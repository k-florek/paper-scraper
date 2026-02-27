"""
Interactive chat session that uses ChromaDB vector search + an Ollama LLM
to help users discover academic papers from the local vector database.

Paper interactions (suggested / read) are persisted in the SQLite database
across sessions so the model never re-suggests a paper it has already shown,
and skips anything the user has marked as read.

Application API
----------
start_chat(config) -> None
    Launch an interactive REPL connected to an Ollama chat model.
"""

import logging
import os
import sqlite3

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------

_CREATE_INTERACTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS paper_interactions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id        INTEGER NOT NULL REFERENCES papers(id),
    status          TEXT    NOT NULL CHECK(status IN ('suggested', 'read')),
    interacted_at   DATETIME DEFAULT (datetime('now')),
    UNIQUE (paper_id, status)
);
"""

_MARK_INTERACTION = """
INSERT OR IGNORE INTO paper_interactions (paper_id, status) VALUES (?, ?);
"""

_GET_INTERACTED_IDS = """
SELECT DISTINCT paper_id FROM paper_interactions;
"""

_GET_READ_IDS = """
SELECT DISTINCT paper_id FROM paper_interactions WHERE status = 'read';
"""

_RESET_ALL = """
DELETE FROM paper_interactions;
"""

_RESET_ALL_STATUS = """
DELETE FROM paper_interactions WHERE status = ?;
"""

_RESET_PAPER = """
DELETE FROM paper_interactions WHERE paper_id = ?;
"""

_RESET_PAPER_STATUS = """
DELETE FROM paper_interactions WHERE paper_id = ? AND status = ?;
"""

# Colour codes (no external dependency)
_C_RESET    = "\033[0m"
_C_BOLD     = "\033[1m"
_C_DIM      = "\033[2m"    # timestamp
_C_CYAN     = "\033[36m"   # suggested label
_C_GREEN    = "\033[32m"   # read label
_C_YELLOW   = "\033[33m"   # You: prompt
_C_BLUE     = "\033[94m"   # Assistant: prefix
_C_RED      = "\033[31m"   # errors
_C_MAGENTA  = "\033[35m"   # banner / accents
_C_TITLE    = "\033[97m"   # paper title (bright white)

# For each paper return its effective status (read beats suggested) and the
# timestamp of that status row, ordered newest-first.
_GET_ALL_INTERACTIONS = """
SELECT p.id, p.title, p.authors, p.year, p.journal,
       CASE WHEN r.paper_id IS NOT NULL THEN 'read' ELSE 'suggested' END
           AS effective_status,
       CASE WHEN r.paper_id IS NOT NULL THEN r.interacted_at
            ELSE s.interacted_at
       END AS status_at
FROM papers p
JOIN  paper_interactions s ON s.paper_id = p.id AND s.status = 'suggested'
LEFT JOIN paper_interactions r ON r.paper_id = p.id AND r.status = 'read'
ORDER BY status_at DESC;
"""


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------


def start_chat(config: dict) -> None:
    """
    Launch an interactive prompt that lets the user converse with an Ollama
    model to discover relevant papers from the local ChromaDB vector store.

    Paper suggestions and read-marks are persisted in the SQLite database so
    the model always knows what it has already shown or what the user has
    already read.

    Args:
        config: Parsed configuration dictionary.  See ``config.example.json``
                for the ``"chat"`` block.

    Commands available inside the session
    --------------------------------------
    mark read <n>   Mark paper #n from the most recent batch as read.
    status          List all suggested/read papers.
    quit / exit     End the session.
    """
    try:
        import chromadb  # noqa: F401
        import ollama as _ollama  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "chromadb and ollama are required: pip install chromadb ollama"
        ) from exc

    import chromadb
    import ollama as ollama_client

    # ------------------------------------------------------------------
    # Resolve configuration
    # ------------------------------------------------------------------
    vec_cfg: dict = config.get("vectorize", {})
    chat_cfg: dict = config.get("chat", {})

    ollama_host: str = chat_cfg.get(
        "ollama_host", vec_cfg.get("ollama_host", "http://localhost:11434")
    )
    chat_model: str = chat_cfg.get("chat_model", "llama3")
    embedding_model: str = chat_cfg.get(
        "embedding_model", vec_cfg.get("embedding_model", "nomic-embed-text")
    )
    chroma_path: str = vec_cfg.get("chroma_path", "databases/vectors")
    collection_name: str = vec_cfg.get("collection_name", "papers")
    top_k: int = int(chat_cfg.get("top_k", 5))
    system_prompt: str = chat_cfg.get(
        "system_prompt",
        (
"""
You are a helpful academic researcher. You have access to a curated database of scientific papers. When papers are provided to you as context, select the 1 to 3 papers that are most relevant to the user's query and recommend only those. Each paper in the context is identified by a unique tag of the form [ID:number]. Do not recommend papers that are not a good match.

For each recommended paper you MUST use EXACTLY this format and no other:

[ID:number] <title of the paper>
URL: <url of the paper, or "N/A" if not available>
<explanation of why this paper is relevant to the user's query>

Do not add any other fields, headers, or prose outside this structure. Do not alter or omit the [ID:number] tag. Do not alter the URL. CRITICAL: the relevance explanation MUST be derived exclusively from the abstract text provided in the context. Do NOT draw on your training knowledge to add, infer, or embellish any findings, methods, conclusions, or claims not explicitly stated in the provided abstract. If no abstract is available, state that in the explanation. Do not fabricate papers or citations outside the provided context.
"""
        ),
    )

    db_path: str = config.get("output", {}).get("db_path", "databases/papers.db")

    # ------------------------------------------------------------------
    # Pre-flight checks
    # ------------------------------------------------------------------
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"SQLite database not found: {db_path}\n"
            "Run the 'scrape' subcommand first."
        )
    if not os.path.exists(chroma_path):
        raise FileNotFoundError(
            f"ChromaDB directory not found: {chroma_path}\n"
            "Run the 'vectorize' subcommand first."
        )

    # ------------------------------------------------------------------
    # Initialise ChromaDB + Ollama client
    # ------------------------------------------------------------------
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    total_vectors = collection.count()
    if total_vectors == 0:
        raise RuntimeError(
            "The ChromaDB collection is empty.  "
            "Run the 'vectorize' subcommand first."
        )

    oll = ollama_client.Client(host=ollama_host)

    # ------------------------------------------------------------------
    # Initialise interaction-tracking table
    # ------------------------------------------------------------------
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(_CREATE_INTERACTIONS_TABLE)
    conn.commit()

    # ------------------------------------------------------------------
    # Interaction helpers
    # ------------------------------------------------------------------

    def get_interacted_ids() -> set[str]:
        """Return the set of ChromaDB string IDs already suggested or read."""
        cursor = conn.execute(_GET_INTERACTED_IDS)
        return {str(row[0]) for row in cursor.fetchall()}

    def mark_papers_suggested(paper_ids: list[int]) -> None:
        conn.executemany(_MARK_INTERACTION, [(pid, "suggested") for pid in paper_ids])
        conn.commit()

    def mark_paper_read(paper_id: int) -> None:
        """Mark a paper as both suggested and read (idempotent)."""
        conn.executemany(
            _MARK_INTERACTION,
            [(paper_id, "suggested"), (paper_id, "read")],
        )
        conn.commit()

    def reset_paper(paper_id: int, status: str | None = None) -> int:
        """Delete interaction rows for one paper. Returns rows deleted."""
        if status:
            cur = conn.execute(_RESET_PAPER_STATUS, (paper_id, status))
        else:
            cur = conn.execute(_RESET_PAPER, (paper_id,))
        conn.commit()
        return cur.rowcount

    def reset_all(status: str | None = None) -> int:
        """Delete all interaction rows (optionally filtered by status). Returns rows deleted."""
        if status:
            cur = conn.execute(_RESET_ALL_STATUS, (status,))
        else:
            cur = conn.execute(_RESET_ALL)
        conn.commit()
        return cur.rowcount

    def print_status() -> None:
        """Print all papers in reverse-chronological order with status labels."""
        conn.row_factory = sqlite3.Row
        rows = conn.execute(_GET_ALL_INTERACTIONS).fetchall()
        conn.row_factory = None

        if not rows:
            print("No papers have been suggested yet.\n")
            return

        print(f"\n{_C_BOLD}--- Paper Interaction History ---{_C_RESET}")
        for row in rows:
            ts     = row["status_at"] or "unknown time"
            status = row["effective_status"]
            if status == "read":
                label = f"{_C_GREEN}[read]{_C_RESET}"
            else:
                label = f"{_C_CYAN}[suggested]{_C_RESET}"
            title   = f"{_C_TITLE}{row['title']}{_C_RESET}"
            meta    = f"({row['year'] or 'n/a'}) – {row['journal'] or ''}"
            print(f"  {_C_DIM}[{ts}]{_C_RESET} {label} {title} {meta}")
        print()

    # ------------------------------------------------------------------
    # Retrieval helper
    # ------------------------------------------------------------------

    def retrieve_papers(query_text: str, exclude_ids: set[str]) -> list[dict]:
        """
        Embed *query_text* and return up to *top_k* ChromaDB results,
        filtering out any IDs present in *exclude_ids*.
        """
        try:
            resp = oll.embeddings(model=embedding_model, prompt=query_text)
            query_embedding = resp["embedding"]
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to embed query: %s", exc)
            return []

        # Fetch extra candidates to account for client-side exclusion
        fetch_k = min(top_k + len(exclude_ids) + 20, total_vectors)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
            include=["documents", "metadatas"],
        )

        papers: list[dict] = []
        ids_list = results.get("ids", [[]])[0]
        metadatas_list = results.get("metadatas", [[]])[0]
        documents_list = results.get("documents", [[]])[0]

        for cid, meta, doc in zip(ids_list, metadatas_list, documents_list):
            if cid in exclude_ids:
                continue
            papers.append({"chroma_id": cid, "meta": meta, "doc": doc})
            if len(papers) >= top_k:
                break

        return papers

    # ------------------------------------------------------------------
    # Formatting helper
    # ------------------------------------------------------------------

    def format_papers_context(papers: list[dict]) -> str:
        """Return a tagged plain-text block describing *papers*."""
        lines: list[str] = []
        for p in papers:
            meta = p["meta"]
            tag = f"[ID:{p['chroma_id']}]"
            lines.append(f"{tag} {meta.get('title', 'Unknown title')}")
            if meta.get("authors"):
                lines.append(f"    Authors : {meta['authors']}")
            if meta.get("year"):
                lines.append(f"    Year    : {meta['year']}")
            if meta.get("journal"):
                lines.append(f"    Journal : {meta['journal']}")
            if meta.get("doi"):
                lines.append(f"    DOI     : {meta['doi']}")
            if meta.get("url"):
                lines.append(f"    URL     : {meta['url']}")
            # Include the full document text (abstract + keywords) so the LLM
            # has actual content to discuss and reference for each paper.
            doc = p.get("doc", "")
            if doc:
                # Strip the title line that is already shown above, then indent
                doc_body = "\n".join(
                    line for line in doc.splitlines()
                    if not line.startswith("Title:")
                ).strip()
                if doc_body:
                    indented = "\n".join(f"    {line}" for line in doc_body.splitlines())
                    lines.append(indented)
            lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Interactive REPL
    # ------------------------------------------------------------------
    conversation: list[dict] = [{"role": "system", "content": system_prompt}]
    last_suggested_by_id: dict[str, dict] = {}

    print(f"\n{_C_BOLD}{_C_MAGENTA}=== Paper Discovery Chat ==={_C_RESET}")
    print(f"  {_C_DIM}Model      :{_C_RESET} {chat_model}")
    print(f"  {_C_DIM}Embedding  :{_C_RESET} {embedding_model}")
    print(f"  {_C_DIM}Papers in DB:{_C_RESET} {total_vectors}")
    print()
    print(f"{_C_BOLD}Commands:{_C_RESET}")
    print(f"  {_C_CYAN}mark read <ID>{_C_RESET}                    – mark a paper as read using its [ID:number]")
    print(f"  {_C_CYAN}reset <ID> suggested|read{_C_RESET}         – reset one status for one paper")
    print(f"  {_C_CYAN}reset all suggested|read{_C_RESET}           – reset one status for all papers")
    print(f"  {_C_CYAN}status{_C_RESET}                      – show all suggested / read papers")
    print(f"  {_C_CYAN}quit / exit{_C_RESET}                 – end the session")
    print()

    try:
        while True:
            try:
                user_input = input(f"{_C_BOLD}{_C_YELLOW}You:{_C_RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n{_C_DIM}Exiting chat.{_C_RESET}")
                break

            if not user_input:
                continue

            # ---- built-in commands ----------------------------------------
            lower = user_input.lower()

            if lower in ("quit", "exit"):
                print(f"{_C_DIM}Goodbye!{_C_RESET}")
                break

            if lower == "status":
                print_status()
                continue

            if lower.startswith("reset"):
                parts = lower.split()
                # Status is required
                status_filter: str | None = parts[-1] if len(parts) >= 2 and parts[-1] in ("suggested", "read") else None

                if status_filter is None:
                    print(
                        f"{_C_DIM}Usage:\n"
                        f"  reset all suggested|read\n"
                        f"  reset <ID> suggested|read{_C_RESET}\n"
                    )
                elif len(parts) == 3 and parts[1] == "all":
                    n = reset_all(status_filter)
                    print(f"{_C_YELLOW}Reset '{status_filter}' for all papers ({n} row(s) removed).{_C_RESET}\n")
                elif len(parts) == 3 and parts[1].isdigit():
                    cid = int(parts[1])
                    n = reset_paper(cid, status_filter)
                    if n:
                        print(f"{_C_YELLOW}Reset '{status_filter}' for paper ID {cid} ({n} row(s) removed).{_C_RESET}\n")
                    else:
                        print(f"{_C_RED}No '{status_filter}' interaction found for paper ID {cid}.{_C_RESET}\n")
                else:
                    print(
                        f"{_C_DIM}Usage:\n"
                        f"  reset all suggested|read\n"
                        f"  reset <ID> suggested|read{_C_RESET}\n"
                    )
                continue

            if lower.startswith("mark read"):
                parts = lower.split()
                if len(parts) == 3 and parts[2].isdigit():
                    cid = parts[2]
                    if cid in last_suggested_by_id:
                        mark_paper_read(int(cid))
                        title = last_suggested_by_id[cid]["meta"].get("title", f"ID:{cid}")
                        print(f"{_C_GREEN}Marked as read:{_C_RESET} \"{title}\"\n")
                    else:
                        print(
                            f"{_C_RED}ID {cid} not found{_C_RESET} in the last batch of "
                            f"suggestions. Use the [ID:number] shown in the response.\n"
                        )
                else:
                    print(f"{_C_DIM}Usage: mark read <ID>   e.g.  mark read 42{_C_RESET}\n")
                continue

            # ---- vector retrieval -----------------------------------------
            already_seen = get_interacted_ids()
            papers = retrieve_papers(user_input, already_seen)
            last_suggested = papers

            if papers:
                context_block = (
                    f"The following {len(papers)} paper(s) were retrieved from the "
                    "vector database as relevant to the user's query. "
                    "Select the 1 to 3 most relevant papers and recommend only those. "
                    "Each paper is identified by its unique [ID:number] tag — you MUST "
                    "use that exact tag when referencing it in your response:\n\n"
                    + format_papers_context(papers)
                )
                # Full message sent to the LLM for this turn (includes abstracts).
                llm_user_message = (
                    f"{context_block}\n"
                    f"User message: {user_input}"
                )
            else:
                llm_user_message = (
                    "No new matching papers were found in the database – "
                    "all relevant papers may have already been suggested or read.\n\n"
                    f"User message: {user_input}"
                )

            # Build the messages list for this inference call: history + current
            # full message.  We do NOT store the bulky context block in history;
            # only the bare user query is kept so the conversation stays within
            # the model's context window across multiple turns.
            inference_messages = conversation + [{"role": "user", "content": llm_user_message}]

            # ---- LLM call -------------------------------------------------
            try:
                resp = oll.chat(model=chat_model, messages=inference_messages)
                assistant_text: str = resp["message"]["content"]
            except Exception as exc:  # noqa: BLE001
                logger.error("Ollama chat error: %s", exc)
                print(f"\n{_C_BOLD}{_C_RED}[Error]{_C_RESET} Could not get a response: {exc}\n")
                continue

            # Resolve [ID:xxx] tags in the response to actual papers.
            import re as _re
            papers_by_id = {p["chroma_id"]: p for p in papers} if papers else {}
            # Collect cited IDs in order of first appearance (deduped).
            seen_ids: list[str] = []
            seen_set: set[str] = set()
            for _m in _re.finditer(r'\[ID:(\d+)\]', assistant_text):
                _cid = _m.group(1)
                if _cid in papers_by_id and _cid not in seen_set:
                    seen_ids.append(_cid)
                    seen_set.add(_cid)

            # Mark only the cited papers as suggested and update last_suggested_by_id
            # so 'mark read <ID>' can look up by chroma ID.
            if seen_ids:
                mark_papers_suggested([int(cid) for cid in seen_ids])
                last_suggested_by_id = {cid: papers_by_id[cid] for cid in seen_ids}
            else:
                last_suggested_by_id = {}

            # Persist only the compact user query + assistant reply to history.
            conversation.append({"role": "user", "content": user_input})
            conversation.append({"role": "assistant", "content": assistant_text})
            print(f"\n{_C_BOLD}{_C_BLUE}Assistant:{_C_RESET} {assistant_text}\n")

    finally:
        conn.close()
