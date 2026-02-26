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
            "You are a helpful academic research assistant. "
            "You have access to a curated database of scientific papers. "
            "When the user asks about a topic, relevant papers from the database "
            "will be provided to you as context. Introduce each paper clearly, "
            "explain why it is relevant to the user's query, and highlight key "
            "findings or contributions. Be concise and scholarly in tone. "
            "Do not fabricate papers or citations outside the provided context."
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
        """Return a numbered plain-text block describing *papers*."""
        lines: list[str] = []
        for i, p in enumerate(papers, 1):
            meta = p["meta"]
            lines.append(f"[{i}] {meta.get('title', 'Unknown title')}")
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
            lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Interactive REPL
    # ------------------------------------------------------------------
    conversation: list[dict] = [{"role": "system", "content": system_prompt}]
    last_suggested: list[dict] = []

    print(f"\n{_C_BOLD}{_C_MAGENTA}=== Paper Discovery Chat ==={_C_RESET}")
    print(f"  {_C_DIM}Model      :{_C_RESET} {chat_model}")
    print(f"  {_C_DIM}Embedding  :{_C_RESET} {embedding_model}")
    print(f"  {_C_DIM}Papers in DB:{_C_RESET} {total_vectors}")
    print()
    print(f"{_C_BOLD}Commands:{_C_RESET}")
    print(f"  {_C_CYAN}mark read <n>{_C_RESET}  – mark paper #n from the last batch as read")
    print(f"  {_C_CYAN}status{_C_RESET}         – show all suggested / read papers")
    print(f"  {_C_CYAN}quit / exit{_C_RESET}    – end the session")
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

            if lower.startswith("mark read"):
                parts = lower.split()
                if len(parts) == 3 and parts[2].isdigit():
                    idx = int(parts[2]) - 1
                    if 0 <= idx < len(last_suggested):
                        pid = int(last_suggested[idx]["chroma_id"])
                        mark_paper_read(pid)
                        title = last_suggested[idx]["meta"].get("title", f"#{idx + 1}")
                        print(f"{_C_GREEN}Marked as read:{_C_RESET} \"{title}\"\n")
                    else:
                        print(
                            f"{_C_RED}Invalid number.{_C_RESET} The last batch had "
                            f"{len(last_suggested)} paper(s).\n"
                        )
                else:
                    print(f"{_C_DIM}Usage: mark read <n>   e.g.  mark read 2{_C_RESET}\n")
                continue

            # ---- vector retrieval -----------------------------------------
            already_seen = get_interacted_ids()
            papers = retrieve_papers(user_input, already_seen)
            last_suggested = papers

            if papers:
                mark_papers_suggested([int(p["chroma_id"]) for p in papers])
                context_block = (
                    "The following papers were retrieved from the vector database "
                    "as relevant to the user's query:\n\n"
                    + format_papers_context(papers)
                )
                user_message = (
                    f"{context_block}\n"
                    f"User message: {user_input}"
                )
            else:
                user_message = (
                    "No new matching papers were found in the database – "
                    "all relevant papers may have already been suggested or read.\n\n"
                    f"User message: {user_input}"
                )

            conversation.append({"role": "user", "content": user_message})

            # ---- LLM call -------------------------------------------------
            try:
                resp = oll.chat(model=chat_model, messages=conversation)
                assistant_text: str = resp["message"]["content"]
            except Exception as exc:  # noqa: BLE001
                logger.error("Ollama chat error: %s", exc)
                print(f"\n{_C_BOLD}{_C_RED}[Error]{_C_RESET} Could not get a response: {exc}\n")
                conversation.pop()  # roll back the failed turn
                continue

            conversation.append({"role": "assistant", "content": assistant_text})
            print(f"\n{_C_BOLD}{_C_BLUE}Assistant:{_C_RESET} {assistant_text}\n")

    finally:
        conn.close()
