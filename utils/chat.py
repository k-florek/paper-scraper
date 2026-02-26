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

_GET_ALL_SUGGESTED = """
SELECT p.id, p.title, p.authors, p.year, p.journal, p.doi, p.url,
       pi.status, pi.interacted_at
FROM papers p
JOIN paper_interactions pi ON pi.paper_id = p.id
ORDER BY pi.interacted_at DESC;
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
        """Print all papers that have been suggested or read this session."""
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(_GET_ALL_SUGGESTED)
        rows = cursor.fetchall()
        conn.row_factory = None
        if not rows:
            print("No papers have been suggested yet.\n")
            return
        print("\n--- Paper Interaction History ---")
        for row in rows:
            status_label = "READ" if row["status"] == "read" else "suggested"
            print(
                f"  [{status_label}] {row['title']} "
                f"({row['year'] or 'n/a'}) – {row['journal'] or ''}"
            )
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

    print("\n=== Paper Discovery Chat ===")
    print(f"  Model      : {chat_model}")
    print(f"  Embedding  : {embedding_model}")
    print(f"  Papers in DB: {total_vectors}")
    print()
    print("Commands:")
    print("  mark read <n>  – mark paper #n from the last batch as read")
    print("  status         – show all suggested / read papers")
    print("  quit / exit    – end the session")
    print()

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting chat.")
                break

            if not user_input:
                continue

            # ---- built-in commands ----------------------------------------
            lower = user_input.lower()

            if lower in ("quit", "exit"):
                print("Goodbye!")
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
                        print(f'Marked as read: "{title}"\n')
                    else:
                        print(
                            f"Invalid number. The last batch had "
                            f"{len(last_suggested)} paper(s).\n"
                        )
                else:
                    print("Usage: mark read <n>   e.g.  mark read 2\n")
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
                print(f"\n[Error] Could not get a response: {exc}\n")
                conversation.pop()  # roll back the failed turn
                continue

            conversation.append({"role": "assistant", "content": assistant_text})
            print(f"\nAssistant: {assistant_text}\n")

    finally:
        conn.close()
