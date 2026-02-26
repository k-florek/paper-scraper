"""
Build a ChromaDB vector database from papers stored in the SQLite database,
using an Ollama model to generate embeddings.

Application API
----------
build_vector_db(config, db_path=None) -> str
    Embed every paper that is not yet in the ChromaDB collection and return
    the absolute path to the ChromaDB directory.
"""

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------


def build_vector_db(config: dict, db_path: Optional[str] = None) -> str:
    """
    Read papers from the SQLite database, embed them with an Ollama model,
    and persist the vectors in a ChromaDB collection.

    Supports incremental updates: papers already present in the collection
    (matched by their integer ``id``) are skipped so re-runs are safe.

    Args:
        config:  Parsed configuration dictionary.  Must contain an
                 ``"output"`` key (for the SQLite path) and optionally a
                 ``"vectorize"`` key with the settings below.
        db_path: Override the SQLite database path from config.

    Returns:
        Absolute path to the ChromaDB persistence directory.

    Configuration keys (all optional, shown with defaults)::

        {
          "vectorize": {
            "ollama_host":       "http://localhost:11434",
            "embedding_model":   "nomic-embed-text",
            "chroma_path":       "databases/chroma",
            "collection_name":   "papers",
            "batch_size":        10
          }
        }
    """
    try:
        import chromadb  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "chromadb is required for vector indexing: pip install chromadb"
        ) from exc

    try:
        import ollama as _ollama  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "ollama is required for vector indexing: pip install ollama"
        ) from exc

    # Deferred imports so the rest of the project doesn't need these deps
    import chromadb
    import ollama as ollama_client

    from utils.exporter import Exporter

    # ------------------------------------------------------------------
    # Resolve configuration
    # ------------------------------------------------------------------
    vec_cfg: dict = config.get("vectorize", {})
    ollama_host: str = vec_cfg.get("ollama_host", "http://localhost:11434")
    embedding_model: str = vec_cfg.get("embedding_model", "nomic-embed-text")
    chroma_path: str = vec_cfg.get("chroma_path", "databases/chroma")
    collection_name: str = vec_cfg.get("collection_name", "papers")
    batch_size: int = int(vec_cfg.get("batch_size", 10))

    if db_path is None:
        db_path = config.get("output", {}).get("db_path", "databases/papers.db")

    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"SQLite database not found: {db_path}\n"
            "Run the 'scrape' subcommand first."
        )

    # ------------------------------------------------------------------
    # Load papers from SQLite
    # ------------------------------------------------------------------
    exporter = Exporter(config)
    rows: list[dict] = exporter.query("SELECT * FROM papers ORDER BY id")

    if not rows:
        logger.warning("No papers found in the database – nothing to embed.")
        return os.path.abspath(chroma_path)

    logger.info("Loaded %d papers from %s", len(rows), db_path)

    # ------------------------------------------------------------------
    # Set up ChromaDB
    # ------------------------------------------------------------------
    os.makedirs(chroma_path, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Incremental: skip papers already indexed
    existing_ids: set[str] = set(collection.get(include=[])["ids"])
    logger.info(
        "%d documents already in ChromaDB collection '%s'.",
        len(existing_ids),
        collection_name,
    )

    new_rows = [r for r in rows if str(r["id"]) not in existing_ids]
    if not new_rows:
        logger.info("ChromaDB collection is already up to date.")
        return os.path.abspath(chroma_path)

    logger.info("%d new papers to embed with model '%s'.", len(new_rows), embedding_model)

    # ------------------------------------------------------------------
    # Embed and store in batches
    # ------------------------------------------------------------------
    oll = ollama_client.Client(host=ollama_host)
    num_batches = (len(new_rows) + batch_size - 1) // batch_size

    for batch_idx, start in enumerate(range(0, len(new_rows), batch_size), start=1):
        batch = new_rows[start : start + batch_size]
        ids, docs, embeddings, metadatas = [], [], [], []

        for row in batch:
            doc_text = _build_document_text(row)
            doc_id = str(row["id"])

            try:
                response = oll.embeddings(model=embedding_model, prompt=doc_text)
                embedding = response["embedding"]
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failed to embed paper id=%s title=%r: %s",
                    doc_id,
                    row.get("title", ""),
                    exc,
                )
                continue

            ids.append(doc_id)
            docs.append(doc_text)
            embeddings.append(embedding)
            metadatas.append(_build_metadata(row))

        if ids:
            collection.add(
                ids=ids,
                documents=docs,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            logger.info(
                "Batch %d/%d – stored %d embeddings.",
                batch_idx,
                num_batches,
                len(ids),
            )

    total = collection.count()
    abs_chroma = os.path.abspath(chroma_path)
    logger.info(
        "ChromaDB collection '%s' now contains %d documents → %s",
        collection_name,
        total,
        abs_chroma,
    )
    return abs_chroma


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_document_text(row: dict) -> str:
    """
    Concatenate the most semantically rich fields of a paper row into a
    single string that will be fed to the embedding model.
    """
    parts: list[str] = []

    if row.get("title"):
        parts.append(f"Title: {row['title']}")

    if row.get("abstract"):
        parts.append(f"Abstract: {row['abstract']}")

    raw_kw = row.get("keywords")
    if raw_kw:
        try:
            kw_list = json.loads(raw_kw) if isinstance(raw_kw, str) else raw_kw
            if kw_list:
                parts.append(f"Keywords: {', '.join(kw_list)}")
        except (json.JSONDecodeError, TypeError):
            pass

    if row.get("journal"):
        parts.append(f"Journal: {row['journal']}")

    return "\n".join(parts)


def _build_metadata(row: dict) -> dict:
    """
    Build the ChromaDB metadata dict for a paper row.

    ChromaDB requires metadata values to be ``str``, ``int``, ``float``,
    or ``bool`` – no ``None`` values are allowed.
    """
    meta: dict = {}

    for field in ("title", "doi", "url", "source", "journal"):
        val = row.get(field)
        if val is not None:
            meta[field] = str(val)

    if row.get("year") is not None:
        meta["year"] = int(row["year"])

    if row.get("citations") is not None:
        meta["citations"] = int(row["citations"])

    raw_authors = row.get("authors")
    if raw_authors:
        try:
            authors_list = (
                json.loads(raw_authors) if isinstance(raw_authors, str) else raw_authors
            )
            meta["authors"] = (
                ", ".join(authors_list)
                if isinstance(authors_list, list)
                else str(authors_list)
            )
        except (json.JSONDecodeError, TypeError):
            meta["authors"] = str(raw_authors)

    return meta
