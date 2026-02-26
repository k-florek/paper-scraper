"""
Persists Paper objects to a SQLite database.
"""

import json
import logging
import os
import sqlite3
from dataclasses import asdict

from scraper.base import Paper

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS papers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT    NOT NULL,
    authors     TEXT,
    abstract    TEXT,
    year        INTEGER,
    doi         TEXT,
    url         TEXT,
    source      TEXT,
    citations   INTEGER,
    journal     TEXT,
    keywords    TEXT,
    extra       TEXT,
    inserted_at DATETIME DEFAULT (datetime('now')),
    UNIQUE (doi, source),
    UNIQUE (title, source)
);
"""

_INSERT_PAPER = """
INSERT OR IGNORE INTO papers
    (title, authors, abstract, year, doi, url, source, citations, journal, keywords, extra)
VALUES
    (:title, :authors, :abstract, :year, :doi, :url, :source, :citations, :journal, :keywords, :extra);
"""


class Exporter:
    """Persists scraped Paper objects into a local SQLite database."""

    def __init__(self, config: dict):
        output_cfg = config.get("output", {})
        db_path: str = output_cfg.get("db_path", "databases/papers.db")

        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

        self.db_path = db_path
        self._init_db()

    # ------------------------------------------------------------------
    # Application interface
    # ------------------------------------------------------------------

    def export(self, papers: list[Paper]) -> str:
        """
        Persist papers to the SQLite database using INSERT OR IGNORE so
        re-runs never create duplicates (keyed on doi+source or title+source).

        Args:
            papers: List of Paper objects to persist.

        Returns:
            Absolute path to the database file.
        """
        if not papers:
            logger.warning("No papers to persist.")
            return os.path.abspath(self.db_path)

        rows = [self._to_row(p) for p in papers]

        with self._connect() as conn:
            result = conn.executemany(_INSERT_PAPER, rows)
            inserted = result.rowcount

        logger.info(
            "Persisted %d new papers (%d duplicates skipped) → %s",
            inserted,
            len(papers) - inserted,
            self.db_path,
        )
        return os.path.abspath(self.db_path)

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        """
        Run an arbitrary SELECT against the database and return rows as dicts.

        Args:
            sql:    SQL query string.
            params: Bound parameters tuple.

        Returns:
            List of row dicts.
        """
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create the papers table if it does not already exist."""
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)
        logger.debug("SQLite database ready: %s", self.db_path)

    def _connect(self) -> sqlite3.Connection:
        """Return a new connection with WAL mode for better concurrency."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    @staticmethod
    def _to_row(paper: Paper) -> dict:
        """Serialise a Paper into the flat dict expected by _INSERT_PAPER."""
        d = asdict(paper)
        d["authors"] = json.dumps(d.get("authors") or [])
        d["keywords"] = json.dumps(d.get("keywords") or [])
        d["extra"] = json.dumps(d.get("extra") or {})
        return d
