#!/usr/bin/env python3
"""
main.py – Entry point for the paper scraper.

Usage:
    python main.py scrape                        # scrape papers into SQLite
    python main.py scrape --config my_config.json
    python main.py vectorize                     # embed papers into ChromaDB
    python main.py vectorize --db results/papers.db
    python main.py chat                          # interactive paper discovery chat
    python main.py chat --config my_config.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from scraper import PubMedScraper, GoogleScholarScraper
from scraper.base import Paper
from utils import Exporter, build_vector_db, start_chat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    """Load and parse the JSON configuration file."""
    config_path = Path(path)
    if not config_path.exists():
        logger.error("Config file not found: %s", path)
        sys.exit(1)

    with config_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def run(config: dict) -> list[Paper]:
    """
    Orchestrate scraping across all enabled sources.

    Args:
        config: Parsed configuration dictionary.

    Returns:
        Deduplicated list of Paper objects.
    """
    keywords: list[str] = config.get("search", {}).get("keywords", [])
    if not keywords:
        logger.error("No keywords found in configuration.")
        sys.exit(1)

    sources_cfg = config.get("sources", {})
    all_papers: list[Paper] = []

    # ---- PubMed ----
    if sources_cfg.get("pubmed", {}).get("enabled", False):
        logger.info("Starting PubMed scraper...")
        try:
            pubmed = PubMedScraper(config)
            papers = pubmed.scrape(keywords)
            logger.info("PubMed returned %d papers.", len(papers))
            all_papers.extend(papers)
        except Exception as exc:  # noqa: BLE001
            logger.error("PubMed scraper failed: %s", exc)

    # ---- Google Scholar ----
    if sources_cfg.get("google_scholar", {}).get("enabled", False):
        logger.info("Starting Google Scholar scraper...")
        try:
            gs = GoogleScholarScraper(config)
            papers = gs.scrape(keywords)
            logger.info("Google Scholar returned %d papers.", len(papers))
            all_papers.extend(papers)
        except Exception as exc:  # noqa: BLE001
            logger.error("Google Scholar scraper failed: %s", exc)

    # ---- Deduplication by DOI / title ----
    all_papers = _deduplicate(all_papers)
    logger.info("Total unique papers after dedup: %d", len(all_papers))
    return all_papers


def _deduplicate(papers: list[Paper]) -> list[Paper]:
    """
    Remove duplicates by DOI (if available) or normalized title.

    Args:
        papers: Raw list of Paper objects, potentially from multiple sources.

    Returns:
        Deduplicated list preserving first occurrence.
    """
    seen: set[str] = set()
    unique: list[Paper] = []
    for paper in papers:
        key = (paper.doi or "").strip().lower() or paper.title.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(paper)
    return unique


def cmd_scrape(args: argparse.Namespace) -> None:
    """Scrape papers and persist them to the SQLite database."""
    config = load_config(args.config)
    papers = run(config)

    if papers:
        exporter = Exporter(config)
        output_path = exporter.export(papers)
        logger.info("Results saved to: %s", output_path)
    else:
        logger.warning("No papers were retrieved.")


def cmd_chat(args: argparse.Namespace) -> None:
    """Start an interactive chat session to discover papers via an Ollama model."""
    config = load_config(args.config)
    try:
        start_chat(config)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except ImportError as exc:
        logger.error("%s", exc)
        sys.exit(1)


def cmd_vectorize(args: argparse.Namespace) -> None:
    """Embed papers from the SQLite database into a ChromaDB vector store."""
    config = load_config(args.config)
    db_path: str | None = args.db if args.db else None

    try:
        chroma_path = build_vector_db(config, db_path=db_path)
        logger.info("Vector database ready: %s", chroma_path)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except ImportError as exc:
        logger.error("%s", exc)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper scraper – scrape, vectorize, and chat about academic papers.",
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to the JSON configuration file (default: config.json)",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ---- scrape subcommand ----
    sub_scrape = subparsers.add_parser(
        "scrape",
        help="Scrape papers from PubMed / Google Scholar and store in SQLite.",
    )
    sub_scrape.add_argument(
        "--config",
        default="config.json",
        dest="config",
        help="Path to the JSON configuration file (default: config.json)",
    )

    # ---- vectorize subcommand ----
    sub_vec = subparsers.add_parser(
        "vectorize",
        help="Embed papers from SQLite into a ChromaDB vector store via Ollama.",
    )
    sub_vec.add_argument(
        "--config",
        default="config.json",
        dest="config",
        help="Path to the JSON configuration file (default: config.json)",
    )
    sub_vec.add_argument(
        "--db",
        default=None,
        metavar="DB_PATH",
        help="Override the SQLite database path from config.",
    )

    # ---- chat subcommand ----
    sub_chat = subparsers.add_parser(
        "chat",
        help="Start an interactive chat session to discover papers via Ollama.",
    )
    sub_chat.add_argument(
        "--config",
        default="config.json",
        dest="config",
        help="Path to the JSON configuration file (default: config.json)",
    )

    args = parser.parse_args()

    # Default to 'scrape' when no subcommand is given (backwards compat)
    if args.command is None:
        args.command = "scrape"
        # args.config already has its default from the top-level --config flag

    if args.command == "scrape":
        cmd_scrape(args)
    elif args.command == "vectorize":
        cmd_vectorize(args)
    elif args.command == "chat":
        cmd_chat(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
