"""
Google Scholar scraper using the `scholarly` library.

Requires:
    pip install scholarly

Note: Google Scholar does not have an official API and will rate-limit
aggressive scrapers. Use the proxy option in the config if needed.
"""

import logging
import time
from typing import Optional

from scholarly import scholarly, ProxyGenerator

from .base import BaseScraper, Paper

logger = logging.getLogger(__name__)

_RATE_LIMIT_SECONDS = 2  # Polite delay between requests


class GoogleScholarScraper(BaseScraper):
    """Scrapes papers from Google Scholar via the `scholarly` library."""

    SOURCE = "google_scholar"

    def __init__(self, config: dict):
        super().__init__(config)
        source_cfg = config.get("sources", {}).get("google_scholar", {})

        if source_cfg.get("use_proxy") and source_cfg.get("proxy"):
            pg = ProxyGenerator()
            pg.SingleProxy(http=source_cfg["proxy"], https=source_cfg["proxy"])
            scholarly.use_proxy(pg)
            logger.info("Google Scholar: proxy configured")

    # ------------------------------------------------------------------
    # Application interface
    # ------------------------------------------------------------------

    def search(self, query: str, **kwargs) -> list[Paper]:
        """
        Search Google Scholar and return a list of Paper objects.

        Args:
            query: Free-text query string.

        Returns:
            List of Paper objects.
        """
        papers: list[Paper] = []
        date_range = self.date_range
        start_year: Optional[int] = self._parse_year(date_range.get("start"))
        end_year: Optional[int] = self._parse_year(date_range.get("end"))

        search_kwargs: dict = {}
        if start_year:
            search_kwargs["year_low"] = start_year
        if end_year:
            search_kwargs["year_high"] = end_year

        logger.debug("Google Scholar query: %s | filters: %s", query, search_kwargs)

        results = scholarly.search_pubs(query, **search_kwargs)

        for i, result in enumerate(results):
            if i >= self.max_results:
                break
            try:
                paper = self._parse_result(result)
                papers.append(paper)
                time.sleep(_RATE_LIMIT_SECONDS)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Google Scholar: failed to parse result %d – %s", i, exc)

        logger.info("Google Scholar: retrieved %d papers", len(papers))
        return papers

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_result(self, result: dict) -> Paper:
        """Convert a raw scholarly result dict into a Paper dataclass."""
        bib = result.get("bib", {})

        title = bib.get("title", "")
        abstract = bib.get("abstract", "")
        authors = bib.get("author", [])
        if isinstance(authors, str):
            authors = [a.strip() for a in authors.split(" and ")]

        year: Optional[int] = None
        try:
            year = int(bib.get("pub_year", 0))
        except (ValueError, TypeError):
            pass

        journal = bib.get("venue", "") or bib.get("journal", "")
        citations = result.get("num_citations")
        url = result.get("pub_url") or result.get("eprint_url")
        doi = result.get("doi")

        return Paper(
            title=title,
            authors=authors,
            abstract=abstract,
            year=year,
            doi=doi,
            url=url,
            source=self.SOURCE,
            citations=citations,
            journal=journal,
            keywords=[],
            extra={"scholar_id": result.get("author_id", "")},
        )

    @staticmethod
    def _parse_year(date_str: Optional[str]) -> Optional[int]:
        """Extract a 4-digit year from a date string like '2023/01/01'."""
        if not date_str:
            return None
        try:
            return int(str(date_str)[:4])
        except (ValueError, TypeError):
            return None
