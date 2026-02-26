"""
Base scraper abstract class that all source scrapers inherit from.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Paper:
    """Normalized representation of a scraped paper."""
    title: str
    authors: list[str]
    abstract: str
    year: Optional[int]
    doi: Optional[str]
    url: Optional[str]
    source: str
    citations: Optional[int] = None
    journal: Optional[str] = None
    keywords: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


class BaseScraper(ABC):
    """
    Abstract base class for all paper scrapers.

    Each subclass must implement:
        - search(query, **kwargs) -> list[Paper]
    """

    def __init__(self, config: dict):
        self.config = config
        self.max_results: int = config.get("search", {}).get("max_results", 50)
        self.date_range: dict = config.get("search", {}).get("date_range", {})

    def build_query(self, keyword_phrase: str) -> str:
        """
        Convert a keyword phrase into a query string.

        Each word in the phrase (split on whitespace) is wrapped in quotes
        and joined with AND so that multi-word phrases become precise queries.

        Example:
            "infectious disease genomics"
            → '"infectious" AND "disease" AND "genomics"'
        """
        words = keyword_phrase.split()
        return " AND ".join(f'"{w}"' for w in words)

    @abstractmethod
    def search(self, query: str, **kwargs) -> list[Paper]:
        """
        Run a search against the source and return a list of Paper objects.

        Args:
            query: The search query string.
            **kwargs: Source-specific parameters.

        Returns:
            List of Paper dataclass instances.
        """
        raise NotImplementedError

    def scrape(self, keywords: list[str]) -> list[Paper]:
        """
        Iterate over each keyword phrase, run an individual search for each,
        and return the combined results.

        Each entry in *keywords* is treated as its own search: words within
        the phrase are joined with AND, so "infectious disease genomics"
        becomes '"infectious" AND "disease" AND "genomics"'.

        Args:
            keywords: List of keyword phrase strings from the config.

        Returns:
            Combined list of Paper dataclass instances across all phrases.
        """
        all_papers: list[Paper] = []
        for phrase in keywords:
            query = self.build_query(phrase)
            logger.info(
                "%s: searching for phrase '%s' → %s",
                self.__class__.__name__,
                phrase,
                query,
            )
            papers = self.search(query)
            all_papers.extend(papers)
        return all_papers
