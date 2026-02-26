"""
PubMed scraper using the NCBI Entrez API via Biopython.

Requires:
    pip install biopython
"""

import logging
from typing import Optional

from Bio import Entrez

from .base import BaseScraper, Paper

logger = logging.getLogger(__name__)


class PubMedScraper(BaseScraper):
    """Scrapes papers from PubMed via the NCBI Entrez E-utilities API."""

    SOURCE = "pubmed"

    def __init__(self, config: dict):
        super().__init__(config)
        source_cfg = config.get("sources", {}).get("pubmed", {})
        email: str = source_cfg.get("email", "")
        api_key: str = source_cfg.get("api_key", "")

        if not email:
            raise ValueError("PubMed requires a valid email address in config.")

        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key

    # ------------------------------------------------------------------
    # Application interface
    # ------------------------------------------------------------------

    def search(self, query: str, **kwargs) -> list[Paper]:
        """
        Search PubMed and return a list of Paper objects.

        Args:
            query: Entrez-formatted query string.

        Returns:
            List of Paper objects.
        """
        ids = self._fetch_ids(query)
        if not ids:
            logger.info("PubMed: no results for query '%s'", query)
            return []

        records = self._fetch_records(ids)
        papers = [self._parse_record(r) for r in records]
        logger.info("PubMed: retrieved %d papers", len(papers))
        return papers

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_entrez_query(self, query: str) -> str:
        """Append date and language filters to the base query."""
        parts = [query]

        filters = self.config.get("filters", {})
        language = filters.get("language", "")
        if language:
            parts.append(f'"{language}"[Language]')

        date_range = self.date_range
        if date_range.get("start") and date_range.get("end"):
            start = date_range["start"].replace("/", "/")
            end = date_range["end"].replace("/", "/")
            parts.append(f'("{start}"[PDAT] : "{end}"[PDAT])')

        article_types = filters.get("article_types", [])
        for at in article_types:
            parts.append(f'"{at}"[Publication Type]')

        return " AND ".join(parts)

    def _fetch_ids(self, query: str) -> list[str]:
        """Use esearch to retrieve PubMed IDs matching the query."""
        full_query = self._build_entrez_query(query)
        logger.debug("PubMed esearch query: %s", full_query)

        handle = Entrez.esearch(
            db="pubmed",
            term=full_query,
            retmax=self.max_results,
            sort="relevance",
        )
        record = Entrez.read(handle)
        handle.close()
        return record.get("IdList", [])

    def _fetch_records(self, ids: list[str]) -> list[dict]:
        """Use efetch to download full records for a list of PubMed IDs."""
        handle = Entrez.efetch(
            db="pubmed",
            id=",".join(ids),
            rettype="xml",
            retmode="xml",
        )
        records = Entrez.read(handle)
        handle.close()
        return records.get("PubmedArticle", [])

    def _parse_record(self, record: dict) -> Paper:
        """Convert a raw Entrez XML record into a Paper dataclass."""
        medline = record.get("MedlineCitation", {})
        article = medline.get("Article", {})

        title = str(article.get("ArticleTitle", ""))
        abstract_texts = (
            article.get("Abstract", {}).get("AbstractText", [])
        )
        abstract = " ".join(str(t) for t in abstract_texts)

        authors = []
        for author in article.get("AuthorList", []):
            last = author.get("LastName", "")
            fore = author.get("ForeName", "")
            authors.append(f"{last}, {fore}".strip(", "))

        pub_date = article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
        year: Optional[int] = None
        try:
            year = int(pub_date.get("Year", pub_date.get("MedlineDate", "")[:4]))
        except (ValueError, TypeError):
            pass

        journal = str(article.get("Journal", {}).get("Title", ""))

        ids_list = medline.get("PMID", None)
        pmid = str(ids_list) if ids_list else ""
        doi = self._extract_doi(article)
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None

        keywords_raw = medline.get("KeywordList", [])
        kw_list = []
        for kw_group in keywords_raw:
            kw_list.extend(str(k) for k in kw_group)

        return Paper(
            title=title,
            authors=authors,
            abstract=abstract,
            year=year,
            doi=doi,
            url=url,
            source=self.SOURCE,
            journal=journal,
            keywords=kw_list,
            extra={"pmid": pmid},
        )

    @staticmethod
    def _extract_doi(article: dict) -> Optional[str]:
        """Extract DOI from ELocationID list if present."""
        for loc in article.get("ELocationID", []):
            if loc.attributes.get("EIdType") == "doi":
                return str(loc)
        return None
