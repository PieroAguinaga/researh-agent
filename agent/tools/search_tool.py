"""
agent/tools/search_tool.py

LangChain tools for retrieving scientific papers from arXiv and Semantic Scholar.

Design notes:
- tenacity retries handle transient API failures gracefully.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import arxiv
import httpx
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class Paper:
    """Normalised representation of a paper from any source."""

    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    published: str              # ISO date string e.g. "2024-03-15"
    source: str                 # "arxiv" | "semantic_scholar"
    categories: list[str] = field(default_factory=list)
    citation_count: int = 0
    pdf_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "url": self.url,
            "published": self.published,
            "source": self.source,
            "categories": self.categories,
            "citation_count": self.citation_count,
            "pdf_url": self.pdf_url,
        }


# ── arXiv ──────────────────────────────────────────────────────────────────────

def _fetch_arxiv(query: str, max_results: int) -> list[Paper]:
    """
    Fetch papers from arXiv using the official Python client (synchronous).

    Args:
        query:       Search string — keywords, arXiv categories, or author names.
        max_results: Maximum number of papers to return.
    """
    client = arxiv.Client(num_retries=3, delay_seconds=2)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    papers: list[Paper] = []
    for result in client.results(search):
        papers.append(Paper(
            paper_id=result.entry_id,
            title=result.title.strip(),
            authors=[a.name for a in result.authors],
            abstract=result.summary.strip(),
            url=result.entry_id,
            published=result.published.date().isoformat(),
            source="arxiv",
            categories=result.categories,
            pdf_url=result.pdf_url or "",
        ))

    logger.info("arXiv → %d papers for query: '%s'", len(papers), query)
    return papers



# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool
def search_papers(query: str, max_results: int = 8) -> list[dict]:
    """
    Search for scientific papers on arXiv.

    Use this tool when the user asks to find, retrieve, or look up research
    papers on a topic. Returns structured metadata for each paper found.

    Args:
        query:       Search query, e.g. "retrieval augmented generation 2024".
        max_results: Number of papers to return per source (default 8, max 20).

    Returns:
        List of paper dicts with keys: paper_id, title, authors, abstract,
        url, published, source, categories, citation_count, pdf_url.
    """
    max_results = min(max_results, 20)
    papers = _fetch_arxiv(query, max_results)
    logger.info("search_papers tool → returning %d results", len(papers))
    return [p.to_dict() for p in papers]
