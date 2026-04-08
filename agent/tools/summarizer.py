"""
agent/tools/summarizer.py

LangChain tool that produces structured, LLM-powered summaries of scientific papers.

Output schema (PaperSummary) uses Pydantic + with_structured_output so the caller
always receives a predictable dict — no prompt-based JSON parsing required.
"""

import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agent.llm import get_llm

logger = logging.getLogger(__name__)


# ── Output schema ──────────────────────────────────────────────────────────────

class PaperSummary(BaseModel):
    """Structured summary returned for every paper."""

    title: str = Field(description="Paper title")
    one_liner: str = Field(description="One sentence capturing the core contribution")
    key_findings: list[str] = Field(description="3 to 5 main findings or contributions")
    methodology: str = Field(description="Brief description of the approach or methods used")
    practical_implications: str = Field(
        description="How an engineer or practitioner could apply this work"
    )
    limitations: str = Field(description="Acknowledged limitations or open questions")
    keywords: list[str] = Field(description="5 to 8 relevant technical keywords")


# ── Prompt ─────────────────────────────────────────────────────────────────────

_SYSTEM = """You are an expert AI research analyst writing for software engineers
and ML practitioners. Produce concise, technically precise summaries.

Rules:
- Be specific. Avoid vague phrases like "the paper explores...".
- key_findings must have between 3 and 5 items.
- keywords must have between 5 and 8 items.
- practical_implications should suggest a concrete engineering use case."""

_HUMAN = """Summarize this paper:

Title: {title}

Abstract:
{abstract}"""

SUMMARIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", _HUMAN),
])


# ── Internal chain builder ─────────────────────────────────────────────────────

def _summarize_one(title: str, abstract: str) -> dict[str, Any]:
    """Run the summarization chain for a single paper. Returns a dict."""
    llm = get_llm(temperature=0)
    structured_llm = llm.with_structured_output(PaperSummary)
    chain = SUMMARIZE_PROMPT | structured_llm

    try:
        result: PaperSummary = chain.invoke({"title": title, "abstract": abstract})
        logger.info("Summarized: '%s'", title[:70])
        return result.model_dump()
    except Exception as exc:
        logger.error("Summarization failed for '%s': %s", title[:70], exc)
        # Graceful degradation — return a minimal valid structure
        return {
            "title": title,
            "one_liner": "Summary could not be generated.",
            "key_findings": [],
            "methodology": "",
            "practical_implications": "",
            "limitations": "",
            "keywords": [],
        }


# ── LangChain tools ────────────────────────────────────────────────────────────

@tool
def summarize_paper(title: str, abstract: str) -> dict[str, Any]:
    """
    Generate a structured summary of a single scientific paper.

    Use this when the user asks to summarize, explain, or get details
    about a specific paper. Always provide both title and abstract.

    Args:
        title:    Full paper title.
        abstract: Paper abstract text.

    Returns:
        Dict with: title, one_liner, key_findings, methodology,
        practical_implications, limitations, keywords.
    """
    return _summarize_one(title, abstract)


@tool
def summarize_multiple_papers(papers: list[dict]) -> list[dict[str, Any]]:
    """
    Generate structured summaries for a list of papers.

    Use this when the user asks to summarize several papers at once,
    or after search_papers returns multiple results and the user wants
    an overview of all of them.

    Each item in the list must contain 'title' and 'abstract' keys.

    Args:
        papers: List of paper dicts (from search_papers output).

    Returns:
        List of structured summary dicts, one per paper.
    """
    summaries = []
    for paper in papers:
        summary = _summarize_one(
            title=paper.get("title", ""),
            abstract=paper.get("abstract", ""),
        )
        summaries.append(summary)
    return summaries