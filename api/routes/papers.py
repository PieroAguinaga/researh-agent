"""
api/routes/papers.py

/api/papers/search — GET: search papers directly (bypasses the agent).
Used by the sidebar quick-search feature in the frontend.
"""

import logging
from flask import Blueprint, jsonify, request
from agent.tools.search_tool import _fetch_arxiv
logger = logging.getLogger(__name__)
papers_bp = Blueprint("papers", __name__)


@papers_bp.get("/papers/search")
def search():
    """
    Search for papers directly without going through the agent.

    Query params:
        q (str):           Search query (required).
        max_results (int): Papers per source, 1–20, default 8.

    Response:
        { "papers": [...], "count": N }
    """
    query = (request.args.get("q") or "").strip()
    if not query:
        return jsonify({"error": "Query parameter 'q' is required."}), 400

    max_results = min(int(request.args.get("max_results", 8)), 20)

    logger.info("Direct paper search | query=%.80s | max=%d", query, max_results)
    papers = _fetch_arxiv(query, max_results=max_results)

    return jsonify({"papers": [p.to_dict() for p in papers], "count": len(papers)})
