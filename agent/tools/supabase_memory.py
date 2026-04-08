"""
agent/tools/supabase_memory.py

Persistent conversation memory backed by Supabase.

This module provides two things:
  1. save_turn()   — writes a single message turn to the DB after each agent response.
  2. load_history() — retrieves the full message history for a thread_id (session).

LangGraph does not yet have a first-party Supabase checkpointer, so we manage
persistence manually: the Flask route calls load_history() to hydrate the state
before invoking the graph, and save_turn() after the graph completes.
"""

import logging
from typing import Any

from supabase import create_client, Client

from config.settings import settings

logger = logging.getLogger(__name__)


# ── Supabase client ────────────────────────────────────────────────────────────

def _get_client() -> Client:
    """Return a Supabase client using the service role key (server-side only)."""
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


# ── Public helpers ─────────────────────────────────────────────────────────────

def save_turn(thread_id: str, role: str, content: str, tool_calls: list | None = None) -> None:
    """
    Persist a single conversation turn to Supabase.

    Args:
        thread_id:  Session / conversation identifier.
        role:       'human' | 'ai' | 'tool'
        content:    Message text content.
        tool_calls: Optional list of tool call dicts for AI turns.
    """
    try:
        client = _get_client()
        client.table(settings.supabase_conversations_table).insert({
            "thread_id": thread_id,
            "role": role,
            "content": content,
            "tool_calls": tool_calls or [],
        }).execute()
        logger.debug("Saved turn | thread=%s role=%s", thread_id, role)
    except Exception as exc:
        # Memory errors should not crash the agent — log and continue
        logger.warning("Failed to save conversation turn: %s", exc)


def load_history(thread_id: str, limit: int = 20) -> list[dict[str, Any]]:
    """
    Load the most recent message turns for a thread from Supabase.

    Returns rows ordered oldest-first so they can be passed directly
    to LangChain as a message sequence.

    Args:
        thread_id: Session / conversation identifier.
        limit:     Maximum number of past turns to load (default 20).

    Returns:
        List of dicts with keys: role, content, tool_calls, created_at.
    """
    try:
        client = _get_client()
        response = (
            client.table(settings.supabase_conversations_table)
            .select("role, content, tool_calls, created_at")
            .eq("thread_id", thread_id)
            .order("created_at", desc=False)
            .limit(limit)
            .execute()
        )
        return response.data or []
    except Exception as exc:
        logger.warning("Failed to load conversation history: %s", exc)
        return []


def save_papers(papers: list[dict]) -> int:
    """
    Upsert a list of papers into the papers table (deduplicates by paper_id).

    Args:
        papers: List of paper dicts from search_papers tool output.

    Returns:
        Number of papers successfully upserted.
    """
    if not papers:
        return 0
    try:
        client = _get_client()
        # upsert ignores conflicts on paper_id (natural key)
        client.table(settings.supabase_papers_table).upsert(
            papers, on_conflict="paper_id"
        ).execute()
        logger.info("Upserted %d papers to Supabase", len(papers))
        return len(papers)
    except Exception as exc:
        logger.warning("Failed to upsert papers: %s", exc)
        return 0
