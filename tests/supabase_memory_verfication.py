"""
tests/test_supabase_memory.py

Unit tests for agent/tools/supabase_memory.py

Tests are isolated using unittest.mock — no real Supabase connection is needed.
Each public function (save_turn, load_history, save_papers) is tested for:
  - happy-path behaviour
  - edge cases (empty input, limit param, etc.)
  - graceful degradation when Supabase raises an exception
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from unittest.mock import patch, MagicMock, call

from config.settings import settings
from agent.tools.supabase_memory import save_turn, load_history, save_papers
from config.settings import settings

# ── helpers ────────────────────────────────────────────────────────────────────

def check(label: str, fn) -> bool:
    """Run a check function and print pass/fail."""
    try:
        fn()
        print(f"  ✓  {label}")
        return True
    except Exception as exc:
        print(f"  ✗  {label}")
        print(f"     └─ {exc}")
        return False


# ── save_turn ──────────────────────────────────────────────────────────────────

def test_save_turn_calls_insert():
    """save_turn() must call .insert() with the correct payload."""
    with patch("agent.tools.supabase_memory._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        save_turn("thread-1", "human", "Hello")

        mock_client.table.assert_called_once_with(settings.supabase_conversations_table)
        mock_client.table().insert.assert_called_once_with({
            "thread_id":  "thread-1",
            "role":       "human",
            "content":    "Hello",
            "tool_calls": [],
        })


def test_save_turn_includes_tool_calls():
    """save_turn() must forward non-empty tool_calls to the insert payload."""
    with patch("agent.tools.supabase_memory._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        calls = [{"name": "search", "args": {}}]
        save_turn("thread-1", "ai", "Searching…", tool_calls=calls)

        inserted = mock_client.table().insert.call_args[0][0]
        assert inserted["tool_calls"] == calls, "tool_calls should be forwarded as-is"


def test_save_turn_swallows_exception():
    """save_turn() must not raise even when Supabase throws."""
    with patch("agent.tools.supabase_memory._get_client") as mock_get_client:
        mock_get_client.side_effect = Exception("connection refused")

        # Should complete without raising
        save_turn("thread-err", "human", "Hi")


# ── load_history ───────────────────────────────────────────────────────────────

def test_load_history_returns_data():
    """load_history() must return the rows from response.data."""
    fake_rows = [
        {"role": "human", "content": "Hi",    "tool_calls": [], "created_at": "2024-01-01T00:00:00"},
        {"role": "ai",    "content": "Hello", "tool_calls": [], "created_at": "2024-01-01T00:00:01"},
    ]
    with patch("agent.tools.supabase_memory._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.table().select().eq().order().limit().execute.return_value = \
            MagicMock(data=fake_rows)
        mock_get_client.return_value = mock_client

        result = load_history("thread-1")

        assert result == fake_rows

def test_load_history_passes_limit():
    """load_history() must forward the limit parameter to the query."""
    with patch("agent.tools.supabase_memory._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_client.table().select().eq().order().limit().execute.return_value = \
            MagicMock(data=[])

        load_history("thread-1", limit=5)

        # Grab the actual limit calls and check that 5 was passed at some point
        limit_calls = mock_client.table().select().eq().order().limit.call_args_list
        assert any(c == call(5) for c in limit_calls), \
            f"Expected limit(5) to be called, got: {limit_calls}"


def test_load_history_returns_empty_list_on_none_data():
    """load_history() must return [] when response.data is None."""
    with patch("agent.tools.supabase_memory._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.table().select().eq().order().limit().execute.return_value = \
            MagicMock(data=None)
        mock_get_client.return_value = mock_client

        result = load_history("thread-empty")

        assert result == []


def test_load_history_swallows_exception():
    """load_history() must return [] and not raise when Supabase throws."""
    with patch("agent.tools.supabase_memory._get_client") as mock_get_client:
        mock_get_client.side_effect = Exception("timeout")

        result = load_history("thread-err")

        assert result == []


# ── save_papers ────────────────────────────────────────────────────────────────

def test_save_papers_returns_zero_on_empty_list():
    """save_papers([]) must short-circuit and return 0 without hitting Supabase."""
    with patch("agent.tools.supabase_memory._get_client") as mock_get_client:
        result = save_papers([])

        mock_get_client.assert_not_called()
        assert result == 0


def test_save_papers_calls_upsert():
    """save_papers() must call .upsert() with the correct papers and conflict key."""
    papers = [
        {"paper_id": "1234.5678", "title": "Attention Is All You Need"},
        {"paper_id": "9999.0000", "title": "BERT"},
    ]
    with patch("agent.tools.supabase_memory._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        result = save_papers(papers)

        mock_client.table.assert_called_once_with(settings.supabase_papers_table)
        mock_client.table().upsert.assert_called_once_with(papers, on_conflict="paper_id")
        assert result == 2


def test_save_papers_returns_zero_on_exception():
    """save_papers() must return 0 and not raise when Supabase throws."""
    with patch("agent.tools.supabase_memory._get_client") as mock_get_client:
        mock_get_client.side_effect = Exception("upsert failed")

        result = save_papers([{"paper_id": "x", "title": "y"}])

        assert result == 0


# ── runner ─────────────────────────────────────────────────────────────────────

def main():
    print("\n── supabase_memory tests ────────────────────────────────\n")

    checks = [
        # save_turn
        ("save_turn: calls insert with correct payload",  test_save_turn_calls_insert),
        ("save_turn: forwards tool_calls",                test_save_turn_includes_tool_calls),
        ("save_turn: swallows Supabase exception",        test_save_turn_swallows_exception),
        # load_history
        ("load_history: returns rows from response.data", test_load_history_returns_data),
        ("load_history: forwards limit param",            test_load_history_passes_limit),
        ("load_history: returns [] when data is None",    test_load_history_returns_empty_list_on_none_data),
        ("load_history: swallows Supabase exception",     test_load_history_swallows_exception),
        # save_papers
        ("save_papers: returns 0 for empty list",         test_save_papers_returns_zero_on_empty_list),
        ("save_papers: calls upsert with correct args",   test_save_papers_calls_upsert),
        ("save_papers: returns 0 on exception",           test_save_papers_returns_zero_on_exception),
    ]

    results = [check(label, fn) for label, fn in checks]
    passed  = sum(results)
    total   = len(results)

    print(f"\n── Result: {passed}/{total} tests passed ────────────────────\n")

    if passed == total:
        print("  All memory tests passed.\n")
        sys.exit(0)
    else:
        print("  Some tests failed — review the output above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()