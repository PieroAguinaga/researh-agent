"""
agent/iata_agent.py

High-level wrapper around the compiled LangGraph multi-agent graph.
Provides the IATAAgent class, which manages conversation state and
Supabase-backed memory across turns.

Flow per invocation:
    1. Load conversation history from Supabase.
    2. Build the initial AgentState with history + new user message.
    3. Run the LangGraph graph (with recursion limit).
    4. Extract the final AI reply and tool calls used.
    5. Persist the turn (human + AI) back to Supabase.

Typical usage:
    agent = IATAAgent()
    result = agent.invoke("Find papers on sparse attention", thread_id="session-123")
    print(result["reply"])
    print(result["tool_calls_used"])
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage, AIMessage, AnyMessage

from agent.graph import build_graph
from agent.state import AgentState
from agent.tools.supabase_memory import load_history, save_turn

logger = logging.getLogger(__name__)

# ── Public interface ───────────────────────────────────────────────────────────

class IATAAgent:
    """
    High-level wrapper around the compiled LangGraph.
    Handles Supabase memory load/save around each invocation.

    Usage:
        agent = IATAAgent()
        result = agent.invoke("Find papers on sparse attention", thread_id="session-123")
        print(result["reply"])
        print(result["tool_calls_used"])
    """

    def __init__(self) -> None:
        self._graph = build_graph()
        logger.info("IATA multi-agent graph initialised")

    def invoke(self, user_message: str, thread_id: str = "default") -> dict[str, Any]:
        """
        Run one full turn of the agent with persistent Supabase memory.

        Args:
            user_message: The user's natural language request.
            thread_id:    Conversation identifier (maps to Supabase rows).

        Returns:
            Dict with keys:
              - reply (str):            The agent's final text response.
              - tool_calls_used (list): Names of tools invoked this turn.
              - thread_id (str):        The session identifier.
        """
        # 1. Load conversation history from Supabase
        history_rows = load_history(thread_id)
        history_messages = _rows_to_messages(history_rows)

        # 2. Build initial state
        initial_state: AgentState = {
            "messages": history_messages + [HumanMessage(content=user_message)],
            "thread_id": thread_id,
            "last_papers": [],
            "next_node": "",
        }

        # 3. Run the graph
        result = self._graph.invoke(
            initial_state,
            config={"recursion_limit": 15},
        )

        # 4. Extract the final AI reply
        final_reply = _extract_last_ai_content(result["messages"])

        # 5. Collect which tools were used this turn
        tool_calls_used = _extract_tool_call_names(result["messages"])

        # 6. Persist this turn to Supabase
        save_turn(thread_id, "human", user_message)
        save_turn(thread_id, "ai", final_reply, tool_calls=[
            {"name": name} for name in tool_calls_used
        ])

        return {
            "reply": final_reply,
            "tool_calls_used": tool_calls_used,
            "thread_id": thread_id,
        }


# ── Private helpers ────────────────────────────────────────────────────────────

def _rows_to_messages(rows: list[dict]) -> list[AnyMessage]:
    """Convert Supabase history rows back into LangChain message objects."""
    messages: list[AnyMessage] = []
    for row in rows:
        role = row.get("role", "")
        content = row.get("content", "")
        if role == "human":
            messages.append(HumanMessage(content=content))
        elif role == "ai":
            messages.append(AIMessage(content=content))
    return messages


def _extract_last_ai_content(messages: list[AnyMessage]) -> str:
    """Walk messages in reverse and return the last AIMessage content string."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
            return msg.content.strip()
    return "I was unable to generate a response. Please try again."


def _extract_tool_call_names(messages: list[AnyMessage]) -> list[str]:
    """Return deduplicated list of tool names called across all messages."""
    seen: set[str] = set()
    names: list[str] = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                if name and name not in seen:
                    seen.add(name)
                    names.append(name)
    return names
