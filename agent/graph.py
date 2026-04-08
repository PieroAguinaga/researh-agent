"""
agent/graph.py

Multi-agent LangGraph orchestration for IATA.

Graph topology
──────────────
                    ┌─────────────────┐
    START ──────► │  supervisor_node  │
                    └────────┬────────┘
                             │  routes to one of:
                ┌────────────┼────────────┐
                ▼            ▼            ▼
         search_node   summarizer_node   END
                │            │
                └────────────┘
                             │
                        supervisor_node  (re-evaluates after tool run)

How it works
────────────
1. supervisor_node: an LLM decides which specialist to call next, or ends.
2. search_node:     runs search_papers tool, saves results to Supabase.
3. summarizer_node: runs summarize_paper / summarize_multiple_papers tool.
4. After each specialist finishes, control returns to supervisor_node which
   can either route to another specialist or end the turn.

Persistent memory
─────────────────
Conversation history is loaded from Supabase before graph invocation and
saved back after. See agent/tools/supabase_memory.py.
"""

from typing import Literal
from langgraph.graph import END, START, StateGraph


from agent.state import AgentState
from agent.nodes.supervisor_node import supervisor_node
from agent.nodes.summarize_agent_node import summarizer_agent_node
from agent.nodes.search_agent_node import search_agent_node




# ── Routing function ───────────────────────────────────────────────────────────

def route_from_supervisor(state: AgentState) -> Literal["search_agent", "summarizer_agent", "__end__"]:
    """Map the supervisor's next_node decision to a LangGraph edge target."""
    mapping = {
        "search_agent":     "search_agent",
        "summarizer_agent": "summarizer_agent",
        "FINISH":           END,
    }
    return mapping.get(state.get("next_node", "FINISH"), END)


# ── Graph construction ─────────────────────────────────────────────────────────

def build_graph():
    """
    Compile and return the multi-agent LangGraph.

    Graph structure:
        START → supervisor → (search_agent | summarizer_agent | END)
        search_agent    → supervisor
        summarizer_agent → supervisor
    """
    graph = StateGraph(AgentState)

    graph.add_node("supervisor",       supervisor_node)
    graph.add_node("search_agent",     search_agent_node)
    graph.add_node("summarizer_agent", summarizer_agent_node)

    # Entry point
    graph.add_edge(START, "supervisor")

    # Supervisor routes conditionally
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "search_agent":     "search_agent",
            "summarizer_agent": "summarizer_agent",
            END:                END,
        },
    )

    # After each specialist, return to supervisor for re-evaluation
    graph.add_edge("search_agent",     "supervisor")
    graph.add_edge("summarizer_agent", "supervisor")

    return graph.compile()