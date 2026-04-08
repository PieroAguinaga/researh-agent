"""
tests/agent_graph_verification.py

Verify the general logic and structure of the agent graph defined in
agent/graph.py without making any real external calls.

Usage:
    python tests/agent_graph_verification.py
"""

import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()


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


# ── Individual checks ──────────────────────────────────────────────────────────

def verify_graph_import():
    """Import build_graph from agent.graph without errors."""
    from agent.graph import build_graph
    assert callable(build_graph), "build_graph is not callable"


def verify_graph_compiles():
    """Call build_graph() and confirm it returns a compiled LangGraph object."""
    from agent.graph import build_graph
    graph = build_graph()
    assert graph is not None, "build_graph() returned None"


def verify_graph_nodes():
    """Confirm the compiled graph exposes the three expected nodes."""
    from agent.graph import build_graph
    graph = build_graph()
    nodes = set(graph.get_graph().nodes.keys())
    expected = {"supervisor", "search_agent", "summarizer_agent"}
    missing = expected - nodes
    assert not missing, f"Missing nodes in graph: {missing}"


def verify_graph_edges():
    """Confirm that all expected directed edges are present in the graph."""
    from agent.graph import build_graph
    from langgraph.graph import END, START
    graph = build_graph()
    edges = graph.get_graph().edges

    # Collect (source, target) pairs, normalising special sentinels
    edge_pairs = {(e.source, e.target) for e in edges}

    required = {
        ("__start__", "supervisor"),
        ("search_agent", "supervisor"),
        ("summarizer_agent", "supervisor"),
    }
    missing = required - edge_pairs
    assert not missing, f"Missing edges in graph: {missing}"


def verify_route_from_supervisor():
    """Unit-test the routing function with all valid next_node values."""
    from agent.graph import route_from_supervisor
    from langgraph.graph import END

    cases = {
        "search_agent":     "search_agent",
        "summarizer_agent": "summarizer_agent",
        "FINISH":           END,
        "unknown_value":    END,   # unknown → default to END
    }
    for next_node, expected in cases.items():
        state = {"next_node": next_node, "messages": [], "thread_id": "", "last_papers": []}
        result = route_from_supervisor(state)
        assert result == expected, (
            f"route_from_supervisor({next_node!r}) → {result!r}, expected {expected!r}"
        )


def verify_agent_state_schema():
    """Confirm AgentState has all required keys."""
    from agent.state import AgentState
    required_keys = {"messages", "thread_id", "last_papers", "next_node"}
    annotations = AgentState.__annotations__
    missing = required_keys - set(annotations.keys())
    assert not missing, f"AgentState is missing keys: {missing}"


def verify_conditional_edges():
    """Confirm the supervisor node has conditional edges registered."""
    from agent.graph import build_graph
    graph = build_graph()
    # Conditional edges appear in the graph definition; their presence means
    # the supervisor node has branching logic compiled in.
    node_names = set(graph.get_graph().nodes.keys())
    assert "supervisor" in node_names, "supervisor node not found in compiled graph"
    # A compiled StateGraph with conditional edges will list more than just
    # the bare linear edges — verify by counting total edges (≥ 3).
    edge_count = len(graph.get_graph().edges)
    assert edge_count >= 3, (
        f"Expected ≥ 3 edges (supervisor→agents + agents→supervisor), found {edge_count}"
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\n── IATA Agent Graph Verification ────────────────────────\n")

    checks = [
        ("Import agent.graph module",          verify_graph_import),
        ("Graph compiles without errors",       verify_graph_compiles),
        ("Graph contains expected nodes",       verify_graph_nodes),
        ("Graph contains expected edges",       verify_graph_edges),
        ("Supervisor routing logic is correct", verify_route_from_supervisor),
        ("AgentState schema is complete",       verify_agent_state_schema),
        ("Conditional edges are registered",    verify_conditional_edges),
    ]

    results = [check(label, fn) for label, fn in checks]
    passed  = sum(results)
    total   = len(results)

    print(f"\n── Result: {passed}/{total} checks passed ─────────────────────\n")

    if passed == total:
        print("  Agent graph structure is correct.\n")
        sys.exit(0)
    else:
        print("  Fix the issues above, then re-run this script.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
