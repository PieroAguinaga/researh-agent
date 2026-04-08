from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from typing import Annotated, TypedDict

class AgentState(TypedDict):
    """
    Shared state threaded through every node in the graph.

    messages:       Full conversation, managed by LangGraph's add_messages reducer.
    thread_id:      Identifies the Supabase conversation row for this session.
    last_papers:    Papers retrieved in the current turn (passed between nodes).
    next_node:      Routing decision made by the supervisor.
    """
    messages: Annotated[list[AnyMessage], add_messages]
    thread_id: str
    last_papers: list[dict]
    next_node: str
