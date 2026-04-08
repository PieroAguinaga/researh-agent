"""
Summarizer agent node for the IATA agent graph.

Receives papers from state or conversation context and uses
summarize_paper / summarize_multiple_papers tools to produce
structured summaries, returning them as new messages.
"""
import json
import logging
from typing import Any

from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from agent.state import AgentState
from agent.prompts import SUMMARIZER_AGENT_SYSTEM
from agent.llm import get_llm
from agent.tools.summarizer import summarize_paper, summarize_multiple_papers

logger = logging.getLogger(__name__)


def summarizer_agent_node(state: AgentState) -> dict[str, Any]:
    """
    Specialist node that summarizes papers either from state or conversation context.
    """
    llm = get_llm(temperature=0)
    agent = create_react_agent(llm, tools=[summarize_paper, summarize_multiple_papers])

    # Inject last_papers into the system prompt so the LLM can reference them
    papers_context = ""
    if state.get("last_papers"):
        papers_json = json.dumps(state["last_papers"], indent=2)
        papers_context = f"\n\nPapers available for summarization:\n{papers_json}"

    system_msg = SystemMessage(content=SUMMARIZER_AGENT_SYSTEM + papers_context)
    messages = [system_msg] + state["messages"]

    result = agent.invoke({"messages": messages})

    return {"messages": result["messages"]}