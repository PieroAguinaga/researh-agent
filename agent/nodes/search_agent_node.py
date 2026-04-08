"""
Search agent node for the IATA agent graph.

Invokes the search_papers tool to query arXiv and Semantic Scholar,
persists the results to Supabase, and appends a follow-up message
summarising the findings to the conversation history.
"""
import json
import logging
from typing import Any

from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from agent.state import AgentState
from agent.prompts import SEARCH_AGENT_SYSTEM
from agent.llm import get_llm
from agent.tools.search_tool import search_papers
from agent.tools.supabase_memory import save_papers

logger = logging.getLogger(__name__)



def search_agent_node(state: AgentState) -> dict[str, Any]:
    """
    Specialist node that calls search_papers and stores results in state.
    """
    llm = get_llm(temperature=0)
    # Forma moderna: llm ya sabe invocar tools directamente
    agent = create_react_agent(llm, tools=[search_papers])

    messages = [SystemMessage(content=SEARCH_AGENT_SYSTEM)] + state["messages"]
    result = agent.invoke({"messages": messages})

    # Extraer papers de los ToolMessages que dejó el agente
    retrieved_papers: list[dict] = []
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            try:
                papers = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                if isinstance(papers, list):
                    retrieved_papers = papers
                    save_papers(papers)
            except Exception as exc:
                logger.warning("Could not parse search tool output: %s", exc)

    return {
        "messages": result["messages"],
        "last_papers": retrieved_papers,
    }