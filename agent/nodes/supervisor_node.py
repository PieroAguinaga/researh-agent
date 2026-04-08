"""
Supervisor node for the IATA agent graph.

Reads the conversation history and uses an LLM with structured output
to decide which specialist agent to invoke next (search_agent or
summarizer_agent), or to end the turn (FINISH).
"""

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage

from agent.state import AgentState
from agent.prompts import SUPERVISOR_SYSTEM
from agent.llm import get_llm

logger = logging.getLogger(__name__)

class SupervisorDecision(BaseModel):
    reasoning: str = Field(description="One sentence explaining the routing decision")
    next: Literal["search_agent", "summarizer_agent", "FINISH"] = Field(
        description="The next agent to invoke, or FINISH if the task is complete"
    )


def supervisor_node(state: AgentState) -> dict[str, Any]:
    """
    Reads the conversation and decides which specialist to invoke next,
    or ends the turn if the task is complete.
    """
    llm = get_llm(temperature=0)
    structured_llm = llm.with_structured_output(SupervisorDecision)
    
    messages = [SystemMessage(content=SUPERVISOR_SYSTEM)] + state["messages"]

    try:
        decision: SupervisorDecision = structured_llm.invoke(messages)
        next_node = decision.next
        reasoning = decision.reasoning
    except Exception as e:
        logger.warning("Supervisor structured output failed: %s", str(e))
        next_node = "FINISH"
        reasoning = "Routing decision unavailable."

    logger.info("Supervisor decision → %s | reason: %s", next_node, reasoning)

    return {"next_node": next_node}