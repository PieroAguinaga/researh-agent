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
    llm = get_llm(temperature=0)
    structured_llm = llm.with_structured_output(SupervisorDecision)

    # ── Contexto explícito del estado actual ──────────────────────────
    papers_found    = len(state.get("papers", []))
    summaries_done  = len(state.get("summaries", []))
    
    state_context = f"""
CURRENT STATE (use this to decide if work is already done):
- Papers retrieved: {papers_found}
- Summaries generated: {summaries_done}

ROUTING RULES:
1. If the user wants papers AND papers_found == 0  → search_agent
2. If papers_found > 0 AND summaries_done == 0     → summarizer_agent
3. If summaries_done > 0                           → FINISH
4. If the user only asked a question (no search/summarize needed) → FINISH
"""

    system_msg = SystemMessage(content=SUPERVISOR_SYSTEM + state_context)
    messages   = [system_msg] + state["messages"]

    try:
        decision: SupervisorDecision = structured_llm.invoke(messages)
        next_node = decision.next
        reasoning = decision.reasoning
    except Exception as e:
        logger.warning("Supervisor structured output failed: %s", str(e))
        next_node = "FINISH"
        reasoning = "Routing decision unavailable."

    logger.info(
        "Supervisor decision → %s | reason: %s | papers=%d summaries=%d",
        next_node, reasoning, papers_found, summaries_done
    )

    return {"next_node": next_node}