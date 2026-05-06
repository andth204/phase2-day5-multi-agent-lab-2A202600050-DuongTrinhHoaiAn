"""Analyst agent — extracts insights from research notes."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are a critical analyst. Given research notes on a topic, produce a structured analysis:
1. **Main Claims** — list the 3-5 strongest factual claims supported by the notes.
2. **Evidence Quality** — rate each claim's evidence as Strong / Moderate / Weak.
3. **Conflicting Viewpoints** — note any disagreements or tensions in the sources.
4. **Key Takeaways** — 2-3 sentences a decision-maker should remember.

Be precise and highlight weak evidence honestly.
"""


class AnalystAgent(BaseAgent):
    """Turns research notes into structured analytical insights."""

    name = "analyst"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        if not state.research_notes:
            state.errors.append("AnalystAgent: research_notes is empty — skipping analysis.")
            logger.warning("Analyst called with no research notes")
            state.analysis_notes = "(no research notes available)"
            return state

        user_prompt = (
            f"Topic: {state.request.query}\n\n"
            f"Research Notes:\n{state.research_notes}"
        )

        with trace_span("analyst.llm") as span:
            response = self._llm.complete(_SYSTEM, user_prompt)
            span["attributes"]["tokens_in"] = response.input_tokens
            span["attributes"]["tokens_out"] = response.output_tokens

        state.analysis_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                metadata={"cost_usd": response.cost_usd},
            )
        )
        state.add_trace_event("analyst", {"analysis_len": len(response.content)})
        logger.info("Analyst done — %d chars of analysis", len(response.content))
        return state
