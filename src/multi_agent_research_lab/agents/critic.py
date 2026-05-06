"""Critic agent — fact-checks and scores the final answer."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are a rigorous fact-checker and quality reviewer. Given:
- The original research notes
- The final answer

Produce a brief review with:
1. **Accuracy** (0-10): Do claims in the answer match the notes?
2. **Completeness** (0-10): Are key points from the notes covered?
3. **Clarity** (0-10): Is the answer well-written for the target audience?
4. **Issues Found**: List any hallucinations, missing citations, or unclear claims.
5. **Overall Score** (0-10): Weighted average.

End with: SCORE: <number>
"""


class CriticAgent(BaseAgent):
    """Fact-checks the final answer and produces a quality score."""

    name = "critic"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        if not state.final_answer:
            logger.warning("Critic called but no final_answer exists — skipping")
            return state

        user_prompt = (
            f"Research Notes:\n{state.research_notes or '(none)'}\n\n"
            f"Final Answer:\n{state.final_answer}"
        )

        with trace_span("critic.llm") as span:
            response = self._llm.complete(_SYSTEM, user_prompt)
            span["attributes"]["tokens_in"] = response.input_tokens

        # Extract numeric score from "SCORE: X" at end of response
        score: float | None = None
        for line in reversed(response.content.splitlines()):
            if line.strip().upper().startswith("SCORE:"):
                try:
                    score = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
                break

        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=response.content,
                metadata={"quality_score": score, "cost_usd": response.cost_usd},
            )
        )
        state.add_trace_event("critic", {"quality_score": score})
        logger.info("Critic done — quality_score=%s", score)
        return state
