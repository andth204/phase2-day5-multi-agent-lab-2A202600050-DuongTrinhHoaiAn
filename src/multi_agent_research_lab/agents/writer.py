"""Writer agent — synthesises research + analysis into the final answer."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are a professional technical writer. Using the research notes and analytical insights
provided, write a clear, well-structured response for the specified audience.

Requirements:
- Write in clear prose (no raw bullet dumps).
- Length: approximately {word_count} words.
- Audience: {audience}.
- End with a "Sources" section listing the titles used.
- Do not hallucinate facts not present in the notes.
"""


class WriterAgent(BaseAgent):
    """Produces the final answer from research and analysis notes."""

    name = "writer"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        research = state.research_notes or "(no research notes)"
        analysis = state.analysis_notes or "(no analysis notes)"
        source_titles = [s.title for s in state.sources]

        system = _SYSTEM.format(
            word_count=500,
            audience=state.request.audience,
        )
        user_prompt = (
            f"Topic: {state.request.query}\n\n"
            f"--- Research Notes ---\n{research}\n\n"
            f"--- Analysis ---\n{analysis}\n\n"
            f"Available sources: {', '.join(source_titles) or 'none'}"
        )

        with trace_span("writer.llm") as span:
            response = self._llm.complete(system, user_prompt)
            span["attributes"]["tokens_in"] = response.input_tokens
            span["attributes"]["tokens_out"] = response.output_tokens

        state.final_answer = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata={"cost_usd": response.cost_usd},
            )
        )
        state.add_trace_event("writer", {"answer_len": len(response.content)})
        logger.info("Writer done — %d chars final answer", len(response.content))
        return state
