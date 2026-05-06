"""Researcher agent — search + LLM summarisation."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are a meticulous research assistant. Given a topic and a set of source snippets,
produce concise, factual research notes. Structure your notes with:
- Key findings (bullet list)
- Important concepts / definitions
- Gaps or open questions

Be objective. Cite sources by their title when relevant.
"""


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self) -> None:
        self._search = SearchClient()
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        query = state.request.query
        max_src = state.request.max_sources

        with trace_span("researcher.search", {"query": query}) as span:
            sources = self._search.search(query, max_results=max_src)
            span["attributes"]["n_sources"] = len(sources)

        # Persist sources in state
        state.sources.extend(sources)

        # Build context block from snippets
        snippets_block = "\n\n".join(
            f"[{i+1}] **{src.title}**\n{src.snippet}" for i, src in enumerate(sources)
        )

        user_prompt = f"Research topic: {query}\n\nSources:\n{snippets_block}"

        with trace_span("researcher.llm") as span:
            response = self._llm.complete(_SYSTEM, user_prompt)
            span["attributes"]["tokens_in"] = response.input_tokens
            span["attributes"]["tokens_out"] = response.output_tokens

        state.research_notes = response.content

        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=response.content,
                metadata={
                    "n_sources": len(sources),
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "researcher",
            {"n_sources": len(sources), "notes_len": len(response.content)},
        )
        logger.info("Researcher done — %d sources, %d chars of notes", len(sources), len(response.content))
        return state
