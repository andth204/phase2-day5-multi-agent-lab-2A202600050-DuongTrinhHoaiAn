"""Supervisor / router — decides which worker runs next."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

ROUTE_RESEARCHER = "researcher"
ROUTE_ANALYST = "analyst"
ROUTE_WRITER = "writer"
ROUTE_DONE = "done"


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop.

    Routing policy (rule-based, no LLM call needed):
      1. No research_notes  → researcher
      2. No analysis_notes  → analyst
      3. No final_answer    → writer
      4. All complete       → done
    Guardrail: if iteration >= max_iterations, route to writer immediately
    (to avoid infinite loops) or done if writer already ran.
    """

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        max_iter = get_settings().max_iterations

        # Guardrail: force termination when iteration limit reached
        if state.iteration >= max_iter:
            logger.warning("Max iterations (%d) reached — forcing DONE", max_iter)
            state.record_route(ROUTE_DONE)
            state.add_trace_event("supervisor", {"decision": ROUTE_DONE, "reason": "max_iterations"})
            return state

        # Rule-based routing
        if state.research_notes is None:
            next_route = ROUTE_RESEARCHER
        elif state.analysis_notes is None:
            next_route = ROUTE_ANALYST
        elif state.final_answer is None:
            next_route = ROUTE_WRITER
        else:
            next_route = ROUTE_DONE

        logger.info(
            "Supervisor → %s (iter=%d, research=%s, analysis=%s, answer=%s)",
            next_route,
            state.iteration,
            state.research_notes is not None,
            state.analysis_notes is not None,
            state.final_answer is not None,
        )

        state.record_route(next_route)
        state.add_trace_event(
            "supervisor",
            {
                "decision": next_route,
                "iteration": state.iteration,
                "has_research": state.research_notes is not None,
                "has_analysis": state.analysis_notes is not None,
                "has_answer": state.final_answer is not None,
            },
        )
        return state
