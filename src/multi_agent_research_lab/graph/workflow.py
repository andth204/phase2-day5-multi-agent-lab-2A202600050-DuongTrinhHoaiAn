"""Multi-agent workflow — pure-Python state machine.

Architecture:
    Supervisor → Researcher → Supervisor → Analyst → Supervisor → Writer → Supervisor → DONE
                                                                          ↳ (optional) Critic

LangGraph integration is straightforward if desired:
replace the while-loop with StateGraph(ResearchState) nodes + conditional edges.
"""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import ROUTE_DONE, SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph."""

    def __init__(self, use_critic: bool = False) -> None:
        self._supervisor = SupervisorAgent()
        self._researcher = ResearcherAgent()
        self._analyst = AnalystAgent()
        self._writer = WriterAgent()
        self._critic = CriticAgent() if use_critic else None
        self._workers = {
            "researcher": self._researcher,
            "analyst": self._analyst,
            "writer": self._writer,
        }

    def build(self) -> "MultiAgentWorkflow":
        """Return self — the state-machine *is* the graph.

        To migrate to LangGraph:
          graph = StateGraph(ResearchState)
          graph.add_node("supervisor", self._supervisor.run)
          graph.add_node("researcher", self._researcher.run)
          ...
          graph.add_conditional_edges("supervisor", lambda s: s.route_history[-1], {...})
          graph.set_entry_point("supervisor")
        """
        return self

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph: supervisor orchestrates workers until DONE."""
        max_iter = get_settings().max_iterations

        with trace_span("workflow.run", {"query": state.request.query}) as span:
            while state.iteration < max_iter:
                # --- Supervisor decides next route ---
                state = self._supervisor.run(state)
                current_route = state.route_history[-1]

                if current_route == ROUTE_DONE:
                    logger.info("Workflow complete after %d iterations", state.iteration)
                    break

                # --- Dispatch to the appropriate worker ---
                worker = self._workers.get(current_route)
                if worker is None:
                    state.errors.append(f"Unknown route: {current_route}")
                    logger.error("Unknown route=%s — aborting", current_route)
                    break

                with trace_span(f"worker.{current_route}"):
                    state = worker.run(state)
            else:
                # Exceeded max_iterations without DONE
                logger.warning("Workflow hit max_iterations=%d without DONE", max_iter)
                state.errors.append("max_iterations reached")

            # Optional critic pass
            if self._critic and state.final_answer:
                state = self._critic.run(state)

            span["attributes"]["total_iterations"] = state.iteration
            span["attributes"]["route_history"] = state.route_history

        return state
