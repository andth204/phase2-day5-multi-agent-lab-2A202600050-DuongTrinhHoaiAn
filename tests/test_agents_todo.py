"""Tests for implemented agents (no longer TODO stubs)."""

import pytest

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routes_to_researcher_when_no_notes() -> None:
    """Supervisor should route to researcher when research_notes is None."""
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "researcher"


def test_supervisor_routes_to_analyst_after_research() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.research_notes = "Some research notes"
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "analyst"


def test_supervisor_routes_to_writer_after_analysis() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.research_notes = "Some research notes"
    state.analysis_notes = "Some analysis"
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "writer"


def test_supervisor_done_when_all_complete() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.research_notes = "Some research notes"
    state.analysis_notes = "Some analysis"
    state.final_answer = "The final answer"
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "done"


def test_supervisor_max_iterations_guard() -> None:
    """Supervisor should force DONE when max_iterations reached."""
    from multi_agent_research_lab.core.config import get_settings
    max_iter = get_settings().max_iterations
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.iteration = max_iter  # already at limit
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "done"


def test_supervisor_records_trace() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    result = SupervisorAgent().run(state)
    assert any(e["name"] == "supervisor" for e in result.trace)
