"""Benchmark: single-agent baseline vs multi-agent workflow."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]


def _extract_quality_score(state: ResearchState) -> float | None:
    """Pull quality score from Critic's AgentResult if it exists."""
    for result in reversed(state.agent_results):
        score = result.metadata.get("quality_score")
        if score is not None:
            try:
                return float(score)
            except (TypeError, ValueError):
                pass
    return None


def _total_cost(state: ResearchState) -> float | None:
    """Sum cost_usd from all agent results."""
    total = 0.0
    found = False
    for result in state.agent_results:
        cost = result.metadata.get("cost_usd")
        if cost is not None:
            total += float(cost)
            found = True
    return round(total, 6) if found else None


def _citation_coverage(state: ResearchState) -> str:
    """Fraction of source titles mentioned in the final answer."""
    if not state.sources or not state.final_answer:
        return ""
    answer_lower = state.final_answer.lower()
    hits = sum(1 for s in state.sources if s.title.lower() in answer_lower)
    return f"{hits}/{len(state.sources)} sources cited"


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, cost, quality score, and citation coverage."""
    logger.info("Benchmark starting — run_name=%s", run_name)
    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    quality = _extract_quality_score(state)
    cost = _total_cost(state)
    citation_note = _citation_coverage(state)
    errors_note = f"{len(state.errors)} error(s)" if state.errors else ""
    notes_parts = [p for p in [citation_note, errors_note] if p]

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=round(latency, 2),
        estimated_cost_usd=cost,
        quality_score=quality,
        notes=" | ".join(notes_parts),
    )
    logger.info(
        "Benchmark done — run=%s latency=%.2fs cost=%s quality=%s",
        run_name,
        latency,
        cost,
        quality,
    )
    return state, metrics
