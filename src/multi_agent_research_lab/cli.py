"""Command-line entrypoint for the lab."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import reset_tracer
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    reset_tracer()  # fresh trace file for each run


# ---------------------------------------------------------------------------
# baseline command — real single-agent LLM call
# ---------------------------------------------------------------------------

@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent LLM baseline (no orchestration)."""
    _init()
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)

    console.print(f"[bold]Baseline[/bold] — running single LLM call for: {query!r}")
    try:
        llm = LLMClient()
        response = llm.complete(
            system_prompt=(
                "You are an expert research assistant. Answer the user's question thoroughly "
                "in approximately 500 words. Include key findings, important concepts, and sources."
            ),
            user_prompt=query,
        )
        state.final_answer = response.content
        state.agent_results  # no agents, direct call
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc

    console.print(Panel.fit(state.final_answer or "", title="Single-Agent Baseline"))
    if response.cost_usd:
        console.print(f"[dim]cost: ${response.cost_usd:.6f} | in: {response.input_tokens} | out: {response.output_tokens}[/dim]")


# ---------------------------------------------------------------------------
# multi-agent command
# ---------------------------------------------------------------------------

@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    critic: Annotated[bool, typer.Option("--critic/--no-critic", help="Enable Critic agent")] = False,
) -> None:
    """Run the full multi-agent workflow (Supervisor → Researcher → Analyst → Writer)."""
    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow(use_critic=critic)

    console.print(f"[bold]Multi-Agent[/bold] — query: {query!r}  critic={critic}")
    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc

    # Display final answer
    console.print(Panel.fit(result.final_answer or "(no answer)", title="Final Answer"))

    # Display route trace table
    table = Table(title="Agent Trace")
    table.add_column("Step", style="dim")
    table.add_column("Route")
    for i, route in enumerate(result.route_history, 1):
        table.add_row(str(i), route)
    console.print(table)

    if result.errors:
        console.print(f"[red]Errors: {result.errors}[/red]")


# ---------------------------------------------------------------------------
# benchmark command
# ---------------------------------------------------------------------------

@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    output: Annotated[str, typer.Option("--output", "-o", help="Report file")] = "benchmark_report.md",
    critic: Annotated[bool, typer.Option("--critic/--no-critic")] = True,
) -> None:
    """Benchmark single-agent vs multi-agent and save a markdown report."""
    _init()
    console.print(f"[bold]Benchmark[/bold] — query: {query!r}")

    # --- Single-agent runner ---
    def _single_runner(q: str) -> ResearchState:
        st = ResearchState(request=ResearchQuery(query=q))
        llm = LLMClient()
        resp = llm.complete(
            "You are an expert research assistant. Answer in ~500 words with key findings and sources.",
            q,
        )
        st.final_answer = resp.content
        from multi_agent_research_lab.core.schemas import AgentName, AgentResult
        st.agent_results.append(
            AgentResult(agent=AgentName.RESEARCHER, content=resp.content,
                        metadata={"cost_usd": resp.cost_usd})
        )
        return st

    # --- Multi-agent runner ---
    def _multi_runner(q: str) -> ResearchState:
        st = ResearchState(request=ResearchQuery(query=q))
        return MultiAgentWorkflow(use_critic=critic).run(st)

    console.print("Running [yellow]single-agent[/yellow] baseline…")
    single_state, single_metrics = run_benchmark("single-agent", query, _single_runner)

    console.print("Running [green]multi-agent[/green] workflow…")
    multi_state, multi_metrics = run_benchmark("multi-agent", query, _multi_runner)

    # Render and save report
    report_md = render_markdown_report([single_metrics, multi_metrics])
    store = LocalArtifactStore()
    path = store.write_text(output, report_md)
    console.print(f"\n[green]Report saved → {path}[/green]")
    console.print(report_md)


if __name__ == "__main__":
    app()
