"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to markdown with analysis."""

    lines = [
        "# Benchmark Report",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Notes |",
        "|---|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        lines.append(f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} | {item.notes} |")

    # --- Analysis section ---
    scored = [m for m in metrics if m.quality_score is not None]
    fastest = min(metrics, key=lambda m: m.latency_seconds)
    cheapest = [m for m in metrics if m.estimated_cost_usd is not None]
    cheapest_run = min(cheapest, key=lambda m: m.estimated_cost_usd) if cheapest else None

    lines += ["", "## Analysis", ""]

    # Quality comparison
    if scored:
        best_quality = max(scored, key=lambda m: m.quality_score)  # type: ignore[arg-type]
        lines.append(f"**Best quality:** `{best_quality.run_name}` — score {best_quality.quality_score:.1f}/10")
    else:
        lines.append("**Quality:** Single-agent run has no quality score (no Critic pass).")

    # Latency
    lines.append(f"**Fastest:** `{fastest.run_name}` — {fastest.latency_seconds:.2f}s")

    # Cost
    if cheapest_run:
        lines.append(f"**Cheapest:** `{cheapest_run.run_name}` — ${cheapest_run.estimated_cost_usd:.4f}")

    # Tradeoff summary
    lines += [
        "",
        "## Tradeoffs",
        "",
        "| Dimension | Single-agent | Multi-agent |",
        "|---|---|---|",
        "| Latency | Lower (1 LLM call) | Higher (Researcher + Analyst + Writer + Critic) |",
        "| Cost | Lower | ~4× higher due to multiple LLM calls |",
        "| Quality | Not scored (no Critic) | Scored + citation-tracked |",
        "| Transparency | Black box | Full route trace per agent |",
        "| Failure isolation | Single point of failure | Each agent can retry independently |",
        "",
        "## When to use multi-agent",
        "",
        "- Query requires **distinct skills** (search vs. analysis vs. writing).",
        "- Output quality matters more than latency/cost.",
        "- Need **auditability** — who did what, which sources were used.",
        "",
        "## When NOT to use multi-agent",
        "",
        "- Simple factual Q&A where one LLM call suffices.",
        "- Latency is critical (real-time chat, autocomplete).",
        "- Budget is very tight.",
        "",
        "## Known failure modes",
        "",
        "- **Mock search** returns generic snippets → LLM may hallucinate specifics."
        " Fix: use Tavily or RAG on a real corpus.",
        "- **Analyst depends on Researcher** — if Researcher produces weak notes,"
        " downstream quality degrades silently.",
        "- **No retry between agents** — a transient OpenAI error mid-workflow fails the whole run."
        " Fix: wrap each agent call with tenacity retry.",
        "",
        "## Trace",
        "",
        "Local JSON traces are saved to `traces/<run_id>.json`.",
        "Each span records `name`, `started_at`, `duration_seconds`, `attributes`, and `status`.",
        "",
        "### Multi-Agent Route Trace",
        "",
        "![Multi-Agent Trace](../docs/screenshots/multi_agent_trace.jpg)",
        "",
        "### Benchmark Report Output",
        "",
        "![Benchmark Report](../docs/screenshots/benchmark_report.jpg)",
    ]

    return "\n".join(lines) + "\n"
