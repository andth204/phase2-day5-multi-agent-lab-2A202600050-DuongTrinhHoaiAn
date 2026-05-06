"""Search client — Tavily (if key available) with mock fallback."""

from __future__ import annotations

import logging

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mock search — no external key required
# ---------------------------------------------------------------------------

def _mock_search(query: str, max_results: int) -> list[SourceDocument]:
    """Return plausible placeholder documents so the pipeline runs without Tavily."""
    logger.warning("Using mock search (no TAVILY_API_KEY set). Results are illustrative only.")
    topics = query.strip().split()[:4]
    tag = " ".join(topics)
    return [
        SourceDocument(
            title=f"Overview of {tag} (Mock Source {i + 1})",
            url=f"https://example.com/mock-{i + 1}",
            snippet=(
                f"This article discusses key aspects of {tag}. "
                f"It covers recent advances, practical applications, and open challenges. "
                f"[Mock result {i + 1} — replace TAVILY_API_KEY for real web search]"
            ),
        )
        for i in range(min(max_results, 4))
    ]


# ---------------------------------------------------------------------------
# Tavily search
# ---------------------------------------------------------------------------

def _tavily_search(query: str, max_results: int, api_key: str) -> list[SourceDocument]:
    try:
        from tavily import TavilyClient  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("tavily-python is required. Run: pip install tavily-python") from exc

    client = TavilyClient(api_key=api_key)
    result = client.search(query=query, max_results=max_results)
    docs: list[SourceDocument] = []
    for item in result.get("results", []):
        docs.append(
            SourceDocument(
                title=item.get("title", "Untitled"),
                url=item.get("url"),
                snippet=item.get("content", ""),
            )
        )
    logger.info("Tavily returned %d results for query=%r", len(docs), query)
    return docs


# ---------------------------------------------------------------------------
# Public client
# ---------------------------------------------------------------------------

class SearchClient:
    """Provider-agnostic search client.

    Uses Tavily when TAVILY_API_KEY is configured, otherwise falls back to a mock.
    """

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        settings = get_settings()
        if settings.tavily_api_key:
            return _tavily_search(query, max_results, settings.tavily_api_key)
        return _mock_search(query, max_results)
