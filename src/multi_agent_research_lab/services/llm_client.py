"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential
from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)

# gpt-4o-mini pricing (USD per 1M tokens, as of 2024)
_PRICE_INPUT_PER_M = 0.150
_PRICE_OUTPUT_PER_M = 0.600


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client — backed by OpenAI."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.openai_api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set. Add it to your .env file.")
        # Import here so the rest of the package still loads without openai installed.
        try:
            from openai import OpenAI  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("openai package is required. Run: pip install openai") from exc

        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model
        logger.info("LLMClient initialised — model=%s", self._model)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion with retry, timeout, and token logging."""
        logger.debug("LLM call — model=%s prompt_len=%d", self._model, len(user_prompt))

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            timeout=get_settings().timeout_seconds,
        )

        usage = response.usage
        in_tok = usage.prompt_tokens if usage else None
        out_tok = usage.completion_tokens if usage else None
        cost = None
        if in_tok is not None and out_tok is not None:
            cost = (in_tok * _PRICE_INPUT_PER_M + out_tok * _PRICE_OUTPUT_PER_M) / 1_000_000

        content = response.choices[0].message.content or ""
        logger.debug("LLM response — in_tok=%s out_tok=%s cost_usd=%.6f", in_tok, out_tok, cost or 0)
        return LLMResponse(content=content, input_tokens=in_tok, output_tokens=out_tok, cost_usd=cost)
