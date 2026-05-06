"""Tracing hooks — local JSON file provider.

Mỗi lần chạy tạo một file trace riêng tại traces/<run_id>.json.
Không cần API key, hoàn toàn offline.

Cấu trúc file:
{
  "run_id": "20260506_153012_abc123",
  "started_at": "2026-05-06T15:30:12Z",
  "spans": [
    {
      "name": "researcher.search",
      "started_at": "...",
      "duration_seconds": 0.42,
      "attributes": {"query": "...", "n_sources": 4},
      "status": "ok"   # hoặc "error"
    },
    ...
  ]
}
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)

_TRACES_DIR = Path("traces")


class _LocalTracer:
    """Singleton-ish tracer — one instance per process run."""

    def __init__(self) -> None:
        _TRACES_DIR.mkdir(exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        short_id = uuid.uuid4().hex[:6]
        self.run_id = f"{ts}_{short_id}"
        self._path = _TRACES_DIR / f"{self.run_id}.json"
        self._data: dict[str, Any] = {
            "run_id": self.run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "spans": [],
        }
        self._flush()
        logger.info("Tracer initialised — file=%s", self._path)

    def record(self, span: dict[str, Any]) -> None:
        self._data["spans"].append(span)
        self._flush()

    def _flush(self) -> None:
        self._path.write_text(json.dumps(self._data, indent=2, default=str), encoding="utf-8")


# Module-level singleton — reset per process
_tracer: _LocalTracer | None = None


def get_tracer() -> _LocalTracer:
    global _tracer
    if _tracer is None:
        _tracer = _LocalTracer()
    return _tracer


def reset_tracer() -> None:
    """Call this at the start of a new run (e.g. in CLI) to get a fresh trace file."""
    global _tracer
    _tracer = _LocalTracer()


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Record a span to the local JSON trace file.

    Usage (unchanged from skeleton):
        with trace_span("researcher.llm", {"query": q}) as span:
            span["attributes"]["tokens_out"] = 123
    """
    tracer = get_tracer()
    started_wall = datetime.now(timezone.utc).isoformat()
    started_perf = perf_counter()

    span: dict[str, Any] = {
        "name": name,
        "started_at": started_wall,
        "duration_seconds": None,
        "attributes": attributes or {},
        "status": "ok",
    }
    try:
        yield span
    except Exception as exc:
        span["status"] = "error"
        span["error"] = str(exc)
        raise
    finally:
        span["duration_seconds"] = round(perf_counter() - started_perf, 4)
        tracer.record(span)
        logger.debug("span=%s duration=%.4fs status=%s", name, span["duration_seconds"], span["status"])