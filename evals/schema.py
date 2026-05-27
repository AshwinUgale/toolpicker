"""Shapes that flow through the eval harness.

A `Case` is one labelled query - the input + the tool(s) the picker should
have surfaced. A `CaseResult` is what the picker actually returned plus the
latency it took. A `RunResult` is the whole run: every CaseResult plus the
aggregated metrics.

All three are plain dataclasses. JSON-serialisable via ``dataclasses.asdict``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["Case", "CaseResult", "RunResult"]


@dataclass(kw_only=True)
class Case:
    """One labelled eval case.

    Attributes:
        query: The user / agent input the router will see.
        expected_tool_ids: The tool id(s) considered correct for this query.
            Order doesn't matter; any expected id in the top-k retrieved
            counts as a hit for that case at that k.
        metadata: Free-form per-case info (domain, difficulty, etc.).
    """

    query: str
    expected_tool_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class CaseResult:
    """What happened when the picker ran one case."""

    case: Case
    retrieved_tool_ids: list[str]  # in rank order, top first
    latency_ms: float
    token_cost: int | None = None  # if a token_budget was applied


@dataclass(kw_only=True)
class RunResult:
    """The whole run - cases, picker config, and the metric block."""

    benchmark: str
    config: dict[str, Any]
    case_results: list[CaseResult]
    metrics: dict[str, Any]
