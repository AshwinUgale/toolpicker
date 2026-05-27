"""EvalRunner - plays one Benchmark's cases through one ToolPicker.

Measures wall-clock latency per case (via ``time.perf_counter()``) and
records the token cost when a token_budget is in play. Pure orchestration;
metric computation lives in ``evals.metrics``.
"""

from __future__ import annotations

import time

from evals.schema import Case, CaseResult
from toolpicker.packer import count_tokens
from toolpicker.router import ToolPicker

__all__ = ["EvalRunner"]


class EvalRunner:
    """Run a list of Cases through a ToolPicker, producing CaseResults.

    Args:
        picker: The configured ``ToolPicker``.
        k: Top-k for retrieval.
        token_budget: Optional. Passed through to ``picker.select(token_budget=...)``.
            When set, each ``CaseResult.token_cost`` is the sum of the
            returned tools' serialised token sizes (using the default
            OpenAI-envelope serialiser).
    """

    def __init__(
        self,
        picker: ToolPicker,
        *,
        k: int = 5,
        token_budget: int | None = None,
    ) -> None:
        self._picker = picker
        self._k = k
        self._token_budget = token_budget

    def run_case(self, case: Case) -> CaseResult:
        """Run a single case and return its result."""
        t0 = time.perf_counter()
        tools = self._picker.select(case.query, k=self._k, token_budget=self._token_budget)
        latency_ms = (time.perf_counter() - t0) * 1000
        token_cost: int | None = None
        if self._token_budget is not None:
            token_cost = sum(count_tokens(t) for t in tools)
        return CaseResult(
            case=case,
            retrieved_tool_ids=[t.id for t in tools],
            latency_ms=latency_ms,
            token_cost=token_cost,
        )

    def run(self, cases: list[Case]) -> list[CaseResult]:
        """Run every case in order. Returns one CaseResult per case."""
        return [self.run_case(c) for c in cases]
