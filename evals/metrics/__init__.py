"""Metrics over a list of CaseResults.

Each metric is a pure function: takes the run output, returns a JSON-
serialisable dict slice. The CLI sums them into the final result block.

Shipped at v0.4:
* `precision_at_k(results, k)` - fraction of cases where any expected
  tool appears in the top-k retrieved.
* `mean_reciprocal_rank(results)` - average 1/rank of the first expected
  tool; 1.0 = always top-1, 0.0 = never retrieved.
* `latency_stats(results)` - p50 / p95 / mean retrieval latency.
* `tokens_saved(results, full_corpus_tokens)` - how much we cut off the
  "send every tool to the LLM" cost. Higher = better routing.
"""

from __future__ import annotations

from evals.schema import CaseResult

__all__ = [
    "latency_stats",
    "mean_reciprocal_rank",
    "precision_at_k",
    "tokens_saved",
]


def precision_at_k(results: list[CaseResult], k: int) -> float:
    """Fraction of cases where >=1 expected tool is in the top-k retrieved.

    Treats each case as binary: hit or miss. The output is in [0.0, 1.0].
    Returns 0.0 on an empty input (rather than dividing by zero).
    """
    if not results or k <= 0:
        return 0.0
    hits = 0
    for r in results:
        topk = set(r.retrieved_tool_ids[:k])
        if any(exp in topk for exp in r.case.expected_tool_ids):
            hits += 1
    return hits / len(results)


def mean_reciprocal_rank(results: list[CaseResult]) -> float:
    """Mean of 1 / rank-of-first-relevant. Higher = better.

    A case where the first expected tool ranks #1 contributes 1.0. Rank #2
    contributes 0.5. Rank #5 contributes 0.2. Cases where no expected tool
    appears contribute 0.0.
    """
    if not results:
        return 0.0
    rr_sum = 0.0
    for r in results:
        expected = set(r.case.expected_tool_ids)
        for rank, tid in enumerate(r.retrieved_tool_ids, 1):
            if tid in expected:
                rr_sum += 1.0 / rank
                break
    return rr_sum / len(results)


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Nearest-rank percentile (no interpolation). Cheap and predictable."""
    if not sorted_vals:
        return 0.0
    if pct <= 0:
        return sorted_vals[0]
    if pct >= 100:
        return sorted_vals[-1]
    # Nearest-rank: ceil(pct/100 * N) - 1
    import math

    idx = max(0, math.ceil(pct / 100 * len(sorted_vals)) - 1)
    return sorted_vals[idx]


def latency_stats(results: list[CaseResult]) -> dict[str, float]:
    """p50 / p95 / mean retrieval latency in milliseconds."""
    if not results:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0}
    latencies = sorted(r.latency_ms for r in results)
    return {
        "p50_ms": _percentile(latencies, 50),
        "p95_ms": _percentile(latencies, 95),
        "mean_ms": sum(latencies) / len(latencies),
    }


def tokens_saved(
    results: list[CaseResult],
    *,
    full_corpus_tokens: int,
) -> dict[str, float]:
    """How many tokens routing saved vs sending every tool to the LLM.

    Args:
        results: CaseResults from a run where ``token_cost`` was populated
            (i.e. the picker was called with a token_budget or the cost
            was measured post-hoc).
        full_corpus_tokens: Tokens it would take to send the whole tool
            corpus (the baseline).

    Returns:
        Dict with the absolute saving, the percentage, and the average
        per-case cost after routing.
    """
    if not results or full_corpus_tokens <= 0:
        return {
            "baseline_tokens": float(full_corpus_tokens),
            "mean_routed_tokens": 0.0,
            "mean_saved_tokens": 0.0,
            "mean_saved_pct": 0.0,
        }
    costs = [r.token_cost for r in results if r.token_cost is not None]
    if not costs:
        return {
            "baseline_tokens": float(full_corpus_tokens),
            "mean_routed_tokens": 0.0,
            "mean_saved_tokens": 0.0,
            "mean_saved_pct": 0.0,
        }
    mean_routed = sum(costs) / len(costs)
    saved = full_corpus_tokens - mean_routed
    return {
        "baseline_tokens": float(full_corpus_tokens),
        "mean_routed_tokens": mean_routed,
        "mean_saved_tokens": saved,
        "mean_saved_pct": (saved / full_corpus_tokens) * 100,
    }
