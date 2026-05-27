"""End-to-end checks for the eval harness.

Runs the synthetic benchmark through the picker and verifies the metrics
module produces sane numbers. Cheap path only - no OpenAI key needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from evals.benchmarks import (
    GorillaAdapter,
    SyntheticBenchmark,
    ToolBenchAdapter,
    get_benchmark,
)
from evals.metrics import (
    latency_stats,
    mean_reciprocal_rank,
    precision_at_k,
    tokens_saved,
)
from evals.runners.base import EvalRunner
from evals.schema import Case, CaseResult

from toolpicker import HashEmbedder, ToolPicker


def _picker_for(bench: SyntheticBenchmark) -> ToolPicker:
    return ToolPicker(bench.tools(), embedder=HashEmbedder(dimensions=32))


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def test_synthetic_benchmark_loads() -> None:
    bench = SyntheticBenchmark()
    assert bench.name == "synthetic"
    assert len(bench.cases()) >= 10
    tools = bench.tools().tools()
    assert len(tools) >= 8


def test_get_benchmark_dispatch() -> None:
    assert isinstance(get_benchmark("synthetic"), SyntheticBenchmark)
    with pytest.raises(ValueError, match="unknown"):
        get_benchmark("nope")


def test_toolbench_adapter_errors_without_data() -> None:
    with pytest.raises(FileNotFoundError, match="ToolBench"):
        ToolBenchAdapter()


def test_gorilla_adapter_errors_without_data() -> None:
    with pytest.raises(FileNotFoundError, match="Gorilla"):
        GorillaAdapter()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def test_runner_produces_one_result_per_case() -> None:
    bench = SyntheticBenchmark()
    runner = EvalRunner(_picker_for(bench), k=3)
    results = runner.run(bench.cases())
    assert len(results) == len(bench.cases())


def test_runner_records_latency() -> None:
    bench = SyntheticBenchmark()
    runner = EvalRunner(_picker_for(bench), k=3)
    results = runner.run(bench.cases())
    assert all(r.latency_ms >= 0 for r in results)


def test_runner_records_token_cost_when_budget_set() -> None:
    bench = SyntheticBenchmark()
    runner = EvalRunner(_picker_for(bench), k=3, token_budget=2000)
    results = runner.run(bench.cases())
    # token_cost is populated only when token_budget is set.
    assert all(r.token_cost is not None for r in results)
    assert all((r.token_cost or 0) <= 2000 for r in results)


def test_runner_token_cost_none_when_no_budget() -> None:
    bench = SyntheticBenchmark()
    runner = EvalRunner(_picker_for(bench), k=3)
    results = runner.run(bench.cases())
    assert all(r.token_cost is None for r in results)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _hit_result(query: str, expected: str, retrieved: list[str]) -> CaseResult:
    return CaseResult(
        case=Case(query=query, expected_tool_ids=[expected]),
        retrieved_tool_ids=retrieved,
        latency_ms=1.0,
    )


def test_precision_at_1_all_hits() -> None:
    results = [_hit_result("q", "a", ["a", "b", "c"])]
    assert precision_at_k(results, 1) == 1.0


def test_precision_at_1_all_misses() -> None:
    results = [_hit_result("q", "a", ["b", "c"])]
    assert precision_at_k(results, 1) == 0.0


def test_precision_at_3_catches_mid_rank() -> None:
    results = [_hit_result("q", "a", ["b", "c", "a"])]
    assert precision_at_k(results, 3) == 1.0
    assert precision_at_k(results, 2) == 0.0


def test_precision_empty_input_returns_zero() -> None:
    assert precision_at_k([], 1) == 0.0


def test_precision_zero_k_returns_zero() -> None:
    results = [_hit_result("q", "a", ["a"])]
    assert precision_at_k(results, 0) == 0.0


def test_mrr_top_one_contributes_full() -> None:
    results = [_hit_result("q", "a", ["a", "b"])]
    assert mean_reciprocal_rank(results) == pytest.approx(1.0)


def test_mrr_rank_two_contributes_half() -> None:
    results = [_hit_result("q", "a", ["b", "a"])]
    assert mean_reciprocal_rank(results) == pytest.approx(0.5)


def test_mrr_no_hit_contributes_zero() -> None:
    results = [_hit_result("q", "a", ["b", "c"])]
    assert mean_reciprocal_rank(results) == pytest.approx(0.0)


def test_mrr_averages_across_cases() -> None:
    results = [
        _hit_result("q1", "a", ["a"]),  # 1.0
        _hit_result("q2", "b", ["x", "b"]),  # 0.5
        _hit_result("q3", "c", ["y", "z"]),  # 0.0
    ]
    assert mean_reciprocal_rank(results) == pytest.approx((1.0 + 0.5 + 0.0) / 3)


def test_mrr_empty_returns_zero() -> None:
    assert mean_reciprocal_rank([]) == 0.0


def test_latency_stats_shape() -> None:
    results = [
        CaseResult(
            case=Case(query=str(i), expected_tool_ids=[]),
            retrieved_tool_ids=[],
            latency_ms=float(i),
        )
        for i in range(1, 11)
    ]
    stats = latency_stats(results)
    assert "p50_ms" in stats
    assert "p95_ms" in stats
    assert "mean_ms" in stats
    assert stats["p50_ms"] <= stats["p95_ms"]


def test_latency_stats_empty() -> None:
    stats = latency_stats([])
    assert stats == {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0}


def test_tokens_saved_computes_pct() -> None:
    results = [
        CaseResult(
            case=Case(query="x", expected_tool_ids=[]),
            retrieved_tool_ids=[],
            latency_ms=1.0,
            token_cost=100,
        ),
        CaseResult(
            case=Case(query="x", expected_tool_ids=[]),
            retrieved_tool_ids=[],
            latency_ms=1.0,
            token_cost=200,
        ),
    ]
    out = tokens_saved(results, full_corpus_tokens=1000)
    assert out["mean_routed_tokens"] == 150
    assert out["mean_saved_tokens"] == 850
    assert out["mean_saved_pct"] == pytest.approx(85.0)


def test_tokens_saved_no_costs_returns_zero() -> None:
    results = [
        CaseResult(
            case=Case(query="x", expected_tool_ids=[]),
            retrieved_tool_ids=[],
            latency_ms=1.0,
            token_cost=None,
        ),
    ]
    out = tokens_saved(results, full_corpus_tokens=1000)
    assert out["mean_routed_tokens"] == 0.0


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_cli_writes_json(tmp_path: Path) -> None:
    from evals.__main__ import main

    out = tmp_path / "result.json"
    rc = main(
        [
            "--benchmark",
            "synthetic",
            "--embedder",
            "hash",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["benchmark"] == "synthetic"
    assert payload["n_cases"] >= 10
    assert "precision_at_1" in payload["metrics"]
    assert "mrr" in payload["metrics"]
    assert "latency" in payload["metrics"]


def test_cli_with_token_budget(tmp_path: Path) -> None:
    from evals.__main__ import main

    out = tmp_path / "result.json"
    rc = main(
        [
            "--benchmark",
            "synthetic",
            "--embedder",
            "none",
            "--token-budget",
            "1500",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    # Every case_result should have a token_cost when budget is set.
    assert all(cr["token_cost"] is not None for cr in payload["case_results"])
    # tokens_saved block should be populated.
    saved = payload["metrics"]["tokens_saved"]
    assert saved["mean_routed_tokens"] >= 0
