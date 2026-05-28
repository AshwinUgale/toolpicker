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
    # v0.5 expanded corpus: 25 tools, 200 cases (8 per tool).
    assert len(bench.cases()) == 200
    tools = bench.tools().tools()
    assert len(tools) == 25


def test_synthetic_every_expected_tool_id_exists_in_corpus() -> None:
    """Sanity check the data: every Case.expected_tool_ids entry must
    correspond to an actual tool in the corpus. Cheap guard against
    typos when authoring more cases.
    """
    bench = SyntheticBenchmark()
    tool_ids = {t.id for t in bench.tools().tools()}
    for case in bench.cases():
        for expected in case.expected_tool_ids:
            assert expected in tool_ids, f"Case {case.query!r} references unknown tool {expected!r}"


def test_synthetic_eight_cases_per_tool() -> None:
    """v0.5 authoring rule: each tool has exactly 8 cases."""
    from collections import Counter

    bench = SyntheticBenchmark()
    per_tool: Counter[str] = Counter()
    for case in bench.cases():
        for expected in case.expected_tool_ids:
            per_tool[expected] += 1
    tool_ids = {t.id for t in bench.tools().tools()}
    for tid in tool_ids:
        assert per_tool[tid] == 8, f"Tool {tid} has {per_tool[tid]} cases (expected 8)"


def test_get_benchmark_dispatch() -> None:
    assert isinstance(get_benchmark("synthetic"), SyntheticBenchmark)
    with pytest.raises(ValueError, match="unknown"):
        get_benchmark("nope")


def test_toolbench_adapter_errors_without_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Strip the env var so the developer's real ToolBench checkout
    # doesn't satisfy the no-arg constructor under test.
    monkeypatch.delenv("TOOLPICKER_TOOLBENCH_DIR", raising=False)
    with pytest.raises(FileNotFoundError, match="ToolBench"):
        ToolBenchAdapter()


def test_gorilla_adapter_errors_without_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Strip the env var so the developer's real Gorilla checkout
    # doesn't satisfy the no-arg constructor under test.
    monkeypatch.delenv("TOOLPICKER_GORILLA_DIR", raising=False)
    with pytest.raises(FileNotFoundError, match="Gorilla"):
        GorillaAdapter()


# ---------------------------------------------------------------------------
# ToolBench / Gorilla parsers - on-disk fixture-driven, no real data needed
# ---------------------------------------------------------------------------


def _write_toolbench_fixture(root: Path) -> None:
    """Build a minimal ToolBench-shaped tree under root."""
    tool_dir = root / "data" / "toolenv" / "tools" / "Weather"
    tool_dir.mkdir(parents=True)
    (tool_dir / "WeatherAPI.json").write_text(
        json.dumps(
            {
                "tool_name": "WeatherAPI",
                "title": "Weather",
                "api_list": [
                    {
                        "name": "GetCurrentWeather",
                        "description": "Get current weather for a city.",
                        "required_parameters": [
                            {"name": "city", "type": "STRING", "description": "City name."}
                        ],
                        "optional_parameters": [
                            {"name": "units", "type": "STRING", "description": "metric/imperial."}
                        ],
                    },
                    {
                        "name": "GetForecast",
                        "description": "Multi-day forecast.",
                        "required_parameters": [],
                        "optional_parameters": [
                            {"name": "days", "type": "NUMBER", "description": "1-7."}
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    # A malformed file - parser should skip without crashing.
    (tool_dir / "Broken.json").write_text("{not valid json", encoding="utf-8")
    # Query files
    instr_dir = root / "data" / "instruction"
    instr_dir.mkdir(parents=True)
    (instr_dir / "G1_query.json").write_text(
        json.dumps(
            [
                {
                    "query": "What's the weather in Boston?",
                    "query_id": 1,
                    "relevant APIs": [["WeatherAPI", "GetCurrentWeather"]],
                },
                {
                    "query": "Give me a 5-day forecast.",
                    "query_id": 2,
                    "relevant APIs": [["WeatherAPI", "GetForecast"]],
                },
                {
                    "query": "Reference a tool we don't have loaded.",
                    "query_id": 3,
                    "relevant APIs": [["GhostAPI", "NopeNope"]],
                },
            ]
        ),
        encoding="utf-8",
    )


def test_toolbench_real_parser_loads_tools_and_cases(tmp_path: Path) -> None:
    _write_toolbench_fixture(tmp_path)
    bench = ToolBenchAdapter(data_dir=tmp_path)
    tools = bench.tools().tools()
    tool_ids = {t.id for t in tools}
    assert "weatherapi__getcurrentweather" in tool_ids
    assert "weatherapi__getforecast" in tool_ids
    cases = bench.cases()
    assert len(cases) == 2  # the GhostAPI case is dropped silently
    assert cases[0].expected_tool_ids == ["weatherapi__getcurrentweather"]
    stats = bench.stats
    assert stats["tool_families_skipped"] >= 1  # the malformed file
    assert stats["cases_dropped_unknown_tool"] == 1


def test_toolbench_max_tools_caps_load(tmp_path: Path) -> None:
    _write_toolbench_fixture(tmp_path)
    bench = ToolBenchAdapter(data_dir=tmp_path, max_tools=1)
    tools = bench.tools().tools()
    assert len(tools) == 1


def _write_gorilla_fixture(root: Path) -> None:
    """Build a minimal Gorilla-shaped tree under root (torchhub only)."""
    api_dir = root / "data" / "api"
    api_dir.mkdir(parents=True)
    (api_dir / "torchhub_api.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "domain": "Computer Vision",
                        "framework": "PyTorch",
                        "functionality": "Image Classification",
                        "api_name": "ResNet50",
                        "api_call": "torch.hub.load('pytorch/vision', 'resnet50')",
                        "description": "ResNet50 image classifier.",
                    }
                ),
                json.dumps(
                    {
                        "domain": "Computer Vision",
                        "framework": "PyTorch",
                        "functionality": "Object Detection",
                        "api_name": "DETR",
                        "api_call": "torch.hub.load('facebookresearch/detr', 'detr_resnet50')",
                        "description": "DETR object detector.",
                    }
                ),
                "{ broken line ignored",
            ]
        ),
        encoding="utf-8",
    )
    # Real Gorilla layout: eval data lives under a nested `gorilla/` subdir.
    q_dir = root / "gorilla" / "eval" / "eval-data" / "questions" / "torchhub"
    q_dir.mkdir(parents=True)
    (q_dir / "questions_torchhub_0_shot.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"question_id": 0, "text": "I want to classify images of cats."}),
                json.dumps({"question_id": 1, "text": "Detect objects in this photo."}),
                json.dumps({"question_id": 2, "text": "No matching response will exist."}),
            ]
        ),
        encoding="utf-8",
    )
    # Oracle responses store a stringified Python dict in the `text` field
    # whose api_call substring contains the gold api_name.
    r_dir = root / "gorilla" / "eval" / "eval-data" / "responses" / "torchhub"
    r_dir.mkdir(parents=True)
    (r_dir / "response_torchhub_Gorilla_FT_oracle.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "question_id": 0,
                        "text": (
                            "{'domain': 'Image Classification', "
                            "'api_call': \"torch.hub.load('pytorch/vision', 'ResNet50')\", "
                            "'api_provider': 'PyTorch'}"
                        ),
                    }
                ),
                json.dumps(
                    {
                        "question_id": 1,
                        "text": (
                            "{'domain': 'Object Detection', "
                            "'api_call': \"torch.hub.load('facebookresearch/detr', 'DETR')\", "
                            "'api_provider': 'PyTorch'}"
                        ),
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )


def test_gorilla_real_parser_loads_tools_and_cases(tmp_path: Path) -> None:
    _write_gorilla_fixture(tmp_path)
    bench = GorillaAdapter(data_dir=tmp_path, hubs=("torchhub",))
    tools = bench.tools().tools()
    tool_ids = {t.id for t in tools}
    assert "torchhub__resnet50" in tool_ids
    assert "torchhub__detr" in tool_ids
    cases = bench.cases()
    # 2 questions with responses; question_id=2 has no response and is
    # dropped (cases_dropped_no_response).
    assert len(cases) == 2
    assert cases[0].expected_tool_ids == ["torchhub__resnet50"]
    assert cases[1].expected_tool_ids == ["torchhub__detr"]
    assert bench.stats["cases_dropped_no_response"] == 1


def test_gorilla_rejects_unknown_hub(tmp_path: Path) -> None:
    _write_gorilla_fixture(tmp_path)
    with pytest.raises(ValueError, match="unknown hub"):
        GorillaAdapter(data_dir=tmp_path, hubs=("not_a_hub",))


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


def test_compare_cli_runs_all_strategies_on_synthetic(tmp_path: Path) -> None:
    """``python -m evals.compare`` over synthetic with hash embedder
    should emit all 5 strategy blocks (bm25 / semantic / hybrid /
    intent-only / bm25+semantic+intent), since the synthetic train
    corpus is bundled."""
    from evals.compare import main as compare_main

    out = tmp_path / "compare.json"
    rc = compare_main(
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
    strategies = [r["strategy"] for r in payload["results"]]
    assert strategies == [
        "bm25-only",
        "semantic-only",
        "hybrid-rrf",
        "intent-only",
        "bm25+semantic+intent",
    ]
    for block in payload["results"]:
        assert "precision_at_1" in block
        assert "mrr" in block
        assert block["n_cases"] == 200
    assert payload["config"]["intent_examples"] == 50


def test_compare_cli_embedder_none_skips_semantic_and_hybrid(tmp_path: Path) -> None:
    """``--embedder none`` -> bm25-only is the only runnable strategy."""
    from evals.compare import main as compare_main

    out = tmp_path / "compare.json"
    rc = compare_main(
        [
            "--benchmark",
            "synthetic",
            "--embedder",
            "none",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    strategies = [r["strategy"] for r in payload["results"]]
    assert strategies == ["bm25-only"]


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
