"""``python -m evals.compare`` - run one benchmark through 3 strategies.

The whole point of hybrid retrieval is that it should beat either half on
its own across a mixed query distribution. This module makes that claim
checkable in one command:

    python -m evals.compare --benchmark synthetic --embedder openai \\
        --output out/compare.json

Strategies (all use the same Benchmark and the same ``k``):

* ``bm25-only``   - ``embedder=None``. Pure lexical.
* ``semantic-only`` - ``bm25_weight=0.0``. Embedder must be set.
* ``hybrid-rrf``  - the default. BM25 + semantic fused with RRF.

Output: a single JSON with one block per strategy holding the same
metrics shape as ``python -m evals`` (precision_at_1, precision_at_3,
mrr, latency), plus a console table.

When ``--embedder none`` the semantic-only and hybrid strategies are
skipped (you can't run them without embeddings) and we emit a one-strategy
report - useful for hashing out the bm25-only baseline on a fresh
benchmark before paying for OpenAI.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

with contextlib.suppress(ImportError):
    from dotenv import load_dotenv

    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env_path, encoding="utf-8-sig")

from evals.benchmarks import get_benchmark
from evals.metrics import (
    latency_stats,
    mean_reciprocal_rank,
    precision_at_k,
)
from evals.runners.base import EvalRunner
from toolpicker.embeddings import EmbeddingProvider, HashEmbedder, OpenAIEmbeddings
from toolpicker.router import ToolPicker

_EMBEDDERS = ("hash", "openai", "none")
_STRATEGIES = ("bm25-only", "semantic-only", "hybrid-rrf")


def _build_embedder(name: str) -> EmbeddingProvider | None:
    if name == "hash":
        return HashEmbedder(dimensions=32)
    if name == "openai":
        return OpenAIEmbeddings()
    if name == "none":
        return None
    raise ValueError(f"unknown embedder: {name!r}")


def _build_picker(
    strategy: str,
    source: Any,
    embedder: EmbeddingProvider | None,
) -> ToolPicker:
    if strategy == "bm25-only":
        return ToolPicker(source, embedder=None)
    if strategy == "semantic-only":
        if embedder is None:
            raise ValueError("semantic-only requires --embedder hash or openai")
        return ToolPicker(source, embedder=embedder, bm25_weight=0.0)
    if strategy == "hybrid-rrf":
        if embedder is None:
            raise ValueError("hybrid-rrf requires --embedder hash or openai")
        return ToolPicker(source, embedder=embedder)
    raise ValueError(f"unknown strategy: {strategy!r}")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m evals.compare",
        description="Compare bm25-only / semantic-only / hybrid-rrf on one benchmark.",
    )
    p.add_argument(
        "--benchmark",
        choices=("synthetic", "toolbench", "gorilla"),
        default="synthetic",
    )
    p.add_argument(
        "--embedder",
        choices=_EMBEDDERS,
        default="hash",
        help="hash = test double; openai = real semantic; none = bm25-only only.",
    )
    p.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k for retrieval (default 5).",
    )
    p.add_argument("--output", type=Path, required=True)
    return p


def _run_strategy(
    strategy: str,
    bench: Any,
    embedder: EmbeddingProvider | None,
    k: int,
) -> dict[str, Any]:
    picker = _build_picker(strategy, bench.tools(), embedder)
    runner = EvalRunner(picker, k=k)
    case_results = runner.run(bench.cases())
    return {
        "strategy": strategy,
        "precision_at_1": precision_at_k(case_results, 1),
        "precision_at_3": precision_at_k(case_results, 3),
        "mrr": mean_reciprocal_rank(case_results),
        "latency": latency_stats(case_results),
        "n_cases": len(case_results),
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    bench = get_benchmark(args.benchmark)
    embedder = _build_embedder(args.embedder)
    # Pick which strategies are runnable given the embedder choice.
    if embedder is None:
        strategies: tuple[str, ...] = ("bm25-only",)
    else:
        strategies = _STRATEGIES
    results = [_run_strategy(s, bench, embedder, args.k) for s in strategies]
    return {
        "benchmark": bench.name,
        "config": {
            "embedder": args.embedder,
            "k": args.k,
        },
        "results": results,
    }


def _format_table(out: dict[str, Any]) -> str:
    rows = out["results"]
    header = f"{'strategy':<16} {'p@1':>6} {'p@3':>6} {'mrr':>6} {'p50_ms':>8} {'p95_ms':>8}"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['strategy']:<16} "
            f"{r['precision_at_1']:>6.3f} "
            f"{r['precision_at_3']:>6.3f} "
            f"{r['mrr']:>6.3f} "
            f"{r['latency']['p50_ms']:>8.1f} "
            f"{r['latency']['p95_ms']:>8.1f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    out = _run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(_format_table(out))
    print(f"\n-> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
