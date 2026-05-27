"""``python -m evals`` - run a benchmark through the picker, dump JSON.

Glues the four pieces together::

    benchmark
        v
    ToolPicker(bench.tools(), embedder=...)
        v
    EvalRunner.run(bench.cases())
        v
    precision_at_k + mrr + latency_stats (+ tokens_saved when budget set)
        v
    JSON written to disk + one-line stdout summary

Examples::

    # cheap path - synthetic benchmark, hash embedder, no key required
    python -m evals --benchmark synthetic --output out/synthetic.json

    # with real OpenAI embeddings
    python -m evals --benchmark synthetic --embedder openai --output out/synthetic_oai.json

    # with token budget (records tokens_saved)
    python -m evals --benchmark synthetic --token-budget 2000 --output out/synthetic_budget.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Optional .env loading (same pattern as Mneme + the smoke script).
with contextlib.suppress(ImportError):
    from dotenv import load_dotenv

    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env_path, encoding="utf-8-sig")

from evals.benchmarks import get_benchmark
from evals.metrics import (
    latency_stats,
    mean_reciprocal_rank,
    precision_at_k,
    tokens_saved,
)
from evals.runners.base import EvalRunner
from toolpicker.embeddings import HashEmbedder, OpenAIEmbeddings
from toolpicker.packer import count_tokens
from toolpicker.router import ToolPicker

_EMBEDDERS = ("hash", "openai", "none")


def _build_embedder(name: str) -> Any:
    if name == "hash":
        return HashEmbedder(dimensions=32)
    if name == "openai":
        return OpenAIEmbeddings()
    if name == "none":
        return None
    raise ValueError(f"unknown embedder: {name!r}")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m evals",
        description="Run a benchmark through ToolPicker and report metrics.",
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
        help="hash = deterministic test double; openai = real semantic; none = BM25-only.",
    )
    p.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k for retrieval (default 5).",
    )
    p.add_argument(
        "--token-budget",
        type=int,
        default=None,
        help="Optional token budget per query (default off).",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed recorded in the result file (no randomness in current path).",
    )
    return p


def _run(args: argparse.Namespace) -> dict[str, Any]:
    bench = get_benchmark(args.benchmark)
    embedder = _build_embedder(args.embedder)
    picker = ToolPicker(bench.tools(), embedder=embedder)

    runner = EvalRunner(picker, k=args.k, token_budget=args.token_budget)
    case_results = runner.run(bench.cases())

    metrics: dict[str, Any] = {
        "precision_at_1": precision_at_k(case_results, 1),
        "precision_at_3": precision_at_k(case_results, 3),
        "mrr": mean_reciprocal_rank(case_results),
        "latency": latency_stats(case_results),
    }

    # tokens_saved makes sense whenever we have a baseline (the cost of
    # sending the whole corpus). Compute the baseline from the picker's
    # full tool list - this is the "no routing" cost the LLM would pay.
    full_corpus_tokens = sum(count_tokens(t) for t in picker.tools)
    metrics["tokens_saved"] = tokens_saved(case_results, full_corpus_tokens=full_corpus_tokens)

    return {
        "benchmark": bench.name,
        "seed": args.seed,
        "config": {
            "embedder": args.embedder,
            "k": args.k,
            "token_budget": args.token_budget,
            "n_tools": len(picker.tools),
        },
        "metrics": metrics,
        "n_cases": len(case_results),
        "case_results": [
            {
                "query": cr.case.query,
                "expected_tool_ids": cr.case.expected_tool_ids,
                "retrieved_tool_ids": cr.retrieved_tool_ids,
                "latency_ms": cr.latency_ms,
                "token_cost": cr.token_cost,
                "metadata": cr.case.metadata,
            }
            for cr in case_results
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    out = _run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    m = out["metrics"]
    print(
        f"benchmark={out['benchmark']} "
        f"n={out['n_cases']} "
        f"p@1={m['precision_at_1']:.3f} "
        f"p@3={m['precision_at_3']:.3f} "
        f"mrr={m['mrr']:.3f} "
        f"p50={m['latency']['p50_ms']:.1f}ms "
        f"-> {args.output}"
    )
    # asdict is unused but imported for parity with future v0.5 expansion.
    _ = asdict
    return 0


if __name__ == "__main__":
    sys.exit(main())
