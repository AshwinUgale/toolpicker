# ToolPicker eval harness

Reproducible benchmark for the router. Plays a `Benchmark`'s labelled cases through a configured `ToolPicker` and reports Precision@1, Precision@3, MRR, latency p50/p95, and tokens-saved-vs-no-routing.

Not installed by `pip install toolpicker` - this is repo-shaped tooling. Run it from a clone.

## Quickstart

```bash
# From the repo root, after uv sync:
uv run python -m evals --benchmark synthetic --output out/synthetic.json
```

Each invocation writes one JSON file plus a one-line stdout summary.

To use real OpenAI embeddings (set `OPENAI_API_KEY` first):

```bash
uv run python -m evals --benchmark synthetic --embedder openai --output out/synthetic_oai.json
```

To see the token-saved numbers:

```bash
uv run python -m evals --benchmark synthetic --token-budget 2000 --output out/synthetic_budget.json
```

## Benchmarks

| Name | Status | What it is |
|---|---|---|
| `synthetic` | ✅ ready | 25 tools / 200 cases across 6 domains, 8 phrasing styles per tool. Cheap path; works without keys. The CI baseline and headline-number benchmark when external datasets aren't fetched. |
| `gorilla` | ✅ ready | UC Berkeley's API-calling benchmark. Set `TOOLPICKER_GORILLA_DIR` and fetch from https://github.com/ShishirPatil/gorilla. Default caps `max_tools=2000` / `max_cases=500`; pass `None` to load everything. |
| `toolbench` | ✅ ready | OpenBMB's large-scale benchmark. Set `TOOLPICKER_TOOLBENCH_DIR` and fetch from https://github.com/OpenBMB/ToolBench. Same default caps as Gorilla. |

## Multi-strategy comparison

`python -m evals.compare` runs the same benchmark through three strategies in one shot - `bm25-only`, `semantic-only`, `hybrid-rrf` - and emits one JSON + a console table:

```bash
uv run python -m evals.compare --benchmark synthetic --embedder openai --output out/compare_oai.json
```

```
strategy             p@1    p@3    mrr   p50_ms   p95_ms
----------------------------------------------------------
bm25-only          0.X     0.X    0.X       X.X      X.X
semantic-only      0.X     0.X    0.X       X.X      X.X
hybrid-rrf         0.X     0.X    0.X       X.X      X.X
```

With `--embedder none`, only `bm25-only` runs (the other two need embeddings).

## What gets measured

- **Precision@k** - fraction of cases where any expected tool appears in the top-k retrieved.
- **MRR (Mean Reciprocal Rank)** - mean of 1/rank-of-first-relevant. 1.0 = always #1; 0.0 = never retrieved.
- **Latency** - p50 / p95 / mean wall-clock per `select()` call.
- **Tokens saved** - mean per-case tokens after routing vs the full-corpus baseline. Higher = routing is paying for itself.

## CLI reference

```
python -m evals \
  --benchmark {synthetic,gorilla,toolbench}
  --embedder {hash,openai,none}        # default: hash
  --k INT                              # default: 5
  --token-budget INT                   # default: off
  --output PATH                        # required
  --seed INT                           # default: 42
```

## Output shape

```jsonc
{
  "benchmark": "synthetic",
  "seed": 42,
  "config": {"embedder": "hash", "k": 5, "token_budget": null, "n_tools": 10},
  "metrics": {
    "precision_at_1": 0.867,
    "precision_at_3": 0.933,
    "mrr": 0.911,
    "latency": {"p50_ms": 0.6, "p95_ms": 1.2, "mean_ms": 0.7},
    "tokens_saved": {"baseline_tokens": 850, "mean_routed_tokens": 0, "mean_saved_tokens": 0, "mean_saved_pct": 0}
  },
  "n_cases": 15,
  "case_results": [
    {"query": "...", "expected_tool_ids": ["..."], "retrieved_tool_ids": ["...", "..."], "latency_ms": 0.5, "token_cost": null, "metadata": {...}}
  ]
}
```

## Adding a benchmark

Implement the `Benchmark` Protocol from `evals/benchmarks/base.py`:

```python
class MyBenchmark:
    name = "mine"

    def tools(self) -> ToolSource:
        return FunctionSchemaSource([...])

    def cases(self) -> list[Case]:
        return [Case(query="...", expected_tool_ids=["..."])]
```

Register it in `evals/benchmarks/__init__.py::_REGISTRY` so the CLI accepts `--benchmark mine`.
