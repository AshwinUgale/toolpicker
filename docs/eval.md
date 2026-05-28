# Eval harness

ToolPicker ships a reproducible eval harness in the `evals/` directory of the repository. It's not part of the wheel — clone the repo and run from source.

Two CLIs:

- `python -m evals` — single configuration, single benchmark, full JSON dump.
- `python -m evals.compare` — multi-strategy comparison on one benchmark, table + JSON out.

---

## Bundled benchmarks

| Name | Status | What it is |
|---|---|---|
| `synthetic` | Ready | 25 tools / 200 cases across 6 domains, 8 phrasing styles per tool. Cheap path; works without API keys. The CI baseline and reference-number benchmark when external datasets aren't fetched. |
| `gorilla` | Ready | UC Berkeley's API-calling benchmark. Set `TOOLPICKER_GORILLA_DIR` and `git clone https://github.com/ShishirPatil/gorilla`. Default caps `max_tools=2000` / `max_cases=500`. |
| `toolbench` | Ready | OpenBMB's large-scale benchmark. Set `TOOLPICKER_TOOLBENCH_DIR` and fetch the data from `https://github.com/OpenBMB/ToolBench` (data lives on Google Drive). Same default caps as Gorilla. |

---

## Single-run eval

```bash
uv run python -m evals --benchmark synthetic --embedder openai --output out/synthetic_oai.json
```

Output is a single JSON plus a one-line stdout summary:

```
benchmark=synthetic n=200 p@1=0.885 p@3=0.970 mrr=0.926 p50=298.4ms -> out\synthetic_oai.json
```

CLI flags:

```
python -m evals \
  --benchmark {synthetic,toolbench,gorilla}
  --embedder {hash,openai,none}        # default: hash
  --k INT                              # default: 5
  --token-budget INT                   # default: off
  --output PATH                        # required
  --seed INT                           # default: 42
```

`--embedder hash` is a deterministic test double (`HashEmbedder`) — useful for sanity-checking ranking changes without paying for OpenAI calls. `--embedder none` runs BM25 alone.

`--token-budget N` activates greedy first-fit packing and populates the `tokens_saved` block in the output, comparing per-case routed cost against the full-corpus cost baseline.

---

## Multi-strategy comparison

This is the headline tool. Runs the same benchmark through every applicable strategy and emits a side-by-side table:

```bash
uv run python -m evals.compare --benchmark synthetic --embedder openai --output out/compare.json
```

```
strategy                    p@1    p@3    mrr   p50_ms   p95_ms
---------------------------------------------------------------
bm25-only                 0.645  0.760  0.701      0.0      0.0
semantic-only             0.885  0.970  0.926    291.9    533.0
hybrid-rrf                0.800  0.960  0.879    283.0    469.3
intent-only               0.715  0.925  0.819    285.5    517.7
bm25+semantic+intent      0.845  0.965  0.908    590.0    901.8
-> out\compare.json
```

The five strategies:

| Strategy | What it does |
|---|---|
| `bm25-only` | Pure lexical (no embedder). |
| `semantic-only` | Embedder-only (BM25 weight 0). |
| `hybrid-rrf` | BM25 + semantic fused with RRF (v0.5 default). |
| `intent-only` | Embedded-example k-NN (BM25 weight 0, no semantic). Synthetic only. |
| `bm25+semantic+intent` | All three signals fused with RRF (v0.6 full hybrid). Synthetic only. |

Intent strategies skip on ToolBench / Gorilla — those benchmarks don't ship with a labelled training corpus. If you want intent on them, construct `EmbeddingNNIntent` yourself with your own examples and call `ToolPicker` directly.

---

## Fetching the real benchmarks

### Gorilla (easy)

```powershell
git clone https://github.com/ShishirPatil/gorilla C:\Users\you\code\gorilla
[Environment]::SetEnvironmentVariable("TOOLPICKER_GORILLA_DIR", "C:\Users\you\code\gorilla", "User")
# restart your shell so the env var becomes visible
uv run python -m evals.compare --benchmark gorilla --embedder openai --output out/compare_gorilla.json
```

The Gorilla repo bundles both the API definitions and the eval data. ~250 MB clone.

### ToolBench (harder)

The ToolBench code lives at `https://github.com/OpenBMB/ToolBench` but the data is hosted separately (Google Drive / Tsinghua Cloud). Once you've extracted the data so that `<root>/data/toolenv/tools/<Category>/*.json` and `<root>/data/instruction/G{1,2}_query.json` exist:

```powershell
[Environment]::SetEnvironmentVariable("TOOLPICKER_TOOLBENCH_DIR", "C:\Users\you\code\ToolBench", "User")
uv run python -m evals.compare --benchmark toolbench --embedder openai --output out/compare_toolbench.json
```

ToolBench's full dataset is ~10 GB uncompressed; the adapter caps at 2000 tools / 500 cases by default to keep eval runtime bounded. Override the caps by constructing the adapter directly if you want the full run.

---

## What gets measured

| Metric | Definition |
|---|---|
| **Precision@k** | Fraction of cases where any expected tool appears in the top-k retrieved. |
| **MRR** | Mean Reciprocal Rank: mean of `1 / rank_of_first_expected`. 1.0 = always #1; 0.0 = never retrieved. |
| **Latency** | p50 / p95 / mean wall-clock per `select()` call (nearest-rank percentile). |
| **Tokens saved** | Mean per-case tokens after routing vs the full-corpus baseline. Only populated when `--token-budget` is set. |

All metrics are pure functions over the runner's `list[CaseResult]`. JSON-serialisable. Same definitions across `evals` and `evals.compare`.

---

## Output shape

```jsonc
{
  "benchmark": "synthetic",
  "config": {
    "embedder": "openai",
    "k": 5,
    "intent_examples": 50
  },
  "results": [
    {
      "strategy": "bm25-only",
      "precision_at_1": 0.645,
      "precision_at_3": 0.760,
      "mrr": 0.701,
      "latency": {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0},
      "n_cases": 200
    },
    {
      "strategy": "semantic-only",
      "precision_at_1": 0.885,
      "...": "..."
    }
  ]
}
```

---

## Adding a benchmark

Implement the `Benchmark` Protocol from `evals/benchmarks/base.py`:

```python
from evals.schema import Case
from toolpicker.sources import FunctionSchemaSource
from toolpicker.types import ToolSource

class MyBenchmark:
    name = "mine"

    def tools(self) -> ToolSource:
        return FunctionSchemaSource([...])

    def cases(self) -> list[Case]:
        return [Case(query="...", expected_tool_ids=["..."])]
```

Register it in `evals/benchmarks/__init__.py::_REGISTRY` so the CLI accepts `--benchmark mine`. Done.
