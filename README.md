# ToolPicker

> Hybrid lexical + semantic tool selection for LLM agents with too many tools to fit in context. Three-stage router (BM25 + embeddings + optional intent classifier), Reciprocal Rank Fusion, token-budget packing.

[![pypi](https://img.shields.io/pypi/v/toolpicker.svg)](https://pypi.org/project/toolpicker/)
[![python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![docs](https://img.shields.io/badge/docs-mkdocs--material-deeppurple)](https://ashwinugale.github.io/toolpicker/)

**Docs:** [ashwinugale.github.io/toolpicker](https://ashwinugale.github.io/toolpicker/) · **Issues:** [GitHub](https://github.com/ashwinugale/toolpicker/issues)

---

## Why

LLM agents have a tool-count ceiling. Past 15-20 tools in the schema, accuracy drops — the model gets confused about which tool to use, hallucinates parameters, takes longer paths. Past 50 tools, performance collapses. Carrying every tool schema also burns prompt tokens linearly while value is sparse: most tools are irrelevant to most queries.

The fix is to **route**: pick the K tools most relevant to the current query and only show those. Naive semantic search over tool descriptions handles some queries and fails on others (lexical-heavy queries like `"get the order for BAN 989678111"` miss semantic matches if no tool description uses the word `"BAN"`). Hybrid retrieval — BM25 + embeddings — fixes that, the same way modern document RAG does.

ToolPicker is the library that does this end to end, with a budget-aware packer, an optional intent classifier, and a reproducible eval harness.

---

## Install

```bash
pip install toolpicker                    # core, zero deps
pip install "toolpicker[openai]"          # add real semantic retrieval
pip install "toolpicker[openai,openapi]"  # parse OpenAPI specs as tool sources
pip install "toolpicker[openai,mcp]"      # introspect MCP servers
pip install "toolpicker[openai,tokens]"   # accurate token-budget packing via tiktoken
```

---

## Quickstart

```python
from toolpicker import FunctionSchemaSource, OpenAIEmbeddings, ToolPicker

tools = [
    {"name": "get_weather", "description": "Get weather for a city.",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}},
    {"name": "send_email", "description": "Send an email.",
     "parameters": {"type": "object", "properties": {"to": {"type": "string"}}}},
    # ... 48 more
]

picker = ToolPicker(FunctionSchemaSource(tools), embedder=OpenAIEmbeddings())
selected = picker.select("send a message to bob about the demo", k=5, token_budget=2000)
# selected = [Tool(name='send_email', ...), ...]  -- ready to hand to the LLM
```

Read the [quickstart](https://ashwinugale.github.io/toolpicker/quickstart/) for the full walkthrough including the intent classifier and token-budget packer.

---

## Headline numbers (v0.6)

Five-strategy comparison on a 200-case in-repo synthetic corpus and a 500-case Gorilla slice, OpenAI `text-embedding-3-small`:

**Synthetic (200 cases, 25 tools):**

| strategy             | p@1   | p@3   | mrr   |
|----------------------|-------|-------|-------|
| bm25-only            | 0.645 | 0.760 | 0.701 |
| **semantic-only**    | **0.885** | **0.970** | **0.926** |
| hybrid-rrf           | 0.800 | 0.960 | 0.879 |
| intent-only          | 0.715 | 0.925 | 0.819 |
| bm25+semantic+intent | 0.845 | 0.965 | 0.908 |

**Gorilla (500 cases, 1726 tools):**

| strategy            | p@1   | p@3   | mrr   |
|---------------------|-------|-------|-------|
| bm25-only           | 0.062 | 0.122 | 0.098 |
| **semantic-only**   | **0.102** | **0.186** | **0.147** |
| hybrid-rrf          | 0.088 | 0.168 | 0.132 |

Honest read: on these corpora under uniform-weight RRF, pure semantic beats every hybrid. Intent narrows the gap (synthetic: 0.800 → 0.845 p@1) but doesn't close it. The library exposes all five strategies and weight knobs so you can find what works for your distribution. Reproducer:

```bash
uv run python -m evals.compare --benchmark synthetic --embedder openai --output out/compare.json
```

More on the [concepts](https://ashwinugale.github.io/toolpicker/concepts/) and [eval harness](https://ashwinugale.github.io/toolpicker/eval/) pages.

---

## What ToolPicker is not

- **Not a tool runner.** Returns tools; you call them.
- **Not an agent framework.** Plugs into LangChain, LlamaIndex, raw OpenAI, Claude Agent SDK — anything that takes a `list[function_schema]`.
- **Not a vector database.** Semantic half stores embeddings in-process; under ~10k tools is the sweet spot. If you have 100k tools, you want a vector DB.

---

## Documentation

Full docs at **[ashwinugale.github.io/toolpicker](https://ashwinugale.github.io/toolpicker/)**:

- [Quickstart](https://ashwinugale.github.io/toolpicker/quickstart/) — install, declare tools, route a query.
- [Concepts](https://ashwinugale.github.io/toolpicker/concepts/) — BM25, semantic, intent, RRF, token packing.
- [Sources](https://ashwinugale.github.io/toolpicker/sources/) — `FunctionSchemaSource`, `OpenAPISource`, `MCPSource`, `MergedSource`.
- [Eval harness](https://ashwinugale.github.io/toolpicker/eval/) — reproduce the headline numbers, run on ToolBench / Gorilla.
- [API reference](https://ashwinugale.github.io/toolpicker/api/) — autogenerated.

---

## Related projects

Part of a 4-project portfolio of production AI engineering:

- **[mneme](https://github.com/AshwinUgale/mneme)** ([PyPI as smolAmem](https://pypi.org/project/smolAmem/)) — Multi-tier memory (working / episodic / semantic) for LLM agents with TTL + decay-based forgetting.
- **[DocChat VS Code extension](https://github.com/AshwinUgale/docchat)** ([Marketplace](https://marketplace.visualstudio.com/items?itemName=AshwinUgale.docchat)) — Version-pinned library-docs Q&A panel. Uses ToolPicker for tool-routing across its 3 agent tools.
- **[docchat-server](https://github.com/AshwinUgale/docchat-mcp)** ([PyPI](https://pypi.org/project/docchat-server/)) — The same retrieval pipeline exposed as an MCP server for Claude Code / Cursor.

## License

MIT. See [LICENSE](./LICENSE).
