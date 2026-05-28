# ToolPicker

> Hybrid lexical + semantic tool selection for LLM agents with too many tools to fit in context. Three-stage router (BM25 + embeddings + optional intent classifier), Reciprocal Rank Fusion, token-budget packing.

[![python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![docs](https://img.shields.io/badge/docs-mkdocs--material-deeppurple)](https://ashwinugale.github.io/toolpicker/)

**Docs:** [ashwinugale.github.io/toolpicker](https://ashwinugale.github.io/toolpicker/) · **Status:** alpha. Public API is being shaped; pin to a specific version if you build on it.

## Why

LLM agents have a tool-count ceiling. Push past 15-20 tools in the schema and accuracy drops - the model gets confused about which tool to use, hallucinates parameters, and starts taking longer paths. Past 50 tools, performance collapses.

The solution is to not show all tools at once. You **route**: pick the K tools most relevant to the current query and only show those. Naive semantic search over tool descriptions works for some queries and fails badly on others (lexical-heavy queries like "get the order for BAN 989678111" miss semantic matches if the tool description doesn't use the word "BAN"). You need hybrid retrieval - the same way modern document RAG uses hybrid retrieval.

ToolPicker is the library that does this end-to-end, with a budget-aware packer and a reproducible eval harness.

## Install

```bash
pip install toolpicker                    # core
pip install "toolpicker[openai]"          # add real semantic retrieval
pip install "toolpicker[openai,openapi]"  # also parse OpenAPI specs as tool sources
pip install "toolpicker[openai,mcp]"      # also introspect MCP servers
```

## Quickstart

_Examples will land once the v0.1 walking skeleton is working._

## Documentation

Docs site lives at **[ashwinugale.github.io/toolpicker](https://ashwinugale.github.io/toolpicker/)** (built and published from v0.7 onward).

## License

MIT. See [LICENSE](./LICENSE).
