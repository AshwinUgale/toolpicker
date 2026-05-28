# Quickstart

Pick the right tool out of many in three steps: install, declare your tools, route a query.

---

## 1. Install

```bash
pip install toolpicker[openai]
```

You can install the bare package (`pip install toolpicker`) and it will still run in BM25-only mode without any extra deps. The `[openai]` extra adds `OpenAIEmbeddings` so the semantic half works.

Set `OPENAI_API_KEY` in your environment before the first call (or in a `.env` file — ToolPicker doesn't auto-load one, but most callers already do).

---

## 2. Declare your tools

`FunctionSchemaSource` takes the standard OpenAI function-call dict format, with or without the `{"type": "function", "function": {...}}` envelope:

```python
tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email message to a recipient.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
        },
    },
    # ... add as many as you have.
]
```

If your tools come from an OpenAPI spec or an MCP server, use `OpenAPISource(spec)` or `MCPSource.from_client(session)` instead — see [Sources](sources.md).

---

## 3. Route a query

```python
from toolpicker import FunctionSchemaSource, OpenAIEmbeddings, ToolPicker

picker = ToolPicker(
    FunctionSchemaSource(tools),
    embedder=OpenAIEmbeddings(),
)

selected = picker.select("send a message to bob@example.com about the launch", k=3)
for tool in selected:
    print(tool.name)
# send_email
# search_inbox    (depending on your corpus)
# ...
```

That's the whole API. `select` returns a `list[Tool]` — your `Tool` objects carry the original name, description, and parameter schema, so you hand them straight to the LLM as the `tools=` argument.

---

## With a token budget

Pass `token_budget=` to cap the total tokens the returned tools will consume in your prompt. ToolPicker uses greedy first-fit packing: a too-big tool at rank N doesn't block smaller tools at rank N+1.

```python
selected = picker.select("schedule a meeting tomorrow", k=10, token_budget=2000)
# Returns up to 10 tools whose serialised schemas fit under ~2000 tokens.
```

Token counting uses `tiktoken` with the `cl100k_base` encoding when available, falling back to a 4-chars-per-token approximation when it isn't. Install the `[tokens]` extra to guarantee the accurate path.

---

## With the intent classifier

If you have labelled examples of "what queries should map to which tool" — call logs, hand-curated examples, anything — feed them to `EmbeddingNNIntent` and pass it to the picker:

```python
from toolpicker import EmbeddingNNIntent, IntentExample, OpenAIEmbeddings, ToolPicker

embedder = OpenAIEmbeddings()
intent = EmbeddingNNIntent(
    examples=[
        IntentExample(query="ping bob", tool_id="send_email"),
        IntentExample(query="notify the team", tool_id="send_email"),
        IntentExample(query="block out friday afternoon", tool_id="create_calendar_event"),
        # ... 20-50 examples gets you meaningful signal
    ],
    embedder=embedder,
)

picker = ToolPicker(
    FunctionSchemaSource(tools),
    embedder=embedder,
    intent_classifier=intent,
)
```

The classifier adds a third ranking signal alongside BM25 and semantic. On our corpora it didn't beat pure semantic — see [Concepts](concepts.md) and the [eval harness](eval.md) for the numbers — but it adds real signal when your queries are indirect ("ping the team" → `send_email`) and your tool descriptions don't mention the same words.

---

## Without OpenAI

If you don't have an API key or you're in CI:

```python
from toolpicker import ToolPicker, FunctionSchemaSource

picker = ToolPicker(FunctionSchemaSource(tools))  # BM25-only, no embedder
```

This works for lexical-heavy queries (parameter names match query tokens, descriptions overlap), gets you 60%+ p@1 on the in-repo synthetic benchmark, and costs zero dollars per call. The trade-off is that natural-language queries that don't share vocabulary with your tool descriptions will miss.
