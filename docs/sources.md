# Sources

A `ToolSource` is anything that yields a `list[Tool]`. Four implementations ship; build your own by satisfying the Protocol.

---

## `FunctionSchemaSource` — the universal format

Accepts the OpenAI function-call schema, with or without the `{"type": "function", "function": {...}}` envelope. Zero extra deps.

```python
from toolpicker import FunctionSchemaSource

source = FunctionSchemaSource([
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
    # The envelope form works too:
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email message.",
            "parameters": {
                "type": "object",
                "properties": {"to": {"type": "string"}},
            },
        },
    },
])
```

Optional `keywords` per tool attach extra retrieval tokens that BM25 indexes but never expose to the LLM:

```python
FunctionSchemaSource(
    tools=[...],
    keywords={"get_order_by_ban": ["BAN", "billing-account-number"]},
)
```

Duplicate tool ids raise loudly at construction. Silent last-write-wins is the worst kind of routing bug.

---

## `OpenAPISource` — OpenAPI 3.0 / 3.1

Accepts a dict, a JSON file path, or a YAML file path (YAML requires PyYAML). Install the `[openapi]` extra:

```bash
pip install toolpicker[openapi]
```

```python
from toolpicker import OpenAPISource

source = OpenAPISource("./petstore.yaml")
# or:
source = OpenAPISource(spec_dict, validate=True)
```

One tool per `(path, method)` operation. Tool id derivation: `operationId` if set, otherwise `{method}_{path_stripped}` (e.g. `GET /pets/{petId}` → `get_pets_petId`). Path / query / header / cookie parameters plus the `application/json` request-body schema's properties merge into one flat parameter namespace — what LLM tool callers actually want.

`$ref` resolution uses `jsonref.replace_refs(..., proxies=False)` plus a JSON round-trip to normalise (jsonref's lazy proxies trip downstream serialisation). Pass `validate=False` to skip `openapi-spec-validator` for trusted specs or to handle minor non-conformances.

---

## `MCPSource` — Model Context Protocol

Accepts MCP-format dicts (`name`, `description`, `inputSchema`) for tools you have in hand:

```python
from toolpicker import MCPSource

source = MCPSource(mcp_tools=[
    {
        "name": "list_repositories",
        "description": "List all repositories for a user.",
        "inputSchema": {
            "type": "object",
            "properties": {"username": {"type": "string"}},
        },
    },
])
```

For live introspection against an MCP server, use the async classmethod:

```python
from mcp import ClientSession
from toolpicker import MCPSource

async def build_source(session: ClientSession) -> MCPSource:
    return await MCPSource.from_client(session)
```

This calls `session.list_tools()`, coerces the response into dicts, and hands off to the sync constructor — so `ToolSource.tools()` stays sync regardless of how the inventory was obtained.

Install the `[mcp]` extra:

```bash
pip install toolpicker[mcp]
```

---

## `MergedSource` — cross-source routing

When you have tools coming from multiple places (your own function-call definitions + an MCP server + an OpenAPI spec for an internal API), wrap them all in one `MergedSource` and hand that to the picker:

```python
from toolpicker import (
    FunctionSchemaSource,
    MCPSource,
    MergedSource,
    OpenAPISource,
    ToolPicker,
)

source = MergedSource(
    FunctionSchemaSource(local_tools),
    OpenAPISource("./internal.yaml"),
    MCPSource(mcp_tools=mcp_inventory),
)
picker = ToolPicker(source, embedder=OpenAIEmbeddings())
```

Order is preserved, and cross-source duplicate ids raise loudly. The picker's API stays a single-source surface — composition lives in `MergedSource`.

---

## Building your own

`ToolSource` is a runtime-checkable `Protocol` with a single method:

```python
class ToolSource(Protocol):
    def tools(self) -> list[Tool]: ...
```

If your tool inventory lives in a database, a config file, a Notion page, or anywhere else — write a tiny class that produces `Tool` objects and pass it to `ToolPicker`. The picker doesn't care where the tools came from; it cares that they have an id, a name, a description, a parameter schema, and (optionally) keywords.

```python
from toolpicker import Tool, ToolPicker

class NotionToolSource:
    def __init__(self, page_id: str) -> None:
        self._page_id = page_id

    def tools(self) -> list[Tool]:
        rows = self._fetch_notion_rows(self._page_id)
        return [
            Tool(
                id=row["name"],
                name=row["name"],
                description=row["description"],
                parameters_schema=row["schema"],
                keywords=row.get("keywords", []),
            )
            for row in rows
        ]
```

That's it. No registration, no inheritance.
