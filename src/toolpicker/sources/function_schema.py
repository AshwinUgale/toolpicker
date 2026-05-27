"""Adapter from OpenAI function-call schemas to ``Tool`` objects.

OpenAI's function-call schema is the universal tool-description format - every
agent framework that does tool calling speaks it. Anthropic's tool schema is
close enough to be 1-for-1 mappable; we accept that shape too.

Schema shape we accept:

    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name."},
            },
            "required": ["city"],
        },
    }

Some callers wrap each schema under a ``{"type": "function", "function": {...}}``
envelope (OpenAI's newer agent-tooling format); we unwrap that transparently.
"""

from __future__ import annotations

from typing import Any

from toolpicker.types import Tool

__all__ = ["FunctionSchemaSource"]


class FunctionSchemaSource:
    """Wrap a list of function-call schema dicts as a ``ToolSource``.

    Args:
        schemas: Function-call schemas. Each may be either the bare schema
            (``{"name": ..., "description": ..., "parameters": ...}``) or the
            wrapped form (``{"type": "function", "function": {...}}``).
        keywords: Optional mapping of tool name → keyword list. Lets callers
            attach domain-specific lexical hints without hand-building the
            ``Tool`` objects.
    """

    def __init__(
        self,
        schemas: list[dict[str, Any]],
        *,
        keywords: dict[str, list[str]] | None = None,
    ) -> None:
        self._tools = [self._parse(s, keywords or {}) for s in schemas]
        ids = [t.id for t in self._tools]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate tool ids in source: {sorted(dupes)}")

    @staticmethod
    def _parse(raw: dict[str, Any], keywords: dict[str, list[str]]) -> Tool:
        # Unwrap the {"type": "function", "function": {...}} envelope if present.
        if raw.get("type") == "function" and "function" in raw:
            raw = raw["function"]
        name = raw.get("name")
        if not name or not isinstance(name, str):
            raise ValueError(f"schema missing string 'name': {raw!r}")
        description = raw.get("description", "")
        if not isinstance(description, str):
            raise ValueError(f"schema 'description' must be str: {raw!r}")
        parameters_schema = raw.get("parameters", {})
        if not isinstance(parameters_schema, dict):
            raise ValueError(f"schema 'parameters' must be dict: {raw!r}")
        return Tool(
            id=name,
            name=name,
            description=description,
            parameters_schema=parameters_schema,
            keywords=list(keywords.get(name, [])),
        )

    def tools(self) -> list[Tool]:
        return list(self._tools)
