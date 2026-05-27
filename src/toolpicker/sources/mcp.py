"""Adapter from MCP tool descriptions to ``Tool`` objects.

The Model Context Protocol describes each tool with three fields::

    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "inputSchema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }

The mapping is 1-for-1: name -> id+name, description -> description,
inputSchema -> parameters_schema.

Two construction paths:

* ``MCPSource(mcp_tools=[{...}, {...}])`` - sync, takes the list of tool
  dicts directly. Useful when you've already introspected the MCP server
  or are wiring up tests.
* ``await MCPSource.from_client(session)`` - async classmethod that calls
  ``session.list_tools()`` and wraps the result. Requires the ``mcp``
  Python package (``pip install 'toolpicker[mcp]'``).

The split keeps the source itself synchronous - ``tools()`` always returns
immediately - while still offering an ergonomic live-introspection path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from toolpicker.types import Tool

if TYPE_CHECKING:
    from mcp import ClientSession

__all__ = ["MCPSource"]


def _mcp_dict_to_tool(raw: dict[str, Any]) -> Tool:
    name = raw.get("name")
    if not name or not isinstance(name, str):
        raise ValueError(f"MCP tool missing string 'name': {raw!r}")
    description = raw.get("description") or ""
    if not isinstance(description, str):
        raise ValueError(f"MCP tool 'description' must be str: {raw!r}")
    schema = raw.get("inputSchema") or {}
    if not isinstance(schema, dict):
        raise ValueError(f"MCP tool 'inputSchema' must be dict: {raw!r}")
    return Tool(
        id=name,
        name=name,
        description=description,
        parameters_schema=schema,
    )


class MCPSource:
    """Wrap a list of MCP tool descriptions as a ``ToolSource``.

    Args:
        mcp_tools: A list of dicts in MCP's tool-description format
            (``name``, ``description``, ``inputSchema``).
    """

    def __init__(self, mcp_tools: list[dict[str, Any]]) -> None:
        self._tools = [_mcp_dict_to_tool(t) for t in mcp_tools]
        ids = [t.id for t in self._tools]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate tool ids in MCP source: {sorted(dupes)}")

    @classmethod
    async def from_client(cls, session: ClientSession) -> MCPSource:
        """Introspect an MCP server and build a source from its advertised tools.

        Args:
            session: An already-initialized ``mcp.ClientSession``. The caller
                is responsible for connecting / closing the underlying
                transport.
        """
        result = await session.list_tools()
        # mcp.types.Tool has .name, .description, .inputSchema attributes.
        # We coerce to dicts so the rest of the pipeline stays plain.
        return cls(
            [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "inputSchema": dict(t.inputSchema) if t.inputSchema else {},
                }
                for t in result.tools
            ]
        )

    def tools(self) -> list[Tool]:
        return list(self._tools)
