"""Tool sources - adapters that parse a format into Tool objects.

* ``FunctionSchemaSource`` (v0.1) - OpenAI function-call schemas. Universal
  format every LLM agent framework speaks.
* ``OpenAPISource`` (v0.3) - parses OpenAPI 3.0 / 3.1 specs. One tool per
  operation; requires the ``[openapi]`` extra.
* ``MCPSource`` (v0.3) - wraps MCP tool descriptions; ``[mcp]`` extra for
  live introspection.
* ``MergedSource`` (v0.3) - combine multiple sources into one.
"""

from toolpicker.sources.function_schema import FunctionSchemaSource
from toolpicker.sources.mcp import MCPSource
from toolpicker.sources.merged import MergedSource
from toolpicker.sources.openapi import OpenAPISource

__all__ = ["FunctionSchemaSource", "MCPSource", "MergedSource", "OpenAPISource"]
