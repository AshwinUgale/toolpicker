"""Core data types shared across every retriever, source, and the router.

`Tool` is the unit of selection. `ToolSource` is the contract every adapter
(function schemas, OpenAPI, MCP) satisfies. Keep this file small - it's the
load-bearing protocol; changes here ripple everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = ["Tool", "ToolSource"]


@dataclass(kw_only=True)
class Tool:
    """One callable surface an LLM agent might invoke.

    Attributes:
        id: Stable opaque identifier. Usually the function name; must be unique
            across the corpus the router sees.
        name: Display name (typically same as `id`; sometimes prettified).
        description: Natural-language description of what the tool does. This is
            what the semantic retriever embeds.
        parameters_schema: JSON-Schema-shaped dict for the tool's parameters.
            Same shape as OpenAI function-call schemas.
        keywords: Optional short tokens that boost lexical recall. Append
            domain-specific terms here that aren't in the name/params/desc
            (e.g. internal codes, account-type abbreviations).
        metadata: Free-form caller-supplied tags. Not used by retrieval; useful
            for downstream routing (group, owner, deprecation, etc.).
    """

    id: str
    name: str
    description: str
    parameters_schema: dict[str, Any] = field(default_factory=dict)
    keywords: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ToolSource(Protocol):
    """Anything that knows how to enumerate a set of tools.

    Implementations: ``FunctionSchemaSource`` (v0.1), ``OpenAPISource`` (v0.3),
    ``MCPSource`` (v0.3). Each parses its input format into ``Tool`` objects.
    The router reads from the source once at construction; reload the source
    and rebuild the router if tools change.
    """

    def tools(self) -> list[Tool]:
        """Return every tool this source knows about."""
        ...
