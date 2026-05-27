"""Combine multiple ``ToolSource`` instances into one.

Use case: an agent has function-schema tools (its own definitions) plus an
OpenAPI-described external API plus a few MCP servers. ``MergedSource``
glues them so a single ``ToolPicker`` can route across all of them.

Duplicate tool ids across sources are a loud error - the router needs to
be able to map a fused result back to a unique ``Tool``, and silent
last-write-wins is the kind of bug that's painful to debug.
"""

from __future__ import annotations

from toolpicker.types import Tool, ToolSource

__all__ = ["MergedSource"]


class MergedSource:
    """Concatenate the tools from multiple sources, preserving order.

    Args:
        *sources: Any number of ``ToolSource``-satisfying objects.
    """

    def __init__(self, *sources: ToolSource) -> None:
        self._tools: list[Tool] = []
        for s in sources:
            self._tools.extend(s.tools())
        ids = [t.id for t in self._tools]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(
                f"duplicate tool ids across merged sources: {sorted(dupes)}; "
                "rename or alias one to disambiguate before merging"
            )

    def tools(self) -> list[Tool]:
        return list(self._tools)
