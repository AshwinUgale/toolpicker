"""Tests for the Tool dataclass + ToolSource protocol."""

from __future__ import annotations

from toolpicker import FunctionSchemaSource, Tool, ToolSource


def test_tool_construction_required_fields() -> None:
    t = Tool(id="x", name="x", description="d")
    assert t.id == "x"
    assert t.name == "x"
    assert t.description == "d"
    assert t.parameters_schema == {}
    assert t.keywords == []
    assert t.metadata == {}


def test_tool_is_kw_only() -> None:
    # Positional construction must fail because of kw_only=True.
    import pytest

    with pytest.raises(TypeError):
        Tool("x", "x", "d")  # type: ignore[misc]


def test_function_schema_source_satisfies_protocol() -> None:
    source = FunctionSchemaSource(
        [{"name": "x", "description": "d", "parameters": {"type": "object"}}]
    )
    assert isinstance(source, ToolSource)
    tools = source.tools()
    assert len(tools) == 1
    assert tools[0].id == "x"
