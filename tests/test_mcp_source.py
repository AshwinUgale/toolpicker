"""Tests for MCPSource."""

from __future__ import annotations

import pytest

from toolpicker import MCPSource, ToolPicker


def test_parses_mcp_tool_dicts() -> None:
    raw = [
        {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "inputSchema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
        {
            "name": "send_email",
            "description": "Send an email message.",
            "inputSchema": {
                "type": "object",
                "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
                "required": ["to"],
            },
        },
    ]
    source = MCPSource(raw)
    tools = source.tools()
    assert {t.id for t in tools} == {"get_weather", "send_email"}

    weather = next(t for t in tools if t.id == "get_weather")
    assert weather.description == "Get the current weather for a city."
    assert weather.parameters_schema["properties"]["city"]["type"] == "string"


def test_empty_list_returns_empty() -> None:
    assert MCPSource([]).tools() == []


def test_missing_input_schema_defaults_to_empty() -> None:
    raw = [{"name": "noop", "description": "Does nothing."}]
    source = MCPSource(raw)
    tool = source.tools()[0]
    assert tool.parameters_schema == {}


def test_missing_description_defaults_to_empty_string() -> None:
    raw = [{"name": "x", "inputSchema": {"type": "object"}}]
    source = MCPSource(raw)
    assert source.tools()[0].description == ""


def test_missing_name_raises() -> None:
    with pytest.raises(ValueError, match="name"):
        MCPSource([{"description": "no name"}])


def test_bad_description_type_raises() -> None:
    with pytest.raises(ValueError, match="description"):
        MCPSource([{"name": "x", "description": 123, "inputSchema": {}}])  # type: ignore[list-item]


def test_bad_input_schema_type_raises() -> None:
    with pytest.raises(ValueError, match="inputSchema"):
        MCPSource([{"name": "x", "description": "d", "inputSchema": "not a dict"}])  # type: ignore[list-item]


def test_duplicate_names_raise() -> None:
    raw = [
        {"name": "x", "description": "a", "inputSchema": {}},
        {"name": "x", "description": "b", "inputSchema": {}},
    ]
    with pytest.raises(ValueError, match="duplicate"):
        MCPSource(raw)


def test_router_can_consume_mcp_source() -> None:
    source = MCPSource(
        [
            {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
            {
                "name": "send_email",
                "description": "Send an email message.",
                "inputSchema": {"type": "object", "properties": {"to": {"type": "string"}}},
            },
        ]
    )
    picker = ToolPicker(source)
    hits = picker.select("what's the weather in San Francisco?", k=2)
    assert any(t.id == "get_weather" for t in hits)
