"""Tests for MergedSource."""

from __future__ import annotations

import pytest

from toolpicker import FunctionSchemaSource, MCPSource, MergedSource, ToolPicker


def _fn(name: str, desc: str = "") -> dict[str, object]:
    return {"name": name, "description": desc or f"do {name}", "parameters": {}}


def _mcp(name: str, desc: str = "") -> dict[str, object]:
    return {"name": name, "description": desc or f"do {name}", "inputSchema": {}}


def test_concatenates_in_order() -> None:
    a = FunctionSchemaSource([_fn("a"), _fn("b")])
    b = MCPSource([_mcp("c"), _mcp("d")])
    merged = MergedSource(a, b)
    assert [t.id for t in merged.tools()] == ["a", "b", "c", "d"]


def test_no_sources_returns_empty() -> None:
    merged = MergedSource()
    assert merged.tools() == []


def test_single_source_passthrough() -> None:
    a = FunctionSchemaSource([_fn("a"), _fn("b")])
    merged = MergedSource(a)
    assert [t.id for t in merged.tools()] == ["a", "b"]


def test_duplicate_ids_across_sources_raise() -> None:
    a = FunctionSchemaSource([_fn("x")])
    b = MCPSource([_mcp("x")])
    with pytest.raises(ValueError, match="duplicate"):
        MergedSource(a, b)


def test_router_routes_across_merged_sources() -> None:
    fn_source = FunctionSchemaSource(
        [
            {
                "name": "send_email",
                "description": "Send an email message.",
                "parameters": {"type": "object", "properties": {"to": {"type": "string"}}},
            }
        ]
    )
    mcp_source = MCPSource(
        [
            {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ]
    )
    merged = MergedSource(fn_source, mcp_source)
    picker = ToolPicker(merged)

    weather_hits = picker.select("what's the weather in SF?", k=1)
    email_hits = picker.select("send an email message", k=1)
    assert weather_hits[0].id == "get_weather"
    assert email_hits[0].id == "send_email"
