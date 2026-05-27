"""Tests for the FunctionSchemaSource adapter."""

from __future__ import annotations

import pytest

from toolpicker import FunctionSchemaSource


def test_parses_bare_schema() -> None:
    source = FunctionSchemaSource(
        [
            {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ]
    )
    tools = source.tools()
    assert len(tools) == 1
    t = tools[0]
    assert t.id == "get_weather"
    assert t.name == "get_weather"
    assert t.description.startswith("Get the current weather")
    assert t.parameters_schema["properties"]["city"]["type"] == "string"


def test_unwraps_function_envelope() -> None:
    source = FunctionSchemaSource(
        [
            {
                "type": "function",
                "function": {
                    "name": "send_email",
                    "description": "Send an email.",
                    "parameters": {"type": "object"},
                },
            }
        ]
    )
    tools = source.tools()
    assert len(tools) == 1
    assert tools[0].id == "send_email"


def test_attaches_keywords_by_name() -> None:
    source = FunctionSchemaSource(
        [
            {"name": "get_order", "description": "Get an order.", "parameters": {}},
        ],
        keywords={"get_order": ["BAN", "billing-account-number"]},
    )
    tools = source.tools()
    assert tools[0].keywords == ["BAN", "billing-account-number"]


def test_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        FunctionSchemaSource(
            [
                {"name": "x", "description": "a", "parameters": {}},
                {"name": "x", "description": "b", "parameters": {}},
            ]
        )


def test_rejects_missing_name() -> None:
    with pytest.raises(ValueError, match="name"):
        FunctionSchemaSource([{"description": "x", "parameters": {}}])


def test_returns_empty_when_no_schemas() -> None:
    source = FunctionSchemaSource([])
    assert source.tools() == []
