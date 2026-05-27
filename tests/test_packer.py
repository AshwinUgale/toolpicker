"""Tests for the token-budget packer."""

from __future__ import annotations

import pytest

from toolpicker import Tool, count_tokens, pack_to_budget


def _tool(name: str, description: str = "") -> Tool:
    return Tool(id=name, name=name, description=description or f"do {name}")


def test_empty_input_returns_empty() -> None:
    assert pack_to_budget([], token_budget=1000) == []


def test_zero_or_negative_budget_returns_empty() -> None:
    tools = [_tool("a"), _tool("b")]
    assert pack_to_budget(tools, token_budget=0) == []
    assert pack_to_budget(tools, token_budget=-5) == []


def test_all_tools_fit_under_budget() -> None:
    # Override the counter so the test is deterministic without tiktoken.
    tools = [_tool("a"), _tool("b"), _tool("c")]
    packed = pack_to_budget(tools, token_budget=100, token_counter=lambda _: 10)
    assert [t.id for t in packed] == ["a", "b", "c"]


def test_budget_truncates_at_first_overflow() -> None:
    tools = [_tool("a"), _tool("b"), _tool("c")]
    # Counter: each is 30 tokens. Budget 75 = 2 fit, 3rd doesn't.
    packed = pack_to_budget(tools, token_budget=75, token_counter=lambda _: 30)
    assert [t.id for t in packed] == ["a", "b"]


def test_skip_then_continue_first_fit() -> None:
    # Greedy first-fit: a too-big tool at position 1 doesn't block smaller
    # tools at positions 2+.
    counts = {"big": 100, "small_a": 10, "small_b": 10}
    tools = [_tool("big"), _tool("small_a"), _tool("small_b")]
    packed = pack_to_budget(
        tools,
        token_budget=25,
        token_counter=lambda t: counts[t.id],
    )
    # "big" doesn't fit; we keep going and take both smalls.
    assert [t.id for t in packed] == ["small_a", "small_b"]


def test_order_preserved_among_included() -> None:
    counts = {"a": 10, "b": 30, "c": 10}
    tools = [_tool("a"), _tool("b"), _tool("c")]
    # Budget 25: "a" fits (10 used, 15 left); "b" doesn't (30 > 15); "c" fits (10 left).
    packed = pack_to_budget(tools, token_budget=25, token_counter=lambda t: counts[t.id])
    assert [t.id for t in packed] == ["a", "c"]


def test_single_too_big_tool_returns_empty() -> None:
    tools = [_tool("huge")]
    packed = pack_to_budget(tools, token_budget=10, token_counter=lambda _: 1000)
    assert packed == []


# ---------------------------------------------------------------------------
# Default token counter (uses tiktoken if installed; ~4 chars/token fallback)
# ---------------------------------------------------------------------------


def test_default_count_tokens_is_positive() -> None:
    t = Tool(
        id="get_weather",
        name="get_weather",
        description="Get current weather for a city.",
        parameters_schema={"type": "object", "properties": {"city": {"type": "string"}}},
    )
    assert count_tokens(t) > 0


def test_default_count_grows_with_description_length() -> None:
    short = Tool(id="x", name="x", description="hi")
    long_ = Tool(id="x", name="x", description="hello " * 50)
    assert count_tokens(long_) > count_tokens(short)


def test_default_count_is_stable() -> None:
    t = Tool(id="x", name="x", description="abc")
    assert count_tokens(t) == count_tokens(t)


def test_default_serialise_wraps_in_function_envelope() -> None:
    from toolpicker import default_serialise

    t = Tool(id="x", name="x", description="d", parameters_schema={"type": "object"})
    out = default_serialise(t)
    assert out["type"] == "function"
    assert out["function"]["name"] == "x"
    assert out["function"]["parameters"] == {"type": "object"}


# ---------------------------------------------------------------------------
# Integration via ToolPicker.select(token_budget=...)
# ---------------------------------------------------------------------------


def test_router_respects_token_budget() -> None:
    from toolpicker import FunctionSchemaSource, ToolPicker

    schemas = [
        {"name": f"tool_{i}", "description": "x" * 1000, "parameters": {}} for i in range(10)
    ]
    picker = ToolPicker(FunctionSchemaSource(schemas))
    full = picker.select("tool", k=10)
    constrained = picker.select("tool", k=10, token_budget=500)
    assert len(constrained) < len(full)


def test_router_token_budget_zero_returns_empty() -> None:
    from toolpicker import FunctionSchemaSource, ToolPicker

    picker = ToolPicker(FunctionSchemaSource([{"name": "a", "description": "a", "parameters": {}}]))
    assert picker.select("a", k=5, token_budget=0) == []


@pytest.mark.parametrize("budget", [10_000, 100_000])
def test_router_huge_budget_matches_unconstrained(budget: int) -> None:
    from toolpicker import FunctionSchemaSource, ToolPicker

    schemas = [{"name": f"tool_{i}", "description": "short", "parameters": {}} for i in range(5)]
    picker = ToolPicker(FunctionSchemaSource(schemas))
    unconstrained = picker.select("tool", k=10)
    constrained = picker.select("tool", k=10, token_budget=budget)
    assert [t.id for t in constrained] == [t.id for t in unconstrained]
