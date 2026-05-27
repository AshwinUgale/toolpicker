"""Token-budget packing.

Given a ranked list of tools and a token budget, return the prefix that fits.

Two pieces:
* `count_tokens(tool)` - token cost of a tool's serialised JSON schema.
  Uses tiktoken (cl100k_base) when installed; falls back to ~4 chars/token.
* `pack_to_budget(tools, token_budget)` - greedy first-fit: iterate the input
  in rank order, include any tool that fits, skip any that doesn't. The order
  among included tools is preserved (rank-respecting), so the LLM sees the
  most-relevant tool first.

We pick first-fit (skip-and-continue) over strict-first-fail because: a
slightly-too-big tool at rank 3 shouldn't block the smaller-but-still-good
tools at rank 4+. First-fit decreasing trades a sliver of rank-purity for
real fill efficiency. Knapsack-optimal packing is deferred (v1.x) - rarely
worth the complexity for typical tool counts.

The serialisation format matches OpenAI's function-call tools field:
``{"type": "function", "function": {"name": ..., "description": ...,
"parameters": ...}}``. Caller can override via the ``serialise`` argument
to match Anthropic's or any other format.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any

from toolpicker.types import Tool

__all__ = ["count_tokens", "default_serialise", "pack_to_budget"]


def default_serialise(tool: Tool) -> dict[str, Any]:
    """OpenAI function-call tool envelope.

    What you'd pass to ``client.chat.completions.create(tools=[...])``.
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters_schema,
        },
    }


_TIKTOKEN_ENCODER: Any = None


def _get_tiktoken_encoder() -> Any:
    """Lazy-load tiktoken's cl100k_base encoder. Cache it module-level.

    Return values: ``None`` (uninitialised — never returned externally),
    ``False`` (tiktoken unavailable — sentinel), or an ``Encoding`` object.
    Typed ``Any`` because the union spans types mypy can't reconcile cleanly.
    """
    global _TIKTOKEN_ENCODER
    if _TIKTOKEN_ENCODER is not None:
        return _TIKTOKEN_ENCODER
    try:
        import tiktoken

        _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
    except ImportError:
        _TIKTOKEN_ENCODER = False  # sentinel: tiktoken unavailable
    return _TIKTOKEN_ENCODER


def count_tokens(
    tool: Tool,
    *,
    serialise: Callable[[Tool], dict[str, Any]] = default_serialise,
) -> int:
    """Count tokens of the serialised tool.

    Uses tiktoken (``cl100k_base``, covers all modern OpenAI chat models)
    when installed. Falls back to ``ceil(len(json) / 4)`` otherwise - close
    enough for budget gating, off by ~10% in pathological cases.
    """
    text = json.dumps(serialise(tool), separators=(",", ":"))
    enc = _get_tiktoken_encoder()
    if enc is False:
        return max(1, math.ceil(len(text) / 4))
    return len(enc.encode(text))


def pack_to_budget(
    tools: list[Tool],
    *,
    token_budget: int,
    token_counter: Callable[[Tool], int] | None = None,
    serialise: Callable[[Tool], dict[str, Any]] = default_serialise,
) -> list[Tool]:
    """Greedy first-fit packing of tools under a token budget.

    Args:
        tools: Tools in rank order (most relevant first).
        token_budget: Max total token cost across the returned tools. Must
            be positive.
        token_counter: Optional override for token cost per tool. Defaults to
            ``count_tokens`` with the provided ``serialise``. Useful in tests
            to make cost deterministic without tiktoken.
        serialise: How to render a tool into a dict for token-counting.
            Defaults to OpenAI function-call envelope.

    Returns:
        The subset of ``tools`` that fits, in their input order. The first
        tool whose cost exceeds the remaining budget is skipped; we keep
        going to try smaller tools further down the ranking.
    """
    if token_budget <= 0:
        return []
    if not tools:
        return []
    counter = token_counter or (lambda t: count_tokens(t, serialise=serialise))
    out: list[Tool] = []
    remaining = token_budget
    for tool in tools:
        cost = counter(tool)
        if cost <= remaining:
            out.append(tool)
            remaining -= cost
        # else: skip this tool, try the next one in case it's smaller.
    return out
