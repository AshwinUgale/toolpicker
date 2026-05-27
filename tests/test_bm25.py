"""Tests for the BM25 retriever."""

from __future__ import annotations

import pytest

from toolpicker import BM25Retriever, Tool


def _tools() -> list[Tool]:
    return [
        Tool(
            id="get_weather",
            name="get_weather",
            description="Get the current weather for a city.",
            parameters_schema={"properties": {"city": {"type": "string"}}},
        ),
        Tool(
            id="send_email",
            name="send_email",
            description="Send an email message to a recipient.",
            parameters_schema={
                "properties": {"to": {"type": "string"}, "subject": {"type": "string"}}
            },
        ),
        Tool(
            id="get_order_by_ban",
            name="get_order_by_ban",
            description="Look up an order by the customer's billing account number.",
            parameters_schema={"properties": {"ban": {"type": "string"}}},
            keywords=["BAN", "billing-account-number"],
        ),
    ]


def test_lexical_match_wins() -> None:
    bm25 = BM25Retriever(_tools())
    # Lexical-heavy query that mentions BAN by name; semantic might miss.
    hits = bm25.retrieve("get order for BAN 989678111", k=3)
    assert hits[0][0] == "get_order_by_ban"


def test_weather_query_finds_weather_tool() -> None:
    bm25 = BM25Retriever(_tools())
    hits = bm25.retrieve("what is the weather in San Francisco?", k=3)
    assert hits[0][0] == "get_weather"


def test_returns_empty_for_no_match() -> None:
    bm25 = BM25Retriever(_tools())
    # Query terms appear nowhere in any tool text.
    hits = bm25.retrieve("xyzzy plugh frobnicate", k=3)
    assert hits == []


def test_k_bounds_results() -> None:
    bm25 = BM25Retriever(_tools())
    hits = bm25.retrieve("get send email weather order", k=2)
    assert len(hits) <= 2


def test_k_zero_returns_empty() -> None:
    bm25 = BM25Retriever(_tools())
    assert bm25.retrieve("anything", k=0) == []


def test_empty_corpus_returns_empty() -> None:
    bm25 = BM25Retriever([])
    assert bm25.retrieve("anything", k=5) == []


def test_invalid_k1_rejected() -> None:
    with pytest.raises(ValueError):
        BM25Retriever([], k1=-1.0)


def test_invalid_b_rejected() -> None:
    with pytest.raises(ValueError):
        BM25Retriever([], b=1.5)


def test_scores_are_descending() -> None:
    bm25 = BM25Retriever(_tools())
    hits = bm25.retrieve("email weather order", k=5)
    scores = [s for _id, s in hits]
    assert scores == sorted(scores, reverse=True)
