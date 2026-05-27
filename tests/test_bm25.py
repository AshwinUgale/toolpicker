"""Tests for the BM25 retriever."""

from __future__ import annotations

import pytest

from toolpicker import BM25Retriever, Tool
from toolpicker.retrievers.bm25 import DEFAULT_STOPWORDS


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


def test_stopwords_dropped_from_query_by_default() -> None:
    """A query that's entirely stopwords should retrieve nothing under
    the default tokeniser. Pre-v0.5 this leaked matches via "a"/"at".
    """
    bm25 = BM25Retriever(_tools())
    hits = bm25.retrieve("a the of to in on at", k=3)
    assert hits == []


def test_stopwords_disabled_returns_v04_behaviour() -> None:
    """Passing frozenset() restores the pre-stopword behaviour."""
    bm25 = BM25Retriever(_tools(), stopwords=frozenset())
    # "a" and "an" appear in send_email's "Send an email message to a recipient"
    # description; without filtering, BM25 will at least score it.
    hits = bm25.retrieve("a an the", k=3)
    assert any(h[0] == "send_email" for h in hits)


def test_default_stopwords_includes_common_english() -> None:
    """Spot-check the curated set so a future trim doesn't silently lose
    coverage on the stopwords we explicitly added the filter to handle.
    """
    for tok in ("a", "an", "the", "at", "of", "to", "in", "on", "for"):
        assert tok in DEFAULT_STOPWORDS


def test_custom_stopwords_override() -> None:
    """Caller-supplied set replaces the default entirely - no merge."""
    bm25 = BM25Retriever(_tools(), stopwords=frozenset({"weather"}))
    # "weather" is now a stopword so it's dropped from the query.
    hits = bm25.retrieve("weather", k=3)
    assert hits == []
    # But "a" is no longer a stopword (custom set doesn't include it).
    hits2 = bm25.retrieve("a", k=3)
    assert hits2 != []
