"""Tests for the SemanticRetriever."""

from __future__ import annotations

from toolpicker import HashEmbedder, SemanticRetriever, Tool


def _tools() -> list[Tool]:
    return [
        Tool(id="a", name="a", description="weather forecast for a location"),
        Tool(id="b", name="b", description="send email message"),
        Tool(id="c", name="c", description="get order by billing account"),
    ]


def test_returns_descending_scores() -> None:
    sem = SemanticRetriever(_tools(), HashEmbedder(dimensions=32))
    hits = sem.retrieve("any query", k=3)
    scores = [s for _id, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_returns_all_tools_within_k() -> None:
    sem = SemanticRetriever(_tools(), HashEmbedder(dimensions=32))
    hits = sem.retrieve("anything", k=10)
    # HashEmbedder doesn't filter; all three should appear.
    assert {h[0] for h in hits} == {"a", "b", "c"}


def test_empty_corpus_returns_empty() -> None:
    sem = SemanticRetriever([], HashEmbedder())
    assert sem.retrieve("anything", k=3) == []


def test_k_zero_returns_empty() -> None:
    sem = SemanticRetriever(_tools(), HashEmbedder())
    assert sem.retrieve("anything", k=0) == []


def test_k_bounds_results() -> None:
    sem = SemanticRetriever(_tools(), HashEmbedder())
    hits = sem.retrieve("anything", k=2)
    assert len(hits) <= 2


def test_falls_back_to_name_when_no_description() -> None:
    # Tool with empty description; SemanticRetriever should embed name instead
    # so we don't end up embedding an empty string (which OpenAI rejects).
    tools = [Tool(id="x", name="x", description="")]
    sem = SemanticRetriever(tools, HashEmbedder())
    hits = sem.retrieve("anything", k=1)
    assert len(hits) == 1
