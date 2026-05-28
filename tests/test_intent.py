"""Tests for IntentClassifier + the synthetic train corpus."""

from __future__ import annotations

import pytest

from toolpicker import (
    EmbeddingNNIntent,
    HashEmbedder,
    IntentClassifier,
    IntentExample,
)


def _examples() -> list[IntentExample]:
    return [
        IntentExample(query="send an email to the team", tool_id="send_email"),
        IntentExample(query="ping bob about the demo", tool_id="send_email"),
        IntentExample(
            query="schedule a calendar event for friday", tool_id="create_calendar_event"
        ),
        IntentExample(query="book a meeting next week", tool_id="create_calendar_event"),
        IntentExample(query="read the contents of a config file", tool_id="read_file"),
    ]


# ---------------------------------------------------------------------------
# Protocol satisfaction
# ---------------------------------------------------------------------------


def test_embedding_nn_intent_satisfies_protocol() -> None:
    classifier = EmbeddingNNIntent(examples=_examples(), embedder=HashEmbedder(dimensions=32))
    assert isinstance(classifier, IntentClassifier)


def test_returns_list_of_tuples() -> None:
    classifier = EmbeddingNNIntent(examples=_examples(), embedder=HashEmbedder(dimensions=32))
    hits = classifier.classify("any query", k=3)
    assert isinstance(hits, list)
    assert all(isinstance(h, tuple) and len(h) == 2 for h in hits)
    assert all(isinstance(h[0], str) and isinstance(h[1], float) for h in hits)


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


def test_empty_examples_returns_empty() -> None:
    classifier = EmbeddingNNIntent(examples=[], embedder=HashEmbedder(dimensions=32))
    assert classifier.classify("anything", k=5) == []


def test_k_zero_returns_empty() -> None:
    classifier = EmbeddingNNIntent(examples=_examples(), embedder=HashEmbedder(dimensions=32))
    assert classifier.classify("anything", k=0) == []


def test_scores_descending() -> None:
    classifier = EmbeddingNNIntent(examples=_examples(), embedder=HashEmbedder(dimensions=32))
    hits = classifier.classify("send an email to the team", k=5)
    scores = [s for _id, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_exact_match_returns_correct_tool_first() -> None:
    """When the query is identical to a training example, that example's
    tool_id should rank #1 - HashEmbedder is deterministic so the same
    text embeds to the same vector.
    """
    examples = _examples()
    classifier = EmbeddingNNIntent(examples=examples, embedder=HashEmbedder(dimensions=32))
    # Exact-text match for the "send_email" cluster.
    hits = classifier.classify("send an email to the team", k=3)
    assert hits
    assert hits[0][0] == "send_email"


def test_invalid_neighbours_rejected() -> None:
    with pytest.raises(ValueError, match="neighbours"):
        EmbeddingNNIntent(examples=_examples(), embedder=HashEmbedder(dimensions=32), neighbours=0)


def test_examples_property_returns_copy() -> None:
    examples = _examples()
    classifier = EmbeddingNNIntent(examples=examples, embedder=HashEmbedder(dimensions=32))
    returned = classifier.examples
    returned.clear()
    # Mutating the returned list must not affect the classifier's state.
    assert classifier.examples == examples


# ---------------------------------------------------------------------------
# Training corpus integrity
# ---------------------------------------------------------------------------


def test_synthetic_train_corpus_loads() -> None:
    from evals.benchmarks.synthetic_train import SYNTHETIC_TRAIN_EXAMPLES

    assert len(SYNTHETIC_TRAIN_EXAMPLES) == 50
    assert all(isinstance(e, IntentExample) for e in SYNTHETIC_TRAIN_EXAMPLES)


def test_synthetic_train_two_examples_per_tool() -> None:
    """v0.6 authoring rule: exactly 2 training examples per tool."""
    from collections import Counter

    from evals.benchmarks.synthetic import SyntheticBenchmark
    from evals.benchmarks.synthetic_train import SYNTHETIC_TRAIN_EXAMPLES

    counts: Counter[str] = Counter(e.tool_id for e in SYNTHETIC_TRAIN_EXAMPLES)
    tool_ids = {t.id for t in SyntheticBenchmark().tools().tools()}
    for tid in tool_ids:
        assert counts[tid] == 2, f"Tool {tid} has {counts[tid]} train examples (expected 2)"


def test_synthetic_train_disjoint_from_test() -> None:
    """No training query may appear verbatim in the test corpus.
    Catches data leakage if the test corpus grows.
    """
    from evals.benchmarks.synthetic import SyntheticBenchmark
    from evals.benchmarks.synthetic_train import SYNTHETIC_TRAIN_EXAMPLES

    test_queries = {c.query for c in SyntheticBenchmark().cases()}
    train_queries = {e.query for e in SYNTHETIC_TRAIN_EXAMPLES}
    overlap = test_queries & train_queries
    assert overlap == set(), f"Train/test overlap: {overlap}"


def test_synthetic_train_tool_ids_all_exist() -> None:
    """Every training example's tool_id must correspond to a real tool."""
    from evals.benchmarks.synthetic import SyntheticBenchmark
    from evals.benchmarks.synthetic_train import SYNTHETIC_TRAIN_EXAMPLES

    tool_ids = {t.id for t in SyntheticBenchmark().tools().tools()}
    for ex in SYNTHETIC_TRAIN_EXAMPLES:
        assert ex.tool_id in tool_ids, f"Train example references unknown tool {ex.tool_id!r}"
