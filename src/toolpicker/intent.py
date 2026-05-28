"""IntentClassifier - the optional third ranker.

v0.6 adds an intent classifier alongside BM25 + semantic retrieval. The
intent half is OFF by default; pass ``intent_classifier=...`` to
``ToolPicker`` to wire it in.

Why a third ranker (not a replacement for semantic): semantic retrieval
matches a query against tool *descriptions*. Intent classification matches
a query against examples of *how that query was actually phrased when it
mapped to a tool*. They answer different questions and the v0.5 finding
that "semantic-only beat hybrid-rrf" suggests routing decisions need both
description-level matching AND label-level matching to get past the
uniform-RRF ceiling.

Library design (ADR-010):

* Protocol-first: ``IntentClassifier`` is a Protocol so users can plug in
  fine-tuned distilbert, hand-written heuristics, k-NN over a different
  embedder, etc. ToolPicker only cares about ``classify(query, *, k)``.
* User-supplied labels: there is no bundled training data. Users construct
  ``EmbeddingNNIntent(examples=[...], embedder=...)`` with their own
  (query, expected_tool_id) pairs. The library ships the mechanism; the
  user owns the corpus. This avoids tying the library to any tool taxonomy
  and removes the data-leakage trap.
* Reference impl is k-NN over embedded queries: cheap, no training, drops
  into any existing OpenAI / hash / cached embedder. For domain-specific
  setups users can replace it without touching the rest of the stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from toolpicker.embeddings import EmbeddingProvider

__all__ = ["EmbeddingNNIntent", "IntentClassifier", "IntentExample"]


@dataclass(frozen=True, kw_only=True)
class IntentExample:
    """A labelled (query, tool_id) example for intent training.

    Args:
        query: The user / agent text the example represents.
        tool_id: The tool that should be selected for this query.
    """

    query: str
    tool_id: str


@runtime_checkable
class IntentClassifier(Protocol):
    """A ranker that scores tools by labelled-example similarity to the query.

    Same shape as ``Retriever``: returns ``(tool_id, score)`` tuples sorted
    by score descending. Scores are aggregated across the k nearest
    training examples; absolute values aren't comparable across classifier
    implementations, but rank order is what RRF needs anyway.
    """

    def classify(self, query: str, *, k: int) -> list[tuple[str, float]]:
        """Return up to ``k`` (tool_id, score) tuples, sorted by score desc.

        Empty list if the example corpus is empty.
        """
        ...


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _unit(v: list[float]) -> list[float]:
    n = _norm(v)
    if n == 0.0:
        return v
    return [x / n for x in v]


class EmbeddingNNIntent:
    """k-NN intent classifier over embedded labelled queries.

    At construction time, embeds every training example's query once and
    keeps the unit-norm vector. At ``classify(query, k=K)``, embeds the
    query, computes cosine similarity against each example, picks the K
    most similar, and aggregates per ``tool_id`` (sum of similarities).
    Tools with multiple similar training examples beat tools with a single
    strong-but-isolated match - which is the right default for routing.

    Args:
        examples: Labelled training examples. Empty list is allowed (the
            classifier just returns ``[]``); makes the construction-time
            ergonomics easier for callers that load lazily.
        embedder: Any ``EmbeddingProvider``. Wrap with ``CachedEmbedder``
            if you'll instantiate this often with the same examples.
        neighbours: Number of nearest training examples to consider when
            classifying. Default 5; raise for noisy corpora.

    Example:
        ```python
        from toolpicker import HashEmbedder
        from toolpicker.intent import EmbeddingNNIntent, IntentExample

        intent = EmbeddingNNIntent(
            examples=[
                IntentExample(query="ping the team", tool_id="send_email"),
                IntentExample(query="block the afternoon", tool_id="create_calendar_event"),
            ],
            embedder=HashEmbedder(dimensions=32),
        )
        hits = intent.classify("notify the team", k=2)
        # Returns [(tool_id, score), ...] sorted by score desc.
        ```
    """

    def __init__(
        self,
        *,
        examples: list[IntentExample],
        embedder: EmbeddingProvider,
        neighbours: int = 5,
    ) -> None:
        if neighbours <= 0:
            raise ValueError(f"neighbours must be positive, got {neighbours}")
        self._embedder = embedder
        self._neighbours = neighbours
        self._examples: list[IntentExample] = list(examples)
        if self._examples:
            # Embed all training queries up front. Use the embedder's batch
            # path - OpenAI batches at 2048, HashEmbedder is per-text.
            raw = self._embedder.embed([e.query for e in self._examples])
            self._vectors: list[list[float]] = [_unit(v) for v in raw]
        else:
            self._vectors = []

    @property
    def examples(self) -> list[IntentExample]:
        """All training examples. Returns a copy to avoid external mutation."""
        return list(self._examples)

    def classify(self, query: str, *, k: int) -> list[tuple[str, float]]:
        if k <= 0 or not self._examples:
            return []
        q_vec = _unit(self._embedder.embed([query])[0])
        # cosine similarity = dot product since vectors are unit-norm
        sims: list[tuple[int, float]] = [(i, _dot(q_vec, v)) for i, v in enumerate(self._vectors)]
        sims.sort(key=lambda x: x[1], reverse=True)
        top = sims[: self._neighbours]
        # Aggregate per tool_id by SUM. Sum-aggregation makes voting natural:
        # if 3 of 5 nearest examples map to ``send_email``, it gets ~3x the
        # signal of a tool with a single strong-but-isolated match.
        agg: dict[str, float] = {}
        for idx, sim in top:
            if sim <= 0:
                continue
            tool_id = self._examples[idx].tool_id
            agg[tool_id] = agg.get(tool_id, 0.0) + sim
        ranked = sorted(agg.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]
