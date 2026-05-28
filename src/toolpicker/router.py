"""ToolPicker facade - the public API.

Constructs the BM25, (optionally) semantic, and (optionally) intent
classifier retrievers internally, owns the tool corpus, and exposes
``select(query, k)`` returning the top-k ``Tool`` objects after RRF fusion.

At v0.6 the picker can fuse up to three ranking signals: BM25 + semantic +
intent. The intent half is opt-in: pass ``intent_classifier=...`` to wire
it in.
"""

from __future__ import annotations

from toolpicker.embeddings import EmbeddingProvider
from toolpicker.fusion import reciprocal_rank_fusion
from toolpicker.intent import IntentClassifier
from toolpicker.packer import pack_to_budget
from toolpicker.retrievers.bm25 import BM25Retriever
from toolpicker.retrievers.semantic import SemanticRetriever
from toolpicker.types import Tool, ToolSource

__all__ = ["ToolPicker"]


# Over-fetch each retriever this many times the final k. RRF benefits from
# longer rankings (so the fusion has more votes to combine), but going much
# past 4x has diminishing returns. Same pattern as Mneme's retrieve().
_OVERFETCH = 4


class ToolPicker:
    """Hybrid lexical + semantic tool selection.

    Args:
        source: Anything satisfying ``ToolSource``. Read once at construction.
        embedder: Optional ``EmbeddingProvider``. If ``None``, semantic
            retrieval is skipped and selection is BM25-only.
        intent_classifier: Optional ``IntentClassifier``. If ``None`` (the
            default), no intent ranking is fused in. Pass an
            ``EmbeddingNNIntent`` (or any classifier satisfying the
            Protocol) to add example-label-based ranking as a third signal
            alongside BM25 and semantic.
        bm25_weight: RRF weight for the BM25 retriever. Default 1.0.
        semantic_weight: RRF weight for the semantic retriever. Default 1.0.
        intent_weight: RRF weight for the intent classifier. Default 1.0.
        rrf_k: RRF damping constant. Default 60.
        bm25_k1: BM25 saturation parameter. Default 1.5.
        bm25_b: BM25 length-normalisation parameter. Default 0.75.
        bm25_stopwords: Optional override for the BM25 stopword set. ``None``
            uses the curated default; pass ``frozenset()`` to disable
            filtering (v0.4 behaviour); pass a custom set to override.

    Example:
        >>> from toolpicker import ToolPicker, FunctionSchemaSource, HashEmbedder
        >>> source = FunctionSchemaSource([
        ...     {"name": "get_weather", "description": "Get weather for a city.",
        ...      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}},
        ...     {"name": "send_email", "description": "Send an email.",
        ...      "parameters": {"type": "object", "properties": {"to": {"type": "string"}}}},
        ... ])
        >>> picker = ToolPicker(source, embedder=HashEmbedder())
        >>> [t.name for t in picker.select("what's the temperature in SF?", k=1)]
        ['get_weather']
    """

    def __init__(
        self,
        source: ToolSource,
        *,
        embedder: EmbeddingProvider | None = None,
        intent_classifier: IntentClassifier | None = None,
        bm25_weight: float = 1.0,
        semantic_weight: float = 1.0,
        intent_weight: float = 1.0,
        rrf_k: int = 60,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        bm25_stopwords: frozenset[str] | None = None,
    ) -> None:
        self._tools: list[Tool] = source.tools()
        self._tool_by_id: dict[str, Tool] = {t.id: t for t in self._tools}
        self._bm25 = BM25Retriever(self._tools, k1=bm25_k1, b=bm25_b, stopwords=bm25_stopwords)
        self._semantic = SemanticRetriever(self._tools, embedder) if embedder else None
        self._intent = intent_classifier
        self._rrf_k = rrf_k
        # Build the weights list in the same order rankings are emitted
        # below: bm25 first, then semantic if present, then intent if
        # present. RRF takes them positionally.
        weights = [bm25_weight]
        if self._semantic is not None:
            weights.append(semantic_weight)
        if self._intent is not None:
            weights.append(intent_weight)
        self._weights: list[float] = weights

    @property
    def tools(self) -> list[Tool]:
        """All tools the picker can return. Useful for debugging the corpus."""
        return list(self._tools)

    def select(
        self,
        query: str,
        *,
        k: int = 5,
        token_budget: int | None = None,
    ) -> list[Tool]:
        """Return tools for the query, ordered by fused relevance.

        Args:
            query: The user / agent input to route against.
            k: Cap on the number of tools returned.
            token_budget: Optional. If set, only return tools whose serialised
                schemas fit under this total token budget. Greedy first-fit
                (skip-and-continue): a too-big tool at rank N doesn't block
                smaller tools at rank N+1. Returned list is bounded by
                ``min(k, number_that_fit)``. If unset, returns the top-k
                without any token-cost consideration.

        Returns:
            ``list[Tool]`` in rank order. Empty if corpus is empty or no
            retriever returns a hit (or token_budget is too small for even
            the smallest tool).
        """
        if k <= 0 or not self._tools:
            return []
        # Over-fetch for fusion headroom AND packer headroom - if a tool
        # doesn't fit the budget we need more candidates further down the
        # ranking to try.
        overfetch_k = k * _OVERFETCH
        rankings = [self._bm25.retrieve(query, k=overfetch_k)]
        if self._semantic is not None:
            rankings.append(self._semantic.retrieve(query, k=overfetch_k))
        if self._intent is not None:
            rankings.append(self._intent.classify(query, k=overfetch_k))
        fused = reciprocal_rank_fusion(rankings, weights=self._weights, rrf_k=self._rrf_k)
        ranked = [self._tool_by_id[tid] for tid, _score in fused if tid in self._tool_by_id]
        if token_budget is not None:
            ranked = pack_to_budget(ranked, token_budget=token_budget)
        return ranked[:k]
