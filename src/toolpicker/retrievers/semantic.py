"""Semantic retriever - cosine similarity between query and tool embeddings.

Embeds each tool's description once at construction (cached in-memory by
``tool_id``). On ``retrieve()``, embeds the query and ranks tools by cosine
similarity to the query vector.

v0.2 will add a disk-backed cache (content-hash keyed) so cold starts don't
re-embed the whole corpus. For v0.1 the in-memory cache is enough.
"""

from __future__ import annotations

import math

from toolpicker.embeddings import EmbeddingProvider
from toolpicker.types import Tool

__all__ = ["SemanticRetriever"]


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Assumes both vectors are non-zero length."""
    if len(a) != len(b):
        raise ValueError(f"vector dim mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class SemanticRetriever:
    """Embedding-based retrieval over tool descriptions.

    Args:
        tools: Tools to index. Each tool's ``description`` is embedded once
            at construction.
        embedder: Any ``EmbeddingProvider``.
    """

    def __init__(self, tools: list[Tool], embedder: EmbeddingProvider) -> None:
        self._embedder = embedder
        self._tool_ids: list[str] = [t.id for t in tools]
        texts = [t.description or t.name for t in tools]
        # One batch embed call; the OpenAI adapter handles auto-batching internally.
        self._vectors: list[list[float]] = embedder.embed(texts) if texts else []
        if len(self._vectors) != len(self._tool_ids):
            raise RuntimeError(
                f"embedder returned {len(self._vectors)} vectors for {len(self._tool_ids)} inputs"
            )

    def retrieve(self, query: str, *, k: int) -> list[tuple[str, float]]:
        if k <= 0 or not self._tool_ids:
            return []
        q_vec = self._embedder.embed([query])[0]
        scored = [
            (tool_id, _cosine(q_vec, vec))
            for tool_id, vec in zip(self._tool_ids, self._vectors, strict=True)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
