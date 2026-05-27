"""Reciprocal Rank Fusion - combines rankings from multiple retrievers.

RRF (Cormack et al., 2009) is the standard parameter-free way to fuse
ranked lists from heterogeneous retrievers. Each retriever votes for a
document with weight ``1 / (rrf_k + rank_i)``; documents accumulate votes
across retrievers; final ranking is by total vote.

The genius of RRF: it ignores raw scores. BM25 scores aren't comparable to
cosine similarities, and trying to normalise them is a tar pit. RRF only
needs the rank order, so the math just works regardless of what each
retriever's score actually means.

The standard ``rrf_k`` constant is 60 (per the original paper; later work
mostly accepts 60 as a sensible default).

Weights let callers say "trust the semantic retriever more than BM25 for
this corpus" - each retriever's votes get scaled by its weight before
summing.
"""

from __future__ import annotations

__all__ = ["reciprocal_rank_fusion"]


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    *,
    weights: list[float] | None = None,
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse N ranked lists into one ranked list.

    Args:
        rankings: List of per-retriever rankings. Each ranking is a list of
            ``(tool_id, score)`` tuples sorted by descending score. RRF only
            uses the position; the score is ignored.
        weights: Optional per-retriever weight (same length as ``rankings``).
            Defaults to uniform 1.0 across all retrievers.
        rrf_k: The damping constant. Larger = less penalty on lower ranks.
            Default 60 per the original paper.

    Returns:
        ``[(tool_id, fused_score), ...]`` sorted by descending fused score.
        A tool only appears if at least one retriever ranked it.
    """
    if not rankings:
        return []
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(f"got {len(rankings)} rankings but {len(weights)} weights")
    if rrf_k <= 0:
        raise ValueError(f"rrf_k must be positive, got {rrf_k}")

    fused: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, (tool_id, _score) in enumerate(ranking):
            # rank is 0-indexed here; the canonical RRF formula uses 1-indexed
            # ranks, so add 1.
            contribution = weight / (rrf_k + rank + 1)
            fused[tool_id] = fused.get(tool_id, 0.0) + contribution

    return sorted(fused.items(), key=lambda x: x[1], reverse=True)
