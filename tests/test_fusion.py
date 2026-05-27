"""Tests for reciprocal rank fusion."""

from __future__ import annotations

import pytest

from toolpicker import reciprocal_rank_fusion


def test_single_ranking_passthrough_order() -> None:
    ranking = [("a", 0.9), ("b", 0.5), ("c", 0.1)]
    fused = reciprocal_rank_fusion([ranking])
    # Single retriever, weights default to 1.0: order should match input.
    assert [tid for tid, _ in fused] == ["a", "b", "c"]


def test_two_rankings_agreeing_top_combined() -> None:
    r1 = [("a", 0.9), ("b", 0.4)]
    r2 = [("a", 0.8), ("c", 0.2)]
    fused = reciprocal_rank_fusion([r1, r2])
    # "a" should win because both rank it first.
    assert fused[0][0] == "a"


def test_two_rankings_disagreeing() -> None:
    r1 = [("a", 1.0), ("b", 0.5)]
    r2 = [("b", 1.0), ("a", 0.5)]
    fused = reciprocal_rank_fusion([r1, r2])
    # Both tools rank 1st in one list and 2nd in the other - tied votes.
    # Sort is stable - so order falls back to insertion order in the dict,
    # which is the order "a" then "b" appears in r1.
    assert {f[0] for f in fused} == {"a", "b"}
    # Scores should be equal.
    assert fused[0][1] == pytest.approx(fused[1][1])


def test_weights_boost_one_retriever() -> None:
    r1 = [("a", 1.0)]
    r2 = [("b", 1.0)]
    fused = reciprocal_rank_fusion([r1, r2], weights=[10.0, 1.0])
    # r1's weight is 10x; "a" should win.
    assert fused[0][0] == "a"


def test_empty_rankings_returns_empty() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_mismatched_weights_raises() -> None:
    with pytest.raises(ValueError, match="weights"):
        reciprocal_rank_fusion([[], []], weights=[1.0])


def test_invalid_rrf_k_raises() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[("a", 1.0)]], rrf_k=0)


def test_only_returns_tools_seen_by_at_least_one_retriever() -> None:
    r1 = [("a", 0.5)]
    r2 = [("b", 0.5)]
    fused = reciprocal_rank_fusion([r1, r2])
    assert {f[0] for f in fused} == {"a", "b"}


def test_first_rank_contributes_more_than_later_rank() -> None:
    r = [("a", 1.0), ("b", 0.5), ("c", 0.1)]
    fused = reciprocal_rank_fusion([r])
    a_score = next(s for tid, s in fused if tid == "a")
    b_score = next(s for tid, s in fused if tid == "b")
    c_score = next(s for tid, s in fused if tid == "c")
    assert a_score > b_score > c_score
