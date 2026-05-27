"""Tests for the embeddings protocol + HashEmbedder."""

from __future__ import annotations

import math

import pytest

from toolpicker import EmbeddingProvider, HashEmbedder


def test_hash_embedder_satisfies_protocol() -> None:
    e = HashEmbedder()
    assert isinstance(e, EmbeddingProvider)


def test_hash_embedder_is_deterministic() -> None:
    a = HashEmbedder(dimensions=16)
    b = HashEmbedder(dimensions=16)
    assert a.embed(["hello"]) == b.embed(["hello"])


def test_hash_embedder_different_inputs_different_vectors() -> None:
    e = HashEmbedder(dimensions=16)
    [v1] = e.embed(["foo"])
    [v2] = e.embed(["bar"])
    assert v1 != v2


def test_hash_embedder_dimensions_match() -> None:
    e = HashEmbedder(dimensions=32)
    assert e.dimensions == 32
    [v] = e.embed(["x"])
    assert len(v) == 32


def test_hash_embedder_unit_norm() -> None:
    e = HashEmbedder(dimensions=16)
    [v] = e.embed(["hello world"])
    norm = math.sqrt(sum(x * x for x in v))
    assert abs(norm - 1.0) < 1e-9


def test_hash_embedder_empty_input_returns_empty() -> None:
    e = HashEmbedder()
    assert e.embed([]) == []


def test_hash_embedder_rejects_zero_dimensions() -> None:
    with pytest.raises(ValueError):
        HashEmbedder(dimensions=0)


def test_hash_embedder_batch_preserves_order() -> None:
    e = HashEmbedder(dimensions=16)
    vecs = e.embed(["a", "b", "c"])
    assert len(vecs) == 3
    # The same input in a batch produces the same vector as alone.
    assert vecs[0] == e.embed(["a"])[0]
    assert vecs[1] == e.embed(["b"])[0]
    assert vecs[2] == e.embed(["c"])[0]
