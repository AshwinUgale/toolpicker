"""Tests for the CachedEmbedder disk cache."""

from __future__ import annotations

from pathlib import Path

from toolpicker import CachedEmbedder, EmbeddingProvider, HashEmbedder


class _CountingEmbedder:
    """Test double that wraps HashEmbedder and counts calls to embed()."""

    def __init__(self, *, dimensions: int = 16) -> None:
        self._inner = HashEmbedder(dimensions=dimensions)
        self.calls = 0
        self.batch_sizes: list[int] = []

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.batch_sizes.append(len(texts))
        return self._inner.embed(texts)


def test_satisfies_protocol(tmp_path: Path) -> None:
    e = CachedEmbedder(HashEmbedder(), cache_path=tmp_path / "c.json")
    assert isinstance(e, EmbeddingProvider)


def test_dimensions_passthrough(tmp_path: Path) -> None:
    inner = HashEmbedder(dimensions=64)
    cached = CachedEmbedder(inner, cache_path=tmp_path / "c.json")
    assert cached.dimensions == 64


def test_first_call_misses_then_caches(tmp_path: Path) -> None:
    counter = _CountingEmbedder()
    cached = CachedEmbedder(counter, cache_path=tmp_path / "c.json")
    cached.embed(["alpha", "beta"])
    assert counter.calls == 1
    assert counter.batch_sizes == [2]


def test_second_call_for_same_inputs_skips_underlying(tmp_path: Path) -> None:
    counter = _CountingEmbedder()
    cached = CachedEmbedder(counter, cache_path=tmp_path / "c.json")
    first = cached.embed(["alpha", "beta"])
    second = cached.embed(["alpha", "beta"])
    assert first == second
    # Only the first call hit the underlying embedder.
    assert counter.calls == 1


def test_mixed_hit_miss_only_misses_get_embedded(tmp_path: Path) -> None:
    counter = _CountingEmbedder()
    cached = CachedEmbedder(counter, cache_path=tmp_path / "c.json")
    cached.embed(["alpha"])  # populate cache for "alpha"
    counter.calls = 0
    counter.batch_sizes.clear()
    cached.embed(["alpha", "beta"])  # only "beta" is a miss
    assert counter.calls == 1
    assert counter.batch_sizes == [1]  # only the one miss


def test_cache_persists_across_instances(tmp_path: Path) -> None:
    counter = _CountingEmbedder()
    path = tmp_path / "c.json"
    a = CachedEmbedder(counter, cache_path=path)
    a.embed(["alpha", "beta"])
    assert counter.calls == 1

    # New instance pointing at the same file should hit the cache.
    counter2 = _CountingEmbedder()
    b = CachedEmbedder(counter2, cache_path=path)
    b.embed(["alpha", "beta"])
    assert counter2.calls == 0  # all cache hits


def test_invalidate_specific_text_forces_re_embed(tmp_path: Path) -> None:
    counter = _CountingEmbedder()
    cached = CachedEmbedder(counter, cache_path=tmp_path / "c.json")
    cached.embed(["alpha", "beta"])
    counter.calls = 0
    cached.invalidate("alpha")
    cached.embed(["alpha", "beta"])  # only alpha re-embeds
    assert counter.calls == 1


def test_invalidate_all_clears_cache(tmp_path: Path) -> None:
    counter = _CountingEmbedder()
    cached = CachedEmbedder(counter, cache_path=tmp_path / "c.json")
    cached.embed(["alpha", "beta"])
    assert len(cached) == 2
    cached.invalidate()  # clear all
    assert len(cached) == 0
    counter.calls = 0
    cached.embed(["alpha"])  # everything is a miss now
    assert counter.calls == 1


def test_contains(tmp_path: Path) -> None:
    cached = CachedEmbedder(HashEmbedder(), cache_path=tmp_path / "c.json")
    assert "alpha" not in cached
    cached.embed(["alpha"])
    assert "alpha" in cached
    assert "beta" not in cached


def test_empty_input_returns_empty(tmp_path: Path) -> None:
    counter = _CountingEmbedder()
    cached = CachedEmbedder(counter, cache_path=tmp_path / "c.json")
    assert cached.embed([]) == []
    assert counter.calls == 0


def test_corrupted_cache_file_is_silently_recovered(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text("not valid json", encoding="utf-8")
    counter = _CountingEmbedder()
    cached = CachedEmbedder(counter, cache_path=path)
    # Should not crash; cache starts empty.
    assert len(cached) == 0
    cached.embed(["x"])
    assert counter.calls == 1


def test_autosave_off_does_not_persist(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    cached = CachedEmbedder(HashEmbedder(), cache_path=path, autosave=False)
    cached.embed(["alpha"])
    assert not path.exists()  # no save happened
    cached.save()  # manual save
    assert path.exists()


def test_save_is_atomic_ish(tmp_path: Path) -> None:
    # The implementation writes to <path>.tmp then renames. Verify the
    # tmp file is cleaned up after save (no leftover .tmp).
    path = tmp_path / "c.json"
    cached = CachedEmbedder(HashEmbedder(), cache_path=path)
    cached.embed(["alpha"])
    tmp = path.with_suffix(path.suffix + ".tmp")
    assert not tmp.exists()


def test_dimensions_match_returned_vectors(tmp_path: Path) -> None:
    inner = HashEmbedder(dimensions=32)
    cached = CachedEmbedder(inner, cache_path=tmp_path / "c.json")
    vecs = cached.embed(["alpha", "beta"])
    assert all(len(v) == 32 for v in vecs)
