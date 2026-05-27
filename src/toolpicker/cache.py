"""Disk-backed embedding cache.

`CachedEmbedder` wraps any `EmbeddingProvider` and interposes a content-hash
cache. Hits skip the underlying embed call entirely; misses are embedded as
usual and written back to the cache.

The cache is keyed by ``sha256(text)`` so identical text always reuses the
same vector. The cache file is JSON: ``{"<sha256 hex>": [floats]}``. JSON
is fine for the sizes we care about (low thousands of tool descriptions);
swap to msgpack or sqlite if anyone ships a corpus past 100k.

Different embedders produce different vectors for the same input. Use a
distinct ``cache_path`` per embedder configuration (model + dimensions) or
the cache will return stale vectors. ``CachedEmbedder.invalidate()`` clears
the cache when you're not sure.

Default cache path: ``~/.toolpicker-cache/embeddings.json``. Override via the
constructor for per-project caches.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path

from toolpicker.embeddings import EmbeddingProvider

__all__ = ["CachedEmbedder"]


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _default_cache_path() -> Path:
    return Path.home() / ".toolpicker-cache" / "embeddings.json"


class CachedEmbedder:
    """Wrap an `EmbeddingProvider` with a disk-backed content-hash cache.

    Args:
        embedder: The underlying provider. Must satisfy `EmbeddingProvider`.
        cache_path: Where to persist the cache. Default
            ``~/.toolpicker-cache/embeddings.json``. The parent directory
            is created on demand.
        autosave: Whether to write the cache to disk after each ``embed()``
            call that produced cache misses. Default ``True``; set ``False``
            and call ``save()`` manually for tight loops.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        *,
        cache_path: Path | None = None,
        autosave: bool = True,
    ) -> None:
        self._embedder = embedder
        self._path = cache_path or _default_cache_path()
        self._autosave = autosave
        self._cache: dict[str, list[float]] = {}
        self._load()

    @property
    def dimensions(self) -> int:
        return self._embedder.dimensions

    @property
    def path(self) -> Path:
        return self._path

    def __contains__(self, text: str) -> bool:
        return _hash_text(text) in self._cache

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # First pass: figure out which texts are cache misses.
        keys = [_hash_text(t) for t in texts]
        miss_indices: list[int] = []
        miss_texts: list[str] = []
        for i, (text, key) in enumerate(zip(texts, keys, strict=True)):
            if key not in self._cache:
                miss_indices.append(i)
                miss_texts.append(text)
        # Second pass: embed misses (single batched call to the underlying
        # provider) and write back.
        if miss_texts:
            new_vectors = self._embedder.embed(miss_texts)
            if len(new_vectors) != len(miss_texts):
                raise RuntimeError(
                    f"underlying embedder returned {len(new_vectors)} vectors "
                    f"for {len(miss_texts)} inputs"
                )
            for idx, vec in zip(miss_indices, new_vectors, strict=True):
                self._cache[keys[idx]] = list(vec)
            if self._autosave:
                self.save()
        # Third pass: assemble in input order.
        return [self._cache[k] for k in keys]

    def invalidate(self, text: str | None = None) -> None:
        """Drop entries from the cache.

        Args:
            text: If given, drop only that single entry. If ``None`` (default),
                clear the entire cache.
        """
        if text is None:
            self._cache.clear()
        else:
            self._cache.pop(_hash_text(text), None)
        if self._autosave:
            self.save()

    def save(self) -> None:
        """Persist the in-memory cache to ``self.path``.

        Creates the parent directory if needed. Atomic-ish: writes to a
        temp file and renames, so a crash mid-write doesn't corrupt the
        existing cache.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._cache), encoding="utf-8")
        tmp.replace(self._path)

    def _load(self) -> None:
        """Load the cache file into memory. Silent if missing or unreadable."""
        if not self._path.exists():
            return
        with contextlib.suppress(json.JSONDecodeError, OSError):
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict):
                self._cache = {str(k): list(v) for k, v in data.items()}

    def __len__(self) -> int:
        return len(self._cache)
