"""Embedding providers - lifted directly from the Mneme pattern.

`EmbeddingProvider` is a small Protocol. Two shipped implementations:

* `OpenAIEmbeddings` - real semantic vectors via OpenAI's API. Lazy-imports
  ``openai`` so the core install stays zero-dep.
* `HashEmbedder` - deterministic test double. Same vector for the same input,
  every time. Not semantic; used in tests and as a no-key fallback.
"""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    from openai import OpenAI

__all__ = ["EmbeddingProvider", "HashEmbedder", "OpenAIEmbeddings"]


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Anything that turns text into fixed-dimensional vectors."""

    @property
    def dimensions(self) -> int:
        """Vector dimension this provider emits."""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in input order."""
        ...


class OpenAIEmbeddings:
    """Real OpenAI embeddings. Defaults to ``text-embedding-3-small`` (1536-d).

    Args:
        model: OpenAI embedding model name. Default ``text-embedding-3-small``.
        api_key: API key. Falls back to ``OPENAI_API_KEY`` env var.

    Auto-batches at 2048 inputs per call (OpenAI's hard limit on the batch
    size). Returned vectors are unit-normalised by the API; cosine similarity
    and dot product yield identical rankings.
    """

    _DIMS: ClassVar[dict[str, int]] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    _BATCH_LIMIT: ClassVar[int] = 2048

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAIEmbeddings requires the 'openai' extra. Install with:\n"
                "    pip install 'toolpicker[openai]'"
            ) from exc
        self._client: OpenAI = OpenAI(api_key=api_key) if api_key else OpenAI()
        self._model = model
        if model not in self._DIMS:
            raise ValueError(
                f"unknown OpenAI embedding model {model!r}; known: {sorted(self._DIMS)}"
            )
        self._dimensions = self._DIMS[model]

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), self._BATCH_LIMIT):
            batch = texts[i : i + self._BATCH_LIMIT]
            response = self._client.embeddings.create(input=batch, model=self._model)
            out.extend(item.embedding for item in response.data)
        return out


class HashEmbedder:
    """Deterministic non-semantic embedder. Same input → same vector, always.

    Useful when:
    * tests need reproducibility without an API key
    * you want a fast no-network baseline for benchmarking

    NOT useful for production semantic retrieval - it's hash-based and has
    no notion of meaning. Lexical retrieval (BM25) will beat it on most
    queries.

    Args:
        dimensions: Output vector dimensionality. Default 16 (fast for tests).
    """

    def __init__(self, *, dimensions: int = 16) -> None:
        if dimensions <= 0:
            raise ValueError(f"dimensions must be positive, got {dimensions}")
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vec(t) for t in texts]

    def _hash_to_vec(self, text: str) -> list[float]:
        # Use SHA-256 of the text and unroll it into floats in [-1, 1].
        # Multiple rounds when we need more dimensions than one hash gives.
        bytes_needed = self._dimensions * 4  # 4 bytes per float
        raw = b""
        counter = 0
        while len(raw) < bytes_needed:
            h = hashlib.sha256(f"{text}:{counter}".encode()).digest()
            raw += h
            counter += 1
        # Convert 4-byte chunks to floats in [-1, 1] via int conversion.
        vec = []
        for i in range(self._dimensions):
            chunk = raw[i * 4 : i * 4 + 4]
            val = int.from_bytes(chunk, "big", signed=False)
            # Map [0, 2^32) to [-1, 1].
            vec.append((val / (2**31)) - 1.0)
        # Normalise to unit length so cosine works cleanly.
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]
