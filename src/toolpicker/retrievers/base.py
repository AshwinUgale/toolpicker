"""Retriever Protocol - the contract every ranking pass satisfies.

A retriever takes a query and returns a ranked list of (tool_id, score)
tuples ordered by descending relevance. The router fuses across multiple
retrievers via RRF, so absolute scores don't need to be comparable - only
ranks do.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["Retriever"]


@runtime_checkable
class Retriever(Protocol):
    """A single ranking pass over the tool corpus.

    Implementations are stateful: they build their index at construction time
    from the full tool list, then `retrieve()` queries the prebuilt index.
    """

    def retrieve(self, query: str, *, k: int) -> list[tuple[str, float]]:
        """Return up to ``k`` (tool_id, score) tuples, sorted by score desc.

        Empty list if the corpus is empty or no tool clears the retriever's
        internal threshold. Score interpretation is retriever-specific.
        """
        ...
