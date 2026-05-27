"""BM25 retriever - the lexical half of hybrid retrieval.

BM25 is the standard probabilistic ranking function for sparse retrieval.
It scores a query against a document by summing per-term contributions:

    BM25(q, d) = sum_t in q: idf(t) * (tf(t,d) * (k1 + 1))
                              / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))

where:
* idf(t) = ln((N - df(t) + 0.5) / (df(t) + 0.5) + 1)   (Robertson-Sparck-Jones+1)
* tf(t,d) = term frequency of t in d
* df(t) = document frequency of t
* N = corpus size
* |d| = doc length (in tokens)
* avgdl = mean doc length
* k1 ~ 1.2-2.0 controls saturation (default 1.5)
* b ~ 0..1 controls length normalisation (default 0.75)

For tool routing the "document" is the tool's name + parameter names +
keywords + description. Lexical-heavy queries ("get the order for BAN 989...")
match the parameter name ``ban`` directly, which is what BM25 buys over pure
semantic retrieval.

We implement BM25 in-process (~40 LOC) to keep core install zero-dep.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

from toolpicker.types import Tool

__all__ = ["BM25Retriever"]


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase + split on non-alphanumeric runs. Filter empty strings.

    Deliberately simple. snake_case identifiers (``get_order_by_ban``) split
    into useful tokens; CamelCase doesn't - we lowercase first, so ``BANId``
    becomes one token ``banid``. Acceptable for v0.1; revisit if recall on
    CamelCase-heavy APIs underperforms.
    """
    return _TOKEN_RE.findall(text.lower())


def _tool_text(tool: Tool) -> str:
    """The text BM25 indexes per tool: name + param names + keywords + description."""
    param_names: Iterable[str] = []
    properties = tool.parameters_schema.get("properties")
    if isinstance(properties, dict):
        param_names = properties.keys()
    parts = [
        tool.name,
        " ".join(param_names),
        " ".join(tool.keywords),
        tool.description,
    ]
    return " ".join(parts)


class BM25Retriever:
    """BM25 over the tool corpus.

    Args:
        tools: Tools to index. Indexed at construction; rebuild for new tools.
        k1: Term-frequency saturation parameter (default 1.5).
        b: Length-normalisation parameter (default 0.75).
    """

    def __init__(
        self,
        tools: list[Tool],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 < 0:
            raise ValueError(f"k1 must be non-negative, got {k1}")
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"b must be in [0, 1], got {b}")
        self._k1 = k1
        self._b = b
        self._tool_ids: list[str] = [t.id for t in tools]
        self._tokens: list[list[str]] = [_tokenize(_tool_text(t)) for t in tools]
        self._lengths: list[int] = [len(toks) for toks in self._tokens]
        self._tfs: list[Counter[str]] = [Counter(toks) for toks in self._tokens]
        n_docs = len(tools)
        self._avg_len: float = (sum(self._lengths) / n_docs) if n_docs > 0 else 0.0
        # df = number of docs each term appears in
        df: Counter[str] = Counter()
        for toks in self._tokens:
            for term in set(toks):
                df[term] += 1
        # idf with Robertson-Sparck-Jones + 1 smoothing (always >= 0)
        self._idf: dict[str, float] = {
            term: math.log(((n_docs - count + 0.5) / (count + 0.5)) + 1.0)
            for term, count in df.items()
        }

    def retrieve(self, query: str, *, k: int) -> list[tuple[str, float]]:
        if k <= 0 or not self._tool_ids:
            return []
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        scored: list[tuple[str, float]] = []
        for i, tool_id in enumerate(self._tool_ids):
            tf = self._tfs[i]
            length = self._lengths[i]
            score = 0.0
            for term in q_tokens:
                if term not in tf:
                    continue
                idf = self._idf.get(term, 0.0)
                freq = tf[term]
                denom = freq + self._k1 * (
                    1 - self._b + self._b * (length / self._avg_len if self._avg_len else 1.0)
                )
                if denom == 0:
                    continue
                score += idf * (freq * (self._k1 + 1.0)) / denom
            if score > 0:
                scored.append((tool_id, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
