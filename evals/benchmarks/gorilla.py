"""Gorilla adapter.

Gorilla is a public API-calling benchmark from UC Berkeley
(https://github.com/ShishirPatil/gorilla). Cleaner shape than ToolBench
(one tool per API entry, ~1700 APIs across torchhub / huggingface /
tensorflowhub) so we wire it up fully in v0.5.

Expected on-disk layout (tweak the globs at the top if your checkout
looks different - schemas have shifted over time):

    <root>/data/api/{torchhub,huggingface,tensorflowhub}_api.jsonl
    <root>/eval/eval-data/questions/{torchhub,huggingface,tensorflowhub}/*0_shot.jsonl
    <root>/eval/eval-data/responses/{torchhub,huggingface,tensorflowhub}/*0_shot.jsonl

Each line of an ``*_api.jsonl`` file is one API: we mint a ``Tool`` per
line. Each line of the questions file is a query; we join to the matching
response line via ``question_id`` to recover the expected api_name. That
api_name maps back to a tool id we built in the first pass.

Tool id format: ``{hub}__{snake(api_name)}``. Collisions inside a single
hub get numeric suffixes.

Limits:
* Default ``max_tools=2000`` / ``max_cases=500`` for tractable eval
  runtime; pass ``None`` to load everything.
* Skips malformed lines instead of crashing.
* Drops cases whose expected api_name didn't appear in the loaded tool
  set (counted in ``stats``).
"""

from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path
from typing import Any

from evals.schema import Case
from toolpicker.sources import FunctionSchemaSource
from toolpicker.types import ToolSource

__all__ = ["GorillaAdapter"]


_ENV_VAR = "TOOLPICKER_GORILLA_DIR"
_HUBS = ("torchhub", "huggingface", "tensorflowhub")
_API_GLOB = "data/api/{hub}_api.jsonl"
# Eval data lives under the nested `gorilla/` subdir in the current repo
# layout (the repo hosts multiple sub-projects under one root).
_QUESTIONS_GLOB = "gorilla/eval/eval-data/questions/{hub}/questions_{hub}_0_shot.jsonl"
# There is no ground-truth file. The "oracle" model run is what the repo
# treats as gold (model had perfect retrieval) so we read its api_call and
# match the api_name back to a tool we loaded.
_ORACLE_GLOB = "gorilla/eval/eval-data/responses/{hub}/response_{hub}_Gorilla_FT_oracle.jsonl"
_SNAKE_RE = re.compile(r"[^a-z0-9]+")


def _to_snake(s: str) -> str:
    s = _SNAKE_RE.sub("_", s.strip().lower()).strip("_")
    return s or "unnamed"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping malformed lines silently."""
    out: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    out.append(obj)
    except OSError:
        return []
    return out


def _pick_api_name(entry: dict[str, Any]) -> str | None:
    """Recover an API name from a Gorilla entry. Schemas have shifted
    over the dataset's lifetime; we check the most common keys.
    """
    for key in ("api_name", "name", "model_id"):
        val = entry.get(key)
        if isinstance(val, str) and val.strip():
            return val
    # Fall back to a hash of the api_call so at least we have an id.
    call = entry.get("api_call")
    if isinstance(call, str) and call.strip():
        return call.strip()[:100]
    return None


# Pattern used as a fallback when ast.literal_eval can't parse the oracle
# response's stringified-dict ``text`` field (some entries have stray
# escapes or unbalanced quotes).
_API_CALL_RE = re.compile(r"['\"]api_call['\"]\s*:\s*['\"](.+?)['\"]\s*[,}]", re.DOTALL)


def _extract_api_call(oracle_text: str) -> str | None:
    """Pull the ``api_call`` string out of an oracle response's text.

    Gorilla's oracle entries store the answer as a stringified Python dict
    in the ``text`` field, e.g.::

        "{'domain': 'Video Classification', 'api_call': \\"model = torch.hub.load(...)\\", ...}"

    Two-tier extraction: try ``ast.literal_eval`` first (handles mixed
    quotes cleanly); fall back to a regex on the literal ``'api_call':``
    key when literal_eval rejects malformed entries.
    """
    if not oracle_text:
        return None
    try:
        parsed = ast.literal_eval(oracle_text)
    except (ValueError, SyntaxError):
        parsed = None
    if isinstance(parsed, dict):
        call = parsed.get("api_call")
        if isinstance(call, str) and call.strip():
            return call
    # Fallback - regex over the raw text.
    match = _API_CALL_RE.search(oracle_text)
    if match:
        return match.group(1)
    return None


def _match_api_name(call: str, api_names: list[str]) -> str | None:
    """Find the longest api_name from ``api_names`` that appears in ``call``.

    Longest-first to handle overlap (``bert`` vs ``bert-base-uncased``).
    Substring match is exact (case-sensitive); api_names are stable model
    identifiers and don't get rewritten between the api JSONL and the
    oracle text.
    """
    if not call:
        return None
    for name in api_names:
        if name and name in call:
            return name
    return None


class GorillaAdapter:
    """Wrap a local Gorilla checkout as a Benchmark.

    Args:
        data_dir: Path to the Gorilla repo root. Defaults to
            ``$TOOLPICKER_GORILLA_DIR`` if set.
        max_tools: Cap on number of tools loaded across all hubs.
            ``None`` = unlimited. Default 2000.
        max_cases: Cap on number of cases loaded across all hubs.
            ``None`` = unlimited. Default 500.
        hubs: Subset of hubs to load (``torchhub``, ``huggingface``,
            ``tensorflowhub``). Defaults to all three.
    """

    name = "gorilla"

    def __init__(
        self,
        data_dir: str | Path | None = None,
        *,
        max_tools: int | None = 2000,
        max_cases: int | None = 500,
        hubs: tuple[str, ...] = _HUBS,
    ) -> None:
        path = data_dir or os.environ.get(_ENV_VAR)
        if not path:
            raise FileNotFoundError(
                "Gorilla data directory not configured. Either:\n"
                "  1. Set the env var: "
                f"$env:{_ENV_VAR} = 'C:\\path\\to\\gorilla'\n"
                "  2. Pass data_dir=... to the constructor.\n"
                "\n"
                "Fetch Gorilla from https://github.com/ShishirPatil/gorilla"
            )
        self._data_dir = Path(path)
        if not self._data_dir.exists():
            raise FileNotFoundError(f"Gorilla data directory does not exist: {self._data_dir}")
        for hub in hubs:
            if hub not in _HUBS:
                raise ValueError(f"unknown hub {hub!r}; expected one of {_HUBS}")
        self._hubs = hubs
        self._max_tools = max_tools
        self._max_cases = max_cases
        self._stats: dict[str, int] = {
            "tools_loaded": 0,
            "cases_loaded": 0,
            "cases_dropped_unknown_tool": 0,
            "cases_dropped_no_response": 0,
        }
        self._cached_tools: list[dict[str, Any]] | None = None
        self._cached_cases: list[Case] | None = None
        # `(hub, api_name)` -> tool id  (case-preserved api_name)
        self._id_map: dict[tuple[str, str], str] = {}
        # Per-hub list of api_names sorted by length descending, for
        # longest-match substring lookup against oracle api_call strings.
        self._api_names_by_hub: dict[str, list[str]] = {}

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def tools(self) -> ToolSource:
        if self._cached_tools is None:
            self._cached_tools = self._load_tools()
        return FunctionSchemaSource(self._cached_tools)

    def cases(self) -> list[Case]:
        if self._cached_tools is None:
            self._cached_tools = self._load_tools()
        if self._cached_cases is None:
            self._cached_cases = self._load_cases()
        return list(self._cached_cases)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_tools(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for hub in self._hubs:
            api_path = self._data_dir / _API_GLOB.format(hub=hub)
            hub_names: list[str] = []
            for entry in _read_jsonl(api_path):
                api_name = _pick_api_name(entry)
                if not api_name:
                    continue
                base_id = f"{hub}__{_to_snake(api_name)}"
                tool_id = base_id
                suffix = 2
                while tool_id in seen_ids:
                    tool_id = f"{base_id}_{suffix}"
                    suffix += 1
                seen_ids.add(tool_id)
                self._id_map[(hub, api_name)] = tool_id
                hub_names.append(api_name)
                description = (
                    entry.get("description")
                    or entry.get("functionality")
                    or entry.get("api_call", "")
                )
                out.append(
                    {
                        "name": tool_id,
                        "description": description if isinstance(description, str) else "",
                        "parameters": {"type": "object", "properties": {}},
                        "keywords": [
                            v
                            for v in (
                                entry.get("domain"),
                                entry.get("framework"),
                                entry.get("functionality"),
                            )
                            if isinstance(v, str)
                        ],
                    }
                )
                if self._max_tools is not None and len(out) >= self._max_tools:
                    # Persist what we collected for this hub before bailing.
                    self._api_names_by_hub[hub] = sorted(hub_names, key=len, reverse=True)
                    self._stats["tools_loaded"] = len(out)
                    return out
            # Sort api_names by length descending so longest-match wins
            # when scanning oracle api_call strings (handles e.g.
            # `bert` vs `bert-base-uncased`).
            self._api_names_by_hub[hub] = sorted(hub_names, key=len, reverse=True)
        self._stats["tools_loaded"] = len(out)
        return out

    def _load_cases(self) -> list[Case]:
        """Build cases by joining questions to oracle responses on question_id.

        Gorilla has no separate ground-truth file. We treat the
        ``response_{hub}_Gorilla_FT_oracle.jsonl`` file as gold - the
        ``oracle`` suffix means the model was given perfect retrieval,
        so its api_call output IS the labelled answer for that query.

        Three drop reasons, all counted in ``stats``:
        * ``cases_dropped_no_response``: question has no matching oracle entry
        * ``cases_dropped_no_api_call``: oracle text didn't yield an api_call
        * ``cases_dropped_unknown_tool``: api_call didn't match any loaded api_name
        """
        self._stats.setdefault("cases_dropped_no_api_call", 0)
        out: list[Case] = []
        for hub in self._hubs:
            q_path = self._data_dir / _QUESTIONS_GLOB.format(hub=hub)
            r_path = self._data_dir / _ORACLE_GLOB.format(hub=hub)
            questions = _read_jsonl(q_path)
            oracle_by_qid: dict[Any, dict[str, Any]] = {}
            for r in _read_jsonl(r_path):
                qid = r.get("question_id")
                if qid is not None:
                    oracle_by_qid[qid] = r
            hub_api_names = self._api_names_by_hub.get(hub, [])
            for q in questions:
                qid = q.get("question_id")
                query = q.get("text") or q.get("question")
                if not isinstance(query, str):
                    continue
                query = query.strip()
                if not query:
                    continue
                oracle = oracle_by_qid.get(qid)
                if oracle is None:
                    self._stats["cases_dropped_no_response"] += 1
                    continue
                oracle_text = oracle.get("text")
                if not isinstance(oracle_text, str):
                    self._stats["cases_dropped_no_api_call"] += 1
                    continue
                # Two-tier match. Prefer matching against the extracted
                # api_call when literal_eval / regex can pull it cleanly -
                # narrower haystack, fewer false positives. Fall back to
                # the entire oracle text when extraction fails: api_names
                # are model identifiers (`slow_r50`, `bert-base-uncased`)
                # which are specific enough that whole-text scan is safe.
                search_text = _extract_api_call(oracle_text) or oracle_text
                matched_name = _match_api_name(search_text, hub_api_names)
                if matched_name is None:
                    self._stats["cases_dropped_unknown_tool"] += 1
                    continue
                tool_id = self._id_map.get((hub, matched_name))
                if tool_id is None:
                    self._stats["cases_dropped_unknown_tool"] += 1
                    continue
                out.append(
                    Case(
                        query=query,
                        expected_tool_ids=[tool_id],
                        metadata={"hub": hub, "question_id": qid},
                    )
                )
                if self._max_cases is not None and len(out) >= self._max_cases:
                    self._stats["cases_loaded"] = len(out)
                    return out
        self._stats["cases_loaded"] = len(out)
        return out
