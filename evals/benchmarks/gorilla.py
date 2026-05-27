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
_QUESTIONS_GLOB = "eval/eval-data/questions/{hub}/questions_{hub}_0_shot.jsonl"
_RESPONSES_GLOB = "eval/eval-data/responses/{hub}/responses_{hub}_0_shot.jsonl"
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
        # `(hub, api_name_lower)` -> tool id
        self._id_map: dict[tuple[str, str], str] = {}

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
                self._id_map[(hub, api_name.lower())] = tool_id
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
                    self._stats["tools_loaded"] = len(out)
                    return out
        self._stats["tools_loaded"] = len(out)
        return out

    def _load_cases(self) -> list[Case]:
        out: list[Case] = []
        for hub in self._hubs:
            q_path = self._data_dir / _QUESTIONS_GLOB.format(hub=hub)
            r_path = self._data_dir / _RESPONSES_GLOB.format(hub=hub)
            questions = _read_jsonl(q_path)
            responses_by_qid: dict[Any, dict[str, Any]] = {}
            for r in _read_jsonl(r_path):
                qid = r.get("question_id")
                if qid is not None:
                    responses_by_qid[qid] = r
            for q in questions:
                qid = q.get("question_id")
                query = q.get("text") or q.get("question")
                if not isinstance(query, str):
                    continue
                response = responses_by_qid.get(qid)
                if response is None:
                    self._stats["cases_dropped_no_response"] += 1
                    continue
                # The expected api_name lives either at the top level of
                # the response or under an ``api_data`` sub-object.
                api_name = _pick_api_name(response)
                if not api_name:
                    api_data = response.get("api_data")
                    if isinstance(api_data, dict):
                        api_name = _pick_api_name(api_data)
                if not api_name:
                    self._stats["cases_dropped_unknown_tool"] += 1
                    continue
                tool_id = self._id_map.get((hub, api_name.lower()))
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
