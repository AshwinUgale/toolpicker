"""ToolBench adapter.

ToolBench is a large public benchmark for tool-using agents
(https://github.com/OpenBMB/ToolBench). It's a hefty dataset (~16k+ API
families, 100k+ instances) so we don't bundle it - the user fetches it
once and points us at the directory via ``TOOLPICKER_TOOLBENCH_DIR`` or
the constructor arg.

Expected on-disk layout (the v0.5 parser is written against this; if your
checkout looks different, tweak ``_TOOLS_GLOB`` / ``_QUERY_FILES`` at the
top of the file):

    <root>/data/toolenv/tools/<Category>/<ApiFamily>.json
    <root>/data/instruction/G{1,2,3}_query.json

Each tool family JSON contains a top-level ``api_list``; each entry is one
API call we expose as a ``Tool``. Query files are arrays of objects with
a ``query`` string and a ``relevant APIs`` list of ``[tool_name, api_name]``
pairs - those become ``Case.expected_tool_ids``.

Tool id format: ``"{tool_family}__{api_name}"`` after lower-snake-casing
both halves. This guarantees uniqueness across families and matches the
shape we use to look up expected ids from query files.

Limits:
* By default we cap at 2000 tools and 500 cases to keep eval runtime
  bounded; pass ``max_tools=None`` / ``max_cases=None`` to load the full
  set. Loading everything pulls 16k+ tools and is slow.
* Skips families with malformed ``api_list`` rather than crashing.
* Drops cases whose ``relevant APIs`` reference a tool we couldn't load
  (silent - we report a count via ``stats``).
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

__all__ = ["ToolBenchAdapter"]


_ENV_VAR = "TOOLPICKER_TOOLBENCH_DIR"
_TOOLS_GLOB = "data/toolenv/tools/*/*.json"
_QUERY_FILES = ("data/instruction/G1_query.json", "data/instruction/G2_query.json")
_SNAKE_RE = re.compile(r"[^a-z0-9]+")


def _to_snake(s: str) -> str:
    """Lower-snake-case a free-form identifier. Empty -> 'unnamed'."""
    s = _SNAKE_RE.sub("_", s.strip().lower()).strip("_")
    return s or "unnamed"


def _make_tool_id(family: str, api_name: str) -> str:
    return f"{_to_snake(family)}__{_to_snake(api_name)}"


def _params_to_schema(api: dict[str, Any]) -> dict[str, Any]:
    """Convert ToolBench param spec into JSON-schema-ish dict the picker uses.

    ToolBench params come as a list of {"name": ..., "type": ..., "description": ...}
    dicts under ``required_parameters`` and ``optional_parameters``. We
    flatten to {"properties": {name: {"type": ..., "description": ...}}}.
    """
    props: dict[str, dict[str, Any]] = {}
    for key in ("required_parameters", "optional_parameters"):
        for p in api.get(key, []) or []:
            if not isinstance(p, dict):
                continue
            name = p.get("name")
            if not isinstance(name, str) or not name:
                continue
            entry: dict[str, Any] = {}
            ptype = p.get("type")
            if isinstance(ptype, str):
                entry["type"] = (
                    ptype.lower()
                    if ptype.lower()
                    in ("string", "integer", "number", "boolean", "array", "object")
                    else "string"
                )
            desc = p.get("description")
            if isinstance(desc, str):
                entry["description"] = desc
            props[name] = entry or {"type": "string"}
    return {"type": "object", "properties": props}


class ToolBenchAdapter:
    """Wrap a local ToolBench checkout as a Benchmark.

    Args:
        data_dir: Path to the ToolBench data directory. Defaults to
            ``$TOOLPICKER_TOOLBENCH_DIR`` if set.
        max_tools: Cap on number of tools loaded. ``None`` = unlimited.
            Default 2000 to keep eval runtime bounded.
        max_cases: Cap on number of cases loaded. ``None`` = unlimited.
            Default 500.
    """

    name = "toolbench"

    def __init__(
        self,
        data_dir: str | Path | None = None,
        *,
        max_tools: int | None = 2000,
        max_cases: int | None = 500,
    ) -> None:
        path = data_dir or os.environ.get(_ENV_VAR)
        if not path:
            raise FileNotFoundError(
                "ToolBench data directory not configured. Either:\n"
                "  1. Set the env var: "
                f"$env:{_ENV_VAR} = 'C:\\path\\to\\toolbench'\n"
                "  2. Pass data_dir=... to the constructor.\n"
                "\n"
                "Fetch ToolBench from https://github.com/OpenBMB/ToolBench"
            )
        self._data_dir = Path(path)
        if not self._data_dir.exists():
            raise FileNotFoundError(f"ToolBench data directory does not exist: {self._data_dir}")
        self._max_tools = max_tools
        self._max_cases = max_cases
        self._stats: dict[str, int] = {
            "tool_files_scanned": 0,
            "tool_families_skipped": 0,
            "tools_loaded": 0,
            "case_files_scanned": 0,
            "cases_loaded": 0,
            "cases_dropped_unknown_tool": 0,
        }
        self._cached_tools: list[dict[str, Any]] | None = None
        self._cached_cases: list[Case] | None = None
        # Maps ``(tool_family_lower, api_name_lower)`` -> our tool id,
        # built while loading tools and consulted while loading cases.
        self._id_map: dict[tuple[str, str], str] = {}

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @property
    def stats(self) -> dict[str, int]:
        """Counts populated by the load - useful for debugging coverage."""
        return dict(self._stats)

    def tools(self) -> ToolSource:
        if self._cached_tools is None:
            self._cached_tools = self._load_tools()
        return FunctionSchemaSource(self._cached_tools)

    def cases(self) -> list[Case]:
        # Make sure tools loaded first so the id_map is populated.
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
        for tool_path in sorted(self._data_dir.glob(_TOOLS_GLOB)):
            self._stats["tool_files_scanned"] += 1
            try:
                data = json.loads(tool_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._stats["tool_families_skipped"] += 1
                continue
            family = data.get("tool_name") or tool_path.stem
            api_list = data.get("api_list")
            if not isinstance(api_list, list):
                self._stats["tool_families_skipped"] += 1
                continue
            for api in api_list:
                if not isinstance(api, dict):
                    continue
                api_name = api.get("name")
                if not isinstance(api_name, str) or not api_name:
                    continue
                tool_id = _make_tool_id(family, api_name)
                if tool_id in seen_ids:
                    # Two distinct (family, api) pairs collided after
                    # snake-casing. Append a numeric suffix.
                    suffix = 2
                    while f"{tool_id}_{suffix}" in seen_ids:
                        suffix += 1
                    tool_id = f"{tool_id}_{suffix}"
                seen_ids.add(tool_id)
                self._id_map[(family.lower(), api_name.lower())] = tool_id
                description = api.get("description") or family
                out.append(
                    {
                        "name": tool_id,
                        "description": description,
                        "parameters": _params_to_schema(api),
                    }
                )
                if self._max_tools is not None and len(out) >= self._max_tools:
                    self._stats["tools_loaded"] = len(out)
                    return out
        self._stats["tools_loaded"] = len(out)
        return out

    def _load_cases(self) -> list[Case]:
        out: list[Case] = []
        for rel in _QUERY_FILES:
            path = self._data_dir / rel
            if not path.exists():
                continue
            self._stats["case_files_scanned"] += 1
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, list):
                continue
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                query = entry.get("query")
                relevant = entry.get("relevant APIs") or entry.get("relevant_apis")
                if not isinstance(query, str) or not isinstance(relevant, list):
                    continue
                expected_ids: list[str] = []
                for pair in relevant:
                    if not isinstance(pair, list | tuple) or len(pair) < 2:
                        continue
                    family, api_name = pair[0], pair[1]
                    if not isinstance(family, str) or not isinstance(api_name, str):
                        continue
                    tool_id = self._id_map.get((family.lower(), api_name.lower()))
                    if tool_id is not None:
                        expected_ids.append(tool_id)
                if not expected_ids:
                    self._stats["cases_dropped_unknown_tool"] += 1
                    continue
                out.append(
                    Case(
                        query=query,
                        expected_tool_ids=expected_ids,
                        metadata={"source": rel, "query_id": entry.get("query_id")},
                    )
                )
                if self._max_cases is not None and len(out) >= self._max_cases:
                    self._stats["cases_loaded"] = len(out)
                    return out
        self._stats["cases_loaded"] = len(out)
        return out
