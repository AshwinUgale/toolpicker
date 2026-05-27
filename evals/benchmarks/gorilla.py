"""Gorilla adapter.

Gorilla is a public API-calling benchmark from UC Berkeley
(https://github.com/ShishirPatil/gorilla). Cleaner shape than ToolBench
(one tool per API call, ~1700 APIs) and likely the first one we wire up
fully in v0.5.

v0.4 ships the stub: clear error if the data isn't locally available.
"""

from __future__ import annotations

import os
from pathlib import Path

from evals.schema import Case
from toolpicker.types import ToolSource

__all__ = ["GorillaAdapter"]


_ENV_VAR = "TOOLPICKER_GORILLA_DIR"


class GorillaAdapter:
    """Wrap a local Gorilla checkout as a Benchmark.

    Args:
        data_dir: Path to the Gorilla data directory. Defaults to
            ``$TOOLPICKER_GORILLA_DIR`` if set.
    """

    name = "gorilla"

    def __init__(self, data_dir: str | Path | None = None) -> None:
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

    def tools(self) -> ToolSource:
        raise NotImplementedError(
            "GorillaAdapter.tools() is not implemented in v0.4. "
            "Real parsing lands in v0.5; for now use the SyntheticBenchmark."
        )

    def cases(self) -> list[Case]:
        raise NotImplementedError(
            "GorillaAdapter.cases() is not implemented in v0.4. "
            "Real parsing lands in v0.5; for now use the SyntheticBenchmark."
        )

    @property
    def data_dir(self) -> Path:
        return self._data_dir
