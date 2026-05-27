"""ToolBench adapter.

ToolBench is a large public benchmark for tool-using agents
(https://github.com/OpenBMB/ToolBench). It's a hefty dataset (~16k+ API
families, 100k+ instances) so we don't bundle it - the user fetches it
once and points us at the directory via ``TOOLPICKER_TOOLBENCH_DIR`` or
the constructor arg.

v0.4 ships the stub: if the data isn't available, the adapter raises a
clear "fetch ToolBench first" error. The real parsing wiring lands in
v0.5 once we've designed the case-extraction strategy.
"""

from __future__ import annotations

import os
from pathlib import Path

from evals.schema import Case
from toolpicker.types import ToolSource

__all__ = ["ToolBenchAdapter"]


_ENV_VAR = "TOOLPICKER_TOOLBENCH_DIR"


class ToolBenchAdapter:
    """Wrap a local ToolBench checkout as a Benchmark.

    Args:
        data_dir: Path to the ToolBench data directory. Defaults to
            ``$TOOLPICKER_TOOLBENCH_DIR`` if set.
    """

    name = "toolbench"

    def __init__(self, data_dir: str | Path | None = None) -> None:
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

    def tools(self) -> ToolSource:
        # v0.5 will parse the ToolBench tool registry. v0.4 stub returns
        # an empty source so callers see a clear "not implemented" path.
        raise NotImplementedError(
            "ToolBenchAdapter.tools() is not implemented in v0.4. "
            "Real parsing lands in v0.5; for now use the SyntheticBenchmark."
        )

    def cases(self) -> list[Case]:
        raise NotImplementedError(
            "ToolBenchAdapter.cases() is not implemented in v0.4. "
            "Real parsing lands in v0.5; for now use the SyntheticBenchmark."
        )

    # Make `data_dir` discoverable for debugging without exposing internals.
    @property
    def data_dir(self) -> Path:
        return self._data_dir
