"""The `Benchmark` Protocol - what every adapter satisfies."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from evals.schema import Case
from toolpicker.types import ToolSource

__all__ = ["Benchmark"]


@runtime_checkable
class Benchmark(Protocol):
    """A tool corpus + a set of labelled query cases."""

    name: str

    def tools(self) -> ToolSource:
        """The tool corpus this benchmark routes over."""
        ...

    def cases(self) -> list[Case]:
        """The labelled query cases."""
        ...
