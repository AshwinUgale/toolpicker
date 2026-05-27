"""Benchmark adapters.

A benchmark provides (a) the tool corpus the router should pick from and
(b) the labelled query cases. Every benchmark satisfies the `Benchmark`
Protocol in `base.py`.

* `SyntheticBenchmark` - tiny in-repo corpus, no downloads, runs everywhere.
* `ToolBenchAdapter` / `GorillaAdapter` - wrap the public benchmarks; both
  raise a clear "fetch the dataset first" error if the data isn't locally
  available. Real wiring lands in v0.5; v0.4 ships the stubs.
"""

from evals.benchmarks.base import Benchmark
from evals.benchmarks.gorilla import GorillaAdapter
from evals.benchmarks.synthetic import SyntheticBenchmark
from evals.benchmarks.toolbench import ToolBenchAdapter

__all__ = ["Benchmark", "GorillaAdapter", "SyntheticBenchmark", "ToolBenchAdapter"]


_REGISTRY: dict[str, type[Benchmark]] = {
    "synthetic": SyntheticBenchmark,
    "toolbench": ToolBenchAdapter,
    "gorilla": GorillaAdapter,
}


def get_benchmark(name: str) -> Benchmark:
    """Construct a benchmark by name (CLI-friendly)."""
    if name not in _REGISTRY:
        raise ValueError(f"unknown benchmark {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()
