"""Smoke test - proves the package imports and the test harness works.

The dumbest possible test on purpose. The point at this stage is to verify:
    1. `uv sync` installed the package in editable mode correctly,
    2. pytest can discover and run tests against installed code,
    3. our src/ layout works without sys.path hacks.
"""

import toolpicker


def test_version_is_a_string() -> None:
    assert isinstance(toolpicker.__version__, str)
    assert toolpicker.__version__ != ""
