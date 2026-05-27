"""ToolPicker's eval harness.

NOT shipped in the installed wheel. This directory holds:

* ``schema.py`` - Case + CaseResult + RunResult dataclasses
* ``benchmarks/`` - synthetic (cheap path), ToolBench, Gorilla
* ``metrics/`` - Precision@k, MRR, latency stats, tokens-saved
* ``runners/`` - EvalRunner: plays cases through a ToolPicker
* ``__main__.py`` - the CLI
* ``README.md`` - how to run + how to add a benchmark

Run with ``python -m evals --help`` from the repo root.
"""
