"""Synthetic benchmark - small in-repo corpus.

10 tools across 5 domains (weather, email, calendar, files, billing).
15 query cases hand-labelled with the expected tool id(s).

Useful for:
* CI: zero-dep, fast (~50ms full run), works without an OpenAI key.
* Sanity-checking ranking changes during development.
* As the headline-number benchmark when ToolBench + Gorilla aren't fetched.

NOT useful as a competitive benchmark - the corpus is too small to read
much into the numbers. Treat it as a smoke test. The v0.5 synthetic
200-pair corpus is the real comparison surface.
"""

from __future__ import annotations

from typing import Any

from evals.schema import Case
from toolpicker.sources import FunctionSchemaSource
from toolpicker.types import ToolSource

__all__ = ["SYNTHETIC_CASES", "SYNTHETIC_TOOLS", "SyntheticBenchmark"]


SYNTHETIC_TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather conditions for a given city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
    {
        "name": "get_weather_forecast",
        "description": "Get the multi-day weather forecast for a given city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
        },
    },
    {
        "name": "send_email",
        "description": "Send an email message to a recipient.",
        "parameters": {
            "type": "object",
            "properties": {"to": {"type": "string"}, "subject": {"type": "string"}},
        },
    },
    {
        "name": "search_inbox",
        "description": "Search the user's email inbox for messages matching a query.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    },
    {
        "name": "create_calendar_event",
        "description": "Create a new event on the user's calendar.",
        "parameters": {
            "type": "object",
            "properties": {"title": {"type": "string"}, "start": {"type": "string"}},
        },
    },
    {
        "name": "list_upcoming_events",
        "description": "List the user's upcoming calendar events.",
        "parameters": {"type": "object", "properties": {"days": {"type": "integer"}}},
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file at the given path.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    },
    {
        "name": "list_directory",
        "description": "List the files and subdirectories under a given path.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    },
    {
        "name": "get_order_by_ban",
        "description": "Look up an order by the customer's billing account number.",
        "parameters": {"type": "object", "properties": {"ban": {"type": "string"}}},
    },
    {
        "name": "run_shell_command",
        "description": "Run a shell command on the host.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
    },
]


SYNTHETIC_CASES: list[Case] = [
    # Weather (3)
    Case(
        query="get the current weather for San Francisco",
        expected_tool_ids=["get_current_weather"],
        metadata={"domain": "weather"},
    ),
    Case(
        query="what's the weather forecast for the next 5 days",
        expected_tool_ids=["get_weather_forecast"],
        metadata={"domain": "weather"},
    ),
    Case(
        query="get current weather conditions for Seattle",
        expected_tool_ids=["get_current_weather"],
        metadata={"domain": "weather"},
    ),
    # Email (3)
    Case(
        query="send an email message about the meeting",
        expected_tool_ids=["send_email"],
        metadata={"domain": "email"},
    ),
    Case(
        query="search my inbox for messages from Bob",
        expected_tool_ids=["search_inbox"],
        metadata={"domain": "email"},
    ),
    Case(
        query="send an email to the team",
        expected_tool_ids=["send_email"],
        metadata={"domain": "email"},
    ),
    # Calendar (2)
    Case(
        query="create a calendar event for tomorrow",
        expected_tool_ids=["create_calendar_event"],
        metadata={"domain": "calendar"},
    ),
    Case(
        query="list my upcoming calendar events this week",
        expected_tool_ids=["list_upcoming_events"],
        metadata={"domain": "calendar"},
    ),
    # Files (3)
    Case(
        query="read the contents of the README file",
        expected_tool_ids=["read_file"],
        metadata={"domain": "files"},
    ),
    Case(
        query="read file at /etc/hosts path",
        expected_tool_ids=["read_file"],
        metadata={"domain": "files"},
    ),
    Case(
        query="list the directory contents under /tmp",
        expected_tool_ids=["list_directory"],
        metadata={"domain": "files"},
    ),
    # Billing - lexical-heavy (2)
    Case(
        query="look up the order for BAN 989678111",
        expected_tool_ids=["get_order_by_ban"],
        metadata={"domain": "billing"},
    ),
    Case(
        query="get order by billing account number",
        expected_tool_ids=["get_order_by_ban"],
        metadata={"domain": "billing"},
    ),
    # Shell (2)
    Case(
        query="run a shell command to list processes",
        expected_tool_ids=["run_shell_command"],
        metadata={"domain": "shell"},
    ),
    Case(
        query="run shell command in the workspace",
        expected_tool_ids=["run_shell_command"],
        metadata={"domain": "shell"},
    ),
]


class SyntheticBenchmark:
    """In-repo 10-tool, 15-case benchmark. Cheap path; works without keys."""

    name = "synthetic"

    def tools(self) -> ToolSource:
        return FunctionSchemaSource(SYNTHETIC_TOOLS)

    def cases(self) -> list[Case]:
        return list(SYNTHETIC_CASES)
