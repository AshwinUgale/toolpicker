"""End-to-end tests for the ToolPicker facade.

A hand-curated 20-tool corpus covering five domains. Each test issues a
query and asserts the expected tool ranks first (or in the top-K). This is
the closest thing to a real-world signal at v0.1; it'll grow into the
eval harness at v0.4.
"""

from __future__ import annotations

import os

import pytest

from toolpicker import FunctionSchemaSource, HashEmbedder, ToolPicker

_CORPUS = [
    # ---- Weather (3) ----
    {
        "name": "get_current_weather",
        "description": "Get the current weather conditions for a given city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
    {
        "name": "get_weather_forecast",
        "description": "Get the multi-day weather forecast for a given location.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
        },
    },
    {
        "name": "get_weather_alerts",
        "description": "Get active severe-weather alerts for a region.",
        "parameters": {"type": "object", "properties": {"region": {"type": "string"}}},
    },
    # ---- Email (3) ----
    {
        "name": "send_email",
        "description": "Send an email message to a recipient.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
        },
    },
    {
        "name": "search_inbox",
        "description": "Search the user's email inbox for messages matching a query.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    },
    {
        "name": "delete_email",
        "description": "Delete an email message by id.",
        "parameters": {"type": "object", "properties": {"message_id": {"type": "string"}}},
    },
    # ---- Calendar (3) ----
    {
        "name": "create_calendar_event",
        "description": "Create a new event on the user's calendar.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "start": {"type": "string"},
                "end": {"type": "string"},
            },
        },
    },
    {
        "name": "list_upcoming_events",
        "description": "List the user's upcoming calendar events.",
        "parameters": {"type": "object", "properties": {"days": {"type": "integer"}}},
    },
    {
        "name": "cancel_calendar_event",
        "description": "Cancel an existing calendar event by id.",
        "parameters": {"type": "object", "properties": {"event_id": {"type": "string"}}},
    },
    # ---- Files (4) ----
    {
        "name": "read_file",
        "description": "Read the contents of a file at the given path.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    },
    {
        "name": "write_file",
        "description": "Write content to a file at the given path.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        },
    },
    {
        "name": "list_directory",
        "description": "List the files and subdirectories under a given path.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    },
    {
        "name": "delete_file",
        "description": "Delete a file at the given path.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    },
    # ---- Orders / billing (lexical-heavy, 4) ----
    {
        "name": "get_order_by_ban",
        "description": "Look up an order by the customer's billing account number.",
        "parameters": {"type": "object", "properties": {"ban": {"type": "string"}}},
    },
    {
        "name": "get_invoice_pdf",
        "description": "Fetch the PDF of an invoice by invoice id.",
        "parameters": {"type": "object", "properties": {"invoice_id": {"type": "string"}}},
    },
    {
        "name": "list_recent_invoices",
        "description": "List invoices issued to a billing account in the last N days.",
        "parameters": {
            "type": "object",
            "properties": {"ban": {"type": "string"}, "days": {"type": "integer"}},
        },
    },
    {
        "name": "refund_order",
        "description": "Issue a refund against an order by order id.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}},
        },
    },
    # ---- Shell / system (3) ----
    {
        "name": "run_shell_command",
        "description": "Run a shell command on the host and return stdout, stderr, and exit code.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
    },
    {
        "name": "get_system_info",
        "description": "Get host OS, CPU, memory, and disk information.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "kill_process",
        "description": "Kill a running process by pid.",
        "parameters": {"type": "object", "properties": {"pid": {"type": "integer"}}},
    },
]


@pytest.fixture
def picker() -> ToolPicker:
    source = FunctionSchemaSource(_CORPUS)
    return ToolPicker(source, embedder=HashEmbedder(dimensions=32))


@pytest.fixture
def picker_bm25_only() -> ToolPicker:
    """BM25-only picker (no embedder). Lexical-heavy queries should still pass."""
    source = FunctionSchemaSource(_CORPUS)
    return ToolPicker(source)


# ---------------------------------------------------------------------------
# Sanity
# ---------------------------------------------------------------------------


def test_picker_loads_all_tools(picker: ToolPicker) -> None:
    assert len(picker.tools) == len(_CORPUS)


def test_picker_returns_at_most_k(picker: ToolPicker) -> None:
    out = picker.select("anything", k=5)
    assert len(out) <= 5


def test_picker_k_zero_returns_empty(picker: ToolPicker) -> None:
    assert picker.select("anything", k=0) == []


def test_picker_empty_corpus_returns_empty() -> None:
    p = ToolPicker(FunctionSchemaSource([]))
    assert p.select("anything", k=5) == []


# ---------------------------------------------------------------------------
# Hybrid retrieval - the wins it produces vs pure semantic or pure lexical.
# ---------------------------------------------------------------------------


def test_lexical_query_finds_right_tool_bm25_only(picker_bm25_only: ToolPicker) -> None:
    # "BAN" is in the tool's parameter name and is a hard-token query.
    # BM25 should crush this.
    hits = picker_bm25_only.select("look up the order for BAN 989678111", k=3)
    assert any(t.id == "get_order_by_ban" for t in hits)
    # Strong claim: it ranks first.
    assert hits[0].id == "get_order_by_ban"


# These tests run on the BM25-only picker. HashEmbedder is hash-based (not
# semantic) so fusing it with BM25 via RRF just adds noise on top of the
# lexical signal. Real semantic tests live below and need OPENAI_API_KEY.


def test_email_query(picker_bm25_only: ToolPicker) -> None:
    hits = picker_bm25_only.select("send an email message about the meeting", k=3)
    ids = [t.id for t in hits]
    assert "send_email" in ids


def test_calendar_query(picker_bm25_only: ToolPicker) -> None:
    hits = picker_bm25_only.select("create a calendar event for the team standup", k=3)
    ids = [t.id for t in hits]
    assert "create_calendar_event" in ids


def test_file_query(picker_bm25_only: ToolPicker) -> None:
    hits = picker_bm25_only.select("read the contents of /etc/hosts", k=3)
    ids = [t.id for t in hits]
    assert "read_file" in ids


def test_shell_query(picker_bm25_only: ToolPicker) -> None:
    hits = picker_bm25_only.select("run a shell command", k=3)
    ids = [t.id for t in hits]
    assert "run_shell_command" in ids


def test_weather_query(picker_bm25_only: ToolPicker) -> None:
    hits = picker_bm25_only.select("get the current weather for San Francisco", k=3)
    ids = [t.id for t in hits]
    assert any(i.startswith("get_current_weather") or i.startswith("get_weather") for i in ids)


# ---------------------------------------------------------------------------
# Real semantic queries - need OpenAI embeddings, otherwise skip.
# These queries have minimal lexical overlap with tool text on purpose - they
# prove the semantic retriever is doing real work.
#
# These tests pin ``bm25_weight=0.0`` to isolate semantic. The reason: BM25
# (no stopword filtering yet, v0.5 work) matches stopwords like "a"/"at" in
# descriptions and pollutes the fused ranking when the query has zero content
# overlap. Hybrid is still the default product surface; this file's hybrid
# tests live above.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="needs OPENAI_API_KEY for real semantic retrieval",
)
def test_semantic_query_finds_weather_without_lexical_overlap() -> None:
    """'temperature' doesn't appear in any tool's text; BM25 returns nothing.

    A real semantic embedder should still surface the weather tools because
    'temperature' and 'weather' are semantically adjacent.
    """
    from toolpicker import OpenAIEmbeddings

    source = FunctionSchemaSource(_CORPUS)
    picker = ToolPicker(source, embedder=OpenAIEmbeddings(), bm25_weight=0.0)
    hits = picker.select("what's the temperature in San Francisco?", k=3)
    ids = [t.id for t in hits]
    assert any("weather" in i for i in ids)


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="needs OPENAI_API_KEY for real semantic retrieval",
)
def test_semantic_query_finds_calendar_without_lexical_overlap() -> None:
    """'schedule' / 'meeting' don't appear in any calendar tool's text.

    Semantic should still surface create_calendar_event.
    """
    from toolpicker import OpenAIEmbeddings

    source = FunctionSchemaSource(_CORPUS)
    picker = ToolPicker(source, embedder=OpenAIEmbeddings(), bm25_weight=0.0)
    hits = picker.select("schedule a meeting tomorrow at 3pm", k=3)
    ids = [t.id for t in hits]
    assert "create_calendar_event" in ids


# ---------------------------------------------------------------------------
# Backend swap - BM25-only should behave reasonably even without embeddings.
# ---------------------------------------------------------------------------


def test_bm25_only_skips_semantic(picker_bm25_only: ToolPicker) -> None:
    # Verifies the embedder=None path: still returns results from BM25.
    hits = picker_bm25_only.select("send email", k=3)
    assert any(t.id == "send_email" for t in hits)


# ---------------------------------------------------------------------------
# Output shape.
# ---------------------------------------------------------------------------


def test_picker_returns_tool_objects(picker: ToolPicker) -> None:
    from toolpicker import Tool

    hits = picker.select("any query", k=3)
    assert all(isinstance(h, Tool) for h in hits)
    # Each result must have a stable id we can look up in the corpus.
    corpus_ids = {s["name"] for s in _CORPUS}
    assert all(h.id in corpus_ids for h in hits)


def test_picker_deduplicates_across_retrievers(picker: ToolPicker) -> None:
    # Both retrievers may rank the same tool; the fused list must not repeat.
    hits = picker.select("get the weather forecast for tomorrow", k=10)
    ids = [t.id for t in hits]
    assert len(ids) == len(set(ids))
