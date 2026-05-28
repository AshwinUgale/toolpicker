"""Synthetic intent-training corpus - 50 labelled examples.

Disjoint from the 200-case test corpus in ``synthetic.py``. Two distinct
queries per tool across the same 25-tool / 6-domain shape, used by the
eval harness to construct an ``EmbeddingNNIntent`` for the
``intent-only`` and ``bm25+semantic+intent`` strategies in ``compare.py``.

Authoring rule: no query here may appear verbatim in
``SYNTHETIC_CASES``. If a future test-corpus addition collides, the
collision is caught by ``test_synthetic_train_disjoint_from_test`` in
``tests/test_intent.py``.
"""

from __future__ import annotations

from toolpicker.intent import IntentExample

__all__ = ["SYNTHETIC_TRAIN_EXAMPLES"]


def _x(query: str, tool_id: str) -> IntentExample:
    return IntentExample(query=query, tool_id=tool_id)


SYNTHETIC_TRAIN_EXAMPLES: list[IntentExample] = [
    # ---- Weather ----
    _x("is it raining in Tokyo this morning", "get_current_weather"),
    _x("how hot is it in Phoenix", "get_current_weather"),
    _x("will it snow this weekend in Aspen", "get_weather_forecast"),
    _x("ten day outlook for Reykjavik", "get_weather_forecast"),
    _x("any flash flood warnings in Houston", "get_weather_alerts"),
    _x("emergency weather notices for the gulf coast", "get_weather_alerts"),
    _x("temperature at 6pm today in Seattle", "get_hourly_forecast"),
    _x("how will the rain pattern change through the afternoon in london", "get_hourly_forecast"),
    # ---- Email ----
    _x("shoot a quick note to my manager", "send_email"),
    _x("drop the team a line about the new release", "send_email"),
    _x("look through correspondence for the renewal thread", "search_inbox"),
    _x("any mail from the legal team about the contract", "search_inbox"),
    _x("yeet that newsletter spam", "delete_email"),
    _x("purge the receipt from amazon last tuesday", "delete_email"),
    _x("acknowledge the daily standup notes message", "mark_email_read"),
    _x("clear notification bubble on the security alert", "mark_email_read"),
    # ---- Calendar ----
    _x("book a working session with sarah next thursday", "create_calendar_event"),
    _x("pencil in lunch with the recruiter friday", "create_calendar_event"),
    _x("what is my afternoon looking like", "list_upcoming_events"),
    _x("which meetings am I in on monday", "list_upcoming_events"),
    _x("scrap the design review at 4", "cancel_calendar_event"),
    _x("yank the optional 1on1 from my calendar", "cancel_calendar_event"),
    _x("when am I free for a quick coffee chat", "find_free_time"),
    _x("show available 90 minute blocks this week", "find_free_time"),
    # ---- Files ----
    _x("dump out the package.json", "read_file"),
    _x("show me the contents of the dockerfile", "read_file"),
    _x("save these notes to disk under journal", "write_file"),
    _x("flush the buffer into output.csv", "write_file"),
    _x("what files do I have under projects", "list_directory"),
    _x("show subdirectories of the backup folder", "list_directory"),
    _x("trash the temp.txt", "delete_file"),
    _x("nuke the leftover lockfile", "delete_file"),
    _x("rename my old draft to final", "move_file"),
    _x("shuttle the report from staging to production", "move_file"),
    # ---- Billing / Orders ----
    _x("which order belongs to BAN 4040", "get_order_by_ban"),
    _x("pull the order tied to this customer's account number", "get_order_by_ban"),
    _x("download invoice 8881 as a pdf", "get_invoice_pdf"),
    _x("bill receipt pdf for INV-7", "get_invoice_pdf"),
    _x("show last 90 days of bills on account 3030", "list_recent_invoices"),
    _x("recent statements for this billing account", "list_recent_invoices"),
    _x("give the customer their money back on order 1234", "refund_order"),
    _x("reverse the charge for that duplicate transaction", "refund_order"),
    # ---- System ----
    _x("execute lsof for me", "run_shell_command"),
    _x("fire off a curl against the staging endpoint", "run_shell_command"),
    _x("show me hardware specs", "get_system_info"),
    _x("what kernel is this box running", "get_system_info"),
    _x("terminate pid 7777", "kill_process"),
    _x("wipe out the runaway python process", "kill_process"),
    _x("how much space left on the data drive", "check_disk_usage"),
    _x("free space report on /opt", "check_disk_usage"),
]
