"""Synthetic benchmark - in-repo 200-pair corpus.

25 tools across 6 domains (weather, email, calendar, files, billing,
system). 200 query cases hand-labelled with the expected tool id.

Per-tool authoring rule: each tool gets 8 cases spanning different
phrasings, so a strategy can't ace the benchmark by exploiting any
single style:

1. Direct keyword match (tool name tokens in the query)
2. Concrete-param phrasing (numbers, paths, emails, account numbers)
3. Conversational ("can you..." / "I need to...")
4. Imperative short ("send email to bob")
5. Stopword-heavy ("just go ahead and...")
6. Semantic paraphrase (different content words, same intent)
7. Indirect/intent ("ping the team" -> send_email)
8. Domain-context ("for the team standup tomorrow" -> create_calendar_event)

This is the corpus the v0.5 multi-strategy comparison (bm25-only vs
semantic-only vs hybrid-rrf) runs against. Numbers from this benchmark
are headline-worthy; the v0.4 15-case version was a smoke test.

CI runs the full 200 cases through HashEmbedder (fast, no key); the
OpenAI run is a one-shot ~$0.001 to validate the semantic + hybrid
strategies on representative queries.
"""

from __future__ import annotations

from typing import Any

from evals.schema import Case
from toolpicker.sources import FunctionSchemaSource
from toolpicker.types import ToolSource

__all__ = ["SYNTHETIC_CASES", "SYNTHETIC_TOOLS", "SyntheticBenchmark"]


SYNTHETIC_TOOLS: list[dict[str, Any]] = [
    # ---- Weather (4) ----
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
        "name": "get_weather_alerts",
        "description": "Get active severe-weather alerts for a region.",
        "parameters": {"type": "object", "properties": {"region": {"type": "string"}}},
    },
    {
        "name": "get_hourly_forecast",
        "description": "Get the hour-by-hour weather forecast for a city for the next 24 hours.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
    # ---- Email (4) ----
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
    {
        "name": "mark_email_read",
        "description": "Mark an email message as read by id.",
        "parameters": {"type": "object", "properties": {"message_id": {"type": "string"}}},
    },
    # ---- Calendar (4) ----
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
    {
        "name": "find_free_time",
        "description": "Find an open time slot on the user's calendar in the next N days.",
        "parameters": {
            "type": "object",
            "properties": {"duration_minutes": {"type": "integer"}, "days": {"type": "integer"}},
        },
    },
    # ---- Files (5) ----
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
    {
        "name": "move_file",
        "description": "Move or rename a file from a source path to a destination path.",
        "parameters": {
            "type": "object",
            "properties": {"src": {"type": "string"}, "dst": {"type": "string"}},
        },
    },
    # ---- Billing / Orders (4) ----
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
    # ---- System (4) ----
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
    {
        "name": "check_disk_usage",
        "description": "Report disk usage and free space on a given filesystem mount.",
        "parameters": {"type": "object", "properties": {"mount": {"type": "string"}}},
    },
]


def _c(query: str, tool_id: str, domain: str, style: str) -> Case:
    """Tiny helper to keep the 200-row table dense and readable."""
    return Case(
        query=query,
        expected_tool_ids=[tool_id],
        metadata={"domain": domain, "style": style},
    )


SYNTHETIC_CASES: list[Case] = [
    # ===================================================================
    # WEATHER
    # ===================================================================
    # get_current_weather
    _c("get the current weather for San Francisco", "get_current_weather", "weather", "direct"),
    _c(
        "current weather conditions in Seattle right now", "get_current_weather", "weather", "param"
    ),
    _c(
        "can you tell me the weather in Boston today?",
        "get_current_weather",
        "weather",
        "conversational",
    ),
    _c("weather in NYC", "get_current_weather", "weather", "imperative"),
    _c(
        "so I just need to know the weather in Austin right now",
        "get_current_weather",
        "weather",
        "stopword_heavy",
    ),
    _c("how warm is it in Miami at the moment", "get_current_weather", "weather", "paraphrase"),
    _c("what should I wear outside in Denver today", "get_current_weather", "weather", "indirect"),
    _c(
        "temperature in Portland for my morning commute",
        "get_current_weather",
        "weather",
        "domain_context",
    ),
    # get_weather_forecast
    _c(
        "get the multi-day weather forecast for Chicago",
        "get_weather_forecast",
        "weather",
        "direct",
    ),
    _c("5-day forecast for Tokyo", "get_weather_forecast", "weather", "param"),
    _c(
        "can you pull the week-long forecast for London?",
        "get_weather_forecast",
        "weather",
        "conversational",
    ),
    _c("forecast Paris 7 days", "get_weather_forecast", "weather", "imperative"),
    _c(
        "just give me the forecast for the next few days in LA",
        "get_weather_forecast",
        "weather",
        "stopword_heavy",
    ),
    _c(
        "rain predictions for the upcoming week in Vancouver",
        "get_weather_forecast",
        "weather",
        "paraphrase",
    ),
    _c(
        "planning a hike next weekend in Yosemite, will it rain?",
        "get_weather_forecast",
        "weather",
        "indirect",
    ),
    _c(
        "forecast for the trip to Amsterdam Friday through Sunday",
        "get_weather_forecast",
        "weather",
        "domain_context",
    ),
    # get_weather_alerts
    _c("get active severe weather alerts for Texas", "get_weather_alerts", "weather", "direct"),
    _c("severe weather alerts in the Bay Area region", "get_weather_alerts", "weather", "param"),
    _c(
        "are there any storm warnings in Florida right now?",
        "get_weather_alerts",
        "weather",
        "conversational",
    ),
    _c("alerts Texas", "get_weather_alerts", "weather", "imperative"),
    _c(
        "just check if there are any weather alerts out for the south",
        "get_weather_alerts",
        "weather",
        "stopword_heavy",
    ),
    _c("tornado warnings active in Oklahoma", "get_weather_alerts", "weather", "paraphrase"),
    _c("is it safe to drive through Kansas tonight", "get_weather_alerts", "weather", "indirect"),
    _c(
        "evacuation advisories for the hurricane in the southeast",
        "get_weather_alerts",
        "weather",
        "domain_context",
    ),
    # get_hourly_forecast
    _c(
        "hourly weather forecast for the next 24 hours in SF",
        "get_hourly_forecast",
        "weather",
        "direct",
    ),
    _c("hour by hour forecast for Boston", "get_hourly_forecast", "weather", "param"),
    _c(
        "can you give me an hourly breakdown of the weather today?",
        "get_hourly_forecast",
        "weather",
        "conversational",
    ),
    _c("hourly forecast Chicago", "get_hourly_forecast", "weather", "imperative"),
    _c(
        "just an hour by hour for the day in NYC please",
        "get_hourly_forecast",
        "weather",
        "stopword_heavy",
    ),
    _c(
        "temperature changes throughout today in Seattle",
        "get_hourly_forecast",
        "weather",
        "paraphrase",
    ),
    _c(
        "when's the best 2-hour window today to walk the dog",
        "get_hourly_forecast",
        "weather",
        "indirect",
    ),
    _c(
        "hour-by-hour weather for the outdoor wedding this afternoon",
        "get_hourly_forecast",
        "weather",
        "domain_context",
    ),
    # ===================================================================
    # EMAIL
    # ===================================================================
    # send_email
    _c("send an email to bob@example.com about the meeting", "send_email", "email", "direct"),
    _c(
        "send email to alice@team.com subject Q4 review body see attached",
        "send_email",
        "email",
        "param",
    ),
    _c(
        "can you send a message to the team about the launch?",
        "send_email",
        "email",
        "conversational",
    ),
    _c("email bob the report", "send_email", "email", "imperative"),
    _c(
        "just go ahead and send an email to all of the team about it",
        "send_email",
        "email",
        "stopword_heavy",
    ),
    _c("compose a message and dispatch it to my manager", "send_email", "email", "paraphrase"),
    _c("ping the team that the build is green", "send_email", "email", "indirect"),
    _c(
        "let stakeholders know the migration shipped, via email",
        "send_email",
        "email",
        "domain_context",
    ),
    # search_inbox
    _c("search my email inbox for messages from Bob", "search_inbox", "email", "direct"),
    _c("search inbox query invoice", "search_inbox", "email", "param"),
    _c(
        "can you look through my email for that thread about Q4?",
        "search_inbox",
        "email",
        "conversational",
    ),
    _c("find emails from sarah last week", "search_inbox", "email", "imperative"),
    _c(
        "just look in my inbox for any of the ones about the review",
        "search_inbox",
        "email",
        "stopword_heavy",
    ),
    _c(
        "hunt down the message thread on the contract renewal",
        "search_inbox",
        "email",
        "paraphrase",
    ),
    _c("what did the legal team send me about that NDA", "search_inbox", "email", "indirect"),
    _c(
        "dig up my old email exchanges with the recruiter at acme",
        "search_inbox",
        "email",
        "domain_context",
    ),
    # delete_email
    _c("delete the email message with id MSG-12345", "delete_email", "email", "direct"),
    _c("delete email message_id MSG-99887", "delete_email", "email", "param"),
    _c("can you remove that email from my inbox?", "delete_email", "email", "conversational"),
    _c("delete email 4567", "delete_email", "email", "imperative"),
    _c(
        "just go ahead and delete that one email out of my inbox",
        "delete_email",
        "email",
        "stopword_heavy",
    ),
    _c("trash the message thread permanently", "delete_email", "email", "paraphrase"),
    _c("get rid of that phishing email I just got", "delete_email", "email", "indirect"),
    _c("clear out the auto-renewal notification email", "delete_email", "email", "domain_context"),
    # mark_email_read
    _c("mark email message id MSG-555 as read", "mark_email_read", "email", "direct"),
    _c("mark email read message_id 8923", "mark_email_read", "email", "param"),
    _c("can you mark this email as read?", "mark_email_read", "email", "conversational"),
    _c("mark as read 12345", "mark_email_read", "email", "imperative"),
    _c("just flag those as read for me in the inbox", "mark_email_read", "email", "stopword_heavy"),
    _c("flip the unread flag off on that message", "mark_email_read", "email", "paraphrase"),
    _c("clear the unread badge from the newsletter", "mark_email_read", "email", "indirect"),
    _c(
        "dismiss the unread indicator on the daily digest message",
        "mark_email_read",
        "email",
        "domain_context",
    ),
    # ===================================================================
    # CALENDAR
    # ===================================================================
    # create_calendar_event
    _c(
        "create a calendar event for the team standup tomorrow",
        "create_calendar_event",
        "calendar",
        "direct",
    ),
    _c(
        "create calendar event title Q4 review start 2026-06-01T14:00 end 2026-06-01T15:00",
        "create_calendar_event",
        "calendar",
        "param",
    ),
    _c(
        "can you put something on my calendar for Friday at 2?",
        "create_calendar_event",
        "calendar",
        "conversational",
    ),
    _c(
        "book a 30 min event next Tuesday at 10am",
        "create_calendar_event",
        "calendar",
        "imperative",
    ),
    _c(
        "just go ahead and put it on the calendar for me on tuesday",
        "create_calendar_event",
        "calendar",
        "stopword_heavy",
    ),
    _c(
        "schedule a meeting with the design team next week",
        "create_calendar_event",
        "calendar",
        "paraphrase",
    ),
    _c(
        "block off an hour for deep work tomorrow morning",
        "create_calendar_event",
        "calendar",
        "indirect",
    ),
    _c(
        "set up the kickoff sync for the migration project monday",
        "create_calendar_event",
        "calendar",
        "domain_context",
    ),
    # list_upcoming_events
    _c(
        "list my upcoming calendar events for this week",
        "list_upcoming_events",
        "calendar",
        "direct",
    ),
    _c("list upcoming events days 7", "list_upcoming_events", "calendar", "param"),
    _c(
        "can you tell me what's on my calendar this week?",
        "list_upcoming_events",
        "calendar",
        "conversational",
    ),
    _c("show calendar next 3 days", "list_upcoming_events", "calendar", "imperative"),
    _c(
        "just give me a list of all of the upcoming things on my calendar",
        "list_upcoming_events",
        "calendar",
        "stopword_heavy",
    ),
    _c("what meetings do I have coming up", "list_upcoming_events", "calendar", "paraphrase"),
    _c(
        "what's my schedule for the rest of the week look like",
        "list_upcoming_events",
        "calendar",
        "indirect",
    ),
    _c(
        "agenda for tomorrow before the all-hands at 11",
        "list_upcoming_events",
        "calendar",
        "domain_context",
    ),
    # cancel_calendar_event
    _c("cancel the calendar event with id EVT-7788", "cancel_calendar_event", "calendar", "direct"),
    _c("cancel calendar event event_id EVT-2200", "cancel_calendar_event", "calendar", "param"),
    _c(
        "can you cancel that event I scheduled for Friday?",
        "cancel_calendar_event",
        "calendar",
        "conversational",
    ),
    _c("cancel meeting 3344", "cancel_calendar_event", "calendar", "imperative"),
    _c(
        "just go ahead and cancel that one for me on the calendar",
        "cancel_calendar_event",
        "calendar",
        "stopword_heavy",
    ),
    _c(
        "remove the scheduled appointment from my agenda",
        "cancel_calendar_event",
        "calendar",
        "paraphrase",
    ),
    _c(
        "drop the 4pm sync, I won't be able to make it",
        "cancel_calendar_event",
        "calendar",
        "indirect",
    ),
    _c(
        "call off the demo session with the client today",
        "cancel_calendar_event",
        "calendar",
        "domain_context",
    ),
    # find_free_time
    _c(
        "find a free time slot on my calendar for a 30 minute call",
        "find_free_time",
        "calendar",
        "direct",
    ),
    _c("find free time duration_minutes 45 days 5", "find_free_time", "calendar", "param"),
    _c(
        "can you find me an open hour this week for a deep dive?",
        "find_free_time",
        "calendar",
        "conversational",
    ),
    _c("free 60 min slot next 7 days", "find_free_time", "calendar", "imperative"),
    _c(
        "just find me any open slot in the next few days for an hour",
        "find_free_time",
        "calendar",
        "stopword_heavy",
    ),
    _c(
        "when do I have an open block for a 90 minute review",
        "find_free_time",
        "calendar",
        "paraphrase",
    ),
    _c(
        "when can I squeeze in a quick chat with sarah this week",
        "find_free_time",
        "calendar",
        "indirect",
    ),
    _c(
        "availability for the offsite planning sync, 2 hours, next two weeks",
        "find_free_time",
        "calendar",
        "domain_context",
    ),
    # ===================================================================
    # FILES
    # ===================================================================
    # read_file
    _c("read the contents of the file at /etc/hosts", "read_file", "files", "direct"),
    _c("read file path ./src/main.py", "read_file", "files", "param"),
    _c("can you open and read this config file for me?", "read_file", "files", "conversational"),
    _c("cat /var/log/syslog", "read_file", "files", "imperative"),
    _c("just go and read what's in the readme for me", "read_file", "files", "stopword_heavy"),
    _c("show me what's inside the json config", "read_file", "files", "paraphrase"),
    _c("what does the .env example file look like", "read_file", "files", "indirect"),
    _c("pull up the contents of the migration sql script", "read_file", "files", "domain_context"),
    # write_file
    _c("write content to a file at /tmp/output.txt", "write_file", "files", "direct"),
    _c("write file path ./notes.md content hello world", "write_file", "files", "param"),
    _c("can you save this text to a file on disk?", "write_file", "files", "conversational"),
    _c("save buffer to /tmp/draft.md", "write_file", "files", "imperative"),
    _c(
        "just go ahead and put the content into a file in the workspace",
        "write_file",
        "files",
        "stopword_heavy",
    ),
    _c("persist the generated report to disk as markdown", "write_file", "files", "paraphrase"),
    _c(
        "dump the api response into a file so I can inspect later",
        "write_file",
        "files",
        "indirect",
    ),
    _c(
        "save the formatted log output to logs/run-2026.txt",
        "write_file",
        "files",
        "domain_context",
    ),
    # list_directory
    _c("list the directory contents under /home/user/docs", "list_directory", "files", "direct"),
    _c("list directory path /var/log", "list_directory", "files", "param"),
    _c(
        "can you show me what files are in this folder?",
        "list_directory",
        "files",
        "conversational",
    ),
    _c("ls /tmp", "list_directory", "files", "imperative"),
    _c(
        "just give me a list of all of the files in that one folder",
        "list_directory",
        "files",
        "stopword_heavy",
    ),
    _c("enumerate the entries in the source directory", "list_directory", "files", "paraphrase"),
    _c("what's in my downloads folder right now", "list_directory", "files", "indirect"),
    _c(
        "show subdirectories under the repo root for the migration audit",
        "list_directory",
        "files",
        "domain_context",
    ),
    # delete_file
    _c("delete the file at /tmp/old.log", "delete_file", "files", "direct"),
    _c("delete file path ./cache/stale.bin", "delete_file", "files", "param"),
    _c("can you remove this file from disk?", "delete_file", "files", "conversational"),
    _c("rm /tmp/scratch.txt", "delete_file", "files", "imperative"),
    _c(
        "just go ahead and delete that one file out of the cache folder",
        "delete_file",
        "files",
        "stopword_heavy",
    ),
    _c("erase the artifact from the build directory", "delete_file", "files", "paraphrase"),
    _c(
        "get rid of the leftover lock file from yesterday's crash",
        "delete_file",
        "files",
        "indirect",
    ),
    _c(
        "clean up the corrupted database snapshot from /backup",
        "delete_file",
        "files",
        "domain_context",
    ),
    # move_file
    _c("move the file from /tmp/a.txt to /home/user/a.txt", "move_file", "files", "direct"),
    _c("move file src ./build/out.js dst ./dist/out.js", "move_file", "files", "param"),
    _c("can you move this file to a different folder?", "move_file", "files", "conversational"),
    _c("mv old.csv data/old.csv", "move_file", "files", "imperative"),
    _c(
        "just go ahead and move that one over to the dist folder for me",
        "move_file",
        "files",
        "stopword_heavy",
    ),
    _c("relocate the asset from staging to production", "move_file", "files", "paraphrase"),
    _c("shift the report into the archive folder once it's done", "move_file", "files", "indirect"),
    _c(
        "rename release-candidate.tar.gz to release-final.tar.gz",
        "move_file",
        "files",
        "domain_context",
    ),
    # ===================================================================
    # BILLING / ORDERS
    # ===================================================================
    # get_order_by_ban
    _c(
        "look up the order by billing account number 989678111",
        "get_order_by_ban",
        "billing",
        "direct",
    ),
    _c("get order by ban 123456789", "get_order_by_ban", "billing", "param"),
    _c(
        "can you pull up the order tied to this customer's BAN?",
        "get_order_by_ban",
        "billing",
        "conversational",
    ),
    _c("order for BAN 5500", "get_order_by_ban", "billing", "imperative"),
    _c(
        "just look up the order on that billing account number for me",
        "get_order_by_ban",
        "billing",
        "stopword_heavy",
    ),
    _c(
        "fetch the purchase record linked to the billing account",
        "get_order_by_ban",
        "billing",
        "paraphrase",
    ),
    _c(
        "what did this customer with BAN 999111 buy last", "get_order_by_ban", "billing", "indirect"
    ),
    _c(
        "retrieve the order placed under TFB billing account 8765432",
        "get_order_by_ban",
        "billing",
        "domain_context",
    ),
    # get_invoice_pdf
    _c("fetch the PDF of the invoice with id INV-7788", "get_invoice_pdf", "billing", "direct"),
    _c("get invoice pdf invoice_id INV-2025-001", "get_invoice_pdf", "billing", "param"),
    _c(
        "can you download the PDF for invoice 4567?", "get_invoice_pdf", "billing", "conversational"
    ),
    _c("invoice pdf INV-9001", "get_invoice_pdf", "billing", "imperative"),
    _c(
        "just go ahead and get the pdf of that one invoice for me",
        "get_invoice_pdf",
        "billing",
        "stopword_heavy",
    ),
    _c(
        "download the billing receipt document for that charge",
        "get_invoice_pdf",
        "billing",
        "paraphrase",
    ),
    _c("send me the bill from last month as a pdf", "get_invoice_pdf", "billing", "indirect"),
    _c(
        "attach the june invoice pdf to the customer support ticket",
        "get_invoice_pdf",
        "billing",
        "domain_context",
    ),
    # list_recent_invoices
    _c(
        "list the invoices issued to billing account 555 in the last 30 days",
        "list_recent_invoices",
        "billing",
        "direct",
    ),
    _c("list recent invoices ban 7788 days 60", "list_recent_invoices", "billing", "param"),
    _c(
        "can you show the invoices from the last month for this account?",
        "list_recent_invoices",
        "billing",
        "conversational",
    ),
    _c("recent invoices ban 4400 last 14 days", "list_recent_invoices", "billing", "imperative"),
    _c(
        "just list out all of the most recent invoices on that account",
        "list_recent_invoices",
        "billing",
        "stopword_heavy",
    ),
    _c(
        "billing statements from the past few weeks for that customer",
        "list_recent_invoices",
        "billing",
        "paraphrase",
    ),
    _c(
        "what has the customer been charged lately for the BAN 998",
        "list_recent_invoices",
        "billing",
        "indirect",
    ),
    _c(
        "invoicing history for the dispute resolution case",
        "list_recent_invoices",
        "billing",
        "domain_context",
    ),
    # refund_order
    _c("issue a refund against the order with id ORD-4400", "refund_order", "billing", "direct"),
    _c("refund order order_id ORD-9911 amount 49.99", "refund_order", "billing", "param"),
    _c("can you refund this order for the customer?", "refund_order", "billing", "conversational"),
    _c("refund ORD-3322", "refund_order", "billing", "imperative"),
    _c(
        "just go ahead and refund the order in the system for them",
        "refund_order",
        "billing",
        "stopword_heavy",
    ),
    _c("reverse the charge on that purchase", "refund_order", "billing", "paraphrase"),
    _c("the customer wants their money back for order 4567", "refund_order", "billing", "indirect"),
    _c(
        "process the chargeback for the duplicate transaction on order ORD-7788",
        "refund_order",
        "billing",
        "domain_context",
    ),
    # ===================================================================
    # SYSTEM
    # ===================================================================
    # run_shell_command
    _c(
        "run a shell command to list processes on the host", "run_shell_command", "system", "direct"
    ),
    _c("run shell command ps aux", "run_shell_command", "system", "param"),
    _c(
        "can you execute this bash command for me?", "run_shell_command", "system", "conversational"
    ),
    _c("shell uname -a", "run_shell_command", "system", "imperative"),
    _c(
        "just run the command on the system for me real quick",
        "run_shell_command",
        "system",
        "stopword_heavy",
    ),
    _c(
        "invoke a terminal command and return the output",
        "run_shell_command",
        "system",
        "paraphrase",
    ),
    _c("check what version of node is installed", "run_shell_command", "system", "indirect"),
    _c(
        "compile the typescript project from the shell with tsc --noEmit",
        "run_shell_command",
        "system",
        "domain_context",
    ),
    # get_system_info
    _c(
        "get system info for the host: OS, CPU, memory, disk", "get_system_info", "system", "direct"
    ),
    _c("get system info", "get_system_info", "system", "param"),
    _c(
        "can you tell me the system specs of this machine?",
        "get_system_info",
        "system",
        "conversational",
    ),
    _c("system info", "get_system_info", "system", "imperative"),
    _c(
        "just give me the system info on that one host for me",
        "get_system_info",
        "system",
        "stopword_heavy",
    ),
    _c("report the hardware and OS details of the box", "get_system_info", "system", "paraphrase"),
    _c("how much RAM does this server have", "get_system_info", "system", "indirect"),
    _c(
        "CPU and memory profile of the production host for the postmortem",
        "get_system_info",
        "system",
        "domain_context",
    ),
    # kill_process
    _c("kill the running process with pid 9876", "kill_process", "system", "direct"),
    _c("kill process pid 1234", "kill_process", "system", "param"),
    _c("can you terminate that process for me?", "kill_process", "system", "conversational"),
    _c("kill -9 4321", "kill_process", "system", "imperative"),
    _c(
        "just go ahead and kill the process by its pid in the system",
        "kill_process",
        "system",
        "stopword_heavy",
    ),
    _c("terminate the runaway worker by its process id", "kill_process", "system", "paraphrase"),
    _c("the node server is hung, take it down", "kill_process", "system", "indirect"),
    _c(
        "end the stuck postgres backend process eating CPU",
        "kill_process",
        "system",
        "domain_context",
    ),
    # check_disk_usage
    _c("check disk usage and free space on the / mount", "check_disk_usage", "system", "direct"),
    _c("check disk usage mount /var", "check_disk_usage", "system", "param"),
    _c(
        "can you tell me how much disk space is left on this volume?",
        "check_disk_usage",
        "system",
        "conversational",
    ),
    _c("df /home", "check_disk_usage", "system", "imperative"),
    _c(
        "just tell me how much disk is free on the system for me",
        "check_disk_usage",
        "system",
        "stopword_heavy",
    ),
    _c(
        "storage capacity and remaining headroom on the data partition",
        "check_disk_usage",
        "system",
        "paraphrase",
    ),
    _c(
        "are we about to run out of space on the data drive",
        "check_disk_usage",
        "system",
        "indirect",
    ),
    _c(
        "free space report on /var/lib/postgresql for the capacity audit",
        "check_disk_usage",
        "system",
        "domain_context",
    ),
]


class SyntheticBenchmark:
    """In-repo 25-tool, 200-case benchmark. Cheap path; works without keys."""

    name = "synthetic"

    def tools(self) -> ToolSource:
        return FunctionSchemaSource(SYNTHETIC_TOOLS)

    def cases(self) -> list[Case]:
        return list(SYNTHETIC_CASES)
