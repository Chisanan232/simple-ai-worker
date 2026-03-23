"""
E2E tests: Dev Agent Slack thread summary — ClickUp backend (E2E-01 through E2E-06).

These tests verify the full path:
  Slack @mention with [dev] tag in a thread
    → dev_handler dispatches crew
      → LLM reads thread via slack/get_messages (MCP JSON-RPC stub)
        → LLM posts comment via clickup/add_comment (stub)
          → LLM replies in thread via slack/reply_to_thread (stub)

The LLM call is **real** (uses OPENAI_API_KEY from environment).
All MCP HTTP calls are intercepted by MCPStubServer (valid JSON-RPC 2.0).
Tests are skipped when no LLM API key is present.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import MagicMock

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

from test.e2e_test.conftest import (
    MCPStubServer,
    build_dev_agent_against_stubs,
    build_e2e_registry,
    skip_without_llm,
)


def _run_dev_handler_sync(event: dict, registry: Any) -> None:
    """Run dev_handler and block until the background crew completes."""
    from src.slack_app.handlers.dev import dev_handler

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        dev_handler(
            text=event.get("text", ""),
            event=event,
            say=MagicMock(),
            registry=registry,
            executor=executor,
        )
        executor.shutdown(wait=True, cancel_futures=False)
    finally:
        executor.shutdown(wait=False)


# ===========================================================================
# E2E-01: Dev reads thread and updates ClickUp task
# ===========================================================================

@skip_without_llm
class TestDevReadsThreadAndUpdatesTicket:
    def test_dev_reads_thread_and_updates_ticket(
        self,
        mcp_stub: MCPStubServer,
        thread_summary_tool_order: None,
    ) -> None:
        """E2E-01: [dev] in thread → LLM reads messages → posts clickup/add_comment."""
        stub = mcp_stub
        stub.register_tool("get_messages", lambda args: {
            "ok": True,
            "messages": [
                {"user": "U1", "text": "We agreed: implement OAuth2. Task: abc123.", "ts": "1.1"},
                {"user": "U2", "text": "Use Google SSO. Task: abc123.", "ts": "1.2"},
            ],
        })
        stub.register_tool("get_thread_permalink", lambda args: {
            "ok": True, "permalink": "https://slack.com/archives/C001/p111222",
        })
        stub.register_tool("add_comment", lambda args: {
            "id": "c-cu-1", "ok": True,
        })
        stub.register_tool("reply_to_thread", lambda args: {"ok": True})
        stub.register_tool("get_task", lambda args: {
            "id": "abc123", "name": "Implement OAuth2",
            "status": {"status": "ACCEPTED"},
        })

        url = stub.url
        dev_agent = build_dev_agent_against_stubs(
            jira_url=url, slack_url=url, github_url=url, clickup_url=url,
        )
        registry = build_e2e_registry(dev_agent)
        _run_dev_handler_sync(
            event={"text": "<@UBOT> [dev] please update the ticket",
                   "channel": "C001", "thread_ts": "111.222", "ts": "111.222"},
            registry=registry,
        )

        assert stub.was_called("get_messages"), (
            "Expected LLM to call slack/get_messages to fetch thread history"
        )
        assert stub.was_called("add_comment"), (
            "Expected LLM to call clickup/add_comment on the task"
        )


# ===========================================================================
# E2E-02: Dev asks for ticket when none found in thread (BR-6)
# ===========================================================================

@skip_without_llm
class TestDevAsksForTicketWhenNoneFound:
    def test_dev_asks_for_ticket_when_none_found(
        self,
        mcp_stub: MCPStubServer,
        reply_only_tool_order: None,
    ) -> None:
        """E2E-02: Thread with no ticket ID → LLM asks supervisor, never calls add_comment."""
        stub = mcp_stub
        stub.register_tool("get_messages", lambda args: {
            "ok": True,
            "messages": [
                {"user": "U1", "text": "We decided to use Redis for caching.", "ts": "1.1"},
                {"user": "U2", "text": "Agreed, Redis Cluster.", "ts": "1.2"},
            ],
        })
        stub.register_tool("get_thread_permalink", lambda args: {
            "ok": True, "permalink": "https://slack.com/archives/C002/p222333",
        })
        stub.register_tool("reply_to_thread", lambda args: {"ok": True})

        url = stub.url
        dev_agent = build_dev_agent_against_stubs(
            jira_url=url, slack_url=url, github_url=url, clickup_url=url,
        )
        registry = build_e2e_registry(dev_agent)
        _run_dev_handler_sync(
            event={"text": "<@UBOT> [dev] read this thread",
                   "channel": "C002", "thread_ts": "222.333", "ts": "222.333"},
            registry=registry,
        )

        assert len(stub.calls_to("add_comment")) == 0, (
            f"add_comment must NOT be called when no ticket ID found (BR-6). "
            f"Got: {stub.calls_to('add_comment')}"
        )
        assert stub.was_called("reply_to_thread"), (
            "Expected LLM to call slack/reply_to_thread to ask for ticket ID (BR-6)"
        )


# ===========================================================================
# E2E-03: Slack permalink included in ticket comment
# ===========================================================================

@skip_without_llm
class TestSlackPermalinkInComment:
    def test_permalink_in_ticket_comment(
        self,
        mcp_stub: MCPStubServer,
        permalink_tool_order: None,
    ) -> None:
        """E2E-03: ClickUp comment body must contain the Slack thread permalink."""
        stub = mcp_stub
        channel_id = "C123"
        expected_permalink = f"https://slack.com/archives/{channel_id}/p111222"
        comment_bodies: list[str] = []

        def _add_comment(args: dict) -> dict:
            comment_bodies.append(str(args))
            return {"id": "c-cu-1", "ok": True}

        stub.register_tool("get_messages", lambda args: {
            "ok": True,
            "messages": [
                {"user": "U1", "text": "Implement feature for task abc123.", "ts": "1.1"},
                {"user": "U2", "text": "Agreed, task abc123.", "ts": "1.2"},
            ],
        })
        stub.register_tool("get_thread_permalink", lambda args: {
            "ok": True, "permalink": expected_permalink,
        })
        stub.register_tool("add_comment", _add_comment)
        stub.register_tool("reply_to_thread", lambda args: {"ok": True})
        stub.register_tool("get_task", lambda args: {
            "id": "abc123", "name": "Implement feature",
            "status": {"status": "ACCEPTED"},
        })

        url = stub.url
        dev_agent = build_dev_agent_against_stubs(
            jira_url=url, slack_url=url, github_url=url, clickup_url=url,
        )
        registry = build_e2e_registry(dev_agent)
        _run_dev_handler_sync(
            event={"text": "<@UBOT> [dev] update ticket with this discussion",
                   "channel": channel_id, "thread_ts": "111.222", "ts": "111.222"},
            registry=registry,
        )

        assert len(comment_bodies) > 0, "Expected clickup/add_comment to be called"
        all_comment_text = " ".join(comment_bodies)
        assert "slack.com/archives" in all_comment_text, (
            f"Expected Slack permalink in ClickUp comment. Got: {all_comment_text[:500]}"
        )


# ===========================================================================
# E2E-05: No new ticket created (guardrail)
# ===========================================================================

@skip_without_llm
class TestNoNewTicketCreated:
    def test_no_new_ticket_created(
        self,
        mcp_stub: MCPStubServer,
    ) -> None:
        """E2E-05: LLM must never call create_task or create_issue."""
        stub = mcp_stub
        create_calls: list = []

        def _create(args: dict) -> dict:
            create_calls.append(args)
            return {"id": "x"}

        stub.register_tool("get_messages", lambda args: {
            "ok": True,
            "messages": [
                {"user": "U1", "text": "Discussed task abc123: use async processing.", "ts": "1.1"},
            ],
        })
        stub.register_tool("get_thread_permalink", lambda args: {
            "ok": True, "permalink": "https://slack.com/archives/C003/p333",
        })
        stub.register_tool("add_comment", lambda args: {"id": "c1"})
        stub.register_tool("reply_to_thread", lambda args: {"ok": True})
        stub.register_tool("create_issue", _create)
        stub.register_tool("create_task", _create)

        url = stub.url
        dev_agent = build_dev_agent_against_stubs(
            jira_url=url, slack_url=url, github_url=url, clickup_url=url,
        )
        registry = build_e2e_registry(dev_agent)
        _run_dev_handler_sync(
            event={"text": "<@UBOT> [dev] update ticket",
                   "channel": "C003", "thread_ts": "333.444", "ts": "333.444"},
            registry=registry,
        )

        assert len(create_calls) == 0, (
            f"LLM must NOT create new tickets. Calls: {create_calls}"
        )


# ===========================================================================
# E2E-06: ClickUp task updated when ClickUp URL found in thread
# ===========================================================================

@skip_without_llm
class TestClickUpTaskUpdated:
    def test_clickup_task_updated_when_clickup_id_found(
        self,
        mcp_stub: MCPStubServer,
        thread_summary_tool_order: None,
    ) -> None:
        """E2E-06: Thread with ClickUp URL → add_comment called for the task."""
        stub = mcp_stub
        add_comment_calls: list = []

        def _add_comment(args: dict) -> dict:
            add_comment_calls.append(args)
            return {"ok": True, "id": "c-cu-2"}

        stub.register_tool("get_messages", lambda args: {
            "ok": True,
            "messages": [
                {
                    "user": "U1",
                    "text": (
                        "Please implement: "
                        "https://app.clickup.com/t/abc123def456 "
                        "(use async queue pattern)"
                    ),
                    "ts": "1.1",
                },
            ],
        })
        stub.register_tool("get_thread_permalink", lambda args: {
            "ok": True, "permalink": "https://slack.com/archives/C004/p444",
        })
        stub.register_tool("add_comment", _add_comment)
        stub.register_tool("reply_to_thread", lambda args: {"ok": True})
        stub.register_tool("get_task", lambda args: {
            "id": "abc123def456", "name": "Implement feature",
            "status": {"status": "ACCEPTED"},
        })

        url = stub.url
        dev_agent = build_dev_agent_against_stubs(
            jira_url=url, slack_url=url, github_url=url, clickup_url=url,
        )
        registry = build_e2e_registry(dev_agent)
        _run_dev_handler_sync(
            event={"text": "<@UBOT> [dev] update the clickup task",
                   "channel": "C004", "thread_ts": "444.555", "ts": "444.555"},
            registry=registry,
        )

        assert len(add_comment_calls) > 0, (
            "Expected LLM to call add_comment for the ClickUp task found in thread"
        )

