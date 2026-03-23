"""
E2E tests: Dev Agent Slack thread summary — JIRA backend variant (E2E-01 through E2E-06).

All tests in this module are currently marked skip because the JIRA MCP
server tooling has not yet been configured for the project.

To enable: remove ``pytest.mark.skip`` from ``pytestmark`` below once
``E2E_ATLASSIAN_URL``, ``E2E_ATLASSIAN_EMAIL``, and ``E2E_MCP_JIRA_TOKEN``
are configured in ``test/e2e_test/.env.e2e``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import MagicMock

import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.skip(reason=(
        "JIRA tooling not yet configured — "
        "will be enabled in a future iteration"
    )),
]

from test.e2e_test.conftest import (
    MCPStubServer,
    build_dev_agent_against_stubs,
    build_e2e_registry,
    skip_without_llm,
)
from test.e2e_test.common.e2e_settings import get_e2e_settings


def _run_dev_handler_sync(event: dict, registry: Any) -> None:
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
# E2E-01: Dev reads thread and updates JIRA ticket
# ===========================================================================

@skip_without_llm
class TestDevReadsThreadAndUpdatesTicket:
    def test_dev_reads_thread_and_updates_ticket(
        self,
        mcp_stub: MCPStubServer,
        thread_summary_tool_order: None,
    ) -> None:
        """E2E-01 (JIRA): [dev] in thread → LLM reads messages → posts jira/add_comment."""
        stub = mcp_stub
        stub.register_tool("get_messages", lambda args: {
            "ok": True,
            "messages": [
                {"user": "U1", "text": "We agreed: implement OAuth2 for PROJ-55.", "ts": "1.1"},
                {"user": "U2", "text": "Use Google SSO. Ticket: PROJ-55.", "ts": "1.2"},
            ],
        })
        stub.register_tool("get_thread_permalink", lambda args: {
            "ok": True, "permalink": "https://slack.com/archives/C001/p111222",
        })
        stub.register_tool("add_comment", lambda args: {
            "id": "c-jira-1", "body": "ok", "issue": {"key": "PROJ-55"},
        })
        stub.register_tool("reply_to_thread", lambda args: {"ok": True})
        stub.register_tool("get_issue", lambda args: {
            "key": "PROJ-55", "fields": {"summary": "Implement OAuth2",
                                          "status": {"name": "ACCEPTED"}},
        })

        url = stub.url
        dev_agent = build_dev_agent_against_stubs(
            jira_url=url, slack_url=url, github_url=url, clickup_url=url,
            e2e_settings=get_e2e_settings(),
        )
        registry = build_e2e_registry(dev_agent)
        _run_dev_handler_sync(
            event={"text": "<@UBOT> [dev] please update the ticket",
                   "channel": "C001", "thread_ts": "111.222", "ts": "111.222"},
            registry=registry,
        )

        assert stub.was_called("get_messages")
        assert stub.was_called("add_comment")


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
        """E2E-02 (JIRA): Thread with no ticket ID → LLM asks supervisor, never calls add_comment."""
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
            e2e_settings=get_e2e_settings(),
        )
        registry = build_e2e_registry(dev_agent)
        _run_dev_handler_sync(
            event={"text": "<@UBOT> [dev] read this thread",
                   "channel": "C002", "thread_ts": "222.333", "ts": "222.333"},
            registry=registry,
        )

        assert len(stub.calls_to("add_comment")) == 0
        assert stub.was_called("reply_to_thread")


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
        """E2E-03 (JIRA): JIRA comment body must contain the Slack thread permalink."""
        stub = mcp_stub
        channel_id = "C123"
        expected_permalink = f"https://slack.com/archives/{channel_id}/p111222"
        comment_bodies: list[str] = []

        def _add_comment(args: dict) -> dict:
            comment_bodies.append(str(args))
            return {"id": "c-jira-1", "issue": {"key": "PROJ-42"}}

        stub.register_tool("get_messages", lambda args: {
            "ok": True,
            "messages": [
                {"user": "U1", "text": "Implement feature PROJ-42.", "ts": "1.1"},
                {"user": "U2", "text": "Agreed, ticket PROJ-42.", "ts": "1.2"},
            ],
        })
        stub.register_tool("get_thread_permalink", lambda args: {
            "ok": True, "permalink": expected_permalink,
        })
        stub.register_tool("add_comment", _add_comment)
        stub.register_tool("reply_to_thread", lambda args: {"ok": True})
        stub.register_tool("get_issue", lambda args: {
            "key": "PROJ-42", "fields": {"summary": "Implement feature",
                                          "status": {"name": "ACCEPTED"}},
        })

        url = stub.url
        dev_agent = build_dev_agent_against_stubs(
            jira_url=url, slack_url=url, github_url=url, clickup_url=url,
            e2e_settings=get_e2e_settings(),
        )
        registry = build_e2e_registry(dev_agent)
        _run_dev_handler_sync(
            event={"text": "<@UBOT> [dev] update ticket with this discussion",
                   "channel": channel_id, "thread_ts": "111.222", "ts": "111.222"},
            registry=registry,
        )

        assert len(comment_bodies) > 0
        assert "slack.com/archives" in " ".join(comment_bodies)


# ===========================================================================
# E2E-05: No new ticket created (guardrail)
# ===========================================================================

@skip_without_llm
class TestNoNewTicketCreated:
    def test_no_new_ticket_created(
        self,
        mcp_stub: MCPStubServer,
    ) -> None:
        """E2E-05 (JIRA): LLM must never call create_issue or create_task."""
        stub = mcp_stub
        create_calls: list = []

        def _create(args: dict) -> dict:
            create_calls.append(args)
            return {"key": "PROJ-NEW", "id": "x"}

        stub.register_tool("get_messages", lambda args: {
            "ok": True,
            "messages": [
                {"user": "U1", "text": "Discussed PROJ-33: use async processing.", "ts": "1.1"},
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
            e2e_settings=get_e2e_settings(),
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
# E2E-06: JIRA issue updated when JIRA key found in thread
# ===========================================================================

@skip_without_llm
class TestJIRAIssueUpdated:
    def test_jira_issue_updated_when_jira_key_found(
        self,
        mcp_stub: MCPStubServer,
        thread_summary_tool_order: None,
    ) -> None:
        """E2E-06 (JIRA): Thread with JIRA key → add_comment called for the issue."""
        stub = mcp_stub
        add_comment_calls: list = []

        def _add_comment(args: dict) -> dict:
            add_comment_calls.append(args)
            return {"id": "c-jira-2", "body": "ok"}

        stub.register_tool("get_messages", lambda args: {
            "ok": True,
            "messages": [
                {
                    "user": "U1",
                    "text": "Please implement PROJ-55 (use async queue pattern)",
                    "ts": "1.1",
                },
            ],
        })
        stub.register_tool("get_thread_permalink", lambda args: {
            "ok": True, "permalink": "https://slack.com/archives/C004/p444",
        })
        stub.register_tool("add_comment", _add_comment)
        stub.register_tool("reply_to_thread", lambda args: {"ok": True})
        stub.register_tool("get_issue", lambda args: {
            "key": "PROJ-55",
            "fields": {"summary": "Implement feature", "status": {"name": "ACCEPTED"}},
        })

        url = stub.url
        dev_agent = build_dev_agent_against_stubs(
            jira_url=url, slack_url=url, github_url=url, clickup_url=url,
            e2e_settings=get_e2e_settings(),
        )
        registry = build_e2e_registry(dev_agent)
        _run_dev_handler_sync(
            event={"text": "<@UBOT> [dev] update the jira issue",
                   "channel": "C004", "thread_ts": "444.555", "ts": "444.555"},
            registry=registry,
        )

        assert len(add_comment_calls) > 0, (
            "Expected LLM to call add_comment for the JIRA issue found in thread"
        )

