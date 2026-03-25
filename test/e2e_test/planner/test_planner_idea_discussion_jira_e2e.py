"""
E2E tests: Planner Idea-Discussion Lifecycle — JIRA backend variant (E2E-PI-01 through E2E-PI-05).

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
]

from test.e2e_test.common.assertions import (
    assert_no_calls_in_stub_mode,
    assert_stub_calls_count,
    assert_stub_was_called,
)
from test.e2e_test.common.hybrid_mode import (
    get_service_url,
    should_assert_stub_calls,
    should_register_stub_tools,
)
from test.e2e_test.common.test_infrastructure import build_planner_registry
from test.e2e_test.conftest import (
    E2ESettings,
    MCPStubServer,
    build_planner_agent_against_stubs,
    skip_without_llm,
)


def _run_planner(message: str, thread_ts: str, stub: MCPStubServer, registry: Any) -> None:
    from src.slack_app.handlers.planner import planner_handler

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        planner_handler(
            text=message,
            event={"text": message, "channel": "C001", "thread_ts": thread_ts, "ts": thread_ts},
            say=MagicMock(),
            registry=registry,
            executor=executor,
        )
        executor.shutdown(wait=True)
    finally:
        executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# E2E-PI-01: Planner surveys a new idea — no ticket created (JIRA)
# ---------------------------------------------------------------------------


@skip_without_llm
class TestPlannerSurveysNewIdea:
    def test_e2e_pi_01_responds_without_creating_tickets(
        self,
        mcp_stub: MCPStubServer,
        mcp_urls: dict[str, str],
        e2e_settings: E2ESettings,
    ) -> None:
        """E2E-PI-01 (JIRA): Ambiguous idea → Planner responds, no issue created."""
        stub = mcp_stub
        url = get_service_url("jira", e2e_settings, mcp_urls, stub)

        create_issue_calls: list = []
        create_task_calls: list = []

        if should_register_stub_tools(e2e_settings):
            stub.register_tool("reply_to_thread", lambda args: {"ok": True})
            stub.register_tool("send_message", lambda args: {"ok": True})
            stub.register_tool(
                "get_messages",
                lambda args: {
                    "ok": True,
                    "messages": [
                        {
                            "user": "U1",
                            "text": "[planner] I want to build a B2B SaaS for restaurant inventory",
                            "ts": "100.1",
                        },
                    ],
                },
            )
            stub.register_tool(
                "create_issue", lambda args: (create_issue_calls.append(args) or {"key": "PROJ-NEW", "id": "99"})
            )
            stub.register_tool("create_task", lambda args: (create_task_calls.append(args) or {"id": "cu-new"}))

        planner_agent = build_planner_agent_against_stubs(url=url, e2e_settings=e2e_settings)
        registry = build_planner_registry(planner_agent)

        _run_planner(
            message="[planner] I want to build a B2B SaaS tool for restaurant inventory",
            thread_ts="100.1",
            stub=stub,
            registry=registry,
        )

        if should_assert_stub_calls(e2e_settings):
            assert_stub_was_called(e2e_settings, stub, "reply_to_thread") or assert_stub_was_called(
                e2e_settings, stub, "send_message"
            )
            assert_no_calls_in_stub_mode(
                e2e_settings, create_issue_calls, "create_issue", f"Got JIRA issues: {create_issue_calls}"
            )
            assert_no_calls_in_stub_mode(
                e2e_settings, create_task_calls, "create_task", f"Got ClickUp tasks: {create_task_calls}"
            )


# ---------------------------------------------------------------------------
# E2E-PI-02: Planner posts complete survey plan (JIRA)
# ---------------------------------------------------------------------------


@skip_without_llm
class TestPlannerPostsSurveyPlan:
    def test_e2e_pi_02_survey_plan_posted_with_key_dimensions(
        self,
        mcp_stub: MCPStubServer,
        mcp_urls: dict[str, str],
        e2e_settings: E2ESettings,
    ) -> None:
        """E2E-PI-02 (JIRA): After exchanges, Planner posts survey plan covering key dimensions."""
        stub = mcp_stub
        url = get_service_url("jira", e2e_settings, mcp_urls, stub)

        reply_bodies: list = []
        create_issue_calls: list = []

        if should_register_stub_tools(e2e_settings):
            stub.register_tool(
                "reply_to_thread",
                lambda args: (
                    reply_bodies.append(args.get("text", args.get("message", ""))) or {"ok": True, "ts": "102.2"}
                ),
            )
            stub.register_tool("send_message", lambda args: {"ok": True})
            stub.register_tool(
                "get_messages",
                lambda args: {
                    "ok": True,
                    "messages": [
                        {"user": "U1", "text": "[planner] Build restaurant inventory SaaS", "ts": "100.1"},
                        {"user": "UBOT", "text": "What type of restaurants?", "ts": "100.2"},
                        {"user": "U1", "text": "Small to medium. Reduce food waste.", "ts": "100.3"},
                        {"user": "UBOT", "text": "Who is the primary buyer?", "ts": "100.4"},
                        {
                            "user": "U1",
                            "text": "Restaurant owners. Budget conscious. Please give me the full survey plan.",
                            "ts": "100.5",
                        },
                    ],
                },
            )
            stub.register_tool("create_issue", lambda args: (create_issue_calls.append(args) or {"key": "PROJ-ERR"}))
            stub.register_tool("search_issues", lambda args: [])
            stub.register_tool("search_tasks", lambda args: [])

        planner_agent = build_planner_agent_against_stubs(url=url, e2e_settings=e2e_settings)
        registry = build_planner_registry(planner_agent)

        _run_planner(
            message="[planner] Please give me the full survey plan now",
            thread_ts="100.5",
            stub=stub,
            registry=registry,
        )

        if should_assert_stub_calls(e2e_settings):
            assert_stub_was_called(e2e_settings, stub, "reply_to_thread") or assert_stub_was_called(
                e2e_settings, stub, "send_message"
            )

            all_reply_text = " ".join(str(b) for b in reply_bodies).lower()
            if not all_reply_text:
                all_reply_text = " ".join(
                    str(c["arguments"]) for c in stub.all_calls if c["tool"] in ("reply_to_thread", "send_message")
                ).lower()

            dimension_hits = sum(
                [
                    "market" in all_reply_text,
                    "business model" in all_reply_text or "revenue" in all_reply_text,
                    "audience" in all_reply_text or "customer" in all_reply_text,
                    "pain" in all_reply_text or "problem" in all_reply_text,
                    "mvp" in all_reply_text or "feature" in all_reply_text,
                    "budget" in all_reply_text or "cost" in all_reply_text,
                ]
            )
            assert dimension_hits >= 3, (
                f"Expected at least 3 dimensions. Found {dimension_hits}. " f"Text: {all_reply_text[:600]}"
            )
            assert_no_calls_in_stub_mode(e2e_settings, create_issue_calls, "create_issue", f"Got: {create_issue_calls}")


# ---------------------------------------------------------------------------
# E2E-PI-03: Human rejects idea — REJECTED issue, no Dev Lead (JIRA)
# ---------------------------------------------------------------------------


@skip_without_llm
class TestPlannerRejectsIdea:
    def test_e2e_pi_03_rejected_issue_created_no_dev_lead(
        self,
        mcp_stub: MCPStubServer,
        mcp_urls: dict[str, str],
        e2e_settings: E2ESettings,
    ) -> None:
        """E2E-PI-03 (JIRA): Human rejects → REJECTED issue + no Dev Lead mention."""
        stub = mcp_stub
        url = get_service_url("jira", e2e_settings, mcp_urls, stub)

        create_issue_calls: list = []
        send_message_calls: list = []
        accepted_status_writes: list = []

        issue_counter = [0]

        def handle_create_issue(args: dict) -> dict:
            create_issue_calls.append(args)
            issue_counter[0] += 1
            status_str = str(args).upper()
            if "ACCEPTED" in status_str and "REJECTED" not in status_str:
                accepted_status_writes.append(args)
            return {"key": f"PROJ-{issue_counter[0]}", "id": str(issue_counter[0])}

        if should_register_stub_tools(e2e_settings):
            stub.register_tool("create_issue", handle_create_issue)
            stub.register_tool("reply_to_thread", lambda args: {"ok": True})
            stub.register_tool("send_message", lambda args: (send_message_calls.append(args) or {"ok": True}))
            stub.register_tool(
                "get_messages",
                lambda args: {
                    "ok": True,
                    "messages": [
                        {"user": "U1", "text": "[planner] Build restaurant inventory SaaS", "ts": "100.1"},
                        {"user": "UBOT", "text": "### Survey Plan ...", "ts": "100.2"},
                        {"user": "U1", "text": "Actually drop this. Market too competitive.", "ts": "100.3"},
                    ],
                },
            )
            stub.register_tool("transition_issue", lambda args: {"ok": True})
            stub.register_tool("search_issues", lambda args: [])
            stub.register_tool("search_tasks", lambda args: [])
            stub.register_tool("create_task", lambda args: {"id": "cu-rej"})

        planner_agent = build_planner_agent_against_stubs(url=url, e2e_settings=e2e_settings)
        registry = build_planner_registry(planner_agent)

        _run_planner(
            message="[planner] Actually drop this. Market too competitive.",
            thread_ts="100.3",
            stub=stub,
            registry=registry,
        )

        if should_assert_stub_calls(e2e_settings):
            assert_stub_was_called(e2e_settings, stub, "reply_to_thread")
            assert_stub_calls_count(
                e2e_settings, create_issue_calls, min_count=1, message=f"Expected JIRA issue. Got: {create_issue_calls}"
            )
            all_issue_text = " ".join(str(c) for c in create_issue_calls).upper()
            assert "REJECTED" in all_issue_text, f"Expected REJECTED status. Got: {create_issue_calls}"
            dev_lead_in_send = any("dev lead" in str(c).lower() for c in send_message_calls)
            assert not dev_lead_in_send, f"BR-12 violated: {send_message_calls}"
            assert_no_calls_in_stub_mode(
                e2e_settings, accepted_status_writes, "create_issue", f"BR-1 violated: {accepted_status_writes}"
            )


# ---------------------------------------------------------------------------
# E2E-PI-04: Human accepts idea — OPEN issue + Dev Lead hand-off (JIRA)
# ---------------------------------------------------------------------------


@skip_without_llm
class TestPlannerAcceptsIdea:
    def test_e2e_pi_04_open_issue_and_dev_lead_handoff(
        self,
        mcp_stub: MCPStubServer,
        mcp_urls: dict[str, str],
        e2e_settings: E2ESettings,
    ) -> None:
        """E2E-PI-04 (JIRA): Human accepts → OPEN issue + Dev Lead mention."""
        stub = mcp_stub
        url = get_service_url("jira", e2e_settings, mcp_urls, stub)

        create_issue_calls: list = []
        send_message_calls: list = []
        accepted_status_writes: list = []

        issue_counter = [0]

        def handle_create_issue(args: dict) -> dict:
            create_issue_calls.append(args)
            issue_counter[0] += 1
            if "ACCEPTED" in str(args).upper():
                accepted_status_writes.append(args)
            return {"key": f"PROJ-{100 + issue_counter[0]}", "id": str(issue_counter[0])}

        if should_register_stub_tools(e2e_settings):
            stub.register_tool("create_issue", handle_create_issue)
            stub.register_tool("reply_to_thread", lambda args: {"ok": True})
            stub.register_tool("send_message", lambda args: (send_message_calls.append(args) or {"ok": True}))
            stub.register_tool(
                "get_messages",
                lambda args: {
                    "ok": True,
                    "messages": [
                        {"user": "U1", "text": "[planner] Build restaurant inventory SaaS", "ts": "100.1"},
                        {
                            "user": "UBOT",
                            "text": (
                                "### 📋 Idea Survey Plan\n"
                                "**1. Marketing Value** — Large F&B opportunity.\n"
                                "**6. MVP Features** — Inventory tracking.\n"
                                "**8. Budget** — $150K."
                            ),
                            "ts": "100.2",
                        },
                        {"user": "U1", "text": "Great plan! I approve this.", "ts": "100.3"},
                    ],
                },
            )
            stub.register_tool("transition_issue", lambda args: {"ok": True})
            stub.register_tool("search_issues", lambda args: [])
            stub.register_tool("search_tasks", lambda args: [])
            stub.register_tool("create_task", lambda args: {"id": "cu-acc"})
            stub.register_tool("link_issues", lambda args: {"ok": True})

        planner_agent = build_planner_agent_against_stubs(url=url, e2e_settings=e2e_settings)
        registry = build_planner_registry(planner_agent)

        _run_planner(
            message="[planner] Great plan! I approve this.",
            thread_ts="100.3",
            stub=stub,
            registry=registry,
        )

        if should_assert_stub_calls(e2e_settings):
            assert_stub_was_called(e2e_settings, stub, "reply_to_thread")
            assert_stub_calls_count(
                e2e_settings, create_issue_calls, min_count=1, message=f"Expected JIRA issue. Got: {create_issue_calls}"
            )
            all_issue_text = " ".join(str(c) for c in create_issue_calls).upper()
            assert "OPEN" in all_issue_text, f"Expected OPEN status. Got: {create_issue_calls}"
            assert_stub_was_called(e2e_settings, stub, "send_message", "BR-13: Expected send_message for hand-off")
            dev_lead_in_send = any("dev lead" in str(c).lower() for c in send_message_calls)
            assert dev_lead_in_send, f"Expected [dev lead] in send_message. Got: {send_message_calls}"
            assert_no_calls_in_stub_mode(
                e2e_settings, accepted_status_writes, "create_issue", f"BR-1 violated: {accepted_status_writes}"
            )


# ---------------------------------------------------------------------------
# E2E-PI-05: Full Planner → Dev Lead lifecycle (JIRA)
# ---------------------------------------------------------------------------


@skip_without_llm
class TestFullPlannerDevLeadLifecycle:
    def test_e2e_pi_05_full_idea_lifecycle_planner_to_dev_lead(
        self,
        mcp_stub: MCPStubServer,
        mcp_urls: dict[str, str],
        e2e_settings: E2ESettings,
    ) -> None:
        """E2E-PI-05 (JIRA): survey → accept → OPEN issue → [dev lead] hand-off."""
        stub = mcp_stub
        url = get_service_url("jira", e2e_settings, mcp_urls, stub)

        create_issue_calls: list = []
        send_message_calls: list = []
        reply_calls: list = []

        issue_counter = [0]

        def handle_create_issue(args: dict) -> dict:
            create_issue_calls.append(args)
            issue_counter[0] += 1
            return {"key": f"PROJ-{200 + issue_counter[0]}", "id": str(issue_counter[0])}

        if should_register_stub_tools(e2e_settings):
            stub.register_tool("create_issue", handle_create_issue)
            stub.register_tool("reply_to_thread", lambda args: (reply_calls.append(args) or {"ok": True}))
            stub.register_tool("send_message", lambda args: (send_message_calls.append(args) or {"ok": True}))
            stub.register_tool(
                "get_messages",
                lambda args: {
                    "ok": True,
                    "messages": [
                        {"user": "U1", "text": "[planner] AI-powered meal planning app", "ts": "105.1"},
                        {"user": "UBOT", "text": "Who is the primary user?", "ts": "105.2"},
                        {"user": "U1", "text": "Parents with kids aged 5-16.", "ts": "105.3"},
                        {
                            "user": "UBOT",
                            "text": (
                                "### 📋 Idea Survey Plan\n"
                                "**3. Business Model** — Subscription $9.99/month.\n"
                                "**6. MVP Features** — AI meal suggestions.\n"
                                "**8. Budget** — $80K."
                            ),
                            "ts": "105.4",
                        },
                        {"user": "U1", "text": "Great! Approved.", "ts": "105.5"},
                    ],
                },
            )
            stub.register_tool("transition_issue", lambda args: {"ok": True})
            stub.register_tool("search_issues", lambda args: [])
            stub.register_tool("search_tasks", lambda args: [])
            stub.register_tool("create_task", lambda args: {"id": "cu-lifecycle"})
            stub.register_tool("link_issues", lambda args: {"ok": True})
            stub.register_tool(
                "get_issue",
                lambda args: {
                    "key": "PROJ-201",
                    "summary": "AI-powered meal planning app",
                    "status": {"name": "OPEN"},
                },
            )

        planner_agent = build_planner_agent_against_stubs(url=url, e2e_settings=e2e_settings)
        registry = build_planner_registry(planner_agent)

        _run_planner(
            message="[planner] Great! Approved.",
            thread_ts="105.5",
            stub=stub,
            registry=registry,
        )

        if should_assert_stub_calls(e2e_settings):
            assert_stub_calls_count(
                e2e_settings, create_issue_calls, min_count=1, message=f"Expected OPEN issue. Got: {create_issue_calls}"
            )
            all_issue_text = " ".join(str(c) for c in create_issue_calls).upper()
            assert "OPEN" in all_issue_text or "REJECTED" not in all_issue_text
            assert_stub_was_called(e2e_settings, stub, "send_message")
            dev_lead_mention = any("dev lead" in str(c).lower() for c in send_message_calls)
            assert dev_lead_mention, f"Expected [dev lead] in send_message. Got: {send_message_calls}"
            accepted_writes = [c for c in create_issue_calls if "ACCEPTED" in str(c).upper()]
            assert_no_calls_in_stub_mode(
                e2e_settings, accepted_writes, "create_issue", f"BR-1 violated: {accepted_writes}"
            )
            reply_with_dev_lead = any("dev lead" in str(c).lower() for c in reply_calls)
            assert not reply_with_dev_lead, f"BR-13 violated: hand-off via reply_to_thread"
