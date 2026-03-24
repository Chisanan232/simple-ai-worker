"""
E2E tests: Planner Idea-Discussion Lifecycle — ClickUp backend (E2E-PI-01 through E2E-PI-05).

Scenario: Human Boss → Planner AI Agent → Idea Survey / Discussion →
          Accept/Reject → Ticket Creation (ClickUp task) → Dev Lead Hand-off

Test coverage:
- E2E-PI-01: Planner responds with survey questions / draft (no ticket created)
- E2E-PI-02: Planner posts complete 8-dimension survey plan as rich Slack message
- E2E-PI-03: Human rejects idea → REJECTED task created, no Dev Lead mention
- E2E-PI-04: Human accepts idea → OPEN task created + Dev Lead mentioned in new message
- E2E-PI-05: Full planner → Dev Lead lifecycle (compound scenario)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import MagicMock

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

from test.e2e_test.common.e2e_settings import get_e2e_settings
from test.e2e_test.conftest import (
    MCPStubServer,
    build_planner_agent_against_stubs,
    skip_without_llm,
)


def _build_planner_registry(planner_agent: Any) -> Any:
    from src.agents.registry import AgentRegistry

    registry = AgentRegistry()
    registry.register("planner", planner_agent)
    return registry


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
# E2E-PI-01: Planner surveys a new idea — no ticket created
# ---------------------------------------------------------------------------


@skip_without_llm
class TestPlannerSurveysNewIdea:
    def test_e2e_pi_01_responds_without_creating_tickets(
        self,
        mcp_stub: MCPStubServer,
        reply_to_thread_tool_order: None,
    ) -> None:
        """E2E-PI-01 (ClickUp): Ambiguous idea → Planner responds, no task created."""
        stub = mcp_stub
        url = stub.url

        create_task_calls: list = []
        create_issue_calls: list = []

        stub.register_tool("reply_to_thread", lambda args: {"ok": True, "ts": "101.1"})
        stub.register_tool("send_message", lambda args: {"ok": True, "ts": "101.2"})
        stub.register_tool(
            "get_messages",
            lambda args: {
                "ok": True,
                "messages": [
                    {
                        "user": "U1",
                        "text": "[planner] I want to build a B2B SaaS for restaurant inventory management",
                        "ts": "100.1",
                    },
                ],
            },
        )
        stub.register_tool("create_task", lambda args: (create_task_calls.append(args) or {"id": "cu-new"}))
        stub.register_tool("create_issue", lambda args: (create_issue_calls.append(args) or {"key": "PROJ-NEW"}))

        planner_agent = build_planner_agent_against_stubs(url=url, e2e_settings=get_e2e_settings())
        registry = _build_planner_registry(planner_agent)

        _run_planner(
            message="[planner] I want to build a B2B SaaS tool for restaurant inventory management",
            thread_ts="100.1",
            stub=stub,
            registry=registry,
        )

        assert stub.was_called("reply_to_thread") or stub.was_called("send_message"), (
            "Expected Planner to call reply_to_thread or send_message. "
            f"All calls: {[c['tool'] for c in stub.all_calls]}"
        )
        assert len(create_task_calls) == 0, (
            f"Planner must NOT create ClickUp tasks during initial idea discussion. " f"Got: {create_task_calls}"
        )
        assert len(create_issue_calls) == 0, (
            f"Planner must NOT create JIRA issues during initial idea discussion. " f"Got: {create_issue_calls}"
        )


# ---------------------------------------------------------------------------
# E2E-PI-02: Planner posts complete survey plan after discussion
# ---------------------------------------------------------------------------


@skip_without_llm
class TestPlannerPostsSurveyPlan:
    def test_e2e_pi_02_survey_plan_posted_with_key_dimensions(
        self,
        mcp_stub: MCPStubServer,
        reply_to_thread_tool_order: None,
    ) -> None:
        """E2E-PI-02 (ClickUp): After exchanges, Planner posts survey plan covering key dimensions."""
        stub = mcp_stub
        url = stub.url

        reply_bodies: list = []
        create_task_calls: list = []

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
                    {
                        "user": "U1",
                        "text": "[planner] I want to build a B2B SaaS for restaurant inventory",
                        "ts": "100.1",
                    },
                    {"user": "UBOT", "text": "What type of restaurants? And what is the main goal?", "ts": "100.2"},
                    {
                        "user": "U1",
                        "text": "Small to medium restaurants. Main goal is to reduce food waste.",
                        "ts": "100.3",
                    },
                    {"user": "UBOT", "text": "Who would be the primary buyer?", "ts": "100.4"},
                    {
                        "user": "U1",
                        "text": "Restaurant owners and managers. Please give me the full survey plan now.",
                        "ts": "100.5",
                    },
                ],
            },
        )
        stub.register_tool("create_task", lambda args: (create_task_calls.append(args) or {"id": "cu-ERR"}))
        stub.register_tool("search_tasks", lambda args: [])
        stub.register_tool("search_issues", lambda args: [])

        planner_agent = build_planner_agent_against_stubs(url=url, e2e_settings=get_e2e_settings())
        registry = _build_planner_registry(planner_agent)

        _run_planner(
            message="[planner] Please give me the full survey plan now",
            thread_ts="100.5",
            stub=stub,
            registry=registry,
        )

        assert stub.was_called("reply_to_thread") or stub.was_called(
            "send_message"
        ), f"Expected Planner to post survey plan. All calls: {[c['tool'] for c in stub.all_calls]}"

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
                "implementation" in all_reply_text or "tech" in all_reply_text,
                "budget" in all_reply_text or "cost" in all_reply_text,
            ]
        )
        assert dimension_hits >= 3, (
            f"Expected at least 3 key dimensions. Found {dimension_hits}. " f"Text (first 600): {all_reply_text[:600]}"
        )
        assert (
            len(create_task_calls) == 0
        ), f"Planner must NOT create tasks during survey plan posting. Got: {create_task_calls}"


# ---------------------------------------------------------------------------
# E2E-PI-03: Human rejects idea — REJECTED task, no Dev Lead
# ---------------------------------------------------------------------------


@skip_without_llm
class TestPlannerRejectsIdea:
    def test_e2e_pi_03_rejected_task_created_no_dev_lead(
        self,
        mcp_stub: MCPStubServer,
        create_task_only_tool_order: None,
    ) -> None:
        """E2E-PI-03 (ClickUp): Human rejects idea → REJECTED task + no Dev Lead mention."""
        stub = mcp_stub
        url = stub.url

        create_task_calls: list = []
        send_message_calls: list = []
        accepted_status_writes: list = []

        task_counter = [0]

        def handle_create_task(args: dict) -> dict:
            create_task_calls.append(args)
            task_counter[0] += 1
            status_str = str(args).upper()
            if "ACCEPTED" in status_str and "REJECTED" not in status_str:
                accepted_status_writes.append(args)
            return {"id": f"cu-{task_counter[0]}", "url": f"https://clickup.com/t/cu-{task_counter[0]}"}

        stub.register_tool("create_task", handle_create_task)
        stub.register_tool("reply_to_thread", lambda args: {"ok": True, "ts": "103.9"})
        stub.register_tool(
            "send_message", lambda args: (send_message_calls.append(args) or {"ok": True, "ts": "103.10"})
        )
        stub.register_tool(
            "get_messages",
            lambda args: {
                "ok": True,
                "messages": [
                    {"user": "U1", "text": "[planner] I want to build a restaurant inventory SaaS", "ts": "100.1"},
                    {"user": "UBOT", "text": "### 📋 Idea Survey Plan\n**1. Marketing Value** ...", "ts": "100.2"},
                    {
                        "user": "U1",
                        "text": "Actually, let's drop this idea. The market is too competitive.",
                        "ts": "100.3",
                    },
                ],
            },
        )
        stub.register_tool("update_task", lambda args: {"ok": True})
        stub.register_tool("search_tasks", lambda args: [])
        stub.register_tool("search_issues", lambda args: [])
        stub.register_tool("create_issue", lambda args: {"key": "PROJ-ERR"})

        planner_agent = build_planner_agent_against_stubs(url=url, e2e_settings=get_e2e_settings())
        registry = _build_planner_registry(planner_agent)

        _run_planner(
            message=(
                "[planner] Actually, let's drop this idea. " "The market is too competitive and we lack the budget."
            ),
            thread_ts="100.3",
            stub=stub,
            registry=registry,
        )

        assert stub.was_called(
            "reply_to_thread"
        ), f"Expected Planner to post conclusion. All calls: {[c['tool'] for c in stub.all_calls]}"
        assert len(create_task_calls) >= 1, f"Expected ClickUp task for rejected conclusion. Got: {create_task_calls}"
        all_task_text = " ".join(str(c) for c in create_task_calls).upper()
        assert "REJECTED" in all_task_text, f"Expected task to have REJECTED status. calls: {create_task_calls}"

        dev_lead_in_send = any(
            "dev lead" in str(c).lower() or "[dev lead]" in str(c).lower() for c in send_message_calls
        )
        assert not dev_lead_in_send, f"BR-12 violation: Dev Lead mentioned on reject path. calls: {send_message_calls}"
        assert len(accepted_status_writes) == 0, f"BR-1 violated: {accepted_status_writes}"


# ---------------------------------------------------------------------------
# E2E-PI-04: Human accepts idea — OPEN task + Dev Lead hand-off
# ---------------------------------------------------------------------------


@skip_without_llm
class TestPlannerAcceptsIdea:
    def test_e2e_pi_04_open_task_and_dev_lead_handoff(
        self,
        mcp_stub: MCPStubServer,
        create_task_and_notify_tool_order: None,
    ) -> None:
        """E2E-PI-04 (ClickUp): Human accepts idea → OPEN task + Dev Lead mentioned."""
        stub = mcp_stub
        url = stub.url

        create_task_calls: list = []
        send_message_calls: list = []
        accepted_status_writes: list = []

        task_counter = [0]

        def handle_create_task(args: dict) -> dict:
            create_task_calls.append(args)
            task_counter[0] += 1
            if "ACCEPTED" in str(args).upper():
                accepted_status_writes.append(args)
            return {
                "id": f"cu-{100 + task_counter[0]}",
                "url": f"https://clickup.com/t/cu-{100 + task_counter[0]}",
            }

        stub.register_tool("create_task", handle_create_task)
        stub.register_tool("reply_to_thread", lambda args: {"ok": True, "ts": "104.9"})
        stub.register_tool(
            "send_message", lambda args: (send_message_calls.append(args) or {"ok": True, "ts": "104.10"})
        )
        stub.register_tool(
            "get_messages",
            lambda args: {
                "ok": True,
                "messages": [
                    {"user": "U1", "text": "[planner] I want to build a restaurant inventory SaaS", "ts": "100.1"},
                    {
                        "user": "UBOT",
                        "text": (
                            "### 📋 Idea Survey Plan\n"
                            "**1. Marketing Value** — Large opportunity in F&B sector.\n"
                            "**6. MVP Features** — Inventory tracking, alerts, reporting.\n"
                            "**8. Budget Estimation** — $150K for 6-month MVP."
                        ),
                        "ts": "100.2",
                    },
                    {"user": "U1", "text": "Great plan! Let's proceed with the MVP. I approve this.", "ts": "100.3"},
                ],
            },
        )
        stub.register_tool("update_task", lambda args: {"ok": True})
        stub.register_tool("search_tasks", lambda args: [])
        stub.register_tool("search_issues", lambda args: [])
        stub.register_tool("create_issue", lambda args: {"key": "PROJ-NEW"})
        stub.register_tool("link_issues", lambda args: {"ok": True})

        planner_agent = build_planner_agent_against_stubs(url=url, e2e_settings=get_e2e_settings())
        registry = _build_planner_registry(planner_agent)

        _run_planner(
            message="[planner] Great plan! Let's proceed with the MVP. I approve this.",
            thread_ts="100.3",
            stub=stub,
            registry=registry,
        )

        assert stub.was_called(
            "reply_to_thread"
        ), f"Expected conclusion reply. All calls: {[c['tool'] for c in stub.all_calls]}"
        assert len(create_task_calls) >= 1, f"Expected ClickUp task for accepted conclusion. Got: {create_task_calls}"
        all_task_text = " ".join(str(c) for c in create_task_calls).upper()
        assert "OPEN" in all_task_text, f"Expected OPEN task on acceptance. calls: {create_task_calls}"
        assert stub.was_called("send_message"), (
            "BR-13: Expected Planner to call send_message for Dev Lead hand-off. "
            f"All calls: {[c['tool'] for c in stub.all_calls]}"
        )
        dev_lead_in_send = any(
            "dev lead" in str(c).lower() or "[dev lead]" in str(c).lower() for c in send_message_calls
        )
        assert dev_lead_in_send, f"Expected [dev lead] mention in send_message. calls: {send_message_calls}"
        assert len(accepted_status_writes) == 0, f"BR-1 violated: {accepted_status_writes}"


# ---------------------------------------------------------------------------
# E2E-PI-05: Full Planner → Dev Lead lifecycle (compound scenario)
# ---------------------------------------------------------------------------


@skip_without_llm
class TestFullPlannerDevLeadLifecycle:
    def test_e2e_pi_05_full_idea_lifecycle_planner_to_dev_lead(
        self,
        mcp_stub: MCPStubServer,
        create_task_and_notify_tool_order: None,
    ) -> None:
        """E2E-PI-05 (ClickUp): survey → accept → OPEN task → [dev lead] hand-off."""
        stub = mcp_stub
        url = stub.url

        create_task_calls: list = []
        send_message_calls: list = []
        reply_calls: list = []

        task_counter = [0]

        def handle_create_task(args: dict) -> dict:
            create_task_calls.append(args)
            task_counter[0] += 1
            return {
                "id": f"cu-{200 + task_counter[0]}",
                "url": f"https://clickup.com/t/cu-{200 + task_counter[0]}",
            }

        stub.register_tool("create_task", handle_create_task)
        stub.register_tool("reply_to_thread", lambda args: (reply_calls.append(args) or {"ok": True, "ts": "105.99"}))
        stub.register_tool(
            "send_message", lambda args: (send_message_calls.append(args) or {"ok": True, "ts": "105.100"})
        )
        stub.register_tool(
            "get_messages",
            lambda args: {
                "ok": True,
                "messages": [
                    {"user": "U1", "text": "[planner] I want to build an AI-powered meal planning app", "ts": "105.1"},
                    {"user": "UBOT", "text": "Who is the primary user?", "ts": "105.2"},
                    {"user": "U1", "text": "Parents with kids aged 5-16.", "ts": "105.3"},
                    {
                        "user": "UBOT",
                        "text": (
                            "### 📋 Idea Survey Plan\n"
                            "**1. Marketing Value** — Significant market.\n"
                            "**3. Business Model** — Subscription $9.99/month.\n"
                            "**6. MVP Features** — AI meal suggestions, grocery lists.\n"
                            "**8. Budget** — $80K for MVP."
                        ),
                        "ts": "105.4",
                    },
                    {"user": "U1", "text": "This looks great! Go ahead, let's do it. Approved.", "ts": "105.5"},
                ],
            },
        )
        stub.register_tool("update_task", lambda args: {"ok": True})
        stub.register_tool("search_tasks", lambda args: [])
        stub.register_tool("search_issues", lambda args: [])
        stub.register_tool("create_issue", lambda args: {"key": "PROJ-201"})
        stub.register_tool("link_issues", lambda args: {"ok": True})
        stub.register_tool(
            "get_issue",
            lambda args: {
                "key": "PROJ-201",
                "summary": "AI-powered meal planning app",
                "status": {"name": "OPEN"},
                "description": "Full survey plan and discussion details.",
            },
        )

        planner_agent = build_planner_agent_against_stubs(url=url, e2e_settings=get_e2e_settings())
        registry = _build_planner_registry(planner_agent)

        _run_planner(
            message="[planner] This looks great! Go ahead, let's do it. Approved.",
            thread_ts="105.5",
            stub=stub,
            registry=registry,
        )

        assert len(create_task_calls) >= 1, f"Expected OPEN task created on acceptance. Got: {create_task_calls}"
        all_task_text = " ".join(str(c) for c in create_task_calls).upper()
        assert (
            "OPEN" in all_task_text or "REJECTED" not in all_task_text
        ), f"Expected OPEN (not REJECTED) task. Got: {create_task_calls}"
        assert stub.was_called("send_message"), (
            "Expected send_message for Dev Lead hand-off. " f"All calls: {[c['tool'] for c in stub.all_calls]}"
        )
        dev_lead_mention = any(
            "dev lead" in str(c).lower() or "[dev lead]" in str(c).lower() for c in send_message_calls
        )
        assert dev_lead_mention, f"Expected [dev lead] in send_message. calls: {send_message_calls}"
        accepted_writes = [c for c in create_task_calls if "ACCEPTED" in str(c).upper()]
        assert len(accepted_writes) == 0, f"BR-1 violated: {accepted_writes}"

        reply_with_dev_lead = any("dev lead" in str(c).lower() or "[dev lead]" in str(c).lower() for c in reply_calls)
        assert (
            not reply_with_dev_lead
        ), f"BR-13 violation: Dev Lead hand-off sent via reply_to_thread. calls: {reply_calls}"
