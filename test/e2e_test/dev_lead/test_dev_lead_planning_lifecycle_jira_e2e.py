"""
E2E tests: Dev Lead Planning Lifecycle — JIRA backend variant (E2E-DL-01 through E2E-DL-06).

All tests in this module are currently marked skip because the JIRA MCP
server tooling has not yet been configured for the project.

To enable: remove ``pytest.mark.skip`` from ``pytestmark`` below once
``E2E_ATLASSIAN_URL``, ``E2E_ATLASSIAN_EMAIL``, and ``E2E_MCP_JIRA_TOKEN``
are configured in ``test/e2e_test/.env.e2e``.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import MagicMock

import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
]

from test.e2e_test.conftest import (
    E2E_WORKFLOW_CONFIG,
    E2ESettings,
    MCPStubServer,
    build_dev_agent_against_stubs,
    build_dev_lead_agent_against_stubs,
    build_e2e_registry,
    skip_without_llm,
)

from src.ticket.models import TicketComment, TicketRecord
from src.ticket.workflow import WorkflowConfig
from test.e2e_test.common.hybrid_mode import get_service_url, should_register_stub_tools, should_assert_stub_calls
from test.e2e_test.common.test_infrastructure import make_settings, make_stub_tracker_registry_planning, build_dev_lead_registry
from test.e2e_test.common.assertions import assert_stub_was_called, assert_stub_calls_count, assert_no_calls_in_stub_mode
from test.e2e_test.common.state_management import reset_planning_state

E2E_PLANNING_WORKFLOW_CONFIG = {
    **E2E_WORKFLOW_CONFIG,
    "open_for_dev": {"status_value": "OPEN", "human_only": False},
    "in_planning": {"status_value": "IN PLANNING", "human_only": True},
}


# ---------------------------------------------------------------------------
# E2E-DL-01: Dev Lead posts clarifying questions (JIRA)
# ---------------------------------------------------------------------------


@skip_without_llm
class TestDevLeadFeasibilityAssessment:
    def test_e2e_dl_01_asks_clarifying_questions_not_creates_tickets(
        self,
        mcp_stub: MCPStubServer,
        mcp_urls: dict[str, str],
        e2e_settings: E2ESettings,
    ) -> None:
        """E2E-DL-01 (JIRA): Ambiguous requirement → Dev Lead asks questions, no issue created."""
        from src.slack_app.handlers.dev_lead import dev_lead_handler

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
                    "messages": [{"user": "U1", "text": "Let's add notifications", "ts": "1.0"}],
                },
            )
            stub.register_tool(
                "create_issue", lambda args: (create_issue_calls.append(args) or {"key": "PROJ-NEW", "id": "99"})
            )
            stub.register_tool("create_task", lambda args: (create_task_calls.append(args) or {"id": "cu-new"}))

        dev_lead_agent = build_dev_lead_agent_against_stubs(url=url, e2e_settings=e2e_settings)
        registry = build_dev_lead_registry(dev_lead_agent)

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            dev_lead_handler(
                text="[dev lead] We want to add a real-time notification system",
                event={
                    "text": "[dev lead] We want to add a real-time notification system",
                    "channel": "C001",
                    "thread_ts": "100.1",
                    "ts": "100.1",
                },
                say=MagicMock(),
                registry=registry,
                executor=executor,
            )
            executor.shutdown(wait=True)
        finally:
            executor.shutdown(wait=False)

        # Stub-specific assertions (only in stub mode with real LLM)
        if not e2e_settings.USE_TESTCONTAINERS and not e2e_settings.USE_FAKE_LLM:
            assert stub.was_called("reply_to_thread")
            assert len(create_issue_calls) == 0, f"Got: {create_issue_calls}"
            assert len(create_task_calls) == 0, f"Got: {create_task_calls}"


# ---------------------------------------------------------------------------
# E2E-DL-02: Dev Lead fetches existing JIRA story
# ---------------------------------------------------------------------------


@skip_without_llm
class TestDevLeadFetchesExistingStory:
    def test_e2e_dl_02_fetches_story_ticket_when_id_present(
        self,
        mcp_stub: MCPStubServer,
        mcp_urls: dict[str, str],
        e2e_settings: E2ESettings,
    ) -> None:
        """E2E-DL-02 (JIRA): Message contains PROJ-50 → Dev Lead calls get_issue."""
        from src.slack_app.handlers.dev_lead import dev_lead_handler

        stub = mcp_stub
        url = get_service_url("jira", e2e_settings, mcp_urls, stub)

        get_issue_calls: list = []

        if should_register_stub_tools(e2e_settings):
            stub.register_tool(
                "get_issue",
                lambda args: (
                    get_issue_calls.append(args)
                    or {
                        "key": "PROJ-50",
                        "summary": "Add notification system",
                        "status": {"name": "To Do"},
                        "description": "We need a real-time notification system for user alerts.",
                    }
                ),
            )
            stub.register_tool("reply_to_thread", lambda args: {"ok": True})
            stub.register_tool("search_issues", lambda args: [])

        dev_lead_agent = build_dev_lead_agent_against_stubs(url=url, e2e_settings=e2e_settings)
        registry = build_dev_lead_registry(dev_lead_agent)

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            dev_lead_handler(
                text="[dev lead] PROJ-50 assess feasibility and provide breakdown plan",
                event={
                    "text": "[dev lead] PROJ-50 assess feasibility",
                    "channel": "C001",
                    "thread_ts": "200.1",
                    "ts": "200.1",
                },
                say=MagicMock(),
                registry=registry,
                executor=executor,
            )
            executor.shutdown(wait=True)
        finally:
            executor.shutdown(wait=False)

        if should_assert_stub_calls(e2e_settings):
            assert_stub_calls_count(e2e_settings, get_issue_calls, min_count=1, message=f"Expected Dev Lead to call get_issue. All calls: {stub.all_calls}")
            fetched_keys = [c.get("issue_key", c.get("key", "")) for c in get_issue_calls]
            assert any("PROJ-50" in str(k) for k in fetched_keys)


# ---------------------------------------------------------------------------
# E2E-DL-03: Dev Lead creates sub-tasks with dependency links (JIRA)
# ---------------------------------------------------------------------------


@skip_without_llm
class TestDevLeadBreakdown:
    def test_e2e_dl_03_creates_subtasks_with_dependency_links(
        self,
        mcp_stub: MCPStubServer,
        mcp_urls: dict[str, str],
        e2e_settings: E2ESettings,
    ) -> None:
        """E2E-DL-03 (JIRA): Explicit breakdown → sub-tasks created + parent notified."""
        from src.slack_app.handlers.dev_lead import dev_lead_handler

        stub = mcp_stub
        url = get_service_url("jira", e2e_settings, mcp_urls, stub)

        create_issue_calls: list = []
        add_comment_calls: list = []
        accepted_write_calls: list = []
        sub_task_counter = [0]

        def handle_create_issue(args: dict) -> dict:
            create_issue_calls.append(args)
            sub_task_counter[0] += 1
            status = str(args.get("status", args.get("fields", {}))).upper()
            if "ACCEPTED" in status:
                accepted_write_calls.append(args)
            return {"key": f"PROJ-10{sub_task_counter[0]}", "id": str(sub_task_counter[0])}

        if should_register_stub_tools(e2e_settings):
            stub.register_tool(
                "get_issue",
                lambda args: {
                    "key": "PROJ-50",
                    "summary": "Add notification system",
                    "status": {"name": "To Do"},
                    "description": (
                        "WebSocket + Redis pub/sub. " "Two components: backend service and frontend subscriber."
                    ),
                },
            )
            stub.register_tool("create_issue", handle_create_issue)
            stub.register_tool("update_issue", lambda args: {"ok": True})
            stub.register_tool("link_issues", lambda args: {"ok": True})
            stub.register_tool("transition_issue", lambda args: {"ok": True})
            stub.register_tool("add_comment", lambda args: (add_comment_calls.append(args) or {"id": "c1"}))
            stub.register_tool("reply_to_thread", lambda args: {"ok": True})
            stub.register_tool("search_issues", lambda args: [])
            stub.register_tool("create_task", lambda args: {"id": "cu-new"})
            stub.register_tool("update_task", lambda args: {"ok": True})

        dev_lead_agent = build_dev_lead_agent_against_stubs(url=url, e2e_settings=e2e_settings)
        registry = build_dev_lead_registry(dev_lead_agent)

        message = (
            "[dev lead] All questions answered — please break down PROJ-50 into sub-tasks. "
            "Architecture: WebSocket backend + Redis pub/sub + frontend subscriber. "
            "Create the implementation tasks with dependencies."
        )

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            dev_lead_handler(
                text=message,
                event={"text": message, "channel": "C001", "thread_ts": "300.1", "ts": "300.1"},
                say=MagicMock(),
                registry=registry,
                executor=executor,
            )
            executor.shutdown(wait=True)
        finally:
            executor.shutdown(wait=False)

        if should_assert_stub_calls(e2e_settings):
            assert_stub_calls_count(e2e_settings, create_issue_calls, min_count=2, message=f"Expected at least 2 sub-tasks. Got: {len(create_issue_calls)}")
            assert_stub_calls_count(e2e_settings, add_comment_calls, min_count=1, message="Expected Dev Lead to add comment on parent ticket PROJ-50")
            assert_stub_was_called(e2e_settings, stub, "reply_to_thread")
            assert_no_calls_in_stub_mode(e2e_settings, accepted_write_calls, "create_issue", f"BR-1 violated: {accepted_write_calls}")


# ---------------------------------------------------------------------------
# E2E-DL-04: Dev Agent generates initial development plan (JIRA)
# ---------------------------------------------------------------------------


@skip_without_llm
class TestDevAgentInitialPlan:
    def test_e2e_dl_04_generates_plan_comment_for_open_issue(
        self,
        mcp_stub: MCPStubServer,
        mcp_urls: dict[str, str],
        e2e_settings: E2ESettings,
    ) -> None:
        """E2E-DL-04 (JIRA): OPEN issue → plan_and_notify_job → plan posted as comment."""
        import src.scheduler.jobs.plan_and_notify as pn_mod
        from src.scheduler.jobs.plan_and_notify import plan_and_notify_job

        stub = mcp_stub
        url = get_service_url("jira", e2e_settings, mcp_urls, stub)

        add_comment_calls: list = []
        transition_calls: list = []

        if should_register_stub_tools(e2e_settings):
            stub.register_tool(
                "get_issue",
                lambda args: {
                    "key": "PROJ-20",
                    "fields": {
                        "summary": "Implement OAuth2 login",
                        "status": {"name": "OPEN"},
                        "description": "Add OAuth2 authentication with Google SSO support.",
                    },
                },
            )
            stub.register_tool("add_comment", lambda args: (add_comment_calls.append(args) or {"id": "c-plan"}))
            stub.register_tool("transition_issue", lambda args: (transition_calls.append(args) or {"ok": True}))
            stub.register_tool(
                "search_issues",
                lambda args: [
                    {"key": "PROJ-20", "fields": {"summary": "OAuth2", "status": {"name": "OPEN"}}},
                ],
            )
            stub.register_tool("search_tasks", lambda args: [])
            stub.register_tool("reply_to_thread", lambda args: {"ok": True})
            stub.register_tool("send_message", lambda args: {"ok": True})

        workflow = WorkflowConfig(E2E_PLANNING_WORKFLOW_CONFIG)
        dev_agent = build_dev_agent_against_stubs(
            jira_url=url,
            slack_url=url,
            github_url=url,
            clickup_url=url,
            e2e_settings=e2e_settings,
        )
        registry = build_e2e_registry(dev_agent)

        reset_planning_state(pn_mod)

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            plan_and_notify_job(
                registry=registry,
                settings=make_settings(),
                executor=executor,
                workflow=workflow,
                tracker_registry=make_stub_tracker_registry_planning(
                    open_tickets=[
                        TicketRecord(
                            id="PROJ-20", source="jira", title="Implement OAuth2 login", url="", raw_status="OPEN"
                        ),
                    ],
                ),
            )
            executor.shutdown(wait=True)
        finally:
            executor.shutdown(wait=False)
            reset_planning_state(pn_mod)

        if should_assert_stub_calls(e2e_settings):
            assert_stub_calls_count(e2e_settings, add_comment_calls, min_count=1, message="Expected Dev Agent to post development plan. " f"All calls: {[c['tool'] for c in stub.all_calls]}")
            plan_bodies = [c.get("comment", c.get("body", c.get("text", ""))) for c in add_comment_calls]
            has_plan = any(
                "plan" in body.lower() or "## " in body or "development" in body.lower() for body in plan_bodies
            )
            assert has_plan, f"Expected plan. Bodies: {plan_bodies}"
            assert not any("ACCEPTED" in str(t).upper() for t in transition_calls), "BR-1 violated"
            assert not any("IN PROGRESS" in str(t).upper() for t in transition_calls), "BR-8 violated"


# ---------------------------------------------------------------------------
# E2E-DL-05: Dev Agent revises plan based on human comments (JIRA)
# ---------------------------------------------------------------------------


@skip_without_llm
class TestDevAgentPlanRevision:
    def test_e2e_dl_05_revises_plan_based_on_human_comments(
        self,
        mcp_stub: MCPStubServer,
        mcp_urls: dict[str, str],
        e2e_settings: E2ESettings,
    ) -> None:
        """E2E-DL-05 (JIRA): IN PLANNING issue + human comment → revised plan posted."""
        import src.scheduler.jobs.plan_and_notify as pn_mod
        from src.scheduler.jobs.plan_and_notify import plan_and_notify_job

        stub = mcp_stub
        url = get_service_url("jira", e2e_settings, mcp_urls, stub)

        add_comment_calls: list = []
        transition_calls: list = []

        if should_register_stub_tools(e2e_settings):
            stub.register_tool(
                "get_issue",
                lambda args: {
                    "key": "PROJ-21",
                    "fields": {
                        "summary": "Implement database migration tool",
                        "status": {"name": "IN PLANNING"},
                        "description": "Build a tool to run database migrations safely.",
                    },
                },
            )
            stub.register_tool("add_comment", lambda args: (add_comment_calls.append(args) or {"id": "c-rev"}))
            stub.register_tool("transition_issue", lambda args: (transition_calls.append(args) or {"ok": True}))
            stub.register_tool("reply_to_thread", lambda args: {"ok": True})
            stub.register_tool("send_message", lambda args: {"ok": True})
            stub.register_tool("search_issues", lambda args: [])
            stub.register_tool("search_tasks", lambda args: [])

        workflow = WorkflowConfig(E2E_PLANNING_WORKFLOW_CONFIG)
        dev_agent = build_dev_agent_against_stubs(
            jira_url=url,
            slack_url=url,
            github_url=url,
            clickup_url=url,
            e2e_settings=e2e_settings,
        )
        registry = build_e2e_registry(dev_agent)

        human_comment = TicketComment(
            id="c-human-1",
            author="alice",
            body="Please clarify how you will handle rollback if the migration fails.",
            created_at=time.time() - 60,
            source="jira",
        )

        reset_planning_state(pn_mod)

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            plan_and_notify_job(
                registry=registry,
                settings=make_settings(),
                executor=executor,
                workflow=workflow,
                tracker_registry=make_stub_tracker_registry_planning(
                    in_planning_tickets=[
                        TicketRecord(
                            id="PROJ-21", source="jira", title="DB migration tool", url="", raw_status="IN PLANNING"
                        ),
                    ],
                    comments_by_ticket={"PROJ-21": [human_comment]},
                ),
            )
            executor.shutdown(wait=True)
        finally:
            executor.shutdown(wait=False)
            reset_planning_state(pn_mod)

        if should_assert_stub_calls(e2e_settings):
            assert_stub_calls_count(e2e_settings, add_comment_calls, min_count=1, message=f"Expected revised plan. All calls: {[c['tool'] for c in stub.all_calls]}")
            plan_bodies = [c.get("comment", c.get("body", c.get("text", ""))) for c in add_comment_calls]
            has_rollback_mention = any(
                "rollback" in body.lower() or "migration" in body.lower() or "revision" in body.lower()
                for body in plan_bodies
            )
            assert has_rollback_mention, f"Expected rollback mention. Bodies: {plan_bodies}"
            bad_transitions = [
                t for t in transition_calls if "ACCEPTED" in str(t).upper() or "IN PROGRESS" in str(t).upper()
            ]
            assert len(bad_transitions) == 0, f"BR violation: {transition_calls}"


# ---------------------------------------------------------------------------
# E2E-DL-06: Full planning lifecycle (JIRA — placeholder, skipped at module level)
# ---------------------------------------------------------------------------


@skip_without_llm
class TestFullPlanningLifecycle:
    def test_e2e_dl_06_full_planning_lifecycle(
        self,
        mcp_stub: MCPStubServer,
    ) -> None:
        """E2E-DL-06 (JIRA): Dev Lead breakdown → Dev Agent plan → no BR violations."""
        # This test is skipped at module level via pytestmark.
        # Body is intentionally identical to ClickUp variant with JIRA tool names.
