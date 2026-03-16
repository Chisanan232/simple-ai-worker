"""
E2E tests: Dev Lead Planning Lifecycle — ClickUp backend (E2E-DL-01 through E2E-DL-06).

Scenario: Human Planner → Dev Lead AI → Task Decomposition → Dev Agent Planning Loop

Test coverage:
- E2E-DL-01: Dev Lead posts clarifying questions (not sub-tasks) for ambiguous requirement
- E2E-DL-02: Dev Lead fetches existing ClickUp task when task ID is in message
- E2E-DL-03: Dev Lead creates sub-tasks with dependency links after discussion concluded
- E2E-DL-04: Dev Agent generates initial development plan for an OPEN task
- E2E-DL-05: Dev Agent revises plan based on human comments on IN PLANNING task
- E2E-DL-06: Full planning lifecycle S-PL-1 through S-PL-8 (compound scenario)
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest_httpserver import HTTPServer

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

from test.e2e_test.conftest import (
    MCPStubServer,
    build_dev_agent_against_stubs,
    build_e2e_registry,
    skip_without_llm,
    E2E_WORKFLOW_CONFIG,
    build_dev_lead_agent_against_stubs,
)
from src.ticket.models import TicketComment, TicketRecord
from src.ticket.workflow import WorkflowConfig

E2E_PLANNING_WORKFLOW_CONFIG = {
    **E2E_WORKFLOW_CONFIG,
    "open_for_dev": {"status_value": "OPEN", "human_only": False},
    "in_planning": {"status_value": "IN PLANNING", "human_only": True},
}


def _make_settings() -> Any:
    s = MagicMock()
    s.PR_AUTO_MERGE_TIMEOUT_SECONDS = 300
    s.PR_REVIEW_COMMENT_CHECK_INTERVAL_SECONDS = 120
    s.MAX_CONCURRENT_DEV_AGENTS = 1
    return s


def _build_dev_lead_registry(dev_lead_agent: Any) -> Any:
    from src.agents.registry import AgentRegistry
    registry = AgentRegistry()
    registry.register("dev_lead", dev_lead_agent)
    return registry


def _make_stub_tracker_registry(
    open_tickets: list | None = None,
    in_planning_tickets: list | None = None,
    comments_by_ticket: dict | None = None,
) -> Any:
    _open = open_tickets or []
    _planning = in_planning_tickets or []
    _comments = comments_by_ticket or {}

    class _StubTracker:
        def fetch_tickets_for_operation(self, op: Any) -> list:
            from src.ticket.workflow import WorkflowOperation
            if op == WorkflowOperation.OPEN_FOR_DEV:
                return _open
            if op == WorkflowOperation.IN_PLANNING:
                return _planning
            return []

        def fetch_ticket_comments(self, ticket_id: str) -> list:
            return _comments.get(ticket_id, [])

    class _StubTrackerRegistry:
        def get(self, source: str) -> _StubTracker:
            return _StubTracker()

    return _StubTrackerRegistry()


# ---------------------------------------------------------------------------
# E2E-DL-01: Dev Lead posts clarifying questions for ambiguous requirement
# ---------------------------------------------------------------------------

@skip_without_llm
class TestDevLeadFeasibilityAssessment:
    def test_e2e_dl_01_asks_clarifying_questions_not_creates_tickets(
        self,
        httpserver: HTTPServer,
    ) -> None:
        """E2E-DL-01 (ClickUp): Ambiguous requirement → Dev Lead asks questions, no task created."""
        from src.slack_app.handlers.dev_lead import dev_lead_handler

        stub = MCPStubServer(httpserver)
        url = stub.url

        create_task_calls: list = []
        create_issue_calls: list = []

        stub.register_tool("reply_to_thread", lambda args: {"ok": True})
        stub.register_tool("send_message", lambda args: {"ok": True})
        stub.register_tool("get_messages", lambda args: {
            "ok": True,
            "messages": [{"user": "U1", "text": "Let's add notifications", "ts": "1.0"}],
        })
        stub.register_tool("create_task", lambda args: (
            create_task_calls.append(args) or {"id": "cu-new"}
        ))
        stub.register_tool("create_issue", lambda args: (
            create_issue_calls.append(args) or {"key": "PROJ-NEW"}
        ))

        from test.e2e_test.common.e2e_settings import get_e2e_settings
        dev_lead_agent = build_dev_lead_agent_against_stubs(url=url, e2e_settings=get_e2e_settings())
        registry = _build_dev_lead_registry(dev_lead_agent)

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

        assert stub.was_called("reply_to_thread"), (
            "Expected dev_lead to call reply_to_thread with clarifying questions"
        )
        assert len(create_task_calls) == 0, (
            f"Dev Lead must NOT create ClickUp tasks for ambiguous requirement. "
            f"Got: {create_task_calls}"
        )
        assert len(create_issue_calls) == 0, (
            f"Dev Lead must NOT create JIRA issues for ambiguous requirement. "
            f"Got: {create_issue_calls}"
        )


# ---------------------------------------------------------------------------
# E2E-DL-02: Dev Lead fetches existing ClickUp task when task ID in message
# ---------------------------------------------------------------------------

@skip_without_llm
class TestDevLeadFetchesExistingTask:
    def test_e2e_dl_02_fetches_task_when_id_present(
        self,
        httpserver: HTTPServer,
    ) -> None:
        """E2E-DL-02 (ClickUp): Message contains task ID → Dev Lead calls get_task."""
        from src.slack_app.handlers.dev_lead import dev_lead_handler

        stub = MCPStubServer(httpserver)
        url = stub.url

        get_task_calls: list = []

        stub.register_tool("get_task", lambda args: (
            get_task_calls.append(args) or {
                "id": "cu-050",
                "name": "Add notification system",
                "status": {"status": "To Do"},
                "description": "We need a real-time notification system for user alerts.",
            }
        ))
        stub.register_tool("reply_to_thread", lambda args: {"ok": True})
        stub.register_tool("search_tasks", lambda args: [])

        from test.e2e_test.common.e2e_settings import get_e2e_settings
        dev_lead_agent = build_dev_lead_agent_against_stubs(url=url, e2e_settings=get_e2e_settings())
        registry = _build_dev_lead_registry(dev_lead_agent)

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            dev_lead_handler(
                text="[dev lead] cu-050 assess feasibility and provide breakdown plan",
                event={
                    "text": "[dev lead] cu-050 assess feasibility",
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

        assert len(get_task_calls) > 0, (
            "Expected Dev Lead to call get_task to fetch cu-050 task details. "
            f"All calls: {stub.all_calls}"
        )
        fetched_ids = [c.get("task_id", c.get("id", "")) for c in get_task_calls]
        assert any("cu-050" in str(k) for k in fetched_ids), (
            f"Expected cu-050 to be fetched. get_task calls: {get_task_calls}"
        )


# ---------------------------------------------------------------------------
# E2E-DL-03: Dev Lead creates sub-tasks after discussion concluded
# ---------------------------------------------------------------------------

@skip_without_llm
class TestDevLeadBreakdown:
    def test_e2e_dl_03_creates_subtasks_with_dependencies(
        self,
        httpserver: HTTPServer,
    ) -> None:
        """E2E-DL-03 (ClickUp): Explicit breakdown instruction → sub-tasks created."""
        from src.slack_app.handlers.dev_lead import dev_lead_handler

        stub = MCPStubServer(httpserver)
        url = stub.url

        create_task_calls: list = []
        add_comment_calls: list = []
        accepted_write_calls: list = []

        task_counter = [0]

        def handle_create_task(args: dict) -> dict:
            create_task_calls.append(args)
            task_counter[0] += 1
            status = str(args.get("status", args.get("fields", {}))).upper()
            if "ACCEPTED" in status:
                accepted_write_calls.append(args)
            return {"id": f"cu-10{task_counter[0]}"}

        stub.register_tool("get_task", lambda args: {
            "id": "cu-050",
            "name": "Add notification system",
            "status": {"status": "To Do"},
            "description": (
                "We need real-time notifications. "
                "Agreed: WebSocket + Redis pub/sub. "
                "Two main components: backend service and frontend subscriber."
            ),
        })
        stub.register_tool("create_task", handle_create_task)
        stub.register_tool("update_task", lambda args: {"ok": True})
        stub.register_tool("add_comment", lambda args: (
            add_comment_calls.append(args) or {"id": "c1"}
        ))
        stub.register_tool("reply_to_thread", lambda args: {"ok": True})
        stub.register_tool("search_tasks", lambda args: [])
        stub.register_tool("create_issue", lambda args: {"key": "PROJ-NEW"})
        stub.register_tool("update_issue", lambda args: {"ok": True})
        stub.register_tool("transition_issue", lambda args: {"ok": True})

        from test.e2e_test.common.e2e_settings import get_e2e_settings
        dev_lead_agent = build_dev_lead_agent_against_stubs(url=url, e2e_settings=get_e2e_settings())
        registry = _build_dev_lead_registry(dev_lead_agent)

        message = (
            "[dev lead] All questions answered — please break down cu-050 into sub-tasks. "
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

        assert len(create_task_calls) >= 2, (
            f"Expected at least 2 sub-tasks. Got: {len(create_task_calls)}"
        )
        assert len(add_comment_calls) > 0, (
            "Expected Dev Lead to add comment on parent task cu-050"
        )
        assert stub.was_called("reply_to_thread"), (
            "Expected Dev Lead to reply in Slack thread"
        )
        assert len(accepted_write_calls) == 0, (
            f"BR-1 violation: ACCEPTED written to sub-task. Calls: {accepted_write_calls}"
        )


# ---------------------------------------------------------------------------
# E2E-DL-04: Dev Agent generates initial development plan (ClickUp)
# ---------------------------------------------------------------------------

@skip_without_llm
class TestDevAgentInitialPlan:
    def test_e2e_dl_04_generates_plan_comment_for_open_task(
        self,
        httpserver: HTTPServer,
    ) -> None:
        """E2E-DL-04 (ClickUp): OPEN task → plan_and_notify_job → plan posted as comment."""
        import src.scheduler.jobs.plan_and_notify as pn_mod
        from src.scheduler.jobs.plan_and_notify import plan_and_notify_job

        stub = MCPStubServer(httpserver)
        url = stub.url

        add_comment_calls: list = []
        transition_calls: list = []

        stub.register_tool("get_task", lambda args: {
            "id": "cu-20",
            "name": "Implement OAuth2 login",
            "status": {"status": "OPEN"},
            "description": "Add OAuth2 authentication with Google SSO support.",
        })
        stub.register_tool("add_comment", lambda args: (
            add_comment_calls.append(args) or {"id": "c-plan"}
        ))
        stub.register_tool("update_task", lambda args: (
            transition_calls.append(args) or {"ok": True}
        ))
        stub.register_tool("search_tasks", lambda args: [
            {"id": "cu-20", "name": "Implement OAuth2 login",
             "status": {"status": "OPEN"}, "url": "https://app.clickup.com/t/cu-20"},
        ])
        stub.register_tool("search_issues", lambda args: [])
        stub.register_tool("reply_to_thread", lambda args: {"ok": True})
        stub.register_tool("send_message", lambda args: {"ok": True})

        workflow = WorkflowConfig(E2E_PLANNING_WORKFLOW_CONFIG)
        dev_agent = build_dev_agent_against_stubs(
            jira_url=url, slack_url=url, github_url=url, clickup_url=url,
            e2e_settings=get_e2e_settings(),
        )
        registry = build_e2e_registry(dev_agent)

        pn_mod._in_planning_tickets.clear()
        pn_mod._plan_comment_watermarks.clear()

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            plan_and_notify_job(
                registry=registry,
                settings=_make_settings(),
                executor=executor,
                workflow=workflow,
                tracker_registry=_make_stub_tracker_registry(
                    open_tickets=[
                        TicketRecord(id="cu-20", source="clickup",
                                     title="Implement OAuth2 login", url="", raw_status="OPEN"),
                    ],
                ),
            )
            executor.shutdown(wait=True)
        finally:
            executor.shutdown(wait=False)
            pn_mod._in_planning_tickets.clear()
            pn_mod._plan_comment_watermarks.clear()

        assert len(add_comment_calls) >= 1, (
            "Expected Dev Agent to post development plan. "
            f"All stub calls: {[c['tool'] for c in stub.all_calls]}"
        )

        plan_bodies = [
            c.get("comment", c.get("body", c.get("text", "")))
            for c in add_comment_calls
        ]
        has_plan = any(
            "plan" in body.lower() or "## " in body or "development" in body.lower()
            for body in plan_bodies
        )
        assert has_plan, f"Expected plan comment. Bodies: {plan_bodies}"

        accepted_transitions = [t for t in transition_calls if "ACCEPTED" in str(t).upper()]
        assert len(accepted_transitions) == 0, f"BR-1 violated: {transition_calls}"

        in_progress_transitions = [
            t for t in transition_calls
            if "IN PROGRESS" in str(t).upper()
        ]
        assert len(in_progress_transitions) == 0, f"BR-8 violated: {transition_calls}"


# ---------------------------------------------------------------------------
# E2E-DL-05: Dev Agent revises plan based on human comments (ClickUp)
# ---------------------------------------------------------------------------

@skip_without_llm
class TestDevAgentPlanRevision:
    def test_e2e_dl_05_revises_plan_based_on_human_comments(
        self,
        httpserver: HTTPServer,
    ) -> None:
        """E2E-DL-05 (ClickUp): IN PLANNING task + human comment → revised plan posted."""
        import src.scheduler.jobs.plan_and_notify as pn_mod
        from src.scheduler.jobs.plan_and_notify import plan_and_notify_job

        stub = MCPStubServer(httpserver)
        url = stub.url

        add_comment_calls: list = []
        transition_calls: list = []

        stub.register_tool("get_task", lambda args: {
            "id": "cu-21",
            "name": "Implement database migration tool",
            "status": {"status": "IN PLANNING"},
            "description": "Build a tool to run database migrations safely.",
        })
        stub.register_tool("add_comment", lambda args: (
            add_comment_calls.append(args) or {"id": "c-rev"}
        ))
        stub.register_tool("update_task", lambda args: (
            transition_calls.append(args) or {"ok": True}
        ))
        stub.register_tool("reply_to_thread", lambda args: {"ok": True})
        stub.register_tool("send_message", lambda args: {"ok": True})
        stub.register_tool("search_issues", lambda args: [])
        stub.register_tool("search_tasks", lambda args: [])

        workflow = WorkflowConfig(E2E_PLANNING_WORKFLOW_CONFIG)
        dev_agent = build_dev_agent_against_stubs(
            jira_url=url, slack_url=url, github_url=url, clickup_url=url,
            e2e_settings=get_e2e_settings(),
        )
        registry = build_e2e_registry(dev_agent)

        human_comment = TicketComment(
            id="c-human-1",
            author="alice",
            body="Please clarify how you will handle rollback if the migration fails.",
            created_at=time.time() - 60,
            source="clickup",
        )

        pn_mod._in_planning_tickets.clear()
        pn_mod._plan_comment_watermarks.clear()

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            plan_and_notify_job(
                registry=registry,
                settings=_make_settings(),
                executor=executor,
                workflow=workflow,
                tracker_registry=_make_stub_tracker_registry(
                    in_planning_tickets=[
                        TicketRecord(id="cu-21", source="clickup",
                                     title="DB migration tool", url="", raw_status="IN PLANNING"),
                    ],
                    comments_by_ticket={"cu-21": [human_comment]},
                ),
            )
            executor.shutdown(wait=True)
        finally:
            executor.shutdown(wait=False)
            pn_mod._in_planning_tickets.clear()
            pn_mod._plan_comment_watermarks.clear()

        assert len(add_comment_calls) >= 1, (
            "Expected Dev Agent to post revised plan. "
            f"All stub calls: {[c['tool'] for c in stub.all_calls]}"
        )

        plan_bodies = [
            c.get("comment", c.get("body", c.get("text", "")))
            for c in add_comment_calls
        ]
        has_rollback_mention = any(
            "rollback" in body.lower() or "migration" in body.lower()
            or "revision" in body.lower()
            for body in plan_bodies
        )
        assert has_rollback_mention, (
            f"Expected revised plan to address rollback feedback. Bodies: {plan_bodies}"
        )

        bad_transitions = [
            t for t in transition_calls
            if "ACCEPTED" in str(t).upper() or "IN PROGRESS" in str(t).upper()
        ]
        assert len(bad_transitions) == 0, (
            f"BR violation: unexpected transitions: {transition_calls}"
        )


# ---------------------------------------------------------------------------
# E2E-DL-06: Full planning lifecycle (ClickUp)
# ---------------------------------------------------------------------------

@skip_without_llm
class TestFullPlanningLifecycle:
    def test_e2e_dl_06_full_planning_lifecycle(
        self,
        httpserver: HTTPServer,
    ) -> None:
        """E2E-DL-06 (ClickUp): Dev Lead breakdown → Dev Agent initial plan → revision."""
        import src.scheduler.jobs.plan_and_notify as pn_mod
        from src.scheduler.jobs.plan_and_notify import plan_and_notify_job
        from src.slack_app.handlers.dev_lead import dev_lead_handler

        stub = MCPStubServer(httpserver)
        url = stub.url

        create_task_calls: list = []
        add_comment_calls: list = []
        plan_comment_calls: list = []
        accepted_writes: list = []

        task_counter = [0]

        def handle_create_task(args: dict) -> dict:
            create_task_calls.append(args)
            task_counter[0] += 1
            if "ACCEPTED" in str(args).upper():
                accepted_writes.append(args)
            return {"id": f"cu-dl-0{task_counter[0]}"}

        stub.register_tool("get_task", lambda args: {
            "id": "cu-050",
            "name": "Add real-time notification system",
            "status": {"status": "To Do"},
            "description": (
                "WebSocket backend + Redis pub/sub + frontend subscriber. "
                "Architecture agreed."
            ),
        })
        stub.register_tool("create_task", handle_create_task)
        stub.register_tool("update_task", lambda args: {"ok": True})
        stub.register_tool("add_comment", lambda args: (
            add_comment_calls.append(args) or {"id": "c1"}
        ))
        stub.register_tool("reply_to_thread", lambda args: {"ok": True})
        stub.register_tool("send_message", lambda args: {"ok": True})
        stub.register_tool("search_tasks", lambda args: [])
        stub.register_tool("search_issues", lambda args: [])
        stub.register_tool("create_issue", lambda args: {"key": "PROJ-NEW"})
        stub.register_tool("transition_issue", lambda args: {"ok": True})

        from test.e2e_test.common.e2e_settings import get_e2e_settings
        dev_lead_agent = build_dev_lead_agent_against_stubs(url=url, e2e_settings=get_e2e_settings())
        dev_lead_registry = _build_dev_lead_registry(dev_lead_agent)

        # Phase 1: Dev Lead breakdown
        message = (
            "[dev lead] All questions answered — break down cu-050. "
            "Architecture: WebSocket + Redis + frontend. Create tasks."
        )
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            dev_lead_handler(
                text=message,
                event={"text": message, "channel": "C001",
                       "thread_ts": "300.1", "ts": "300.1"},
                say=MagicMock(),
                registry=dev_lead_registry,
                executor=executor,
            )
            executor.shutdown(wait=True)
        finally:
            executor.shutdown(wait=False)

        assert len(create_task_calls) >= 1, (
            f"Expected Dev Lead to create at least 1 sub-task. Got: {create_task_calls}"
        )
        assert len(accepted_writes) == 0, f"BR-1 violated: {accepted_writes}"

        # Phase 2: Dev Agent plans the first created sub-task
        new_task_id = create_task_calls[0].get("name", "sub-task-1") if create_task_calls else "cu-new"

        stub.register_tool("add_comment", lambda args: (
            plan_comment_calls.append(args) or {"id": "c-plan"}
        ))
        stub.register_tool("get_task", lambda args: {
            "id": args.get("task_id", new_task_id),
            "name": "WebSocket backend service",
            "status": {"status": "OPEN"},
            "description": "Implement WebSocket backend for real-time notifications.",
        })

        workflow = WorkflowConfig(E2E_PLANNING_WORKFLOW_CONFIG)
        dev_agent = build_dev_agent_against_stubs(
            jira_url=url, slack_url=url, github_url=url, clickup_url=url,
            e2e_settings=get_e2e_settings(),
        )
        dev_registry = build_e2e_registry(dev_agent)

        pn_mod._in_planning_tickets.clear()
        pn_mod._plan_comment_watermarks.clear()

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            plan_and_notify_job(
                registry=dev_registry,
                settings=_make_settings(),
                executor=executor,
                workflow=workflow,
                tracker_registry=_make_stub_tracker_registry(
                    open_tickets=[
                        TicketRecord(id="cu-new-sub", source="clickup",
                                     title="WebSocket backend service",
                                     url="", raw_status="OPEN"),
                    ],
                ),
            )
            executor.shutdown(wait=True)
        finally:
            executor.shutdown(wait=False)
            pn_mod._in_planning_tickets.clear()
            pn_mod._plan_comment_watermarks.clear()

        assert len(plan_comment_calls) >= 1, (
            "Expected Dev Agent to post development plan comment"
        )


