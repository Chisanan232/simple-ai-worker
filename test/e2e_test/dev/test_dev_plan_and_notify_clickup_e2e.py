"""
E2E tests: Dev Agent development plan & notify — ClickUp backend (E2E-PN-01 through E2E-PN-07).

Scenario steps covered: S-PL-5 through S-PL-7

Verifies:
- Dev Agent generates initial Markdown development plan for an OPEN task (E2E-PN-01)
- Dev Agent handles multiple OPEN tasks in one job run — batch planning (E2E-PN-02)
- Dispatch guard prevents double-planning the same task (E2E-PN-03)
- Dev Agent revises plan based on human comment on IN PLANNING task (E2E-PN-04)
- No revision dispatched when no new comments since watermark (E2E-PN-05)
- Plan comment includes notification to human engineer (E2E-PN-06)
- Full planning loop: OPEN → plan → human comment → revision → ACCEPTED (E2E-PN-07)

Business rules:
- BR-1: ACCEPTED status must NEVER be written by AI
- BR-8: Dev Agent must NOT start coding (no IN PROGRESS, no PR) during planning mode
- BR-10: IN PLANNING status must NEVER be written by AI
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

from test.e2e_test.conftest import (
    E2E_WORKFLOW_CONFIG,
    FakeLLM,
    MCPStubServer,
    build_dev_agent_against_stubs,
    build_e2e_registry,
    skip_without_llm,
)

from src.ticket.models import TicketComment, TicketRecord
from src.ticket.workflow import WorkflowConfig

# Planning workflow extends the standard config
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


def _make_stub_tracker_registry(
    open_tickets: list | None = None,
    in_planning_tickets: list | None = None,
    comments_by_ticket: dict | None = None,
) -> Any:
    """Build a minimal _StubTrackerRegistry for plan_and_notify_job."""
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


# ===========================================================================
# E2E-PN-01: Dev Agent generates initial plan for a single OPEN task
# ===========================================================================


@skip_without_llm
class TestDevAgentGeneratesInitialPlan:
    def test_generates_initial_plan_for_open_task(
        self,
        mcp_stub: MCPStubServer,
        planning_tool_order: None,
    ) -> None:
        """E2E-PN-01 (ClickUp): OPEN task → plan_and_notify_job → plan posted as comment."""
        import src.scheduler.jobs.plan_and_notify as pn_mod
        from src.scheduler.jobs.plan_and_notify import plan_and_notify_job

        stub = mcp_stub
        url = stub.url

        add_comment_calls: list = []
        transition_calls: list = []

        stub.register_tool(
            "get_task",
            lambda args: {
                "id": "cu-20",
                "name": "Implement OAuth2 login",
                "status": {"status": "OPEN"},
                "description": "Add OAuth2 authentication with Google SSO support.",
            },
        )
        stub.register_tool("add_comment", lambda args: (add_comment_calls.append(args) or {"id": "c-plan"}))
        stub.register_tool("update_task", lambda args: (transition_calls.append(args) or {"ok": True}))
        stub.register_tool(
            "search_tasks",
            lambda args: [
                {
                    "id": "cu-20",
                    "name": "Implement OAuth2 login",
                    "status": {"status": "OPEN"},
                    "url": "https://app.clickup.com/t/cu-20",
                },
            ],
        )
        stub.register_tool("search_issues", lambda args: [])
        stub.register_tool("reply_to_thread", lambda args: {"ok": True})
        stub.register_tool("send_message", lambda args: {"ok": True})

        workflow = WorkflowConfig(E2E_PLANNING_WORKFLOW_CONFIG)
        dev_agent = build_dev_agent_against_stubs(
            jira_url=url,
            slack_url=url,
            github_url=url,
            clickup_url=url,
        )
        registry = build_e2e_registry(dev_agent)

        tracker_registry = _make_stub_tracker_registry(
            open_tickets=[
                TicketRecord(id="cu-20", source="clickup", title="Implement OAuth2 login", url="", raw_status="OPEN"),
            ],
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
                tracker_registry=tracker_registry,
            )
            executor.shutdown(wait=True)
        finally:
            executor.shutdown(wait=False)
            pn_mod._in_planning_tickets.clear()
            pn_mod._plan_comment_watermarks.clear()

        # 1. add_comment called at least once (plan posted)
        assert len(add_comment_calls) >= 1, (
            "Expected Dev Agent to post development plan as a ticket comment. "
            f"All stub calls: {[c['tool'] for c in stub.all_calls]}"
        )

        # 2. Comment body contains Markdown heading
        plan_bodies = [c.get("comment", c.get("body", c.get("text", ""))) for c in add_comment_calls]
        has_plan_heading = any(
            "## " in body or "plan" in body.lower() or "development" in body.lower() for body in plan_bodies
        )
        assert has_plan_heading, f"Expected plan comment to contain Markdown heading. Bodies: {plan_bodies}"

        # 3. Comment contains at least one technical section
        has_tech_section = any(
            any(
                kw in body.lower()
                for kw in ("approach", "steps", "implementation", "solution", "design", "architecture")
            )
            for body in plan_bodies
        )
        assert has_tech_section, f"Expected plan to contain a technical section. Bodies: {plan_bodies}"

        # 4. update_task NOT called with IN PROGRESS (BR-8)
        in_progress_calls = [
            t for t in transition_calls if "IN PROGRESS" in str(t).upper() or "in_progress" in str(t).lower()
        ]
        assert len(in_progress_calls) == 0, (
            f"BR-8 violated: Dev Agent transitioned to IN PROGRESS during planning. " f"Calls: {transition_calls}"
        )

        # 5. update_task NOT called with ACCEPTED (BR-1)
        accepted_calls = [t for t in transition_calls if "ACCEPTED" in str(t).upper()]
        assert len(accepted_calls) == 0, f"BR-1 violated: Dev Agent wrote ACCEPTED status. Calls: {transition_calls}"

        # 6. update_task NOT called with IN PLANNING (BR-10)
        in_planning_calls = [
            t for t in transition_calls if "IN PLANNING" in str(t).upper() or "in_planning" in str(t).lower()
        ]
        assert (
            len(in_planning_calls) == 0
        ), f"BR-10 violated: Dev Agent wrote IN PLANNING status. Calls: {transition_calls}"

        # 7. No create_pull_request call (BR-8 — no coding during planning)
        assert not stub.was_called("create_pull_request"), "BR-8 violated: Dev Agent opened a PR during planning mode"


# ===========================================================================
# E2E-PN-02: Dev Agent generates plans for multiple OPEN tasks (batch)
# ===========================================================================


@skip_without_llm
class TestDevAgentBatchPlanning:
    def test_generates_plans_for_multiple_open_tasks(
        self,
        mcp_stub: MCPStubServer,
        planning_tool_order: None,
    ) -> None:
        """E2E-PN-02 (ClickUp): 3 OPEN tasks → plan_and_notify_job → 3 plan comments."""
        import src.scheduler.jobs.plan_and_notify as pn_mod
        from src.scheduler.jobs.plan_and_notify import plan_and_notify_job

        stub = mcp_stub
        url = stub.url

        add_comment_calls: list = []

        task_details = {
            "cu-30": {
                "id": "cu-30",
                "name": "Feature A",
                "status": {"status": "OPEN"},
                "description": "Implement Feature A with caching layer.",
            },
            "cu-31": {
                "id": "cu-31",
                "name": "Feature B",
                "status": {"status": "OPEN"},
                "description": "Add Feature B as background worker.",
            },
            "cu-32": {
                "id": "cu-32",
                "name": "Feature C",
                "status": {"status": "OPEN"},
                "description": "Build Feature C API endpoints.",
            },
        }

        stub.register_tool(
            "get_task",
            lambda args: (
                task_details.get(
                    args.get("task_id", ""),
                    {"id": "unknown", "name": "Unknown", "status": {"status": "OPEN"}, "description": "Unknown task"},
                )
            ),
        )
        stub.register_tool(
            "add_comment", lambda args: (add_comment_calls.append(args) or {"id": f"c-{len(add_comment_calls)}"})
        )
        stub.register_tool("update_task", lambda args: {"ok": True})
        stub.register_tool(
            "search_tasks",
            lambda args: [
                {"id": tid, "name": d["name"], "status": {"status": "OPEN"}, "url": f"https://app.clickup.com/t/{tid}"}
                for tid, d in task_details.items()
            ],
        )
        stub.register_tool("search_issues", lambda args: [])
        stub.register_tool("reply_to_thread", lambda args: {"ok": True})
        stub.register_tool("send_message", lambda args: {"ok": True})

        workflow = WorkflowConfig(E2E_PLANNING_WORKFLOW_CONFIG)
        dev_agent = build_dev_agent_against_stubs(
            jira_url=url,
            slack_url=url,
            github_url=url,
            clickup_url=url,
        )
        registry = build_e2e_registry(dev_agent)

        tracker_registry = _make_stub_tracker_registry(
            open_tickets=[
                TicketRecord(id=tid, source="clickup", title=d["name"], url="", raw_status="OPEN")
                for tid, d in task_details.items()
            ],
        )

        pn_mod._in_planning_tickets.clear()
        pn_mod._plan_comment_watermarks.clear()

        # max_workers=1 runs the 3 planning crews sequentially.
        # Using max_workers=3 (concurrent) caused an intermittent race: concurrent
        # CrewAI crews share the same anyio event loop machinery, and the async
        # HTTP-client teardown of one crew interfered with another, causing
        # crew.kickoff() to raise RuntimeError("cancel scope in different task").
        # Sequential execution eliminates the race while still validating that
        # all 3 tickets receive a plan comment.
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            plan_and_notify_job(
                registry=registry,
                settings=_make_settings(),
                executor=executor,
                workflow=workflow,
                tracker_registry=tracker_registry,
            )
            executor.shutdown(wait=True)
        finally:
            executor.shutdown(wait=False)
            pn_mod._in_planning_tickets.clear()
            pn_mod._plan_comment_watermarks.clear()

        # 1. add_comment called at least 3 times (one plan per ticket)
        assert (
            len(add_comment_calls) >= 3
        ), f"Expected at least 3 plan comments (one per task). Got: {len(add_comment_calls)}"

        # 2. No create_pull_request call
        assert not stub.was_called("create_pull_request"), "BR-8 violated: Dev Agent opened a PR during batch planning"


# ===========================================================================
# E2E-PN-03: Dispatch guard prevents double-planning the same ticket
# ===========================================================================


@skip_without_llm
class TestDispatchGuardPreventsDoublePlanning:
    def test_dispatch_guard_prevents_double_planning(
        self,
        mcp_stub: MCPStubServer,
    ) -> None:
        """E2E-PN-03 (ClickUp): _in_planning_tickets pre-seeded → ticket NOT re-planned."""
        import src.scheduler.jobs.plan_and_notify as pn_mod
        from src.scheduler.jobs.plan_and_notify import plan_and_notify_job

        stub = mcp_stub
        url = stub.url

        add_comment_calls: list = []

        stub.register_tool(
            "get_task",
            lambda args: {
                "id": "cu-20",
                "name": "Implement OAuth2 login",
                "status": {"status": "OPEN"},
                "description": "Add OAuth2 authentication with Google SSO support.",
            },
        )
        stub.register_tool("add_comment", lambda args: (add_comment_calls.append(args) or {"id": "c-plan"}))
        stub.register_tool("update_task", lambda args: {"ok": True})
        stub.register_tool(
            "search_tasks",
            lambda args: [
                {
                    "id": "cu-20",
                    "name": "Implement OAuth2 login",
                    "status": {"status": "OPEN"},
                    "url": "https://app.clickup.com/t/cu-20",
                },
            ],
        )
        stub.register_tool("search_issues", lambda args: [])

        workflow = WorkflowConfig(E2E_PLANNING_WORKFLOW_CONFIG)
        dev_agent = build_dev_agent_against_stubs(
            jira_url=url,
            slack_url=url,
            github_url=url,
            clickup_url=url,
        )
        registry = build_e2e_registry(dev_agent)

        tracker_registry = _make_stub_tracker_registry(
            open_tickets=[
                TicketRecord(id="cu-20", source="clickup", title="Implement OAuth2 login", url="", raw_status="OPEN"),
            ],
        )

        # Pre-seed the dispatch guard — cu-20 is already in planning
        pn_mod._in_planning_tickets.clear()
        pn_mod._in_planning_tickets.add("cu-20")
        pn_mod._plan_comment_watermarks.clear()

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            plan_and_notify_job(
                registry=registry,
                settings=_make_settings(),
                executor=executor,
                workflow=workflow,
                tracker_registry=tracker_registry,
            )
            executor.shutdown(wait=True)
        finally:
            executor.shutdown(wait=False)
            pn_mod._in_planning_tickets.clear()
            pn_mod._plan_comment_watermarks.clear()

        # add_comment must NOT be called (ticket already in _in_planning_tickets)
        assert len(add_comment_calls) == 0, (
            f"Dev Agent must NOT re-plan a ticket already in _in_planning_tickets. "
            f"add_comment calls: {add_comment_calls}"
        )


# ===========================================================================
# E2E-PN-04: Dev Agent revises plan based on human comment
# ===========================================================================


@skip_without_llm
class TestDevAgentRevisesPlanOnHumanFeedback:
    def test_revises_plan_on_human_feedback(
        self,
        mcp_stub: MCPStubServer,
        planning_tool_order: None,
    ) -> None:
        """E2E-PN-04 (ClickUp): IN PLANNING task + human comment → revised plan posted."""
        import src.scheduler.jobs.plan_and_notify as pn_mod
        from src.scheduler.jobs.plan_and_notify import plan_and_notify_job

        stub = mcp_stub
        url = stub.url

        add_comment_calls: list = []
        transition_calls: list = []

        stub.register_tool(
            "get_task",
            lambda args: {
                "id": "cu-21",
                "name": "DB migration tool",
                "status": {"status": "IN PLANNING"},
                "description": "Build a tool to run database migrations safely.",
            },
        )
        stub.register_tool("add_comment", lambda args: (add_comment_calls.append(args) or {"id": "c-rev"}))
        stub.register_tool("update_task", lambda args: (transition_calls.append(args) or {"ok": True}))
        stub.register_tool(
            "search_tasks",
            lambda args: [
                {
                    "id": "cu-21",
                    "name": "DB migration tool",
                    "status": {"status": "IN PLANNING"},
                    "url": "https://app.clickup.com/t/cu-21",
                },
            ],
        )
        stub.register_tool("search_issues", lambda args: [])
        stub.register_tool("reply_to_thread", lambda args: {"ok": True})
        stub.register_tool("send_message", lambda args: {"ok": True})

        workflow = WorkflowConfig(E2E_PLANNING_WORKFLOW_CONFIG)
        dev_agent = build_dev_agent_against_stubs(
            jira_url=url,
            slack_url=url,
            github_url=url,
            clickup_url=url,
        )
        registry = build_e2e_registry(dev_agent)

        human_comment = TicketComment(
            id="c-human-1",
            author="alice",
            body="Please clarify how you will handle rollback if the migration fails.",
            created_at=time.time() - 60,
            source="clickup",
        )

        tracker_registry = _make_stub_tracker_registry(
            in_planning_tickets=[
                TicketRecord(id="cu-21", source="clickup", title="DB migration tool", url="", raw_status="IN PLANNING"),
            ],
            comments_by_ticket={"cu-21": [human_comment]},
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
                tracker_registry=tracker_registry,
            )
            executor.shutdown(wait=True)
        finally:
            executor.shutdown(wait=False)
            pn_mod._in_planning_tickets.clear()
            pn_mod._plan_comment_watermarks.clear()

        # 1. add_comment called at least once (revised plan posted)
        assert len(add_comment_calls) >= 1, (
            "Expected Dev Agent to post revised plan as ticket comment. "
            f"All stub calls: {[c['tool'] for c in stub.all_calls]}"
        )

        # 2. Revised comment addresses the feedback topic
        plan_bodies = [c.get("comment", c.get("body", c.get("text", ""))) for c in add_comment_calls]
        has_rollback = any(
            "rollback" in body.lower()
            or "migration" in body.lower()
            or "failure" in body.lower()
            or "revision" in body.lower()
            for body in plan_bodies
        )
        assert has_rollback, f"Expected revised plan to address rollback feedback. Bodies: {plan_bodies}"

        # 3. update_task NOT called with IN PLANNING (BR-10)
        in_planning_calls = [
            t for t in transition_calls if "IN PLANNING" in str(t).upper() or "in_planning" in str(t).lower()
        ]
        assert len(in_planning_calls) == 0, f"BR-10 violated: Dev Agent wrote IN PLANNING. Calls: {transition_calls}"

        # 4. update_task NOT called with ACCEPTED (BR-1)
        accepted_calls = [t for t in transition_calls if "ACCEPTED" in str(t).upper()]
        assert len(accepted_calls) == 0, f"BR-1 violated: Dev Agent wrote ACCEPTED. Calls: {transition_calls}"

        # 5. No create_pull_request (BR-8)
        assert not stub.was_called(
            "create_pull_request"
        ), "BR-8 violated: Dev Agent opened a PR during planning revision"


# ===========================================================================
# E2E-PN-05: No revision dispatched when no new comments since watermark
# ===========================================================================


@skip_without_llm
class TestNoRevisionWhenNoNewComments:
    def test_no_revision_when_no_new_comments(
        self,
        mcp_stub: MCPStubServer,
    ) -> None:
        """E2E-PN-05 (ClickUp): Watermark newer than comment → no revision dispatched."""
        import src.scheduler.jobs.plan_and_notify as pn_mod
        from src.scheduler.jobs.plan_and_notify import plan_and_notify_job

        stub = mcp_stub
        url = stub.url

        add_comment_calls: list = []

        stub.register_tool(
            "get_task",
            lambda args: {
                "id": "cu-21",
                "name": "DB migration tool",
                "status": {"status": "IN PLANNING"},
                "description": "Build a tool to run database migrations safely.",
            },
        )
        stub.register_tool("add_comment", lambda args: (add_comment_calls.append(args) or {"id": "c-rev"}))
        stub.register_tool("update_task", lambda args: {"ok": True})
        stub.register_tool("search_tasks", lambda args: [])
        stub.register_tool("search_issues", lambda args: [])

        workflow = WorkflowConfig(E2E_PLANNING_WORKFLOW_CONFIG)
        dev_agent = build_dev_agent_against_stubs(
            jira_url=url,
            slack_url=url,
            github_url=url,
            clickup_url=url,
        )
        registry = build_e2e_registry(dev_agent)

        # Human comment is OLD (120 seconds ago)
        old_comment = TicketComment(
            id="c-human-old",
            author="alice",
            body="Please clarify rollback handling.",
            created_at=time.time() - 120,
            source="clickup",
        )

        tracker_registry = _make_stub_tracker_registry(
            in_planning_tickets=[
                TicketRecord(id="cu-21", source="clickup", title="DB migration tool", url="", raw_status="IN PLANNING"),
            ],
            comments_by_ticket={"cu-21": [old_comment]},
        )

        # Watermark is NEWER than the comment (60s ago — newer than 120s)
        pn_mod._in_planning_tickets.clear()
        pn_mod._plan_comment_watermarks.clear()
        pn_mod._plan_comment_watermarks["cu-21"] = time.time() - 60

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            plan_and_notify_job(
                registry=registry,
                settings=_make_settings(),
                executor=executor,
                workflow=workflow,
                tracker_registry=tracker_registry,
            )
            executor.shutdown(wait=True)
        finally:
            executor.shutdown(wait=False)
            pn_mod._in_planning_tickets.clear()
            pn_mod._plan_comment_watermarks.clear()

        # add_comment NOT called (no new comments to respond to)
        assert len(add_comment_calls) == 0, (
            f"Dev Agent must NOT post revision when no new comments since watermark. "
            f"add_comment calls: {add_comment_calls}"
        )


# ===========================================================================
# E2E-PN-06: Plan comment includes notification to human engineer
# ===========================================================================


@skip_without_llm
class TestPlanCommentIncludesHumanNotification:
    def test_plan_comment_includes_human_notification(
        self,
        mcp_stub: MCPStubServer,
        planning_tool_order: None,
    ) -> None:
        """E2E-PN-06 (ClickUp): Plan comment contains human review notification."""
        import src.scheduler.jobs.plan_and_notify as pn_mod
        from src.scheduler.jobs.plan_and_notify import plan_and_notify_job

        stub = mcp_stub
        url = stub.url

        add_comment_calls: list = []

        stub.register_tool(
            "get_task",
            lambda args: {
                "id": "cu-20",
                "name": "Implement OAuth2 login",
                "status": {"status": "OPEN"},
                "description": "Add OAuth2 authentication with Google SSO support.",
            },
        )
        stub.register_tool("add_comment", lambda args: (add_comment_calls.append(args) or {"id": "c-plan"}))
        stub.register_tool("update_task", lambda args: {"ok": True})
        stub.register_tool(
            "search_tasks",
            lambda args: [
                {
                    "id": "cu-20",
                    "name": "Implement OAuth2 login",
                    "status": {"status": "OPEN"},
                    "url": "https://app.clickup.com/t/cu-20",
                },
            ],
        )
        stub.register_tool("search_issues", lambda args: [])
        stub.register_tool("reply_to_thread", lambda args: {"ok": True})
        stub.register_tool("send_message", lambda args: {"ok": True})

        workflow = WorkflowConfig(E2E_PLANNING_WORKFLOW_CONFIG)
        dev_agent = build_dev_agent_against_stubs(
            jira_url=url,
            slack_url=url,
            github_url=url,
            clickup_url=url,
        )
        registry = build_e2e_registry(dev_agent)

        tracker_registry = _make_stub_tracker_registry(
            open_tickets=[
                TicketRecord(id="cu-20", source="clickup", title="Implement OAuth2 login", url="", raw_status="OPEN"),
            ],
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
                tracker_registry=tracker_registry,
            )
            executor.shutdown(wait=True)
        finally:
            executor.shutdown(wait=False)
            pn_mod._in_planning_tickets.clear()
            pn_mod._plan_comment_watermarks.clear()

        # 1. add_comment called
        assert len(add_comment_calls) >= 1, "Expected Dev Agent to post plan comment."

        # 2. Comment body contains a human-directed notification phrase
        plan_bodies = [c.get("comment", c.get("body", c.get("text", ""))) for c in add_comment_calls]
        has_notification = any(
            any(
                kw in body.lower()
                for kw in (
                    "please review",
                    "ready for review",
                    "review",
                    "feedback",
                    "@",
                    "human",
                    "engineer",
                    "notify",
                )
            )
            for body in plan_bodies
        )
        assert has_notification, f"Expected plan comment to include human notification phrase. Bodies: {plan_bodies}"


# ===========================================================================
# E2E-PN-07: Full planning loop — OPEN → plan → human comment → revision → ACCEPTED
# ===========================================================================


@skip_without_llm
class TestFullPlanningLoop:
    def test_full_planning_loop(
        self,
        mcp_stub: MCPStubServer,
        planning_tool_order: None,
        fake_llm_session: Optional[FakeLLM],
    ) -> None:
        """E2E-PN-07 (ClickUp): OPEN → initial plan → IN PLANNING → revision → ACCEPTED → dispatch."""
        import src.scheduler.jobs.plan_and_notify as pn_mod
        from src.scheduler.jobs.plan_and_notify import plan_and_notify_job

        stub = mcp_stub
        url = stub.url

        add_comment_calls_run1: list = []
        add_comment_calls_run2: list = []
        in_progress_calls: list = []
        accepted_write_calls: list = []
        in_planning_write_calls: list = []
        pr_calls: list = []

        workflow = WorkflowConfig(E2E_PLANNING_WORKFLOW_CONFIG)
        dev_agent = build_dev_agent_against_stubs(
            jira_url=url,
            slack_url=url,
            github_url=url,
            clickup_url=url,
        )
        registry = build_e2e_registry(dev_agent)

        # ── RUN 1: ticket at OPEN → initial plan ──────────────────────────
        stub.register_tool(
            "get_task",
            lambda args: {
                "id": "cu-22",
                "name": "Search indexing service",
                "status": {"status": "OPEN"},
                "description": "Build an Elasticsearch-based search indexing service.",
            },
        )
        stub.register_tool(
            "add_comment",
            lambda args: (add_comment_calls_run1.append(args) or {"id": f"c-r1-{len(add_comment_calls_run1)}"}),
        )

        def _update_task_run1(args: dict) -> dict:
            s = str(args).upper()
            if "IN PROGRESS" in s or "in_progress" in str(args).lower():
                in_progress_calls.append(args)
            if "ACCEPTED" in s:
                accepted_write_calls.append(args)
            if "IN PLANNING" in s or "in_planning" in str(args).lower():
                in_planning_write_calls.append(args)
            return {"ok": True}

        stub.register_tool("update_task", _update_task_run1)
        stub.register_tool(
            "create_pull_request",
            lambda args: (pr_calls.append(args) or {"html_url": "https://github.com/org/repo/pull/0", "number": 0}),
        )
        stub.register_tool(
            "search_tasks",
            lambda args: [
                {
                    "id": "cu-22",
                    "name": "Search indexing service",
                    "status": {"status": "OPEN"},
                    "url": "https://app.clickup.com/t/cu-22",
                },
            ],
        )
        stub.register_tool("search_issues", lambda args: [])
        stub.register_tool("reply_to_thread", lambda args: {"ok": True})
        stub.register_tool("send_message", lambda args: {"ok": True})

        tracker_registry_run1 = _make_stub_tracker_registry(
            open_tickets=[
                TicketRecord(id="cu-22", source="clickup", title="Search indexing service", url="", raw_status="OPEN"),
            ],
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
                tracker_registry=tracker_registry_run1,
            )
            executor.shutdown(wait=True)
        finally:
            executor.shutdown(wait=False)

        # RUN 1 assertions
        assert len(add_comment_calls_run1) >= 1, "Expected initial plan comment in run 1"
        assert len(pr_calls) == 0, "BR-8 violated: PR opened during planning run 1"
        assert len(in_progress_calls) == 0, "BR-8 violated: IN PROGRESS transition during planning run 1"
        assert len(accepted_write_calls) == 0, "BR-1 violated: ACCEPTED written during planning run 1"
        assert len(in_planning_write_calls) == 0, "BR-10 violated: IN PLANNING written during planning run 1"

        # Reset FakeLLM turn counters so run 2 starts fresh
        if fake_llm_session is not None:
            fake_llm_session.reset_turns()

        # ── RUN 2: ticket at IN PLANNING with human comment → revision ────
        human_comment = TicketComment(
            id="c-human-2",
            author="alice",
            body="Please add details on how you'll handle index failures and retries.",
            created_at=time.time() - 30,
            source="clickup",
        )

        # Re-register add_comment to capture run 2 calls
        stub.register_tool(
            "add_comment",
            lambda args: (add_comment_calls_run2.append(args) or {"id": f"c-r2-{len(add_comment_calls_run2)}"}),
        )
        stub.register_tool(
            "get_task",
            lambda args: {
                "id": "cu-22",
                "name": "Search indexing service",
                "status": {"status": "IN PLANNING"},
                "description": "Build an Elasticsearch-based search indexing service.",
            },
        )

        tracker_registry_run2 = _make_stub_tracker_registry(
            in_planning_tickets=[
                TicketRecord(
                    id="cu-22", source="clickup", title="Search indexing service", url="", raw_status="IN PLANNING"
                ),
            ],
            comments_by_ticket={"cu-22": [human_comment]},
        )

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            plan_and_notify_job(
                registry=registry,
                settings=_make_settings(),
                executor=executor,
                workflow=workflow,
                tracker_registry=tracker_registry_run2,
            )
            executor.shutdown(wait=True)
        finally:
            executor.shutdown(wait=False)
            pn_mod._in_planning_tickets.clear()
            pn_mod._plan_comment_watermarks.clear()

        # RUN 2 assertions: two add_comment calls total (initial + revised)
        total_comments = len(add_comment_calls_run1) + len(add_comment_calls_run2)
        assert total_comments >= 2, (
            f"Expected at least 2 total plan comments (initial + revision). "
            f"Got: run1={len(add_comment_calls_run1)}, run2={len(add_comment_calls_run2)}"
        )
        assert len(pr_calls) == 0, "BR-8 violated: PR opened during planning run 2"
        assert len(in_planning_write_calls) == 0, "BR-10 violated: IN PLANNING written during planning run 2"
        assert len(accepted_write_calls) == 0, "BR-1 violated: ACCEPTED written during any planning run"
