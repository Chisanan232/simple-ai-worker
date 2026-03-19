"""
E2E tests: Dev Agent development plan & notify — JIRA backend variant (E2E-PN-01 through E2E-PN-07).

All tests in this module are currently marked skip because the JIRA MCP
server tooling has not yet been configured for the project.

To enable: remove ``pytest.mark.skip`` from ``pytestmark`` below once
``E2E_ATLASSIAN_URL``, ``E2E_ATLASSIAN_EMAIL``, and ``E2E_MCP_JIRA_TOKEN``
are configured in ``test/e2e_test/.env.e2e``.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest
from pytest_httpserver import HTTPServer

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
    FakeLLM,
    build_dev_agent_against_stubs,
    build_e2e_registry,
    skip_without_llm,
    E2E_WORKFLOW_CONFIG,
)
from test.e2e_test.common.e2e_settings import get_e2e_settings
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


# ===========================================================================
# E2E-PN-01: Dev Agent generates initial plan for a single OPEN issue
# ===========================================================================

@skip_without_llm
class TestDevAgentGeneratesInitialPlan:
    def test_generates_initial_plan_for_open_issue(
        self,
        httpserver: HTTPServer,
        planning_tool_order: None,
    ) -> None:
        """E2E-PN-01 (JIRA): OPEN issue → plan_and_notify_job → plan posted as comment."""
        import src.scheduler.jobs.plan_and_notify as pn_mod
        from src.scheduler.jobs.plan_and_notify import plan_and_notify_job

        stub = MCPStubServer(httpserver)
        url = stub.url

        add_comment_calls: list = []
        transition_calls: list = []

        stub.register_tool("get_issue", lambda args: {
            "key": "PROJ-20",
            "fields": {
                "summary": "Implement OAuth2 login",
                "status": {"name": "OPEN"},
                "description": "Add OAuth2 authentication with Google SSO support.",
            },
        })
        stub.register_tool("add_comment", lambda args: (
            add_comment_calls.append(args) or {"id": "c-plan"}
        ))
        stub.register_tool("transition_issue", lambda args: (
            transition_calls.append(args) or {"ok": True}
        ))
        stub.register_tool("search_issues", lambda args: [
            {"key": "PROJ-20", "fields": {"summary": "Implement OAuth2 login",
                                           "status": {"name": "OPEN"}}},
        ])
        stub.register_tool("search_tasks", lambda args: [])
        stub.register_tool("reply_to_thread", lambda args: {"ok": True})
        stub.register_tool("send_message", lambda args: {"ok": True})

        workflow = WorkflowConfig(E2E_PLANNING_WORKFLOW_CONFIG)
        dev_agent = build_dev_agent_against_stubs(
            jira_url=url, slack_url=url, github_url=url, clickup_url=url,
            e2e_settings=get_e2e_settings(),
        )
        registry = build_e2e_registry(dev_agent)

        tracker_registry = _make_stub_tracker_registry(
            open_tickets=[
                TicketRecord(id="PROJ-20", source="jira",
                             title="Implement OAuth2 login", url="", raw_status="OPEN"),
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

        assert len(add_comment_calls) >= 1
        plan_bodies = [
            c.get("comment", c.get("body", c.get("text", "")))
            for c in add_comment_calls
        ]
        has_plan = any(
            "## " in body or "plan" in body.lower() or "development" in body.lower()
            for body in plan_bodies
        )
        assert has_plan, f"Expected plan comment. Bodies: {plan_bodies}"

        in_progress_calls = [
            t for t in transition_calls
            if "IN PROGRESS" in str(t).upper()
        ]
        assert len(in_progress_calls) == 0, f"BR-8 violated: {transition_calls}"

        accepted_calls = [
            t for t in transition_calls
            if "ACCEPTED" in str(t).upper()
        ]
        assert len(accepted_calls) == 0, f"BR-1 violated: {transition_calls}"

        in_planning_calls = [
            t for t in transition_calls
            if "IN PLANNING" in str(t).upper()
        ]
        assert len(in_planning_calls) == 0, f"BR-10 violated: {transition_calls}"

        assert not stub.was_called("create_pull_request"), "BR-8 violated: PR opened"


# ===========================================================================
# E2E-PN-02: Dev Agent generates plans for multiple OPEN issues (batch)
# ===========================================================================

@skip_without_llm
class TestDevAgentBatchPlanning:
    def test_generates_plans_for_multiple_open_issues(
        self,
        httpserver: HTTPServer,
        planning_tool_order: None,
    ) -> None:
        """E2E-PN-02 (JIRA): 3 OPEN issues → plan_and_notify_job → 3 plan comments."""
        import src.scheduler.jobs.plan_and_notify as pn_mod
        from src.scheduler.jobs.plan_and_notify import plan_and_notify_job

        stub = MCPStubServer(httpserver)
        url = stub.url

        add_comment_calls: list = []

        issue_details = {
            "PROJ-30": {"key": "PROJ-30", "fields": {
                "summary": "Feature A", "status": {"name": "OPEN"},
                "description": "Implement Feature A with caching."}},
            "PROJ-31": {"key": "PROJ-31", "fields": {
                "summary": "Feature B", "status": {"name": "OPEN"},
                "description": "Add Feature B as background worker."}},
            "PROJ-32": {"key": "PROJ-32", "fields": {
                "summary": "Feature C", "status": {"name": "OPEN"},
                "description": "Build Feature C API endpoints."}},
        }

        stub.register_tool("get_issue", lambda args: (
            issue_details.get(args.get("issue_key", args.get("key", "")),
                              {"key": "UNKNOWN", "fields": {}})
        ))
        stub.register_tool("add_comment", lambda args: (
            add_comment_calls.append(args) or {"id": f"c-{len(add_comment_calls)}"}
        ))
        stub.register_tool("transition_issue", lambda args: {"ok": True})
        stub.register_tool("search_issues", lambda args: [
            {"key": key, "fields": d["fields"]}
            for key, d in issue_details.items()
        ])
        stub.register_tool("search_tasks", lambda args: [])
        stub.register_tool("reply_to_thread", lambda args: {"ok": True})
        stub.register_tool("send_message", lambda args: {"ok": True})

        workflow = WorkflowConfig(E2E_PLANNING_WORKFLOW_CONFIG)
        dev_agent = build_dev_agent_against_stubs(
            jira_url=url, slack_url=url, github_url=url, clickup_url=url,
            e2e_settings=get_e2e_settings(),
        )
        registry = build_e2e_registry(dev_agent)

        tracker_registry = _make_stub_tracker_registry(
            open_tickets=[
                TicketRecord(id=key, source="jira", title=d["fields"]["summary"],
                             url="", raw_status="OPEN")
                for key, d in issue_details.items()
            ],
        )

        pn_mod._in_planning_tickets.clear()
        pn_mod._plan_comment_watermarks.clear()

        executor = ThreadPoolExecutor(max_workers=3)
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

        assert len(add_comment_calls) >= 3, (
            f"Expected at least 3 plan comments. Got: {len(add_comment_calls)}"
        )
        assert not stub.was_called("create_pull_request"), "BR-8 violated"


# ===========================================================================
# E2E-PN-03: Dispatch guard prevents double-planning the same issue
# ===========================================================================

@skip_without_llm
class TestDispatchGuardPreventsDoublePlanning:
    def test_dispatch_guard_prevents_double_planning(
        self,
        httpserver: HTTPServer,
    ) -> None:
        """E2E-PN-03 (JIRA): _in_planning_tickets pre-seeded → issue NOT re-planned."""
        import src.scheduler.jobs.plan_and_notify as pn_mod
        from src.scheduler.jobs.plan_and_notify import plan_and_notify_job

        stub = MCPStubServer(httpserver)
        url = stub.url

        add_comment_calls: list = []

        stub.register_tool("get_issue", lambda args: {
            "key": "PROJ-20",
            "fields": {"summary": "Implement OAuth2 login",
                       "status": {"name": "OPEN"},
                       "description": "OAuth2 SSO."},
        })
        stub.register_tool("add_comment", lambda args: (
            add_comment_calls.append(args) or {"id": "c-plan"}
        ))
        stub.register_tool("transition_issue", lambda args: {"ok": True})
        stub.register_tool("search_issues", lambda args: [
            {"key": "PROJ-20", "fields": {"summary": "Implement OAuth2 login",
                                           "status": {"name": "OPEN"}}},
        ])
        stub.register_tool("search_tasks", lambda args: [])

        workflow = WorkflowConfig(E2E_PLANNING_WORKFLOW_CONFIG)
        dev_agent = build_dev_agent_against_stubs(
            jira_url=url, slack_url=url, github_url=url, clickup_url=url,
            e2e_settings=get_e2e_settings(),
        )
        registry = build_e2e_registry(dev_agent)

        tracker_registry = _make_stub_tracker_registry(
            open_tickets=[
                TicketRecord(id="PROJ-20", source="jira",
                             title="Implement OAuth2 login", url="", raw_status="OPEN"),
            ],
        )

        pn_mod._in_planning_tickets.clear()
        pn_mod._in_planning_tickets.add("PROJ-20")
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
        httpserver: HTTPServer,
        planning_tool_order: None,
    ) -> None:
        """E2E-PN-04 (JIRA): IN PLANNING issue + human comment → revised plan posted."""
        import src.scheduler.jobs.plan_and_notify as pn_mod
        from src.scheduler.jobs.plan_and_notify import plan_and_notify_job

        stub = MCPStubServer(httpserver)
        url = stub.url

        add_comment_calls: list = []
        transition_calls: list = []

        stub.register_tool("get_issue", lambda args: {
            "key": "PROJ-21",
            "fields": {
                "summary": "DB migration tool",
                "status": {"name": "IN PLANNING"},
                "description": "Build a tool to run database migrations safely.",
            },
        })
        stub.register_tool("add_comment", lambda args: (
            add_comment_calls.append(args) or {"id": "c-rev"}
        ))
        stub.register_tool("transition_issue", lambda args: (
            transition_calls.append(args) or {"ok": True}
        ))
        stub.register_tool("search_issues", lambda args: [])
        stub.register_tool("search_tasks", lambda args: [])
        stub.register_tool("reply_to_thread", lambda args: {"ok": True})
        stub.register_tool("send_message", lambda args: {"ok": True})

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
            source="jira",
        )

        tracker_registry = _make_stub_tracker_registry(
            in_planning_tickets=[
                TicketRecord(id="PROJ-21", source="jira",
                             title="DB migration tool", url="", raw_status="IN PLANNING"),
            ],
            comments_by_ticket={"PROJ-21": [human_comment]},
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

        assert len(add_comment_calls) >= 1
        plan_bodies = [
            c.get("comment", c.get("body", c.get("text", "")))
            for c in add_comment_calls
        ]
        has_rollback = any(
            "rollback" in body.lower() or "migration" in body.lower()
            or "failure" in body.lower() or "revision" in body.lower()
            for body in plan_bodies
        )
        assert has_rollback, f"Expected rollback mention. Bodies: {plan_bodies}"

        in_planning_calls = [t for t in transition_calls if "IN PLANNING" in str(t).upper()]
        assert len(in_planning_calls) == 0, f"BR-10 violated: {transition_calls}"

        accepted_calls = [t for t in transition_calls if "ACCEPTED" in str(t).upper()]
        assert len(accepted_calls) == 0, f"BR-1 violated: {transition_calls}"

        assert not stub.was_called("create_pull_request"), "BR-8 violated"


# ===========================================================================
# E2E-PN-05: No revision dispatched when no new comments since watermark
# ===========================================================================

@skip_without_llm
class TestNoRevisionWhenNoNewComments:
    def test_no_revision_when_no_new_comments(
        self,
        httpserver: HTTPServer,
    ) -> None:
        """E2E-PN-05 (JIRA): Watermark newer than comment → no revision dispatched."""
        import src.scheduler.jobs.plan_and_notify as pn_mod
        from src.scheduler.jobs.plan_and_notify import plan_and_notify_job

        stub = MCPStubServer(httpserver)
        url = stub.url

        add_comment_calls: list = []

        stub.register_tool("get_issue", lambda args: {
            "key": "PROJ-21",
            "fields": {"summary": "DB migration tool",
                       "status": {"name": "IN PLANNING"},
                       "description": "Build a migration tool."},
        })
        stub.register_tool("add_comment", lambda args: (
            add_comment_calls.append(args) or {"id": "c-rev"}
        ))
        stub.register_tool("transition_issue", lambda args: {"ok": True})
        stub.register_tool("search_issues", lambda args: [])
        stub.register_tool("search_tasks", lambda args: [])

        workflow = WorkflowConfig(E2E_PLANNING_WORKFLOW_CONFIG)
        dev_agent = build_dev_agent_against_stubs(
            jira_url=url, slack_url=url, github_url=url, clickup_url=url,
            e2e_settings=get_e2e_settings(),
        )
        registry = build_e2e_registry(dev_agent)

        old_comment = TicketComment(
            id="c-old",
            author="alice",
            body="Please clarify rollback.",
            created_at=time.time() - 120,
            source="jira",
        )

        tracker_registry = _make_stub_tracker_registry(
            in_planning_tickets=[
                TicketRecord(id="PROJ-21", source="jira",
                             title="DB migration tool", url="", raw_status="IN PLANNING"),
            ],
            comments_by_ticket={"PROJ-21": [old_comment]},
        )

        pn_mod._in_planning_tickets.clear()
        pn_mod._plan_comment_watermarks.clear()
        pn_mod._plan_comment_watermarks["PROJ-21"] = time.time() - 60

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
        httpserver: HTTPServer,
        planning_tool_order: None,
    ) -> None:
        """E2E-PN-06 (JIRA): Plan comment contains human review notification."""
        import src.scheduler.jobs.plan_and_notify as pn_mod
        from src.scheduler.jobs.plan_and_notify import plan_and_notify_job

        stub = MCPStubServer(httpserver)
        url = stub.url

        add_comment_calls: list = []

        stub.register_tool("get_issue", lambda args: {
            "key": "PROJ-20",
            "fields": {"summary": "Implement OAuth2 login",
                       "status": {"name": "OPEN"},
                       "description": "Add OAuth2 authentication with Google SSO support."},
        })
        stub.register_tool("add_comment", lambda args: (
            add_comment_calls.append(args) or {"id": "c-plan"}
        ))
        stub.register_tool("transition_issue", lambda args: {"ok": True})
        stub.register_tool("search_issues", lambda args: [
            {"key": "PROJ-20", "fields": {"summary": "OAuth2",
                                           "status": {"name": "OPEN"}}},
        ])
        stub.register_tool("search_tasks", lambda args: [])
        stub.register_tool("reply_to_thread", lambda args: {"ok": True})
        stub.register_tool("send_message", lambda args: {"ok": True})

        workflow = WorkflowConfig(E2E_PLANNING_WORKFLOW_CONFIG)
        dev_agent = build_dev_agent_against_stubs(
            jira_url=url, slack_url=url, github_url=url, clickup_url=url,
            e2e_settings=get_e2e_settings(),
        )
        registry = build_e2e_registry(dev_agent)

        tracker_registry = _make_stub_tracker_registry(
            open_tickets=[
                TicketRecord(id="PROJ-20", source="jira",
                             title="Implement OAuth2 login", url="", raw_status="OPEN"),
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

        assert len(add_comment_calls) >= 1
        plan_bodies = [
            c.get("comment", c.get("body", c.get("text", "")))
            for c in add_comment_calls
        ]
        has_notification = any(
            any(kw in body.lower() for kw in (
                "please review", "ready for review", "review", "feedback",
                "@", "human", "engineer", "notify",
            ))
            for body in plan_bodies
        )
        assert has_notification, (
            f"Expected plan comment to include human notification. Bodies: {plan_bodies}"
        )


# ===========================================================================
# E2E-PN-07: Full planning loop (JIRA)
# ===========================================================================

@skip_without_llm
class TestFullPlanningLoop:
    def test_full_planning_loop(
        self,
        httpserver: HTTPServer,
        planning_tool_order: None,
        fake_llm_session: Optional[FakeLLM],
    ) -> None:
        """E2E-PN-07 (JIRA): OPEN → initial plan → IN PLANNING → revision → no BR violations."""
        import src.scheduler.jobs.plan_and_notify as pn_mod
        from src.scheduler.jobs.plan_and_notify import plan_and_notify_job

        stub = MCPStubServer(httpserver)
        url = stub.url

        add_comment_calls_run1: list = []
        add_comment_calls_run2: list = []
        in_progress_calls: list = []
        accepted_write_calls: list = []
        in_planning_write_calls: list = []
        pr_calls: list = []

        workflow = WorkflowConfig(E2E_PLANNING_WORKFLOW_CONFIG)
        dev_agent = build_dev_agent_against_stubs(
            jira_url=url, slack_url=url, github_url=url, clickup_url=url,
            e2e_settings=get_e2e_settings(),
        )
        registry = build_e2e_registry(dev_agent)

        # ── RUN 1: OPEN → initial plan ────────────────────────────────────
        stub.register_tool("get_issue", lambda args: {
            "key": "PROJ-22",
            "fields": {"summary": "Search indexing service",
                       "status": {"name": "OPEN"},
                       "description": "Build Elasticsearch search indexing."},
        })
        stub.register_tool("add_comment", lambda args: (
            add_comment_calls_run1.append(args) or {"id": f"c-r1-{len(add_comment_calls_run1)}"}
        ))

        def _transition_run1(args: dict) -> dict:
            s = str(args).upper()
            if "IN PROGRESS" in s:
                in_progress_calls.append(args)
            if "ACCEPTED" in s:
                accepted_write_calls.append(args)
            if "IN PLANNING" in s:
                in_planning_write_calls.append(args)
            return {"ok": True}

        stub.register_tool("transition_issue", _transition_run1)
        stub.register_tool("create_pull_request", lambda args: (
            pr_calls.append(args) or {"html_url": "", "number": 0}
        ))
        stub.register_tool("search_issues", lambda args: [
            {"key": "PROJ-22", "fields": {"summary": "Search indexing",
                                           "status": {"name": "OPEN"}}},
        ])
        stub.register_tool("search_tasks", lambda args: [])
        stub.register_tool("reply_to_thread", lambda args: {"ok": True})
        stub.register_tool("send_message", lambda args: {"ok": True})

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
                        TicketRecord(id="PROJ-22", source="jira",
                                     title="Search indexing service", url="", raw_status="OPEN"),
                    ],
                ),
            )
            executor.shutdown(wait=True)
        finally:
            executor.shutdown(wait=False)

        assert len(add_comment_calls_run1) >= 1, "Expected initial plan in run 1"
        assert len(pr_calls) == 0, "BR-8 violated in run 1"
        assert len(in_progress_calls) == 0, "BR-8 violated in run 1"
        assert len(accepted_write_calls) == 0, "BR-1 violated in run 1"
        assert len(in_planning_write_calls) == 0, "BR-10 violated in run 1"

        # Reset FakeLLM turn counters so run 2 starts fresh
        if fake_llm_session is not None:
            fake_llm_session.reset_turns()

        # ── RUN 2: IN PLANNING + human comment → revision ─────────────────
        human_comment = TicketComment(
            id="c-human-j",
            author="alice",
            body="Please add failure handling and retry logic for index errors.",
            created_at=time.time() - 30,
            source="jira",
        )

        stub.register_tool("add_comment", lambda args: (
            add_comment_calls_run2.append(args) or {"id": f"c-r2-{len(add_comment_calls_run2)}"}
        ))
        stub.register_tool("get_issue", lambda args: {
            "key": "PROJ-22",
            "fields": {"summary": "Search indexing service",
                       "status": {"name": "IN PLANNING"},
                       "description": "Build Elasticsearch search indexing."},
        })

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            plan_and_notify_job(
                registry=registry,
                settings=_make_settings(),
                executor=executor,
                workflow=workflow,
                tracker_registry=_make_stub_tracker_registry(
                    in_planning_tickets=[
                        TicketRecord(id="PROJ-22", source="jira",
                                     title="Search indexing service",
                                     url="", raw_status="IN PLANNING"),
                    ],
                    comments_by_ticket={"PROJ-22": [human_comment]},
                ),
            )
            executor.shutdown(wait=True)
        finally:
            executor.shutdown(wait=False)
            pn_mod._in_planning_tickets.clear()
            pn_mod._plan_comment_watermarks.clear()

        total_comments = len(add_comment_calls_run1) + len(add_comment_calls_run2)
        assert total_comments >= 2, (
            f"Expected at least 2 total comments. "
            f"run1={len(add_comment_calls_run1)}, run2={len(add_comment_calls_run2)}"
        )
        assert len(pr_calls) == 0, "BR-8 violated in run 2"
        assert len(in_planning_write_calls) == 0, "BR-10 violated in run 2"
        assert len(accepted_write_calls) == 0, "BR-1 violated across all runs"

