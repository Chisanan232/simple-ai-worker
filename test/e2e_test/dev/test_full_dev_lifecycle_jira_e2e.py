"""
E2E tests: Full dev lifecycle — JIRA backend variant (E2E-20 through E2E-24).

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
    build_dev_agent_against_stubs,
    build_e2e_registry,
    skip_without_llm,
    E2E_WORKFLOW_CONFIG,
)
from src.ticket.models import PRRecord
from src.ticket.workflow import WorkflowConfig


def _make_e2e_settings(timeout: int = 300) -> Any:
    s = MagicMock()
    s.PR_AUTO_MERGE_TIMEOUT_SECONDS = timeout
    s.PR_REVIEW_COMMENT_CHECK_INTERVAL_SECONDS = 120
    s.MAX_CONCURRENT_DEV_AGENTS = 1
    return s


def _pre_populate_pr(ticket_id: str, pr_url: str, age_seconds: float = 310.0) -> None:
    import src.scheduler.jobs.scan_tickets as scan_mod
    scan_mod._open_prs[ticket_id] = PRRecord(
        ticket_id=ticket_id,
        pr_url=pr_url,
        opened_at_utc=time.time() - age_seconds,
    )
    scan_mod._prs_under_review[ticket_id] = pr_url


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
        executor.shutdown(wait=True)
    finally:
        executor.shutdown(wait=False)


# ===========================================================================
# E2E-20: Full path S1 → COMPLETE with auto-merge (JIRA)
# ===========================================================================

@skip_without_llm
class TestFullPathS1ToComplete:
    def test_full_path_s1_to_complete_auto_merge(
        self,
        httpserver: HTTPServer,
    ) -> None:
        """E2E-20 (JIRA): Thread summary → ACCEPTED → dev → PR → approve → auto-merge → COMPLETE."""
        import src.scheduler.jobs.scan_tickets as scan_mod
        from src.scheduler.jobs.scan_tickets import scan_and_dispatch_job
        from src.scheduler.jobs.pr_merge_watcher import pr_merge_watcher_job

        stub = MCPStubServer(httpserver)
        url = stub.url
        workflow = WorkflowConfig(E2E_WORKFLOW_CONFIG)

        accepted_write_calls: list = []
        merge_calls: list = []
        pr_url = "https://github.com/org/repo/pull/999"
        pr_opened = [False]

        stub.register_tool("get_messages", lambda args: {
            "ok": True,
            "messages": [
                {"user": "U1", "text": "Discuss PROJ-20 — use OAuth2.", "ts": "1.1"},
                {"user": "U2", "text": "Agreed. Ticket: PROJ-20.", "ts": "1.2"},
            ],
        })
        stub.register_tool("reply_to_thread", lambda args: {"ok": True})
        stub.register_tool("send_message", lambda args: {"ok": True})
        stub.register_tool("search_issues", lambda args: [
            {"key": "PROJ-20", "fields": {"summary": "OAuth2 login",
                                           "status": {"name": "ACCEPTED"}}},
        ])
        stub.register_tool("search_tasks", lambda args: [])
        stub.register_tool("get_issue", lambda args: {
            "key": "PROJ-20",
            "fields": {"summary": "OAuth2 login", "status": {"name": "ACCEPTED"},
                       "description": "Implement OAuth2 with Google SSO."},
        })
        stub.register_tool("add_comment", lambda args: {"id": "c1"})

        def _transition(args: dict) -> dict:
            if "ACCEPTED" in str(args).upper():
                accepted_write_calls.append(args)
            return {"ok": True}

        stub.register_tool("transition_issue", _transition)
        stub.register_tool("create_pull_request", lambda args: (
            pr_opened.__setitem__(0, True) or {"html_url": pr_url, "number": 999}
        ))
        stub.register_tool("get_pull_request", lambda args: {
            "merged": False, "is_merged": False, "approval_count": 1,
        })
        stub.register_tool("get_pull_request_reviews", lambda args: [
            {"state": "APPROVED"},
        ])
        stub.register_tool("merge_pull_request", lambda args: (
            merge_calls.append(args) or {"merged": True, "sha": "abc"}
        ))

        dev_agent = build_dev_agent_against_stubs(
            jira_url=url, slack_url=url, github_url=url, clickup_url=url,
            e2e_settings=get_e2e_settings(),
        )
        registry = build_e2e_registry(dev_agent)

        _run_dev_handler_sync(
            event={
                "text": "<@UBOT> [dev] read this thread",
                "channel": "C010",
                "thread_ts": "100.100",
                "ts": "100.100",
            },
            registry=registry,
        )

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            scan_and_dispatch_job(
                registry=registry,
                settings=_make_e2e_settings(),
                executor=executor,
                workflow=workflow,
            )
            executor.shutdown(wait=True)
        finally:
            executor.shutdown(wait=False)

        if not scan_mod._open_prs:
            _pre_populate_pr("PROJ-20", pr_url, age_seconds=310)

        pr_merge_watcher_job(
            registry=registry,
            settings=_make_e2e_settings(timeout=300),
            executor=ThreadPoolExecutor(max_workers=1),
            workflow=workflow,
        )

        assert len(accepted_write_calls) == 0, (
            f"BR-1 VIOLATED: ACCEPTED was written by AI. Calls: {accepted_write_calls}"
        )


# ===========================================================================
# E2E-21: Full path — user merges before timeout (JIRA)
# ===========================================================================

@skip_without_llm
class TestFullPathUserMergeBeforeTimeout:
    def test_full_path_user_merge_before_timeout(
        self,
        httpserver: HTTPServer,
    ) -> None:
        """E2E-21 (JIRA): Dev opens PR → user merges before timeout → COMPLETE transition."""
        import src.scheduler.jobs.scan_tickets as scan_mod
        from src.scheduler.jobs.scan_tickets import scan_and_dispatch_job
        from src.scheduler.jobs.pr_merge_watcher import pr_merge_watcher_job

        stub = MCPStubServer(httpserver)
        url = stub.url
        workflow = WorkflowConfig(E2E_WORKFLOW_CONFIG)

        pr_url = "https://github.com/org/repo/pull/998"
        merge_calls: list = []

        stub.register_tool("search_issues", lambda args: [
            {"key": "PROJ-21", "fields": {"summary": "Auth feature",
                                           "status": {"name": "ACCEPTED"}}},
        ])
        stub.register_tool("search_tasks", lambda args: [])
        stub.register_tool("get_issue", lambda args: {
            "key": "PROJ-21",
            "fields": {"summary": "Auth feature", "status": {"name": "ACCEPTED"},
                       "description": "OAuth2."},
        })
        stub.register_tool("transition_issue", lambda args: {"ok": True})
        stub.register_tool("create_pull_request", lambda args: {
            "html_url": pr_url, "number": 998,
        })
        stub.register_tool("add_comment", lambda args: {"id": "c1"})
        stub.register_tool("send_message", lambda args: {"ok": True})
        stub.register_tool("get_pull_request", lambda args: {
            "merged": True, "is_merged": True, "approval_count": 1,
        })
        stub.register_tool("merge_pull_request", lambda args: (
            merge_calls.append(args) or {"merged": True}
        ))

        dev_agent = build_dev_agent_against_stubs(
            jira_url=url, slack_url=url, github_url=url, clickup_url=url,
            e2e_settings=get_e2e_settings(),
        )
        registry = build_e2e_registry(dev_agent)

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            scan_and_dispatch_job(
                registry=registry,
                settings=_make_e2e_settings(),
                executor=executor,
                workflow=workflow,
            )
            executor.shutdown(wait=True)
        finally:
            executor.shutdown(wait=False)

        if not scan_mod._open_prs:
            _pre_populate_pr("PROJ-21", pr_url, age_seconds=60)

        pr_merge_watcher_job(
            registry=registry,
            settings=_make_e2e_settings(timeout=300),
            executor=ThreadPoolExecutor(max_workers=1),
            workflow=workflow,
        )

        assert len(merge_calls) == 0, "Already-merged PR must not be re-merged"


# ===========================================================================
# E2E-22: Full path with review comment fix cycle (JIRA)
# ===========================================================================

@skip_without_llm
class TestFullPathWithReviewCommentFix:
    def test_full_path_with_review_comment_fix_cycle(
        self,
        httpserver: HTTPServer,
    ) -> None:
        """E2E-22 (JIRA): Dev opens PR → reviewer requests changes → no self-approve."""
        from src.scheduler.jobs.pr_review_comment_handler import pr_review_comment_handler_job

        stub = MCPStubServer(httpserver)
        url = stub.url
        workflow = WorkflowConfig(E2E_WORKFLOW_CONFIG)

        pr_url = "https://github.com/org/repo/pull/997"
        approve_calls: list = []

        _pre_populate_pr("PROJ-22", pr_url, age_seconds=310)
        import src.scheduler.jobs.scan_tickets as scan_mod
        scan_mod._prs_under_review["PROJ-22"] = pr_url

        stub.register_tool("get_pull_request_reviews", lambda args: [
            {"state": "CHANGES_REQUESTED", "user": {"login": "reviewer1"}},
        ])
        stub.register_tool("get_pull_request_comments", lambda args: [
            {"id": "c1", "body": "Add error handling", "path": "main.py",
             "line": 10, "resolved": False},
        ])
        stub.register_tool("reply_to_review_comment", lambda args: {"ok": True})

        def _submit_review(args: dict) -> dict:
            if "APPROVE" in str(args).upper():
                approve_calls.append(args)
            return {"ok": True}

        stub.register_tool("approve_pull_request", _submit_review)
        stub.register_tool("submit_review", _submit_review)
        stub.register_tool("transition_issue", lambda args: {"ok": True})
        stub.register_tool("send_message", lambda args: {"ok": True})

        dev_agent = build_dev_agent_against_stubs(
            jira_url=url, slack_url=url, github_url=url, clickup_url=url,
            e2e_settings=get_e2e_settings(),
        )
        registry = build_e2e_registry(dev_agent)

        pr_review_comment_handler_job(
            registry=registry,
            settings=_make_e2e_settings(timeout=300),
            executor=ThreadPoolExecutor(max_workers=1),
            workflow=workflow,
        )

        assert len(approve_calls) == 0, (
            f"Dev Agent must NOT self-approve PRs. Got: {approve_calls}"
        )


# ===========================================================================
# E2E-23: Ask and wait when no ticket ID in thread (BR-6, JIRA)
# ===========================================================================

@skip_without_llm
class TestAskAndWaitWhenNoTicketId:
    def test_ask_and_wait_when_no_ticket_id(
        self,
        httpserver: HTTPServer,
    ) -> None:
        """E2E-23 (JIRA): No ticket ID in thread → ask, add_comment NOT called."""
        stub = MCPStubServer(httpserver)
        url = stub.url

        add_comment_calls: list = []
        reply_calls: list = []

        stub.register_tool("get_messages", lambda args: {
            "ok": True,
            "messages": [
                {"user": "U1", "text": "We decided to use Redis.", "ts": "1.1"},
                {"user": "U2", "text": "Great, Redis Cluster.", "ts": "1.2"},
            ],
        })
        stub.register_tool("add_comment", lambda args: (
            add_comment_calls.append(args) or {"id": "c1"}
        ))
        stub.register_tool("reply_to_thread", lambda args: (
            reply_calls.append(args) or {"ok": True}
        ))

        dev_agent = build_dev_agent_against_stubs(
            jira_url=url, slack_url=url, github_url=url, clickup_url=url,
            e2e_settings=get_e2e_settings(),
        )
        registry = build_e2e_registry(dev_agent)

        _run_dev_handler_sync(
            event={
                "text": "<@UBOT> [dev] read this discussion",
                "channel": "C020",
                "thread_ts": "200.200",
                "ts": "200.200",
            },
            registry=registry,
        )

        assert len(add_comment_calls) == 0, (
            f"add_comment must not be called when no ticket ID found (BR-6). "
            f"Calls: {add_comment_calls}"
        )
        assert len(reply_calls) > 0, (
            "Expected LLM to ask for ticket ID via slack/reply_to_thread (BR-6)"
        )


# ===========================================================================
# E2E-24: REJECTED ticket halts all processing (BR-3, JIRA)
# ===========================================================================

@skip_without_llm
class TestRejectionHaltsProcessing:
    def test_rejection_halts_all_processing(
        self,
        httpserver: HTTPServer,
    ) -> None:
        """E2E-24 (JIRA): REJECTED ticket → no dispatch, no PR, not in watcher dicts."""
        import src.scheduler.jobs.scan_tickets as scan_mod
        from src.scheduler.jobs.scan_tickets import scan_and_dispatch_job

        stub = MCPStubServer(httpserver)
        url = stub.url
        workflow = WorkflowConfig(E2E_WORKFLOW_CONFIG)

        pr_calls: list = []

        stub.register_tool("search_issues", lambda args: [
            {"key": "PROJ-23", "fields": {"summary": "Cancelled work",
                                           "status": {"name": "REJECTED"}}},
        ])
        stub.register_tool("search_tasks", lambda args: [])
        stub.register_tool("transition_issue", lambda args: {"ok": True})
        stub.register_tool("create_pull_request", lambda args: (
            pr_calls.append(args) or {"html_url": "", "number": 0}
        ))

        dev_agent = build_dev_agent_against_stubs(
            jira_url=url, slack_url=url, github_url=url, clickup_url=url,
            e2e_settings=get_e2e_settings(),
        )
        registry = build_e2e_registry(dev_agent)
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            scan_and_dispatch_job(
                registry=registry,
                settings=_make_e2e_settings(),
                executor=executor,
                workflow=workflow,
            )
            executor.shutdown(wait=True)
        finally:
            executor.shutdown(wait=False)

        assert len(pr_calls) == 0, (
            f"No PR must be created for REJECTED ticket. Got: {pr_calls}"
        )
        assert "PROJ-23" not in scan_mod._open_prs
        assert "PROJ-23" not in scan_mod._prs_under_review


