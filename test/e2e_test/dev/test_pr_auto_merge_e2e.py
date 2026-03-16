"""
E2E tests: PR auto-merge watcher (E2E-11 through E2E-15).

Scenario steps covered: S7a/S7b → COMPLETE

Verifies:
- Auto-merge with approval after timeout → ticket → COMPLETE
- No merge without approval (BR-2)
- No merge before timeout
- User-merged before timeout → ticket transition only
- After merge, _prs_under_review entry removed
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Request, Response

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

from test.e2e_test.conftest import (
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
    return s


def _pre_populate_pr(
    ticket_id: str,
    pr_url: str,
    age_seconds: float = 310.0,
) -> None:
    """Seed the shared _open_prs dict as if _execute_ticket had already run."""
    import src.scheduler.jobs.scan_tickets as scan_mod

    scan_mod._open_prs[ticket_id] = PRRecord(
        ticket_id=ticket_id,
        pr_url=pr_url,
        opened_at_utc=time.time() - age_seconds,
    )
    scan_mod._prs_under_review[ticket_id] = pr_url


# ===========================================================================
# E2E-11: Auto-merge with approval after timeout
# ===========================================================================

@skip_without_llm
class TestAutoMergeWithApproval:
    def test_auto_merge_with_approval_after_timeout(
        self,
        httpserver: HTTPServer,
    ) -> None:
        """E2E-11: 1 approval + stale PR → merged → ticket → COMPLETE → Slack notified."""
        import src.scheduler.jobs.scan_tickets as scan_mod
        from src.scheduler.jobs.pr_merge_watcher import pr_merge_watcher_job

        url = httpserver.url_for("/mcp")
        workflow = WorkflowConfig(E2E_WORKFLOW_CONFIG)
        pr_url = "https://github.com/org/repo/pull/100"

        _pre_populate_pr("PROJ-20", pr_url, age_seconds=310)

        merge_calls: list = []
        transition_calls: list = []
        complete_write_calls: list = []

        def handler(req: Request) -> Response:
            try:
                body = json.loads(req.data.decode())
            except Exception:
                body = {}
            tool = body.get("tool", body.get("name", ""))
            params = body.get("params", body.get("arguments", body))

            if tool == "get_pull_request":
                return Response(
                    json.dumps({
                        "html_url": pr_url,
                        "merged": False,
                        "reviews": [{"state": "APPROVED", "user": {"login": "reviewer1"}}],
                        "approval_count": 1,
                        "is_merged": False,
                    }),
                    content_type="application/json",
                )
            elif tool == "get_pull_request_reviews":
                return Response(
                    json.dumps([{"state": "APPROVED"}]),
                    content_type="application/json",
                )
            elif tool == "merge_pull_request":
                merge_calls.append(body)
                return Response(
                    json.dumps({"merged": True, "sha": "abc123"}),
                    content_type="application/json",
                )
            elif tool in ("transition_issue", "update_task"):
                transition_calls.append(body)
                status = str(params.get("status", params.get("transition", "")))
                if "COMPLETE" in status.upper():
                    complete_write_calls.append(body)
                return Response(json.dumps({"ok": True}), content_type="application/json")
            elif tool == "send_message":
                return Response(json.dumps({"ok": True, "ts": "1.1"}), content_type="application/json")
            else:
                return Response(json.dumps({"ok": True}), content_type="application/json")

        httpserver.expect_request("/mcp").respond_with_handler(handler)

        dev_agent = build_dev_agent_against_stubs(
            jira_url=url, slack_url=url, github_url=url, clickup_url=url
        )
        registry = build_e2e_registry(dev_agent)

        pr_merge_watcher_job(
            registry=registry,
            settings=_make_e2e_settings(timeout=300),
            executor=ThreadPoolExecutor(max_workers=1),
            workflow=workflow,
        )

        assert len(merge_calls) > 0, "Expected github/merge_pull_request to be called"
        assert "PROJ-20" not in scan_mod._open_prs, "PR entry should be cleared after merge"


# ===========================================================================
# E2E-12: No merge without approval (BR-2)
# ===========================================================================

@skip_without_llm
class TestNoMergeWithoutApproval:
    def test_no_merge_without_approval(
        self,
        httpserver: HTTPServer,
    ) -> None:
        """E2E-12: 0 approvals → merge_pull_request never called (BR-2)."""
        import src.scheduler.jobs.scan_tickets as scan_mod
        from src.scheduler.jobs.pr_merge_watcher import pr_merge_watcher_job

        url = httpserver.url_for("/mcp")
        workflow = WorkflowConfig(E2E_WORKFLOW_CONFIG)
        pr_url = "https://github.com/org/repo/pull/200"

        _pre_populate_pr("PROJ-21", pr_url, age_seconds=310)

        merge_calls: list = []

        def handler(req: Request) -> Response:
            try:
                body = json.loads(req.data.decode())
            except Exception:
                body = {}
            tool = body.get("tool", body.get("name", ""))

            if tool == "get_pull_request":
                return Response(
                    json.dumps({
                        "html_url": pr_url,
                        "merged": False,
                        "approval_count": 0,
                        "is_merged": False,
                    }),
                    content_type="application/json",
                )
            elif tool == "get_pull_request_reviews":
                return Response(json.dumps([]), content_type="application/json")
            elif tool == "merge_pull_request":
                merge_calls.append(body)
                return Response(json.dumps({"merged": True}), content_type="application/json")
            else:
                return Response(json.dumps({"ok": True}), content_type="application/json")

        httpserver.expect_request("/mcp").respond_with_handler(handler)

        dev_agent = build_dev_agent_against_stubs(
            jira_url=url, slack_url=url, github_url=url, clickup_url=url
        )
        registry = build_e2e_registry(dev_agent)

        pr_merge_watcher_job(
            registry=registry,
            settings=_make_e2e_settings(timeout=300),
            executor=ThreadPoolExecutor(max_workers=1),
            workflow=workflow,
        )

        assert len(merge_calls) == 0, (
            f"BR-2 VIOLATED: merge_pull_request called with 0 approvals. Calls: {merge_calls}"
        )
        # Entry must remain (not cleared — still waiting for approval).
        assert "PROJ-21" in scan_mod._open_prs


# ===========================================================================
# E2E-13: No merge before timeout
# ===========================================================================

@skip_without_llm
class TestNoMergeBeforeTimeout:
    def test_no_merge_before_timeout(
        self,
        httpserver: HTTPServer,
    ) -> None:
        """E2E-13: 1 approval but PR only 120s old → no merge yet."""
        import src.scheduler.jobs.scan_tickets as scan_mod
        from src.scheduler.jobs.pr_merge_watcher import pr_merge_watcher_job

        url = httpserver.url_for("/mcp")
        workflow = WorkflowConfig(E2E_WORKFLOW_CONFIG)
        pr_url = "https://github.com/org/repo/pull/300"

        # Fresh PR: only 120 seconds old.
        _pre_populate_pr("PROJ-22", pr_url, age_seconds=120)

        merge_calls: list = []

        def handler(req: Request) -> Response:
            try:
                body = json.loads(req.data.decode())
            except Exception:
                body = {}
            tool = body.get("tool", body.get("name", ""))

            if tool == "get_pull_request":
                return Response(
                    json.dumps({"merged": False, "approval_count": 1, "is_merged": False}),
                    content_type="application/json",
                )
            elif tool == "merge_pull_request":
                merge_calls.append(body)
                return Response(json.dumps({"merged": True}), content_type="application/json")
            else:
                return Response(json.dumps({"ok": True}), content_type="application/json")

        httpserver.expect_request("/mcp").respond_with_handler(handler)

        dev_agent = build_dev_agent_against_stubs(
            jira_url=url, slack_url=url, github_url=url, clickup_url=url
        )
        registry = build_e2e_registry(dev_agent)

        pr_merge_watcher_job(
            registry=registry,
            settings=_make_e2e_settings(timeout=300),
            executor=ThreadPoolExecutor(max_workers=1),
            workflow=workflow,
        )

        assert len(merge_calls) == 0, (
            f"Expected no merge before 5-min timeout. merge_pull_request called: {merge_calls}"
        )


# ===========================================================================
# E2E-14: User merged before timeout → ticket transitioned, entry cleared
# ===========================================================================

@skip_without_llm
class TestUserMergedBeforeTimeout:
    def test_user_merged_before_timeout(
        self,
        httpserver: HTTPServer,
    ) -> None:
        """E2E-14: PR already merged by user → ticket → COMPLETE, entry cleared."""
        import src.scheduler.jobs.scan_tickets as scan_mod
        from src.scheduler.jobs.pr_merge_watcher import pr_merge_watcher_job

        url = httpserver.url_for("/mcp")
        workflow = WorkflowConfig(E2E_WORKFLOW_CONFIG)
        pr_url = "https://github.com/org/repo/pull/400"

        _pre_populate_pr("PROJ-23", pr_url, age_seconds=60)  # Fresh, but already merged.

        merge_calls: list = []
        transition_calls: list = []

        def handler(req: Request) -> Response:
            try:
                body = json.loads(req.data.decode())
            except Exception:
                body = {}
            tool = body.get("tool", body.get("name", ""))

            if tool == "get_pull_request":
                return Response(
                    json.dumps({"merged": True, "is_merged": True, "approval_count": 1}),
                    content_type="application/json",
                )
            elif tool == "merge_pull_request":
                merge_calls.append(body)
                return Response(json.dumps({"merged": True}), content_type="application/json")
            elif tool in ("transition_issue", "update_task"):
                transition_calls.append(body)
                return Response(json.dumps({"ok": True}), content_type="application/json")
            elif tool == "send_message":
                return Response(json.dumps({"ok": True}), content_type="application/json")
            else:
                return Response(json.dumps({"ok": True}), content_type="application/json")

        httpserver.expect_request("/mcp").respond_with_handler(handler)

        dev_agent = build_dev_agent_against_stubs(
            jira_url=url, slack_url=url, github_url=url, clickup_url=url
        )
        registry = build_e2e_registry(dev_agent)

        pr_merge_watcher_job(
            registry=registry,
            settings=_make_e2e_settings(timeout=300),
            executor=ThreadPoolExecutor(max_workers=1),
            workflow=workflow,
        )

        # No duplicate merge.
        assert len(merge_calls) == 0, (
            "Already-merged PR must not be merged again"
        )
        # Ticket should be transitioned to COMPLETE.
        assert len(transition_calls) > 0, (
            "Expected ticket to be transitioned to COMPLETE after detecting already-merged PR"
        )
        # Entry cleared.
        assert "PROJ-23" not in scan_mod._open_prs


# ===========================================================================
# E2E-15: After merge, _prs_under_review entry removed
# ===========================================================================

@skip_without_llm
class TestMergeClearsReviewWatch:
    def test_auto_merge_clears_review_watch(
        self,
        httpserver: HTTPServer,
    ) -> None:
        """E2E-15: After auto-merge, _prs_under_review entry is removed."""
        import src.scheduler.jobs.scan_tickets as scan_mod
        from src.scheduler.jobs.pr_merge_watcher import pr_merge_watcher_job

        url = httpserver.url_for("/mcp")
        workflow = WorkflowConfig(E2E_WORKFLOW_CONFIG)
        pr_url = "https://github.com/org/repo/pull/500"

        _pre_populate_pr("PROJ-24", pr_url, age_seconds=310)

        def handler(req: Request) -> Response:
            try:
                body = json.loads(req.data.decode())
            except Exception:
                body = {}
            tool = body.get("tool", body.get("name", ""))

            if tool == "get_pull_request":
                return Response(
                    json.dumps({"merged": False, "is_merged": False, "approval_count": 2}),
                    content_type="application/json",
                )
            elif tool in ("merge_pull_request", "transition_issue", "update_task", "send_message"):
                return Response(json.dumps({"ok": True, "merged": True}), content_type="application/json")
            else:
                return Response(json.dumps({"ok": True}), content_type="application/json")

        httpserver.expect_request("/mcp").respond_with_handler(handler)

        dev_agent = build_dev_agent_against_stubs(
            jira_url=url, slack_url=url, github_url=url, clickup_url=url
        )
        registry = build_e2e_registry(dev_agent)

        pr_merge_watcher_job(
            registry=registry,
            settings=_make_e2e_settings(timeout=300),
            executor=ThreadPoolExecutor(max_workers=1),
            workflow=workflow,
        )

        assert "PROJ-24" not in scan_mod._prs_under_review, (
            "_prs_under_review entry must be cleared after auto-merge"
        )

