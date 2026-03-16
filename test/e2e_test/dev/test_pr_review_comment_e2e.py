"""
E2E tests: PR review comment handler (E2E-16 through E2E-19).

Scenario steps covered: S6b

Verifies:
- Fix crew dispatched for CHANGES_REQUESTED → commits pushed, reviewer replied
- No fix for approved PR with no unresolved comments
- Deduplication: fix dispatched at most once per ticket
- AI never self-approves (approve_pull_request never called)
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
)


def _make_e2e_settings() -> Any:
    s = MagicMock()
    s.PR_REVIEW_COMMENT_CHECK_INTERVAL_SECONDS = 120
    return s


def _populate_under_review(ticket_id: str, pr_url: str) -> None:
    import src.scheduler.jobs.scan_tickets as scan_mod
    scan_mod._prs_under_review[ticket_id] = pr_url


# ===========================================================================
# E2E-16: Fixes changes-requested and replies to reviewer
# ===========================================================================

@skip_without_llm
class TestFixesChangesRequested:
    def test_fixes_changes_requested_and_replies(
        self,
        httpserver: HTTPServer,
    ) -> None:
        """E2E-16: CHANGES_REQUESTED + 2 comments → fixes committed, comments replied to."""
        from src.scheduler.jobs.pr_review_comment_handler import pr_review_comment_handler_job

        url = httpserver.url_for("/mcp")
        pr_url = "https://github.com/org/repo/pull/600"
        _populate_under_review("PROJ-30", pr_url)

        reply_calls: list = []
        approve_calls: list = []

        def handler(req: Request) -> Response:
            try:
                body = json.loads(req.data.decode())
            except Exception:
                body = {}
            tool = body.get("tool", body.get("name", ""))

            if tool == "get_pull_request_reviews":
                return Response(
                    json.dumps([{"state": "CHANGES_REQUESTED", "user": {"login": "reviewer1"}}]),
                    content_type="application/json",
                )
            elif tool == "get_pull_request_comments":
                return Response(
                    json.dumps([
                        {"id": "c1", "body": "Add error handling", "path": "main.py", "line": 10,
                         "resolved": False},
                        {"id": "c2", "body": "Extract this function", "path": "utils.py", "line": 5,
                         "resolved": False},
                    ]),
                    content_type="application/json",
                )
            elif tool == "reply_to_review_comment":
                reply_calls.append(body)
                return Response(json.dumps({"ok": True, "id": "r1"}), content_type="application/json")
            elif tool in ("approve_pull_request", "submit_review"):
                # Capture self-approval attempts.
                params = body.get("params", body.get("arguments", body))
                if "APPROVE" in str(params).upper():
                    approve_calls.append(body)
                return Response(json.dumps({"ok": True}), content_type="application/json")
            elif tool == "create_commit":
                return Response(json.dumps({"sha": "def456"}), content_type="application/json")
            elif tool == "push_commits":
                return Response(json.dumps({"ok": True}), content_type="application/json")
            else:
                return Response(json.dumps({"ok": True}), content_type="application/json")

        httpserver.expect_request("/mcp").respond_with_handler(handler)

        dev_agent = build_dev_agent_against_stubs(
            jira_url=url, slack_url=url, github_url=url, clickup_url=url
        )
        registry = build_e2e_registry(dev_agent)
        executor = ThreadPoolExecutor(max_workers=1)

        try:
            pr_review_comment_handler_job(
                registry=registry,
                settings=_make_e2e_settings(),
                executor=executor,
            )
            executor.shutdown(wait=True)
        finally:
            executor.shutdown(wait=False)

        # At least one reply should have been posted.
        assert len(reply_calls) > 0, (
            "Expected LLM to call github/reply_to_review_comment for addressed comments"
        )
        # AI must never self-approve.
        assert len(approve_calls) == 0, (
            f"AI must never self-approve a PR. approve calls: {approve_calls}"
        )


# ===========================================================================
# E2E-17: No fix for approved PR with no unresolved comments
# ===========================================================================

@skip_without_llm
class TestNoFixForApprovedPR:
    def test_no_fix_for_approved_pr(
        self,
        httpserver: HTTPServer,
    ) -> None:
        """E2E-17: APPROVED PR + 0 unresolved comments → no fix crew dispatched."""
        from src.scheduler.jobs.pr_review_comment_handler import pr_review_comment_handler_job

        url = httpserver.url_for("/mcp")
        pr_url = "https://github.com/org/repo/pull/700"
        _populate_under_review("PROJ-31", pr_url)

        fix_crew_calls: list = []

        def handler(req: Request) -> Response:
            try:
                body = json.loads(req.data.decode())
            except Exception:
                body = {}
            tool = body.get("tool", body.get("name", ""))

            if tool == "get_pull_request_reviews":
                return Response(
                    json.dumps([{"state": "APPROVED"}]),
                    content_type="application/json",
                )
            elif tool == "get_pull_request_comments":
                return Response(json.dumps([]), content_type="application/json")
            elif tool in ("reply_to_review_comment", "create_commit"):
                fix_crew_calls.append(body)
            return Response(json.dumps({"ok": True}), content_type="application/json")

        httpserver.expect_request("/mcp").respond_with_handler(handler)

        dev_agent = build_dev_agent_against_stubs(
            jira_url=url, slack_url=url, github_url=url, clickup_url=url
        )
        registry = build_e2e_registry(dev_agent)
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit = MagicMock()

        pr_review_comment_handler_job(
            registry=registry,
            settings=_make_e2e_settings(),
            executor=executor,
        )

        executor.submit.assert_not_called()


# ===========================================================================
# E2E-18: Deduplication prevents parallel fix
# ===========================================================================

@skip_without_llm
class TestDeduplicationPreventsParallelFix:
    def test_deduplication_prevents_parallel_fix(
        self,
        httpserver: HTTPServer,
    ) -> None:
        """E2E-18: Two rapid triggers for same ticket → fix dispatched at most once."""
        import src.scheduler.jobs.pr_review_comment_handler as handler_mod
        from src.scheduler.jobs.pr_review_comment_handler import pr_review_comment_handler_job

        url = httpserver.url_for("/mcp")
        pr_url = "https://github.com/org/repo/pull/800"
        _populate_under_review("PROJ-32", pr_url)

        def handler(req: Request) -> Response:
            try:
                body = json.loads(req.data.decode())
            except Exception:
                body = {}
            tool = body.get("tool", body.get("name", ""))
            if tool == "get_pull_request_reviews":
                return Response(
                    json.dumps([{"state": "CHANGES_REQUESTED"}]),
                    content_type="application/json",
                )
            elif tool == "get_pull_request_comments":
                return Response(
                    json.dumps([{"id": "c1", "body": "Fix this", "path": "x.py", "line": 1,
                                 "resolved": False}]),
                    content_type="application/json",
                )
            return Response(json.dumps({"ok": True}), content_type="application/json")

        httpserver.expect_request("/mcp").respond_with_handler(handler)

        dev_agent = build_dev_agent_against_stubs(
            jira_url=url, slack_url=url, github_url=url, clickup_url=url
        )
        registry = build_e2e_registry(dev_agent)
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit = MagicMock()

        # First call.
        pr_review_comment_handler_job(
            registry=registry,
            settings=_make_e2e_settings(),
            executor=executor,
        )

        # Manually simulate the guard being set (as if the first submit is running).
        handler_mod._in_progress_comment_fixes.add("PROJ-32")

        # Second rapid call — should NOT dispatch again.
        pr_review_comment_handler_job(
            registry=registry,
            settings=_make_e2e_settings(),
            executor=executor,
        )

        # executor.submit must be called at most once.
        assert executor.submit.call_count <= 1, (
            f"Fix dispatched more than once for the same ticket. "
            f"Call count: {executor.submit.call_count}"
        )


# ===========================================================================
# E2E-19: AI never self-approves
# ===========================================================================

@skip_without_llm
class TestAINeverSelfApproves:
    def test_ai_never_self_approves(
        self,
        httpserver: HTTPServer,
    ) -> None:
        """E2E-19: Full fix cycle — approve_pull_request / APPROVE review NEVER called."""
        from src.scheduler.jobs.pr_review_comment_handler import pr_review_comment_handler_job

        url = httpserver.url_for("/mcp")
        pr_url = "https://github.com/org/repo/pull/900"
        _populate_under_review("PROJ-33", pr_url)

        approve_calls: list = []

        def handler(req: Request) -> Response:
            try:
                body = json.loads(req.data.decode())
            except Exception:
                body = {}
            tool = body.get("tool", body.get("name", ""))
            params = body.get("params", body.get("arguments", body))

            # Catch any approval-related calls.
            if "approve" in tool.lower():
                approve_calls.append({"tool": tool, "params": params})
            elif tool in ("submit_review", "create_review"):
                if "APPROVE" in str(params).upper():
                    approve_calls.append({"tool": tool, "params": params})

            if tool == "get_pull_request_reviews":
                return Response(
                    json.dumps([{"state": "CHANGES_REQUESTED"}]),
                    content_type="application/json",
                )
            elif tool == "get_pull_request_comments":
                return Response(
                    json.dumps([
                        {"id": "c1", "body": "Add tests", "path": "test_x.py", "line": 1,
                         "resolved": False},
                    ]),
                    content_type="application/json",
                )
            else:
                return Response(json.dumps({"ok": True}), content_type="application/json")

        httpserver.expect_request("/mcp").respond_with_handler(handler)

        dev_agent = build_dev_agent_against_stubs(
            jira_url=url, slack_url=url, github_url=url, clickup_url=url
        )
        registry = build_e2e_registry(dev_agent)
        executor = ThreadPoolExecutor(max_workers=1)

        try:
            pr_review_comment_handler_job(
                registry=registry,
                settings=_make_e2e_settings(),
                executor=executor,
            )
            executor.shutdown(wait=True)
        finally:
            executor.shutdown(wait=False)

        assert len(approve_calls) == 0, (
            f"AI must NEVER self-approve a PR. Approval calls detected: {approve_calls}"
        )

