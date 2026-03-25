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

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

from test.e2e_test.common.pr_state_management import populate_under_review
from test.e2e_test.conftest import (
    MCPStubServer,
    build_dev_agent_against_stubs,
    build_e2e_registry,
    skip_without_llm,
)

# ===========================================================================
# E2E-16: Fixes changes-requested and replies to reviewer
# ===========================================================================


@skip_without_llm
class TestFixesChangesRequested:
    def test_fixes_changes_requested_and_replies(
        self,
        mcp_stub: MCPStubServer,
        review_reply_tool_order: None,
        pr_review_settings,
    ) -> None:
        """E2E-16: CHANGES_REQUESTED + 2 comments → fixes committed, comments replied to."""
        from src.scheduler.jobs.pr_review_comment_handler import (
            pr_review_comment_handler_job,
        )

        stub = mcp_stub
        url = stub.url
        pr_url = "https://github.com/org/repo/pull/600"
        populate_under_review("PROJ-30", pr_url)

        reply_calls: list = []
        approve_calls: list = []

        stub.register_tool(
            "get_pull_request_reviews",
            lambda args: [
                {"state": "CHANGES_REQUESTED", "user": {"login": "reviewer1"}},
            ],
        )
        stub.register_tool(
            "get_pull_request_comments",
            lambda args: [
                {"id": "c1", "body": "Add error handling", "path": "main.py", "line": 10, "resolved": False},
                {"id": "c2", "body": "Extract this function", "path": "utils.py", "line": 5, "resolved": False},
            ],
        )

        def _reply(args: dict) -> dict:
            reply_calls.append(args)
            return {"ok": True, "id": "r1"}

        stub.register_tool("reply_to_review_comment", _reply)

        def _approve(args: dict) -> dict:
            if "APPROVE" in str(args).upper():
                approve_calls.append(args)
            return {"ok": True}

        stub.register_tool("approve_pull_request", _approve)
        stub.register_tool("submit_review", _approve)
        stub.register_tool("create_commit", lambda args: {"sha": "def456"})
        stub.register_tool("push_commits", lambda args: {"ok": True})

        dev_agent = build_dev_agent_against_stubs(jira_url=url, slack_url=url, github_url=url, clickup_url=url)
        registry = build_e2e_registry(dev_agent)
        executor = ThreadPoolExecutor(max_workers=1)

        try:
            pr_review_comment_handler_job(
                registry=registry,
                settings=pr_review_settings,
                executor=executor,
            )
            executor.shutdown(wait=True)
        finally:
            executor.shutdown(wait=False)

        # At least one reply should have been posted.
        assert len(reply_calls) > 0, "Expected LLM to call github/reply_to_review_comment for addressed comments"
        # AI must never self-approve.
        assert len(approve_calls) == 0, f"AI must never self-approve a PR. approve calls: {approve_calls}"


# ===========================================================================
# E2E-17: No fix for approved PR with no unresolved comments
# ===========================================================================


@skip_without_llm
class TestNoFixForApprovedPR:
    def test_no_fix_for_approved_pr(
        self,
        mcp_stub: MCPStubServer,
        pr_review_settings,
    ) -> None:
        """E2E-17: APPROVED PR + 0 unresolved comments → no fix crew dispatched."""
        from src.scheduler.jobs.pr_review_comment_handler import (
            pr_review_comment_handler_job,
        )

        stub = mcp_stub
        url = stub.url
        pr_url = "https://github.com/org/repo/pull/700"
        populate_under_review("PROJ-31", pr_url)

        stub.register_tool(
            "get_pull_request_reviews",
            lambda args: [
                {"state": "APPROVED"},
            ],
        )
        stub.register_tool("get_pull_request_comments", lambda args: [])

        dev_agent = build_dev_agent_against_stubs(jira_url=url, slack_url=url, github_url=url, clickup_url=url)
        registry = build_e2e_registry(dev_agent)
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit = MagicMock()

        pr_review_comment_handler_job(
            registry=registry,
            settings=pr_review_settings,
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
        mcp_stub: MCPStubServer,
        pr_review_settings,
    ) -> None:
        """E2E-18: Two rapid triggers for same ticket → fix dispatched at most once."""
        import src.scheduler.jobs.pr_review_comment_handler as handler_mod
        from src.scheduler.jobs.pr_review_comment_handler import (
            pr_review_comment_handler_job,
        )

        stub = mcp_stub
        url = stub.url
        pr_url = "https://github.com/org/repo/pull/800"
        populate_under_review("PROJ-32", pr_url)

        stub.register_tool(
            "get_pull_request_reviews",
            lambda args: [
                {"state": "CHANGES_REQUESTED"},
            ],
        )
        stub.register_tool(
            "get_pull_request_comments",
            lambda args: [
                {"id": "c1", "body": "Fix this", "path": "x.py", "line": 1, "resolved": False},
            ],
        )

        dev_agent = build_dev_agent_against_stubs(jira_url=url, slack_url=url, github_url=url, clickup_url=url)
        registry = build_e2e_registry(dev_agent)
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit = MagicMock()

        # First call.
        pr_review_comment_handler_job(
            registry=registry,
            settings=pr_review_settings,
            executor=executor,
        )

        # Manually simulate the guard being set (as if the first submit is running).
        handler_mod._in_progress_comment_fixes.add("PROJ-32")

        # Second rapid call — should NOT dispatch again.
        pr_review_comment_handler_job(
            registry=registry,
            settings=pr_review_settings,
            executor=executor,
        )

        # executor.submit must be called at most once.
        assert executor.submit.call_count <= 1, (
            f"Fix dispatched more than once for the same ticket. " f"Call count: {executor.submit.call_count}"
        )


# ===========================================================================
# E2E-19: AI never self-approves
# ===========================================================================


@skip_without_llm
class TestAINeverSelfApproves:
    def test_ai_never_self_approves(
        self,
        mcp_stub: MCPStubServer,
        pr_review_settings,
    ) -> None:
        """E2E-19: Full fix cycle — approve_pull_request / APPROVE review NEVER called."""
        from src.scheduler.jobs.pr_review_comment_handler import (
            pr_review_comment_handler_job,
        )

        stub = mcp_stub
        url = stub.url
        pr_url = "https://github.com/org/repo/pull/900"
        populate_under_review("PROJ-33", pr_url)

        approve_calls: list = []

        def _capture_approve(args: dict) -> dict:
            if "APPROVE" in str(args).upper():
                approve_calls.append(args)
            return {"ok": True}

        stub.register_tool(
            "get_pull_request_reviews",
            lambda args: [
                {"state": "CHANGES_REQUESTED"},
            ],
        )
        stub.register_tool(
            "get_pull_request_comments",
            lambda args: [
                {"id": "c1", "body": "Add tests", "path": "test_x.py", "line": 1, "resolved": False},
            ],
        )
        stub.register_tool("reply_to_review_comment", lambda args: {"ok": True, "id": "r1"})
        stub.register_tool("approve_pull_request", _capture_approve)
        stub.register_tool("submit_review", _capture_approve)

        dev_agent = build_dev_agent_against_stubs(jira_url=url, slack_url=url, github_url=url, clickup_url=url)
        registry = build_e2e_registry(dev_agent)
        executor = ThreadPoolExecutor(max_workers=1)

        try:
            pr_review_comment_handler_job(
                registry=registry,
                settings=pr_review_settings,
                executor=executor,
            )
            executor.shutdown(wait=True)
        finally:
            executor.shutdown(wait=False)

        assert len(approve_calls) == 0, f"AI must NEVER self-approve a PR. Approval calls detected: {approve_calls}"
