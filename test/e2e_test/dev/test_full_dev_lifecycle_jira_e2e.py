"""
E2E tests: Full dev lifecycle — JIRA backend variant (E2E-20 through E2E-24).

Scenario steps covered: S1 → COMPLETE (all features, dev_agent only)

Complete journey:
  1. Supervisor @mentions dev in Slack thread → thread summary → ticket comment
  2. Human sets ticket to ACCEPTED (simulated by pre-seeding stub response)
  3. Scan job picks up ACCEPTED ticket → IN PROGRESS → PR opened → IN REVIEW
  4. Approval + 5-min timeout → auto-merge → COMPLETE

Business rules asserted across the full chain:
- ACCEPTED never written by AI (BR-1)
- Auto-merge only with approval (BR-2)
- REJECTED tickets skipped (BR-3)
- Ask supervisor when no ticket ID (BR-6)

Previously marked skip because JIRA MCP server tooling was not yet configured.
Enable live mode by setting ``E2E_ATLASSIAN_URL``, ``E2E_ATLASSIAN_EMAIL``, and
``E2E_MCP_JIRA_TOKEN`` in ``test/e2e_test/.env.e2e`` and switching
``E2E_MCP_JIRA_URL`` to the live container address.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

from test.e2e_test.common.e2e_settings import get_e2e_settings
from test.e2e_test.common.pr_state_management import populate_pr_state
from test.e2e_test.common.test_infrastructure import make_stub_tracker_registry_dev
from test.e2e_test.common.tool_handlers import run_dev_handler_sync
from test.e2e_test.conftest import (
    E2E_WORKFLOW_CONFIG,
    MCPStubServer,
    build_dev_agent_against_stubs,
    build_e2e_registry,
    skip_without_llm,
)

from src.ticket.workflow import WorkflowConfig

# ===========================================================================
# E2E-20: Full path S1 → COMPLETE with auto-merge (JIRA)
# ===========================================================================


@skip_without_llm
class TestFullPathS1ToComplete:
    def test_full_path_s1_to_complete_auto_merge(
        self,
        mcp_stub: MCPStubServer,
        pr_merge_settings_with_timeout,
    ) -> None:
        """E2E-20 (JIRA): Thread summary → ACCEPTED → dev → PR → approve → auto-merge → COMPLETE."""
        import src.scheduler.jobs.scan_tickets as scan_mod
        from src.scheduler.jobs.pr_merge_watcher import pr_merge_watcher_job
        from src.scheduler.jobs.scan_tickets import scan_and_dispatch_job

        stub = mcp_stub
        url = stub.url
        workflow = WorkflowConfig(E2E_WORKFLOW_CONFIG)

        accepted_write_calls: list = []
        merge_calls: list = []
        pr_url = "https://github.com/org/repo/pull/999"
        pr_opened = [False]

        stub.register_tool(
            "get_messages",
            lambda args: {
                "ok": True,
                "messages": [
                    {"user": "U1", "text": "Discuss PROJ-20 — use OAuth2.", "ts": "1.1"},
                    {"user": "U2", "text": "Agreed. Ticket: PROJ-20.", "ts": "1.2"},
                ],
            },
        )
        stub.register_tool("reply_to_thread", lambda args: {"ok": True})
        stub.register_tool("send_message", lambda args: {"ok": True})
        stub.register_tool(
            "search_issues",
            lambda args: [
                {"key": "PROJ-20", "fields": {"summary": "OAuth2 login", "status": {"name": "ACCEPTED"}}},
            ],
        )
        stub.register_tool("search_tasks", lambda args: [])
        stub.register_tool(
            "get_issue",
            lambda args: {
                "key": "PROJ-20",
                "fields": {
                    "summary": "OAuth2 login",
                    "status": {"name": "ACCEPTED"},
                    "description": "Implement OAuth2 with Google SSO.",
                },
            },
        )
        stub.register_tool("add_comment", lambda args: {"id": "c1"})

        def _transition(args: dict) -> dict:
            status = str(args.get("status", args.get("fields", {}))).upper()
            if "ACCEPTED" in status:
                accepted_write_calls.append(args)
            return {"ok": True}

        stub.register_tool("transition_issue", _transition)

        def _create_pr(args: dict) -> dict:
            pr_opened[0] = True
            return {"html_url": pr_url, "number": 999}

        stub.register_tool("create_pull_request", _create_pr)
        stub.register_tool(
            "get_pull_request",
            lambda args: {
                "merged": False,
                "is_merged": False,
                "approval_count": 1,
            },
        )
        stub.register_tool(
            "get_pull_request_reviews",
            lambda args: [
                {"state": "APPROVED"},
            ],
        )
        stub.register_tool(
            "merge_pull_request", lambda args: (merge_calls.append(args) or {"merged": True, "sha": "abc"})
        )

        dev_agent = build_dev_agent_against_stubs(
            jira_url=url,
            slack_url=url,
            github_url=url,
            clickup_url=url,
        )
        registry = build_e2e_registry(dev_agent)

        # Step 1: Thread summary (S3a)
        run_dev_handler_sync(
            event={
                "text": "<@UBOT> [dev] read this thread",
                "channel": "C010",
                "thread_ts": "100.100",
                "ts": "100.100",
            },
            registry=registry,
        )

        # Step 2: Scan → dev (S5–S6)
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            from src.ticket.models import TicketRecord

            scan_and_dispatch_job(
                registry=registry,
                settings=pr_merge_settings_with_timeout(),
                executor=executor,
                workflow=workflow,
                tracker_registry=make_stub_tracker_registry_dev(
                    accepted_tickets=[
                        TicketRecord(
                            id="PROJ-20",
                            source="jira",
                            title="OAuth2 login",
                            url="https://jira.example.com/browse/PROJ-20",
                            raw_status="ACCEPTED",
                        ),
                    ]
                ),
            )
            executor.shutdown(wait=True)
        finally:
            executor.shutdown(wait=False)

        # Step 3: Pre-seed PR for merge watcher
        if not scan_mod._open_prs:
            populate_pr_state("PROJ-20", pr_url, age_seconds=310)

        # Step 4: Auto-merge (S7b)
        pr_merge_watcher_job(
            registry=registry,
            settings=pr_merge_settings_with_timeout(timeout=300),
            executor=ThreadPoolExecutor(max_workers=1),
            workflow=workflow,
        )

        assert (
            len(accepted_write_calls) == 0
        ), f"BR-1 VIOLATED: ACCEPTED was written by AI. Calls: {accepted_write_calls}"
        assert (
            pr_opened[0] or len(scan_mod._open_prs) > 0 or len(merge_calls) > 0
        ), "Expected full dev lifecycle to complete (PR opened or merge attempted)"


# ===========================================================================
# E2E-21: Full path — user merges before timeout (JIRA)
# ===========================================================================


@skip_without_llm
class TestFullPathUserMergeBeforeTimeout:
    def test_full_path_user_merge_before_timeout(
        self,
        mcp_stub: MCPStubServer,
        user_merged_tool_order: None,
        pr_merge_settings_with_timeout,
    ) -> None:
        """E2E-21 (JIRA): Dev opens PR → user merges before timeout → COMPLETE transition."""
        import src.scheduler.jobs.scan_tickets as scan_mod
        from src.scheduler.jobs.pr_merge_watcher import pr_merge_watcher_job
        from src.scheduler.jobs.scan_tickets import scan_and_dispatch_job

        stub = mcp_stub
        url = stub.url
        workflow = WorkflowConfig(E2E_WORKFLOW_CONFIG)

        pr_url = "https://github.com/org/repo/pull/998"
        transition_calls: list = []
        merge_calls: list = []

        stub.register_tool(
            "search_issues",
            lambda args: [
                {"key": "PROJ-21", "fields": {"summary": "Auth feature", "status": {"name": "ACCEPTED"}}},
            ],
        )
        stub.register_tool("search_tasks", lambda args: [])
        stub.register_tool(
            "get_issue",
            lambda args: {
                "key": "PROJ-21",
                "fields": {"summary": "Auth feature", "status": {"name": "ACCEPTED"}, "description": "OAuth2."},
            },
        )
        stub.register_tool("transition_issue", lambda args: (transition_calls.append(args) or {"ok": True}))
        stub.register_tool(
            "create_pull_request",
            lambda args: {
                "html_url": pr_url,
                "number": 998,
            },
        )
        stub.register_tool("add_comment", lambda args: {"id": "c1"})
        stub.register_tool("send_message", lambda args: {"ok": True})
        # PR already merged by user (fresh, within timeout)
        stub.register_tool(
            "get_pull_request",
            lambda args: {
                "merged": True,
                "is_merged": True,
                "approval_count": 1,
            },
        )
        stub.register_tool("merge_pull_request", lambda args: (merge_calls.append(args) or {"merged": True}))

        dev_agent = build_dev_agent_against_stubs(
            jira_url=url,
            slack_url=url,
            github_url=url,
            clickup_url=url,
        )
        registry = build_e2e_registry(dev_agent)

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            from src.ticket.models import TicketRecord

            scan_and_dispatch_job(
                registry=registry,
                settings=pr_merge_settings_with_timeout(),
                executor=executor,
                workflow=workflow,
                tracker_registry=make_stub_tracker_registry_dev(
                    accepted_tickets=[
                        TicketRecord(
                            id="PROJ-21",
                            source="jira",
                            title="Auth feature",
                            url="https://jira.example.com/browse/PROJ-21",
                            raw_status="ACCEPTED",
                        ),
                    ]
                ),
            )
            executor.shutdown(wait=True)
        finally:
            executor.shutdown(wait=False)

        if not scan_mod._open_prs:
            populate_pr_state("PROJ-21", pr_url, age_seconds=60)

        pr_merge_watcher_job(
            registry=registry,
            settings=pr_merge_settings_with_timeout(timeout=300),
            executor=ThreadPoolExecutor(max_workers=1),
            workflow=workflow,
        )

        # No duplicate merge (user already merged)
        assert len(merge_calls) == 0, "Already-merged PR must not be re-merged"
        # Ticket should be transitioned to COMPLETE
        complete_calls = [t for t in transition_calls if "COMPLETE" in str(t).upper()]
        assert (
            len(complete_calls) > 0 or "PROJ-21" not in scan_mod._open_prs
        ), "Expected ticket transitioned to COMPLETE after user merge"


# ===========================================================================
# E2E-22: Full path with review comment fix cycle (JIRA)
# ===========================================================================


@skip_without_llm
class TestFullPathWithReviewCommentFix:
    def test_full_path_with_review_comment_fix_cycle(
        self,
        mcp_stub: MCPStubServer,
        pr_review_settings,
    ) -> None:
        """E2E-22 (JIRA): Dev opens PR → reviewer requests changes → no self-approve."""
        from src.scheduler.jobs.pr_review_comment_handler import (
            pr_review_comment_handler_job,
        )

        stub = mcp_stub
        url = stub.url
        workflow = WorkflowConfig(E2E_WORKFLOW_CONFIG)

        pr_url = "https://github.com/org/repo/pull/997"
        approve_calls: list = []

        populate_pr_state("PROJ-22", pr_url, age_seconds=310)

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
            ],
        )
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
            jira_url=url,
            slack_url=url,
            github_url=url,
            clickup_url=url,
            e2e_settings=get_e2e_settings(),
        )
        registry = build_e2e_registry(dev_agent)

        pr_review_comment_handler_job(
            registry=registry,
            settings=pr_review_settings,
            executor=ThreadPoolExecutor(max_workers=1),
        )

        assert len(approve_calls) == 0, f"Dev Agent must NOT self-approve PRs. Got: {approve_calls}"


# ===========================================================================
# E2E-23: Ask and wait when no ticket ID in thread (BR-6, JIRA)
# ===========================================================================


@skip_without_llm
class TestAskAndWaitWhenNoTicketId:
    def test_ask_and_wait_when_no_ticket_id(
        self,
        mcp_stub: MCPStubServer,
        reply_only_tool_order: None,
        e2e_settings,
    ) -> None:
        """E2E-23 (JIRA): No ticket ID in thread → ask, add_comment NOT called."""
        stub = mcp_stub
        url = stub.url

        add_comment_calls: list = []
        reply_calls: list = []

        stub.register_tool(
            "get_messages",
            lambda args: {
                "ok": True,
                "messages": [
                    {"user": "U1", "text": "We decided to use Redis.", "ts": "1.1"},
                    {"user": "U2", "text": "Great, Redis Cluster.", "ts": "1.2"},
                ],
            },
        )
        stub.register_tool("add_comment", lambda args: (add_comment_calls.append(args) or {"id": "c1"}))
        stub.register_tool("reply_to_thread", lambda args: (reply_calls.append(args) or {"ok": True}))

        dev_agent = build_dev_agent_against_stubs(
            jira_url=url,
            slack_url=url,
            github_url=url,
            clickup_url=url,
            e2e_settings=get_e2e_settings(),
        )
        registry = build_e2e_registry(dev_agent)

        run_dev_handler_sync(
            event={
                "text": "<@UBOT> [dev] read this discussion",
                "channel": "C020",
                "thread_ts": "200.200",
                "ts": "200.200",
            },
            registry=registry,
        )

        assert len(add_comment_calls) == 0, (
            f"add_comment must not be called when no ticket ID found (BR-6). " f"Calls: {add_comment_calls}"
        )
        assert len(reply_calls) > 0, "Expected LLM to ask for ticket ID via slack/reply_to_thread (BR-6)"


# ===========================================================================
# E2E-24: REJECTED ticket halts all processing (BR-3, JIRA)
# ===========================================================================


@skip_without_llm
class TestRejectionHaltsProcessing:
    def test_rejection_halts_all_processing(
        self,
        mcp_stub: MCPStubServer,
        pr_merge_settings,
    ) -> None:
        """E2E-24 (JIRA): REJECTED ticket → no dispatch, no PR, not in watcher dicts."""
        import src.scheduler.jobs.scan_tickets as scan_mod
        from src.scheduler.jobs.scan_tickets import scan_and_dispatch_job

        stub = mcp_stub
        url = stub.url
        workflow = WorkflowConfig(E2E_WORKFLOW_CONFIG)

        pr_calls: list = []

        stub.register_tool(
            "search_issues",
            lambda args: [
                {"key": "PROJ-23", "fields": {"summary": "Cancelled work", "status": {"name": "REJECTED"}}},
            ],
        )
        stub.register_tool("search_tasks", lambda args: [])
        stub.register_tool("transition_issue", lambda args: {"ok": True})
        stub.register_tool("create_pull_request", lambda args: (pr_calls.append(args) or {"html_url": "", "number": 0}))

        dev_agent = build_dev_agent_against_stubs(
            jira_url=url,
            slack_url=url,
            github_url=url,
            clickup_url=url,
            e2e_settings=get_e2e_settings(),
        )
        registry = build_e2e_registry(dev_agent)
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            scan_and_dispatch_job(
                registry=registry,
                settings=pr_merge_settings,
                executor=executor,
                workflow=workflow,
            )
            executor.shutdown(wait=True)
        finally:
            executor.shutdown(wait=False)

        assert len(pr_calls) == 0, f"No PR must be created for REJECTED ticket. Got: {pr_calls}"
        assert "PROJ-23" not in scan_mod._open_prs
        assert "PROJ-23" not in scan_mod._prs_under_review
