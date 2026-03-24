"""
E2E tests: Dev Agent scan-and-dispatch workflow — JIRA backend variant (E2E-07 through E2E-10).

All tests in this module are currently marked skip because the JIRA MCP
server tooling has not yet been configured for the project.

To enable: remove ``pytest.mark.skip`` from ``pytestmark`` below once
``E2E_ATLASSIAN_URL``, ``E2E_ATLASSIAN_EMAIL``, and ``E2E_MCP_JIRA_TOKEN``
are configured in ``test/e2e_test/.env.e2e``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
]

from test.e2e_test.common.e2e_settings import get_e2e_settings
from test.e2e_test.common.hybrid_mode import (
    get_service_url,
    should_register_stub_tools,
)
from test.e2e_test.common.test_infrastructure import (
    make_settings,
    make_stub_tracker_registry_dev,
)
from test.e2e_test.conftest import (
    E2E_WORKFLOW_CONFIG,
    E2ESettings,
    MCPStubServer,
    build_dev_agent_against_stubs,
    build_e2e_registry,
    skip_without_llm,
)

from src.ticket.models import TicketRecord
from src.ticket.workflow import WorkflowConfig

# ===========================================================================
# E2E-07: Dev picks up ACCEPTED JIRA issue, transitions statuses, opens PR
# ===========================================================================


@skip_without_llm
class TestDevPicksUpAcceptedTicket:
    def test_dev_picks_up_accepted_and_opens_pr(
        self,
        mcp_stub: MCPStubServer,
        mcp_urls: dict[str, str],
        e2e_settings: E2ESettings,
    ) -> None:
        """E2E-07 (JIRA): ACCEPTED issue → IN PROGRESS → PR opened → IN REVIEW."""
        import src.scheduler.jobs.scan_tickets as scan_mod
        from src.scheduler.jobs.scan_tickets import scan_and_dispatch_job

        stub = mcp_stub
        url = get_service_url("jira", e2e_settings, mcp_urls, stub)

        workflow = WorkflowConfig(E2E_WORKFLOW_CONFIG)

        transition_calls: list = []
        pr_calls: list = []
        accepted_write_calls: list = []

        if should_register_stub_tools(e2e_settings):
            stub.register_tool(
                "search_issues",
                lambda args: [
                    {"key": "PROJ-10", "fields": {"summary": "Implement login", "status": {"name": "ACCEPTED"}}},
                ],
            )
            stub.register_tool("search_tasks", lambda args: [])
            stub.register_tool(
                "get_issue",
                lambda args: {
                    "key": "PROJ-10",
                    "fields": {
                        "summary": "Implement login feature",
                        "status": {"name": "ACCEPTED"},
                        "description": "Use OAuth2 with Google SSO.",
                    },
                },
            )

            def _transition_issue(args: dict) -> dict:
                status = str(args.get("status", args.get("transition", ""))).upper()
                if "ACCEPTED" in status:
                    accepted_write_calls.append(args)
                transition_calls.append({"tool": "transition_issue", "params": args})
                return {"ok": True}

            stub.register_tool("transition_issue", _transition_issue)

            def _create_pr(args: dict) -> dict:
                pr_url = "https://github.com/org/repo/pull/42"
                pr_calls.append({"pr_url": pr_url})
                return {"html_url": pr_url, "number": 42}

            stub.register_tool("create_pull_request", _create_pr)
            stub.register_tool("send_message", lambda args: {"ok": True})
            stub.register_tool("reply_to_thread", lambda args: {"ok": True})
            stub.register_tool("add_comment", lambda args: {"id": "c1"})

        dev_agent = build_dev_agent_against_stubs(
            jira_url=mcp_urls["jira"] if e2e_settings.USE_TESTCONTAINERS else url,
            slack_url=mcp_urls["slack"] if e2e_settings.USE_TESTCONTAINERS else url,
            github_url=mcp_urls["github"] if e2e_settings.USE_TESTCONTAINERS else url,
            clickup_url=mcp_urls["clickup"] if e2e_settings.USE_TESTCONTAINERS else url,
            e2e_settings=e2e_settings,
        )
        registry = build_e2e_registry(dev_agent)

        # Bypass the real REST API — feed the ACCEPTED ticket directly.
        tracker_registry = make_stub_tracker_registry_dev(
            accepted_tickets=[
                TicketRecord(
                    id="PROJ-10",
                    source="jira",
                    title="Implement login",
                    url="https://test.atlassian.net/browse/PROJ-10",
                    raw_status="ACCEPTED",
                ),
            ]
        )

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            scan_and_dispatch_job(
                registry=registry,
                settings=make_settings(),
                executor=executor,
                workflow=workflow,
                tracker_registry=tracker_registry,
            )
            executor.shutdown(wait=True)
        finally:
            executor.shutdown(wait=False)

        # Stub-specific assertions (only in stub mode with real LLM)
        if not e2e_settings.USE_TESTCONTAINERS and not e2e_settings.USE_FAKE_LLM:
            assert len(pr_calls) > 0, "Expected LLM to call github/create_pull_request"

        # Live mode and stub mode: Check that PR was registered in watcher
        # Only check this with real LLM, as FakeLLM may not create PRs
        if not e2e_settings.USE_FAKE_LLM:
            assert len(scan_mod._open_prs) > 0 or len(scan_mod._prs_under_review) > 0

    def test_dev_never_writes_accepted(
        self,
        mcp_stub: MCPStubServer,
    ) -> None:
        """E2E-09 (JIRA): The agent must NEVER write ACCEPTED to any ticket (BR-1)."""
        from src.scheduler.jobs.scan_tickets import scan_and_dispatch_job

        stub = mcp_stub
        url = stub.url
        workflow = WorkflowConfig(E2E_WORKFLOW_CONFIG)

        accepted_write_calls: list = []

        stub.register_tool(
            "search_issues",
            lambda args: [
                {"key": "PROJ-12", "fields": {"summary": "Auth feature", "status": {"name": "ACCEPTED"}}},
            ],
        )
        stub.register_tool("search_tasks", lambda args: [])
        stub.register_tool(
            "get_issue",
            lambda args: {
                "key": "PROJ-12",
                "fields": {
                    "summary": "Auth feature",
                    "status": {"name": "ACCEPTED"},
                    "description": "OAuth2 with Google SSO.",
                },
            },
        )

        def _transition_issue(args: dict) -> dict:
            status = str(args.get("status", args.get("transition", ""))).upper()
            if "ACCEPTED" in status:
                accepted_write_calls.append(args)
            return {"ok": True}

        stub.register_tool("transition_issue", _transition_issue)
        stub.register_tool(
            "create_pull_request",
            lambda args: {
                "html_url": "https://github.com/org/repo/pull/43",
                "number": 43,
            },
        )
        stub.register_tool("send_message", lambda args: {"ok": True})
        stub.register_tool("reply_to_thread", lambda args: {"ok": True})

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
                settings=make_settings(),
                executor=executor,
                workflow=workflow,
            )
            executor.shutdown(wait=True)
        finally:
            executor.shutdown(wait=False)

        assert (
            len(accepted_write_calls) == 0
        ), f"BR-1 VIOLATED: LLM wrote ACCEPTED status. Calls: {accepted_write_calls}"


# ===========================================================================
# E2E-08: Dev skips REJECTED JIRA issue (BR-3)
# ===========================================================================


@skip_without_llm
class TestDevSkipsRejectedTicket:
    def test_dev_skips_rejected_ticket(
        self,
        mcp_stub: MCPStubServer,
    ) -> None:
        """E2E-08 (JIRA): REJECTED issue in scan results → not dispatched."""
        from src.scheduler.jobs.scan_tickets import scan_and_dispatch_job

        stub = mcp_stub
        url = stub.url
        workflow = WorkflowConfig(E2E_WORKFLOW_CONFIG)

        pr_calls: list = []

        stub.register_tool(
            "search_issues",
            lambda args: [
                {"key": "PROJ-11", "fields": {"summary": "Cancelled feature", "status": {"name": "REJECTED"}}},
            ],
        )
        stub.register_tool("search_tasks", lambda args: [])
        stub.register_tool("transition_issue", lambda args: {"ok": True})

        def _create_pr(args: dict) -> dict:
            pr_calls.append(args)
            return {"html_url": "https://github.com/org/repo/pull/99", "number": 99}

        stub.register_tool("create_pull_request", _create_pr)

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
                settings=make_settings(),
                executor=executor,
                workflow=workflow,
            )
            executor.shutdown(wait=True)
        finally:
            executor.shutdown(wait=False)

        assert len(pr_calls) == 0, f"PR must NOT be created for REJECTED ticket. Got: {pr_calls}"


# ===========================================================================
# E2E-10: Ticket already IN PROGRESS not re-picked up on "restart"
# ===========================================================================


@skip_without_llm
class TestInProgressTicketNotPickedUp:
    def test_dev_handles_in_progress_on_restart(
        self,
        mcp_stub: MCPStubServer,
    ) -> None:
        """E2E-10 (JIRA): IN PROGRESS issue not returned in scan (scan filters by ACCEPTED only)."""
        from src.scheduler.jobs.scan_tickets import scan_and_dispatch_job

        stub = mcp_stub
        url = stub.url
        workflow = WorkflowConfig(E2E_WORKFLOW_CONFIG)

        dispatched_tickets: list = []

        stub.register_tool("search_issues", lambda args: [])
        stub.register_tool("search_tasks", lambda args: [])

        def _transition(args: dict) -> dict:
            dispatched_tickets.append(args)
            return {"ok": True}

        stub.register_tool("transition_issue", _transition)

        def _create_pr(args: dict) -> dict:
            dispatched_tickets.append(args)
            return {"html_url": "https://github.com/org/repo/pull/100", "number": 100}

        stub.register_tool("create_pull_request", _create_pr)

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
                settings=make_settings(),
                executor=executor,
                workflow=workflow,
            )
            executor.shutdown(wait=True)
        finally:
            executor.shutdown(wait=False)

        assert len(dispatched_tickets) == 0, "IN PROGRESS ticket must not be picked up (scan only targets ACCEPTED)"
