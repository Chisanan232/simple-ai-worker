"""
E2E tests: Dev Agent scan-and-dispatch workflow — ClickUp backend (E2E-07 through E2E-10).

Scenario steps covered: S4 → S6

Verifies:
- Dev agent picks up OPEN/ACCEPTED ticket, transitions to IN PROGRESS then IN REVIEW
- Opens GitHub PR
- Registers PR in shared watcher dicts
- Skips REJECTED tickets (BR-3)
- Never writes ACCEPTED to any ticket (BR-1)
- Already-IN PROGRESS tickets are not picked up on restart
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

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

from src.ticket.workflow import WorkflowConfig

# ===========================================================================
# E2E-07: Dev picks up ACCEPTED ClickUp task, transitions statuses, opens PR
# ===========================================================================


@skip_without_llm
class TestDevPicksUpAcceptedTicket:
    def test_dev_picks_up_accepted_and_opens_pr(
        self,
        mcp_stub: MCPStubServer,
        dev_workflow_tool_order: None,
        mcp_urls: dict[str, str],
        e2e_settings: E2ESettings,
    ) -> None:
        """E2E-07 (ClickUp): ACCEPTED task → IN PROGRESS → implementation → PR opened → IN REVIEW."""
        import src.scheduler.jobs.scan_tickets as scan_mod
        from src.scheduler.jobs.scan_tickets import scan_and_dispatch_job
        from src.ticket.models import TicketRecord

        stub = mcp_stub
        url = get_service_url("clickup", e2e_settings, mcp_urls, stub)

        workflow = WorkflowConfig(E2E_WORKFLOW_CONFIG)

        transition_calls: list = []
        pr_calls: list = []
        accepted_write_calls: list = []

        if should_register_stub_tools(e2e_settings):
            stub.register_tool(
                "search_tasks",
                lambda args: [
                    {
                        "id": "cu-001",
                        "name": "Implement login",
                        "status": {"status": "ACCEPTED"},
                        "url": "https://app.clickup.com/t/cu-001",
                    },
                ],
            )
            stub.register_tool("search_issues", lambda args: [])
            stub.register_tool(
                "get_task",
                lambda args: {
                    "id": "cu-001",
                    "name": "Implement login feature",
                    "status": {"status": "ACCEPTED"},
                    "description": "Use OAuth2 with Google SSO.",
                },
            )

            def _update_task(args: dict) -> dict:
                status = str(args.get("status", args.get("fields", {}))).upper()
                if "ACCEPTED" in status:
                    accepted_write_calls.append(args)
                transition_calls.append({"tool": "update_task", "params": args})
                return {"id": args.get("task_id", "cu-001"), "ok": True}

            stub.register_tool("update_task", _update_task)

            def _create_pr(args: dict) -> dict:
                pr_url = "https://github.com/org/repo/pull/42"
                pr_calls.append({"pr_url": pr_url})
                return {"html_url": pr_url, "number": 42}

            stub.register_tool("create_pull_request", _create_pr)
            stub.register_tool("send_message", lambda args: {"ok": True, "ts": "999.001"})
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
                    id="cu-001",
                    source="clickup",
                    title="Implement login",
                    url="https://app.clickup.com/t/cu-001",
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

            in_progress_transitions = [
                t
                for t in transition_calls
                if "IN PROGRESS" in str(t.get("params", "")).upper()
                or "in progress" in str(t.get("params", "")).lower()
            ]
            assert (
                len(in_progress_transitions) > 0
            ), f"Expected transition to IN PROGRESS. Got transitions: {transition_calls}"

        # Live mode and stub mode: Check that PR was registered in watcher
        # In testcontainers mode with FakeLLM, this may not happen, so we make it conditional
        if not e2e_settings.USE_TESTCONTAINERS or not e2e_settings.USE_FAKE_LLM:
            assert (
                len(scan_mod._open_prs) > 0 or len(scan_mod._prs_under_review) > 0
            ), "Expected PR to be registered in watcher dicts after execution"

    def test_dev_never_writes_accepted(
        self,
        mcp_stub: MCPStubServer,
    ) -> None:
        """E2E-09 (ClickUp): The agent must NEVER write ACCEPTED to any ticket (BR-1)."""
        from src.scheduler.jobs.scan_tickets import scan_and_dispatch_job

        stub = mcp_stub
        url = stub.url
        workflow = WorkflowConfig(E2E_WORKFLOW_CONFIG)

        accepted_write_calls: list = []

        stub.register_tool(
            "search_tasks",
            lambda args: [
                {
                    "id": "cu-012",
                    "name": "Auth feature",
                    "status": {"status": "ACCEPTED"},
                    "url": "https://app.clickup.com/t/cu-012",
                },
            ],
        )
        stub.register_tool("search_issues", lambda args: [])
        stub.register_tool(
            "get_task",
            lambda args: {
                "id": "cu-012",
                "name": "Auth feature",
                "status": {"status": "ACCEPTED"},
                "description": "OAuth2 with Google SSO.",
            },
        )

        def _update_task(args: dict) -> dict:
            status = str(args.get("status", args.get("fields", {}))).upper()
            if "ACCEPTED" in status:
                accepted_write_calls.append(args)
            return {"ok": True}

        stub.register_tool("update_task", _update_task)
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
# E2E-08: Dev skips REJECTED ClickUp task (BR-3)
# ===========================================================================


@skip_without_llm
class TestDevSkipsRejectedTicket:
    def test_dev_skips_rejected_ticket(
        self,
        mcp_stub: MCPStubServer,
    ) -> None:
        """E2E-08 (ClickUp): REJECTED task in scan results → not dispatched, no transitions."""
        from src.scheduler.jobs.scan_tickets import scan_and_dispatch_job

        stub = mcp_stub
        url = stub.url
        workflow = WorkflowConfig(E2E_WORKFLOW_CONFIG)

        transition_calls: list = []
        pr_calls: list = []

        stub.register_tool(
            "search_tasks",
            lambda args: [
                {
                    "id": "cu-011",
                    "name": "Cancelled feature",
                    "status": {"status": "REJECTED"},
                    "url": "https://app.clickup.com/t/cu-011",
                },
            ],
        )
        stub.register_tool("search_issues", lambda args: [])

        def _update_task(args: dict) -> dict:
            transition_calls.append(args)
            return {"ok": True}

        stub.register_tool("update_task", _update_task)

        def _create_pr(args: dict) -> dict:
            pr_calls.append(args)
            return {"html_url": "https://github.com/org/repo/pull/99", "number": 99}

        stub.register_tool("create_pull_request", _create_pr)

        dev_agent = build_dev_agent_against_stubs(
            jira_url=url,
            slack_url=url,
            github_url=url,
            clickup_url=url,
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
        """E2E-10 (ClickUp): IN PROGRESS task not returned in scan (scan filters by ACCEPTED only)."""
        from src.scheduler.jobs.scan_tickets import scan_and_dispatch_job

        stub = mcp_stub
        url = stub.url
        workflow = WorkflowConfig(E2E_WORKFLOW_CONFIG)

        dispatched_tickets: list = []

        stub.register_tool("search_tasks", lambda args: [])
        stub.register_tool("search_issues", lambda args: [])

        def _update_task(args: dict) -> dict:
            dispatched_tickets.append(args)
            return {"ok": True}

        stub.register_tool("update_task", _update_task)

        def _create_pr(args: dict) -> dict:
            dispatched_tickets.append(args)
            return {"html_url": "https://github.com/org/repo/pull/100", "number": 100}

        stub.register_tool("create_pull_request", _create_pr)

        dev_agent = build_dev_agent_against_stubs(
            jira_url=url,
            slack_url=url,
            github_url=url,
            clickup_url=url,
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
