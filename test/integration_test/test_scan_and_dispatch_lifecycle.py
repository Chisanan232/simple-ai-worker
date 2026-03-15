"""
Integration tests for scan_and_dispatch lifecycle (INT-SD-01 through INT-SD-10).

Covers:
- Direct REST API fetch: TrackerRegistry is called, no scanner crew built (INT-SD-01–03)
- REJECTED guard in dispatch loop (BR-3) (INT-SD-07)
- Dev task description injects START_DEVELOPMENT and OPEN_FOR_REVIEW statuses (INT-SD-04–06)
- PR registration in _open_prs and _prs_under_review after execute (INT-SD-08–09)
- Double-dispatch prevention guard (INT-SD-10)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch, call

import pytest

from src.ticket.models import TicketRecord
from src.ticket.rest_client import TicketFetchError
from src.ticket.workflow import WorkflowConfig, WorkflowOperation

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings() -> MagicMock:
    settings = MagicMock()
    settings.MAX_CONCURRENT_DEV_AGENTS = 2
    settings.PR_AUTO_MERGE_TIMEOUT_SECONDS = 300
    return settings


def _make_tracker_registry(
    jira_tickets: list[TicketRecord] | None = None,
    clickup_tickets: list[TicketRecord] | None = None,
    jira_error: Exception | None = None,
    clickup_error: Exception | None = None,
) -> MagicMock:
    """Build a mock TrackerRegistry that returns given tickets or raises errors."""
    registry = MagicMock()

    def _get(source: str) -> MagicMock:
        tracker = MagicMock()
        if source == "jira":
            if jira_error:
                tracker.fetch_tickets_for_operation.side_effect = jira_error
            else:
                tracker.fetch_tickets_for_operation.return_value = jira_tickets or []
        elif source == "clickup":
            if clickup_error:
                tracker.fetch_tickets_for_operation.side_effect = clickup_error
            else:
                tracker.fetch_tickets_for_operation.return_value = clickup_tickets or []
        return tracker

    registry.get.side_effect = _get
    return registry


# ---------------------------------------------------------------------------
# INT-SD-01 through INT-SD-03 — Direct REST API fetch (no scanner crew)
# ---------------------------------------------------------------------------

class TestDirectRestApiFetch:
    def test_calls_tracker_registry_for_jira(
        self, default_workflow_config: WorkflowConfig, mock_dev_registry: MagicMock
    ) -> None:
        """INT-SD-01: scan_and_dispatch_job calls tracker_registry.get('jira') — no scanner crew."""
        from src.scheduler.jobs.scan_tickets import scan_and_dispatch_job

        mock_tr = _make_tracker_registry()
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit = MagicMock()

        scan_and_dispatch_job(
            registry=mock_dev_registry,
            settings=_make_settings(),
            executor=executor,
            workflow=default_workflow_config,
            tracker_registry=mock_tr,
        )

        mock_tr.get.assert_any_call("jira")

    def test_calls_tracker_registry_for_clickup(
        self, default_workflow_config: WorkflowConfig, mock_dev_registry: MagicMock
    ) -> None:
        """INT-SD-02: scan_and_dispatch_job calls tracker_registry.get('clickup') — no scanner crew."""
        from src.scheduler.jobs.scan_tickets import scan_and_dispatch_job

        mock_tr = _make_tracker_registry()
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit = MagicMock()

        scan_and_dispatch_job(
            registry=mock_dev_registry,
            settings=_make_settings(),
            executor=executor,
            workflow=default_workflow_config,
            tracker_registry=mock_tr,
        )

        mock_tr.get.assert_any_call("clickup")

    def test_no_scanner_crew_built(
        self, default_workflow_config: WorkflowConfig, mock_dev_registry: MagicMock
    ) -> None:
        """INT-SD-03: No scanner Task or CrewBuilder.build is called during scan step."""
        from src.scheduler.jobs.scan_tickets import scan_and_dispatch_job

        mock_tr = _make_tracker_registry()
        executor = MagicMock(spec=ThreadPoolExecutor)

        with (
            patch("src.scheduler.jobs.scan_tickets.Task") as mock_task,
            patch("src.scheduler.jobs.scan_tickets.CrewBuilder") as mock_cb,
        ):
            scan_and_dispatch_job(
                registry=mock_dev_registry,
                settings=_make_settings(),
                executor=executor,
                workflow=default_workflow_config,
                tracker_registry=mock_tr,
            )
            # Task and CrewBuilder must NOT be called during the scan step.
            # (They may be called later by _execute_ticket — that's fine,
            # but the scan itself must not invoke them.)
            # With no tickets returned, _execute_ticket is never called.
            mock_task.assert_not_called()
            mock_cb.build.assert_not_called()

    def test_fetch_error_from_one_source_does_not_block_other(
        self, default_workflow_config: WorkflowConfig, mock_dev_registry: MagicMock
    ) -> None:
        """INT-SD-03b: TicketFetchError from jira is logged and clickup still dispatches."""
        from src.scheduler.jobs.scan_tickets import scan_and_dispatch_job

        clickup_ticket = TicketRecord(
            id="cu-1", source="clickup", title="Task A", url="", raw_status="ACCEPTED"
        )
        mock_tr = _make_tracker_registry(
            jira_error=TicketFetchError("JIRA down", source="jira"),
            clickup_tickets=[clickup_ticket],
        )
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit = MagicMock()

        with (
            patch("src.scheduler.jobs.scan_tickets.Task"),
            patch("src.scheduler.jobs.scan_tickets.CrewBuilder") as mock_cb,
        ):
            mock_cb.build.return_value = MagicMock()
            scan_and_dispatch_job(
                registry=mock_dev_registry,
                settings=_make_settings(),
                executor=executor,
                workflow=default_workflow_config,
                tracker_registry=mock_tr,
            )

        # ClickUp ticket must still be dispatched despite JIRA error.
        executor.submit.assert_called_once()
        call_args = executor.submit.call_args
        assert call_args.args[1] == "cu-1"  # ticket_id is the 2nd positional arg

    def test_tickets_from_both_sources_are_all_dispatched(
        self, default_workflow_config: WorkflowConfig, mock_dev_registry: MagicMock
    ) -> None:
        """INT-SD-03c: Tickets from both JIRA and ClickUp are all dispatched."""
        from src.scheduler.jobs.scan_tickets import scan_and_dispatch_job

        jira_ticket = TicketRecord(id="P-1", source="jira", title="J1", url="", raw_status="ACCEPTED")
        clickup_ticket = TicketRecord(id="cu-1", source="clickup", title="C1", url="", raw_status="ACCEPTED")
        mock_tr = _make_tracker_registry(
            jira_tickets=[jira_ticket],
            clickup_tickets=[clickup_ticket],
        )
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit = MagicMock()

        with (
            patch("src.scheduler.jobs.scan_tickets.Task"),
            patch("src.scheduler.jobs.scan_tickets.CrewBuilder") as mock_cb,
        ):
            mock_cb.build.return_value = MagicMock()
            scan_and_dispatch_job(
                registry=mock_dev_registry,
                settings=_make_settings(),
                executor=executor,
                workflow=default_workflow_config,
                tracker_registry=mock_tr,
            )

        assert executor.submit.call_count == 2
        submitted_ids = {c.args[1] for c in executor.submit.call_args_list}
        assert submitted_ids == {"P-1", "cu-1"}


# ---------------------------------------------------------------------------
# INT-SD-04 / INT-SD-05 — Dev task description injects START/REVIEW statuses
# ---------------------------------------------------------------------------

class TestDevTaskDescription:
    def test_dev_task_injects_start_development_status(self) -> None:
        """INT-SD-04: dev task description contains 'Doing' (Team B start_development)."""
        from src.scheduler.jobs.scan_tickets import _build_dev_task_description

        team_b = WorkflowConfig(
            {
                "scan_for_work": {"status_value": "Approved", "human_only": True},
                "skip_rejected": {"status_value": "Cancelled"},
                "start_development": {"status_value": "Doing"},
                "open_for_review": {"status_value": "PR Raised"},
                "mark_complete": {"status_value": "Finished"},
                "update_with_context": {"status_value": ""},
            }
        )
        desc = _build_dev_task_description("TASK-1", "jira", "Fix bug", team_b)
        assert "Doing" in desc
        assert "IN PROGRESS" not in desc  # Must not fall back to hardcoded string

    def test_dev_task_injects_open_for_review_status(self) -> None:
        """INT-SD-05: dev task description contains 'PR Raised' (Team B open_for_review)."""
        from src.scheduler.jobs.scan_tickets import _build_dev_task_description

        team_b = WorkflowConfig(
            {
                "scan_for_work": {"status_value": "Approved", "human_only": True},
                "skip_rejected": {"status_value": "Cancelled"},
                "start_development": {"status_value": "Doing"},
                "open_for_review": {"status_value": "PR Raised"},
                "mark_complete": {"status_value": "Finished"},
                "update_with_context": {"status_value": ""},
            }
        )
        desc = _build_dev_task_description("TASK-1", "jira", "Fix bug", team_b)
        assert "PR Raised" in desc
        assert "IN REVIEW" not in desc

    def test_dev_task_instructs_start_development_first(
        self, default_workflow_config: WorkflowConfig
    ) -> None:
        """INT-SD-06: Dev task instructs transitioning to START_DEVELOPMENT first."""
        from src.scheduler.jobs.scan_tickets import _build_dev_task_description

        desc = _build_dev_task_description(
            "PROJ-10", "jira", "Implement feature", default_workflow_config
        )
        # Check that the instruction to set IN PROGRESS FIRST is present.
        upper = desc.upper()
        assert "FIRST" in upper or "VERY FIRST" in upper or "IMMEDIATELY" in upper

    def test_dev_task_contains_pr_url_instruction(
        self, default_workflow_config: WorkflowConfig
    ) -> None:
        """Dev task instructs agent to output PR_URL: line."""
        from src.scheduler.jobs.scan_tickets import _build_dev_task_description

        desc = _build_dev_task_description("PROJ-10", "jira", "Fix", default_workflow_config)
        assert "PR_URL:" in desc


# ---------------------------------------------------------------------------
# INT-SD-07 — Dispatch guard skips REJECTED tickets
# ---------------------------------------------------------------------------

class TestDispatchRejectedGuard:
    def test_dispatch_guard_skips_rejected_ticket(
        self,
        default_workflow_config: WorkflowConfig,
        mock_dev_registry: MagicMock,
    ) -> None:
        """INT-SD-07: scan_and_dispatch_job skips tickets with REJECTED raw_status (BR-3)."""
        from src.ticket.models import TicketRecord
        from src.scheduler.jobs.scan_tickets import scan_and_dispatch_job

        rejected_ticket = TicketRecord(
            id="PROJ-99", source="jira", title="Rejected ticket",
            url="", raw_status="REJECTED",
        )
        mock_tr = _make_tracker_registry(jira_tickets=[rejected_ticket])
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit = MagicMock()

        with (
            patch("src.scheduler.jobs.scan_tickets.Task"),
            patch("src.scheduler.jobs.scan_tickets.CrewBuilder"),
        ):
            scan_and_dispatch_job(
                registry=mock_dev_registry,
                settings=_make_settings(),
                executor=executor,
                workflow=default_workflow_config,
                tracker_registry=mock_tr,
            )

        executor.submit.assert_not_called()


# ---------------------------------------------------------------------------
# INT-SD-08 / INT-SD-09 — PR registration after _execute_ticket
# ---------------------------------------------------------------------------

class TestPRRegistration:
    def test_execute_ticket_registers_pr_in_open_prs(
        self,
        default_workflow_config: WorkflowConfig,
        mock_dev_registry: MagicMock,
    ) -> None:
        """INT-SD-08: _execute_ticket registers PRRecord in _open_prs when PR_URL found."""
        import src.scheduler.jobs.scan_tickets as scan_mod
        from src.scheduler.jobs.scan_tickets import _execute_ticket

        result_text = "Done.\nPR_URL: https://github.com/org/repo/pull/42\nDone."

        with (
            patch("src.scheduler.jobs.scan_tickets.Task"),
            patch("src.scheduler.jobs.scan_tickets.CrewBuilder") as mock_cb,
        ):
            mock_crew = MagicMock()
            mock_crew.kickoff.return_value = MagicMock(__str__=lambda s: result_text)
            mock_cb.build.return_value = mock_crew

            _execute_ticket(
                ticket_id="PROJ-1",
                source="jira",
                title="Test ticket",
                raw_status="ACCEPTED",
                registry=mock_dev_registry,
                workflow=default_workflow_config,
            )

        assert "PROJ-1" in scan_mod._open_prs, "_open_prs should contain PROJ-1 after execute"
        assert scan_mod._open_prs["PROJ-1"].pr_url == "https://github.com/org/repo/pull/42"

    def test_execute_ticket_registers_pr_in_prs_under_review(
        self,
        default_workflow_config: WorkflowConfig,
        mock_dev_registry: MagicMock,
    ) -> None:
        """INT-SD-09: _execute_ticket registers pr_url in _prs_under_review."""
        import src.scheduler.jobs.scan_tickets as scan_mod
        from src.scheduler.jobs.scan_tickets import _execute_ticket

        result_text = "PR_URL: https://github.com/org/repo/pull/55"

        with (
            patch("src.scheduler.jobs.scan_tickets.Task"),
            patch("src.scheduler.jobs.scan_tickets.CrewBuilder") as mock_cb,
        ):
            mock_crew = MagicMock()
            mock_crew.kickoff.return_value = MagicMock(__str__=lambda s: result_text)
            mock_cb.build.return_value = mock_crew

            _execute_ticket(
                ticket_id="PROJ-2",
                source="jira",
                title="Another ticket",
                raw_status="ACCEPTED",
                registry=mock_dev_registry,
                workflow=default_workflow_config,
            )

        assert "PROJ-2" in scan_mod._prs_under_review
        assert scan_mod._prs_under_review["PROJ-2"] == "https://github.com/org/repo/pull/55"


# ---------------------------------------------------------------------------
# INT-SD-10 — Double-dispatch prevention guard
# ---------------------------------------------------------------------------

class TestDoubleDispatchPrevention:
    def test_dispatch_guard_prevents_double_dispatch(
        self,
        default_workflow_config: WorkflowConfig,
        mock_dev_registry: MagicMock,
    ) -> None:
        """INT-SD-10: ticket already in _in_progress_tickets is not re-dispatched."""
        from src.ticket.models import TicketRecord
        import src.scheduler.jobs.scan_tickets as scan_mod
        from src.scheduler.jobs.scan_tickets import scan_and_dispatch_job

        # Pre-populate the in-progress guard.
        scan_mod._in_progress_tickets.add("PROJ-99")

        in_progress_ticket = TicketRecord(
            id="PROJ-99", source="jira", title="Already in progress",
            url="", raw_status="ACCEPTED",
        )
        mock_tr = _make_tracker_registry(jira_tickets=[in_progress_ticket])
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit = MagicMock()

        with (
            patch("src.scheduler.jobs.scan_tickets.Task"),
            patch("src.scheduler.jobs.scan_tickets.CrewBuilder"),
        ):
            scan_and_dispatch_job(
                registry=mock_dev_registry,
                settings=_make_settings(),
                executor=executor,
                workflow=default_workflow_config,
                tracker_registry=mock_tr,
            )

        executor.submit.assert_not_called()




