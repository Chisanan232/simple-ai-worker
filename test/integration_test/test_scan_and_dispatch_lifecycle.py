"""
Integration tests for scan_and_dispatch lifecycle (INT-SD-01 through INT-SD-10).

Covers:
- Scan task description uses WorkflowConfig-resolved status strings (not hardcoded)
- REJECTED guard in dispatch loop (BR-3)
- Dev task description injects START_DEVELOPMENT and OPEN_FOR_REVIEW statuses
- Dev task instructs starting development first (GAP-7 mitigation)
- PR registration in _open_prs and _prs_under_review after execute
- Double-dispatch prevention guard
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch, call
import time

import pytest

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


# ---------------------------------------------------------------------------
# INT-SD-01 — scan task description uses SCAN_FOR_WORK status
# ---------------------------------------------------------------------------

class TestScanTaskDescription:
    def test_scan_queries_scan_for_work_status(
        self, default_workflow_config: WorkflowConfig, mock_dev_registry: MagicMock
    ) -> None:
        """INT-SD-01: _build_scan_task_description contains 'ACCEPTED'."""
        from src.scheduler.jobs.scan_tickets import _build_scan_task_description

        desc = _build_scan_task_description(default_workflow_config)
        assert "ACCEPTED" in desc

    def test_scan_excludes_skip_rejected_status(
        self, default_workflow_config: WorkflowConfig
    ) -> None:
        """INT-SD-02: Scan task description contains 'REJECTED' exclusion."""
        from src.scheduler.jobs.scan_tickets import _build_scan_task_description

        desc = _build_scan_task_description(default_workflow_config)
        assert "REJECTED" in desc

    def test_scan_description_forbids_writing_scan_for_work_status(
        self, default_workflow_config: WorkflowConfig
    ) -> None:
        """INT-SD-03: Scan description forbids writing ACCEPTED (BR-1 instruction)."""
        from src.scheduler.jobs.scan_tickets import _build_scan_task_description

        desc = _build_scan_task_description(default_workflow_config)
        assert "human-only" in desc.lower() or "BR-1" in desc or "Do NOT set" in desc

    def test_scan_uses_team_b_status_names(self, team_b_workflow_config: WorkflowConfig) -> None:
        """Scan description uses 'Approved' and 'Cancelled' for Team B."""
        from src.scheduler.jobs.scan_tickets import _build_scan_task_description

        desc = _build_scan_task_description(team_b_workflow_config)
        assert "Approved" in desc
        assert "Cancelled" in desc


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
        import json
        from src.scheduler.jobs.scan_tickets import scan_and_dispatch_job

        tickets_json = json.dumps([
            {"id": "PROJ-99", "source": "jira", "title": "Rejected ticket",
             "url": "", "status": "REJECTED"},
        ])

        mock_crew_result = MagicMock()
        mock_crew_result.__str__ = lambda self: tickets_json  # type: ignore[assignment]

        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit = MagicMock()

        with (
            patch("src.scheduler.jobs.scan_tickets.Task"),
            patch("src.scheduler.jobs.scan_tickets.CrewBuilder") as mock_cb,
        ):
            mock_crew = MagicMock()
            mock_crew.kickoff.return_value = mock_crew_result
            mock_cb.build.return_value = mock_crew

            scan_and_dispatch_job(
                registry=mock_dev_registry,
                settings=_make_settings(),
                executor=executor,
                workflow=default_workflow_config,
            )

        # Executor.submit must NOT be called for REJECTED tickets.
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
        import json
        import src.scheduler.jobs.scan_tickets as scan_mod
        from src.scheduler.jobs.scan_tickets import scan_and_dispatch_job

        # Pre-populate the in-progress guard.
        scan_mod._in_progress_tickets.add("PROJ-99")

        tickets_json = json.dumps([
            {"id": "PROJ-99", "source": "jira", "title": "Already in progress",
             "url": "", "status": "ACCEPTED"},
        ])

        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit = MagicMock()

        with (
            patch("src.scheduler.jobs.scan_tickets.Task"),
            patch("src.scheduler.jobs.scan_tickets.CrewBuilder") as mock_cb,
        ):
            mock_crew = MagicMock()
            mock_crew.kickoff.return_value = MagicMock(__str__=lambda s: tickets_json)
            mock_cb.build.return_value = mock_crew

            scan_and_dispatch_job(
                registry=mock_dev_registry,
                settings=_make_settings(),
                executor=executor,
                workflow=default_workflow_config,
            )

        executor.submit.assert_not_called()




