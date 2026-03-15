"""
Integration tests for the PR auto-merge watcher (INT-PR-01 through INT-PR-07).

Covers:
- Merge triggered with ≥1 approval after timeout (BR-2)
- No merge without approvals (BR-2)
- No merge before timeout elapsed
- Already-merged PR → only transition, no second merge call
- Entry cleared after merge
- Mixed PRs (only approved+stale merged)
- Team B custom status (mark_complete.status_value="Finished")
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch, call

import pytest

from src.ticket.models import PRRecord
from src.ticket.workflow import WorkflowConfig, WorkflowOperation

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(timeout: int = 300) -> MagicMock:
    s = MagicMock()
    s.PR_AUTO_MERGE_TIMEOUT_SECONDS = timeout
    return s


def _make_pr_record(
    ticket_id: str,
    pr_url: str,
    age_seconds: float = 310.0,
    approval_count: int = 0,
    is_merged: bool = False,
) -> PRRecord:
    return PRRecord(
        ticket_id=ticket_id,
        pr_url=pr_url,
        opened_at_utc=time.time() - age_seconds,
        approval_count=approval_count,
        is_merged=is_merged,
    )


def _mock_pr_status(is_merged: bool = False, approval_count: int = 0) -> str:
    """Return JSON that a fake crew would output for PR status."""
    import json
    return json.dumps({"is_merged": is_merged, "approval_count": approval_count, "pr_url": "https://github.com/x"})


# ---------------------------------------------------------------------------
# INT-PR-01 — Merge triggered with approval after timeout
# ---------------------------------------------------------------------------

class TestMergeTriggered:
    def test_merge_triggered_with_approval_after_timeout(
        self,
        default_workflow_config: WorkflowConfig,
        mock_dev_registry: MagicMock,
    ) -> None:
        """INT-PR-01: approved + stale PR → _run_pr_merge called."""
        import src.scheduler.jobs.scan_tickets as scan_mod
        from src.scheduler.jobs.pr_merge_watcher import pr_merge_watcher_job

        pr_record = _make_pr_record("PROJ-1", "https://github.com/org/repo/pull/10", age_seconds=310)
        scan_mod._open_prs["PROJ-1"] = pr_record

        executor = MagicMock(spec=ThreadPoolExecutor)

        with (
            patch("src.scheduler.jobs.pr_merge_watcher._run_pr_status_check") as mock_status,
            patch("src.scheduler.jobs.pr_merge_watcher._run_pr_merge") as mock_merge,
        ):
            mock_status.return_value = {"is_merged": False, "approval_count": 1}
            mock_merge.return_value = True

            pr_merge_watcher_job(
                registry=mock_dev_registry,
                settings=_make_settings(timeout=300),
                executor=executor,
                workflow=default_workflow_config,
            )

        mock_merge.assert_called_once()
        merge_call_args = mock_merge.call_args
        # complete_status should be 'COMPLETE' for Team A
        assert merge_call_args[0][2] == "COMPLETE" or merge_call_args[1].get("complete_status") == "COMPLETE"


# ---------------------------------------------------------------------------
# INT-PR-02 — No merge without approval (BR-2)
# ---------------------------------------------------------------------------

class TestNoMergeWithoutApproval:
    def test_no_merge_without_approval(
        self,
        default_workflow_config: WorkflowConfig,
        mock_dev_registry: MagicMock,
    ) -> None:
        """INT-PR-02: 0 approvals → PR not merged (BR-2)."""
        import src.scheduler.jobs.scan_tickets as scan_mod
        from src.scheduler.jobs.pr_merge_watcher import pr_merge_watcher_job

        pr_record = _make_pr_record("PROJ-2", "https://github.com/org/repo/pull/20", age_seconds=310)
        scan_mod._open_prs["PROJ-2"] = pr_record

        executor = MagicMock(spec=ThreadPoolExecutor)

        with (
            patch("src.scheduler.jobs.pr_merge_watcher._run_pr_status_check") as mock_status,
            patch("src.scheduler.jobs.pr_merge_watcher._run_pr_merge") as mock_merge,
        ):
            mock_status.return_value = {"is_merged": False, "approval_count": 0}
            mock_merge.return_value = True

            pr_merge_watcher_job(
                registry=mock_dev_registry,
                settings=_make_settings(timeout=300),
                executor=executor,
                workflow=default_workflow_config,
            )

        mock_merge.assert_not_called()
        # Entry should remain in _open_prs
        assert "PROJ-2" in scan_mod._open_prs


# ---------------------------------------------------------------------------
# INT-PR-03 — No merge before timeout
# ---------------------------------------------------------------------------

class TestNoMergeBeforeTimeout:
    def test_no_merge_before_timeout(
        self,
        default_workflow_config: WorkflowConfig,
        mock_dev_registry: MagicMock,
    ) -> None:
        """INT-PR-03: 2 approvals but PR only 100s old → no merge."""
        import src.scheduler.jobs.scan_tickets as scan_mod
        from src.scheduler.jobs.pr_merge_watcher import pr_merge_watcher_job

        pr_record = _make_pr_record("PROJ-3", "https://github.com/org/repo/pull/30", age_seconds=100)
        scan_mod._open_prs["PROJ-3"] = pr_record

        executor = MagicMock(spec=ThreadPoolExecutor)

        with (
            patch("src.scheduler.jobs.pr_merge_watcher._run_pr_status_check") as mock_status,
            patch("src.scheduler.jobs.pr_merge_watcher._run_pr_merge") as mock_merge,
        ):
            mock_status.return_value = {"is_merged": False, "approval_count": 2}

            pr_merge_watcher_job(
                registry=mock_dev_registry,
                settings=_make_settings(timeout=300),
                executor=executor,
                workflow=default_workflow_config,
            )

        mock_merge.assert_not_called()
        assert "PROJ-3" in scan_mod._open_prs


# ---------------------------------------------------------------------------
# INT-PR-04 — Already-merged PR
# ---------------------------------------------------------------------------

class TestAlreadyMergedPR:
    def test_handles_already_merged_pr(
        self,
        default_workflow_config: WorkflowConfig,
        mock_dev_registry: MagicMock,
    ) -> None:
        """INT-PR-04: already merged PR → transition ticket, clear entry."""
        import src.scheduler.jobs.scan_tickets as scan_mod
        from src.scheduler.jobs.pr_merge_watcher import pr_merge_watcher_job

        pr_record = _make_pr_record("PROJ-4", "https://github.com/org/repo/pull/40", age_seconds=50)
        scan_mod._open_prs["PROJ-4"] = pr_record

        executor = MagicMock(spec=ThreadPoolExecutor)

        with (
            patch("src.scheduler.jobs.pr_merge_watcher._run_pr_status_check") as mock_status,
            patch("src.scheduler.jobs.pr_merge_watcher._run_pr_merge") as mock_merge,
        ):
            mock_status.return_value = {"is_merged": True, "approval_count": 0}
            mock_merge.return_value = True

            pr_merge_watcher_job(
                registry=mock_dev_registry,
                settings=_make_settings(timeout=300),
                executor=executor,
                workflow=default_workflow_config,
            )

        # Should still call _run_pr_merge to do the ticket transition + notification.
        mock_merge.assert_called_once()


# ---------------------------------------------------------------------------
# INT-PR-05 — Entry cleared after merge
# ---------------------------------------------------------------------------

class TestEntryClearedAfterMerge:
    def test_entry_cleared_after_merge(
        self,
        default_workflow_config: WorkflowConfig,
        mock_dev_registry: MagicMock,
    ) -> None:
        """INT-PR-05: _open_prs and _prs_under_review cleared after successful merge."""
        import src.scheduler.jobs.scan_tickets as scan_mod
        from src.scheduler.jobs.pr_merge_watcher import pr_merge_watcher_job

        pr_record = _make_pr_record("PROJ-5", "https://github.com/org/repo/pull/50", age_seconds=310)
        scan_mod._open_prs["PROJ-5"] = pr_record
        scan_mod._prs_under_review["PROJ-5"] = "https://github.com/org/repo/pull/50"

        executor = MagicMock(spec=ThreadPoolExecutor)

        with (
            patch("src.scheduler.jobs.pr_merge_watcher._run_pr_status_check") as mock_status,
            patch("src.scheduler.jobs.pr_merge_watcher._run_pr_merge") as mock_merge,
        ):
            mock_status.return_value = {"is_merged": False, "approval_count": 1}
            mock_merge.return_value = True

            pr_merge_watcher_job(
                registry=mock_dev_registry,
                settings=_make_settings(timeout=300),
                executor=executor,
                workflow=default_workflow_config,
            )

        assert "PROJ-5" not in scan_mod._open_prs
        assert "PROJ-5" not in scan_mod._prs_under_review


# ---------------------------------------------------------------------------
# INT-PR-06 — Mixed PRs: only approved+stale merged
# ---------------------------------------------------------------------------

class TestMixedPRs:
    def test_only_approved_stale_merged(
        self,
        default_workflow_config: WorkflowConfig,
        mock_dev_registry: MagicMock,
    ) -> None:
        """INT-PR-06: only the approved+stale PR is merged; others are not."""
        import src.scheduler.jobs.scan_tickets as scan_mod
        from src.scheduler.jobs.pr_merge_watcher import pr_merge_watcher_job

        # PR A: approved + stale → SHOULD merge
        scan_mod._open_prs["A"] = _make_pr_record("A", "https://github.com/r/pull/1", age_seconds=310)
        # PR B: unapproved + stale → should NOT merge (BR-2)
        scan_mod._open_prs["B"] = _make_pr_record("B", "https://github.com/r/pull/2", age_seconds=310)
        # PR C: approved + fresh → should NOT merge (timeout not elapsed)
        scan_mod._open_prs["C"] = _make_pr_record("C", "https://github.com/r/pull/3", age_seconds=100)

        pr_statuses = {
            "https://github.com/r/pull/1": {"is_merged": False, "approval_count": 1},
            "https://github.com/r/pull/2": {"is_merged": False, "approval_count": 0},
            "https://github.com/r/pull/3": {"is_merged": False, "approval_count": 1},
        }

        executor = MagicMock(spec=ThreadPoolExecutor)

        with (
            patch("src.scheduler.jobs.pr_merge_watcher._run_pr_status_check") as mock_status,
            patch("src.scheduler.jobs.pr_merge_watcher._run_pr_merge") as mock_merge,
        ):
            def status_side_effect(pr_url: str, dev_agent: object) -> dict:
                return pr_statuses[pr_url]

            mock_status.side_effect = status_side_effect
            mock_merge.return_value = True

            pr_merge_watcher_job(
                registry=mock_dev_registry,
                settings=_make_settings(timeout=300),
                executor=executor,
                workflow=default_workflow_config,
            )

        assert mock_merge.call_count == 1
        merged_url = mock_merge.call_args[0][0]
        assert merged_url == "https://github.com/r/pull/1"


# ---------------------------------------------------------------------------
# INT-PR-07 — Team B mark_complete status
# ---------------------------------------------------------------------------

class TestTeamBMarkComplete:
    def test_uses_team_b_mark_complete_status(
        self,
        team_b_workflow_config: WorkflowConfig,
        mock_dev_registry: MagicMock,
    ) -> None:
        """INT-PR-07: merge crew called with 'Finished' for Team B."""
        import src.scheduler.jobs.scan_tickets as scan_mod
        from src.scheduler.jobs.pr_merge_watcher import pr_merge_watcher_job

        pr_record = _make_pr_record("TASK-1", "https://github.com/r/pull/77", age_seconds=310)
        scan_mod._open_prs["TASK-1"] = pr_record

        executor = MagicMock(spec=ThreadPoolExecutor)

        with (
            patch("src.scheduler.jobs.pr_merge_watcher._run_pr_status_check") as mock_status,
            patch("src.scheduler.jobs.pr_merge_watcher._run_pr_merge") as mock_merge,
        ):
            mock_status.return_value = {"is_merged": False, "approval_count": 1}
            mock_merge.return_value = True

            pr_merge_watcher_job(
                registry=mock_dev_registry,
                settings=_make_settings(timeout=300),
                executor=executor,
                workflow=team_b_workflow_config,
            )

        mock_merge.assert_called_once()
        # Third positional arg is complete_status.
        complete_status_used = mock_merge.call_args[0][2]
        assert complete_status_used == "Finished"

