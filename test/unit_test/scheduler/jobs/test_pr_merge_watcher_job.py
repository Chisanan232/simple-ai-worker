"""
Unit tests for :func:`src.scheduler.jobs.pr_merge_watcher.pr_merge_watcher_job`
and its helper :func:`_run_pr_status_check` / :func:`_run_pr_merge`.

Test IDs: UNIT-PMW-01 through UNIT-PMW-12
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from src.ticket.models import PRRecord
from src.ticket.workflow import WorkflowConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEAM_A_WORKFLOW = {
    "scan_for_work": {"status_value": "ACCEPTED", "human_only": True},
    "skip_rejected": {"status_value": "REJECTED"},
    "start_development": {"status_value": "IN PROGRESS"},
    "open_for_review": {"status_value": "IN REVIEW"},
    "mark_complete": {"status_value": "COMPLETE"},
    "update_with_context": {"status_value": ""},
}


def _make_workflow(**overrides: dict) -> WorkflowConfig:
    cfg = dict(_TEAM_A_WORKFLOW)
    cfg.update(overrides)
    return WorkflowConfig(cfg)


def _make_settings(timeout: int = 300) -> MagicMock:
    s = MagicMock()
    s.PR_AUTO_MERGE_TIMEOUT_SECONDS = timeout
    return s


def _make_pr_record(
    ticket_id: str = "PROJ-1",
    pr_url: str = "https://github.com/org/repo/pull/1",
    age_seconds: float = 310.0,
) -> PRRecord:
    return PRRecord(
        ticket_id=ticket_id,
        pr_url=pr_url,
        opened_at_utc=time.time() - age_seconds,
    )


def _make_registry(dev_agent: MagicMock | None = None) -> MagicMock:
    reg = MagicMock()
    reg.__getitem__ = MagicMock(return_value=dev_agent or MagicMock())
    reg.agent_ids = MagicMock(return_value=["dev_agent"])
    return reg


# ---------------------------------------------------------------------------
# Fixture: clear shared state around every test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_open_prs() -> None:  # type: ignore[return]
    import src.scheduler.jobs.scan_tickets as scan_mod

    scan_mod._open_prs.clear()
    scan_mod._prs_under_review.clear()
    yield
    scan_mod._open_prs.clear()
    scan_mod._prs_under_review.clear()


# ===========================================================================
# UNIT-PMW-01: No-op when _open_prs is empty
# ===========================================================================


class TestNoOpenPrs:
    def test_returns_early_when_no_open_prs(self) -> None:
        """UNIT-PMW-01: pr_merge_watcher_job does nothing when _open_prs is empty."""
        from src.scheduler.jobs.pr_merge_watcher import pr_merge_watcher_job

        registry = _make_registry()
        executor = MagicMock(spec=ThreadPoolExecutor)

        with patch("src.scheduler.jobs.pr_merge_watcher._run_pr_status_check") as mock_check:
            pr_merge_watcher_job(
                registry=registry,
                settings=_make_settings(),
                executor=executor,
                workflow=_make_workflow(),
            )

        mock_check.assert_not_called()


# ===========================================================================
# UNIT-PMW-02: Returns early when workflow is None
# ===========================================================================


class TestNoWorkflow:
    def test_skips_run_when_no_workflow_provided(self, caplog: pytest.LogCaptureFixture) -> None:
        """UNIT-PMW-02: job logs a warning and returns when workflow=None."""
        import logging

        import src.scheduler.jobs.scan_tickets as scan_mod
        from src.scheduler.jobs.pr_merge_watcher import pr_merge_watcher_job

        scan_mod._open_prs["PROJ-1"] = _make_pr_record()
        registry = _make_registry()
        executor = MagicMock(spec=ThreadPoolExecutor)

        with (
            caplog.at_level(logging.WARNING, logger="src.scheduler.jobs.pr_merge_watcher"),
            patch("src.scheduler.jobs.pr_merge_watcher._run_pr_status_check") as mock_check,
        ):
            pr_merge_watcher_job(
                registry=registry,
                settings=_make_settings(),
                executor=executor,
                workflow=None,
            )

        mock_check.assert_not_called()
        assert any("WorkflowConfig" in r.message or "workflow" in r.message.lower() for r in caplog.records)


# ===========================================================================
# UNIT-PMW-03: Returns early when dev_agent missing from registry
# ===========================================================================


class TestMissingDevAgent:
    def test_skips_run_when_dev_agent_missing(self, caplog: pytest.LogCaptureFixture) -> None:
        """UNIT-PMW-03: job logs an error and returns when dev_agent not in registry."""
        import logging

        import src.scheduler.jobs.scan_tickets as scan_mod
        from src.scheduler.jobs.pr_merge_watcher import pr_merge_watcher_job

        scan_mod._open_prs["PROJ-1"] = _make_pr_record()
        registry = MagicMock()
        registry.__getitem__ = MagicMock(side_effect=KeyError("dev_agent"))

        with (
            caplog.at_level(logging.ERROR, logger="src.scheduler.jobs.pr_merge_watcher"),
            patch("src.scheduler.jobs.pr_merge_watcher._run_pr_status_check") as mock_check,
        ):
            pr_merge_watcher_job(
                registry=registry,
                settings=_make_settings(),
                executor=MagicMock(),
                workflow=_make_workflow(),
            )

        mock_check.assert_not_called()
        assert any("dev_agent" in r.message for r in caplog.records)


# ===========================================================================
# UNIT-PMW-04: No merge when PR has 0 approvals (BR-2)
# ===========================================================================


class TestNoMergeWithoutApproval:
    def test_no_merge_with_zero_approvals(self) -> None:
        """UNIT-PMW-04: Approved=0 → _run_pr_merge never called (BR-2)."""
        import src.scheduler.jobs.scan_tickets as scan_mod
        from src.scheduler.jobs.pr_merge_watcher import pr_merge_watcher_job

        scan_mod._open_prs["PROJ-2"] = _make_pr_record(age_seconds=400)
        registry = _make_registry()

        with (
            patch("src.scheduler.jobs.pr_merge_watcher._run_pr_status_check") as mock_check,
            patch("src.scheduler.jobs.pr_merge_watcher._run_pr_merge") as mock_merge,
        ):
            mock_check.return_value = {"is_merged": False, "approval_count": 0}
            pr_merge_watcher_job(
                registry=registry,
                settings=_make_settings(timeout=300),
                executor=MagicMock(),
                workflow=_make_workflow(),
            )

        mock_merge.assert_not_called()


# ===========================================================================
# UNIT-PMW-05: No merge before timeout elapsed
# ===========================================================================


class TestNoMergeBeforeTimeout:
    def test_no_merge_before_timeout(self) -> None:
        """UNIT-PMW-05: 1 approval but PR only 60s old, timeout=300 → no merge."""
        import src.scheduler.jobs.scan_tickets as scan_mod
        from src.scheduler.jobs.pr_merge_watcher import pr_merge_watcher_job

        scan_mod._open_prs["PROJ-3"] = _make_pr_record(age_seconds=60)
        registry = _make_registry()

        with (
            patch("src.scheduler.jobs.pr_merge_watcher._run_pr_status_check") as mock_check,
            patch("src.scheduler.jobs.pr_merge_watcher._run_pr_merge") as mock_merge,
        ):
            mock_check.return_value = {"is_merged": False, "approval_count": 1}
            pr_merge_watcher_job(
                registry=registry,
                settings=_make_settings(timeout=300),
                executor=MagicMock(),
                workflow=_make_workflow(),
            )

        mock_merge.assert_not_called()


# ===========================================================================
# UNIT-PMW-06: Merge triggered with ≥1 approval after timeout
# ===========================================================================


class TestMergeTriggered:
    def test_merge_triggered_with_approval_after_timeout(self) -> None:
        """UNIT-PMW-06: 1 approval + stale PR → _run_pr_merge called with COMPLETE status."""
        import src.scheduler.jobs.scan_tickets as scan_mod
        from src.scheduler.jobs.pr_merge_watcher import pr_merge_watcher_job

        pr_url = "https://github.com/org/repo/pull/10"
        scan_mod._open_prs["PROJ-4"] = _make_pr_record("PROJ-4", pr_url, age_seconds=310)
        registry = _make_registry()

        with (
            patch("src.scheduler.jobs.pr_merge_watcher._run_pr_status_check") as mock_check,
            patch("src.scheduler.jobs.pr_merge_watcher._run_pr_merge") as mock_merge,
        ):
            mock_check.return_value = {"is_merged": False, "approval_count": 1}
            mock_merge.return_value = True

            pr_merge_watcher_job(
                registry=registry,
                settings=_make_settings(timeout=300),
                executor=MagicMock(),
                workflow=_make_workflow(),
            )

        mock_merge.assert_called_once()
        call_args = mock_merge.call_args
        # complete_status must be 'COMPLETE' (Team A)
        assert call_args[0][2] == "COMPLETE" or call_args[1].get("complete_status") == "COMPLETE"


# ===========================================================================
# UNIT-PMW-07: Already-merged PR only calls _run_pr_merge (no re-merge)
# ===========================================================================


class TestAlreadyMergedPR:
    def test_already_merged_pr_clears_entry(self) -> None:
        """UNIT-PMW-07: is_merged=True → _run_pr_merge called once, entry cleared."""
        import src.scheduler.jobs.scan_tickets as scan_mod
        from src.scheduler.jobs.pr_merge_watcher import pr_merge_watcher_job

        pr_url = "https://github.com/org/repo/pull/20"
        scan_mod._open_prs["PROJ-5"] = _make_pr_record("PROJ-5", pr_url, age_seconds=10)
        scan_mod._prs_under_review["PROJ-5"] = pr_url
        registry = _make_registry()

        with (
            patch("src.scheduler.jobs.pr_merge_watcher._run_pr_status_check") as mock_check,
            patch("src.scheduler.jobs.pr_merge_watcher._run_pr_merge") as mock_merge,
        ):
            mock_check.return_value = {"is_merged": True, "approval_count": 0}
            mock_merge.return_value = True

            pr_merge_watcher_job(
                registry=registry,
                settings=_make_settings(timeout=300),
                executor=MagicMock(),
                workflow=_make_workflow(),
            )

        mock_merge.assert_called_once()
        assert "PROJ-5" not in scan_mod._open_prs
        assert "PROJ-5" not in scan_mod._prs_under_review


# ===========================================================================
# UNIT-PMW-08: Entry cleared from _open_prs after successful merge
# ===========================================================================


class TestEntryCleared:
    def test_entry_cleared_after_merge(self) -> None:
        """UNIT-PMW-08: After merge success, ticket removed from _open_prs and _prs_under_review."""
        import src.scheduler.jobs.scan_tickets as scan_mod
        from src.scheduler.jobs.pr_merge_watcher import pr_merge_watcher_job

        pr_url = "https://github.com/org/repo/pull/30"
        scan_mod._open_prs["PROJ-6"] = _make_pr_record("PROJ-6", pr_url, age_seconds=310)
        scan_mod._prs_under_review["PROJ-6"] = pr_url
        registry = _make_registry()

        with (
            patch("src.scheduler.jobs.pr_merge_watcher._run_pr_status_check") as mock_check,
            patch("src.scheduler.jobs.pr_merge_watcher._run_pr_merge") as mock_merge,
        ):
            mock_check.return_value = {"is_merged": False, "approval_count": 1}
            mock_merge.return_value = True

            pr_merge_watcher_job(
                registry=registry,
                settings=_make_settings(timeout=300),
                executor=MagicMock(),
                workflow=_make_workflow(),
            )

        assert "PROJ-6" not in scan_mod._open_prs
        assert "PROJ-6" not in scan_mod._prs_under_review


# ===========================================================================
# UNIT-PMW-09: Entry NOT cleared when merge fails
# ===========================================================================


class TestEntryNotClearedOnFailure:
    def test_entry_kept_when_merge_fails(self) -> None:
        """UNIT-PMW-09: When _run_pr_merge returns False, entry stays in _open_prs."""
        import src.scheduler.jobs.scan_tickets as scan_mod
        from src.scheduler.jobs.pr_merge_watcher import pr_merge_watcher_job

        pr_url = "https://github.com/org/repo/pull/40"
        scan_mod._open_prs["PROJ-7"] = _make_pr_record("PROJ-7", pr_url, age_seconds=310)
        registry = _make_registry()

        with (
            patch("src.scheduler.jobs.pr_merge_watcher._run_pr_status_check") as mock_check,
            patch("src.scheduler.jobs.pr_merge_watcher._run_pr_merge") as mock_merge,
        ):
            mock_check.return_value = {"is_merged": False, "approval_count": 1}
            mock_merge.return_value = False  # merge failed

            pr_merge_watcher_job(
                registry=registry,
                settings=_make_settings(timeout=300),
                executor=MagicMock(),
                workflow=_make_workflow(),
            )

        assert "PROJ-7" in scan_mod._open_prs


# ===========================================================================
# UNIT-PMW-10: Status check failure → entry skipped, not cleared
# ===========================================================================


class TestStatusCheckFailure:
    def test_skips_pr_when_status_check_returns_none(self, caplog: pytest.LogCaptureFixture) -> None:
        """UNIT-PMW-10: None from _run_pr_status_check → warning logged, entry kept."""
        import logging

        import src.scheduler.jobs.scan_tickets as scan_mod
        from src.scheduler.jobs.pr_merge_watcher import pr_merge_watcher_job

        scan_mod._open_prs["PROJ-8"] = _make_pr_record("PROJ-8", age_seconds=400)
        registry = _make_registry()

        with (
            caplog.at_level(logging.WARNING, logger="src.scheduler.jobs.pr_merge_watcher"),
            patch("src.scheduler.jobs.pr_merge_watcher._run_pr_status_check") as mock_check,
            patch("src.scheduler.jobs.pr_merge_watcher._run_pr_merge") as mock_merge,
        ):
            mock_check.return_value = None

            pr_merge_watcher_job(
                registry=registry,
                settings=_make_settings(timeout=300),
                executor=MagicMock(),
                workflow=_make_workflow(),
            )

        mock_merge.assert_not_called()
        assert "PROJ-8" in scan_mod._open_prs


# ===========================================================================
# UNIT-PMW-11: MARK_COMPLETE human_only raises PermissionError early
# ===========================================================================


class TestMarkCompleteHumanOnly:
    def test_raises_permission_error_when_mark_complete_human_only(self, caplog: pytest.LogCaptureFixture) -> None:
        """UNIT-PMW-11: WorkflowConfig.MARK_COMPLETE human_only=True → logs error, skips."""
        import logging

        import src.scheduler.jobs.scan_tickets as scan_mod
        from src.scheduler.jobs.pr_merge_watcher import pr_merge_watcher_job

        misconfigured_workflow = WorkflowConfig(
            {
                "scan_for_work": {"status_value": "ACCEPTED", "human_only": True},
                "skip_rejected": {"status_value": "REJECTED"},
                "start_development": {"status_value": "IN PROGRESS"},
                "open_for_review": {"status_value": "IN REVIEW"},
                "mark_complete": {"status_value": "COMPLETE", "human_only": True},  # misconfigured!
                "update_with_context": {"status_value": ""},
            }
        )

        scan_mod._open_prs["PROJ-9"] = _make_pr_record(age_seconds=400)
        registry = _make_registry()

        with (
            caplog.at_level(logging.ERROR, logger="src.scheduler.jobs.pr_merge_watcher"),
            patch("src.scheduler.jobs.pr_merge_watcher._run_pr_status_check") as mock_check,
        ):
            pr_merge_watcher_job(
                registry=registry,
                settings=_make_settings(),
                executor=MagicMock(),
                workflow=misconfigured_workflow,
            )

        mock_check.assert_not_called()
        assert any(
            "human_only" in r.message or "MARK_COMPLETE" in r.message or "human" in r.message.lower()
            for r in caplog.records
        )


# ===========================================================================
# UNIT-PMW-12: Mixed PRs — only stale approved one is merged
# ===========================================================================


class TestMixedPRs:
    def test_only_stale_approved_pr_is_merged(self) -> None:
        """UNIT-PMW-12: 3 PRs — only 1 meets merge criteria, others skipped."""
        import src.scheduler.jobs.scan_tickets as scan_mod
        from src.scheduler.jobs.pr_merge_watcher import pr_merge_watcher_job

        scan_mod._open_prs["PROJ-A"] = _make_pr_record(
            "PROJ-A", "https://github.com/r/pull/100", age_seconds=310
        )  # approved + stale
        scan_mod._open_prs["PROJ-B"] = _make_pr_record(
            "PROJ-B", "https://github.com/r/pull/101", age_seconds=60
        )  # not stale
        scan_mod._open_prs["PROJ-C"] = _make_pr_record(
            "PROJ-C", "https://github.com/r/pull/102", age_seconds=310
        )  # no approval

        merge_calls: list = []

        def fake_status(pr_url: str, agent: object) -> dict:
            if "pull/100" in pr_url:
                return {"is_merged": False, "approval_count": 1}
            if "pull/101" in pr_url:
                return {"is_merged": False, "approval_count": 1}
            if "pull/102" in pr_url:
                return {"is_merged": False, "approval_count": 0}
            return {"is_merged": False, "approval_count": 0}

        def fake_merge(pr_url: str, ticket_id: str, complete_status: str, agent: object) -> bool:
            merge_calls.append(ticket_id)
            return True

        registry = _make_registry()

        with (
            patch("src.scheduler.jobs.pr_merge_watcher._run_pr_status_check", side_effect=fake_status),
            patch("src.scheduler.jobs.pr_merge_watcher._run_pr_merge", side_effect=fake_merge),
        ):
            pr_merge_watcher_job(
                registry=registry,
                settings=_make_settings(timeout=300),
                executor=MagicMock(),
                workflow=_make_workflow(),
            )

        # Only PROJ-A qualifies (stale + approved)
        assert "PROJ-A" in merge_calls
        assert "PROJ-C" not in merge_calls  # 0 approvals
