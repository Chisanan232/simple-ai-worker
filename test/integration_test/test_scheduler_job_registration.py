"""
Integration tests for scheduler job registration (INT-SCH-01 through INT-SCH-03).

Verifies that all Phase 6 + Phase 8 jobs are registered with the correct IDs
and intervals when registry/settings/executor are provided.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runner(interval: int = 60) -> "SchedulerRunner":  # type: ignore[name-defined]
    from src.scheduler.runner import SchedulerRunner
    from src.ticket.workflow import WorkflowConfig

    registry = MagicMock()
    settings = MagicMock()
    settings.PR_REVIEW_COMMENT_CHECK_INTERVAL_SECONDS = 120
    executor = MagicMock(spec=ThreadPoolExecutor)
    workflow = WorkflowConfig(
        {
            "scan_for_work": {"status_value": "ACCEPTED", "human_only": True},
            "skip_rejected": {"status_value": "REJECTED"},
            "start_development": {"status_value": "IN PROGRESS"},
            "open_for_review": {"status_value": "IN REVIEW"},
            "mark_complete": {"status_value": "COMPLETE"},
            "update_with_context": {"status_value": ""},
        }
    )

    return SchedulerRunner(
        interval_seconds=interval,
        registry=registry,
        settings=settings,
        executor=executor,
        workflow=workflow,
    )


# ---------------------------------------------------------------------------
# INT-SCH-01 — All Phase 6 + Phase 8 jobs registered
# ---------------------------------------------------------------------------


class TestAllJobsRegistered:
    def test_all_phase8_jobs_registered(self) -> None:
        """INT-SCH-01: All expected job IDs are present after runner.start()."""
        runner = _make_runner()
        runner.start()
        try:
            job_ids = {job.id for job in runner._scheduler.get_jobs()}
            expected = {
                "hello_world",
                "scan_and_dispatch",
                "planner_listener",
                "dev_lead_listener",
                "pr_merge_watcher",
                "pr_review_comment_handler",
            }
            assert expected.issubset(job_ids), f"Missing jobs: {expected - job_ids}. Registered: {job_ids}"
        finally:
            runner.stop(wait=False)


# ---------------------------------------------------------------------------
# INT-SCH-02 — PR merge watcher has 60s interval
# ---------------------------------------------------------------------------


class TestPRMergeWatcherInterval:
    def test_pr_merge_watcher_has_60s_interval(self) -> None:
        """INT-SCH-02: pr_merge_watcher job has 60-second interval."""
        runner = _make_runner()
        runner.start()
        try:
            jobs = {job.id: job for job in runner._scheduler.get_jobs()}
            assert "pr_merge_watcher" in jobs
            job = jobs["pr_merge_watcher"]
            # APScheduler IntervalTrigger stores interval in trigger.interval.total_seconds()
            trigger = job.trigger
            interval_seconds = trigger.interval.total_seconds()
            assert interval_seconds == 60, f"Expected pr_merge_watcher interval=60s, got {interval_seconds}s"
        finally:
            runner.stop(wait=False)


# ---------------------------------------------------------------------------
# INT-SCH-03 — PR review comment handler uses settings.PR_REVIEW_COMMENT_CHECK_INTERVAL_SECONDS
# ---------------------------------------------------------------------------


class TestPRCommentHandlerInterval:
    def test_pr_comment_handler_has_configured_interval(self) -> None:
        """INT-SCH-03: pr_review_comment_handler uses PR_REVIEW_COMMENT_CHECK_INTERVAL_SECONDS."""
        runner = _make_runner()
        # settings.PR_REVIEW_COMMENT_CHECK_INTERVAL_SECONDS is mocked to 120.
        runner.start()
        try:
            jobs = {job.id: job for job in runner._scheduler.get_jobs()}
            assert "pr_review_comment_handler" in jobs
            job = jobs["pr_review_comment_handler"]
            trigger = job.trigger
            interval_seconds = trigger.interval.total_seconds()
            assert interval_seconds == 120, f"Expected pr_review_comment_handler interval=120s, got {interval_seconds}s"
        finally:
            runner.stop(wait=False)

    def test_minimal_mode_has_only_hello_world(self) -> None:
        """When no registry/settings/executor → only hello_world job is registered."""
        from src.scheduler.runner import SchedulerRunner

        runner = SchedulerRunner(interval_seconds=60)
        runner.start()
        try:
            job_ids = {job.id for job in runner._scheduler.get_jobs()}
            assert job_ids == {"hello_world"}
        finally:
            runner.stop(wait=False)
