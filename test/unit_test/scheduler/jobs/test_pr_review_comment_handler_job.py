"""
Unit tests for :func:`src.scheduler.jobs.pr_review_comment_handler.pr_review_comment_handler_job`
and its helpers :func:`_check_pr_review_status` / :func:`_fix_review_comments`.

Test IDs: UNIT-RCH-01 through UNIT-RCH-10
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings() -> MagicMock:
    s = MagicMock()
    s.PR_REVIEW_COMMENT_CHECK_INTERVAL_SECONDS = 120
    return s


def _make_registry(dev_agent: MagicMock | None = None) -> MagicMock:
    reg = MagicMock()
    reg.__getitem__ = MagicMock(return_value=dev_agent or MagicMock())
    reg.agent_ids = MagicMock(return_value=["dev_agent"])
    return reg


def _populate_under_review(ticket_id: str, pr_url: str) -> None:
    import src.scheduler.jobs.scan_tickets as scan_mod

    scan_mod._prs_under_review[ticket_id] = pr_url


# ---------------------------------------------------------------------------
# Fixture: clear shared state around every test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_state() -> None:  # type: ignore[return]
    import src.scheduler.jobs.pr_review_comment_handler as review_mod
    import src.scheduler.jobs.scan_tickets as scan_mod

    scan_mod._prs_under_review.clear()
    review_mod._in_progress_comment_fixes.clear()
    yield
    scan_mod._prs_under_review.clear()
    review_mod._in_progress_comment_fixes.clear()


# ===========================================================================
# UNIT-RCH-01: No-op when _prs_under_review is empty
# ===========================================================================


class TestNoOpenPRs:
    def test_returns_early_when_no_prs_under_review(self) -> None:
        """UNIT-RCH-01: job does nothing when _prs_under_review is empty."""
        from src.scheduler.jobs.pr_review_comment_handler import (
            pr_review_comment_handler_job,
        )

        registry = _make_registry()
        executor = MagicMock(spec=ThreadPoolExecutor)

        with patch("src.scheduler.jobs.pr_review_comment_handler._check_pr_review_status") as mock_check:
            pr_review_comment_handler_job(
                registry=registry,
                settings=_make_settings(),
                executor=executor,
            )

        mock_check.assert_not_called()
        executor.submit.assert_not_called()


# ===========================================================================
# UNIT-RCH-02: Returns early when dev_agent missing
# ===========================================================================


class TestMissingDevAgent:
    def test_skips_run_when_dev_agent_missing(self, caplog: pytest.LogCaptureFixture) -> None:
        """UNIT-RCH-02: KeyError on registry['dev_agent'] → logs error and returns."""
        import logging

        from src.scheduler.jobs.pr_review_comment_handler import (
            pr_review_comment_handler_job,
        )

        _populate_under_review("PROJ-1", "https://github.com/r/pull/1")
        registry = MagicMock()
        registry.__getitem__ = MagicMock(side_effect=KeyError("dev_agent"))

        executor = MagicMock(spec=ThreadPoolExecutor)

        with (
            caplog.at_level(logging.ERROR, logger="src.scheduler.jobs.pr_review_comment_handler"),
            patch("src.scheduler.jobs.pr_review_comment_handler._check_pr_review_status") as mock_check,
        ):
            pr_review_comment_handler_job(
                registry=registry,
                settings=_make_settings(),
                executor=executor,
            )

        mock_check.assert_not_called()
        executor.submit.assert_not_called()
        assert any("dev_agent" in r.message for r in caplog.records)


# ===========================================================================
# UNIT-RCH-03: No dispatch when PR is approved with no unresolved comments
# ===========================================================================


class TestNoDispatchWhenNoFeedback:
    def test_no_dispatch_when_approved_no_comments(self) -> None:
        """UNIT-RCH-03: APPROVED + 0 unresolved comments → executor.submit not called."""
        from src.scheduler.jobs.pr_review_comment_handler import (
            pr_review_comment_handler_job,
        )

        _populate_under_review("PROJ-2", "https://github.com/r/pull/2")
        registry = _make_registry()
        executor = MagicMock(spec=ThreadPoolExecutor)

        with patch("src.scheduler.jobs.pr_review_comment_handler._check_pr_review_status") as mock_check:
            mock_check.return_value = {
                "has_changes_requested": False,
                "unresolved_comment_count": 0,
                "comments": [],
            }
            pr_review_comment_handler_job(
                registry=registry,
                settings=_make_settings(),
                executor=executor,
            )

        executor.submit.assert_not_called()


# ===========================================================================
# UNIT-RCH-04: Dispatch when CHANGES_REQUESTED
# ===========================================================================


class TestDispatchOnChangesRequested:
    def test_dispatches_fix_for_changes_requested(self) -> None:
        """UNIT-RCH-04: CHANGES_REQUESTED → executor.submit called once."""
        from src.scheduler.jobs.pr_review_comment_handler import (
            pr_review_comment_handler_job,
        )

        _populate_under_review("PROJ-3", "https://github.com/r/pull/3")
        registry = _make_registry()
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit = MagicMock()

        with patch("src.scheduler.jobs.pr_review_comment_handler._check_pr_review_status") as mock_check:
            mock_check.return_value = {
                "has_changes_requested": True,
                "unresolved_comment_count": 0,
                "comments": [],
            }
            pr_review_comment_handler_job(
                registry=registry,
                settings=_make_settings(),
                executor=executor,
            )

        executor.submit.assert_called_once()


# ===========================================================================
# UNIT-RCH-05: Dispatch when unresolved inline comments (even without CHANGES_REQUESTED)
# ===========================================================================


class TestDispatchOnUnresolvedComments:
    def test_dispatches_fix_for_unresolved_comments(self) -> None:
        """UNIT-RCH-05: unresolved_comment_count > 0 → executor.submit called."""
        from src.scheduler.jobs.pr_review_comment_handler import (
            pr_review_comment_handler_job,
        )

        _populate_under_review("PROJ-4", "https://github.com/r/pull/4")
        registry = _make_registry()
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit = MagicMock()

        with patch("src.scheduler.jobs.pr_review_comment_handler._check_pr_review_status") as mock_check:
            mock_check.return_value = {
                "has_changes_requested": False,
                "unresolved_comment_count": 2,
                "comments": [
                    {"id": "c1", "body": "Fix typo", "path": "main.py", "line": 5},
                    {"id": "c2", "body": "Add tests", "path": "test.py", "line": 10},
                ],
            }
            pr_review_comment_handler_job(
                registry=registry,
                settings=_make_settings(),
                executor=executor,
            )

        executor.submit.assert_called_once()


# ===========================================================================
# UNIT-RCH-06: Deduplication prevents double dispatch
# ===========================================================================


class TestDeduplicationGuard:
    def test_deduplication_prevents_double_dispatch(self) -> None:
        """UNIT-RCH-06: ticket already in _in_progress_comment_fixes → not re-dispatched."""
        import src.scheduler.jobs.pr_review_comment_handler as review_mod
        from src.scheduler.jobs.pr_review_comment_handler import (
            pr_review_comment_handler_job,
        )

        _populate_under_review("PROJ-5", "https://github.com/r/pull/5")
        review_mod._in_progress_comment_fixes.add("PROJ-5")  # already in progress

        registry = _make_registry()
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit = MagicMock()

        with patch("src.scheduler.jobs.pr_review_comment_handler._check_pr_review_status") as mock_check:
            pr_review_comment_handler_job(
                registry=registry,
                settings=_make_settings(),
                executor=executor,
            )

        # _check_pr_review_status should not even be called — guarded before
        mock_check.assert_not_called()
        executor.submit.assert_not_called()


# ===========================================================================
# UNIT-RCH-07: Status check failure → entry skipped
# ===========================================================================


class TestStatusCheckFailure:
    def test_skips_pr_when_status_check_returns_none(self, caplog: pytest.LogCaptureFixture) -> None:
        """UNIT-RCH-07: None from _check_pr_review_status → warning, no dispatch."""
        import logging

        from src.scheduler.jobs.pr_review_comment_handler import (
            pr_review_comment_handler_job,
        )

        _populate_under_review("PROJ-6", "https://github.com/r/pull/6")
        registry = _make_registry()
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit = MagicMock()

        with (
            caplog.at_level(logging.WARNING, logger="src.scheduler.jobs.pr_review_comment_handler"),
            patch("src.scheduler.jobs.pr_review_comment_handler._check_pr_review_status") as mock_check,
        ):
            mock_check.return_value = None
            pr_review_comment_handler_job(
                registry=registry,
                settings=_make_settings(),
                executor=executor,
            )

        executor.submit.assert_not_called()


# ===========================================================================
# UNIT-RCH-08: Dedup guard added before executor.submit
# ===========================================================================


class TestDedupGuardAddedBeforeSubmit:
    def test_dedup_guard_added_before_submit(self) -> None:
        """UNIT-RCH-08: ticket added to _in_progress_comment_fixes before executor.submit."""
        import src.scheduler.jobs.pr_review_comment_handler as review_mod
        from src.scheduler.jobs.pr_review_comment_handler import (
            pr_review_comment_handler_job,
        )

        _populate_under_review("PROJ-7", "https://github.com/r/pull/7")
        registry = _make_registry()

        captured_guard_state: list[bool] = []

        def capturing_submit(fn: object, *args: object, **kwargs: object) -> MagicMock:
            # At the moment submit is called, the guard must already be set.
            captured_guard_state.append("PROJ-7" in review_mod._in_progress_comment_fixes)
            return MagicMock()

        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit = capturing_submit

        with patch("src.scheduler.jobs.pr_review_comment_handler._check_pr_review_status") as mock_check:
            mock_check.return_value = {
                "has_changes_requested": True,
                "unresolved_comment_count": 1,
                "comments": [{"id": "c1", "body": "Fix it", "path": "f.py", "line": 1}],
            }
            pr_review_comment_handler_job(
                registry=registry,
                settings=_make_settings(),
                executor=executor,
            )

        assert captured_guard_state == [True], "Dedup guard must be set BEFORE executor.submit is called"


# ===========================================================================
# UNIT-RCH-09: Fix task description contains FIXES_APPLIED instruction
# ===========================================================================


class TestFixTaskDescription:
    def test_fix_task_description_contains_fixes_applied(self) -> None:
        """UNIT-RCH-09: _PR_FIX_TASK_TEMPLATE contains 'FIXES_APPLIED' output marker."""
        from src.scheduler.jobs.pr_review_comment_handler import _PR_FIX_TASK_TEMPLATE

        rendered = _PR_FIX_TASK_TEMPLATE.format(
            pr_url="https://github.com/r/pull/9",
            ticket_id="PROJ-9",
            comments_text="[1] Fix the bug",
        )
        assert "FIXES_APPLIED" in rendered

    def test_fix_task_description_forbids_approval(self) -> None:
        """UNIT-RCH-09b: Fix task explicitly says NOT to approve the PR."""
        from src.scheduler.jobs.pr_review_comment_handler import _PR_FIX_TASK_TEMPLATE

        rendered = _PR_FIX_TASK_TEMPLATE.format(
            pr_url="https://github.com/r/pull/9",
            ticket_id="PROJ-9",
            comments_text="[1] Fix the bug",
        )
        upper = rendered.upper()
        assert "DO NOT APPROVE" in upper or "NOT APPROVE" in upper

    def test_fix_task_description_forbids_merge(self) -> None:
        """UNIT-RCH-09c: Fix task explicitly says NOT to merge the PR."""
        from src.scheduler.jobs.pr_review_comment_handler import _PR_FIX_TASK_TEMPLATE

        rendered = _PR_FIX_TASK_TEMPLATE.format(
            pr_url="https://github.com/r/pull/9",
            ticket_id="PROJ-9",
            comments_text="[1] Fix the bug",
        )
        upper = rendered.upper()
        assert "DO NOT MERGE" in upper or "NOT MERGE" in upper


# ===========================================================================
# UNIT-RCH-10: Multiple PRs — only actionable ones dispatched
# ===========================================================================


class TestMultiplePRs:
    def test_only_actionable_prs_dispatched(self) -> None:
        """UNIT-RCH-10: 2 PRs, only 1 has changes_requested → 1 submit."""
        import src.scheduler.jobs.scan_tickets as scan_mod
        from src.scheduler.jobs.pr_review_comment_handler import (
            pr_review_comment_handler_job,
        )

        scan_mod._prs_under_review["PROJ-A"] = "https://github.com/r/pull/10"
        scan_mod._prs_under_review["PROJ-B"] = "https://github.com/r/pull/11"

        registry = _make_registry()
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit = MagicMock()

        def fake_status(pr_url: str, agent: object) -> dict:
            if "pull/10" in pr_url:
                return {
                    "has_changes_requested": True,
                    "unresolved_comment_count": 1,
                    "comments": [{"id": "c1", "body": "Fix", "path": "f.py", "line": 1}],
                }
            # pull/11: no feedback
            return {"has_changes_requested": False, "unresolved_comment_count": 0, "comments": []}

        with patch(
            "src.scheduler.jobs.pr_review_comment_handler._check_pr_review_status",
            side_effect=fake_status,
        ):
            pr_review_comment_handler_job(
                registry=registry,
                settings=_make_settings(),
                executor=executor,
            )

        # Only PROJ-A (pull/10) should be submitted
        assert executor.submit.call_count == 1
