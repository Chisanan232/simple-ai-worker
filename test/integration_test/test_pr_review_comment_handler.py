"""
Integration tests for PR review comment handler (INT-RC-01 through INT-RC-06).

Covers:
- No action when PR has no changes requested and no unresolved comments
- Fix dispatched for CHANGES_REQUESTED
- Fix dispatched for unresolved inline comments
- Fix task description includes all comment texts
- Fix task explicitly forbids self-approval
- Deduplication prevents parallel fix dispatch
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings() -> MagicMock:
    s = MagicMock()
    s.PR_REVIEW_COMMENT_CHECK_INTERVAL_SECONDS = 120
    return s


def _populate_prs_under_review(ticket_id: str, pr_url: str) -> None:
    """Helper: add a PR to the shared _prs_under_review dict."""
    import src.scheduler.jobs.scan_tickets as scan_mod

    scan_mod._prs_under_review[ticket_id] = pr_url


# ---------------------------------------------------------------------------
# INT-RC-01 — No action when no changes requested and no unresolved comments
# ---------------------------------------------------------------------------


class TestNoActionWhenNoFeedback:
    def test_no_action_when_approved_and_no_comments(self, mock_dev_registry: MagicMock) -> None:
        """INT-RC-01: APPROVED PR with 0 unresolved comments → no executor.submit."""
        from src.scheduler.jobs.pr_review_comment_handler import (
            pr_review_comment_handler_job,
        )

        _populate_prs_under_review("PROJ-10", "https://github.com/r/pull/10")

        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit = MagicMock()

        with patch("src.scheduler.jobs.pr_review_comment_handler._check_pr_review_status") as mock_check:
            mock_check.return_value = {
                "has_changes_requested": False,
                "unresolved_comment_count": 0,
                "comments": [],
            }

            pr_review_comment_handler_job(
                registry=mock_dev_registry,
                settings=_make_settings(),
                executor=executor,
            )

        executor.submit.assert_not_called()


# ---------------------------------------------------------------------------
# INT-RC-02 — Dispatches fix for CHANGES_REQUESTED
# ---------------------------------------------------------------------------


class TestDispatchForChangesRequested:
    def test_dispatches_fix_for_changes_requested(self, mock_dev_registry: MagicMock) -> None:
        """INT-RC-02: CHANGES_REQUESTED → executor.submit called once."""
        from src.scheduler.jobs.pr_review_comment_handler import (
            pr_review_comment_handler_job,
        )

        _populate_prs_under_review("PROJ-11", "https://github.com/r/pull/11")

        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit = MagicMock()

        with patch("src.scheduler.jobs.pr_review_comment_handler._check_pr_review_status") as mock_check:
            mock_check.return_value = {
                "has_changes_requested": True,
                "unresolved_comment_count": 1,
                "comments": [{"id": "c1", "body": "Fix typo", "path": "main.py", "line": 10}],
            }

            pr_review_comment_handler_job(
                registry=mock_dev_registry,
                settings=_make_settings(),
                executor=executor,
            )

        executor.submit.assert_called_once()


# ---------------------------------------------------------------------------
# INT-RC-03 — Dispatches fix for unresolved inline comments
# ---------------------------------------------------------------------------


class TestDispatchForUnresolvedComments:
    def test_dispatches_fix_for_unresolved_inline_comments(self, mock_dev_registry: MagicMock) -> None:
        """INT-RC-03: No CHANGES_REQUESTED but unresolved comments → executor.submit called."""
        from src.scheduler.jobs.pr_review_comment_handler import (
            pr_review_comment_handler_job,
        )

        _populate_prs_under_review("PROJ-12", "https://github.com/r/pull/12")

        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit = MagicMock()

        with patch("src.scheduler.jobs.pr_review_comment_handler._check_pr_review_status") as mock_check:
            mock_check.return_value = {
                "has_changes_requested": False,
                "unresolved_comment_count": 2,
                "comments": [
                    {"id": "c1", "body": "Rename variable", "path": "foo.py", "line": 5},
                    {"id": "c2", "body": "Add type hint", "path": "bar.py", "line": 12},
                ],
            }

            pr_review_comment_handler_job(
                registry=mock_dev_registry,
                settings=_make_settings(),
                executor=executor,
            )

        executor.submit.assert_called_once()


# ---------------------------------------------------------------------------
# INT-RC-04 — Fix task includes all comment texts
# ---------------------------------------------------------------------------


class TestFixTaskContainsCommentTexts:
    def test_fix_task_includes_all_comment_texts(self, mock_dev_registry: MagicMock) -> None:
        """INT-RC-04: Fix task description contains all 3 comment bodies."""
        from src.scheduler.jobs.pr_review_comment_handler import _fix_review_comments

        captured_descriptions: list[str] = []

        comments = [
            {"id": "c1", "body": "Add error handling here", "path": "main.py", "line": 20},
            {"id": "c2", "body": "Extract this into a helper function", "path": "utils.py", "line": 5},
            {"id": "c3", "body": "Missing unit test for this branch", "path": "test_x.py", "line": 1},
        ]

        def capture_task(**kwargs: object) -> MagicMock:
            captured_descriptions.append(str(kwargs.get("description", "")))
            return MagicMock()

        with (
            patch("src.scheduler.jobs.pr_review_comment_handler.Task", side_effect=capture_task),
            patch("src.scheduler.jobs.pr_review_comment_handler.CrewBuilder") as mock_cb,
        ):
            mock_crew = MagicMock()
            mock_crew.kickoff.return_value = MagicMock(__str__=lambda s: "FIXES_APPLIED: PROJ-13")
            mock_cb.build.return_value = mock_crew

            dev_agent = mock_dev_registry["dev_agent"]
            _fix_review_comments(
                ticket_id="PROJ-13",
                pr_url="https://github.com/r/pull/13",
                comments=comments,
                dev_agent=dev_agent,
            )

        full_text = " ".join(captured_descriptions)
        assert "Add error handling here" in full_text
        assert "Extract this into a helper function" in full_text
        assert "Missing unit test for this branch" in full_text


# ---------------------------------------------------------------------------
# INT-RC-05 — Fix task explicitly forbids self-approval
# ---------------------------------------------------------------------------


class TestNoSelfApproval:
    def test_fix_task_explicitly_forbids_self_approval(self, mock_dev_registry: MagicMock) -> None:
        """INT-RC-05: Fix task description contains 'Do NOT approve'."""
        from src.scheduler.jobs.pr_review_comment_handler import _fix_review_comments

        captured_descriptions: list[str] = []

        def capture(**kwargs: object) -> MagicMock:
            captured_descriptions.append(str(kwargs.get("description", "")))
            return MagicMock()

        with (
            patch("src.scheduler.jobs.pr_review_comment_handler.Task", side_effect=capture),
            patch("src.scheduler.jobs.pr_review_comment_handler.CrewBuilder") as mock_cb,
        ):
            mock_crew = MagicMock()
            mock_crew.kickoff.return_value = MagicMock(__str__=lambda s: "FIXES_APPLIED: PROJ-14")
            mock_cb.build.return_value = mock_crew

            dev_agent = mock_dev_registry["dev_agent"]
            _fix_review_comments(
                ticket_id="PROJ-14",
                pr_url="https://github.com/r/pull/14",
                comments=[{"id": "c1", "body": "Fix this", "path": "x.py", "line": 1}],
                dev_agent=dev_agent,
            )

        full_text = " ".join(captured_descriptions)
        assert "Do NOT approve" in full_text or "do not approve" in full_text.lower()

    def test_fix_task_explicitly_forbids_merge(self, mock_dev_registry: MagicMock) -> None:
        """Fix task description also forbids merging."""
        from src.scheduler.jobs.pr_review_comment_handler import _fix_review_comments

        captured_descriptions: list[str] = []

        def capture(**kwargs: object) -> MagicMock:
            captured_descriptions.append(str(kwargs.get("description", "")))
            return MagicMock()

        with (
            patch("src.scheduler.jobs.pr_review_comment_handler.Task", side_effect=capture),
            patch("src.scheduler.jobs.pr_review_comment_handler.CrewBuilder") as mock_cb,
        ):
            mock_crew = MagicMock()
            mock_crew.kickoff.return_value = MagicMock(__str__=lambda s: "done")
            mock_cb.build.return_value = mock_crew

            dev_agent = mock_dev_registry["dev_agent"]
            _fix_review_comments(
                ticket_id="PROJ-15",
                pr_url="https://github.com/r/pull/15",
                comments=[{"id": "c1", "body": "test", "path": "x.py", "line": 1}],
                dev_agent=dev_agent,
            )

        full_text = " ".join(captured_descriptions)
        assert "Do NOT merge" in full_text or "do not merge" in full_text.lower()


# ---------------------------------------------------------------------------
# INT-RC-06 — Deduplication prevents parallel fix dispatch
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_deduplication_prevents_parallel_fix(self, mock_dev_registry: MagicMock) -> None:
        """INT-RC-06: ticket already in _in_progress_comment_fixes → no re-dispatch."""
        import src.scheduler.jobs.pr_review_comment_handler as handler_mod
        from src.scheduler.jobs.pr_review_comment_handler import (
            pr_review_comment_handler_job,
        )

        _populate_prs_under_review("PROJ-16", "https://github.com/r/pull/16")
        # Pre-populate the dedup guard.
        handler_mod._in_progress_comment_fixes.add("PROJ-16")

        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit = MagicMock()

        with patch("src.scheduler.jobs.pr_review_comment_handler._check_pr_review_status") as mock_check:
            mock_check.return_value = {
                "has_changes_requested": True,
                "unresolved_comment_count": 1,
                "comments": [{"id": "c1", "body": "Fix this", "path": "x.py", "line": 1}],
            }

            pr_review_comment_handler_job(
                registry=mock_dev_registry,
                settings=_make_settings(),
                executor=executor,
            )

        executor.submit.assert_not_called()

    def test_deduplication_guard_cleared_after_fix(self, mock_dev_registry: MagicMock) -> None:
        """After _fix_review_comments completes, guard is cleared."""
        import src.scheduler.jobs.pr_review_comment_handler as handler_mod
        from src.scheduler.jobs.pr_review_comment_handler import _fix_review_comments

        dev_agent = mock_dev_registry["dev_agent"]

        # Manually set the guard (as if it was set before executor.submit).
        handler_mod._in_progress_comment_fixes.add("PROJ-17")

        with (
            patch("src.scheduler.jobs.pr_review_comment_handler.Task"),
            patch("src.scheduler.jobs.pr_review_comment_handler.CrewBuilder") as mock_cb,
        ):
            mock_crew = MagicMock()
            mock_crew.kickoff.return_value = MagicMock(__str__=lambda s: "FIXES_APPLIED: PROJ-17")
            mock_cb.build.return_value = mock_crew

            _fix_review_comments(
                ticket_id="PROJ-17",
                pr_url="https://github.com/r/pull/17",
                comments=[{"id": "c1", "body": "fix", "path": "x.py", "line": 1}],
                dev_agent=dev_agent,
            )

        # Guard must be cleared after completion.
        assert "PROJ-17" not in handler_mod._in_progress_comment_fixes
