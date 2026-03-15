"""
Unit tests for TicketComment model and fetch_ticket_comments implementations.

Covers INT-TC-01 through INT-TC-03:
- INT-TC-01: TicketComment validates required fields
- INT-TC-02: JiraTracker.fetch_ticket_comments calls rest_client.get_issue_comments()
             and returns List[TicketComment]
- INT-TC-03: ClickUpTracker.fetch_ticket_comments calls rest_client.get_task_comments()
             and returns List[TicketComment]
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.ticket.models import TicketComment
from src.ticket.workflow import WorkflowConfig

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Shared workflow config for tracker construction
# ---------------------------------------------------------------------------

_WORKFLOW_CFG = {
    "scan_for_work": {"status_value": "ACCEPTED", "human_only": True},
    "skip_rejected": {"status_value": "REJECTED"},
    "start_development": {"status_value": "IN PROGRESS"},
    "open_for_review": {"status_value": "IN REVIEW"},
    "mark_complete": {"status_value": "COMPLETE"},
    "update_with_context": {"status_value": ""},
    "open_for_dev": {"status_value": "OPEN", "human_only": False},
    "in_planning": {"status_value": "IN PLANNING", "human_only": True},
}


def _make_workflow() -> WorkflowConfig:
    return WorkflowConfig(_WORKFLOW_CFG)


# ---------------------------------------------------------------------------
# INT-TC-01: TicketComment model validation
# ---------------------------------------------------------------------------


class TestTicketCommentModel:
    def test_valid_comment_all_fields(self) -> None:
        """INT-TC-01a: TicketComment accepts all valid fields."""
        comment = TicketComment(
            id="c-123",
            author="alice",
            body="Please clarify the database migration strategy.",
            created_at=1700000000.0,
            source="jira",
        )
        assert comment.id == "c-123"
        assert comment.author == "alice"
        assert comment.body == "Please clarify the database migration strategy."
        assert comment.created_at == 1700000000.0
        assert comment.source == "jira"

    def test_valid_comment_minimal_fields(self) -> None:
        """INT-TC-01b: TicketComment requires only id and source; others have defaults."""
        comment = TicketComment(id="c-999", source="clickup")
        assert comment.id == "c-999"
        assert comment.source == "clickup"
        assert comment.author == ""
        assert comment.body == ""
        assert comment.created_at == 0.0

    def test_comment_is_frozen(self) -> None:
        """INT-TC-01c: TicketComment is immutable (frozen=True)."""
        from pydantic import ValidationError
        comment = TicketComment(id="c-1", source="jira")
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            comment.body = "mutated"  # type: ignore[misc]

    def test_comment_missing_required_id_raises(self) -> None:
        """INT-TC-01d: TicketComment raises ValidationError when id is missing."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            TicketComment(source="jira")  # type: ignore[call-arg]

    def test_comment_missing_required_source_raises(self) -> None:
        """INT-TC-01e: TicketComment raises ValidationError when source is missing."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            TicketComment(id="c-1")  # type: ignore[call-arg]

    def test_clickup_source(self) -> None:
        """INT-TC-01f: TicketComment accepts 'clickup' as source."""
        comment = TicketComment(id="t-42", source="clickup", created_at=1234567890.5)
        assert comment.source == "clickup"
        assert comment.created_at == 1234567890.5

    def test_comments_sorted_by_created_at(self) -> None:
        """INT-TC-01g: TicketComment objects can be sorted by created_at."""
        c1 = TicketComment(id="c1", source="jira", created_at=300.0)
        c2 = TicketComment(id="c2", source="jira", created_at=100.0)
        c3 = TicketComment(id="c3", source="jira", created_at=200.0)
        sorted_comments = sorted([c1, c2, c3], key=lambda x: x.created_at)
        assert [c.id for c in sorted_comments] == ["c2", "c3", "c1"]


# ---------------------------------------------------------------------------
# INT-TC-02: JiraTracker.fetch_ticket_comments
# ---------------------------------------------------------------------------


class TestJiraTrackerFetchComments:
    def _make_jira_tracker(self, raw_comments: list) -> "object":
        from src.ticket.jira_tracker import JiraTracker

        mock_rest = MagicMock()
        mock_rest.get_issue_comments.return_value = raw_comments
        workflow = _make_workflow()
        mock_agent = MagicMock()
        mock_crew_builder = MagicMock()
        return JiraTracker(
            workflow=workflow,
            dev_agent=mock_agent,
            crew_builder=mock_crew_builder,
            rest_client=mock_rest,
        )

    def test_calls_get_issue_comments(self) -> None:
        """INT-TC-02a: fetch_ticket_comments calls rest_client.get_issue_comments."""
        from src.ticket.jira_tracker import JiraTracker

        mock_rest = MagicMock()
        mock_rest.get_issue_comments.return_value = []
        tracker = JiraTracker(
            workflow=_make_workflow(),
            dev_agent=MagicMock(),
            crew_builder=MagicMock(),
            rest_client=mock_rest,
        )
        tracker.fetch_ticket_comments("PROJ-42")
        mock_rest.get_issue_comments.assert_called_once_with("PROJ-42")

    def test_returns_list_of_ticket_comments(self) -> None:
        """INT-TC-02b: Returns a List[TicketComment] from raw REST data."""
        raw = [
            {"id": "1001", "author": "alice", "body": "LGTM", "created_at": 1000.0},
            {"id": "1002", "author": "bob", "body": "Please add tests", "created_at": 2000.0},
        ]
        tracker = self._make_jira_tracker(raw)
        result = tracker.fetch_ticket_comments("PROJ-10")
        assert len(result) == 2
        assert all(isinstance(c, TicketComment) for c in result)
        assert result[0].id == "1001"
        assert result[0].source == "jira"
        assert result[1].author == "bob"

    def test_sorted_oldest_first(self) -> None:
        """INT-TC-02c: fetch_ticket_comments returns comments sorted by created_at ascending."""
        raw = [
            {"id": "c2", "author": "bob", "body": "second", "created_at": 500.0},
            {"id": "c1", "author": "alice", "body": "first", "created_at": 100.0},
        ]
        tracker = self._make_jira_tracker(raw)
        result = tracker.fetch_ticket_comments("PROJ-5")
        assert result[0].id == "c1"
        assert result[1].id == "c2"

    def test_skips_comments_without_id(self) -> None:
        """INT-TC-02d: Items with empty id are silently skipped."""
        raw = [
            {"id": "", "author": "anon", "body": "no id", "created_at": 100.0},
            {"id": "c-valid", "author": "alice", "body": "valid", "created_at": 200.0},
        ]
        tracker = self._make_jira_tracker(raw)
        result = tracker.fetch_ticket_comments("PROJ-9")
        assert len(result) == 1
        assert result[0].id == "c-valid"

    def test_empty_comments_returns_empty_list(self) -> None:
        """INT-TC-02e: No comments → empty list."""
        tracker = self._make_jira_tracker([])
        result = tracker.fetch_ticket_comments("PROJ-99")
        assert result == []


# ---------------------------------------------------------------------------
# INT-TC-03: ClickUpTracker.fetch_ticket_comments
# ---------------------------------------------------------------------------


class TestClickUpTrackerFetchComments:
    def _make_clickup_tracker(self, raw_comments: list) -> "object":
        from src.ticket.clickup_tracker import ClickUpTracker

        mock_rest = MagicMock()
        mock_rest.get_task_comments.return_value = raw_comments
        workflow = _make_workflow()
        mock_agent = MagicMock()
        mock_crew_builder = MagicMock()
        return ClickUpTracker(
            workflow=workflow,
            dev_agent=mock_agent,
            crew_builder=mock_crew_builder,
            rest_client=mock_rest,
        )

    def test_calls_get_task_comments(self) -> None:
        """INT-TC-03a: fetch_ticket_comments calls rest_client.get_task_comments."""
        from src.ticket.clickup_tracker import ClickUpTracker

        mock_rest = MagicMock()
        mock_rest.get_task_comments.return_value = []
        tracker = ClickUpTracker(
            workflow=_make_workflow(),
            dev_agent=MagicMock(),
            crew_builder=MagicMock(),
            rest_client=mock_rest,
        )
        tracker.fetch_ticket_comments("cu-123")
        mock_rest.get_task_comments.assert_called_once_with("cu-123")

    def test_returns_list_of_ticket_comments(self) -> None:
        """INT-TC-03b: Returns a List[TicketComment] from raw REST data."""
        raw = [
            {"id": "cu-c1", "author": "charlie", "body": "Looks good!", "created_at": 1500.0},
        ]
        tracker = self._make_clickup_tracker(raw)
        result = tracker.fetch_ticket_comments("cu-55")
        assert len(result) == 1
        assert isinstance(result[0], TicketComment)
        assert result[0].source == "clickup"
        assert result[0].author == "charlie"

    def test_sorted_oldest_first(self) -> None:
        """INT-TC-03c: fetch_ticket_comments sorted by created_at ascending."""
        raw = [
            {"id": "c3", "author": "d", "body": "third", "created_at": 900.0},
            {"id": "c1", "author": "a", "body": "first", "created_at": 100.0},
            {"id": "c2", "author": "b", "body": "second", "created_at": 500.0},
        ]
        tracker = self._make_clickup_tracker(raw)
        result = tracker.fetch_ticket_comments("cu-7")
        assert [c.id for c in result] == ["c1", "c2", "c3"]

    def test_empty_comments_returns_empty_list(self) -> None:
        """INT-TC-03d: No comments → empty list."""
        tracker = self._make_clickup_tracker([])
        result = tracker.fetch_ticket_comments("cu-0")
        assert result == []


