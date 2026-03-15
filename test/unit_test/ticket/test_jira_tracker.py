"""
Unit tests for :mod:`src.ticket.jira_tracker` (UNIT-JT-01 through UNIT-JT-20).

Covers:
- fetch_tickets_for_operation: correct JQL built from WorkflowConfig, crew called once,
  TicketRecords returned, empty list on empty crew output
- transition: status_for_write called, crew description contains status, BR-1 passthrough
- add_comment: crew description contains ticket ID and comment body
- _parse_ticket_records: JSON parsing, markdown code fence stripping, missing id skipped,
  invalid JSON returns empty list
- Source is always "jira"
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from src.ticket.jira_tracker import JiraTracker
from src.ticket.models import TicketRecord
from src.ticket.workflow import WorkflowConfig, WorkflowOperation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_WORKFLOW_CFG = {
    "scan_for_work": {"status_value": "ACCEPTED", "human_only": True},
    "skip_rejected": {"status_value": "REJECTED"},
    "start_development": {"status_value": "IN PROGRESS"},
    "open_for_review": {"status_value": "IN REVIEW"},
    "mark_complete": {"status_value": "COMPLETE"},
    "update_with_context": {"status_value": ""},
}


def _make_parts(crew_result: str = "[]") -> tuple:
    """Return (workflow, dev_agent, mock_builder). Caller must patch Task separately."""
    workflow = WorkflowConfig(_WORKFLOW_CFG)
    dev_agent = MagicMock()
    mock_crew = MagicMock()
    mock_crew.kickoff.return_value = crew_result
    mock_builder = MagicMock()
    mock_builder.build.return_value = mock_crew
    return workflow, dev_agent, mock_builder


@pytest.fixture(autouse=True)
def _patch_task():
    """Patch crewai.Task inside jira_tracker so MagicMock agent passes Pydantic validation."""
    with patch("src.ticket.jira_tracker.Task") as mock_task_cls:
        mock_task_cls.return_value = MagicMock()
        yield mock_task_cls


# ===========================================================================
# fetch_tickets_for_operation
# ===========================================================================


class TestFetchTicketsForOperation:
    def test_builds_crew_once(self, _patch_task: MagicMock) -> None:
        """UNIT-JT-01: A single crew is built and kicked off for a fetch."""
        workflow, dev_agent, mock_builder = _make_parts("[]")
        tracker = JiraTracker(workflow=workflow, dev_agent=dev_agent, crew_builder=mock_builder)
        tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK)
        mock_builder.build.assert_called_once()

    def test_task_description_contains_scan_status(self, _patch_task: MagicMock) -> None:
        """UNIT-JT-02: Task description contains 'ACCEPTED' (scan_for_work status)."""
        workflow, dev_agent, mock_builder = _make_parts("[]")
        tracker = JiraTracker(workflow=workflow, dev_agent=dev_agent, crew_builder=mock_builder)
        tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK)
        assert "ACCEPTED" in _patch_task.call_args[1]["description"]

    def test_task_description_contains_skip_status(self, _patch_task: MagicMock) -> None:
        """UNIT-JT-03: Task description contains 'REJECTED' (skip_rejected status)."""
        workflow, dev_agent, mock_builder = _make_parts("[]")
        tracker = JiraTracker(workflow=workflow, dev_agent=dev_agent, crew_builder=mock_builder)
        tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK)
        assert "REJECTED" in _patch_task.call_args[1]["description"]

    def test_returns_empty_list_on_empty_crew_output(self, _patch_task: MagicMock) -> None:
        """UNIT-JT-04: Empty JSON array from crew returns empty list."""
        workflow, dev_agent, mock_builder = _make_parts("[]")
        tracker = JiraTracker(workflow=workflow, dev_agent=dev_agent, crew_builder=mock_builder)
        assert tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK) == []

    def test_returns_ticket_records_from_crew_output(self, _patch_task: MagicMock) -> None:
        """UNIT-JT-05: Valid JSON from crew is parsed into TicketRecord objects."""
        tickets = [
            {"id": "PROJ-1", "title": "Login", "url": "https://j.io/1", "status": "ACCEPTED"},
            {"id": "PROJ-2", "title": "OAuth",  "url": "https://j.io/2", "status": "ACCEPTED"},
        ]
        workflow, dev_agent, mock_builder = _make_parts(json.dumps(tickets))
        tracker = JiraTracker(workflow=workflow, dev_agent=dev_agent, crew_builder=mock_builder)
        result = tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK)
        assert len(result) == 2
        assert result[0].id == "PROJ-1"
        assert result[0].source == "jira"
        assert result[0].title == "Login"
        assert result[0].url == "https://j.io/1"
        assert result[0].raw_status == "ACCEPTED"
        assert result[1].id == "PROJ-2"

    def test_source_is_always_jira(self, _patch_task: MagicMock) -> None:
        """UNIT-JT-06: All returned TicketRecords have source='jira'."""
        workflow, dev_agent, mock_builder = _make_parts(
            json.dumps([{"id": "P-1", "title": "T", "url": "", "status": ""}])
        )
        tracker = JiraTracker(workflow=workflow, dev_agent=dev_agent, crew_builder=mock_builder)
        result = tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK)
        assert all(r.source == "jira" for r in result)

    def test_strips_markdown_code_fence(self, _patch_task: MagicMock) -> None:
        """UNIT-JT-07: Markdown fence is stripped before JSON parsing."""
        tickets = [{"id": "P-1", "title": "T", "url": "", "status": "A"}]
        fenced = "```json\n" + json.dumps(tickets) + "\n```"
        workflow, dev_agent, mock_builder = _make_parts(fenced)
        tracker = JiraTracker(workflow=workflow, dev_agent=dev_agent, crew_builder=mock_builder)
        result = tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK)
        assert len(result) == 1 and result[0].id == "P-1"

    def test_returns_empty_list_on_invalid_json(self, _patch_task: MagicMock) -> None:
        """UNIT-JT-08: Non-JSON crew output returns empty list."""
        workflow, dev_agent, mock_builder = _make_parts("Sorry, no tickets.")
        tracker = JiraTracker(workflow=workflow, dev_agent=dev_agent, crew_builder=mock_builder)
        assert tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK) == []

    def test_skips_items_with_missing_id(self, _patch_task: MagicMock) -> None:
        """UNIT-JT-09: Items with empty or missing 'id' are silently skipped."""
        tickets = [
            {"title": "No ID", "url": "", "status": ""},
            {"id": "", "title": "Empty", "url": "", "status": ""},
            {"id": "P-1", "title": "Valid", "url": "", "status": ""},
        ]
        workflow, dev_agent, mock_builder = _make_parts(json.dumps(tickets))
        tracker = JiraTracker(workflow=workflow, dev_agent=dev_agent, crew_builder=mock_builder)
        result = tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK)
        assert len(result) == 1 and result[0].id == "P-1"

    def test_title_falls_back_to_id_when_missing(self, _patch_task: MagicMock) -> None:
        """UNIT-JT-10: Missing 'title' falls back to ticket ID."""
        workflow, dev_agent, mock_builder = _make_parts(
            json.dumps([{"id": "P-5", "url": "", "status": ""}])
        )
        tracker = JiraTracker(workflow=workflow, dev_agent=dev_agent, crew_builder=mock_builder)
        result = tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK)
        assert result[0].title == "P-5"


# ===========================================================================
# transition
# ===========================================================================


class TestTransition:
    def test_builds_crew_for_transition(self, _patch_task: MagicMock) -> None:
        """UNIT-JT-11: transition builds and kicks off exactly one crew."""
        workflow, dev_agent, mock_builder = _make_parts("ok")
        tracker = JiraTracker(workflow=workflow, dev_agent=dev_agent, crew_builder=mock_builder)
        tracker.transition("PROJ-1", WorkflowOperation.START_DEVELOPMENT)
        mock_builder.build.assert_called_once()

    def test_task_description_contains_ticket_id(self, _patch_task: MagicMock) -> None:
        """UNIT-JT-12: Task description includes the ticket ID."""
        workflow, dev_agent, mock_builder = _make_parts("ok")
        tracker = JiraTracker(workflow=workflow, dev_agent=dev_agent, crew_builder=mock_builder)
        tracker.transition("PROJ-42", WorkflowOperation.START_DEVELOPMENT)
        assert "PROJ-42" in _patch_task.call_args[1]["description"]

    def test_task_description_contains_target_status(self, _patch_task: MagicMock) -> None:
        """UNIT-JT-13: Task description contains the resolved status string."""
        workflow, dev_agent, mock_builder = _make_parts("ok")
        tracker = JiraTracker(workflow=workflow, dev_agent=dev_agent, crew_builder=mock_builder)
        tracker.transition("PROJ-42", WorkflowOperation.START_DEVELOPMENT)
        assert "IN PROGRESS" in _patch_task.call_args[1]["description"]

    def test_transition_raises_permission_error_for_human_only_op(self, _patch_task: MagicMock) -> None:
        """UNIT-JT-14: PermissionError raised for human_only operation (BR-1)."""
        workflow, dev_agent, mock_builder = _make_parts("ok")
        tracker = JiraTracker(workflow=workflow, dev_agent=dev_agent, crew_builder=mock_builder)
        with pytest.raises(PermissionError):
            tracker.transition("PROJ-1", WorkflowOperation.SCAN_FOR_WORK)


# ===========================================================================
# add_comment
# ===========================================================================


class TestAddComment:
    def test_builds_crew_for_comment(self, _patch_task: MagicMock) -> None:
        """UNIT-JT-15: add_comment builds and kicks off exactly one crew."""
        workflow, dev_agent, mock_builder = _make_parts("ok")
        tracker = JiraTracker(workflow=workflow, dev_agent=dev_agent, crew_builder=mock_builder)
        tracker.add_comment("PROJ-1", "A comment.")
        mock_builder.build.assert_called_once()

    def test_task_description_contains_ticket_id(self, _patch_task: MagicMock) -> None:
        """UNIT-JT-16: Task description includes the ticket ID."""
        workflow, dev_agent, mock_builder = _make_parts("ok")
        tracker = JiraTracker(workflow=workflow, dev_agent=dev_agent, crew_builder=mock_builder)
        tracker.add_comment("PROJ-77", "Body.")
        assert "PROJ-77" in _patch_task.call_args[1]["description"]

    def test_task_description_contains_comment_body(self, _patch_task: MagicMock) -> None:
        """UNIT-JT-17: Task description includes the full comment body."""
        workflow, dev_agent, mock_builder = _make_parts("ok")
        tracker = JiraTracker(workflow=workflow, dev_agent=dev_agent, crew_builder=mock_builder)
        tracker.add_comment("PROJ-77", "Thread summary: deployed OAuth2.")
        assert "Thread summary: deployed OAuth2." in _patch_task.call_args[1]["description"]


# ===========================================================================
# _parse_ticket_records — static method, no Task involved
# ===========================================================================

class TestParseTicketRecords:
    def test_parses_valid_json_list(self) -> None:
        """UNIT-JT-18: _parse_ticket_records correctly parses a valid JSON array."""
        raw = json.dumps([{"id": "X-1", "title": "T", "url": "u", "status": "s"}])
        result = JiraTracker._parse_ticket_records(raw, source="jira")
        assert len(result) == 1
        assert result[0].id == "X-1"
        assert result[0].source == "jira"

    def test_returns_empty_list_on_invalid_json(self) -> None:
        """UNIT-JT-19: Invalid JSON returns an empty list (no exception raised)."""
        assert JiraTracker._parse_ticket_records("not json", source="jira") == []

    def test_strips_code_fence_before_parsing(self) -> None:
        """UNIT-JT-20: Backtick code fence is stripped before JSON parsing."""
        raw = "```\n" + json.dumps([{"id": "X-2", "title": "T", "url": "", "status": ""}]) + "\n```"
        result = JiraTracker._parse_ticket_records(raw, source="jira")
        assert len(result) == 1 and result[0].id == "X-2"

