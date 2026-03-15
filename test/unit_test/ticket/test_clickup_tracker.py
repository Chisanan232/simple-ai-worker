"""
Unit tests for :mod:`src.ticket.clickup_tracker` (UNIT-CT-01 through UNIT-CT-20).

Mirrors the JiraTracker test structure but validates ClickUp-specific
task descriptions (clickup/search_tasks, clickup/update_task, clickup/add_comment).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.ticket.clickup_tracker import ClickUpTracker
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
    """Return (workflow, dev_agent, mock_builder). Caller must have Task patched."""
    workflow = WorkflowConfig(_WORKFLOW_CFG)
    dev_agent = MagicMock()
    mock_crew = MagicMock()
    mock_crew.kickoff.return_value = crew_result
    mock_builder = MagicMock()
    mock_builder.build.return_value = mock_crew
    return workflow, dev_agent, mock_builder


@pytest.fixture(autouse=True)
def _patch_task():
    """Patch crewai.Task inside clickup_tracker so MagicMock agent passes validation."""
    with patch("src.ticket.clickup_tracker.Task") as mock_task_cls:
        mock_task_cls.return_value = MagicMock()
        yield mock_task_cls


# ===========================================================================
# fetch_tickets_for_operation
# ===========================================================================


class TestFetchTicketsForOperation:
    def test_builds_crew_once(self, _patch_task: MagicMock) -> None:
        """UNIT-CT-01: A single crew is built and kicked off for a fetch."""
        workflow, dev_agent, mock_builder = _make_parts("[]")
        tracker = ClickUpTracker(workflow=workflow, dev_agent=dev_agent, crew_builder=mock_builder)
        tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK)
        mock_builder.build.assert_called_once()

    def test_task_description_contains_scan_status(self, _patch_task: MagicMock) -> None:
        """UNIT-CT-02: Task description contains 'ACCEPTED' (scan_for_work status)."""
        workflow, dev_agent, mock_builder = _make_parts("[]")
        tracker = ClickUpTracker(workflow=workflow, dev_agent=dev_agent, crew_builder=mock_builder)
        tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK)
        assert "ACCEPTED" in _patch_task.call_args[1]["description"]

    def test_task_description_contains_skip_status(self, _patch_task: MagicMock) -> None:
        """UNIT-CT-03: Task description contains 'REJECTED' (skip_rejected status)."""
        workflow, dev_agent, mock_builder = _make_parts("[]")
        tracker = ClickUpTracker(workflow=workflow, dev_agent=dev_agent, crew_builder=mock_builder)
        tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK)
        assert "REJECTED" in _patch_task.call_args[1]["description"]

    def test_returns_empty_list_on_empty_crew_output(self, _patch_task: MagicMock) -> None:
        """UNIT-CT-04: Empty JSON array from crew returns empty list."""
        workflow, dev_agent, mock_builder = _make_parts("[]")
        tracker = ClickUpTracker(workflow=workflow, dev_agent=dev_agent, crew_builder=mock_builder)
        assert tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK) == []

    def test_returns_ticket_records_from_crew_output(self, _patch_task: MagicMock) -> None:
        """UNIT-CT-05: Valid JSON is parsed into TicketRecord objects."""
        tasks = [
            {"id": "cu-1", "title": "Build API", "url": "https://app.clickup.com/t/cu-1", "status": "ACCEPTED"},
            {"id": "cu-2", "title": "Write tests", "url": "https://app.clickup.com/t/cu-2", "status": "ACCEPTED"},
        ]
        workflow, dev_agent, mock_builder = _make_parts(json.dumps(tasks))
        tracker = ClickUpTracker(workflow=workflow, dev_agent=dev_agent, crew_builder=mock_builder)
        result = tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK)
        assert len(result) == 2
        assert result[0].id == "cu-1"
        assert result[0].source == "clickup"
        assert result[0].title == "Build API"
        assert result[1].id == "cu-2"

    def test_source_is_always_clickup(self, _patch_task: MagicMock) -> None:
        """UNIT-CT-06: All returned TicketRecords have source='clickup'."""
        workflow, dev_agent, mock_builder = _make_parts(
            json.dumps([{"id": "cu-1", "title": "T", "url": "", "status": ""}])
        )
        tracker = ClickUpTracker(workflow=workflow, dev_agent=dev_agent, crew_builder=mock_builder)
        result = tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK)
        assert all(r.source == "clickup" for r in result)

    def test_strips_markdown_code_fence(self, _patch_task: MagicMock) -> None:
        """UNIT-CT-07: Markdown fence is stripped before JSON parsing."""
        tasks = [{"id": "cu-1", "title": "T", "url": "", "status": ""}]
        fenced = "```json\n" + json.dumps(tasks) + "\n```"
        workflow, dev_agent, mock_builder = _make_parts(fenced)
        tracker = ClickUpTracker(workflow=workflow, dev_agent=dev_agent, crew_builder=mock_builder)
        result = tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK)
        assert len(result) == 1

    def test_returns_empty_list_on_invalid_json(self, _patch_task: MagicMock) -> None:
        """UNIT-CT-08: Non-JSON crew output returns empty list."""
        workflow, dev_agent, mock_builder = _make_parts("No tasks found.")
        tracker = ClickUpTracker(workflow=workflow, dev_agent=dev_agent, crew_builder=mock_builder)
        assert tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK) == []

    def test_skips_items_with_missing_id(self, _patch_task: MagicMock) -> None:
        """UNIT-CT-09: Items without 'id' are silently skipped."""
        tasks = [
            {"title": "No ID", "url": "", "status": ""},
            {"id": "cu-1", "title": "Valid", "url": "", "status": ""},
        ]
        workflow, dev_agent, mock_builder = _make_parts(json.dumps(tasks))
        tracker = ClickUpTracker(workflow=workflow, dev_agent=dev_agent, crew_builder=mock_builder)
        result = tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK)
        assert len(result) == 1 and result[0].id == "cu-1"

    def test_title_falls_back_to_id_when_missing(self, _patch_task: MagicMock) -> None:
        """UNIT-CT-10: Missing 'title' falls back to the task ID."""
        workflow, dev_agent, mock_builder = _make_parts(
            json.dumps([{"id": "cu-9", "url": "", "status": ""}])
        )
        tracker = ClickUpTracker(workflow=workflow, dev_agent=dev_agent, crew_builder=mock_builder)
        result = tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK)
        assert result[0].title == "cu-9"


# ===========================================================================
# transition
# ===========================================================================


class TestTransition:
    def test_builds_crew_for_transition(self, _patch_task: MagicMock) -> None:
        """UNIT-CT-11: transition builds and kicks off exactly one crew."""
        workflow, dev_agent, mock_builder = _make_parts("ok")
        tracker = ClickUpTracker(workflow=workflow, dev_agent=dev_agent, crew_builder=mock_builder)
        tracker.transition("cu-1", WorkflowOperation.START_DEVELOPMENT)
        mock_builder.build.assert_called_once()

    def test_task_description_contains_ticket_id(self, _patch_task: MagicMock) -> None:
        """UNIT-CT-12: Task description includes the task ID."""
        workflow, dev_agent, mock_builder = _make_parts("ok")
        tracker = ClickUpTracker(workflow=workflow, dev_agent=dev_agent, crew_builder=mock_builder)
        tracker.transition("cu-42", WorkflowOperation.START_DEVELOPMENT)
        assert "cu-42" in _patch_task.call_args[1]["description"]

    def test_task_description_contains_target_status(self, _patch_task: MagicMock) -> None:
        """UNIT-CT-13: Task description contains the resolved status string."""
        workflow, dev_agent, mock_builder = _make_parts("ok")
        tracker = ClickUpTracker(workflow=workflow, dev_agent=dev_agent, crew_builder=mock_builder)
        tracker.transition("cu-42", WorkflowOperation.OPEN_FOR_REVIEW)
        assert "IN REVIEW" in _patch_task.call_args[1]["description"]

    def test_transition_raises_permission_error_for_human_only_op(self, _patch_task: MagicMock) -> None:
        """UNIT-CT-14: BR-1 — PermissionError raised for human_only operation."""
        workflow, dev_agent, mock_builder = _make_parts("ok")
        tracker = ClickUpTracker(workflow=workflow, dev_agent=dev_agent, crew_builder=mock_builder)
        with pytest.raises(PermissionError):
            tracker.transition("cu-1", WorkflowOperation.SCAN_FOR_WORK)


# ===========================================================================
# add_comment
# ===========================================================================


class TestAddComment:
    def test_builds_crew_for_comment(self, _patch_task: MagicMock) -> None:
        """UNIT-CT-15: add_comment builds and kicks off exactly one crew."""
        workflow, dev_agent, mock_builder = _make_parts("ok")
        tracker = ClickUpTracker(workflow=workflow, dev_agent=dev_agent, crew_builder=mock_builder)
        tracker.add_comment("cu-1", "A comment.")
        mock_builder.build.assert_called_once()

    def test_task_description_contains_ticket_id(self, _patch_task: MagicMock) -> None:
        """UNIT-CT-16: Task description includes the task ID."""
        workflow, dev_agent, mock_builder = _make_parts("ok")
        tracker = ClickUpTracker(workflow=workflow, dev_agent=dev_agent, crew_builder=mock_builder)
        tracker.add_comment("cu-55", "Some comment.")
        assert "cu-55" in _patch_task.call_args[1]["description"]

    def test_task_description_contains_comment_body(self, _patch_task: MagicMock) -> None:
        """UNIT-CT-17: Task description includes the full comment text."""
        workflow, dev_agent, mock_builder = _make_parts("ok")
        tracker = ClickUpTracker(workflow=workflow, dev_agent=dev_agent, crew_builder=mock_builder)
        tracker.add_comment("cu-55", "Sprint summary: delivered feature X.")
        assert "Sprint summary: delivered feature X." in _patch_task.call_args[1]["description"]


# ===========================================================================
# _parse_ticket_records (static method — no Task involved)
# ===========================================================================


class TestParseTicketRecords:
    def test_parses_valid_json_list(self) -> None:
        """UNIT-CT-18: _parse_ticket_records correctly parses a valid JSON array."""
        raw = json.dumps([{"id": "cu-1", "title": "T", "url": "u", "status": "s"}])
        result = ClickUpTracker._parse_ticket_records(raw, source="clickup")
        assert len(result) == 1
        assert result[0].id == "cu-1"
        assert result[0].source == "clickup"

    def test_returns_empty_list_on_invalid_json(self) -> None:
        """UNIT-CT-19: Invalid JSON returns empty list without raising."""
        assert ClickUpTracker._parse_ticket_records("not json at all", source="clickup") == []

    def test_strips_code_fence_before_parsing(self) -> None:
        """UNIT-CT-20: Backtick code fence is stripped before JSON parsing."""
        raw = "```\n" + json.dumps([{"id": "cu-2", "title": "T", "url": "", "status": ""}]) + "\n```"
        result = ClickUpTracker._parse_ticket_records(raw, source="clickup")
        assert len(result) == 1 and result[0].id == "cu-2"



