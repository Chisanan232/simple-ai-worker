"""
Unit tests for :mod:`src.ticket.clickup_tracker` (UNIT-CT-01 through UNIT-CT-20).

UNIT-CT-01 through UNIT-CT-10 cover the new direct REST API fetch path
(ClickUpRestClient.search_tasks is called directly — no LLM crew).

UNIT-CT-11 through UNIT-CT-20 cover transition, add_comment, and
_parse_ticket_records (still crew-driven — unchanged).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.ticket.clickup_tracker import ClickUpTracker
from src.ticket.rest_client import ClickUpRestClient, TicketFetchError
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


def _make_parts(
    rest_return: list | None = None,
    crew_result: str = "ok",
) -> tuple:
    """Return (tracker, mock_rest_client, mock_builder).

    *rest_return* is the list returned by ``ClickUpRestClient.search_tasks``.
    *crew_result* is the string returned by ``crew.kickoff()`` for write ops.
    """
    workflow = WorkflowConfig(_WORKFLOW_CFG)
    dev_agent = MagicMock()

    mock_rest = MagicMock(spec=ClickUpRestClient)
    mock_rest.search_tasks.return_value = rest_return if rest_return is not None else []

    mock_crew = MagicMock()
    mock_crew.kickoff.return_value = crew_result
    mock_builder = MagicMock()
    mock_builder.build.return_value = mock_crew

    tracker = ClickUpTracker(
        workflow=workflow,
        dev_agent=dev_agent,
        crew_builder=mock_builder,
        rest_client=mock_rest,
    )
    return tracker, mock_rest, mock_builder


@pytest.fixture(autouse=True)
def _patch_task():
    """Patch crewai.Task inside clickup_tracker for write-op tests."""
    with patch("src.ticket.clickup_tracker.Task") as mock_task_cls:
        mock_task_cls.return_value = MagicMock()
        yield mock_task_cls


# ===========================================================================
# fetch_tickets_for_operation — direct REST API (UNIT-CT-01 through UNIT-CT-10)
# ===========================================================================


class TestFetchTicketsForOperation:
    def test_calls_rest_client_not_crew(self, _patch_task: MagicMock) -> None:
        """UNIT-CT-01: fetch_tickets_for_operation calls REST client, not crew builder."""
        tracker, mock_rest, mock_builder = _make_parts([])
        tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK)
        mock_rest.search_tasks.assert_called_once()
        mock_builder.build.assert_not_called()

    def test_passes_scan_status_to_rest_client(self, _patch_task: MagicMock) -> None:
        """UNIT-CT-02: search_tasks is called with the SCAN_FOR_WORK status string."""
        tracker, mock_rest, _ = _make_parts([])
        tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK)
        call_kwargs = mock_rest.search_tasks.call_args
        assert call_kwargs.kwargs.get("status") == "ACCEPTED" or (
            call_kwargs.args and call_kwargs.args[0] == "ACCEPTED"
        )

    def test_passes_exclude_status_to_rest_client(self, _patch_task: MagicMock) -> None:
        """UNIT-CT-03: search_tasks is called with the SKIP_REJECTED status string."""
        tracker, mock_rest, _ = _make_parts([])
        tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK)
        call_kwargs = mock_rest.search_tasks.call_args
        assert call_kwargs.kwargs.get("exclude_status") == "REJECTED" or (
            len(call_kwargs.args) > 1 and call_kwargs.args[1] == "REJECTED"
        )

    def test_returns_empty_list_on_empty_api_response(self, _patch_task: MagicMock) -> None:
        """UNIT-CT-04: Empty list from REST client returns empty list."""
        tracker, _, _ = _make_parts([])
        assert tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK) == []

    def test_maps_api_items_to_ticket_records(self, _patch_task: MagicMock) -> None:
        """UNIT-CT-05: API response items are mapped to TicketRecord objects."""
        items = [
            {"id": "cu-1", "title": "Build API", "url": "https://app.clickup.com/t/cu-1", "status": "ACCEPTED"},
            {"id": "cu-2", "title": "Write tests", "url": "https://app.clickup.com/t/cu-2", "status": "ACCEPTED"},
        ]
        tracker, _, _ = _make_parts(items)
        result = tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK)
        assert len(result) == 2
        assert result[0].id == "cu-1"
        assert result[0].source == "clickup"
        assert result[0].title == "Build API"
        assert result[1].id == "cu-2"

    def test_source_is_always_clickup(self, _patch_task: MagicMock) -> None:
        """UNIT-CT-06: All returned TicketRecords have source='clickup'."""
        tracker, _, _ = _make_parts([{"id": "cu-1", "title": "T", "url": "", "status": "ACCEPTED"}])
        result = tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK)
        assert all(r.source == "clickup" for r in result)

    def test_br3_guard_filters_rejected_items(self, _patch_task: MagicMock) -> None:
        """UNIT-CT-07: Belt-and-suspenders BR-3 — REJECTED items filtered post-fetch."""
        items = [
            {"id": "cu-1", "title": "Good", "url": "", "status": "ACCEPTED"},
            {"id": "cu-2", "title": "Bad", "url": "", "status": "REJECTED"},
        ]
        tracker, _, _ = _make_parts(items)
        result = tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK)
        assert len(result) == 1
        assert result[0].id == "cu-1"

    def test_skips_items_with_missing_id(self, _patch_task: MagicMock) -> None:
        """UNIT-CT-08: Items without 'id' are silently skipped."""
        items = [
            {"title": "No ID", "url": "", "status": ""},
            {"id": "cu-1", "title": "Valid", "url": "", "status": "ACCEPTED"},
        ]
        tracker, _, _ = _make_parts(items)
        result = tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK)
        assert len(result) == 1 and result[0].id == "cu-1"

    def test_title_falls_back_to_id_when_missing(self, _patch_task: MagicMock) -> None:
        """UNIT-CT-09: Missing 'title' falls back to the task ID."""
        tracker, _, _ = _make_parts([{"id": "cu-9", "url": "", "status": "ACCEPTED"}])
        result = tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK)
        assert result[0].title == "cu-9"

    def test_ticket_fetch_error_propagates(self, _patch_task: MagicMock) -> None:
        """UNIT-CT-10: TicketFetchError from REST client propagates to caller."""
        tracker, mock_rest, _ = _make_parts()
        mock_rest.search_tasks.side_effect = TicketFetchError("API 503", source="clickup", status_code=503)
        with pytest.raises(TicketFetchError):
            tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK)


# ===========================================================================
# transition — still crew-driven (UNIT-CT-11 through UNIT-CT-14)
# ===========================================================================


class TestTransition:
    def test_builds_crew_for_transition(self, _patch_task: MagicMock) -> None:
        """UNIT-CT-11: transition builds and kicks off exactly one crew."""
        tracker, _, mock_builder = _make_parts(crew_result="ok")
        tracker.transition("cu-1", WorkflowOperation.START_DEVELOPMENT)
        mock_builder.build.assert_called_once()

    def test_task_description_contains_ticket_id(self, _patch_task: MagicMock) -> None:
        """UNIT-CT-12: Task description includes the task ID."""
        tracker, _, _ = _make_parts(crew_result="ok")
        tracker.transition("cu-42", WorkflowOperation.START_DEVELOPMENT)
        assert "cu-42" in _patch_task.call_args[1]["description"]

    def test_task_description_contains_target_status(self, _patch_task: MagicMock) -> None:
        """UNIT-CT-13: Task description contains the resolved status string."""
        tracker, _, _ = _make_parts(crew_result="ok")
        tracker.transition("cu-42", WorkflowOperation.OPEN_FOR_REVIEW)
        assert "IN REVIEW" in _patch_task.call_args[1]["description"]

    def test_transition_raises_permission_error_for_human_only_op(self, _patch_task: MagicMock) -> None:
        """UNIT-CT-14: BR-1 — PermissionError raised for human_only operation."""
        tracker, _, _ = _make_parts(crew_result="ok")
        with pytest.raises(PermissionError):
            tracker.transition("cu-1", WorkflowOperation.SCAN_FOR_WORK)


# ===========================================================================
# add_comment — still crew-driven (UNIT-CT-15 through UNIT-CT-17)
# ===========================================================================


class TestAddComment:
    def test_builds_crew_for_comment(self, _patch_task: MagicMock) -> None:
        """UNIT-CT-15: add_comment builds and kicks off exactly one crew."""
        tracker, _, mock_builder = _make_parts(crew_result="ok")
        tracker.add_comment("cu-1", "A comment.")
        mock_builder.build.assert_called_once()

    def test_task_description_contains_ticket_id(self, _patch_task: MagicMock) -> None:
        """UNIT-CT-16: Task description includes the task ID."""
        tracker, _, _ = _make_parts(crew_result="ok")
        tracker.add_comment("cu-55", "Some comment.")
        assert "cu-55" in _patch_task.call_args[1]["description"]

    def test_task_description_contains_comment_body(self, _patch_task: MagicMock) -> None:
        """UNIT-CT-17: Task description includes the full comment text."""
        tracker, _, _ = _make_parts(crew_result="ok")
        tracker.add_comment("cu-55", "Sprint summary: delivered feature X.")
        assert "Sprint summary: delivered feature X." in _patch_task.call_args[1]["description"]


# ===========================================================================
# _parse_ticket_records_from_api (new static method — UNIT-CT-18 through UNIT-CT-20)
# ===========================================================================


class TestParseTicketRecordsFromApi:
    def test_parses_normalised_dict_list(self) -> None:
        """UNIT-CT-18: _parse_ticket_records_from_api maps dicts to TicketRecords."""
        items = [{"id": "cu-1", "title": "T", "url": "u", "status": "ACCEPTED"}]
        result = ClickUpTracker._parse_ticket_records_from_api(items, source="clickup")
        assert len(result) == 1
        assert result[0].id == "cu-1"
        assert result[0].source == "clickup"

    def test_skips_items_with_missing_id(self) -> None:
        """UNIT-CT-19: Items without 'id' are silently skipped."""
        items = [{"title": "No id", "url": "", "status": ""}, {"id": "cu-2", "title": "T", "url": "", "status": ""}]
        result = ClickUpTracker._parse_ticket_records_from_api(items, source="clickup")
        assert len(result) == 1 and result[0].id == "cu-2"

    def test_legacy_parse_ticket_records_still_works(self) -> None:
        """UNIT-CT-20: Legacy _parse_ticket_records (JSON string) still parses correctly."""
        raw = json.dumps([{"id": "cu-2", "title": "T", "url": "", "status": ""}])
        result = ClickUpTracker._parse_ticket_records(raw, source="clickup")
        assert len(result) == 1 and result[0].id == "cu-2"
