"""
Unit tests for :mod:`src.ticket.jira_tracker` (UNIT-JT-01 through UNIT-JT-20).

UNIT-JT-01 through UNIT-JT-10 cover the new direct REST API fetch path
(JiraRestClient.search_issues is called directly — no LLM crew).

UNIT-JT-11 through UNIT-JT-20 cover transition, add_comment, and
_parse_ticket_records (still crew-driven — unchanged).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.ticket.jira_tracker import JiraTracker
from src.ticket.rest_client import JiraRestClient, TicketFetchError
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
    project_key: str | None = None,
) -> tuple:
    """Return (tracker, mock_rest, mock_builder).

    *rest_return* is the list returned by ``JiraRestClient.search_issues``.
    *crew_result* is the string returned by ``crew.kickoff()`` for write ops.
    """
    workflow = WorkflowConfig(_WORKFLOW_CFG)
    dev_agent = MagicMock()

    mock_rest = MagicMock(spec=JiraRestClient)
    mock_rest.search_issues.return_value = rest_return if rest_return is not None else []

    mock_crew = MagicMock()
    mock_crew.kickoff.return_value = crew_result
    mock_builder = MagicMock()
    mock_builder.build.return_value = mock_crew

    tracker = JiraTracker(
        workflow=workflow,
        dev_agent=dev_agent,
        crew_builder=mock_builder,
        rest_client=mock_rest,
        project_key=project_key,
    )
    return tracker, mock_rest, mock_builder


@pytest.fixture(autouse=True)
def _patch_task():
    """Patch crewai.Task inside jira_tracker so MagicMock agent passes Pydantic validation."""
    with patch("src.ticket.jira_tracker.Task") as mock_task_cls:
        mock_task_cls.return_value = MagicMock()
        yield mock_task_cls


# ===========================================================================
# fetch_tickets_for_operation — direct REST API (UNIT-JT-01 through UNIT-JT-10)
# ===========================================================================


class TestFetchTicketsForOperation:
    def test_calls_rest_client_not_crew(self, _patch_task: MagicMock) -> None:
        """UNIT-JT-01: fetch_tickets_for_operation calls REST client, not crew builder."""
        tracker, mock_rest, mock_builder = _make_parts([])
        tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK)
        mock_rest.search_issues.assert_called_once()
        mock_builder.build.assert_not_called()

    def test_passes_scan_status_to_rest_client(self, _patch_task: MagicMock) -> None:
        """UNIT-JT-02: search_issues is called with the SCAN_FOR_WORK status string."""
        tracker, mock_rest, _ = _make_parts([])
        tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK)
        call_kwargs = mock_rest.search_issues.call_args
        assert call_kwargs.kwargs.get("status") == "ACCEPTED" or (
            call_kwargs.args and call_kwargs.args[0] == "ACCEPTED"
        )

    def test_passes_exclude_status_to_rest_client(self, _patch_task: MagicMock) -> None:
        """UNIT-JT-03: search_issues is called with the SKIP_REJECTED status string."""
        tracker, mock_rest, _ = _make_parts([])
        tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK)
        call_kwargs = mock_rest.search_issues.call_args
        assert call_kwargs.kwargs.get("exclude_status") == "REJECTED" or (
            len(call_kwargs.args) > 1 and call_kwargs.args[1] == "REJECTED"
        )

    def test_returns_empty_list_on_empty_api_response(self, _patch_task: MagicMock) -> None:
        """UNIT-JT-04: Empty list from REST client returns empty list."""
        tracker, _, _ = _make_parts([])
        assert tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK) == []

    def test_maps_api_items_to_ticket_records(self, _patch_task: MagicMock) -> None:
        """UNIT-JT-05: API response items are mapped to TicketRecord objects."""
        items = [
            {"id": "PROJ-1", "title": "Login", "url": "https://j.io/PROJ-1", "status": "ACCEPTED"},
            {"id": "PROJ-2", "title": "OAuth", "url": "https://j.io/PROJ-2", "status": "ACCEPTED"},
        ]
        tracker, _, _ = _make_parts(items)
        result = tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK)
        assert len(result) == 2
        assert result[0].id == "PROJ-1"
        assert result[0].source == "jira"
        assert result[0].title == "Login"
        assert result[1].id == "PROJ-2"

    def test_source_is_always_jira(self, _patch_task: MagicMock) -> None:
        """UNIT-JT-06: All returned TicketRecords have source='jira'."""
        tracker, _, _ = _make_parts([{"id": "P-1", "title": "T", "url": "", "status": "ACCEPTED"}])
        result = tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK)
        assert all(r.source == "jira" for r in result)

    def test_br3_guard_filters_rejected_items(self, _patch_task: MagicMock) -> None:
        """UNIT-JT-07: Belt-and-suspenders BR-3 — REJECTED items filtered post-fetch."""
        items = [
            {"id": "P-1", "title": "Good", "url": "", "status": "ACCEPTED"},
            {"id": "P-2", "title": "Bad", "url": "", "status": "REJECTED"},
        ]
        tracker, _, _ = _make_parts(items)
        result = tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK)
        assert len(result) == 1 and result[0].id == "P-1"

    def test_skips_items_with_missing_id(self, _patch_task: MagicMock) -> None:
        """UNIT-JT-08: Items without 'id' are silently skipped."""
        items = [
            {"title": "No ID", "url": "", "status": ""},
            {"id": "P-1", "title": "Valid", "url": "", "status": "ACCEPTED"},
        ]
        tracker, _, _ = _make_parts(items)
        result = tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK)
        assert len(result) == 1 and result[0].id == "P-1"

    def test_title_falls_back_to_id_when_missing(self, _patch_task: MagicMock) -> None:
        """UNIT-JT-09: Missing 'title' falls back to ticket ID."""
        tracker, _, _ = _make_parts([{"id": "P-5", "url": "", "status": "ACCEPTED"}])
        result = tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK)
        assert result[0].title == "P-5"

    def test_ticket_fetch_error_propagates(self, _patch_task: MagicMock) -> None:
        """UNIT-JT-10: TicketFetchError from REST client propagates to caller."""
        tracker, mock_rest, _ = _make_parts()
        mock_rest.search_issues.side_effect = TicketFetchError("JIRA 401", source="jira", status_code=401)
        with pytest.raises(TicketFetchError):
            tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK)


# ===========================================================================
# transition — still crew-driven (UNIT-JT-11 through UNIT-JT-14)
# ===========================================================================


class TestTransition:
    def test_builds_crew_for_transition(self, _patch_task: MagicMock) -> None:
        """UNIT-JT-11: transition builds and kicks off exactly one crew."""
        tracker, _, mock_builder = _make_parts(crew_result="ok")
        tracker.transition("PROJ-1", WorkflowOperation.START_DEVELOPMENT)
        mock_builder.build.assert_called_once()

    def test_task_description_contains_ticket_id(self, _patch_task: MagicMock) -> None:
        """UNIT-JT-12: Task description includes the ticket ID."""
        tracker, _, _ = _make_parts(crew_result="ok")
        tracker.transition("PROJ-42", WorkflowOperation.START_DEVELOPMENT)
        assert "PROJ-42" in _patch_task.call_args[1]["description"]

    def test_task_description_contains_target_status(self, _patch_task: MagicMock) -> None:
        """UNIT-JT-13: Task description contains the resolved status string."""
        tracker, _, _ = _make_parts(crew_result="ok")
        tracker.transition("PROJ-42", WorkflowOperation.START_DEVELOPMENT)
        assert "IN PROGRESS" in _patch_task.call_args[1]["description"]

    def test_transition_raises_permission_error_for_human_only_op(self, _patch_task: MagicMock) -> None:
        """UNIT-JT-14: PermissionError raised for human_only operation (BR-1)."""
        tracker, _, _ = _make_parts(crew_result="ok")
        with pytest.raises(PermissionError):
            tracker.transition("PROJ-1", WorkflowOperation.SCAN_FOR_WORK)


# ===========================================================================
# add_comment — still crew-driven (UNIT-JT-15 through UNIT-JT-17)
# ===========================================================================


class TestAddComment:
    def test_builds_crew_for_comment(self, _patch_task: MagicMock) -> None:
        """UNIT-JT-15: add_comment builds and kicks off exactly one crew."""
        tracker, _, mock_builder = _make_parts(crew_result="ok")
        tracker.add_comment("PROJ-1", "A comment.")
        mock_builder.build.assert_called_once()

    def test_task_description_contains_ticket_id(self, _patch_task: MagicMock) -> None:
        """UNIT-JT-16: Task description includes the ticket ID."""
        tracker, _, _ = _make_parts(crew_result="ok")
        tracker.add_comment("PROJ-77", "Body.")
        assert "PROJ-77" in _patch_task.call_args[1]["description"]

    def test_task_description_contains_comment_body(self, _patch_task: MagicMock) -> None:
        """UNIT-JT-17: Task description includes the full comment body."""
        tracker, _, _ = _make_parts(crew_result="ok")
        tracker.add_comment("PROJ-77", "Thread summary: deployed OAuth2.")
        assert "Thread summary: deployed OAuth2." in _patch_task.call_args[1]["description"]


# ===========================================================================
# _parse_ticket_records_from_api and _parse_ticket_records
# ===========================================================================


class TestParseTicketRecordsFromApi:
    def test_parses_normalised_dict_list(self) -> None:
        """UNIT-JT-18: _parse_ticket_records_from_api maps dicts to TicketRecords."""
        items = [{"id": "X-1", "title": "T", "url": "u", "status": "s"}]
        result = JiraTracker._parse_ticket_records_from_api(items, source="jira")
        assert len(result) == 1
        assert result[0].id == "X-1"
        assert result[0].source == "jira"

    def test_skips_items_with_missing_id(self) -> None:
        """UNIT-JT-19: Items without 'id' are silently skipped."""
        items = [{"title": "No id", "url": "", "status": ""}, {"id": "X-2", "title": "T", "url": "", "status": ""}]
        result = JiraTracker._parse_ticket_records_from_api(items, source="jira")
        assert len(result) == 1 and result[0].id == "X-2"

    def test_legacy_parse_ticket_records_still_works(self) -> None:
        """UNIT-JT-20: Legacy _parse_ticket_records (JSON string) still parses correctly."""
        raw = json.dumps([{"id": "X-2", "title": "T", "url": "", "status": ""}])
        result = JiraTracker._parse_ticket_records(raw, source="jira")
        assert len(result) == 1 and result[0].id == "X-2"
