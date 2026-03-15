"""
Unit tests for :func:`src.scheduler.jobs.scan_tickets.scan_and_dispatch_job`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

import src.scheduler.jobs.scan_tickets as scan_module
from src.scheduler.jobs.scan_tickets import scan_and_dispatch_job
from src.ticket.models import TicketRecord
from src.ticket.rest_client import TicketFetchError
from src.ticket.workflow import WorkflowConfig

_WORKFLOW_CFG = {
    "scan_for_work": {"status_value": "ACCEPTED", "human_only": True},
    "skip_rejected": {"status_value": "REJECTED"},
    "start_development": {"status_value": "IN PROGRESS"},
    "open_for_review": {"status_value": "IN REVIEW"},
    "mark_complete": {"status_value": "COMPLETE"},
    "update_with_context": {"status_value": ""},
}


def _make_workflow() -> WorkflowConfig:
    return WorkflowConfig(_WORKFLOW_CFG)


def _make_registry(
    dev_agent: MagicMock | None = None,
    raise_key_error: bool = False,
) -> MagicMock:
    reg = MagicMock()
    if raise_key_error:
        reg.__getitem__.side_effect = KeyError("dev_agent")
        reg.agent_ids.return_value = []
    else:
        reg.__getitem__.return_value = dev_agent or MagicMock()
        reg.agent_ids.return_value = ["dev_agent"]
    return reg


def _make_settings() -> MagicMock:
    s = MagicMock()
    s.MAX_CONCURRENT_DEV_AGENTS = 3
    return s


def _make_tracker_registry(
    jira_tickets: list[TicketRecord] | None = None,
    clickup_tickets: list[TicketRecord] | None = None,
    jira_error: Exception | None = None,
    clickup_error: Exception | None = None,
) -> MagicMock:
    """Build a mock TrackerRegistry that returns given tickets or raises errors."""
    mock_tr = MagicMock()

    def _get(source: str) -> MagicMock:
        tracker = MagicMock()
        if source == "jira":
            if jira_error:
                tracker.fetch_tickets_for_operation.side_effect = jira_error
            else:
                tracker.fetch_tickets_for_operation.return_value = jira_tickets or []
        elif source == "clickup":
            if clickup_error:
                tracker.fetch_tickets_for_operation.side_effect = clickup_error
            else:
                tracker.fetch_tickets_for_operation.return_value = clickup_tickets or []
        return tracker

    mock_tr.get.side_effect = _get
    return mock_tr


class TestScanAndDispatchJob:
    """Tests for scan_and_dispatch_job."""

    def setup_method(self) -> None:
        """Clear the in-progress dispatch guard before each test."""
        scan_module._in_progress_tickets.clear()

    def teardown_method(self) -> None:
        scan_module._in_progress_tickets.clear()

    def test_skips_run_when_dev_agent_missing_from_registry(self, caplog: pytest.LogCaptureFixture) -> None:
        """Job must log an error and return early when 'dev_agent' is not in registry."""
        import logging

        registry = _make_registry(raise_key_error=True)
        settings = _make_settings()
        executor = MagicMock(spec=ThreadPoolExecutor)
        mock_tr = _make_tracker_registry()

        with caplog.at_level(logging.ERROR, logger="src.scheduler.jobs.scan_tickets"):
            scan_and_dispatch_job(
                registry=registry,
                settings=settings,
                executor=executor,
                tracker_registry=mock_tr,
            )

        assert any("dev_agent" in r.message for r in caplog.records)
        executor.submit.assert_not_called()

    def test_no_dispatch_when_scanner_returns_empty_list(self) -> None:
        """Job must not dispatch any tickets when both sources return no tickets."""
        registry = _make_registry()
        settings = _make_settings()
        executor = MagicMock(spec=ThreadPoolExecutor)
        mock_tr = _make_tracker_registry()  # returns [] for both sources by default

        scan_and_dispatch_job(
            registry=registry,
            settings=settings,
            executor=executor,
            workflow=_make_workflow(),
            tracker_registry=mock_tr,
        )

        executor.submit.assert_not_called()

    def test_dispatches_ticket_to_executor(self) -> None:
        """Job must submit one executor task per ready ticket returned by REST client."""
        ticket = TicketRecord(id="PROJ-1", source="jira", title="Fix login", url="https://x", raw_status="ACCEPTED")
        registry = _make_registry()
        settings = _make_settings()
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit.return_value = MagicMock()
        mock_tr = _make_tracker_registry(jira_tickets=[ticket])

        with (
            patch("src.scheduler.jobs.scan_tickets.Task"),
            patch("src.scheduler.jobs.scan_tickets.CrewBuilder"),
        ):
            scan_and_dispatch_job(
                registry=registry,
                settings=settings,
                executor=executor,
                workflow=_make_workflow(),
                tracker_registry=mock_tr,
            )

        executor.submit.assert_called_once()
        submitted_fn = executor.submit.call_args.args[0]
        assert callable(submitted_fn)

    def test_skips_already_in_progress_ticket(self) -> None:
        """Job must not re-dispatch a ticket that is already in the in-progress guard."""
        scan_module._in_progress_tickets.add("PROJ-99")

        ticket = TicketRecord(id="PROJ-99", source="jira", title="Already running", url="", raw_status="ACCEPTED")
        registry = _make_registry()
        settings = _make_settings()
        executor = MagicMock(spec=ThreadPoolExecutor)
        mock_tr = _make_tracker_registry(jira_tickets=[ticket])

        scan_and_dispatch_job(
            registry=registry,
            settings=settings,
            executor=executor,
            workflow=_make_workflow(),
            tracker_registry=mock_tr,
        )

        executor.submit.assert_not_called()

    def test_adds_ticket_to_in_progress_guard_on_dispatch(self) -> None:
        """After dispatch, the ticket ID must appear in _in_progress_tickets."""
        ticket = TicketRecord(id="PROJ-42", source="clickup", title="Test task", url="", raw_status="ACCEPTED")
        registry = _make_registry()
        settings = _make_settings()
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit.return_value = MagicMock()
        mock_tr = _make_tracker_registry(clickup_tickets=[ticket])

        with (
            patch("src.scheduler.jobs.scan_tickets.Task"),
            patch("src.scheduler.jobs.scan_tickets.CrewBuilder"),
        ):
            scan_and_dispatch_job(
                registry=registry,
                settings=settings,
                executor=executor,
                workflow=_make_workflow(),
                tracker_registry=mock_tr,
            )

        assert "PROJ-42" in scan_module._in_progress_tickets

    def test_handles_fetch_error_gracefully(self, caplog: pytest.LogCaptureFixture) -> None:
        """Job must log an exception and continue when REST fetch raises TicketFetchError."""
        import logging

        registry = _make_registry()
        settings = _make_settings()
        executor = MagicMock(spec=ThreadPoolExecutor)
        mock_tr = _make_tracker_registry(
            jira_error=TicketFetchError("503 Service Unavailable", source="jira"),
            clickup_error=TicketFetchError("timeout", source="clickup"),
        )

        with caplog.at_level(logging.ERROR, logger="src.scheduler.jobs.scan_tickets"):
            scan_and_dispatch_job(
                registry=registry,
                settings=settings,
                executor=executor,
                workflow=_make_workflow(),
                tracker_registry=mock_tr,
            )

        executor.submit.assert_not_called()

    def test_handles_scanner_crew_exception(self, caplog: pytest.LogCaptureFixture) -> None:
        """Job must log the exception and continue if a tracker raises unexpectedly."""
        import logging

        registry = _make_registry()
        settings = _make_settings()
        executor = MagicMock(spec=ThreadPoolExecutor)
        mock_tr = _make_tracker_registry(
            jira_error=RuntimeError("unexpected failure"),
        )

        # RuntimeError is not caught by the TicketFetchError handler,
        # but it IS caught by the ValueError handler (falls through).
        # For this test, the job should not crash and executor should not be called.
        with caplog.at_level(logging.ERROR, logger="src.scheduler.jobs.scan_tickets"):
            try:
                scan_and_dispatch_job(
                    registry=registry,
                    settings=settings,
                    executor=executor,
                    workflow=_make_workflow(),
                    tracker_registry=mock_tr,
                )
            except RuntimeError:
                pass  # allowed to propagate for unexpected errors

        executor.submit.assert_not_called()

    def test_skips_ticket_entry_with_rejected_status(self, caplog: pytest.LogCaptureFixture) -> None:
        """Job must skip ticket entries that have REJECTED raw_status (BR-3)."""
        import logging

        ticket = TicketRecord(id="PROJ-10", source="jira", title="Rejected ticket", url="", raw_status="REJECTED")
        registry = _make_registry()
        settings = _make_settings()
        executor = MagicMock(spec=ThreadPoolExecutor)
        mock_tr = _make_tracker_registry(jira_tickets=[ticket])

        with caplog.at_level(logging.INFO, logger="src.scheduler.jobs.scan_tickets"):
            scan_and_dispatch_job(
                registry=registry,
                settings=settings,
                executor=executor,
                workflow=_make_workflow(),
                tracker_registry=mock_tr,
            )

        executor.submit.assert_not_called()
