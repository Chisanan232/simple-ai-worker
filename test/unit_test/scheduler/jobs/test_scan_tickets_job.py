"""
Unit tests for :func:`src.scheduler.jobs.scan_tickets.scan_and_dispatch_job`.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

import src.scheduler.jobs.scan_tickets as scan_module
from src.scheduler.jobs.scan_tickets import scan_and_dispatch_job


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


class TestScanAndDispatchJob:
    """Tests for scan_and_dispatch_job."""

    def setup_method(self) -> None:
        """Clear the in-progress dispatch guard before each test."""
        scan_module._in_progress_tickets.clear()

    def teardown_method(self) -> None:
        scan_module._in_progress_tickets.clear()

    def test_skips_run_when_dev_agent_missing_from_registry(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Job must log an error and return early when 'dev_agent' is not in registry."""
        import logging

        registry = _make_registry(raise_key_error=True)
        settings = _make_settings()
        executor = MagicMock(spec=ThreadPoolExecutor)

        with caplog.at_level(logging.ERROR, logger="src.scheduler.jobs.scan_tickets"):
            scan_and_dispatch_job(registry=registry, settings=settings, executor=executor)

        assert any("dev_agent" in r.message for r in caplog.records)
        executor.submit.assert_not_called()

    def test_no_dispatch_when_scanner_returns_empty_list(self) -> None:
        """Job must not dispatch any tickets when scanner crew returns []."""
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "[]"

        registry = _make_registry()
        settings = _make_settings()
        executor = MagicMock(spec=ThreadPoolExecutor)

        with (
            patch("src.scheduler.jobs.scan_tickets.Task"),
            patch("src.scheduler.jobs.scan_tickets.CrewBuilder") as MockBuilder,
        ):
            MockBuilder.build.return_value = mock_crew
            scan_and_dispatch_job(registry=registry, settings=settings, executor=executor)

        executor.submit.assert_not_called()

    def test_dispatches_ticket_to_executor(self) -> None:
        """Job must submit one executor task per ready ticket found by scanner."""
        tickets = [{"id": "PROJ-1", "source": "jira", "title": "Fix login", "url": "https://x"}]
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = json.dumps(tickets)

        registry = _make_registry()
        settings = _make_settings()
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit.return_value = MagicMock()

        with (
            patch("src.scheduler.jobs.scan_tickets.Task"),
            patch("src.scheduler.jobs.scan_tickets.CrewBuilder") as MockBuilder,
        ):
            MockBuilder.build.return_value = mock_crew
            scan_and_dispatch_job(registry=registry, settings=settings, executor=executor)

        executor.submit.assert_called_once()
        # First positional arg to submit is the target function
        submitted_fn = executor.submit.call_args.args[0]
        assert callable(submitted_fn)

    def test_skips_already_in_progress_ticket(self) -> None:
        """Job must not re-dispatch a ticket that is already in the in-progress guard."""
        scan_module._in_progress_tickets.add("PROJ-99")

        tickets = [{"id": "PROJ-99", "source": "jira", "title": "Already running", "url": ""}]
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = json.dumps(tickets)

        registry = _make_registry()
        settings = _make_settings()
        executor = MagicMock(spec=ThreadPoolExecutor)

        with (
            patch("src.scheduler.jobs.scan_tickets.Task"),
            patch("src.scheduler.jobs.scan_tickets.CrewBuilder") as MockBuilder,
        ):
            MockBuilder.build.return_value = mock_crew
            scan_and_dispatch_job(registry=registry, settings=settings, executor=executor)

        executor.submit.assert_not_called()

    def test_adds_ticket_to_in_progress_guard_on_dispatch(self) -> None:
        """After dispatch, the ticket ID must appear in _in_progress_tickets."""
        tickets = [{"id": "PROJ-42", "source": "clickup", "title": "Test task", "url": ""}]
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = json.dumps(tickets)

        registry = _make_registry()
        settings = _make_settings()
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit.return_value = MagicMock()

        with (
            patch("src.scheduler.jobs.scan_tickets.Task"),
            patch("src.scheduler.jobs.scan_tickets.CrewBuilder") as MockBuilder,
        ):
            MockBuilder.build.return_value = mock_crew
            scan_and_dispatch_job(registry=registry, settings=settings, executor=executor)

        assert "PROJ-42" in scan_module._in_progress_tickets

    def test_handles_non_json_scanner_output_gracefully(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Job must log a warning and return early if scanner output is not valid JSON."""
        import logging

        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "Sorry, I could not find any tickets."

        registry = _make_registry()
        settings = _make_settings()
        executor = MagicMock(spec=ThreadPoolExecutor)

        with (
            caplog.at_level(logging.WARNING, logger="src.scheduler.jobs.scan_tickets"),
            patch("src.scheduler.jobs.scan_tickets.Task"),
            patch("src.scheduler.jobs.scan_tickets.CrewBuilder") as MockBuilder,
        ):
            MockBuilder.build.return_value = mock_crew
            scan_and_dispatch_job(registry=registry, settings=settings, executor=executor)

        executor.submit.assert_not_called()
        assert any("JSON" in r.message for r in caplog.records)

    def test_handles_scanner_crew_exception(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Job must log the exception and return early if scanner crew.kickoff() raises."""
        import logging

        mock_crew = MagicMock()
        mock_crew.kickoff.side_effect = RuntimeError("LLM unavailable")

        registry = _make_registry()
        settings = _make_settings()
        executor = MagicMock(spec=ThreadPoolExecutor)

        with (
            caplog.at_level(logging.ERROR, logger="src.scheduler.jobs.scan_tickets"),
            patch("src.scheduler.jobs.scan_tickets.Task"),
            patch("src.scheduler.jobs.scan_tickets.CrewBuilder") as MockBuilder,
        ):
            MockBuilder.build.return_value = mock_crew
            scan_and_dispatch_job(registry=registry, settings=settings, executor=executor)

        executor.submit.assert_not_called()

    def test_strips_markdown_code_fence_from_scanner_output(self) -> None:
        """Job must strip ```json … ``` fences before JSON parsing."""
        tickets = [{"id": "PROJ-7", "source": "jira", "title": "Fenced", "url": ""}]
        raw_output = f"```json\n{json.dumps(tickets)}\n```"

        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = raw_output

        registry = _make_registry()
        settings = _make_settings()
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit.return_value = MagicMock()

        with (
            patch("src.scheduler.jobs.scan_tickets.Task"),
            patch("src.scheduler.jobs.scan_tickets.CrewBuilder") as MockBuilder,
        ):
            MockBuilder.build.return_value = mock_crew
            scan_and_dispatch_job(registry=registry, settings=settings, executor=executor)

        executor.submit.assert_called_once()

    def test_skips_ticket_entry_with_missing_id(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Job must warn and skip ticket entries that have no 'id' field."""
        import logging

        tickets = [{"source": "jira", "title": "No ID ticket", "url": ""}]
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = json.dumps(tickets)

        registry = _make_registry()
        settings = _make_settings()
        executor = MagicMock(spec=ThreadPoolExecutor)

        with (
            caplog.at_level(logging.WARNING, logger="src.scheduler.jobs.scan_tickets"),
            patch("src.scheduler.jobs.scan_tickets.Task"),
            patch("src.scheduler.jobs.scan_tickets.CrewBuilder") as MockBuilder,
        ):
            MockBuilder.build.return_value = mock_crew
            scan_and_dispatch_job(registry=registry, settings=settings, executor=executor)

        executor.submit.assert_not_called()


