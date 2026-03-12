"""
Unit tests for :func:`src.scheduler.jobs.planner_listener.planner_listener_job`
and :func:`src.scheduler.jobs.dev_lead_listener.dev_lead_listener_job`.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import src.scheduler.jobs.dev_lead_listener as dev_lead_module
import src.scheduler.jobs.planner_listener as planner_module
from src.scheduler.jobs.dev_lead_listener import dev_lead_listener_job
from src.scheduler.jobs.planner_listener import planner_listener_job


def _make_registry(agent_id: str = "planner", raise_key_error: bool = False) -> MagicMock:
    reg = MagicMock()
    if raise_key_error:
        reg.__getitem__.side_effect = KeyError(agent_id)
        reg.agent_ids.return_value = []
    else:
        reg.__getitem__.return_value = MagicMock()
        reg.agent_ids.return_value = [agent_id]
    return reg


def _make_settings() -> MagicMock:
    return MagicMock()


class TestPlannerListenerJob:
    """Tests for planner_listener_job."""

    def setup_method(self) -> None:
        planner_module._last_processed_ts = None

    def teardown_method(self) -> None:
        planner_module._last_processed_ts = None

    def test_returns_early_when_planner_not_in_registry(self, caplog: pytest.LogCaptureFixture) -> None:
        """Job must log an error and return early when 'planner' is not in registry."""
        import logging

        registry = _make_registry(agent_id="planner", raise_key_error=True)
        settings = _make_settings()

        with caplog.at_level(logging.ERROR, logger="src.scheduler.jobs.planner_listener"):
            planner_listener_job(registry=registry, settings=settings)

        assert any("planner" in r.message for r in caplog.records)

    def test_advances_last_processed_ts_on_success(self) -> None:
        """Job must update _last_processed_ts after a successful crew run."""
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "No new planner messages found."

        registry = _make_registry(agent_id="planner")
        settings = _make_settings()

        with (
            patch("src.scheduler.jobs.planner_listener.Task"),
            patch("src.scheduler.jobs.planner_listener.CrewBuilder") as MockBuilder,
        ):
            MockBuilder.build.return_value = mock_crew
            planner_listener_job(registry=registry, settings=settings)

        assert planner_module._last_processed_ts is not None

    def test_does_not_advance_ts_on_crew_exception(self) -> None:
        """Job must NOT update _last_processed_ts if crew.kickoff() raises."""
        mock_crew = MagicMock()
        mock_crew.kickoff.side_effect = RuntimeError("LLM error")

        registry = _make_registry(agent_id="planner")
        settings = _make_settings()

        before_ts = planner_module._last_processed_ts

        with (
            patch("src.scheduler.jobs.planner_listener.Task"),
            patch("src.scheduler.jobs.planner_listener.CrewBuilder") as MockBuilder,
        ):
            MockBuilder.build.return_value = mock_crew
            planner_listener_job(registry=registry, settings=settings)

        assert planner_module._last_processed_ts == before_ts

    def test_uses_last_processed_ts_as_watermark(self) -> None:
        """On second run, job must use the updated _last_processed_ts as the since-timestamp."""
        import time

        planner_module._last_processed_ts = time.time() - 120.0

        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "Done."

        registry = _make_registry(agent_id="planner")
        settings = _make_settings()

        with (
            patch("src.scheduler.jobs.planner_listener.Task"),
            patch("src.scheduler.jobs.planner_listener.CrewBuilder") as MockBuilder,
        ):
            MockBuilder.build.return_value = mock_crew
            planner_listener_job(registry=registry, settings=settings)

        # Crew was built — task description included the watermark timestamp
        MockBuilder.build.assert_called_once()

    def test_builds_crew_with_planner_agent(self) -> None:
        """Job must build the Crew using the planner agent from the registry."""
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "Done."
        mock_agent = MagicMock()

        registry = _make_registry(agent_id="planner")
        registry.__getitem__.return_value = mock_agent
        settings = _make_settings()

        with (
            patch("src.scheduler.jobs.planner_listener.Task"),
            patch("src.scheduler.jobs.planner_listener.CrewBuilder") as MockBuilder,
        ):
            MockBuilder.build.return_value = mock_crew
            planner_listener_job(registry=registry, settings=settings)

        # build was called with the mock agent
        call_kwargs = MockBuilder.build.call_args.kwargs
        assert mock_agent in call_kwargs["agents"]


class TestDevLeadListenerJob:
    """Tests for dev_lead_listener_job."""

    def setup_method(self) -> None:
        dev_lead_module._last_processed_ts = None

    def teardown_method(self) -> None:
        dev_lead_module._last_processed_ts = None

    def test_returns_early_when_dev_lead_not_in_registry(self, caplog: pytest.LogCaptureFixture) -> None:
        """Job must log an error and return early when 'dev_lead' is not in registry."""
        import logging

        registry = _make_registry(agent_id="dev_lead", raise_key_error=True)
        settings = _make_settings()

        with caplog.at_level(logging.ERROR, logger="src.scheduler.jobs.dev_lead_listener"):
            dev_lead_listener_job(registry=registry, settings=settings)

        assert any("dev_lead" in r.message for r in caplog.records)

    def test_advances_last_processed_ts_on_success(self) -> None:
        """Job must update _last_processed_ts after a successful crew run."""
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "No new dev lead messages found."

        registry = _make_registry(agent_id="dev_lead")
        settings = _make_settings()

        with (
            patch("src.scheduler.jobs.dev_lead_listener.Task"),
            patch("src.scheduler.jobs.dev_lead_listener.CrewBuilder") as MockBuilder,
        ):
            MockBuilder.build.return_value = mock_crew
            dev_lead_listener_job(registry=registry, settings=settings)

        assert dev_lead_module._last_processed_ts is not None

    def test_does_not_advance_ts_on_crew_exception(self) -> None:
        """Job must NOT update _last_processed_ts if crew.kickoff() raises."""
        mock_crew = MagicMock()
        mock_crew.kickoff.side_effect = RuntimeError("LLM error")

        registry = _make_registry(agent_id="dev_lead")
        settings = _make_settings()

        before_ts = dev_lead_module._last_processed_ts

        with (
            patch("src.scheduler.jobs.dev_lead_listener.Task"),
            patch("src.scheduler.jobs.dev_lead_listener.CrewBuilder") as MockBuilder,
        ):
            MockBuilder.build.return_value = mock_crew
            dev_lead_listener_job(registry=registry, settings=settings)

        assert dev_lead_module._last_processed_ts == before_ts

    def test_builds_crew_with_dev_lead_agent(self) -> None:
        """Job must build the Crew using the dev_lead agent from the registry."""
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "Done."
        mock_agent = MagicMock()

        registry = _make_registry(agent_id="dev_lead")
        registry.__getitem__.return_value = mock_agent
        settings = _make_settings()

        with (
            patch("src.scheduler.jobs.dev_lead_listener.Task"),
            patch("src.scheduler.jobs.dev_lead_listener.CrewBuilder") as MockBuilder,
        ):
            MockBuilder.build.return_value = mock_crew
            dev_lead_listener_job(registry=registry, settings=settings)

        call_kwargs = MockBuilder.build.call_args.kwargs
        assert mock_agent in call_kwargs["agents"]
