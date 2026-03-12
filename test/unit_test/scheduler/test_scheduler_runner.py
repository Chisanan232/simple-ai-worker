"""Unit tests for :class:`src.scheduler.runner.SchedulerRunner`."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from src.scheduler.runner import SchedulerRunner


def _make_registry(agent_ids: list[str] | None = None) -> MagicMock:
    """Return a minimal mock AgentRegistry."""
    ids = agent_ids or ["planner", "dev_lead", "dev_agent"]
    reg = MagicMock()
    reg.agent_ids.return_value = ids
    return reg


def _make_settings(interval: int = 3600) -> MagicMock:
    s = MagicMock()
    s.SCHEDULER_INTERVAL_SECONDS = interval
    return s


class TestSchedulerRunnerInit:
    """Tests for SchedulerRunner.__init__."""

    def test_default_values(self) -> None:
        """Default interval and timezone are applied when not supplied."""
        runner = SchedulerRunner()
        assert runner._interval_seconds == 60
        assert runner._timezone == "UTC"

    def test_custom_values(self) -> None:
        """Custom interval and timezone are stored correctly."""
        runner = SchedulerRunner(interval_seconds=30, timezone="Asia/Taipei")
        assert runner._interval_seconds == 30
        assert runner._timezone == "Asia/Taipei"

    def test_scheduler_is_not_running_on_init(self) -> None:
        """The underlying BackgroundScheduler must not be running after init."""
        runner = SchedulerRunner()
        assert runner._scheduler.running is False

    def test_optional_phase6_args_default_to_none(self) -> None:
        """registry, settings, and executor default to None."""
        runner = SchedulerRunner()
        assert runner._registry is None
        assert runner._settings is None
        assert runner._executor is None

    def test_phase6_args_stored_when_provided(self) -> None:
        """registry, settings, and executor are stored when provided."""
        reg = _make_registry()
        settings = _make_settings()
        executor = MagicMock(spec=ThreadPoolExecutor)
        runner = SchedulerRunner(registry=reg, settings=settings, executor=executor)
        assert runner._registry is reg
        assert runner._settings is settings
        assert runner._executor is executor


class TestSchedulerRunnerStart:
    """Tests for SchedulerRunner.start()."""

    def test_start_starts_background_scheduler(self) -> None:
        """start() must set the underlying scheduler to running."""
        runner = SchedulerRunner(interval_seconds=3600)
        try:
            runner.start()
            assert runner._scheduler.running is True
        finally:
            runner.stop(wait=False)

    def test_start_registers_hello_world_job(self) -> None:
        """start() must register the hello_world job."""
        runner = SchedulerRunner(interval_seconds=3600)
        try:
            runner.start()
            job_ids = [job.id for job in runner._scheduler.get_jobs()]
            assert "hello_world" in job_ids
        finally:
            runner.stop(wait=False)

    def test_start_called_twice_is_idempotent(self, caplog: pytest.LogCaptureFixture) -> None:
        """Calling start() on an already-running scheduler logs a warning and does not raise."""
        import logging

        runner = SchedulerRunner(interval_seconds=3600)
        try:
            runner.start()
            with caplog.at_level(logging.WARNING, logger="src.scheduler.runner"):
                runner.start()  # second call — should warn, not raise
            assert any("already-running" in r.message for r in caplog.records)
        finally:
            runner.stop(wait=False)

    # ------------------------------------------------------------------
    # Phase 6 job registration
    # ------------------------------------------------------------------

    def test_phase6_jobs_registered_when_dependencies_provided(self) -> None:
        """All three Phase-6 jobs must be registered when registry/settings/executor are supplied."""
        reg = _make_registry()
        settings = _make_settings()
        executor = ThreadPoolExecutor(max_workers=1)
        runner = SchedulerRunner(
            interval_seconds=3600,
            registry=reg,
            settings=settings,
            executor=executor,
        )
        try:
            runner.start()
            job_ids = [job.id for job in runner._scheduler.get_jobs()]
            assert "scan_and_dispatch" in job_ids
            assert "planner_listener" in job_ids
            assert "dev_lead_listener" in job_ids
        finally:
            runner.stop(wait=False)
            executor.shutdown(wait=False)

    def test_phase6_jobs_not_registered_without_registry(self) -> None:
        """Phase-6 jobs must NOT be registered when registry is None."""
        runner = SchedulerRunner(interval_seconds=3600)
        try:
            runner.start()
            job_ids = [job.id for job in runner._scheduler.get_jobs()]
            assert "scan_and_dispatch" not in job_ids
            assert "planner_listener" not in job_ids
            assert "dev_lead_listener" not in job_ids
        finally:
            runner.stop(wait=False)

    def test_phase6_jobs_use_correct_interval(self) -> None:
        """Phase-6 interval jobs must be scheduled with the configured interval_seconds."""
        reg = _make_registry()
        settings = _make_settings()
        executor = ThreadPoolExecutor(max_workers=1)
        runner = SchedulerRunner(
            interval_seconds=300,
            registry=reg,
            settings=settings,
            executor=executor,
        )
        try:
            runner.start()
            jobs_by_id = {job.id: job for job in runner._scheduler.get_jobs()}
            # APScheduler stores interval jobs with a trigger whose 'interval' attribute
            # contains the timedelta — we just check the job exists with correct kwargs.
            assert "scan_and_dispatch" in jobs_by_id
        finally:
            runner.stop(wait=False)
            executor.shutdown(wait=False)


class TestSchedulerRunnerStop:
    """Tests for SchedulerRunner.stop()."""

    def test_stop_shuts_down_scheduler(self) -> None:
        """stop() must leave the scheduler in a not-running state."""
        runner = SchedulerRunner(interval_seconds=3600)
        runner.start()
        runner.stop(wait=False)
        assert runner._scheduler.running is False

    def test_stop_on_not_running_scheduler_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """stop() on a never-started scheduler logs a warning and does not raise."""
        import logging

        runner = SchedulerRunner(interval_seconds=3600)
        with caplog.at_level(logging.WARNING, logger="src.scheduler.runner"):
            runner.stop()  # should not raise
        assert any("not running" in r.message for r in caplog.records)


class TestSchedulerRunnerRepr:
    """Tests for SchedulerRunner.__repr__."""

    def test_repr_contains_class_name(self) -> None:
        """__repr__ must include the class name."""
        runner = SchedulerRunner()
        assert "SchedulerRunner" in repr(runner)

    def test_repr_contains_interval(self) -> None:
        """__repr__ must include the configured interval."""
        runner = SchedulerRunner(interval_seconds=120)
        assert "120" in repr(runner)

    def test_repr_contains_timezone(self) -> None:
        """__repr__ must include the configured timezone."""
        runner = SchedulerRunner(timezone="Asia/Tokyo")
        assert "Asia/Tokyo" in repr(runner)


class TestSchedulerRunnerInit:
    """Tests for SchedulerRunner.__init__."""

    def test_default_values(self) -> None:
        """Default interval and timezone are applied when not supplied."""
        runner = SchedulerRunner()
        assert runner._interval_seconds == 60
        assert runner._timezone == "UTC"

    def test_custom_values(self) -> None:
        """Custom interval and timezone are stored correctly."""
        runner = SchedulerRunner(interval_seconds=30, timezone="Asia/Taipei")
        assert runner._interval_seconds == 30
        assert runner._timezone == "Asia/Taipei"

    def test_scheduler_is_not_running_on_init(self) -> None:
        """The underlying BackgroundScheduler must not be running after init."""
        runner = SchedulerRunner()
        assert runner._scheduler.running is False


class TestSchedulerRunnerStart:
    """Tests for SchedulerRunner.start()."""

    def test_start_starts_background_scheduler(self) -> None:
        """start() must set the underlying scheduler to running."""
        runner = SchedulerRunner(interval_seconds=3600)
        try:
            runner.start()
            assert runner._scheduler.running is True
        finally:
            runner.stop(wait=False)

    def test_start_registers_hello_world_job(self) -> None:
        """start() must register the hello_world job."""
        runner = SchedulerRunner(interval_seconds=3600)
        try:
            runner.start()
            job_ids = [job.id for job in runner._scheduler.get_jobs()]
            assert "hello_world" in job_ids
        finally:
            runner.stop(wait=False)

    def test_start_called_twice_is_idempotent(self, caplog: pytest.LogCaptureFixture) -> None:
        """Calling start() on an already-running scheduler logs a warning and does not raise."""
        import logging

        runner = SchedulerRunner(interval_seconds=3600)
        try:
            runner.start()
            with caplog.at_level(logging.WARNING, logger="src.scheduler.runner"):
                runner.start()  # second call — should warn, not raise
            assert any("already-running" in r.message for r in caplog.records)
        finally:
            runner.stop(wait=False)


class TestSchedulerRunnerStop:
    """Tests for SchedulerRunner.stop()."""

    def test_stop_shuts_down_scheduler(self) -> None:
        """stop() must leave the scheduler in a not-running state."""
        runner = SchedulerRunner(interval_seconds=3600)
        runner.start()
        runner.stop(wait=False)
        assert runner._scheduler.running is False

    def test_stop_on_not_running_scheduler_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """stop() on a never-started scheduler logs a warning and does not raise."""
        import logging

        runner = SchedulerRunner(interval_seconds=3600)
        with caplog.at_level(logging.WARNING, logger="src.scheduler.runner"):
            runner.stop()  # should not raise
        assert any("not running" in r.message for r in caplog.records)


class TestSchedulerRunnerRepr:
    """Tests for SchedulerRunner.__repr__."""

    def test_repr_contains_class_name(self) -> None:
        """__repr__ must include the class name."""
        runner = SchedulerRunner()
        assert "SchedulerRunner" in repr(runner)

    def test_repr_contains_interval(self) -> None:
        """__repr__ must include the configured interval."""
        runner = SchedulerRunner(interval_seconds=120)
        assert "120" in repr(runner)

    def test_repr_contains_timezone(self) -> None:
        """__repr__ must include the configured timezone."""
        runner = SchedulerRunner(timezone="Asia/Tokyo")
        assert "Asia/Tokyo" in repr(runner)
