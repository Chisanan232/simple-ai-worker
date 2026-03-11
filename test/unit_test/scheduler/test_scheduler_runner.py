"""Unit tests for :class:`src.scheduler.runner.SchedulerRunner`."""

from __future__ import annotations

import pytest

from src.scheduler.runner import SchedulerRunner


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

    def test_stop_on_not_running_scheduler_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
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


