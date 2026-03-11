"""Unit tests for :func:`src.scheduler.jobs.hello_world.hello_world_job`."""

from __future__ import annotations

import logging

import pytest

from src.scheduler.jobs.hello_world import hello_world_job


class TestHelloWorldJob:
    """Tests for the Phase-1 placeholder scheduler job."""

    def test_job_runs_without_exception(self) -> None:
        """hello_world_job() must complete without raising."""
        hello_world_job()  # should not raise

    def test_job_logs_info_message(self, caplog: pytest.LogCaptureFixture) -> None:
        """hello_world_job() must emit exactly one INFO-level log record."""
        with caplog.at_level(logging.INFO, logger="src.scheduler.jobs.hello_world"):
            hello_world_job()

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.levelno == logging.INFO
        assert "Hello World" in record.message

    def test_job_log_contains_iso_timestamp(self, caplog: pytest.LogCaptureFixture) -> None:
        """The log message must contain a UTC ISO-8601 timestamp."""
        with caplog.at_level(logging.INFO, logger="src.scheduler.jobs.hello_world"):
            hello_world_job()

        message: str = caplog.records[0].message
        # UTC ISO timestamp ends with "+00:00"
        assert "+00:00" in message


