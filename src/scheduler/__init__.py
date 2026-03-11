"""
Scheduler package for the simple-ai-worker application.

This package provides the APScheduler-based scheduling infrastructure used
to trigger AI agent jobs on a configurable interval.

Primary entry-point:
    :class:`~src.scheduler.runner.SchedulerRunner` — import directly from
    the submodule::

        from src.scheduler.runner import SchedulerRunner
"""

from __future__ import annotations

__all__: list[str] = []
