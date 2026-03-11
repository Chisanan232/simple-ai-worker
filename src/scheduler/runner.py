"""
Scheduler runner module.

Provides :class:`SchedulerRunner`, a thin wrapper around
``apscheduler.schedulers.background.BackgroundScheduler`` that:

- Registers all application jobs in one place.
- Exposes a clean ``start()`` / ``stop()`` lifecycle API.
- Is intentionally **not** a singleton so it can be instantiated and
  torn down freely in tests.

Typical usage::

    from src.scheduler.runner import SchedulerRunner

    runner = SchedulerRunner(interval_seconds=60, timezone="UTC")
    runner.start()
    # … main thread blocks on signal …
    runner.stop()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from apscheduler.schedulers.background import BackgroundScheduler

from .jobs.hello_world import hello_world_job  # noqa: E402

if TYPE_CHECKING:
    from apscheduler.job import Job

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_SECONDS: int = 60
_DEFAULT_TIMEZONE: str = "UTC"


class SchedulerRunner:
    """Manages the lifecycle of the APScheduler ``BackgroundScheduler``.

    The scheduler runs in a daemon background thread so the main thread
    remains free for signal handling.  All jobs are registered via
    :meth:`_register_jobs` which is called automatically inside
    :meth:`start`.

    Args:
        interval_seconds: Default polling interval in seconds applied to
            all registered interval-based jobs.  Defaults to ``60``.
        timezone: Timezone string recognised by APScheduler / pytz (e.g.
            ``"UTC"``, ``"Asia/Taipei"``).  Defaults to ``"UTC"``.
    """

    def __init__(
        self,
        interval_seconds: int = _DEFAULT_INTERVAL_SECONDS,
        timezone: str = _DEFAULT_TIMEZONE,
    ) -> None:
        self._interval_seconds = interval_seconds
        self._timezone = timezone
        self._scheduler: BackgroundScheduler = BackgroundScheduler(
            timezone=self._timezone,
        )

    # ------------------------------------------------------------------
    # Public lifecycle API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Register all jobs and start the background scheduler.

        Safe to call only once per instance.  Calling a second time on an
        already-running scheduler will be logged and ignored.
        """
        if self._scheduler.running:
            logger.warning("SchedulerRunner.start() called on an already-running scheduler — ignored.")
            return

        self._register_jobs()
        self._scheduler.start()
        logger.info(
            "Scheduler started (interval=%ds, timezone=%s).",
            self._interval_seconds,
            self._timezone,
        )

    def stop(self, wait: bool = True) -> None:
        """Shut down the background scheduler gracefully.

        Args:
            wait: When ``True`` (default) the call blocks until all
                currently-executing jobs have finished before returning.
        """
        if not self._scheduler.running:
            logger.warning("SchedulerRunner.stop() called on a scheduler that is not running — ignored.")
            return

        self._scheduler.shutdown(wait=wait)
        logger.info("Scheduler stopped (wait=%s).", wait)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register_jobs(self) -> None:
        """Register all application jobs with the scheduler.

        Each job is added with ``replace_existing=True`` so that
        re-registration (e.g. during tests) never raises a
        ``ConflictingIdError``.

        Phase-1 jobs:
            - :func:`~src.scheduler.jobs.hello_world.hello_world_job`
              — no-op placeholder, fires every ``interval_seconds``.

        Later phases will register additional jobs here.
        """
        hello_world: Job = self._scheduler.add_job(
            func=hello_world_job,
            trigger="interval",
            seconds=self._interval_seconds,
            id="hello_world",
            name="Hello World (placeholder)",
            replace_existing=True,
        )
        logger.debug("Registered job: %s (id=%s).", hello_world.name, hello_world.id)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"interval_seconds={self._interval_seconds!r}, "
            f"timezone={self._timezone!r}, "
            f"running={self._scheduler.running!r})"
        )


