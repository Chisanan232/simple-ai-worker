"""
Scheduler runner module.

Provides :class:`SchedulerRunner`, a thin wrapper around
``apscheduler.schedulers.background.BackgroundScheduler`` that:

- Registers all application jobs in one place.
- Exposes a clean ``start()`` / ``stop()`` lifecycle API.
- Is intentionally **not** a singleton so it can be instantiated and
  torn down freely in tests.

Typical usage (Phase 6)::

    from concurrent.futures import ThreadPoolExecutor
    from src.scheduler.runner import SchedulerRunner
    from src.agents.registry import build_registry
    from src.config import get_settings, load_agent_config

    settings = get_settings()
    team_config = load_agent_config(settings.AGENT_CONFIG_PATH)
    registry = build_registry(team_config, settings)
    executor = ThreadPoolExecutor(max_workers=settings.MAX_CONCURRENT_DEV_AGENTS)

    runner = SchedulerRunner(
        interval_seconds=settings.SCHEDULER_INTERVAL_SECONDS,
        timezone=settings.SCHEDULER_TIMEZONE,
        registry=registry,
        settings=settings,
        executor=executor,
    )
    runner.start()
    # … main thread blocks on signal …
    runner.stop()
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional

from apscheduler.schedulers.background import BackgroundScheduler

from .jobs.dev_lead_listener import dev_lead_listener_job
from .jobs.hello_world import hello_world_job
from .jobs.planner_listener import planner_listener_job
from .jobs.scan_tickets import scan_and_dispatch_job

if TYPE_CHECKING:
    from apscheduler.job import Job

    from src.agents.registry import AgentRegistry
    from src.config.settings import AppSettings

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
        registry: Optional :class:`~src.agents.registry.AgentRegistry`.
            When provided, Phase 6 agent jobs are registered.  When
            ``None`` (default) only the Phase 1 hello-world job is
            registered — useful in unit tests that do not need agents.
        settings: Optional :class:`~src.config.settings.AppSettings`.
            Required when *registry* is provided.
        executor: Optional :class:`~concurrent.futures.ThreadPoolExecutor`
            shared with the Slack Bolt handlers.  Required when *registry*
            is provided.
    """

    def __init__(
        self,
        interval_seconds: int = _DEFAULT_INTERVAL_SECONDS,
        timezone: str = _DEFAULT_TIMEZONE,
        registry: Optional["AgentRegistry"] = None,
        settings: Optional["AppSettings"] = None,
        executor: Optional[ThreadPoolExecutor] = None,
    ) -> None:
        self._interval_seconds = interval_seconds
        self._timezone = timezone
        self._registry = registry
        self._settings = settings
        self._executor = executor
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

        Phase-6 jobs (registered only when *registry*, *settings*, and
        *executor* were supplied to the constructor):
            - :func:`~src.scheduler.jobs.scan_tickets.scan_and_dispatch_job`
              — pull-model Dev Agent dispatcher, fires every
              ``interval_seconds``.
            - :func:`~src.scheduler.jobs.planner_listener.planner_listener_job`
              — fallback Planner Slack polling, fires every
              ``interval_seconds``.
            - :func:`~src.scheduler.jobs.dev_lead_listener.dev_lead_listener_job`
              — fallback Dev Lead Slack polling, fires every
              ``interval_seconds``.
        """
        hello_world: "Job" = self._scheduler.add_job(
            func=hello_world_job,
            trigger="interval",
            seconds=self._interval_seconds,
            id="hello_world",
            name="Hello World (placeholder)",
            replace_existing=True,
        )
        logger.debug("Registered job: %s (id=%s).", hello_world.name, hello_world.id)

        # Phase 6 jobs — only when agent infrastructure is available.
        if self._registry is not None and self._settings is not None and self._executor is not None:
            scan_job: "Job" = self._scheduler.add_job(
                func=scan_and_dispatch_job,
                kwargs={
                    "registry": self._registry,
                    "settings": self._settings,
                    "executor": self._executor,
                },
                trigger="interval",
                seconds=self._interval_seconds,
                id="scan_and_dispatch",
                name="Scan & Dispatch Dev Tickets (pull model)",
                replace_existing=True,
            )
            logger.debug("Registered job: %s (id=%s).", scan_job.name, scan_job.id)

            planner_job: "Job" = self._scheduler.add_job(
                func=planner_listener_job,
                kwargs={
                    "registry": self._registry,
                    "settings": self._settings,
                },
                trigger="interval",
                seconds=self._interval_seconds,
                id="planner_listener",
                name="Planner Slack Listener (fallback polling)",
                replace_existing=True,
            )
            logger.debug("Registered job: %s (id=%s).", planner_job.name, planner_job.id)

            dev_lead_job: "Job" = self._scheduler.add_job(
                func=dev_lead_listener_job,
                kwargs={
                    "registry": self._registry,
                    "settings": self._settings,
                },
                trigger="interval",
                seconds=self._interval_seconds,
                id="dev_lead_listener",
                name="Dev Lead Slack Listener (fallback polling)",
                replace_existing=True,
            )
            logger.debug("Registered job: %s (id=%s).", dev_lead_job.name, dev_lead_job.id)

            logger.info(
                "Phase-6 scheduler jobs registered: scan_and_dispatch, "
                "planner_listener, dev_lead_listener."
            )
        else:
            logger.info(
                "Phase-6 scheduler jobs NOT registered "
                "(registry/settings/executor not provided — running in minimal mode)."
            )

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
