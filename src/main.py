"""
Application entry-point for simple-ai-worker (scheduler process).

Startup sequence:
    1. Configure root logging.
    2. Load :class:`~src.config.settings.AppSettings` from ``.env`` via
       :func:`~src.config.get_settings`.
    3. Register ``SIGINT`` / ``SIGTERM`` handlers for graceful shutdown.
    4. Load :class:`~src.config.agent_config.AgentTeamConfig` from YAML via
       :func:`~src.config.load_agent_config`.
    5. Build the :class:`~src.agents.registry.AgentRegistry` via
       :func:`~src.agents.registry.build_registry`.
    6. Create the bounded :class:`~concurrent.futures.ThreadPoolExecutor`
       used by the Dev Agent dispatcher.
    7. Instantiate :class:`~src.scheduler.runner.SchedulerRunner` using
       values from :class:`~src.config.settings.AppSettings`.
    8. Start the scheduler.
    9. Block the main thread until a signal is received.
    10. Graceful teardown: stop scheduler → shut down executor.

This process runs **no Slack code at all**.  The Slack Events API webhook
is served by the separate ``simple-ai-slack`` process
(entry point: ``src/slack_main.py``).
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import FrameType
from typing import Optional

from src.agents.registry import AgentRegistry, build_registry
from src.config import get_settings, load_agent_config
from src.scheduler.runner import SchedulerRunner

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

_LOG_FORMAT: str = "%(asctime)s [%(levelname)8s] %(name)s — %(message)s"
_LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=_LOG_FORMAT,
    datefmt=_LOG_DATE_FORMAT,
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful shutdown helpers
# ---------------------------------------------------------------------------

# Shared flag: set to True by signal handlers to break the main loop.
_shutdown_requested: bool = False
_runner: Optional[SchedulerRunner] = None
_registry: Optional[AgentRegistry] = None
_executor: Optional[ThreadPoolExecutor] = None


def _handle_signal(signum: int, frame: Optional["FrameType"]) -> None:  # noqa: ARG001
    """Handle ``SIGINT`` / ``SIGTERM`` by requesting a graceful shutdown.

    Args:
        signum: The signal number received.
        frame: The current stack frame (unused).
    """
    global _shutdown_requested  # noqa: PLW0603

    sig_name: str = signal.Signals(signum).name
    logger.info("Received signal %s — initiating graceful shutdown …", sig_name)
    _shutdown_requested = True


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """Start the simple-ai-worker scheduler process.

    This function is referenced by the ``[project.scripts]`` entry in
    ``pyproject.toml`` so it can be invoked via::

        uv run simple-ai-worker

    Returns:
        None
    """
    global _runner, _registry, _executor  # noqa: PLW0603

    logger.info("simple-ai-worker starting …")

    # Register OS signal handlers before starting any background threads.
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Load AppSettings from .env (via pydantic-settings).
    settings = get_settings()
    logger.info(
        "Settings loaded (interval=%ds, timezone=%s, max_dev_agents=%d).",
        settings.SCHEDULER_INTERVAL_SECONDS,
        settings.SCHEDULER_TIMEZONE,
        settings.MAX_CONCURRENT_DEV_AGENTS,
    )

    # Load agent team config and build the registry.
    agent_team = load_agent_config(settings.AGENT_CONFIG_PATH)
    _registry = build_registry(agent_team, settings)
    logger.info(
        "Agent registry ready: %d agent(s) — %s.",
        len(_registry),
        ", ".join(_registry.agent_ids()),
    )

    # Create the bounded ThreadPoolExecutor used by scan_and_dispatch_job.
    _executor = ThreadPoolExecutor(
        max_workers=settings.MAX_CONCURRENT_DEV_AGENTS,
        thread_name_prefix="ai-worker",
    )
    logger.info(
        "ThreadPoolExecutor created (max_workers=%d).",
        settings.MAX_CONCURRENT_DEV_AGENTS,
    )

    # Start APScheduler with all registered jobs.
    _runner = SchedulerRunner(
        interval_seconds=settings.SCHEDULER_INTERVAL_SECONDS,
        timezone=settings.SCHEDULER_TIMEZONE,
        registry=_registry,
        settings=settings,
        executor=_executor,
    )
    _runner.start()

    logger.info("simple-ai-worker is running. Press Ctrl-C to stop.")

    # Block the main thread in a tight loop so signal handlers can fire.
    while not _shutdown_requested:
        time.sleep(1)

    # ------------------------------------------------------------------
    # Graceful teardown
    # ------------------------------------------------------------------
    logger.info("Shutting down …")

    _runner.stop(wait=True)
    logger.info("Scheduler stopped.")

    if _executor is not None:
        _executor.shutdown(wait=True)
        logger.info("ThreadPoolExecutor shut down.")

    logger.info("simple-ai-worker stopped cleanly.")


if __name__ == "__main__":
    main()
