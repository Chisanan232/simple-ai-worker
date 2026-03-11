"""
Application entry-point for simple-ai-worker.

Startup sequence (Phase 2):
    1. Configure root logging.
    2. Load :class:`~src.config.settings.AppSettings` from ``.env`` via
       :func:`~src.config.get_settings`.
    3. Register ``SIGINT`` / ``SIGTERM`` handlers for graceful shutdown.
    4. Instantiate :class:`~src.scheduler.runner.SchedulerRunner` using
       values from :class:`~src.config.settings.AppSettings`.
    5. Start the scheduler.
    6. Block the main thread until a signal is received.

Later phases will extend this sequence with:
    - Loading agent configuration from ``config/agents.yaml``.
    - Building the :class:`~src.agents.registry.AgentRegistry`.
    - Starting the Slack Bolt Socket-Mode server in a background thread.
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from types import FrameType

from src.config import get_settings
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


def _handle_signal(signum: int, frame: Optional[FrameType]) -> None:  # noqa: ARG001
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
    """Start the simple-ai-worker process.

    This function is referenced by the ``[project.scripts]`` entry in
    ``pyproject.toml`` so it can be invoked via::

        uv run simple-ai-worker

    Returns:
        None
    """
    global _runner  # noqa: PLW0603

    logger.info("simple-ai-worker starting …")

    # Register OS signal handlers before starting any background threads.
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Phase 2: load AppSettings from .env (via pydantic-settings).
    # Interval and timezone are now sourced from the settings model.
    settings = get_settings()
    logger.info(
        "Settings loaded (interval=%ds, timezone=%s, max_dev_agents=%d).",
        settings.SCHEDULER_INTERVAL_SECONDS,
        settings.SCHEDULER_TIMEZONE,
        settings.MAX_CONCURRENT_DEV_AGENTS,
    )

    _runner = SchedulerRunner(
        interval_seconds=settings.SCHEDULER_INTERVAL_SECONDS,
        timezone=settings.SCHEDULER_TIMEZONE,
    )
    _runner.start()

    logger.info("simple-ai-worker is running. Press Ctrl-C to stop.")

    # Block the main thread in a tight loop so signal handlers can fire.
    # ``time.sleep(1)`` keeps CPU usage negligible while remaining
    # responsive to signals within ~1 second.
    while not _shutdown_requested:
        time.sleep(1)

    # Graceful teardown.
    logger.info("Shutting down …")
    _runner.stop(wait=True)
    logger.info("simple-ai-worker stopped cleanly.")


if __name__ == "__main__":
    main()
