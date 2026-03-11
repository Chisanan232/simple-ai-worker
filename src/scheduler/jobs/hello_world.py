"""
Hello-world placeholder job.

This module provides :func:`hello_world_job`, a no-op APScheduler job
used in **Phase 1** to validate the scheduler lifecycle end-to-end
before any real agent logic is wired in.

The job simply logs a timestamped message at ``INFO`` level.  It will be
replaced / supplemented by real jobs in Phase 6.
"""

from __future__ import annotations

import datetime
import logging

logger = logging.getLogger(__name__)


def hello_world_job() -> None:
    """Log a timestamped hello-world message.

    This is the Phase-1 placeholder job.  It has no side effects beyond
    writing to the application logger and is safe to run in any
    environment including CI.

    Returns:
        None
    """
    now: datetime.datetime = datetime.datetime.now(tz=datetime.timezone.utc)
    logger.info("Hello World from scheduler — %s", now.isoformat())
