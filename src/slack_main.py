"""
Entry-point for the Slack Events API webhook process (simple-ai-slack).

Run with::

    uv run simple-ai-slack

Startup sequence:
    1. Configure root logging.
    2. Load :class:`~src.config.settings.AppSettings` from ``.env`` via
       :func:`~src.config.get_settings`.
    3. Load :class:`~src.config.agent_config.AgentTeamConfig` + build
       :class:`~src.agents.registry.AgentRegistry`.
    4. Create the bounded :class:`~concurrent.futures.ThreadPoolExecutor`.
    5. Build the :class:`~slack_bolt.async_app.AsyncApp` via
       :func:`~src.slack_app.app.create_bolt_app`.
    6. Start the built-in Bolt HTTP server:
       ``await app.start(port=settings.SLACK_PORT)``.

       - Binds to ``0.0.0.0:<SLACK_PORT>``.
       - ``POST /slack/events`` — Bolt verifies HMAC signatures, handles the
         URL-challenge handshake, and dispatches ``app_mention`` / ``message``
         events to the registered async handlers.
       - Blocks until ``SIGINT`` / ``SIGTERM``; Bolt handles graceful shutdown
         internally — no manual signal wiring is needed here.

This process runs **no scheduler code at all**.  APScheduler jobs are run by
the separate ``simple-ai-worker`` process (entry point: ``src/main.py``).
"""

from __future__ import annotations

import asyncio
import logging
import sys
from concurrent.futures import ThreadPoolExecutor

from src.agents.registry import build_registry
from src.config import get_settings, load_agent_config
from src.slack_app.app import create_bolt_app

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
# Async runner
# ---------------------------------------------------------------------------


async def _run() -> None:
    """Async body of the Slack webhook process."""
    settings = get_settings()
    logger.info(
        "Settings loaded (slack_port=%d, max_dev_agents=%d).",
        settings.SLACK_PORT,
        settings.MAX_CONCURRENT_DEV_AGENTS,
    )

    agent_team = load_agent_config(settings.AGENT_CONFIG_PATH)
    registry = build_registry(agent_team, settings)
    logger.info(
        "Agent registry ready: %d agent(s) — %s.",
        len(registry),
        ", ".join(registry.agent_ids()),
    )

    executor = ThreadPoolExecutor(
        max_workers=settings.MAX_CONCURRENT_DEV_AGENTS,
        thread_name_prefix="ai-slack-worker",
    )
    logger.info(
        "ThreadPoolExecutor created (max_workers=%d).",
        settings.MAX_CONCURRENT_DEV_AGENTS,
    )

    app = create_bolt_app(settings=settings, registry=registry, executor=executor)

    logger.info(
        "Starting Slack Events API HTTP server on port %d (POST /slack/events) …",
        settings.SLACK_PORT,
    )
    # Blocks until SIGINT/SIGTERM.  Bolt handles:
    #   - URL challenge verification (Slack → our server handshake)
    #   - HMAC signature verification on every request
    #   - Event dispatch to registered async handlers
    await app.start(port=settings.SLACK_PORT)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """Slack webhook entry-point — referenced by ``pyproject.toml`` ``[project.scripts]``.

    Invoked via::

        uv run simple-ai-slack
    """
    asyncio.run(_run())


if __name__ == "__main__":
    main()
