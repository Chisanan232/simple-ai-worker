"""
Slack Bolt application factory (Phase 6 — Events API revision).

Provides :func:`create_bolt_app` which builds a configured
:class:`slack_bolt.async_app.AsyncApp` instance for the Events API HTTP webhook.

The app registers two event types:

- ``app_mention``  — fires when the bot (``@ai-worker``) is mentioned in a
  channel.
- ``message``      — fires for direct messages (DMs) sent to ``@ai-worker``.

Both event types are routed through :func:`~src.slack_app.router.role_router`
which parses the ``[planner]`` / ``[dev lead]`` role tag and calls the
appropriate handler.

The caller starts the built-in Bolt HTTP server by calling
``await app.start(port=settings.SLACK_PORT)``, which:

- Binds an ``aiohttp`` server to ``0.0.0.0:<SLACK_PORT>``.
- Handles URL-challenge verification automatically.
- Verifies HMAC signatures on every incoming request.
- Dispatches events to the registered async handlers.

No ``SocketModeHandler``, no uvicorn, no FastAPI — just the single
``await app.start(port=...)`` call in ``src/slack_main.py``.

Usage::

    from src.slack_app.app import create_bolt_app

    settings = get_settings()
    registry = build_registry(...)

    app = create_bolt_app(settings, registry, executor)
    await app.start(port=settings.SLACK_PORT)   # blocks until SIGINT/SIGTERM
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, List

from slack_bolt.async_app import AsyncApp

from .router import role_router

if TYPE_CHECKING:
    from src.agents.registry import AgentRegistry
    from src.config.settings import AppSettings

__all__: List[str] = ["create_bolt_app"]

logger = logging.getLogger(__name__)


def create_bolt_app(
    settings: "AppSettings",
    registry: "AgentRegistry",
    executor: ThreadPoolExecutor,
) -> AsyncApp:
    """Build and return a configured Slack Bolt ``AsyncApp`` for the Events API.

    The app listens for:

    - ``app_mention`` events (``@ai-worker`` mentioned in a channel).
    - ``message`` events with ``subtype`` absent (direct messages to the bot).

    Both are dispatched through :func:`~src.slack_app.router.role_router`.

    The returned ``AsyncApp`` is ready to serve HTTP requests; the caller
    starts the built-in server with ``await app.start(port=...)``.

    Args:
        settings: Application settings containing Slack credentials.
            ``SLACK_BOT_TOKEN`` and ``SLACK_SIGNING_SECRET`` must both be set.
        registry: The shared :class:`~src.agents.registry.AgentRegistry`
            populated at startup.
        executor: The bounded :class:`~concurrent.futures.ThreadPoolExecutor`
            used to run Crew executions without blocking the Bolt event loop.

    Returns:
        A configured :class:`~slack_bolt.async_app.AsyncApp` instance.
        Call ``await app.start(port=<SLACK_PORT>)`` to begin serving requests.

    Raises:
        ValueError: If any required Slack credential is missing from settings.
    """
    # ------------------------------------------------------------------
    # Validate required credentials
    # ------------------------------------------------------------------
    missing: list[str] = []
    if not settings.SLACK_BOT_TOKEN:
        missing.append("SLACK_BOT_TOKEN")
    if not settings.SLACK_SIGNING_SECRET:
        missing.append("SLACK_SIGNING_SECRET")

    if missing:
        raise ValueError(
            f"create_bolt_app: missing required Slack credentials in settings: "
            f"{', '.join(missing)}. Set them in .env before starting the Slack app."
        )

    bot_token: str = settings.SLACK_BOT_TOKEN.get_secret_value()  # type: ignore[union-attr]
    signing_secret: str = settings.SLACK_SIGNING_SECRET.get_secret_value()  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Build the AsyncApp (Events API — no SocketModeHandler)
    # ------------------------------------------------------------------
    app = AsyncApp(
        token=bot_token,
        signing_secret=signing_secret,
    )

    # ------------------------------------------------------------------
    # Register async event handlers
    # ------------------------------------------------------------------

    @app.event("app_mention")
    async def handle_app_mention(event: dict[str, Any], say: Any) -> None:
        """Handle @ai-worker mentions in channels.

        Delegates to :func:`~src.slack_app.router.role_router` to parse
        the ``[planner]`` / ``[dev lead]`` tag and invoke the correct handler.
        """
        logger.debug("Bolt: received app_mention event: %s", event.get("ts"))
        role_router(event=event, say=say, registry=registry, executor=executor)

    @app.event("message")
    async def handle_dm(event: dict[str, Any], say: Any) -> None:
        """Handle direct messages sent to @ai-worker.

        Only processes DMs (``channel_type == "im"``).  Any other message
        subtypes (e.g. bot messages, file shares) are silently ignored.
        """
        # Filter out bot messages and other non-user subtypes to prevent
        # infinite loops where the bot responds to its own messages.
        if event.get("subtype"):
            return
        if event.get("bot_id"):
            return
        if event.get("channel_type") != "im":
            return

        logger.debug("Bolt: received DM event: %s", event.get("ts"))
        role_router(event=event, say=say, registry=registry, executor=executor)

    logger.info(
        "Slack Bolt AsyncApp created (Events API). " "Call `await app.start(port=...)` to begin serving requests."
    )

    return app
