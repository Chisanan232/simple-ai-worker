"""
Slack Bolt application factory (Phase 6).

Provides :func:`create_bolt_app` which builds a configured
:class:`slack_bolt.App` instance operating in Socket Mode.

The app registers two event types:
- ``app_mention``  — fires when the bot (``@ai-worker``) is mentioned in a
  channel.
- ``message``      — fires for direct messages (DMs) sent to ``@ai-worker``.

Both event types are routed through :func:`~src.slack_app.router.role_router`
which parses the ``[planner]`` / ``[dev lead]`` role tag and calls the
appropriate handler.

A :class:`slack_bolt.adapter.socket_mode.SocketModeHandler` wraps the app to
establish the persistent WebSocket connection without requiring a publicly
reachable HTTP endpoint.

Usage::

    from src.slack_app.app import create_bolt_app
    from slack_bolt.adapter.socket_mode import SocketModeHandler

    settings = get_settings()
    registry = build_registry(...)

    bolt_app, handler = create_bolt_app(settings, registry, executor)
    handler.connect()   # non-blocking background thread
    # … application runs …
    handler.close()
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, List, Tuple

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

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
) -> Tuple[App, SocketModeHandler]:
    """Build and return a configured Slack Bolt app with a Socket Mode handler.

    The app listens for:
    - ``app_mention`` events (``@ai-worker`` mentioned in a channel).
    - ``message`` events with ``subtype`` absent (direct messages to the bot).

    Both are dispatched through :func:`~src.slack_app.router.role_router`.

    Args:
        settings: Application settings containing Slack credentials.
            ``SLACK_BOT_TOKEN``, ``SLACK_APP_TOKEN``, and
            ``SLACK_SIGNING_SECRET`` must all be set.
        registry: The shared :class:`~src.agents.registry.AgentRegistry`
            populated at startup.
        executor: The bounded :class:`~concurrent.futures.ThreadPoolExecutor`
            used to run Crew executions without blocking the Bolt event loop.

    Returns:
        A tuple of ``(bolt_app, socket_mode_handler)``.  Call
        ``socket_mode_handler.connect()`` to start the WebSocket connection
        in a background thread.

    Raises:
        ValueError: If any required Slack credential is missing from settings.
    """
    # ------------------------------------------------------------------
    # Validate required credentials
    # ------------------------------------------------------------------
    missing: list[str] = []
    if not settings.SLACK_BOT_TOKEN:
        missing.append("SLACK_BOT_TOKEN")
    if not settings.SLACK_APP_TOKEN:
        missing.append("SLACK_APP_TOKEN")
    if not settings.SLACK_SIGNING_SECRET:
        missing.append("SLACK_SIGNING_SECRET")

    if missing:
        raise ValueError(
            f"create_bolt_app: missing required Slack credentials in settings: "
            f"{', '.join(missing)}. Set them in .env before starting the Slack app."
        )

    bot_token: str = settings.SLACK_BOT_TOKEN.get_secret_value()  # type: ignore[union-attr]
    app_token: str = settings.SLACK_APP_TOKEN.get_secret_value()  # type: ignore[union-attr]
    signing_secret: str = settings.SLACK_SIGNING_SECRET.get_secret_value()  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Build the Bolt app
    # ------------------------------------------------------------------
    bolt_app = App(
        token=bot_token,
        signing_secret=signing_secret,
    )

    # ------------------------------------------------------------------
    # Register event handlers
    # ------------------------------------------------------------------

    @bolt_app.event("app_mention")
    def handle_app_mention(event: dict, say: object) -> None:  # type: ignore[type-arg]
        """Handle @ai-worker mentions in channels.

        Delegates to :func:`~src.slack_app.router.role_router` to parse
        the ``[planner]`` / ``[dev lead]`` tag and invoke the correct handler.
        """
        logger.debug("Bolt: received app_mention event: %s", event.get("ts"))
        role_router(event=event, say=say, registry=registry, executor=executor)

    @bolt_app.event("message")
    def handle_dm(event: dict, say: object) -> None:  # type: ignore[type-arg]
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

    logger.info("Slack Bolt app created. Registering Socket Mode handler …")

    # ------------------------------------------------------------------
    # Wrap with Socket Mode handler (WebSocket transport)
    # ------------------------------------------------------------------
    socket_handler = SocketModeHandler(app=bolt_app, app_token=app_token)

    return bolt_app, socket_handler

