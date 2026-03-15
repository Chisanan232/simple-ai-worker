"""
Role router for the Slack Bolt event handlers (Phase 6 + Phase 8a).

Provides :func:`role_router`, which parses the incoming Slack message body for
a ``[planner]``, ``[dev lead]``, or ``[dev]`` role tag and dispatches to the
appropriate handler module.

Routing rules
-------------
- Message contains ``[dev]``      → :func:`~src.slack_app.handlers.dev.dev_handler`
- Message contains ``[planner]``  → :func:`~src.slack_app.handlers.planner.planner_handler`
- Message contains ``[dev lead]`` → :func:`~src.slack_app.handlers.dev_lead.dev_lead_handler`
- No recognised tag               → bot replies with a usage hint

All checks are case-insensitive and the match is done on the **full text**
after stripping the ``<@BOTID>`` mention prefix.  ``[dev]`` is checked first
to avoid matching the ``[dev lead]`` tag (which also contains "dev").

Design notes
------------
- The router **must** return quickly (Slack requires ACK within 3 seconds).
  All Crew execution is therefore submitted to the supplied
  :class:`~concurrent.futures.ThreadPoolExecutor`; handlers never block.
- ``say`` is typed as ``Any`` to avoid importing Slack SDK internal types.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from src.agents.registry import AgentRegistry

from .handlers.dev import dev_handler
from .handlers.dev_lead import dev_lead_handler
from .handlers.planner import planner_handler

__all__: List[str] = ["role_router"]

logger = logging.getLogger(__name__)

# Regex patterns for role tag detection (case-insensitive).
# NOTE: _DEV_TAG_RE must match "[dev]" but NOT "[dev lead]".
# The pattern \[\s*dev\s*\] uses strict word boundary via the closing ].
_DEV_TAG_RE: re.Pattern[str] = re.compile(r"\[\s*dev\s*\]", re.IGNORECASE)
_PLANNER_TAG_RE: re.Pattern[str] = re.compile(r"\[planner\]", re.IGNORECASE)
_DEV_LEAD_TAG_RE: re.Pattern[str] = re.compile(r"\[dev\s+lead\]", re.IGNORECASE)

# Strip Slack user-mention tokens like <@U12345678> from the message text.
_MENTION_RE: re.Pattern[str] = re.compile(r"<@[A-Z0-9]+>", re.IGNORECASE)

_USAGE_HINT: str = (
    "👋 Hi! I'm *@ai-worker*. To chat with me, use one of these role tags:\n"
    "• `[dev] <update>` — talk to the Dev Agent (use inside a Slack thread)\n"
    "• `[planner] <your product requirement>` — talk to the Planner Agent\n"
    "• `[dev lead] <your directive>` — talk to the Dev Lead Agent"
)


def role_router(
    event: dict[str, Any],
    say: Any,
    registry: "AgentRegistry",
    executor: ThreadPoolExecutor,
) -> None:
    """Parse the Slack event and dispatch to the correct role handler.

    Parses the ``text`` field of the Slack event, strips ``<@BOTID>``
    mention tokens, and checks for ``[planner]`` / ``[dev lead]`` tags.
    Dispatches to the corresponding handler or sends a usage hint if no
    recognised tag is present.

    Args:
        event:    The Slack event payload dict (``app_mention`` or ``message``).
        say:      The Slack Bolt ``say()`` function for posting messages.
        registry: The shared :class:`~src.agents.registry.AgentRegistry`.
        executor: The bounded :class:`~concurrent.futures.ThreadPoolExecutor`
            for non-blocking Crew execution.
    """
    raw_text: str = event.get("text", "") or ""
    thread_ts: str | None = event.get("thread_ts") or event.get("ts")

    # Strip <@BOTID> mention tokens so they don't confuse tag detection.
    cleaned_text: str = _MENTION_RE.sub("", raw_text).strip()

    logger.debug(
        "role_router: raw_text=%.200s, cleaned=%.200s, thread_ts=%s",
        raw_text,
        cleaned_text,
        thread_ts,
    )

    if _DEV_TAG_RE.search(cleaned_text):
        logger.info("role_router: [dev] tag detected — dispatching to dev_handler.")
        dev_handler(
            text=cleaned_text,
            event=event,
            say=say,
            registry=registry,
            executor=executor,
        )

    elif _PLANNER_TAG_RE.search(cleaned_text):
        logger.info("role_router: [planner] tag detected — dispatching to planner_handler.")
        planner_handler(
            text=cleaned_text,
            event=event,
            say=say,
            registry=registry,
            executor=executor,
        )

    elif _DEV_LEAD_TAG_RE.search(cleaned_text):
        logger.info("role_router: [dev lead] tag detected — dispatching to dev_lead_handler.")
        dev_lead_handler(
            text=cleaned_text,
            event=event,
            say=say,
            registry=registry,
            executor=executor,
        )

    else:
        logger.info(
            "role_router: no recognised role tag in message (ts=%s). Sending usage hint.",
            thread_ts,
        )
        say(text=_USAGE_HINT, thread_ts=thread_ts)
