"""
Planner Slack Bolt handler (Phase 6).

Provides :func:`planner_handler`, called by
:func:`~src.slack_app.router.role_router` when a Slack message contains the
``[planner]`` tag.

Execution flow
--------------
1. Strip the ``[planner]`` prefix from the message text.
2. Extract ``thread_ts`` from the event so replies land in the same thread.
3. Post an immediate "thinking…" acknowledgement so the human knows the bot
   is working (Slack requires ACK within 3 s; the real Crew runs async).
4. Submit Crew execution to the bounded :class:`~concurrent.futures.ThreadPoolExecutor`.
5. Crew result (or error message) is posted back to the thread via
   ``say(thread_ts=...)``.

Note: The Planner Agent itself also uses ``slack/reply_to_thread`` as part of
its native CrewAI Enterprise app integrations, but the Python-level ``say()``
call here is used for the "thinking…" acknowledgement and error recovery —
these are lightweight Bolt-layer posts that do not go through the CrewAI
platform.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, List

from crewai import Task

from src.crew.builder import CrewBuilder

if TYPE_CHECKING:
    from src.agents.registry import AgentRegistry

__all__: List[str] = ["planner_handler"]

logger = logging.getLogger(__name__)

# Strip the role tag prefix (and any leading/trailing whitespace) from the
# message body.  Handles variations like [Planner], [ planner ], etc.
_PLANNER_TAG_RE: re.Pattern[str] = re.compile(r"\[\s*planner\s*\]", re.IGNORECASE)

_THINKING_MSG: str = "⏳ On it! Let me think about your product requirement …"

_PLANNER_TASK_DESCRIPTION_TEMPLATE: str = """
A human stakeholder has sent you the following product requirement via Slack:

---
{human_message}
---

Your task:
1. Carefully read and understand the requirement.
2. If the requirement is **ambiguous or incomplete**, ask targeted clarifying
   questions to resolve uncertainty before creating any tickets.
3. If the requirement is **clear and actionable**:
   a. Create a JIRA epic via jira/create_issue with a descriptive title,
      acceptance criteria, and priority.
   b. Create a matching ClickUp task via clickup/create_task that mirrors
      the JIRA epic.
4. Reply in the Slack thread via slack/reply_to_thread with:
   - A brief summary of your understanding of the requirement.
   - Either: the JIRA epic key + ClickUp task ID that were created, OR
   - The clarifying questions you need answered before proceeding.

Be concise, professional, and confirm your understanding before acting.
"""

_PLANNER_TASK_EXPECTED_OUTPUT: str = (
    "A structured reply posted in the Slack thread: either clarifying questions "
    "OR a confirmation that the JIRA epic and ClickUp task have been created "
    "(include ticket keys/IDs and URLs)."
)


def _run_planner_crew(
    human_message: str,
    thread_ts: str | None,
    say: Any,
    registry: "AgentRegistry",
) -> None:
    """Execute the Planner Crew in a background thread.

    Args:
        human_message: The clean message body (role tag already stripped).
        thread_ts:     Slack thread timestamp for reply threading.
        say:           Slack Bolt ``say()`` callable for error fallback.
        registry:      The shared :class:`~src.agents.registry.AgentRegistry`.
    """
    try:
        planner_agent = registry["planner"]
    except KeyError:
        logger.error("planner_handler: 'planner' agent not found in registry.")
        say(
            text="❌ Configuration error: the Planner agent is not available. Please contact the admin.",
            thread_ts=thread_ts,
        )
        return

    task = Task(
        description=_PLANNER_TASK_DESCRIPTION_TEMPLATE.format(
            human_message=human_message,
        ),
        expected_output=_PLANNER_TASK_EXPECTED_OUTPUT,
        agent=planner_agent,
    )

    crew = CrewBuilder.build(
        agents=[planner_agent],
        tasks=[task],
        process="sequential",
    )

    try:
        result = crew.kickoff()
        logger.info(
            "planner_handler: crew completed. Result preview: %.300s",
            str(result),
        )
        # The agent posts the full reply via slack/reply_to_thread natively.
        # We log the result but do not post it again through say() to avoid
        # duplicate messages in the thread.

    except Exception as exc:  # noqa: BLE001
        logger.exception("planner_handler: crew raised an exception.")
        say(
            text=(
                f"❌ Something went wrong while the Planner was processing your request:\n"
                f"```{exc}```\nPlease try again or contact the admin."
            ),
            thread_ts=thread_ts,
        )


def planner_handler(
    text: str,
    event: dict[str, Any],
    say: Any,
    registry: "AgentRegistry",
    executor: ThreadPoolExecutor,
) -> None:
    """Handle an ``@ai-worker [planner]`` Slack message.

    Strips the role tag, posts an acknowledgement, then submits the Planner
    Crew to the background executor so Bolt ACKs within the 3-second window.

    Args:
        text:     The cleaned message text (``<@BOTID>`` tokens already removed
                  by the router).
        event:    The full Slack event payload dict.
        say:      Slack Bolt ``say()`` callable.
        registry: The shared :class:`~src.agents.registry.AgentRegistry`.
        executor: The bounded :class:`~concurrent.futures.ThreadPoolExecutor`.
    """
    thread_ts: str | None = event.get("thread_ts") or event.get("ts")

    # Strip the [planner] tag to get the clean human message.
    human_message: str = _PLANNER_TAG_RE.sub("", text).strip()

    if not human_message:
        say(
            text="👋 You mentioned *[planner]* but didn't include a message. What would you like to plan?",
            thread_ts=thread_ts,
        )
        return

    logger.info(
        "planner_handler: dispatching crew for thread_ts=%s, message=%.100s …",
        thread_ts,
        human_message,
    )

    # Post immediate acknowledgement before submitting to executor.
    say(text=_THINKING_MSG, thread_ts=thread_ts)

    executor.submit(
        _run_planner_crew,
        human_message,
        thread_ts,
        say,
        registry,
    )

