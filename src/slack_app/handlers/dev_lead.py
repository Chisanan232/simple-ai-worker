"""
Dev Lead Slack Bolt handler (Phase 6).

Provides :func:`dev_lead_handler`, called by
:func:`~src.slack_app.router.role_router` when a Slack message contains the
``[dev lead]`` tag.

Execution flow
--------------
1. Strip the ``[dev lead]`` prefix from the message text.
2. Extract ``thread_ts`` from the event so replies land in the same thread.
3. Post an immediate "thinking…" acknowledgement (Bolt ACK within 3 s).
4. Submit Crew execution to the bounded :class:`~concurrent.futures.ThreadPoolExecutor`.
5. The Dev Lead Agent fetches the referenced ticket, applies changes
   (sub-task creation, dependency annotation, implementation notes), and
   uses ``slack/reply_to_thread`` to reply in the original Slack thread.
6. On error, a Python-level ``say()`` fallback posts the error to the thread.
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

__all__: List[str] = ["dev_lead_handler"]

logger = logging.getLogger(__name__)

# Strip the role tag prefix from the message body.
_DEV_LEAD_TAG_RE: re.Pattern[str] = re.compile(r"\[\s*dev\s+lead\s*\]", re.IGNORECASE)

_THINKING_MSG: str = "⏳ On it! Analysing your request and reviewing the tickets …"

_DEV_LEAD_TASK_DESCRIPTION_TEMPLATE: str = """
A human stakeholder has sent you the following directive via Slack:

---
{human_message}
---

The message may contain one of the following request types:

**A. Epic/Story Breakdown Request**
   The human wants you to decompose a high-level JIRA/ClickUp epic or story
   into well-scoped, independently executable sub-tasks.
   Steps:
   1. Extract the ticket ID or description from the message.
   2. Fetch the parent ticket via jira/search_issues or clickup/search_tasks.
   3. Decompose into sub-tasks: each sub-task must have:
      - A clear title and acceptance criteria.
      - Explicit dependencies on other sub-tasks (if any).
      - Implementation notes so a developer can begin immediately.
   4. Create the sub-tasks via jira/create_issue (or clickup/create_task).
   5. Update the parent ticket to link the sub-tasks (jira/update_issue or
      clickup/update_task).
   6. Reply in the Slack thread via slack/reply_to_thread with a structured
      summary of the breakdown (sub-task keys, titles, dependencies).

**B. Ticket Amendment / Refinement Request**
   The human wants to update, refine, or add context to an existing ticket.
   Steps:
   1. Extract the ticket ID and the specific amendment from the message.
   2. Fetch the ticket via jira/get_issue or clickup/get_task.
   3. Apply the requested changes:
      - Update fields, status, assignee, or priority via jira/update_issue
        or clickup/update_task.
      - Add implementation notes / context via jira/add_comment or
        clickup/add_comment.
      - Update dependencies via jira/update_issue or clickup/update_task.
   4. Transition the ticket if a new status is specified (jira/transition_issue
      or clickup/update_task with the new status).
   5. Reply in the Slack thread via slack/reply_to_thread with a confirmation
      of every change made (field names, old → new values where applicable).

Determine which request type applies from the message content and execute the
appropriate steps. Be thorough, structured, and always confirm your actions.
"""

_DEV_LEAD_TASK_EXPECTED_OUTPUT: str = (
    "A structured reply posted in the Slack thread confirming all actions taken: "
    "either a breakdown summary (sub-task keys, titles, dependencies) or an "
    "amendment summary (ticket ID, fields updated, comments added)."
)


def _run_dev_lead_crew(
    human_message: str,
    thread_ts: str | None,
    say: Any,
    registry: "AgentRegistry",
) -> None:
    """Execute the Dev Lead Crew in a background thread.

    Args:
        human_message: The clean message body (role tag already stripped).
        thread_ts:     Slack thread timestamp for reply threading.
        say:           Slack Bolt ``say()`` callable for error fallback.
        registry:      The shared :class:`~src.agents.registry.AgentRegistry`.
    """
    try:
        dev_lead_agent = registry["dev_lead"]
    except KeyError:
        logger.error("dev_lead_handler: 'dev_lead' agent not found in registry.")
        say(
            text="❌ Configuration error: the Dev Lead agent is not available. Please contact the admin.",
            thread_ts=thread_ts,
        )
        return

    task = Task(
        description=_DEV_LEAD_TASK_DESCRIPTION_TEMPLATE.format(
            human_message=human_message,
        ),
        expected_output=_DEV_LEAD_TASK_EXPECTED_OUTPUT,
        agent=dev_lead_agent,
    )

    crew = CrewBuilder.build(
        agents=[dev_lead_agent],
        tasks=[task],
        process="sequential",
    )

    try:
        result = crew.kickoff()
        logger.info(
            "dev_lead_handler: crew completed. Result preview: %.300s",
            str(result),
        )
        # The agent posts the full reply via slack/reply_to_thread natively.

    except Exception as exc:  # noqa: BLE001
        logger.exception("dev_lead_handler: crew raised an exception.")
        say(
            text=(
                f"❌ Something went wrong while the Dev Lead was processing your request:\n"
                f"```{exc}```\nPlease try again or contact the admin."
            ),
            thread_ts=thread_ts,
        )


def dev_lead_handler(
    text: str,
    event: dict[str, Any],
    say: Any,
    registry: "AgentRegistry",
    executor: ThreadPoolExecutor,
) -> None:
    """Handle an ``@ai-worker [dev lead]`` Slack message.

    Strips the role tag, posts an acknowledgement, then submits the Dev Lead
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

    # Strip the [dev lead] tag to get the clean human message.
    human_message: str = _DEV_LEAD_TAG_RE.sub("", text).strip()

    if not human_message:
        say(
            text=(
                "👋 You mentioned *[dev lead]* but didn't include a directive. "
                "Try something like: `[dev lead] Break down PROJ-42 into sub-tasks.`"
            ),
            thread_ts=thread_ts,
        )
        return

    logger.info(
        "dev_lead_handler: dispatching crew for thread_ts=%s, message=%.100s …",
        thread_ts,
        human_message,
    )

    # Post immediate acknowledgement before submitting to executor.
    say(text=_THINKING_MSG, thread_ts=thread_ts)

    executor.submit(
        _run_dev_lead_crew,
        human_message,
        thread_ts,
        say,
        registry,
    )

