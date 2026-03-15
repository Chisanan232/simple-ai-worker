"""
Dev Agent Slack Bolt handler (Phase 8a — S3a/S3b).

Provides :func:`dev_handler`, called by
:func:`~src.slack_app.router.role_router` when a Slack message contains the
``[dev]`` tag.

This handler implements **S3a** (read thread → summarise → update ticket) and
**S3b / BR-6** (ask supervisor when no ticket ID is found in the thread).

Execution flow
--------------
1. Validate that the mention arrives in an **existing thread** (``thread_ts``
   present in event).  If not, reply with a usage hint and return — the Dev
   Agent only acts on concluded discussions.
2. Post an immediate "⏳ Reading the thread…" acknowledgement.
3. Submit :func:`_run_dev_thread_summary_crew` to the bounded
   :class:`~concurrent.futures.ThreadPoolExecutor`.
4. The crew:
   a. Calls ``slack/get_messages`` to fetch the full thread history.
   b. Extracts key conclusions and the ticket ID (JIRA key or ClickUp task ID).
   c. If no ticket ID found → asks supervisor in thread, stops (BR-6).
   d. Composes a structured comment with Slack permalink.
   e. Posts comment via ``jira/add_comment`` or ``clickup/add_comment``.
   f. Replies in thread confirming the update.
5. On error, a Python-level ``say()`` fallback posts the error to the thread.

Design notes
------------
- The handler **never** transitions ticket state (S3a only posts a comment).
- ``Do NOT create any new ticket`` and ``Do NOT transition the ticket state``
  are explicitly stated in the task description (defensive prompt guardrails).
- ``Do NOT set any ticket to the scan_for_work status`` (BR-1) is also
  stated even though this handler does no state changes — belt-and-suspenders.
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

__all__: List[str] = ["dev_handler"]

logger = logging.getLogger(__name__)

# Strip the [dev] role tag from the message body.
_DEV_TAG_RE: re.Pattern[str] = re.compile(r"\[\s*dev\s*\]", re.IGNORECASE)

_THINKING_MSG: str = "⏳ Reading the thread and updating the ticket …"

_NOT_IN_THREAD_MSG: str = (
    "👋 Mention me with `[dev]` inside an *existing* Slack thread after your team "
    "has reached a conclusion. I'll read the thread, summarise it, and update the "
    "relevant ticket."
)

_DEV_THREAD_SUMMARY_TASK_TEMPLATE: str = """
Your supervisor and the product planner have discussed a task in Slack and
reached a conclusion. The supervisor has mentioned you to pass along the
domain knowledge and decisions so you can begin development.

Slack context:
  - Channel ID:   {channel_id}
  - Thread ts:    {thread_ts}
  - Permalink:    https://slack.com/archives/{channel_id}/p{thread_ts_nodot}

Steps:
1. Use slack/get_messages with channel="{channel_id}" and thread_ts="{thread_ts}"
   to fetch ALL messages in the thread. Read every message carefully.

2. Extract key conclusions:
   - Decisions about the implementation approach.
   - Domain constraints and business rules.
   - Clarifications about acceptance criteria.

3. Identify the task ticket ID:
   a. Scan all thread messages for a JIRA key (e.g. PROJ-42) or ClickUp URL/ID.
   b. If found: proceed to step 4.
   c. If NOT found: reply via slack/reply_to_thread:
      "I couldn't identify a ticket from this thread. Please reply with the
       ticket ID (e.g., PROJ-42) and I'll update it right away."
      Then STOP — do not post any comment until the supervisor replies.

4. Compose the ticket comment:
   ## Discussion Summary
   - [key conclusion bullet points]

   **Reference:** [Slack thread](https://slack.com/archives/{channel_id}/p{thread_ts_nodot})

5. Post the comment:
   - JIRA: use jira/add_comment with the ticket key and comment body.
   - ClickUp: use clickup/add_comment with the task ID and comment body.

6. Reply in the Slack thread via slack/reply_to_thread:
   "✅ Got it! I've summarised the discussion and added a comment to
    [ticket ID] with a link back to this thread. Ready to start development."

IMPORTANT:
  - Do NOT create any new ticket.
  - Do NOT transition the ticket state.
  - Do NOT set any ticket to the scan_for_work status — it is human-only (BR-1).
  - Only update the existing ticket with the discussion summary comment.
"""

_DEV_THREAD_SUMMARY_EXPECTED_OUTPUT: str = (
    "A confirmation that the Slack thread was read, the ticket comment was posted "
    "(include ticket ID and a note that the Slack permalink was included), and a "
    "reply was posted in the Slack thread. OR a question to the supervisor asking "
    "for the ticket ID if none was found in the thread."
)


def _run_dev_thread_summary_crew(
    channel_id: str,
    thread_ts: str,
    thread_ts_nodot: str,
    say: Any,
    registry: "AgentRegistry",
) -> None:
    """Execute the Dev Thread Summary Crew in a background thread.

    Args:
        channel_id:       Slack channel ID.
        thread_ts:        Thread timestamp (dotted, e.g. ``"111.222"``).
        thread_ts_nodot:  Thread timestamp without the dot (``"111222"``).
        say:              Slack Bolt ``say()`` callable for error fallback.
        registry:         The shared :class:`~src.agents.registry.AgentRegistry`.
    """
    try:
        dev_agent = registry["dev_agent"]
    except KeyError:
        logger.error("dev_handler: 'dev_agent' not found in registry.")
        say(
            text="❌ Configuration error: the Dev agent is not available. Please contact the admin.",
            thread_ts=thread_ts,
        )
        return

    task = Task(
        description=_DEV_THREAD_SUMMARY_TASK_TEMPLATE.format(
            channel_id=channel_id,
            thread_ts=thread_ts,
            thread_ts_nodot=thread_ts_nodot,
        ),
        expected_output=_DEV_THREAD_SUMMARY_EXPECTED_OUTPUT,
        agent=dev_agent,
    )

    crew = CrewBuilder.build(
        agents=[dev_agent],
        tasks=[task],
        process="sequential",
    )

    try:
        result = crew.kickoff()
        logger.info(
            "dev_handler: crew completed. Result preview: %.300s",
            str(result),
        )
        # The agent posts the full reply via slack/reply_to_thread natively.
        # We log but do not post again via say() to avoid duplicate messages.

    except Exception as exc:  # noqa: BLE001
        logger.exception("dev_handler: crew raised an exception.")
        say(
            text=(
                f"❌ Something went wrong while the Dev agent was processing the thread:\n"
                f"```{exc}```\nPlease try again or contact the admin."
            ),
            thread_ts=thread_ts,
        )


def dev_handler(
    text: str,
    event: dict[str, Any],
    say: Any,
    registry: "AgentRegistry",
    executor: ThreadPoolExecutor,
) -> None:
    """Handle an ``@ai-worker [dev]`` Slack message in an existing thread.

    Validates that the mention is inside an existing thread (``thread_ts``
    present), posts an acknowledgement, then submits the Dev Thread Summary
    Crew to the background executor.

    Args:
        text:     The cleaned message text (``<@BOTID>`` tokens already removed
                  by the router).
        event:    The full Slack event payload dict.
        say:      Slack Bolt ``say()`` callable.
        registry: The shared :class:`~src.agents.registry.AgentRegistry`.
        executor: The bounded :class:`~concurrent.futures.ThreadPoolExecutor`.
    """
    thread_ts: str | None = event.get("thread_ts") or event.get("ts")
    channel_id: str = event.get("channel", "")

    # Only act when the mention is inside an existing thread.
    if not event.get("thread_ts"):
        logger.info("dev_handler: [dev] mention is not inside an existing thread — sending hint.")
        say(text=_NOT_IN_THREAD_MSG, thread_ts=thread_ts)
        return

    # Sanitise: remove dot for permalink construction.
    thread_ts_str: str = str(thread_ts)
    thread_ts_nodot: str = thread_ts_str.replace(".", "")

    logger.info(
        "dev_handler: dispatching thread summary crew for channel=%s, thread_ts=%s …",
        channel_id,
        thread_ts,
    )

    # Post immediate acknowledgement before submitting to executor.
    say(text=_THINKING_MSG, thread_ts=thread_ts)

    executor.submit(
        _run_dev_thread_summary_crew,
        channel_id,
        thread_ts_str,
        thread_ts_nodot,
        say,
        registry,
    )
