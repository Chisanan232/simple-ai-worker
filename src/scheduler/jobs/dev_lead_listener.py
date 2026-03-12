"""
Dev Lead listener fallback job (Phase 6 — APScheduler polling).

Provides :func:`dev_lead_listener_job`, an interval-based APScheduler job that
acts as the **fallback / batch-polling path** for the Dev Lead Agent.

Under normal operation the Slack Bolt Socket Mode handler
(:mod:`src.slack_app.handlers.dev_lead`) processes ``@ai-worker [dev lead]``
messages in real-time.  This job exists to catch any messages that were missed
during a Bolt downtime window.  It:

1. Builds a short-lived Crew with the Dev Lead Agent.
2. Asks it to check ``slack/get_messages`` for unprocessed ``[dev lead]``
   messages since the last processed timestamp.
3. For each new directive, fetches the referenced ticket, applies the
   amendments (sub-task creation, dependency annotations, implementation
   notes), and replies in the Slack thread.
4. Updates ``_last_processed_ts`` so already-handled messages are not
   re-processed on the next run.

Usage (injected by :class:`~src.scheduler.runner.SchedulerRunner`)::

    from src.scheduler.jobs.dev_lead_listener import dev_lead_listener_job

    scheduler.add_job(
        func=dev_lead_listener_job,
        kwargs={"registry": registry, "settings": settings},
        trigger="interval",
        seconds=300,
        id="dev_lead_listener",
    )
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, List, Optional

from crewai import Task

from src.crew.builder import CrewBuilder

if TYPE_CHECKING:
    from src.agents.registry import AgentRegistry
    from src.config.settings import AppSettings

__all__: List[str] = ["dev_lead_listener_job"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State: watermark timestamp for deduplication (see module docstring).
# ---------------------------------------------------------------------------
_last_processed_ts: Optional[float] = None

_DEV_LEAD_FALLBACK_TASK_DESCRIPTION_TEMPLATE: str = """
You are the Dev Lead Agent operating in **fallback batch-polling mode**.

The real-time Slack Bolt handler may have been temporarily unavailable.
Your job is to check for any unprocessed messages that were missed.

Steps:
1. Use slack/get_messages to fetch recent messages from the monitored Slack
   channel. Look for messages posted after timestamp {since_ts} that mention
   @ai-worker and contain the tag '[dev lead]' in the message text.

2. For each such unprocessed message:
   a. Strip the '@ai-worker [dev lead]' prefix from the message body to get
      the clean directive text. The message may contain:
      - A request to break down a JIRA/ClickUp epic into sub-tasks.
      - A ticket ID + specific implementation directive or refinement request.
   b. For a breakdown request:
      - Use jira/search_issues or clickup/search_tasks to fetch the parent
        epic/task.
      - Create well-scoped sub-tasks with acceptance criteria and dependency
        annotations (jira/create_issue, clickup/create_task).
      - Update the parent ticket with sub-task links (jira/update_issue,
        clickup/update_task).
   c. For an amendment request on an existing ticket:
      - Fetch the ticket using jira/get_issue or clickup/get_task.
      - Apply the requested changes (update fields, add implementation notes
        via jira/add_comment or clickup/add_comment, set dependencies).
   d. Reply in the Slack thread via slack/reply_to_thread with a structured
      summary of all changes made.

3. If no new '[dev lead]' messages are found, do nothing and report
   "No new dev lead messages found since {since_ts}."

Important: Do NOT re-process messages with timestamps at or before {since_ts}.
"""

_DEV_LEAD_FALLBACK_TASK_EXPECTED_OUTPUT: str = (
    "A summary of actions taken: either 'No new dev lead messages found' or "
    "a list of messages processed with ticket IDs created/updated and sub-tasks "
    "created for each breakdown request."
)


def dev_lead_listener_job(
    registry: "AgentRegistry",
    settings: "AppSettings",  # noqa: ARG001  (reserved for future use)
) -> None:
    """APScheduler fallback job: poll Slack for missed ``[dev lead]`` messages.

    This job is the safety net for the Slack Bolt Socket Mode dev-lead handler.
    Under normal operation it will find no new messages to process.

    Args:
        registry: The shared :class:`~src.agents.registry.AgentRegistry`
            populated at startup.
        settings: The application :class:`~src.config.settings.AppSettings`
            singleton (reserved for future configuration; currently unused).
    """
    global _last_processed_ts  # noqa: PLW0603

    logger.info("dev_lead_listener_job: checking for missed [dev lead] messages …")

    since_ts: float = _last_processed_ts if _last_processed_ts is not None else (time.time() - 300.0)
    since_ts_str: str = f"{since_ts:.6f}"

    # ------------------------------------------------------------------
    # Resolve agent from registry
    # ------------------------------------------------------------------
    try:
        dev_lead_agent = registry["dev_lead"]
    except KeyError:
        logger.error(
            "dev_lead_listener_job: 'dev_lead' agent not found in registry — " "available ids: %s. Skipping run.",
            registry.agent_ids(),
        )
        return

    # ------------------------------------------------------------------
    # Build short-lived Dev Lead Crew for the fallback scan
    # ------------------------------------------------------------------
    task = Task(
        description=_DEV_LEAD_FALLBACK_TASK_DESCRIPTION_TEMPLATE.format(
            since_ts=since_ts_str,
        ),
        expected_output=_DEV_LEAD_FALLBACK_TASK_EXPECTED_OUTPUT,
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
            "dev_lead_listener_job: completed. Result preview: %.300s",
            str(result),
        )
        # Advance the watermark so the next run won't re-process these messages.
        _last_processed_ts = time.time()

    except Exception:  # noqa: BLE001
        logger.exception("dev_lead_listener_job: crew raised an exception.")
