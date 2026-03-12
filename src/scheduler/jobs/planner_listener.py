"""
Planner listener fallback job (Phase 6 — APScheduler polling).

Provides :func:`planner_listener_job`, an interval-based APScheduler job that
acts as the **fallback / batch-polling path** for the Planner Agent.

Under normal operation the Slack Bolt Socket Mode handler
(:mod:`src.slack_app.handlers.planner`) processes ``@ai-worker [planner]``
messages in real-time.  This job exists to catch any messages that were
missed during a Bolt downtime window.  It:

1. Builds a short-lived Crew with the Planner Agent.
2. Asks it to check ``slack/get_messages`` for unprocessed ``[planner]``
   messages since the last processed timestamp.
3. For each new request, creates a JIRA epic / ClickUp task and replies in
   the Slack thread.
4. Updates ``_last_processed_ts`` so already-handled messages are not
   re-processed on the next run.

Usage (injected by :class:`~src.scheduler.runner.SchedulerRunner`)::

    from src.scheduler.jobs.planner_listener import planner_listener_job

    scheduler.add_job(
        func=planner_listener_job,
        kwargs={"registry": registry, "settings": settings},
        trigger="interval",
        seconds=300,
        id="planner_listener",
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

__all__: List[str] = ["planner_listener_job"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State: tracks the most recent Slack message timestamp that has been
# processed by this fallback job.  Stored as a module-level variable so it
# persists across job runs within the same process.
# ---------------------------------------------------------------------------
_last_processed_ts: Optional[float] = None

_PLANNER_FALLBACK_TASK_DESCRIPTION_TEMPLATE: str = """
You are the Planner Agent operating in **fallback batch-polling mode**.

The real-time Slack Bolt handler may have been temporarily unavailable.
Your job is to check for any unprocessed messages that were missed.

Steps:
1. Use slack/get_messages to fetch recent messages from the monitored Slack
   channel. Look for messages posted after timestamp {since_ts} that mention
   @ai-worker and contain the tag '[planner]' in the message text.

2. For each such unprocessed message:
   a. Strip the '@ai-worker [planner]' prefix from the message body to get
      the clean product requirement text.
   b. Understand the requirement — if it is ambiguous ask a clarifying
      question in the thread; if it is clear enough:
      - Create a JIRA epic via jira/create_issue.
      - Create a matching ClickUp task via clickup/create_task.
   c. Reply in the Slack thread via slack/reply_to_thread with a summary of
      what was created or what clarification is needed.

3. If no new '[planner]' messages are found, do nothing and report
   "No new planner messages found since {since_ts}."

Important: Do NOT re-process messages with timestamps at or before {since_ts}.
"""

_PLANNER_FALLBACK_TASK_EXPECTED_OUTPUT: str = (
    "A summary of actions taken: either 'No new planner messages found' or "
    "a list of messages processed with the JIRA epic keys and ClickUp task IDs created."
)


def planner_listener_job(
    registry: "AgentRegistry",
    settings: "AppSettings",  # noqa: ARG001  (reserved for future use)
) -> None:
    """APScheduler fallback job: poll Slack for missed ``[planner]`` messages.

    This job is the safety net for the Slack Bolt Socket Mode planner handler.
    Under normal operation it will find no new messages to process.

    Args:
        registry: The shared :class:`~src.agents.registry.AgentRegistry`
            populated at startup.
        settings: The application :class:`~src.config.settings.AppSettings`
            singleton (reserved for future configuration; currently unused).
    """
    global _last_processed_ts  # noqa: PLW0603

    logger.info("planner_listener_job: checking for missed [planner] messages …")

    since_ts: float = _last_processed_ts if _last_processed_ts is not None else (time.time() - 300.0)
    since_ts_str: str = f"{since_ts:.6f}"

    # ------------------------------------------------------------------
    # Resolve agent from registry
    # ------------------------------------------------------------------
    try:
        planner_agent = registry["planner"]
    except KeyError:
        logger.error(
            "planner_listener_job: 'planner' agent not found in registry — "
            "available ids: %s. Skipping run.",
            registry.agent_ids(),
        )
        return

    # ------------------------------------------------------------------
    # Build short-lived Planner Crew for the fallback scan
    # ------------------------------------------------------------------
    task = Task(
        description=_PLANNER_FALLBACK_TASK_DESCRIPTION_TEMPLATE.format(
            since_ts=since_ts_str,
        ),
        expected_output=_PLANNER_FALLBACK_TASK_EXPECTED_OUTPUT,
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
            "planner_listener_job: completed. Result preview: %.300s",
            str(result),
        )
        # Advance the watermark so the next run won't re-process these messages.
        _last_processed_ts = time.time()

    except Exception:  # noqa: BLE001
        logger.exception("planner_listener_job: crew raised an exception.")

