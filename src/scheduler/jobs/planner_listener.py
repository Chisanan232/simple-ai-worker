"""
Planner listener fallback job (Phase 6 + Phase 10 — Idea-Discussion Workflow).

Provides :func:`planner_listener_job`, an interval-based APScheduler job that
acts as the **fallback / batch-polling path** for the Planner Agent.

Under normal operation the Slack Bolt Socket Mode handler
(:mod:`src.slack_app.handlers.planner`) processes ``@ai-worker [planner]``
messages in real-time.  This job exists to catch any messages that were
missed during a Bolt downtime window.  It:

1. Builds a short-lived Crew with the Planner Agent.
2. Asks it to check ``slack/get_messages`` for unprocessed ``[planner]``
   messages since the last processed timestamp.
3. For each new message, applies the same Type A / Type B / Type C logic
   as the Bolt handler (idea survey, conclusion, or actionable requirement).
4. Updates ``_last_processed_ts`` so already-handled messages are not
   re-processed on the next run.

Usage (injected by :class:`~src.scheduler.runner.SchedulerRunner`)::

    from src.scheduler.jobs.planner_listener import planner_listener_job

    scheduler.add_job(
        func=planner_listener_job,
        kwargs={{"registry": registry, "settings": settings}},
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

2. If no new '[planner]' messages are found, do nothing and report
   "No new planner messages found since {since_ts}."

3. For each new '[planner]' message found, determine which request type
   applies and execute the corresponding steps:

   ════════════════════════════════════════════════════════════════
   **Request Type A — Actionable Requirement**
   ════════════════════════════════════════════════════════════════
   Applies when the message contains a clear, scoped product requirement
   ready to be turned into a development ticket.
   Steps:
   a. Strip the '@ai-worker [planner]' prefix from the message body.
   b. If clear and actionable: create a JIRA epic via jira/create_issue and
      a matching ClickUp task via clickup/create_task.
   c. Reply in the Slack thread via slack/reply_to_thread with a summary and
      the created ticket keys/IDs, OR clarifying questions if needed.

   ════════════════════════════════════════════════════════════════
   **Request Type B — Idea Survey & Discussion**
   ════════════════════════════════════════════════════════════════
   Applies when the message describes an exploratory product idea not yet
   ready for a development ticket.
   Steps:
   a. Use slack/get_messages to read the full thread for that message.
   b. Assess the discussion stage (initial / mid-discussion / survey ready).
   c. Respond with survey questions or the full 8-dimension Idea Survey Plan:
      ### 📋 Idea Survey Plan
      1. Marketing Value  2. Market Scope  3. Business Model
      4. Target Audience  5. Customer Pain Points Resolved
      6. MVP Features     7. Quick Implementation Path  8. Budget Estimation
   d. GUARDRAIL (BR-11): Do NOT create any tickets during Type B mode.

   ════════════════════════════════════════════════════════════════
   **Request Type C — Idea Discussion Conclusion**
   ════════════════════════════════════════════════════════════════
   Applies when the human's message contains a final accept or reject decision.
   Accept signals: "let's do it", "approved", "proceed", "go ahead", "LGTM".
   Reject signals: "rejected", "not now", "cancel", "drop it", "too risky".

   REJECT path:
   a. Read the full thread via slack/get_messages.
   b. Post conclusion message in the thread via slack/reply_to_thread.
   c. Create 1 JIRA issue + 1 ClickUp task with status "REJECTED" containing
      the full discussion summary.
   d. Reply with ticket URLs.
   e. GUARDRAIL (BR-12): Do NOT mention or notify the Dev Lead.

   ACCEPT path:
   a. Read the full thread via slack/get_messages.
   b. Post conclusion message in the thread via slack/reply_to_thread.
   c. Create JIRA issue(s) + ClickUp task(s) with status "OPEN" containing
      the full survey plan and discussion details.
   d. Reply with ticket URLs.
   e. Send a NEW message via slack/send_message (BR-13):
      "[dev lead] A new product idea has been accepted: <idea name>.
       Tickets: <ticket URLs>. Please start the development planning."
   f. GUARDRAIL (BR-1): Never set any ticket to "ACCEPTED" status.
      GUARDRAIL (BR-14): REJECTED tickets must use status "REJECTED".

4. After processing all messages, report a summary of every action taken.

Important: Do NOT re-process messages with timestamps at or before {since_ts}.
"""

_PLANNER_FALLBACK_TASK_EXPECTED_OUTPUT: str = (
    "A summary of actions taken for each missed [planner] message: "
    "'No new planner messages found' if none were missed, OR "
    "for each message processed: the request type detected (A/B/C) and "
    "the actions taken (survey posted / tickets created / Dev Lead notified)."
)


def planner_listener_job(
    registry: "AgentRegistry",
    settings: "AppSettings",  # noqa: ARG001  (reserved for future use)
) -> None:
    """APScheduler fallback job: poll Slack for missed ``[planner]`` messages.

    This job is the safety net for the Slack Bolt Socket Mode planner handler.
    Under normal operation it will find no new messages to process.

    Handles the same three request types as the Bolt handler:
    - **Type A**: Actionable requirement → create JIRA epic + ClickUp task.
    - **Type B**: Idea survey → multi-turn discussion + Markdown plan.
    - **Type C**: Conclusion → REJECTED or OPEN ticket + Dev Lead hand-off.

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
            "planner_listener_job: 'planner' agent not found in registry — available ids: %s. Skipping run.",
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
