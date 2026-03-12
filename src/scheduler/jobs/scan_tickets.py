"""
Scan-and-dispatch scheduler job (Phase 6 — Pull Model).

Provides :func:`scan_and_dispatch_job`, an APScheduler interval job that
implements the **pull model** for Dev Agents:

1. Build a short-lived "Scanner" Crew with the Dev Agent.
2. Ask it to find all JIRA / ClickUp tickets in ``Ready`` status with no
   open dependencies.
3. For each returned ticket, submit a fresh Dev Crew execution to the
   bounded :class:`~concurrent.futures.ThreadPoolExecutor`.
4. On success → transition ticket to "In Review", open a GitHub PR, and
   notify Slack.
5. On failure → log the error and notify Slack.

An in-memory *dispatch guard* (``set[str]``) prevents the same ticket from
being dispatched twice while it is still being processed.

Usage (injected by :class:`~src.scheduler.runner.SchedulerRunner`)::

    from src.scheduler.jobs.scan_tickets import scan_and_dispatch_job

    scheduler.add_job(
        func=scan_and_dispatch_job,
        kwargs={"registry": registry, "settings": settings, "executor": executor},
        trigger="interval",
        seconds=60,
        id="scan_and_dispatch",
    )
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, List, Set

from crewai import Task

from src.crew.builder import CrewBuilder

if TYPE_CHECKING:
    from src.agents.registry import AgentRegistry
    from src.config.settings import AppSettings

__all__: List[str] = ["scan_and_dispatch_job"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory dispatch guard: ticket IDs currently being processed.
# Cleared when each future completes (success or failure).
# ---------------------------------------------------------------------------
_in_progress_tickets: Set[str] = set()

_SCAN_TASK_DESCRIPTION: str = """
Use jira/search_issues with JQL 'status = Ready' to find all tickets that are
in 'Ready' status with no open sub-task dependencies.
Also use clickup/search_tasks to check for ClickUp tasks with status='ready'
and no unresolved dependencies.

Return a JSON array of objects, one per ready ticket, with these fields:
  - "id":     the ticket/task identifier (e.g. "PROJ-42" or a ClickUp task ID)
  - "source": either "jira" or "clickup"
  - "title":  the ticket title / summary
  - "url":    the ticket URL (if available)

If no ready tickets are found, return an empty JSON array: [].
"""

_SCAN_TASK_EXPECTED_OUTPUT: str = (
    "A JSON array of ready ticket objects, e.g.: "
    '[{"id": "PROJ-42", "source": "jira", "title": "Implement login", "url": "https://..."}]. '
    "Return an empty array if no tickets are ready."
)

_DEV_TASK_TEMPLATE: str = """
You have been assigned ticket {ticket_id} from {source}.

Title: {title}

Steps:
1. Fetch the full ticket details using {fetch_action}.
2. Self-assign the ticket and update its status to "In Progress" via the
   appropriate update/transition action.
3. Implement the changes described in the ticket (write or modify code as
   needed, following the acceptance criteria and any implementation notes in
   the ticket comments).
4. Open a GitHub Pull Request summarising your changes via github/create_pull_request.
5. Transition the ticket to "In Review" (jira/transition_issue or
   clickup/update_task with status="In Review").
6. Post a Slack notification with the PR link and ticket reference using
   slack/send_message.

Complete all steps in order and confirm what was done.
"""

_DEV_TASK_EXPECTED_OUTPUT: str = (
    "A structured summary: ticket self-assigned and moved to In Progress, "
    "implementation completed, GitHub PR opened (include PR URL), ticket "
    "transitioned to In Review, Slack notification posted."
)


def _execute_ticket(
    ticket_id: str,
    source: str,
    title: str,
    registry: "AgentRegistry",
) -> None:
    """Execute a single ticket inside a ThreadPoolExecutor worker.

    Builds a fresh, short-lived Dev Crew, runs ``kickoff()``, then cleans
    up the dispatch guard on completion or failure.

    Args:
        ticket_id: The ticket identifier (e.g. ``"PROJ-42"``).
        source:    ``"jira"`` or ``"clickup"``.
        title:     Human-readable ticket title for logging.
        registry:  The shared :class:`~src.agents.registry.AgentRegistry`.
    """
    fetch_action = "jira/get_issue" if source == "jira" else "clickup/get_task"

    task_description = _DEV_TASK_TEMPLATE.format(
        ticket_id=ticket_id,
        source=source,
        title=title,
        fetch_action=fetch_action,
    )

    try:
        dev_agent = registry["dev_agent"]
        task = Task(
            description=task_description,
            expected_output=_DEV_TASK_EXPECTED_OUTPUT,
            agent=dev_agent,
        )
        crew = CrewBuilder.build(
            agents=[dev_agent],
            tasks=[task],
            process="sequential",
        )

        logger.info("Dev Crew starting for ticket %s (%s).", ticket_id, source)
        result = crew.kickoff()
        logger.info(
            "Dev Crew completed for ticket %s. Result preview: %.200s",
            ticket_id,
            str(result),
        )

    except Exception:  # noqa: BLE001
        logger.exception("Dev Crew failed for ticket %s (%s).", ticket_id, source)

    finally:
        _in_progress_tickets.discard(ticket_id)
        logger.debug("Dispatch guard released for ticket %s.", ticket_id)


def scan_and_dispatch_job(
    registry: "AgentRegistry",
    settings: "AppSettings",
    executor: ThreadPoolExecutor,
) -> None:
    """APScheduler job: scan for ready tickets and dispatch Dev Agent crews.

    This is the pull-model entry point.  It is registered by
    :class:`~src.scheduler.runner.SchedulerRunner` and called on the
    configured interval.

    Args:
        registry: The shared :class:`~src.agents.registry.AgentRegistry`
            populated at startup.
        settings: The application :class:`~src.config.settings.AppSettings`
            singleton (currently used for guard logging; reserved for future
            per-job interval overrides).
        executor: The bounded :class:`~concurrent.futures.ThreadPoolExecutor`
            shared across all dev-agent dispatches.  Pool size is
            ``settings.MAX_CONCURRENT_DEV_AGENTS``.
    """
    logger.info("scan_and_dispatch_job: scanning for ready tickets …")

    # ------------------------------------------------------------------
    # Step 1 — Scanner Crew: find all ready tickets
    # ------------------------------------------------------------------
    try:
        dev_agent = registry["dev_agent"]
    except KeyError:
        logger.error(
            "scan_and_dispatch_job: 'dev_agent' not found in registry — available ids: %s. Skipping run.",
            registry.agent_ids(),
        )
        return

    scan_task = Task(
        description=_SCAN_TASK_DESCRIPTION,
        expected_output=_SCAN_TASK_EXPECTED_OUTPUT,
        agent=dev_agent,
    )
    scanner_crew = CrewBuilder.build(
        agents=[dev_agent],
        tasks=[scan_task],
        process="sequential",
    )

    try:
        scan_result = scanner_crew.kickoff()
    except Exception:  # noqa: BLE001
        logger.exception("scan_and_dispatch_job: scanner crew raised an exception.")
        return

    # ------------------------------------------------------------------
    # Step 2 — Parse scan result
    # ------------------------------------------------------------------
    import json  # local import keeps module-level imports clean

    raw_output: str = str(scan_result).strip()

    # The LLM may wrap the JSON in a markdown code block — strip it.
    if raw_output.startswith("```"):
        lines = raw_output.splitlines()
        raw_output = "\n".join(line for line in lines if not line.strip().startswith("```"))

    try:
        tickets: list[dict] = json.loads(raw_output)
    except json.JSONDecodeError:
        logger.warning(
            "scan_and_dispatch_job: could not parse scanner output as JSON. Raw output (first 500 chars): %.500s",
            raw_output,
        )
        return

    if not tickets:
        logger.info("scan_and_dispatch_job: no ready tickets found.")
        return

    logger.info("scan_and_dispatch_job: found %d ready ticket(s).", len(tickets))

    # ------------------------------------------------------------------
    # Step 3 — Dispatch one Dev Crew per ticket (bounded pool)
    # ------------------------------------------------------------------
    for ticket in tickets:
        ticket_id: str = str(ticket.get("id", ""))
        source: str = str(ticket.get("source", "jira"))
        title: str = str(ticket.get("title", ticket_id))

        if not ticket_id:
            logger.warning("scan_and_dispatch_job: ticket entry missing 'id' field — skipped: %s", ticket)
            continue

        if ticket_id in _in_progress_tickets:
            logger.debug(
                "scan_and_dispatch_job: ticket %s already in progress — skipped.",
                ticket_id,
            )
            continue

        _in_progress_tickets.add(ticket_id)
        logger.info(
            "scan_and_dispatch_job: dispatching ticket %s (%s) to executor.",
            ticket_id,
            source,
        )

        future: Future[None] = executor.submit(
            _execute_ticket,
            ticket_id,
            source,
            title,
            registry,
        )

        # Attach a done-callback for structured error logging.
        def _on_done(fut: Future[None], tid: str = ticket_id) -> None:
            exc = fut.exception()
            if exc:
                logger.error(
                    "scan_and_dispatch_job: executor future for ticket %s raised: %s",
                    tid,
                    exc,
                )

        future.add_done_callback(_on_done)
