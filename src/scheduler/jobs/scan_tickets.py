"""
Scan-and-dispatch scheduler job (Phase 8b — WorkflowConfig integration).

Provides :func:`scan_and_dispatch_job`, an APScheduler interval job that
implements the **pull model** for Dev Agents:

1. Build a short-lived "Scanner" Crew with the Dev Agent.
2. Ask it to find all JIRA / ClickUp tickets in the ``SCAN_FOR_WORK``-configured
   status (excluding ``SKIP_REJECTED``-configured status).
3. For each returned ticket, apply the Python-level ``SKIP_REJECTED`` guard,
   then submit a fresh Dev Crew execution to the bounded
   :class:`~concurrent.futures.ThreadPoolExecutor`.
4. On success → the crew opens a GitHub PR, transitions to ``OPEN_FOR_REVIEW``
   status, and notifies Slack.  The PR is registered in the shared
   ``_open_prs`` and ``_prs_under_review`` dicts for the watcher jobs.
5. On failure → log the error and notify Slack.

An in-memory *dispatch guard* (``set[str]``) prevents the same ticket from
being dispatched twice while it is still being processed.

Business Rules Enforced
-----------------------
- **BR-1:** The scan query targets the ``SCAN_FOR_WORK`` operation status
  (human-only).  The task description explicitly forbids the LLM from writing
  that status.
- **BR-3:** Tickets in ``SKIP_REJECTED`` status are excluded from the query
  *and* guarded in the Python dispatch loop.
- **GAP-7 mitigation:** The dev task description instructs the agent to
  transition to ``START_DEVELOPMENT`` as its **very first action**, so a
  restarted process sees the in-progress status (not the scan_for_work status)
  and skips it.
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Dict, List, Optional, Set

from crewai import Task

from src.crew.builder import CrewBuilder
from src.ticket.models import PRRecord
from src.ticket.workflow import WorkflowConfig, WorkflowOperation

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

# ---------------------------------------------------------------------------
# Shared PR tracking dicts — written after _execute_ticket opens a PR.
# Read by pr_merge_watcher_job and pr_review_comment_handler_job.
# ---------------------------------------------------------------------------
_open_prs: Dict[str, PRRecord] = {}       # ticket_id → PRRecord
_prs_under_review: Dict[str, str] = {}    # ticket_id → pr_url


def _build_scan_task_description(workflow: WorkflowConfig) -> str:
    """Build the scanner task description with operation-resolved status strings."""
    scan_status = workflow.status_for(WorkflowOperation.SCAN_FOR_WORK)
    skip_status = workflow.status_for(WorkflowOperation.SKIP_REJECTED)
    return (
        f"Use jira/search_issues with JQL "
        f"'status = \"{scan_status}\" AND status != \"{skip_status}\"' "
        f"to find all tickets that are in '{scan_status}' status "
        f"with no open sub-task dependencies.\n"
        f"Also use clickup/search_tasks to check for ClickUp tasks with "
        f"status='{scan_status}' (excluding '{skip_status}') "
        f"and no unresolved dependencies.\n\n"
        "Return a JSON array of objects, one per ready ticket, with these fields:\n"
        '  - "id":     the ticket/task identifier (e.g. "PROJ-42")\n'
        '  - "source": either "jira" or "clickup"\n'
        '  - "title":  the ticket title / summary\n'
        '  - "url":    the ticket URL (if available)\n'
        '  - "status": the current status string of the ticket\n\n'
        "If no ready tickets are found, return an empty JSON array: [].\n\n"
        "IMPORTANT:\n"
        f"  - Do NOT set any ticket to '{scan_status}' — that status is "
        "human-only (BR-1). Only humans may set a ticket to that state.\n"
        f"  - Do NOT process or modify tickets in '{skip_status}' status — "
        "they are silently skipped (BR-3)."
    )


_SCAN_TASK_EXPECTED_OUTPUT: str = (
    "A JSON array of ready ticket objects, e.g.: "
    '[{"id": "PROJ-42", "source": "jira", "title": "Implement login", '
    '"url": "https://...", "status": "ACCEPTED"}]. '
    "Return an empty array if no tickets are ready."
)


def _build_dev_task_description(
    ticket_id: str,
    source: str,
    title: str,
    workflow: WorkflowConfig,
) -> str:
    """Build the dev task description with WorkflowConfig-resolved status strings."""
    fetch_action = "jira/get_issue" if source == "jira" else "clickup/get_task"
    start_status = workflow.status_for_write(WorkflowOperation.START_DEVELOPMENT)
    review_status = workflow.status_for_write(WorkflowOperation.OPEN_FOR_REVIEW)
    scan_status = workflow.status_for(WorkflowOperation.SCAN_FOR_WORK)

    return (
        f"You have been assigned ticket {ticket_id} from {source}.\n\n"
        f"Title: {title}\n\n"
        "Steps — execute in order:\n"
        f"1. Fetch full ticket details: {fetch_action} for '{ticket_id}'.\n"
        f"2. IMMEDIATELY transition the ticket to '{start_status}' as your VERY FIRST "
        "action (before writing any code). This prevents double-pickup if the "
        "process restarts.\n"
        "3. Self-assign the ticket.\n"
        "4. Implement all changes described in the ticket: write or modify code as "
        "needed, following acceptance criteria and implementation notes in ticket "
        "comments.\n"
        "5. Open a GitHub Pull Request summarising your changes via "
        "github/create_pull_request. The PR description must include the ticket ID "
        f"'{ticket_id}' and a link to the ticket.\n"
        f"6. Transition the ticket to '{review_status}' using the appropriate "
        "transition/update action.\n"
        "7. Post a Slack notification with the PR URL and ticket reference using "
        "slack/send_message.\n"
        "8. Output the PR URL on a line by itself prefixed with 'PR_URL:' so it can "
        "be recorded (e.g. 'PR_URL: https://github.com/org/repo/pull/42').\n\n"
        "IMPORTANT:\n"
        f"  - Do NOT set any ticket to '{scan_status}' — that status is "
        "human-only (BR-1).\n"
        f"  - Transition to '{start_status}' as your very first action (step 2).\n"
        "  - Complete all steps in order and confirm what was done."
    )


_DEV_TASK_EXPECTED_OUTPUT: str = (
    "A structured summary: ticket fetched, transitioned to start-development status, "
    "self-assigned, implementation completed, GitHub PR opened (include PR URL on a "
    "line prefixed with 'PR_URL:'), ticket transitioned to open-for-review status, "
    "Slack notification posted."
)


def _extract_pr_url(result_text: str) -> Optional[str]:
    """Extract the PR URL from the crew result text.

    Looks for a line containing ``PR_URL:`` (case-insensitive) followed by a URL.

    Args:
        result_text: The raw string output from ``crew.kickoff()``.

    Returns:
        The extracted URL string, or ``None`` if not found.
    """
    match = re.search(r"PR_URL:\s*(https?://\S+)", result_text, re.IGNORECASE)
    if match:
        return match.group(1).rstrip(".,;)")
    return None


def _execute_ticket(
    ticket_id: str,
    source: str,
    title: str,
    raw_status: str,
    registry: "AgentRegistry",
    workflow: WorkflowConfig,
) -> None:
    """Execute a single ticket inside a ThreadPoolExecutor worker.

    Builds a fresh, short-lived Dev Crew, runs ``kickoff()``, extracts the
    PR URL from the result, registers the PR in the shared watcher dicts,
    then cleans up the dispatch guard.

    Args:
        ticket_id:  The ticket identifier (e.g. ``"PROJ-42"``).
        source:     ``"jira"`` or ``"clickup"``.
        title:      Human-readable ticket title for logging.
        raw_status: The raw status string from the scan result.
        registry:   The shared :class:`~src.agents.registry.AgentRegistry`.
        workflow:   The :class:`~src.ticket.workflow.WorkflowConfig` instance.
    """
    # Belt-and-suspenders BR-3 guard inside the worker thread.
    if workflow.matches(WorkflowOperation.SKIP_REJECTED, raw_status):
        logger.info(
            "_execute_ticket: ticket %s is in '%s' (SKIP_REJECTED) — skipping (BR-3).",
            ticket_id,
            raw_status,
        )
        _in_progress_tickets.discard(ticket_id)
        return

    task_description = _build_dev_task_description(ticket_id, source, title, workflow)

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
        result_text = str(result).strip()
        logger.info(
            "Dev Crew completed for ticket %s. Result preview: %.300s",
            ticket_id,
            result_text,
        )

        # Register the PR in the shared watcher dicts.
        pr_url = _extract_pr_url(result_text)
        if pr_url:
            pr_record = PRRecord(
                ticket_id=ticket_id,
                pr_url=pr_url,
                opened_at_utc=time.time(),
            )
            _open_prs[ticket_id] = pr_record
            _prs_under_review[ticket_id] = pr_url
            logger.info(
                "_execute_ticket: PR registered for ticket %s: %s",
                ticket_id,
                pr_url,
            )
        else:
            logger.warning(
                "_execute_ticket: no PR_URL found in crew output for ticket %s "
                "(cannot register in watcher dicts).",
                ticket_id,
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
    workflow: Optional[WorkflowConfig] = None,
) -> None:
    """APScheduler job: scan for ready tickets and dispatch Dev Agent crews.

    This is the pull-model entry point.  It is registered by
    :class:`~src.scheduler.runner.SchedulerRunner` and called on the
    configured interval.

    Args:
        registry: The shared :class:`~src.agents.registry.AgentRegistry`
            populated at startup.
        settings: The application :class:`~src.config.settings.AppSettings`
            singleton.
        executor: The bounded :class:`~concurrent.futures.ThreadPoolExecutor`
            shared across all dev-agent dispatches.
        workflow: Optional :class:`~src.ticket.workflow.WorkflowConfig`
            instance.  When ``None``, a permissive fallback is constructed
            using legacy status strings with a warning.
    """
    logger.info("scan_and_dispatch_job: scanning for ready tickets …")

    # ------------------------------------------------------------------
    # Step 0 — Resolve WorkflowConfig
    # ------------------------------------------------------------------
    if workflow is None:
        logger.warning(
            "scan_and_dispatch_job: no WorkflowConfig provided — using legacy "
            "fallback (status='Ready').  Add a 'workflow:' block to agents.yaml."
        )
        workflow = WorkflowConfig(
            {
                "scan_for_work": {"status_value": "Ready", "human_only": True},
                "skip_rejected": {"status_value": "Rejected"},
                "start_development": {"status_value": "In Progress"},
                "open_for_review": {"status_value": "In Review"},
                "mark_complete": {"status_value": "Complete"},
                "update_with_context": {"status_value": ""},
            }
        )

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

    scan_description = _build_scan_task_description(workflow)
    scan_task = Task(
        description=scan_description,
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
    raw_output: str = str(scan_result).strip()

    # The LLM may wrap the JSON in a markdown code block — strip it.
    if raw_output.startswith("```"):
        lines = raw_output.splitlines()
        raw_output = "\n".join(line for line in lines if not line.strip().startswith("```"))

    try:
        tickets: list[dict] = json.loads(raw_output)
    except json.JSONDecodeError:
        logger.warning(
            "scan_and_dispatch_job: could not parse scanner output as JSON. "
            "Raw output (first 500 chars): %.500s",
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
        raw_status: str = str(ticket.get("status", ""))

        if not ticket_id:
            logger.warning(
                "scan_and_dispatch_job: ticket entry missing 'id' field — skipped: %s",
                ticket,
            )
            continue

        # BR-3 guard: skip REJECTED tickets in the dispatch loop.
        if workflow.matches(WorkflowOperation.SKIP_REJECTED, raw_status):
            logger.info(
                "scan_and_dispatch_job: ticket %s is in '%s' (SKIP_REJECTED) — skipping (BR-3).",
                ticket_id,
                raw_status,
            )
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
            raw_status,
            registry,
            workflow,
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
