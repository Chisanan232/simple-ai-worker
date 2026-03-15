"""
PR auto-merge watcher scheduler job (Phase 8c).

Provides :func:`pr_merge_watcher_job`, an APScheduler interval job that
watches open PRs registered by :func:`~src.scheduler.jobs.scan_tickets._execute_ticket`
and auto-merges them when:

1. The PR has been open for at least ``PR_AUTO_MERGE_TIMEOUT_SECONDS`` seconds.
2. The PR has **at least one approving review** (BR-2: never merge without approval).

After a successful merge (or when the PR was already merged by the human before
the watcher fires), the ticket is transitioned to ``MARK_COMPLETE`` via the
:class:`~src.ticket.workflow.WorkflowConfig` abstraction, and a Slack
notification is sent.

Business Rules Enforced
-----------------------
- **BR-2:** Auto-merge is only triggered when ``approval_count >= 1``.
  Zero approvals → log "no approvals, skipping" and leave the entry in
  ``_open_prs``.
- The watcher **never self-approves** — it only calls ``merge_pull_request``,
  never ``approve_pull_request``.
- ``MARK_COMPLETE`` uses
  :meth:`~src.ticket.workflow.WorkflowConfig.status_for_write`, so a
  misconfigured ``human_only: true`` on that operation raises
  :exc:`PermissionError` before any API call.

Shared state
------------
``_open_prs`` (imported from ``scan_tickets``) is a dict of
``ticket_id → PRRecord`` populated by ``_execute_ticket``.  This module
reads and clears entries from that dict.

``_prs_under_review`` (imported from ``scan_tickets``) is also cleared here
after a merge, so ``pr_review_comment_handler_job`` stops watching merged PRs.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, List, Optional

from crewai import Task

from src.crew.builder import CrewBuilder
from src.scheduler.jobs.scan_tickets import _open_prs, _prs_under_review
from src.ticket.workflow import WorkflowConfig, WorkflowOperation

if TYPE_CHECKING:
    from src.agents.registry import AgentRegistry
    from src.config.settings import AppSettings

__all__: List[str] = ["pr_merge_watcher_job"]

logger = logging.getLogger(__name__)


_PR_STATUS_TASK_TEMPLATE: str = """
Check the status of a GitHub pull request and return a structured JSON summary.

Pull Request URL: {pr_url}

Steps:
1. Use github/get_pull_request to fetch the current PR state.
2. Check whether the PR has been merged (is_merged field).
3. Count the number of approving reviews (approval_count).
4. Return a JSON object with exactly these fields:
   {{
     "is_merged": true | false,
     "approval_count": <integer>,
     "pr_url": "{pr_url}"
   }}

Return ONLY the JSON object — no surrounding text.
"""

_PR_MERGE_TASK_TEMPLATE: str = """
Merge a GitHub pull request that has received the required approvals.

Pull Request URL: {pr_url}
Ticket ID:        {ticket_id}

Steps:
1. Use github/merge_pull_request to merge the PR at {pr_url}.
2. Use the appropriate ticket tool to transition ticket '{ticket_id}' to
   status '{complete_status}':
   - JIRA: jira/transition_issue
   - ClickUp: clickup/update_task with status='{complete_status}'
3. Post a Slack notification via slack/send_message:
   "✅ PR merged for ticket {ticket_id}: {pr_url}
    Ticket transitioned to '{complete_status}'."
4. Output 'MERGED: {pr_url}' on a line by itself to confirm success.

IMPORTANT:
  - Do NOT approve the PR — only merge it.
  - Do NOT create any new tickets.
"""

_MERGE_TASK_EXPECTED_OUTPUT: str = (
    "Confirmation that the PR was merged, the ticket was transitioned to the "
    "complete status, and a Slack notification was posted. "
    "Include 'MERGED: <pr_url>' on a separate line."
)


def _run_pr_status_check(
    pr_url: str,
    dev_agent: Any,
) -> Optional[dict]:
    """Run a crew task to fetch the current PR status.

    Args:
        pr_url:    The GitHub pull request URL.
        dev_agent: The CrewAI agent object.

    Returns:
        A dict with ``is_merged`` (bool) and ``approval_count`` (int), or
        ``None`` if the status could not be determined.
    """
    import json

    description = _PR_STATUS_TASK_TEMPLATE.format(pr_url=pr_url)
    expected_output = (
        "JSON with is_merged (bool) and approval_count (int): "
        '{"is_merged": false, "approval_count": 1, "pr_url": "..."}'
    )
    task = Task(description=description, expected_output=expected_output, agent=dev_agent)
    crew = CrewBuilder.build(agents=[dev_agent], tasks=[task], process="sequential")

    try:
        result = crew.kickoff()
        raw = str(result).strip()
        # Strip markdown code fence if present.
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(ln for ln in lines if not ln.strip().startswith("```"))
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        logger.exception("pr_merge_watcher: failed to fetch PR status for %s.", pr_url)
        return None


def _run_pr_merge(
    pr_url: str,
    ticket_id: str,
    complete_status: str,
    dev_agent: Any,
) -> bool:
    """Run a crew task to merge the PR and transition the ticket.

    Args:
        pr_url:          The GitHub pull request URL.
        ticket_id:       The ticket to transition after merge.
        complete_status: The configured status for ``MARK_COMPLETE``.
        dev_agent:       The CrewAI agent object.

    Returns:
        ``True`` if the merge task completed without raising an exception.
    """
    description = _PR_MERGE_TASK_TEMPLATE.format(
        pr_url=pr_url,
        ticket_id=ticket_id,
        complete_status=complete_status,
    )
    task = Task(
        description=description,
        expected_output=_MERGE_TASK_EXPECTED_OUTPUT,
        agent=dev_agent,
    )
    crew = CrewBuilder.build(agents=[dev_agent], tasks=[task], process="sequential")

    try:
        result = crew.kickoff()
        logger.info(
            "pr_merge_watcher: merge crew completed for %s. Preview: %.300s",
            pr_url,
            str(result),
        )
        return True
    except Exception:  # noqa: BLE001
        logger.exception("pr_merge_watcher: merge crew failed for PR %s.", pr_url)
        return False


def pr_merge_watcher_job(
    registry: "AgentRegistry",
    settings: "AppSettings",
    executor: ThreadPoolExecutor,
    workflow: Optional[WorkflowConfig] = None,
) -> None:
    """APScheduler job: auto-merge approved PRs after the configured timeout.

    Iterates over ``_open_prs`` (populated by ``_execute_ticket``) and for
    each entry:

    - Checks whether the timeout has elapsed.
    - Fetches current PR state (merged?, approval count).
    - If already merged → transitions ticket to ``MARK_COMPLETE``, clears entry.
    - If not merged + 0 approvals → skips (BR-2).
    - If not merged + ≥1 approvals + timeout elapsed → merges, transitions,
      clears entry.

    Args:
        registry: The shared :class:`~src.agents.registry.AgentRegistry`.
        settings: The :class:`~src.config.settings.AppSettings` (reads
            ``PR_AUTO_MERGE_TIMEOUT_SECONDS``).
        executor: The bounded ``ThreadPoolExecutor`` (not used directly here;
            all ops run synchronously inside the scheduler thread for simplicity).
        workflow: Optional :class:`~src.ticket.workflow.WorkflowConfig`.
            When ``None``, the watcher logs a warning and skips the run.
    """
    if not _open_prs:
        logger.debug("pr_merge_watcher_job: no open PRs to watch.")
        return

    if workflow is None:
        logger.warning(
            "pr_merge_watcher_job: no WorkflowConfig provided — skipping run. " "Pass workflow= to the job kwargs."
        )
        return

    try:
        dev_agent = registry["dev_agent"]
    except KeyError:
        logger.error("pr_merge_watcher_job: 'dev_agent' not found in registry — skipping run.")
        return

    timeout_seconds: int = getattr(settings, "PR_AUTO_MERGE_TIMEOUT_SECONDS", 300)

    # Resolve the complete status string via WorkflowConfig (raises PermissionError
    # if misconfigured as human_only — catches misconfigurations early).
    try:
        complete_status = workflow.status_for_write(WorkflowOperation.MARK_COMPLETE)
    except PermissionError:
        logger.error(
            "pr_merge_watcher_job: MARK_COMPLETE operation is marked human_only "
            "in WorkflowConfig — cannot auto-merge. Check agents.yaml."
        )
        return

    now = time.time()
    tickets_to_clear: List[str] = []

    for ticket_id, pr_record in list(_open_prs.items()):
        age_seconds = now - pr_record.opened_at_utc
        pr_url = pr_record.pr_url

        logger.debug(
            "pr_merge_watcher: checking PR for ticket %s (age=%.0fs, url=%s).",
            ticket_id,
            age_seconds,
            pr_url,
        )

        # Fetch current PR state.
        status = _run_pr_status_check(pr_url, dev_agent)
        if status is None:
            logger.warning("pr_merge_watcher: could not fetch status for PR %s — skipping.", pr_url)
            continue

        is_merged: bool = bool(status.get("is_merged", False))
        approval_count: int = int(status.get("approval_count", 0))

        if is_merged:
            # PR was already merged by a human — just transition the ticket.
            logger.info(
                "pr_merge_watcher: PR %s already merged — transitioning ticket %s to '%s'.",
                pr_url,
                ticket_id,
                complete_status,
            )
            _run_pr_merge(pr_url, ticket_id, complete_status, dev_agent)
            tickets_to_clear.append(ticket_id)
            continue

        if approval_count == 0:
            logger.info(
                "pr_merge_watcher: PR %s has 0 approvals — skipping (BR-2). " "Will re-check next interval.",
                pr_url,
            )
            continue

        # approval_count >= 1 and not yet merged.
        if age_seconds < timeout_seconds:
            logger.debug(
                "pr_merge_watcher: PR %s has %d approval(s) but timeout not yet elapsed " "(%.0fs < %ds). Waiting.",
                pr_url,
                approval_count,
                age_seconds,
                timeout_seconds,
            )
            continue

        # Timeout elapsed AND at least 1 approval → merge.
        logger.info(
            "pr_merge_watcher: PR %s has %d approval(s) and is %.0fs old — auto-merging.",
            pr_url,
            approval_count,
            age_seconds,
        )
        success = _run_pr_merge(pr_url, ticket_id, complete_status, dev_agent)
        if success:
            tickets_to_clear.append(ticket_id)

    # Clear entries for completed PRs.
    for ticket_id in tickets_to_clear:
        _open_prs.pop(ticket_id, None)
        _prs_under_review.pop(ticket_id, None)
        logger.info("pr_merge_watcher: cleared watcher entries for ticket %s.", ticket_id)
