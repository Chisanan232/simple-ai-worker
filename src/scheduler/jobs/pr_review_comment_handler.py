"""
PR review comment handler scheduler job (Phase 8d).

Provides :func:`pr_review_comment_handler_job`, an APScheduler interval job
that detects pull requests with ``CHANGES_REQUESTED`` review status or
unresolved inline comments, dispatches a fix crew to implement the requested
changes, and replies to the reviewer's comments.

Business Rules Enforced
-----------------------
- The Dev Agent **never self-approves** a PR.  The task description explicitly
  states ``"Do NOT approve the PR"`` and ``"Do NOT merge the PR"``.
- **Deduplication guard:** ``_in_progress_comment_fixes`` prevents a second
  fix from being dispatched for the same ticket while a fix is already running.

Shared state
------------
``_prs_under_review`` (imported from ``scan_tickets``) is a dict of
``ticket_id → pr_url`` populated by ``_execute_ticket``.  This module reads
entries from that dict.

``_in_progress_comment_fixes`` is a module-level set that tracks which
ticket IDs currently have a fix crew running.  It is automatically populated
before ``executor.submit()`` and cleared in the ``finally`` block of
``_fix_review_comments``.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, List, Optional, Set

from crewai import Task

from src.crew.builder import CrewBuilder
from src.scheduler.jobs.scan_tickets import _prs_under_review

if TYPE_CHECKING:
    from src.agents.registry import AgentRegistry
    from src.config.settings import AppSettings

__all__: List[str] = ["pr_review_comment_handler_job"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deduplication guard: ticket IDs that currently have a fix crew running.
# ---------------------------------------------------------------------------
_in_progress_comment_fixes: Set[str] = set()


_PR_REVIEW_STATUS_TASK_TEMPLATE: str = """
Check whether a GitHub pull request has actionable review feedback.

Pull Request URL: {pr_url}

Steps:
1. Use github/get_pull_request_reviews to fetch all reviews.
2. Use github/get_pull_request_comments to fetch all inline comments.
3. Return a JSON object:
   {{
     "has_changes_requested": true | false,
     "unresolved_comment_count": <integer>,
     "comments": [
       {{"id": "<comment_id>", "body": "<comment_text>", "path": "<file_path>", "line": <line_number>}}
     ]
   }}

Return ONLY the JSON object — no surrounding text or markdown.
"""

_PR_FIX_TASK_TEMPLATE: str = """
A pull request reviewer has requested changes. Implement all the requested fixes.

Pull Request URL: {pr_url}
Ticket ID:        {ticket_id}

Review comments to address:
{comments_text}

Steps:
1. Read each review comment carefully.
2. Implement the requested changes in the codebase (write/modify files as needed).
3. Commit and push the changes to the same PR branch.
4. For each comment you addressed, reply via github/reply_to_review_comment:
   "Addressed: <brief description of what was changed>"
5. Output 'FIXES_APPLIED: {ticket_id}' on a line by itself to confirm.

CRITICAL RULES:
  - Do NOT approve the PR under any circumstances.
  - Do NOT merge the PR.
  - Do NOT create new tickets.
  - Only implement the changes requested in the comments above.
"""

_FIX_TASK_EXPECTED_OUTPUT: str = (
    "Confirmation that all requested changes were implemented, committed, and pushed. "
    "Each addressed review comment was replied to. "
    "Include 'FIXES_APPLIED: <ticket_id>' on a separate line. "
    "The PR was NOT approved and NOT merged."
)


def _check_pr_review_status(
    pr_url: str,
    dev_agent: Any,
) -> Optional[dict]:
    """Fetch the review status for a pull request.

    Args:
        pr_url:    The GitHub pull request URL.
        dev_agent: The CrewAI agent object.

    Returns:
        A dict with ``has_changes_requested`` (bool),
        ``unresolved_comment_count`` (int), and ``comments`` (list),
        or ``None`` on error.
    """
    import json

    description = _PR_REVIEW_STATUS_TASK_TEMPLATE.format(pr_url=pr_url)
    expected_output = 'JSON: {"has_changes_requested": false, "unresolved_comment_count": 0, "comments": []}'
    task = Task(description=description, expected_output=expected_output, agent=dev_agent)
    crew = CrewBuilder.build(agents=[dev_agent], tasks=[task], process="sequential")

    try:
        result = crew.kickoff()
        raw = str(result).strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(ln for ln in lines if not ln.strip().startswith("```"))
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        logger.exception("pr_review_comment_handler: failed to fetch review status for %s.", pr_url)
        return None


def _fix_review_comments(
    ticket_id: str,
    pr_url: str,
    comments: List[dict],
    dev_agent: Any,
) -> None:
    """Implement fixes for PR review comments in a background thread.

    Dispatched via the ThreadPoolExecutor.  Clears the deduplication guard
    in the ``finally`` block so subsequent fix cycles can fire.

    Args:
        ticket_id:  The ticket this PR belongs to.
        pr_url:     The GitHub pull request URL.
        comments:   List of comment dicts (id, body, path, line).
        dev_agent:  The CrewAI agent object.
    """
    try:
        if not comments:
            logger.info("pr_review_comment_handler: no comments to fix for ticket %s.", ticket_id)
            return

        # Format the comment list for the task description.
        comments_text = "\n".join(
            f"  [{i+1}] File: {c.get('path', 'unknown')}:{c.get('line', '?')} — " f"{c.get('body', '(no body)')}"
            for i, c in enumerate(comments)
        )

        description = _PR_FIX_TASK_TEMPLATE.format(
            pr_url=pr_url,
            ticket_id=ticket_id,
            comments_text=comments_text,
        )
        task = Task(
            description=description,
            expected_output=_FIX_TASK_EXPECTED_OUTPUT,
            agent=dev_agent,
        )
        crew = CrewBuilder.build(
            agents=[dev_agent],
            tasks=[task],
            process="sequential",
        )

        logger.info(
            "pr_review_comment_handler: fix crew starting for ticket %s (%d comments).",
            ticket_id,
            len(comments),
        )
        result = crew.kickoff()
        logger.info(
            "pr_review_comment_handler: fix crew completed for ticket %s. Preview: %.300s",
            ticket_id,
            str(result),
        )

    except Exception:  # noqa: BLE001
        logger.exception("pr_review_comment_handler: fix crew failed for ticket %s.", ticket_id)

    finally:
        _in_progress_comment_fixes.discard(ticket_id)
        logger.debug("pr_review_comment_handler: fix guard cleared for ticket %s.", ticket_id)


def pr_review_comment_handler_job(
    registry: "AgentRegistry",
    settings: "AppSettings",
    executor: ThreadPoolExecutor,
) -> None:
    """APScheduler job: detect PR review comments and dispatch fix crews.

    For each PR in ``_prs_under_review``:

    1. Skip if a fix is already running (deduplication guard).
    2. Fetch review status (``CHANGES_REQUESTED`` or unresolved comments?).
    3. If actionable → add to dedup guard → submit fix crew to executor.

    The fix crew implements the requested changes, pushes commits to the
    existing branch, and replies to each addressed comment.  It **never**
    approves or merges the PR.

    Args:
        registry: The shared :class:`~src.agents.registry.AgentRegistry`.
        settings: The :class:`~src.config.settings.AppSettings` singleton.
        executor: The bounded ``ThreadPoolExecutor`` for fix crew dispatch.
    """
    if not _prs_under_review:
        logger.debug("pr_review_comment_handler_job: no PRs under review to check.")
        return

    try:
        dev_agent = registry["dev_agent"]
    except KeyError:
        logger.error("pr_review_comment_handler_job: 'dev_agent' not found in registry — skipping run.")
        return

    for ticket_id, pr_url in list(_prs_under_review.items()):
        # Deduplication guard.
        if ticket_id in _in_progress_comment_fixes:
            logger.debug(
                "pr_review_comment_handler: fix already in progress for ticket %s — skipping.",
                ticket_id,
            )
            continue

        logger.debug(
            "pr_review_comment_handler: checking PR %s for ticket %s.",
            pr_url,
            ticket_id,
        )

        status = _check_pr_review_status(pr_url, dev_agent)
        if status is None:
            logger.warning(
                "pr_review_comment_handler: could not fetch review status for PR %s — skipping.",
                pr_url,
            )
            continue

        has_changes_requested: bool = bool(status.get("has_changes_requested", False))
        unresolved_count: int = int(status.get("unresolved_comment_count", 0))
        comments: List[dict] = status.get("comments", [])

        is_actionable = has_changes_requested or unresolved_count > 0

        if not is_actionable:
            logger.debug(
                "pr_review_comment_handler: no actionable feedback for ticket %s PR %s.",
                ticket_id,
                pr_url,
            )
            continue

        logger.info(
            "pr_review_comment_handler: actionable feedback found for ticket %s "
            "(changes_requested=%s, unresolved=%d) — dispatching fix crew.",
            ticket_id,
            has_changes_requested,
            unresolved_count,
        )

        # Mark as in-progress BEFORE submitting to prevent race conditions.
        _in_progress_comment_fixes.add(ticket_id)

        executor.submit(
            _fix_review_comments,
            ticket_id,
            pr_url,
            comments,
            dev_agent,
        )
