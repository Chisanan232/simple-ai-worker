"""
Pure data models for ticket and PR state (Phase 7).

These dataclasses are intentionally dependency-free so they can be imported
anywhere without pulling in CrewAI or MCP machinery.

Classes
-------
TicketRecord
    Immutable snapshot of a ticket returned by a tracker query.
PRRecord
    Mutable record tracking a pull request opened by the Dev Agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

__all__: List[str] = ["TicketRecord", "PRRecord"]


@dataclass(frozen=True)
class TicketRecord:
    """Immutable snapshot of a single ticket from JIRA or ClickUp.

    Attributes:
        id:         The ticket identifier (e.g. ``"PROJ-42"`` or a ClickUp
                    task ID).
        source:     The ticket management system.  Either ``"jira"`` or
                    ``"clickup"``.
        title:      Human-readable ticket title / summary.
        url:        Full URL to the ticket (empty string if unavailable).
        raw_status: The raw status string as returned by the tracker API
                    (e.g. ``"In Progress"``, ``"ACCEPTED"``).  Used by
                    :class:`~src.ticket.workflow.WorkflowConfig` for
                    operation matching.
    """

    id: str
    source: str
    title: str
    url: str
    raw_status: str = ""


@dataclass
class PRRecord:
    """Mutable record tracking a pull request opened by the Dev Agent.

    Populated by ``_execute_ticket`` after the Dev Agent opens a PR and
    stored in the shared ``_open_prs`` dict in
    :mod:`src.scheduler.jobs.pr_merge_watcher`.

    Attributes:
        ticket_id:      The ticket this PR was created for.
        pr_url:         Full URL to the GitHub pull request.
        opened_at_utc:  Unix timestamp (seconds) when the PR was opened.
        approval_count: Number of approving reviews observed at last poll.
        is_merged:      Whether the PR has been merged.
    """

    ticket_id: str
    pr_url: str
    opened_at_utc: float
    approval_count: int = 0
    is_merged: bool = False

