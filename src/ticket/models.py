"""
Pure data models for ticket and PR state (Phase 7 + Phase 9).

All models use **Pydantic** ``BaseModel`` for field validation, clear
schema documentation, and IDE-friendly type hints.

Classes
-------
TicketRecord
    Immutable snapshot of a ticket returned by a tracker query.
    ``frozen=True`` makes instances hashable and prevents accidental mutation.
PRRecord
    Mutable record tracking a pull request opened by the Dev Agent.
    Fields are updated in-place by the watcher jobs as the PR progresses.
TicketComment
    Immutable snapshot of a single comment on a ticket (JIRA or ClickUp).
    Used by the plan-and-notify job to detect new human feedback on
    ``IN PLANNING`` tickets.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field

__all__: List[str] = ["TicketRecord", "PRRecord", "TicketComment"]


class TicketRecord(BaseModel):
    """Immutable snapshot of a single ticket from JIRA or ClickUp.

    Attributes:
        id:         The ticket identifier (e.g. ``"PROJ-42"`` or a ClickUp
                    task ID).
        source:     The ticket management system — ``"jira"`` or
                    ``"clickup"``.
        title:      Human-readable ticket title / summary.
        url:        Full URL to the ticket (empty string if unavailable).
        raw_status: The raw status string as returned by the tracker API
                    (e.g. ``"In Progress"``, ``"ACCEPTED"``).  Used by
                    :class:`~src.ticket.workflow.WorkflowConfig` for
                    operation matching.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., description="Ticket identifier, e.g. 'PROJ-42' or a ClickUp task ID.")
    source: str = Field(..., description="Ticket system: 'jira' or 'clickup'.")
    title: str = Field(..., description="Human-readable ticket title.")
    url: str = Field(..., description="Full URL to the ticket; empty string if unavailable.")
    raw_status: str = Field(default="", description="Raw status string from the tracker API.")


class PRRecord(BaseModel):
    """Mutable record tracking a pull request opened by the Dev Agent.

    Populated by ``_execute_ticket`` after the Dev Agent opens a PR and
    stored in the shared ``_open_prs`` dict in
    :mod:`src.scheduler.jobs.scan_tickets`.  Fields are updated in-place
    by the watcher jobs as review and merge events are observed.

    Attributes:
        ticket_id:      The ticket this PR was created for.
        pr_url:         Full URL to the GitHub pull request.
        opened_at_utc:  Unix timestamp (seconds) when the PR was opened.
        approval_count: Number of approving reviews observed at last poll.
        is_merged:      Whether the PR has been merged.
    """

    ticket_id: str = Field(..., description="Ticket ID this PR was created for.")
    pr_url: str = Field(..., description="Full GitHub pull request URL.")
    opened_at_utc: float = Field(..., description="Unix timestamp (UTC) when the PR was opened.")
    approval_count: int = Field(default=0, description="Approving review count observed at last poll.", ge=0)
    is_merged: bool = Field(default=False, description="Whether the PR has been merged.")


class TicketComment(BaseModel):
    """Immutable snapshot of a single comment on a JIRA or ClickUp ticket.

    Used by :func:`~src.scheduler.jobs.plan_and_notify.plan_and_notify_job`
    to detect new human feedback on ``IN PLANNING`` tickets and trigger the
    Dev Agent plan-revision loop.

    Attributes:
        id:          The comment identifier (string; numeric for JIRA, may be
                     alphanumeric for ClickUp).
        author:      Display name or username of the comment author.
        body:        The raw comment text (may be Markdown or plain text).
        created_at:  Unix timestamp (UTC seconds) when the comment was posted.
                     Used as a watermark to detect new comments since the last
                     plan-revision run.
        source:      The ticket system this comment came from — ``"jira"``
                     or ``"clickup"``.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., description="Comment identifier.")
    author: str = Field(default="", description="Display name of the comment author.")
    body: str = Field(default="", description="Comment text body.")
    created_at: float = Field(default=0.0, description="Unix timestamp (UTC) when the comment was created.")
    source: str = Field(..., description="Ticket system: 'jira' or 'clickup'.")
