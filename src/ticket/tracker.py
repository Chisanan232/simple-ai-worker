"""
Abstract base class for ticket tracker implementations (Phase 7).

Provides :class:`TicketTracker` — the interface all concrete tracker
implementations (JIRA, ClickUp) must satisfy.

All public methods accept :class:`~src.ticket.workflow.WorkflowOperation`
values rather than raw status strings.  Status string resolution and BR-1
enforcement happen inside the concrete implementation (or, for
:meth:`transition`, inside
:meth:`~src.ticket.workflow.WorkflowConfig.status_for_write` which raises
:exc:`PermissionError` on human-only operations).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from .models import TicketComment, TicketRecord
from .workflow import WorkflowConfig, WorkflowOperation

__all__: List[str] = ["TicketTracker"]


class TicketTracker(ABC):
    """Abstract interface for ticket management systems.

    Concrete subclasses provide JIRA and ClickUp implementations.  All methods
    use :class:`~src.ticket.workflow.WorkflowOperation` names — status string
    resolution and BR-1 enforcement happen internally via
    :class:`~src.ticket.workflow.WorkflowConfig`.

    Parameters
    ----------
    workflow:
        The shared :class:`~src.ticket.workflow.WorkflowConfig` instance that
        maps operations to team-specific status strings.
    """

    def __init__(self, workflow: WorkflowConfig) -> None:
        self._workflow = workflow

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fetch_tickets_for_operation(
        self,
        operation: WorkflowOperation,
    ) -> List[TicketRecord]:
        """Query tickets whose current status matches *operation*'s configured value.

        Used by :func:`~src.scheduler.jobs.scan_tickets.scan_and_dispatch_job`
        with ``WorkflowOperation.SCAN_FOR_WORK`` to find work items the Dev
        Agent should pick up.

        Args:
            operation: The workflow operation whose configured status value
                is used as the query filter.

        Returns:
            List of :class:`~src.ticket.models.TicketRecord` objects matching
            the query.  Returns an empty list if nothing is found.
        """
        ...

    @abstractmethod
    def transition(
        self,
        ticket_id: str,
        operation: WorkflowOperation,
    ) -> None:
        """Transition *ticket_id* to the status configured for *operation*.

        Internally calls
        :meth:`~src.ticket.workflow.WorkflowConfig.status_for_write` which
        raises :exc:`PermissionError` if the operation is ``human_only: true``
        (BR-1 enforcement at the Python layer).

        Args:
            ticket_id: The ticket identifier (e.g. ``"PROJ-42"``).
            operation: The workflow operation whose target status is used.

        Raises:
            PermissionError: If the operation is ``human_only: true`` (BR-1).
        """
        ...

    @abstractmethod
    def add_comment(self, ticket_id: str, comment: str) -> None:
        """Post *comment* to *ticket_id*.

        Used by ``UPDATE_WITH_CONTEXT`` — posts the Slack thread summary as
        a structured comment.  No state transition is performed.

        Args:
            ticket_id: The ticket identifier.
            comment:   The comment body (Markdown-formatted).
        """
        ...

    @abstractmethod
    def fetch_ticket_comments(self, ticket_id: str) -> List[TicketComment]:
        """Return all comments posted on *ticket_id*, ordered oldest-first.

        Used by :func:`~src.scheduler.jobs.plan_and_notify.plan_and_notify_job`
        to detect new human feedback on ``IN PLANNING`` tickets and trigger the
        plan-revision loop.

        Calls the tracker REST API directly (no LLM crew) — comment fetching
        is a deterministic read that does not require AI reasoning.

        Args:
            ticket_id: The ticket identifier (e.g. ``"PROJ-42"`` or a
                ClickUp task ID).

        Returns:
            List of :class:`~src.ticket.models.TicketComment` objects sorted
            by ``created_at`` ascending.  Returns an empty list if the ticket
            has no comments or cannot be reached.

        Raises:
            :class:`~src.ticket.rest_client.TicketFetchError`: On API error.
        """
        ...

