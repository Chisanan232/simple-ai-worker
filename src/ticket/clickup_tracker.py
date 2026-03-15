"""
Concrete ClickUp ticket tracker implementation (Phase 7).

:class:`ClickUpTracker` satisfies the :class:`~src.ticket.tracker.TicketTracker`
ABC and routes ticket operations as follows:

- **fetch_tickets_for_operation** — calls the ClickUp REST API v2 directly via
  :class:`~src.ticket.rest_client.ClickUpRestClient`, bypassing the LLM entirely.
  Fetching "which tasks are ACCEPTED?" is a deterministic read with a fixed
  schema — no AI reasoning is needed, and avoiding an LLM crew on every
  scheduler tick saves both cost and latency.
- **transition / add_comment** — still route through a Dev Agent CrewAI crew
  using ClickUp MCP tools.  These are write operations where LLM confirmation
  and error handling add value.

Status strings are resolved exclusively via
:class:`~src.ticket.workflow.WorkflowConfig`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List

from crewai import Task

from .models import TicketRecord
from .rest_client import ClickUpRestClient
from .tracker import TicketTracker
from .workflow import WorkflowConfig, WorkflowOperation

if TYPE_CHECKING:
    pass

__all__: List[str] = ["ClickUpTracker"]

logger = logging.getLogger(__name__)


class ClickUpTracker(TicketTracker):
    """ClickUp implementation of :class:`~src.ticket.tracker.TicketTracker`.

    Parameters
    ----------
    workflow:
        The shared :class:`~src.ticket.workflow.WorkflowConfig` instance.
    dev_agent:
        The CrewAI ``Agent`` object for the ``dev_agent`` role.
    crew_builder:
        The :class:`~src.crew.builder.CrewBuilder` class (or any callable
        that returns a crew with a ``kickoff()`` method).
    rest_client:
        A :class:`~src.ticket.rest_client.ClickUpRestClient` instance used
        to query the ClickUp REST API directly for
        :meth:`fetch_tickets_for_operation`.  Constructed and injected by
        :class:`~src.ticket.registry.TrackerRegistry`.
    """

    def __init__(
        self,
        workflow: WorkflowConfig,
        dev_agent: Any,
        crew_builder: Any,
        rest_client: ClickUpRestClient,
    ) -> None:
        super().__init__(workflow)
        self._dev_agent = dev_agent
        self._crew_builder = crew_builder
        self._rest_client = rest_client

    # ------------------------------------------------------------------
    # TicketTracker implementation
    # ------------------------------------------------------------------

    def fetch_tickets_for_operation(
        self,
        operation: WorkflowOperation,
    ) -> List[TicketRecord]:
        """Query ClickUp directly via REST API for tasks matching *operation*'s status.

        Calls :meth:`~src.ticket.rest_client.ClickUpRestClient.search_tasks`
        directly — no LLM crew is created.  This removes LLM token cost and
        latency from every scheduler poll cycle.

        Args:
            operation: The workflow operation whose configured status value
                is used as the query filter.

        Returns:
            List of :class:`~src.ticket.models.TicketRecord` objects.
            Returns an empty list if nothing is found.

        Raises:
            :class:`~src.ticket.rest_client.TicketFetchError`: On API error.
        """
        scan_status = self._workflow.status_for(operation)
        skip_status = self._workflow.status_for(WorkflowOperation.SKIP_REJECTED)

        raw_items = self._rest_client.search_tasks(
            status=scan_status,
            exclude_status=skip_status,
        )
        records = self._parse_ticket_records_from_api(raw_items, source="clickup")

        # Belt-and-suspenders BR-3 guard: the REST client already excludes
        # the skip_status server-side, but we double-check here in case of
        # case-sensitivity differences or partial matches.
        return [r for r in records if not self._workflow.matches(WorkflowOperation.SKIP_REJECTED, r.raw_status)]

    def transition(
        self,
        ticket_id: str,
        operation: WorkflowOperation,
    ) -> None:
        """Transition *ticket_id* in ClickUp to the status configured for *operation*.

        Raises :exc:`PermissionError` via :meth:`WorkflowConfig.status_for_write`
        if the operation is ``human_only: true`` (BR-1).
        """
        target_status = self._workflow.status_for_write(operation)  # raises on human_only
        description = (
            f"Use clickup/update_task to update task '{ticket_id}' "
            f"setting status to '{target_status}'. Confirm the update was successful."
        )
        expected_output = f"Confirmation that task '{ticket_id}' was updated " f"to status '{target_status}'."
        self._run_crew_task(description, expected_output)
        logger.info(
            "ClickUpTracker: transitioned %s to '%s' (%s).",
            ticket_id,
            target_status,
            operation.value,
        )

    def add_comment(self, ticket_id: str, comment: str) -> None:
        """Post *comment* to ClickUp task *ticket_id*."""
        description = (
            f"Use clickup/add_comment to post the following comment to task '{ticket_id}':\n\n"
            f"{comment}\n\n"
            "Confirm the comment was posted successfully."
        )
        expected_output = f"Confirmation that a comment was posted to '{ticket_id}'."
        self._run_crew_task(description, expected_output)
        logger.info("ClickUpTracker: added comment to %s.", ticket_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_crew_task(self, description: str, expected_output: str) -> str:
        """Run a single-task crew and return the raw string result.

        Used by :meth:`transition` and :meth:`add_comment` — write operations
        that still benefit from LLM-driven confirmation and error handling.
        """
        task = Task(
            description=description,
            expected_output=expected_output,
            agent=self._dev_agent,
        )
        crew = self._crew_builder.build(
            agents=[self._dev_agent],
            tasks=[task],
            process="sequential",
        )
        result = crew.kickoff()
        return str(result).strip()

    @staticmethod
    def _parse_ticket_records_from_api(
        items: List[dict],
        source: str,
    ) -> List[TicketRecord]:
        """Map already-parsed API dicts to :class:`~src.ticket.models.TicketRecord` objects.

        Unlike :meth:`_parse_ticket_records`, this method receives the
        already-deserialized list from the REST client (no JSON parsing needed).

        Args:
            items:  List of normalised dicts from
                    :class:`~src.ticket.rest_client.ClickUpRestClient`.
                    Each dict has keys ``id``, ``title``, ``url``, ``status``.
            source: Ticket source string (always ``"clickup"`` here).

        Returns:
            List of :class:`~src.ticket.models.TicketRecord` objects.
            Items missing an ``id`` are silently skipped.
        """
        records: List[TicketRecord] = []
        for item in items:
            ticket_id = str(item.get("id", "")).strip()
            if not ticket_id:
                continue
            records.append(
                TicketRecord(
                    id=ticket_id,
                    source=source,
                    title=str(item.get("title", ticket_id)),
                    url=str(item.get("url", "")),
                    raw_status=str(item.get("status", "")),
                )
            )
        return records

    @staticmethod
    def _parse_ticket_records(raw: str, source: str) -> List[TicketRecord]:
        """Parse a JSON string (from crew output) into TicketRecord objects.

        Kept for backward-compatibility — no longer used by
        :meth:`fetch_tickets_for_operation` but may be used by subclasses
        or future extensions.
        """
        import json

        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(ln for ln in lines if not ln.strip().startswith("```"))

        try:
            items: list = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(
                "ClickUpTracker: could not parse crew output as JSON (first 300 chars): %.300s",
                raw,
            )
            return []

        return ClickUpTracker._parse_ticket_records_from_api(items, source)
