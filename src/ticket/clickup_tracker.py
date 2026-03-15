"""
Concrete ClickUp ticket tracker implementation (Phase 7).

:class:`ClickUpTracker` satisfies the :class:`~src.ticket.tracker.TicketTracker`
ABC and routes all ticket operations through a Dev Agent CrewAI crew using
ClickUp MCP tools.  Status strings are resolved exclusively via
:class:`~src.ticket.workflow.WorkflowConfig`.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, List

from crewai import Task

from .models import TicketRecord
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
    """

    def __init__(
        self,
        workflow: WorkflowConfig,
        dev_agent: Any,
        crew_builder: Any,
    ) -> None:
        super().__init__(workflow)
        self._dev_agent = dev_agent
        self._crew_builder = crew_builder

    # ------------------------------------------------------------------
    # TicketTracker implementation
    # ------------------------------------------------------------------

    def fetch_tickets_for_operation(
        self,
        operation: WorkflowOperation,
    ) -> List[TicketRecord]:
        """Query ClickUp for tasks matching the operation's configured status."""
        scan_status = self._workflow.status_for(operation)
        skip_status = self._workflow.status_for(WorkflowOperation.SKIP_REJECTED)

        description = (
            f"Use clickup/search_tasks with status='{scan_status}' "
            f"to find all tasks in '{scan_status}' state "
            f"that are NOT in '{skip_status}' state and have no unresolved dependencies.\n"
            "Return a JSON array of objects with fields: id, title, url, status.\n"
            "Return [] if nothing found."
        )
        expected_output = (
            "A JSON array of ClickUp task objects: "
            '[{"id": "task-123", "title": "...", "url": "...", "status": "..."}]. '
            "Return [] if no tasks match."
        )

        raw = self._run_crew_task(description, expected_output)
        return self._parse_ticket_records(raw, source="clickup")

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
        expected_output = (
            f"Confirmation that task '{ticket_id}' was updated "
            f"to status '{target_status}'."
        )
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
        """Run a single-task crew and return the raw string result."""
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
    def _parse_ticket_records(raw: str, source: str) -> List[TicketRecord]:
        """Parse the crew's JSON output into a list of :class:`TicketRecord`."""
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

