"""
Concrete JIRA ticket tracker implementation (Phase 7).

:class:`JiraTracker` satisfies the :class:`~src.ticket.tracker.TicketTracker`
ABC and routes ticket operations as follows:

- **fetch_tickets_for_operation** — calls the JIRA REST API v3 directly via
  :class:`~src.ticket.rest_client.JiraRestClient`, bypassing the LLM entirely.
  Fetching "which tickets are ACCEPTED?" is a deterministic read with a fixed
  schema — no AI reasoning is needed, and avoiding an LLM crew on every
  scheduler tick saves both cost and latency.
- **transition / add_comment** — still route through a Dev Agent CrewAI crew
  using JIRA MCP tools.  These are write operations where LLM confirmation
  and error handling add value.

Status strings are resolved exclusively via
:class:`~src.ticket.workflow.WorkflowConfig` — no status string is ever
hardcoded in this module.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional

from crewai import Task

from .models import TicketRecord
from .rest_client import JiraRestClient
from .tracker import TicketTracker
from .workflow import WorkflowConfig, WorkflowOperation

if TYPE_CHECKING:
    pass

__all__: List[str] = ["JiraTracker"]

logger = logging.getLogger(__name__)


class JiraTracker(TicketTracker):
    """JIRA implementation of :class:`~src.ticket.tracker.TicketTracker`.

    Parameters
    ----------
    workflow:
        The shared :class:`~src.ticket.workflow.WorkflowConfig` instance.
    dev_agent:
        The CrewAI ``Agent`` object for the ``dev_agent`` role.
    crew_builder:
        The :class:`~src.crew.builder.CrewBuilder` class (or any callable that
        returns a crew with a ``kickoff()`` method).
    rest_client:
        A :class:`~src.ticket.rest_client.JiraRestClient` instance used to
        query the JIRA REST API directly for :meth:`fetch_tickets_for_operation`.
        Constructed and injected by :class:`~src.ticket.registry.TrackerRegistry`.
    project_key:
        Optional JIRA project key used to scope JQL queries (e.g. ``"PROJ"``).
        When ``None``, queries the entire workspace.  Sourced from
        ``AppSettings.JIRA_PROJECT_KEY``.
    """

    def __init__(
        self,
        workflow: WorkflowConfig,
        dev_agent: Any,
        crew_builder: Any,
        rest_client: JiraRestClient,
        project_key: Optional[str] = None,
    ) -> None:
        super().__init__(workflow)
        self._dev_agent = dev_agent
        self._crew_builder = crew_builder
        self._rest_client = rest_client
        self._project_key = project_key or None

    # ------------------------------------------------------------------
    # TicketTracker implementation
    # ------------------------------------------------------------------

    def fetch_tickets_for_operation(
        self,
        operation: WorkflowOperation,
    ) -> List[TicketRecord]:
        """Query JIRA directly via REST API for tickets matching *operation*'s status.

        Calls :meth:`~src.ticket.rest_client.JiraRestClient.search_issues`
        directly — no LLM crew is created.  This removes LLM token cost and
        latency from every scheduler poll cycle.

        Args:
            operation: The workflow operation whose configured status value
                is used as the JQL filter.

        Returns:
            List of :class:`~src.ticket.models.TicketRecord` objects.
            Returns an empty list if nothing is found.

        Raises:
            :class:`~src.ticket.rest_client.TicketFetchError`: On API error.
        """
        scan_status = self._workflow.status_for(operation)
        skip_status = self._workflow.status_for(WorkflowOperation.SKIP_REJECTED)

        raw_items = self._rest_client.search_issues(
            status=scan_status,
            exclude_status=skip_status,
            project_key=self._project_key,
        )
        records = self._parse_ticket_records_from_api(raw_items, source="jira")

        # Belt-and-suspenders BR-3 guard (JQL already excludes skip_status,
        # but we double-check for case-sensitivity edge cases).
        return [r for r in records if not self._workflow.matches(WorkflowOperation.SKIP_REJECTED, r.raw_status)]

    def transition(
        self,
        ticket_id: str,
        operation: WorkflowOperation,
    ) -> None:
        """Transition *ticket_id* in JIRA to the status configured for *operation*.

        Raises :exc:`PermissionError` via :meth:`WorkflowConfig.status_for_write`
        if the operation is ``human_only: true`` (BR-1).
        """
        target_status = self._workflow.status_for_write(operation)  # raises on human_only
        description = (
            f"Use jira/transition_issue to transition ticket '{ticket_id}' "
            f"to status '{target_status}'. Confirm the transition was successful."
        )
        expected_output = f"Confirmation that ticket '{ticket_id}' was transitioned " f"to '{target_status}'."
        self._run_crew_task(description, expected_output)
        logger.info(
            "JiraTracker: transitioned %s to '%s' (%s).",
            ticket_id,
            target_status,
            operation.value,
        )

    def add_comment(self, ticket_id: str, comment: str) -> None:
        """Post *comment* to JIRA ticket *ticket_id*."""
        description = (
            f"Use jira/add_comment to post the following comment to ticket '{ticket_id}':\n\n"
            f"{comment}\n\n"
            "Confirm the comment was posted successfully."
        )
        expected_output = f"Confirmation that a comment was posted to '{ticket_id}'."
        self._run_crew_task(description, expected_output)
        logger.info("JiraTracker: added comment to %s.", ticket_id)

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
                    :class:`~src.ticket.rest_client.JiraRestClient`.
                    Each dict has keys ``id``, ``title``, ``url``, ``status``.
            source: Ticket source string (always ``"jira"`` here).

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
                "JiraTracker: could not parse crew output as JSON (first 300 chars): %.300s",
                raw,
            )
            return []

        return JiraTracker._parse_ticket_records_from_api(items, source)
