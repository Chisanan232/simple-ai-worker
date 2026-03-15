"""
Tracker registry — selects the correct :class:`~src.ticket.tracker.TicketTracker`
by ticket source string (Phase 7).

:class:`TrackerRegistry` is constructed once at startup with a shared
:class:`~src.ticket.workflow.WorkflowConfig` instance.  Callers request a
tracker via :meth:`TrackerRegistry.get` with a ``source`` string
(``"jira"`` or ``"clickup"``).
"""

from __future__ import annotations

from typing import Any, List

from .clickup_tracker import ClickUpTracker
from .jira_tracker import JiraTracker
from .tracker import TicketTracker
from .workflow import WorkflowConfig

__all__: List[str] = ["TrackerRegistry"]


class TrackerRegistry:
    """Selects the correct :class:`~src.ticket.tracker.TicketTracker` by source.

    The :class:`~src.ticket.workflow.WorkflowConfig` is shared across all
    trackers so that status strings are consistent regardless of which
    backing system (JIRA or ClickUp) is used.

    Parameters
    ----------
    workflow:
        The shared :class:`~src.ticket.workflow.WorkflowConfig` instance.
    dev_agent:
        The CrewAI ``Agent`` object for the ``dev_agent`` role.
    crew_builder:
        The :class:`~src.crew.builder.CrewBuilder` class (or a compatible
        factory) used to build short-lived crews for each operation.
    """

    def __init__(
        self,
        workflow: WorkflowConfig,
        dev_agent: Any,
        crew_builder: Any,
    ) -> None:
        self._workflow = workflow
        self._dev_agent = dev_agent
        self._crew_builder = crew_builder

    def get(self, source: str) -> TicketTracker:
        """Return the tracker for the given *source* string.

        Args:
            source: Either ``"jira"`` or ``"clickup"``.

        Returns:
            A freshly constructed tracker instance (cheap — no I/O at
            construction time).

        Raises:
            ValueError: If *source* is not ``"jira"`` or ``"clickup"``.
        """
        if source == "jira":
            return JiraTracker(self._workflow, self._dev_agent, self._crew_builder)
        if source == "clickup":
            return ClickUpTracker(self._workflow, self._dev_agent, self._crew_builder)
        raise ValueError(
            f"Unknown ticket source: {source!r}. "
            "Supported sources: 'jira', 'clickup'."
        )

