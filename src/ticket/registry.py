"""
Tracker registry — selects the correct :class:`~src.ticket.tracker.TicketTracker`
by ticket source string (Phase 7).

:class:`TrackerRegistry` is constructed once at startup with a shared
:class:`~src.ticket.workflow.WorkflowConfig` instance and an
:class:`~src.config.settings.AppSettings` instance.  It builds the appropriate
:class:`~src.ticket.rest_client.ClickUpRestClient` /
:class:`~src.ticket.rest_client.JiraRestClient` and injects them into each
tracker so that :meth:`~src.ticket.tracker.TicketTracker.fetch_tickets_for_operation`
calls the REST API directly instead of spinning up an LLM crew.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional

from .clickup_tracker import ClickUpTracker
from .jira_tracker import JiraTracker
from .rest_client import ClickUpRestClient, JiraRestClient
from .tracker import TicketTracker
from .workflow import WorkflowConfig

if TYPE_CHECKING:
    from src.config.settings import AppSettings

__all__: List[str] = ["TrackerRegistry"]

logger = logging.getLogger(__name__)


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
        factory) used to build short-lived crews for ``transition`` and
        ``add_comment`` operations.
    settings:
        The application :class:`~src.config.settings.AppSettings` instance.
        Used to build the REST API clients for
        :meth:`~src.ticket.tracker.TicketTracker.fetch_tickets_for_operation`.
        When ``None`` (e.g. in legacy tests), REST clients are not built and
        a ``ValueError`` is raised if a ticket source that needs them is
        requested.
    """

    def __init__(
        self,
        workflow: WorkflowConfig,
        dev_agent: Any,
        crew_builder: Any,
        settings: Optional["AppSettings"] = None,
    ) -> None:
        self._workflow = workflow
        self._dev_agent = dev_agent
        self._crew_builder = crew_builder
        self._settings = settings
        self._clickup_client: Optional[ClickUpRestClient] = None
        self._jira_client: Optional[JiraRestClient] = None

        if settings is not None:
            self._clickup_client = self._build_clickup_client(settings)
            self._jira_client = self._build_jira_client(settings)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, source: str) -> TicketTracker:
        """Return the tracker for the given *source* string.

        Args:
            source: Either ``"jira"`` or ``"clickup"``.

        Returns:
            A freshly constructed tracker instance (cheap — no I/O at
            construction time).

        Raises:
            ValueError: If *source* is not ``"jira"`` or ``"clickup"``, or
                if the required REST client could not be built (missing config).
        """
        if source == "jira":
            if self._jira_client is None:
                raise ValueError(
                    "TrackerRegistry: JiraRestClient is not available. "
                    "Ensure ATLASSIAN_URL, ATLASSIAN_EMAIL, and MCP_JIRA_TOKEN "
                    "are set in your .env file and that 'settings' was passed to "
                    "TrackerRegistry."
                )
            project_key: Optional[str] = (
                getattr(self._settings, "JIRA_PROJECT_KEY", None)
                if self._settings is not None
                else None
            )
            return JiraTracker(
                self._workflow,
                self._dev_agent,
                self._crew_builder,
                self._jira_client,
                project_key=project_key,
            )

        if source == "clickup":
            if self._clickup_client is None:
                raise ValueError(
                    "TrackerRegistry: ClickUpRestClient is not available. "
                    "Ensure MCP_CLICKUP_TOKEN and CLICKUP_LIST_ID are set in "
                    "your .env file and that 'settings' was passed to "
                    "TrackerRegistry."
                )
            return ClickUpTracker(
                self._workflow,
                self._dev_agent,
                self._crew_builder,
                self._clickup_client,
            )

        raise ValueError(
            f"Unknown ticket source: {source!r}. "
            "Supported sources: 'jira', 'clickup'."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_clickup_client(settings: "AppSettings") -> Optional[ClickUpRestClient]:
        """Build a :class:`~src.ticket.rest_client.ClickUpRestClient` from settings.

        Returns ``None`` (with a warning) if required settings are absent so
        that the registry can still be constructed even when ClickUp is not
        configured — the error surfaces only when ``get("clickup")`` is called.
        """
        from pydantic import SecretStr

        token_field = getattr(settings, "MCP_CLICKUP_TOKEN", None)
        list_id = getattr(settings, "CLICKUP_LIST_ID", None)

        if not token_field or not list_id:
            logger.warning(
                "TrackerRegistry: ClickUp REST client not configured — "
                "MCP_CLICKUP_TOKEN=%s, CLICKUP_LIST_ID=%s. "
                "Set both in .env to enable ClickUp ticket fetching.",
                "set" if token_field else "missing",
                "set" if list_id else "missing",
            )
            return None

        api_token = (
            token_field.get_secret_value()
            if isinstance(token_field, SecretStr)
            else str(token_field)
        )
        return ClickUpRestClient(api_token=api_token, list_id=str(list_id))

    @staticmethod
    def _build_jira_client(settings: "AppSettings") -> Optional[JiraRestClient]:
        """Build a :class:`~src.ticket.rest_client.JiraRestClient` from settings.

        Returns ``None`` (with a warning) if required settings are absent.
        """
        from pydantic import SecretStr

        token_field = getattr(settings, "MCP_JIRA_TOKEN", None)
        base_url = getattr(settings, "ATLASSIAN_URL", None)
        email = getattr(settings, "ATLASSIAN_EMAIL", None)

        if not token_field or not base_url or not email:
            logger.warning(
                "TrackerRegistry: JIRA REST client not configured — "
                "MCP_JIRA_TOKEN=%s, ATLASSIAN_URL=%s, ATLASSIAN_EMAIL=%s. "
                "Set all three in .env to enable JIRA ticket fetching.",
                "set" if token_field else "missing",
                "set" if base_url else "missing",
                "set" if email else "missing",
            )
            return None

        api_token = (
            token_field.get_secret_value()
            if isinstance(token_field, SecretStr)
            else str(token_field)
        )
        return JiraRestClient(
            base_url=str(base_url),
            api_token=api_token,
            email=str(email),
        )
