"""Test infrastructure factories for E2E tests.

Provides factory functions for creating common test objects like
settings, stub tracker registries, and agent registries.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock

from src.agents.registry import AgentRegistry
from src.ticket.models import TicketRecord


def make_settings() -> Any:
    """Create a mock settings object for E2E tests.

    Returns:
        A MagicMock settings object with standard E2E test values
    """
    s = MagicMock()
    s.PR_AUTO_MERGE_TIMEOUT_SECONDS = 300
    s.PR_REVIEW_COMMENT_CHECK_INTERVAL_SECONDS = 120
    s.MAX_CONCURRENT_DEV_AGENTS = 1
    return s


def make_stub_tracker_registry_simple(
    accepted_tickets: List[TicketRecord] | None = None,
) -> Any:
    """Create a stub TrackerRegistry for simple workflow tests.

    Used to bypass the real REST API calls in ``scan_and_dispatch_job`` when
    running against the MCP stub server (E2E_USE_TESTCONTAINERS=false).

    Args:
        accepted_tickets: List of TicketRecord objects to return for SCAN_FOR_WORK

    Returns:
        A stub TrackerRegistry object
    """
    _tickets = list(accepted_tickets or [])

    class _StubTracker:
        def fetch_tickets_for_operation(self, op: Any) -> List[TicketRecord]:
            from src.ticket.workflow import WorkflowOperation

            if op == WorkflowOperation.SCAN_FOR_WORK:
                return _tickets
            return []

    class _StubTrackerRegistry:
        def get_tracker(self, tracker_type: str) -> _StubTracker:
            return _StubTracker()

    return _StubTrackerRegistry()


def make_stub_tracker_registry_planning(
    open_tickets: List[TicketRecord] | None = None,
    in_planning_tickets: List[TicketRecord] | None = None,
    comments_by_ticket: Dict[str, List[Any]] | None = None,
) -> Any:
    """Create a stub TrackerRegistry for planning workflow tests.

    Supports multiple ticket states (open, in_planning) and ticket comments.

    Args:
        open_tickets: List of open TicketRecord objects
        in_planning_tickets: List of in-planning TicketRecord objects
        comments_by_ticket: Dictionary mapping ticket keys to comment lists

    Returns:
        A stub TrackerRegistry object
    """
    _open = list(open_tickets or [])
    _in_planning = list(in_planning_tickets or [])
    _comments = dict(comments_by_ticket or {})

    class _StubTracker:
        def fetch_tickets_for_operation(self, op: Any) -> List[TicketRecord]:
            from src.ticket.workflow import WorkflowOperation

            if op == WorkflowOperation.SCAN_FOR_WORK or op == WorkflowOperation.OPEN_FOR_DEV:
                return _open
            elif op == WorkflowOperation.IN_PLANNING:
                return _in_planning
            return []

        def get_ticket(self, key: str) -> TicketRecord | None:
            for ticket in _open + _in_planning:
                if ticket.key == key:
                    return ticket
            return None

        def get_comments(self, ticket_key: str) -> List[Any]:
            return _comments.get(ticket_key, [])

        def fetch_ticket_comments(self, ticket_id: str) -> List[Any]:
            return _comments.get(ticket_id, [])

    class _StubTrackerRegistry:
        def get_tracker(self, tracker_type: str) -> _StubTracker:
            return _StubTracker()

        def get(self, source: str) -> _StubTracker:
            return _StubTracker()

    return _StubTrackerRegistry()


def make_stub_tracker_registry_dev(
    accepted_tickets: List[TicketRecord] | None = None,
) -> Any:
    """Create a stub TrackerRegistry for dev agent workflow tests.

    Used to bypass the real REST API calls in ``scan_and_dispatch_job`` when
    running against the MCP stub server (E2E_USE_TESTCONTAINERS=false).

    Args:
        accepted_tickets: List of TicketRecord objects to return for SCAN_FOR_WORK

    Returns:
        A stub TrackerRegistry object
    """
    _tickets = list(accepted_tickets or [])

    class _StubTracker:
        def fetch_tickets_for_operation(self, op: Any) -> List[TicketRecord]:
            from src.ticket.workflow import WorkflowOperation

            if op == WorkflowOperation.SCAN_FOR_WORK:
                return _tickets
            return []

        def fetch_ticket_comments(self, ticket_id: str) -> List[Any]:
            return []

    class _StubTrackerRegistry:
        def get(self, source: str) -> _StubTracker:
            return _StubTracker()

    return _StubTrackerRegistry()


def build_planner_registry(planner_agent: Any) -> AgentRegistry:
    """Build an AgentRegistry for planner agent tests.

    Args:
        planner_agent: The planner agent instance

    Returns:
        An AgentRegistry with the planner agent registered
    """
    registry = AgentRegistry()
    registry.register("planner", planner_agent)
    return registry


def build_dev_lead_registry(dev_lead_agent: Any) -> AgentRegistry:
    """Build an AgentRegistry for dev_lead agent tests.

    Args:
        dev_lead_agent: The dev_lead agent instance

    Returns:
        An AgentRegistry with the dev_lead agent registered
    """
    registry = AgentRegistry()
    registry.register("dev_lead", dev_lead_agent)
    return registry
