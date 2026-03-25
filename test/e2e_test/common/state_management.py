"""State management utilities for E2E tests.

Provides helpers for managing and cleaning up shared state used in tests,
particularly for the plan_and_notify module.
"""

from typing import Any


def reset_planning_state(pn_module: Any) -> None:
    """Reset the planning state in plan_and_notify module.

    Clears both _in_planning_tickets and _plan_comment_watermarks.

    Args:
        pn_module: The plan_and_notify module object
    """
    pn_module._in_planning_tickets.clear()
    pn_module._plan_comment_watermarks.clear()


def reset_in_planning_tickets(pn_module: Any) -> None:
    """Reset only the in_planning_tickets set.

    Args:
        pn_module: The plan_and_notify module object
    """
    pn_module._in_planning_tickets.clear()


def reset_plan_comment_watermarks(pn_module: Any) -> None:
    """Reset only the plan_comment_watermarks dictionary.

    Args:
        pn_module: The plan_and_notify module object
    """
    pn_module._plan_comment_watermarks.clear()


def add_in_planning_ticket(pn_module: Any, ticket_key: str) -> None:
    """Add a ticket to the in_planning_tickets set.

    Args:
        pn_module: The plan_and_notify module object
        ticket_key: The ticket key to add
    """
    pn_module._in_planning_tickets.add(ticket_key)


def set_plan_comment_watermark(pn_module: Any, ticket_key: str, timestamp: float) -> None:
    """Set a watermark timestamp for a ticket's plan comment.

    Args:
        pn_module: The plan_and_notify module object
        ticket_key: The ticket key
        timestamp: The watermark timestamp
    """
    pn_module._plan_comment_watermarks[ticket_key] = timestamp
