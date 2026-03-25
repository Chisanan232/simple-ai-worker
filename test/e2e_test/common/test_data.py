"""Test data factories for E2E tests.

Provides factory functions for creating common test objects like
TicketRecord, TicketComment, and workflow configurations.
"""

from typing import Any, Dict

from src.ticket.models import TicketComment, TicketRecord


def make_ticket_record(
    key: str,
    summary: str = "Test ticket",
    description: str = "Test description",
    status: str = "OPEN",
    assignee: str | None = None,
    fields: Dict[str, Any] | None = None,
) -> TicketRecord:
    """Create a TicketRecord for testing.

    Args:
        key: Ticket key (e.g., 'PROJ-1')
        summary: Ticket summary
        description: Ticket description
        status: Ticket status
        assignee: Optional assignee
        fields: Optional additional fields

    Returns:
        A TicketRecord instance
    """
    _fields = fields or {}
    _fields.setdefault("summary", summary)
    _fields.setdefault("status", {"name": status})
    _fields.setdefault("description", description)
    if assignee:
        _fields.setdefault("assignee", {"displayName": assignee})

    return TicketRecord(
        key=key,
        url=f"https://jira.example.com/browse/{key}",
        fields=_fields,
    )


def make_ticket_comment(
    id: str,
    body: str,
    author: str = "Test User",
    created: str = "2026-03-24T00:00:00Z",
) -> TicketComment:
    """Create a TicketComment for testing.

    Args:
        id: Comment ID
        body: Comment body text
        author: Author name
        created: Creation timestamp

    Returns:
        A TicketComment instance
    """
    return TicketComment(
        id=id,
        body=body,
        author=author,
        created=created,
    )


def make_planning_workflow_config() -> Dict[str, Any]:
    """Create a planning workflow configuration for E2E tests.

    Returns:
        A workflow configuration dictionary with planning states
    """
    from test.e2e_test.conftest import E2E_WORKFLOW_CONFIG

    return {
        **E2E_WORKFLOW_CONFIG,
        "open_for_dev": {"status_value": "OPEN", "human_only": False},
        "in_planning": {"status_value": "IN PLANNING", "human_only": True},
    }


def make_jira_issue_response(
    key: str,
    summary: str = "Test issue",
    status: str = "OPEN",
    description: str = "Test description",
) -> Dict[str, Any]:
    """Create a mock JIRA issue response for testing.

    Args:
        key: Issue key
        summary: Issue summary
        status: Issue status
        description: Issue description

    Returns:
        A dictionary representing a JIRA issue response
    """
    return {
        "key": key,
        "fields": {
            "summary": summary,
            "status": {"name": status},
            "description": description,
        },
    }


def make_clickup_task_response(
    id: str,
    name: str = "Test task",
    status: str = "OPEN",
    description: str = "Test description",
) -> Dict[str, Any]:
    """Create a mock ClickUp task response for testing.

    Args:
        id: Task ID
        name: Task name
        status: Task status
        description: Task description

    Returns:
        A dictionary representing a ClickUp task response
    """
    return {
        "id": id,
        "name": name,
        "status": {"status": status},
        "description": description,
    }


def make_github_pr_response(
    number: int,
    title: str = "Test PR",
    html_url: str = "https://github.com/test/test/pull/1",
    state: str = "open",
) -> Dict[str, Any]:
    """Create a mock GitHub PR response for testing.

    Args:
        number: PR number
        title: PR title
        html_url: PR URL
        state: PR state (open, closed, merged)

    Returns:
        A dictionary representing a GitHub PR response
    """
    return {
        "number": number,
        "title": title,
        "html_url": html_url,
        "state": state,
    }
