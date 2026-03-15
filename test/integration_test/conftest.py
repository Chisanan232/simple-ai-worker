"""
Integration test shared fixtures.

Provides:
- ``default_workflow_config``   — Team A (ACCEPTED, IN PROGRESS, IN REVIEW, COMPLETE, REJECTED)
- ``team_b_workflow_config``    — Team B (Approved, Developing, PR Raised, Finished, Cancelled)
- ``mock_dev_registry``         — AgentRegistry stub with a mock dev_agent
- ``dev_agent_state_reset``     — autouse fixture that clears shared module-level dicts
                                   before every test
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.ticket.workflow import WorkflowConfig

# ---------------------------------------------------------------------------
# Standard workflow configs for re-use across all integration test modules
# ---------------------------------------------------------------------------

_TEAM_A_WORKFLOW = {
    "scan_for_work": {"status_value": "ACCEPTED", "human_only": True},
    "skip_rejected": {"status_value": "REJECTED"},
    "start_development": {"status_value": "IN PROGRESS"},
    "open_for_review": {"status_value": "IN REVIEW"},
    "mark_complete": {"status_value": "COMPLETE"},
    "update_with_context": {"status_value": ""},
}

_TEAM_B_WORKFLOW = {
    "scan_for_work": {"status_value": "Approved", "human_only": True},
    "skip_rejected": {"status_value": "Cancelled"},
    "start_development": {"status_value": "Developing"},
    "open_for_review": {"status_value": "PR Raised"},
    "mark_complete": {"status_value": "Finished"},
    "update_with_context": {"status_value": ""},
}


@pytest.fixture
def default_workflow_config() -> WorkflowConfig:
    """WorkflowConfig for Team A (standard status names)."""
    return WorkflowConfig(_TEAM_A_WORKFLOW)


@pytest.fixture
def team_b_workflow_config() -> WorkflowConfig:
    """WorkflowConfig for Team B (different status vocabulary)."""
    return WorkflowConfig(_TEAM_B_WORKFLOW)


@pytest.fixture
def mock_dev_agent() -> MagicMock:
    """A MagicMock standing in for a CrewAI dev Agent object."""
    agent = MagicMock()
    agent.name = "dev_agent"
    return agent


@pytest.fixture
def mock_dev_registry(mock_dev_agent: MagicMock) -> MagicMock:
    """AgentRegistry stub that returns mock_dev_agent for key 'dev_agent'."""
    registry = MagicMock()
    registry.__getitem__ = MagicMock(return_value=mock_dev_agent)
    registry.agent_ids = MagicMock(return_value=["dev_agent"])
    return registry


@pytest.fixture(autouse=True)
def dev_agent_state_reset() -> None:  # type: ignore[return]
    """Clear all shared module-level state dicts/sets before each test.

    This prevents cross-test pollution when tests write to the shared
    in-memory dicts used by scan_tickets, pr_merge_watcher, and
    pr_review_comment_handler.
    """
    import src.scheduler.jobs.scan_tickets as scan_mod
    import src.scheduler.jobs.pr_review_comment_handler as review_mod

    # Clear all shared collections before the test.
    scan_mod._in_progress_tickets.clear()
    scan_mod._open_prs.clear()
    scan_mod._prs_under_review.clear()
    review_mod._in_progress_comment_fixes.clear()

    yield

    # Also clear after — belt-and-suspenders.
    scan_mod._in_progress_tickets.clear()
    scan_mod._open_prs.clear()
    scan_mod._prs_under_review.clear()
    review_mod._in_progress_comment_fixes.clear()

