"""
Unit tests for :mod:`src.ticket.registry` (UNIT-TR-01 through UNIT-TR-10).

Covers:
- get("jira")     → returns JiraTracker instance
- get("clickup")  → returns ClickUpTracker instance
- get(unknown)    → raises ValueError with the unknown source in message
- Returned trackers share the same WorkflowConfig
- Each call to get() returns a fresh tracker instance
- Tracker instances are bound to the correct dev_agent and crew_builder
- REST clients are injected into the trackers
- get() raises ValueError when REST client is not available (missing config)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.ticket.clickup_tracker import ClickUpTracker
from src.ticket.jira_tracker import JiraTracker
from src.ticket.registry import TrackerRegistry
from src.ticket.rest_client import ClickUpRestClient, JiraRestClient
from src.ticket.workflow import WorkflowConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_WORKFLOW_CFG = {
    "scan_for_work": {"status_value": "ACCEPTED", "human_only": True},
    "skip_rejected": {"status_value": "REJECTED"},
    "start_development": {"status_value": "IN PROGRESS"},
    "open_for_review": {"status_value": "IN REVIEW"},
    "mark_complete": {"status_value": "COMPLETE"},
    "update_with_context": {"status_value": ""},
}


def _make_registry() -> tuple[TrackerRegistry, WorkflowConfig, MagicMock, MagicMock]:
    """Build a TrackerRegistry with mock REST clients injected directly."""
    workflow = WorkflowConfig(_WORKFLOW_CFG)
    dev_agent = MagicMock()
    crew_builder = MagicMock()

    # Build with no settings — inject pre-built mock REST clients instead.
    registry = TrackerRegistry(
        workflow=workflow,
        dev_agent=dev_agent,
        crew_builder=crew_builder,
        settings=None,
    )
    # Inject mock REST clients directly so get() can build trackers.
    registry._clickup_client = MagicMock(spec=ClickUpRestClient)
    registry._jira_client = MagicMock(spec=JiraRestClient)
    return registry, workflow, dev_agent, crew_builder


# ===========================================================================
# TrackerRegistry.get
# ===========================================================================


class TestTrackerRegistryGet:
    def test_get_jira_returns_jira_tracker(self) -> None:
        """UNIT-REG-01: get('jira') returns a JiraTracker instance."""
        registry, _, _, _ = _make_registry()
        tracker = registry.get("jira")
        assert isinstance(tracker, JiraTracker)

    def test_get_clickup_returns_clickup_tracker(self) -> None:
        """UNIT-REG-02: get('clickup') returns a ClickUpTracker instance."""
        registry, _, _, _ = _make_registry()
        tracker = registry.get("clickup")
        assert isinstance(tracker, ClickUpTracker)

    def test_get_unknown_source_raises_value_error(self) -> None:
        """UNIT-REG-03: get() with an unknown source raises ValueError."""
        registry, _, _, _ = _make_registry()
        with pytest.raises(ValueError):
            registry.get("linear")

    def test_value_error_message_contains_unknown_source(self) -> None:
        """UNIT-REG-04: ValueError message names the unknown source string."""
        registry, _, _, _ = _make_registry()
        with pytest.raises(ValueError) as exc_info:
            registry.get("notion")
        assert "notion" in str(exc_info.value)

    def test_jira_tracker_uses_shared_workflow(self) -> None:
        """UNIT-REG-05: The JiraTracker returned shares the registry's WorkflowConfig."""
        registry, workflow, _, _ = _make_registry()
        tracker = registry.get("jira")
        assert tracker._workflow is workflow

    def test_clickup_tracker_uses_shared_workflow(self) -> None:
        """UNIT-REG-06: The ClickUpTracker returned shares the registry's WorkflowConfig."""
        registry, workflow, _, _ = _make_registry()
        tracker = registry.get("clickup")
        assert tracker._workflow is workflow

    def test_each_call_returns_new_instance(self) -> None:
        """UNIT-REG-07: Two consecutive get('jira') calls return distinct instances."""
        registry, _, _, _ = _make_registry()
        t1 = registry.get("jira")
        t2 = registry.get("jira")
        assert t1 is not t2

    def test_tracker_bound_to_correct_dev_agent(self) -> None:
        """UNIT-REG-08: Returned tracker has the dev_agent passed to the registry."""
        registry, _, dev_agent, _ = _make_registry()
        tracker = registry.get("jira")
        assert tracker._dev_agent is dev_agent

    def test_tracker_bound_to_correct_crew_builder(self) -> None:
        """UNIT-REG-09: Returned tracker has the crew_builder passed to the registry."""
        registry, _, _, crew_builder = _make_registry()
        tracker = registry.get("clickup")
        assert tracker._crew_builder is crew_builder

    def test_value_error_mentions_supported_sources(self) -> None:
        """UNIT-REG-10: ValueError message lists supported sources for easier debugging."""
        registry, _, _, _ = _make_registry()
        with pytest.raises(ValueError) as exc_info:
            registry.get("asana")
        msg = str(exc_info.value)
        assert "jira" in msg or "clickup" in msg

    def test_get_jira_raises_when_jira_client_not_configured(self) -> None:
        """UNIT-REG-11: get('jira') raises ValueError when jira REST client is None."""
        workflow = WorkflowConfig(_WORKFLOW_CFG)
        registry = TrackerRegistry(
            workflow=workflow,
            dev_agent=MagicMock(),
            crew_builder=MagicMock(),
            settings=None,
        )
        # _jira_client is None (no settings provided, no manual injection)
        with pytest.raises(ValueError, match="JiraRestClient"):
            registry.get("jira")

    def test_get_clickup_raises_when_clickup_client_not_configured(self) -> None:
        """UNIT-REG-12: get('clickup') raises ValueError when clickup REST client is None."""
        workflow = WorkflowConfig(_WORKFLOW_CFG)
        registry = TrackerRegistry(
            workflow=workflow,
            dev_agent=MagicMock(),
            crew_builder=MagicMock(),
            settings=None,
        )
        with pytest.raises(ValueError, match="ClickUpRestClient"):
            registry.get("clickup")

    def test_jira_tracker_receives_rest_client(self) -> None:
        """UNIT-REG-13: JiraTracker is injected with the registry's jira_client."""
        registry, _, _, _ = _make_registry()
        tracker = registry.get("jira")
        assert tracker._rest_client is registry._jira_client

    def test_clickup_tracker_receives_rest_client(self) -> None:
        """UNIT-REG-14: ClickUpTracker is injected with the registry's clickup_client."""
        registry, _, _, _ = _make_registry()
        tracker = registry.get("clickup")
        assert tracker._rest_client is registry._clickup_client
