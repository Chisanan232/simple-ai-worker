"""
Unit tests for :mod:`src.ticket.tracker` (UNIT-TR-01 through UNIT-TR-06).

Covers:
- TicketTracker ABC cannot be instantiated directly
- Concrete subclass satisfying the ABC compiles and works
- __init__ stores workflow on self._workflow
- fetch_tickets_for_operation, transition, add_comment, fetch_ticket_comments are abstract
"""

from __future__ import annotations

from typing import List

import pytest

from src.ticket.models import TicketComment, TicketRecord
from src.ticket.tracker import TicketTracker
from src.ticket.workflow import WorkflowConfig, WorkflowOperation

# ---------------------------------------------------------------------------
# Minimal concrete subclass used only in this test module
# ---------------------------------------------------------------------------


class _ConcreteTracker(TicketTracker):
    """Minimal concrete implementation for ABC testing."""

    def fetch_tickets_for_operation(self, operation: WorkflowOperation) -> List[TicketRecord]:
        return []

    def transition(self, ticket_id: str, operation: WorkflowOperation) -> None:
        pass

    def add_comment(self, ticket_id: str, comment: str) -> None:
        pass

    def fetch_ticket_comments(self, ticket_id: str) -> List[TicketComment]:
        return []


_WORKFLOW_CFG = {
    "scan_for_work": {"status_value": "ACCEPTED", "human_only": True},
    "skip_rejected": {"status_value": "REJECTED"},
    "start_development": {"status_value": "IN PROGRESS"},
    "open_for_review": {"status_value": "IN REVIEW"},
    "mark_complete": {"status_value": "COMPLETE"},
    "update_with_context": {"status_value": ""},
}


class TestTicketTrackerABC:
    def test_cannot_instantiate_abstract_class_directly(self) -> None:
        """UNIT-TR-01: TicketTracker cannot be instantiated (abstract class)."""
        with pytest.raises(TypeError):
            TicketTracker(workflow=WorkflowConfig(_WORKFLOW_CFG))  # type: ignore[abstract]

    def test_concrete_subclass_instantiates_successfully(self) -> None:
        """UNIT-TR-02: A fully-implemented subclass can be instantiated."""
        workflow = WorkflowConfig(_WORKFLOW_CFG)
        tracker = _ConcreteTracker(workflow=workflow)
        assert tracker is not None

    def test_workflow_stored_on_instance(self) -> None:
        """UNIT-TR-03: __init__ stores the WorkflowConfig on self._workflow."""
        workflow = WorkflowConfig(_WORKFLOW_CFG)
        tracker = _ConcreteTracker(workflow=workflow)
        assert tracker._workflow is workflow

    def test_concrete_fetch_returns_empty_list(self) -> None:
        """UNIT-TR-05: fetch_tickets_for_operation returns empty list in minimal impl."""
        workflow = WorkflowConfig(_WORKFLOW_CFG)
        tracker = _ConcreteTracker(workflow=workflow)
        result = tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK)
        assert result == []

    def test_concrete_fetch_comments_returns_empty_list(self) -> None:
        """UNIT-TR-06: fetch_ticket_comments returns empty list in minimal impl."""
        workflow = WorkflowConfig(_WORKFLOW_CFG)
        tracker = _ConcreteTracker(workflow=workflow)
        result = tracker.fetch_ticket_comments("PROJ-1")
        assert result == []

    def test_concrete_transition_and_add_comment_callable(self) -> None:
        """UNIT-TR-07: transition() and add_comment() can be called without error."""
        workflow = WorkflowConfig(_WORKFLOW_CFG)
        tracker = _ConcreteTracker(workflow=workflow)
        tracker.transition("PROJ-1", WorkflowOperation.START_DEVELOPMENT)
        tracker.add_comment("PROJ-1", "test comment")

    def test_subclass_missing_one_abstract_method_cannot_instantiate(self) -> None:
        """UNIT-TR-04: Subclass that omits any abstract method cannot be instantiated."""

        class _Incomplete(TicketTracker):
            def fetch_tickets_for_operation(self, operation: WorkflowOperation) -> List[TicketRecord]:
                return []

            def transition(self, ticket_id: str, operation: WorkflowOperation) -> None:
                pass

            # add_comment and fetch_ticket_comments intentionally omitted

        with pytest.raises(TypeError):
            _Incomplete(workflow=WorkflowConfig(_WORKFLOW_CFG))  # type: ignore[abstract]

    def test_concrete_fetch_returns_empty_list(self) -> None:
        """UNIT-TR-05: Minimal concrete implementation returns empty list from fetch."""
        tracker = _ConcreteTracker(workflow=WorkflowConfig(_WORKFLOW_CFG))
        result = tracker.fetch_tickets_for_operation(WorkflowOperation.SCAN_FOR_WORK)
        assert result == []

    def test_concrete_transition_and_add_comment_callable(self) -> None:
        """UNIT-TR-06: transition and add_comment on concrete subclass are callable."""
        tracker = _ConcreteTracker(workflow=WorkflowConfig(_WORKFLOW_CFG))
        tracker.transition("PROJ-1", WorkflowOperation.START_DEVELOPMENT)
        tracker.add_comment("PROJ-1", "comment body")
