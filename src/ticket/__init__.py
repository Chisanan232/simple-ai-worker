"""
``src.ticket`` package — Workflow operation abstraction for the Dev Agent (Phase 7).

Public exports
--------------
Data models:
    :class:`~src.ticket.models.TicketRecord`
    :class:`~src.ticket.models.PRRecord`

Workflow abstraction:
    :class:`~src.ticket.workflow.WorkflowOperation`
    :class:`~src.ticket.workflow.OperationConfig`
    :class:`~src.ticket.workflow.WorkflowConfig`

Tracker interface and implementations:
    :class:`~src.ticket.tracker.TicketTracker`
    :class:`~src.ticket.jira_tracker.JiraTracker`
    :class:`~src.ticket.clickup_tracker.ClickUpTracker`
    :class:`~src.ticket.registry.TrackerRegistry`
"""

from __future__ import annotations

from typing import List

from .clickup_tracker import ClickUpTracker
from .jira_tracker import JiraTracker
from .models import PRRecord, TicketRecord
from .registry import TrackerRegistry
from .tracker import TicketTracker
from .workflow import OperationConfig, WorkflowConfig, WorkflowOperation

__all__: List[str] = [
    # models
    "TicketRecord",
    "PRRecord",
    # workflow
    "WorkflowOperation",
    "OperationConfig",
    "WorkflowConfig",
    # trackers
    "TicketTracker",
    "JiraTracker",
    "ClickUpTracker",
    "TrackerRegistry",
]
