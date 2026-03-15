"""
Integration tests for plan_and_notify_job (INT-PL-01 through INT-PL-05).

Tests cover:
- INT-PL-01: Job calls tracker_registry.get('jira') and get('clickup') for OPEN_FOR_DEV tickets
- INT-PL-02: Each OPEN_FOR_DEV ticket dispatches _create_initial_plan crew via executor
- INT-PL-03: Dispatch guard prevents double-planning the same ticket
- INT-PL-04: IN_PLANNING tickets with new comments dispatch _revise_plan crew
- INT-PL-05: No revision crew when no new comments since watermark
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from src.ticket.models import TicketComment, TicketRecord
from src.ticket.rest_client import TicketFetchError
from src.ticket.workflow import WorkflowConfig, WorkflowOperation

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Workflow config with Phase-9 ops
# ---------------------------------------------------------------------------

_WORKFLOW_CFG = {
    "scan_for_work": {"status_value": "ACCEPTED", "human_only": True},
    "skip_rejected": {"status_value": "REJECTED"},
    "open_for_dev": {"status_value": "OPEN", "human_only": False},
    "in_planning": {"status_value": "IN PLANNING", "human_only": True},
    "start_development": {"status_value": "IN PROGRESS"},
    "open_for_review": {"status_value": "IN REVIEW"},
    "mark_complete": {"status_value": "COMPLETE"},
    "update_with_context": {"status_value": ""},
}


def _make_workflow() -> WorkflowConfig:
    return WorkflowConfig(_WORKFLOW_CFG)


def _make_settings() -> MagicMock:
    s = MagicMock()
    return s


def _make_registry(has_dev_agent: bool = True) -> MagicMock:
    registry = MagicMock()
    if has_dev_agent:
        agent = MagicMock()
        agent.name = "dev_agent"
        registry.__getitem__ = MagicMock(return_value=agent)
        registry.agent_ids = MagicMock(return_value=["dev_agent"])
    else:
        registry.__getitem__ = MagicMock(side_effect=KeyError("dev_agent"))
        registry.agent_ids = MagicMock(return_value=[])
    return registry


def _make_open_ticket(ticket_id: str = "PROJ-1", source: str = "jira") -> TicketRecord:
    return TicketRecord(
        id=ticket_id,
        source=source,
        title=f"Task {ticket_id}",
        url="",
        raw_status="OPEN",
    )


def _make_in_planning_ticket(ticket_id: str = "PROJ-2", source: str = "jira") -> TicketRecord:
    return TicketRecord(
        id=ticket_id,
        source=source,
        title=f"Task {ticket_id}",
        url="",
        raw_status="IN PLANNING",
    )


def _make_tracker_registry(
    open_tickets: list | None = None,
    in_planning_tickets: list | None = None,
    comments: list | None = None,
    open_error: Exception | None = None,
) -> MagicMock:
    """Build a mock TrackerRegistry for plan_and_notify_job tests."""
    tr = MagicMock()

    def _get(source: str) -> MagicMock:
        tracker = MagicMock()

        def fetch_for_op(op: WorkflowOperation) -> list:
            if op == WorkflowOperation.OPEN_FOR_DEV:
                if open_error:
                    raise open_error
                return [t for t in (open_tickets or []) if t.source == source]
            elif op == WorkflowOperation.IN_PLANNING:
                return [t for t in (in_planning_tickets or []) if t.source == source]
            return []

        tracker.fetch_tickets_for_operation.side_effect = fetch_for_op
        tracker.fetch_ticket_comments.return_value = comments or []
        return tracker

    tr.get.side_effect = _get
    return tr


# ---------------------------------------------------------------------------
# Autouse: clear plan_and_notify module-level state before each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_plan_and_notify_state() -> None:  # type: ignore[return]
    import src.scheduler.jobs.plan_and_notify as pn_mod
    pn_mod._in_planning_tickets.clear()
    pn_mod._plan_comment_watermarks.clear()
    yield
    pn_mod._in_planning_tickets.clear()
    pn_mod._plan_comment_watermarks.clear()


# ---------------------------------------------------------------------------
# INT-PL-01 — Job queries both jira and clickup for OPEN_FOR_DEV tickets
# ---------------------------------------------------------------------------


class TestPlanAndNotifyFetch:
    def test_int_pl_01_queries_jira_for_open_for_dev(self) -> None:
        """INT-PL-01a: plan_and_notify_job calls tracker_registry.get('jira')."""
        from src.scheduler.jobs.plan_and_notify import plan_and_notify_job

        mock_tr = _make_tracker_registry()
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit = MagicMock()

        plan_and_notify_job(
            registry=_make_registry(),
            settings=_make_settings(),
            executor=executor,
            workflow=_make_workflow(),
            tracker_registry=mock_tr,
        )

        mock_tr.get.assert_any_call("jira")

    def test_int_pl_01b_queries_clickup_for_open_for_dev(self) -> None:
        """INT-PL-01b: plan_and_notify_job calls tracker_registry.get('clickup')."""
        from src.scheduler.jobs.plan_and_notify import plan_and_notify_job

        mock_tr = _make_tracker_registry()
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit = MagicMock()

        plan_and_notify_job(
            registry=_make_registry(),
            settings=_make_settings(),
            executor=executor,
            workflow=_make_workflow(),
            tracker_registry=mock_tr,
        )

        mock_tr.get.assert_any_call("clickup")

    def test_int_pl_01c_no_workflow_skips_gracefully(self) -> None:
        """INT-PL-01c: No WorkflowConfig → job logs warning and returns without error."""
        from src.scheduler.jobs.plan_and_notify import plan_and_notify_job

        mock_tr = _make_tracker_registry()
        executor = MagicMock(spec=ThreadPoolExecutor)

        plan_and_notify_job(
            registry=_make_registry(),
            settings=_make_settings(),
            executor=executor,
            workflow=None,
            tracker_registry=mock_tr,
        )

        executor.submit.assert_not_called()

    def test_int_pl_01d_no_tracker_registry_skips_gracefully(self) -> None:
        """INT-PL-01d: No TrackerRegistry → job logs warning and returns without error."""
        from src.scheduler.jobs.plan_and_notify import plan_and_notify_job

        executor = MagicMock(spec=ThreadPoolExecutor)
        plan_and_notify_job(
            registry=_make_registry(),
            settings=_make_settings(),
            executor=executor,
            workflow=_make_workflow(),
            tracker_registry=None,
        )

        executor.submit.assert_not_called()

    def test_int_pl_01e_missing_dev_agent_skips(self) -> None:
        """INT-PL-01e: Missing dev_agent in registry → job skips without error."""
        from src.scheduler.jobs.plan_and_notify import plan_and_notify_job

        executor = MagicMock(spec=ThreadPoolExecutor)
        plan_and_notify_job(
            registry=_make_registry(has_dev_agent=False),
            settings=_make_settings(),
            executor=executor,
            workflow=_make_workflow(),
            tracker_registry=_make_tracker_registry(),
        )

        executor.submit.assert_not_called()

    def test_int_pl_01f_rest_fetch_error_does_not_crash(self) -> None:
        """INT-PL-01f: TicketFetchError from one source is logged, other continues."""
        from src.scheduler.jobs.plan_and_notify import plan_and_notify_job

        clickup_ticket = _make_open_ticket("cu-1", "clickup")
        mock_tr = _make_tracker_registry(
            open_error=TicketFetchError("JIRA down", source="jira"),
            open_tickets=[clickup_ticket],
        )
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit = MagicMock()

        # Should not raise even if jira fetch fails
        plan_and_notify_job(
            registry=_make_registry(),
            settings=_make_settings(),
            executor=executor,
            workflow=_make_workflow(),
            tracker_registry=mock_tr,
        )


# ---------------------------------------------------------------------------
# INT-PL-02 — OPEN_FOR_DEV tickets dispatch initial plan crew
# ---------------------------------------------------------------------------


class TestPlanAndNotifyInitialPlan:
    def test_int_pl_02_open_ticket_dispatches_to_executor(self) -> None:
        """INT-PL-02: Each OPEN_FOR_DEV ticket dispatches _create_initial_plan via executor.submit."""
        from src.scheduler.jobs.plan_and_notify import plan_and_notify_job

        tickets = [
            _make_open_ticket("PROJ-1", "jira"),
            _make_open_ticket("PROJ-2", "jira"),
        ]
        mock_tr = _make_tracker_registry(open_tickets=tickets)
        executor = MagicMock(spec=ThreadPoolExecutor)
        future_mock = MagicMock()
        future_mock.exception.return_value = None
        executor.submit = MagicMock(return_value=future_mock)

        plan_and_notify_job(
            registry=_make_registry(),
            settings=_make_settings(),
            executor=executor,
            workflow=_make_workflow(),
            tracker_registry=mock_tr,
        )

        assert executor.submit.call_count == 2

    def test_int_pl_02b_single_open_ticket_dispatched_once(self) -> None:
        """INT-PL-02b: Single OPEN_FOR_DEV ticket → submit called exactly once."""
        from src.scheduler.jobs.plan_and_notify import plan_and_notify_job

        ticket = _make_open_ticket("PROJ-5", "jira")
        mock_tr = _make_tracker_registry(open_tickets=[ticket])
        executor = MagicMock(spec=ThreadPoolExecutor)
        future_mock = MagicMock()
        future_mock.exception.return_value = None
        executor.submit = MagicMock(return_value=future_mock)

        plan_and_notify_job(
            registry=_make_registry(),
            settings=_make_settings(),
            executor=executor,
            workflow=_make_workflow(),
            tracker_registry=mock_tr,
        )

        executor.submit.assert_called_once()


# ---------------------------------------------------------------------------
# INT-PL-03 — Dispatch guard prevents double-planning
# ---------------------------------------------------------------------------


class TestPlanAndNotifyDispatchGuard:
    def test_int_pl_03_dispatch_guard_prevents_double_submit(self) -> None:
        """INT-PL-03: Same ticket already in _in_planning_tickets → no second submit."""
        import src.scheduler.jobs.plan_and_notify as pn_mod
        from src.scheduler.jobs.plan_and_notify import plan_and_notify_job

        ticket = _make_open_ticket("PROJ-99", "jira")
        # Pre-populate the dispatch guard
        pn_mod._in_planning_tickets.add("PROJ-99")

        mock_tr = _make_tracker_registry(open_tickets=[ticket])
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit = MagicMock()

        plan_and_notify_job(
            registry=_make_registry(),
            settings=_make_settings(),
            executor=executor,
            workflow=_make_workflow(),
            tracker_registry=mock_tr,
        )

        executor.submit.assert_not_called()

    def test_int_pl_03b_guard_released_after_worker_completes(self) -> None:
        """INT-PL-03b: Dispatch guard is released in _create_initial_plan finally block."""
        import src.scheduler.jobs.plan_and_notify as pn_mod
        from src.scheduler.jobs.plan_and_notify import _create_initial_plan

        pn_mod._in_planning_tickets.add("PROJ-10")
        registry = _make_registry()

        with patch("src.scheduler.jobs.plan_and_notify.Task"), \
             patch("src.scheduler.jobs.plan_and_notify.CrewBuilder") as mock_cb:
            mock_crew = MagicMock()
            mock_crew.kickoff.return_value = "plan posted"
            mock_cb.build.return_value = mock_crew

            _create_initial_plan("PROJ-10", "jira", "Test ticket", registry)

        assert "PROJ-10" not in pn_mod._in_planning_tickets


# ---------------------------------------------------------------------------
# INT-PL-04 — IN_PLANNING tickets with new comments dispatch revision crew
# ---------------------------------------------------------------------------


class TestPlanAndNotifyRevision:
    def test_int_pl_04_new_comments_dispatch_revision(self) -> None:
        """INT-PL-04: IN_PLANNING ticket with new human comments → revision crew dispatched."""
        from src.scheduler.jobs.plan_and_notify import plan_and_notify_job

        ticket = _make_in_planning_ticket("PROJ-3", "jira")
        new_comment = TicketComment(
            id="c-1",
            author="alice",
            body="Please add migration strategy details.",
            created_at=time.time(),
            source="jira",
        )
        mock_tr = _make_tracker_registry(
            in_planning_tickets=[ticket],
            comments=[new_comment],
        )
        executor = MagicMock(spec=ThreadPoolExecutor)
        future_mock = MagicMock()
        future_mock.exception.return_value = None
        executor.submit = MagicMock(return_value=future_mock)

        plan_and_notify_job(
            registry=_make_registry(),
            settings=_make_settings(),
            executor=executor,
            workflow=_make_workflow(),
            tracker_registry=mock_tr,
        )

        executor.submit.assert_called_once()

    def test_int_pl_04b_watermark_updated_after_revision(self) -> None:
        """INT-PL-04b: _plan_comment_watermarks[ticket_id] updated after revision crew runs."""
        import src.scheduler.jobs.plan_and_notify as pn_mod
        from src.scheduler.jobs.plan_and_notify import _revise_plan

        registry = _make_registry()
        latest_ts = 9999.0

        with patch("src.scheduler.jobs.plan_and_notify.Task"), \
             patch("src.scheduler.jobs.plan_and_notify.CrewBuilder") as mock_cb:
            mock_crew = MagicMock()
            mock_crew.kickoff.return_value = "revised plan posted"
            mock_cb.build.return_value = mock_crew

            _revise_plan(
                ticket_id="PROJ-3",
                source="jira",
                comments_text="[1] alice: Please clarify migration.",
                version=2,
                registry=registry,
                latest_comment_ts=latest_ts,
            )

        assert pn_mod._plan_comment_watermarks.get("PROJ-3") == latest_ts


# ---------------------------------------------------------------------------
# INT-PL-05 — No revision crew when no new comments since watermark
# ---------------------------------------------------------------------------


class TestPlanAndNotifyNoRevision:
    def test_int_pl_05_no_new_comments_skips_revision(self) -> None:
        """INT-PL-05: No comments newer than watermark → no executor.submit."""
        import src.scheduler.jobs.plan_and_notify as pn_mod
        from src.scheduler.jobs.plan_and_notify import plan_and_notify_job

        ticket = _make_in_planning_ticket("PROJ-4", "jira")
        old_comment = TicketComment(
            id="c-old",
            author="bob",
            body="Old feedback",
            created_at=1000.0,
            source="jira",
        )
        # Watermark is AFTER the comment → nothing new
        pn_mod._plan_comment_watermarks["PROJ-4"] = 2000.0

        mock_tr = _make_tracker_registry(
            in_planning_tickets=[ticket],
            comments=[old_comment],
        )
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit = MagicMock()

        plan_and_notify_job(
            registry=_make_registry(),
            settings=_make_settings(),
            executor=executor,
            workflow=_make_workflow(),
            tracker_registry=mock_tr,
        )

        executor.submit.assert_not_called()

    def test_int_pl_05b_no_comments_at_all_skips_revision(self) -> None:
        """INT-PL-05b: IN_PLANNING ticket with no comments → no executor.submit."""
        from src.scheduler.jobs.plan_and_notify import plan_and_notify_job

        ticket = _make_in_planning_ticket("PROJ-5", "jira")
        mock_tr = _make_tracker_registry(
            in_planning_tickets=[ticket],
            comments=[],  # no comments
        )
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit = MagicMock()

        plan_and_notify_job(
            registry=_make_registry(),
            settings=_make_settings(),
            executor=executor,
            workflow=_make_workflow(),
            tracker_registry=mock_tr,
        )

        executor.submit.assert_not_called()

    def test_int_pl_05c_empty_open_for_dev_status_skips_mode1(self) -> None:
        """INT-PL-05c: WorkflowConfig with empty open_for_dev status → Mode 1 skipped gracefully."""
        from src.scheduler.jobs.plan_and_notify import plan_and_notify_job

        # Six-key config: open_for_dev defaults to empty string
        six_key_workflow = WorkflowConfig({
            "scan_for_work": {"status_value": "ACCEPTED", "human_only": True},
            "skip_rejected": {"status_value": "REJECTED"},
            "start_development": {"status_value": "IN PROGRESS"},
            "open_for_review": {"status_value": "IN REVIEW"},
            "mark_complete": {"status_value": "COMPLETE"},
            "update_with_context": {"status_value": ""},
        })
        mock_tr = _make_tracker_registry()
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit = MagicMock()

        # Should not crash; Mode 1 is skipped because status is empty
        plan_and_notify_job(
            registry=_make_registry(),
            settings=_make_settings(),
            executor=executor,
            workflow=six_key_workflow,
            tracker_registry=mock_tr,
        )

        executor.submit.assert_not_called()


# ---------------------------------------------------------------------------
# INT-PL-06 — Task description content (initial plan)
# ---------------------------------------------------------------------------


class TestPlanAndNotifyTaskContent:
    def test_int_pl_06_initial_plan_task_description_has_guardrails(self) -> None:
        """INT-PL-06: Initial plan task description contains BR-8 guardrails."""
        from src.scheduler.jobs.plan_and_notify import _INITIAL_PLAN_TASK_TEMPLATE

        desc = _INITIAL_PLAN_TASK_TEMPLATE.format(ticket_id="PROJ-1", source="jira")
        assert "Do NOT write any code" in desc
        assert "Do NOT open a GitHub Pull Request" in desc
        assert "Do NOT transition" in desc

    def test_int_pl_06b_initial_plan_has_plan_structure(self) -> None:
        """INT-PL-06b: Initial plan template instructs agent to produce structured Markdown."""
        from src.scheduler.jobs.plan_and_notify import _INITIAL_PLAN_TASK_TEMPLATE

        desc = _INITIAL_PLAN_TASK_TEMPLATE.format(ticket_id="PROJ-1", source="jira")
        assert "Development Plan" in desc
        assert "Technical Approach" in desc
        assert "Implementation Steps" in desc

    def test_int_pl_06c_revision_template_has_guardrails(self) -> None:
        """INT-PL-06c: Revision plan task description contains BR-8 guardrails."""
        from src.scheduler.jobs.plan_and_notify import _REVISE_PLAN_TASK_TEMPLATE

        desc = _REVISE_PLAN_TASK_TEMPLATE.format(
            ticket_id="PROJ-2",
            source="jira",
            comments_text="[1] Alice: please clarify",
            version=2,
        )
        assert "Do NOT write any code" in desc
        assert "Do NOT open a GitHub Pull Request" in desc
        assert "Revision Notes" in desc

