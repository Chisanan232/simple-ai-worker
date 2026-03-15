"""
Integration tests for the Dev Lead Slack handler — Feasibility Assessment mode
(INT-DL-01 through INT-DL-06).

Tests cover:
- INT-DL-01: Message with no ticket ID dispatches crew via executor.submit
- INT-DL-02: Task description contains feasibility assessment instructions
- INT-DL-03: Task description has guardrail: no sub-tasks before planner confirms
- INT-DL-04: Task description references jira/get_issue when a JIRA ticket ID present
- INT-DL-05: Existing epic breakdown (type A) still works — not broken by Phase 9
- INT-DL-06: Missing dev_lead agent in registry → say() error, no executor.submit
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_message_event(
    text: str = "[dev lead] We want to add real-time notifications",
    thread_ts: str | None = "100.200",
) -> dict:
    event: dict = {"text": text, "channel": "C001", "ts": "100.200"}
    if thread_ts:
        event["thread_ts"] = thread_ts
    return event


@pytest.fixture
def mock_say() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_executor() -> MagicMock:
    executor = MagicMock(spec=ThreadPoolExecutor)
    executor.submit = MagicMock()
    return executor


@pytest.fixture
def mock_registry_with_dev_lead() -> MagicMock:
    agent = MagicMock()
    agent.name = "dev_lead"
    registry = MagicMock()
    registry.__getitem__ = MagicMock(return_value=agent)
    return registry


@pytest.fixture
def mock_registry_missing_dev_lead() -> MagicMock:
    registry = MagicMock()
    registry.__getitem__ = MagicMock(side_effect=KeyError("dev_lead"))
    return registry


# ---------------------------------------------------------------------------
# INT-DL-01 — Plain requirement message dispatches crew
# ---------------------------------------------------------------------------


class TestDevLeadFeasibilityDispatch:
    def test_int_dl_01_plain_requirement_dispatches_executor(
        self,
        mock_say: MagicMock,
        mock_executor: MagicMock,
        mock_registry_with_dev_lead: MagicMock,
    ) -> None:
        """INT-DL-01: [dev lead] plain requirement → executor.submit called."""
        from src.slack_app.handlers.dev_lead import dev_lead_handler

        event = _make_message_event("[dev lead] We want to add real-time push notifications to the app")
        dev_lead_handler(
            text="[dev lead] We want to add real-time push notifications to the app",
            event=event,
            say=mock_say,
            registry=mock_registry_with_dev_lead,
            executor=mock_executor,
        )

        mock_executor.submit.assert_called_once()

    def test_int_dl_01b_ack_posted_before_submit(
        self,
        mock_say: MagicMock,
        mock_executor: MagicMock,
        mock_registry_with_dev_lead: MagicMock,
    ) -> None:
        """INT-DL-01b: say() acknowledgement is posted before executor.submit."""
        from src.slack_app.handlers.dev_lead import dev_lead_handler

        call_order: list[str] = []
        mock_say.side_effect = lambda **kwargs: call_order.append("say")
        mock_executor.submit.side_effect = lambda *a, **kw: call_order.append("submit")

        event = _make_message_event()
        dev_lead_handler(
            text="[dev lead] new requirement",
            event=event,
            say=mock_say,
            registry=mock_registry_with_dev_lead,
            executor=mock_executor,
        )

        assert "say" in call_order
        assert "submit" in call_order
        assert call_order.index("say") < call_order.index("submit")


# ---------------------------------------------------------------------------
# INT-DL-02 — Task description contains feasibility assessment instructions
# ---------------------------------------------------------------------------


class TestDevLeadFeasibilityTaskDescription:
    def _capture_task_description(
        self,
        message: str,
        mock_say: MagicMock,
        mock_executor: MagicMock,
        mock_registry: MagicMock,
    ) -> str:
        """Run dev_lead_handler with a patched Task and capture the task description."""
        from src.slack_app.handlers.dev_lead import dev_lead_handler

        captured_tasks: list = []

        with (
            patch("src.slack_app.handlers.dev_lead.Task") as mock_task_cls,
            patch("src.slack_app.handlers.dev_lead.CrewBuilder") as mock_cb,
        ):
            mock_crew = MagicMock()
            mock_crew.kickoff.return_value = "OK"
            mock_cb.build.return_value = mock_crew

            def capture_task(*args, **kwargs):
                captured_tasks.append(kwargs.get("description", ""))
                return MagicMock()

            mock_task_cls.side_effect = capture_task

            event = _make_message_event(text=message)
            executor = ThreadPoolExecutor(max_workers=1)
            try:
                dev_lead_handler(
                    text=message,
                    event=event,
                    say=mock_say,
                    registry=mock_registry,
                    executor=executor,
                )
                executor.shutdown(wait=True)
            finally:
                executor.shutdown(wait=False)

        return captured_tasks[0] if captured_tasks else ""

    def test_int_dl_02_feasibility_instructions_in_description(
        self,
        mock_say: MagicMock,
        mock_registry_with_dev_lead: MagicMock,
    ) -> None:
        """INT-DL-02: Task description contains 'Feasibility Assessment' for plain requirement."""
        desc = self._capture_task_description(
            message="[dev lead] We want to add real-time notifications",
            mock_say=mock_say,
            mock_executor=MagicMock(spec=ThreadPoolExecutor),
            mock_registry=mock_registry_with_dev_lead,
        )
        assert "Feasibility Assessment" in desc or "feasibility" in desc.lower()

    def test_int_dl_02b_all_three_request_types_present(
        self,
        mock_say: MagicMock,
        mock_registry_with_dev_lead: MagicMock,
    ) -> None:
        """INT-DL-02b: Task description has all three request type labels."""
        desc = self._capture_task_description(
            message="[dev lead] breakdown PROJ-10",
            mock_say=mock_say,
            mock_executor=MagicMock(spec=ThreadPoolExecutor),
            mock_registry=mock_registry_with_dev_lead,
        )
        assert "**A." in desc or "A. Epic" in desc
        assert "**B." in desc or "B. Ticket" in desc
        assert "**C." in desc or "C. Feasibility" in desc

    def test_int_dl_03_guardrail_no_subtasks_before_confirm(
        self,
        mock_say: MagicMock,
        mock_registry_with_dev_lead: MagicMock,
    ) -> None:
        """INT-DL-03: Task description contains guardrail against premature sub-task creation."""
        desc = self._capture_task_description(
            message="[dev lead] new requirement to assess",
            mock_say=mock_say,
            mock_executor=MagicMock(spec=ThreadPoolExecutor),
            mock_registry=mock_registry_with_dev_lead,
        )
        # Should contain a guardrail about not creating sub-tasks prematurely
        assert "Do NOT create" in desc or "do not create" in desc.lower()

    def test_int_dl_04_get_issue_mentioned_when_ticket_id_context(
        self,
        mock_say: MagicMock,
        mock_registry_with_dev_lead: MagicMock,
    ) -> None:
        """INT-DL-04: Task description references jira/get_issue for reading existing tickets."""
        desc = self._capture_task_description(
            message="[dev lead] PROJ-42 assess feasibility",
            mock_say=mock_say,
            mock_executor=MagicMock(spec=ThreadPoolExecutor),
            mock_registry=mock_registry_with_dev_lead,
        )
        # The template should mention jira/get_issue for reading existing stories
        assert "jira/get_issue" in desc

    def test_int_dl_05_epic_breakdown_still_works(
        self,
        mock_say: MagicMock,
        mock_registry_with_dev_lead: MagicMock,
    ) -> None:
        """INT-DL-05: Epic breakdown request type A instructions are still present."""
        desc = self._capture_task_description(
            message="[dev lead] break down PROJ-10",
            mock_say=mock_say,
            mock_executor=MagicMock(spec=ThreadPoolExecutor),
            mock_registry=mock_registry_with_dev_lead,
        )
        assert "jira/create_issue" in desc or "create_issue" in desc
        assert "clickup/create_task" in desc or "create_task" in desc

    def test_int_dl_05b_dependency_annotation_in_breakdown(
        self,
        mock_say: MagicMock,
        mock_registry_with_dev_lead: MagicMock,
    ) -> None:
        """INT-DL-05b: Breakdown type A includes dependency annotation instructions."""
        desc = self._capture_task_description(
            message="[dev lead] decompose PROJ-20",
            mock_say=mock_say,
            mock_executor=MagicMock(spec=ThreadPoolExecutor),
            mock_registry=mock_registry_with_dev_lead,
        )
        assert "dependenc" in desc.lower()

    def test_int_dl_05c_planner_notification_in_breakdown(
        self,
        mock_say: MagicMock,
        mock_registry_with_dev_lead: MagicMock,
    ) -> None:
        """INT-DL-05c: Breakdown type A includes step to notify planner via ticket comment."""
        desc = self._capture_task_description(
            message="[dev lead] breakdown request",
            mock_say=mock_say,
            mock_executor=MagicMock(spec=ThreadPoolExecutor),
            mock_registry=mock_registry_with_dev_lead,
        )
        assert "add_comment" in desc


# ---------------------------------------------------------------------------
# INT-DL-06 — Missing dev_lead agent → error handling
# ---------------------------------------------------------------------------


class TestDevLeadMissingAgent:
    def test_int_dl_06_missing_agent_posts_error_say(
        self,
        mock_say: MagicMock,
        mock_executor: MagicMock,
        mock_registry_missing_dev_lead: MagicMock,
    ) -> None:
        """INT-DL-06: Missing dev_lead agent → say() called with config error message."""
        from src.slack_app.handlers.dev_lead import dev_lead_handler

        event = _make_message_event()

        with patch("src.slack_app.handlers.dev_lead.Task"), patch("src.slack_app.handlers.dev_lead.CrewBuilder"):
            executor = ThreadPoolExecutor(max_workers=1)
            try:
                dev_lead_handler(
                    text="[dev lead] test",
                    event=event,
                    say=mock_say,
                    registry=mock_registry_missing_dev_lead,
                    executor=executor,
                )
                executor.shutdown(wait=True)
            finally:
                executor.shutdown(wait=False)

        # say() must be called with an error message
        assert mock_say.called
        all_calls = str(mock_say.call_args_list)
        assert "error" in all_calls.lower() or "not available" in all_calls.lower() or "❌" in all_calls

    def test_int_dl_06b_missing_agent_no_crew_kickoff(
        self,
        mock_say: MagicMock,
        mock_executor: MagicMock,
        mock_registry_missing_dev_lead: MagicMock,
    ) -> None:
        """INT-DL-06b: Missing dev_lead agent → no crew kickoff attempted."""
        from src.slack_app.handlers.dev_lead import dev_lead_handler

        event = _make_message_event()

        with patch("src.slack_app.handlers.dev_lead.CrewBuilder") as mock_cb:
            executor = ThreadPoolExecutor(max_workers=1)
            try:
                dev_lead_handler(
                    text="[dev lead] test",
                    event=event,
                    say=mock_say,
                    registry=mock_registry_missing_dev_lead,
                    executor=executor,
                )
                executor.shutdown(wait=True)
            finally:
                executor.shutdown(wait=False)

        mock_cb.build.assert_not_called()


# ---------------------------------------------------------------------------
# INT-DL-07 — Empty message after stripping tag sends hint
# ---------------------------------------------------------------------------


class TestDevLeadEmptyMessage:
    def test_int_dl_07_empty_message_after_tag_strip(
        self,
        mock_say: MagicMock,
        mock_executor: MagicMock,
        mock_registry_with_dev_lead: MagicMock,
    ) -> None:
        """INT-DL-07: '[dev lead]' with no body → say() hint, no executor.submit."""
        from src.slack_app.handlers.dev_lead import dev_lead_handler

        event = _make_message_event(text="[dev lead]")
        dev_lead_handler(
            text="[dev lead]",
            event=event,
            say=mock_say,
            registry=mock_registry_with_dev_lead,
            executor=mock_executor,
        )

        mock_say.assert_called()
        mock_executor.submit.assert_not_called()
