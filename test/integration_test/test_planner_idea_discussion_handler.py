"""
Integration tests for the Planner Slack handler — Idea-Discussion Workflow
(INT-PI-01 through INT-PI-15).

Tests cover:
- INT-PI-01: [planner] message with idea description dispatches crew via executor.submit
- INT-PI-02: Acknowledgement message posted via say() before executor.submit
- INT-PI-03: Empty message after stripping [planner] tag → say() hint, no executor.submit
- INT-PI-04: Missing planner agent in registry → say() error, no executor.submit
- INT-PI-05: Task description contains Type B "Idea Survey" instructions
- INT-PI-06: Task description instructs agent to call slack/get_messages for thread context
- INT-PI-07: Task description covers the 8 survey dimensions (marketing, MVP, budget, etc.)
- INT-PI-08: Task description contains guardrail: no ticket creation during survey/discussion
- INT-PI-09: Task description still contains original Type A epic-creation instructions
- INT-PI-10: Task description contains Type C "Conclusion" instructions
- INT-PI-11: Task description has accept path: OPEN-status ticket + Dev Lead hand-off
- INT-PI-12: Task description has reject path: REJECTED-status ticket
- INT-PI-13: Task description has guardrail: reject path does NOT mention Dev Lead
- INT-PI-14: Task description instructs to use slack/send_message for Dev Lead hand-off
- INT-PI-15: Task description instructs to post conclusion message before creating tickets
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(
    text: str = "[planner] I have a new product idea",
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
def mock_registry_with_planner() -> MagicMock:
    agent = MagicMock()
    agent.name = "planner"
    registry = MagicMock()
    registry.__getitem__ = MagicMock(return_value=agent)
    return registry


@pytest.fixture
def mock_registry_missing_planner() -> MagicMock:
    registry = MagicMock()
    registry.__getitem__ = MagicMock(side_effect=KeyError("planner"))
    return registry


def _capture_task_description(
    message: str,
    mock_say: MagicMock,
    mock_registry: MagicMock,
    thread_ts: str | None = "100.200",
) -> str:
    """Run planner_handler with real ThreadPoolExecutor and capture the task description."""
    from src.slack_app.handlers.planner import planner_handler

    captured_descriptions: list[str] = []

    with (
        patch("src.slack_app.handlers.planner.Task") as mock_task_cls,
        patch("src.slack_app.handlers.planner.CrewBuilder") as mock_cb,
    ):
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "OK"
        mock_cb.build.return_value = mock_crew

        def capture_task(*args, **kwargs):
            captured_descriptions.append(kwargs.get("description", ""))
            return MagicMock()

        mock_task_cls.side_effect = capture_task

        event = _make_event(text=message, thread_ts=thread_ts)
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            planner_handler(
                text=message,
                event=event,
                say=mock_say,
                registry=mock_registry,
                executor=executor,
            )
            executor.shutdown(wait=True)
        finally:
            executor.shutdown(wait=False)

    return captured_descriptions[0] if captured_descriptions else ""


# ---------------------------------------------------------------------------
# INT-PI-01 through INT-PI-04 — Dispatch behaviour
# ---------------------------------------------------------------------------


class TestPlannerDispatch:
    def test_int_pi_01_idea_message_dispatches_executor(
        self,
        mock_say: MagicMock,
        mock_executor: MagicMock,
        mock_registry_with_planner: MagicMock,
    ) -> None:
        """INT-PI-01: [planner] message with idea → executor.submit called once."""
        from src.slack_app.handlers.planner import planner_handler

        event = _make_event("[planner] I want to build a B2B SaaS for restaurant inventory")
        planner_handler(
            text="[planner] I want to build a B2B SaaS for restaurant inventory",
            event=event,
            say=mock_say,
            registry=mock_registry_with_planner,
            executor=mock_executor,
        )

        mock_executor.submit.assert_called_once()

    def test_int_pi_02_ack_posted_before_submit(
        self,
        mock_say: MagicMock,
        mock_executor: MagicMock,
        mock_registry_with_planner: MagicMock,
    ) -> None:
        """INT-PI-02: say() acknowledgement is posted before executor.submit."""
        from src.slack_app.handlers.planner import planner_handler

        call_order: list[str] = []
        mock_say.side_effect = lambda **kwargs: call_order.append("say")
        mock_executor.submit.side_effect = lambda *a, **kw: call_order.append("submit")

        event = _make_event()
        planner_handler(
            text="[planner] new idea",
            event=event,
            say=mock_say,
            registry=mock_registry_with_planner,
            executor=mock_executor,
        )

        assert "say" in call_order
        assert "submit" in call_order
        assert call_order.index("say") < call_order.index("submit")

    def test_int_pi_03_empty_message_after_tag_strip(
        self,
        mock_say: MagicMock,
        mock_executor: MagicMock,
        mock_registry_with_planner: MagicMock,
    ) -> None:
        """INT-PI-03: '[planner]' with no body → say() hint, no executor.submit."""
        from src.slack_app.handlers.planner import planner_handler

        event = _make_event(text="[planner]")
        planner_handler(
            text="[planner]",
            event=event,
            say=mock_say,
            registry=mock_registry_with_planner,
            executor=mock_executor,
        )

        mock_say.assert_called()
        mock_executor.submit.assert_not_called()

    def test_int_pi_03b_whitespace_only_message_after_tag(
        self,
        mock_say: MagicMock,
        mock_executor: MagicMock,
        mock_registry_with_planner: MagicMock,
    ) -> None:
        """INT-PI-03b: '[planner]   ' with whitespace only → say() hint, no submit."""
        from src.slack_app.handlers.planner import planner_handler

        event = _make_event(text="[planner]   ")
        planner_handler(
            text="[planner]   ",
            event=event,
            say=mock_say,
            registry=mock_registry_with_planner,
            executor=mock_executor,
        )

        mock_say.assert_called()
        mock_executor.submit.assert_not_called()

    def test_int_pi_04_missing_planner_agent_posts_error(
        self,
        mock_say: MagicMock,
        mock_executor: MagicMock,
        mock_registry_missing_planner: MagicMock,
    ) -> None:
        """INT-PI-04: Missing planner agent → say() error message, no executor.submit."""
        from src.slack_app.handlers.planner import planner_handler

        event = _make_event()

        with (
            patch("src.slack_app.handlers.planner.Task"),
            patch("src.slack_app.handlers.planner.CrewBuilder"),
        ):
            executor = ThreadPoolExecutor(max_workers=1)
            try:
                planner_handler(
                    text="[planner] some idea",
                    event=event,
                    say=mock_say,
                    registry=mock_registry_missing_planner,
                    executor=executor,
                )
                executor.shutdown(wait=True)
            finally:
                executor.shutdown(wait=False)

        # say() must have been called with an error message
        assert mock_say.called
        all_calls_str = str(mock_say.call_args_list)
        assert "error" in all_calls_str.lower() or "not available" in all_calls_str.lower() or "❌" in all_calls_str

    def test_int_pi_04b_missing_planner_no_crew_built(
        self,
        mock_say: MagicMock,
        mock_executor: MagicMock,
        mock_registry_missing_planner: MagicMock,
    ) -> None:
        """INT-PI-04b: Missing planner agent → no crew kickoff attempted."""
        from src.slack_app.handlers.planner import planner_handler

        event = _make_event()
        with patch("src.slack_app.handlers.planner.CrewBuilder") as mock_cb:
            executor = ThreadPoolExecutor(max_workers=1)
            try:
                planner_handler(
                    text="[planner] some idea",
                    event=event,
                    say=mock_say,
                    registry=mock_registry_missing_planner,
                    executor=executor,
                )
                executor.shutdown(wait=True)
            finally:
                executor.shutdown(wait=False)

        mock_cb.build.assert_not_called()


# ---------------------------------------------------------------------------
# INT-PI-05 through INT-PI-09 — Task description: Idea Survey Mode (Type B)
# ---------------------------------------------------------------------------


class TestPlannerIdeaSurveyTaskDescription:
    def test_int_pi_05_survey_instructions_present(
        self,
        mock_say: MagicMock,
        mock_registry_with_planner: MagicMock,
    ) -> None:
        """INT-PI-05: Task description contains Type B Idea Survey instructions."""
        desc = _capture_task_description(
            message="[planner] I want to build a restaurant inventory SaaS",
            mock_say=mock_say,
            mock_registry=mock_registry_with_planner,
        )
        assert (
            "Idea Survey" in desc or "survey" in desc.lower()
        ), f"Expected 'Idea Survey' or 'survey' in task description. Got:\n{desc[:500]}"

    def test_int_pi_06_get_messages_instruction_present(
        self,
        mock_say: MagicMock,
        mock_registry_with_planner: MagicMock,
    ) -> None:
        """INT-PI-06: Task description instructs agent to call slack/get_messages."""
        desc = _capture_task_description(
            message="[planner] new product idea to discuss",
            mock_say=mock_say,
            mock_registry=mock_registry_with_planner,
        )
        assert "get_messages" in desc, (
            f"Expected 'get_messages' in task description for thread context reading. " f"Got:\n{desc[:500]}"
        )

    def test_int_pi_07_all_survey_dimensions_covered(
        self,
        mock_say: MagicMock,
        mock_registry_with_planner: MagicMock,
    ) -> None:
        """INT-PI-07: Task description covers the required survey dimensions."""
        desc = _capture_task_description(
            message="[planner] I want to build a new B2B product",
            mock_say=mock_say,
            mock_registry=mock_registry_with_planner,
        )
        desc_lower = desc.lower()
        # Check for at least 6 of 8 key survey dimension terms
        dimension_hits = sum(
            [
                "marketing" in desc_lower,
                "market scope" in desc_lower or "market size" in desc_lower,
                "business model" in desc_lower,
                "target audience" in desc_lower or "audience" in desc_lower,
                "pain" in desc_lower,
                "mvp" in desc_lower,
                "implementation" in desc_lower or "implement" in desc_lower,
                "budget" in desc_lower,
            ]
        )
        assert dimension_hits >= 6, (
            f"Expected at least 6 survey dimensions covered in task description. "
            f"Only found {dimension_hits}. Description snippet:\n{desc[:800]}"
        )

    def test_int_pi_08_guardrail_no_tickets_during_survey(
        self,
        mock_say: MagicMock,
        mock_registry_with_planner: MagicMock,
    ) -> None:
        """INT-PI-08: Task description contains guardrail against ticket creation in survey mode."""
        desc = _capture_task_description(
            message="[planner] exploratory product idea",
            mock_say=mock_say,
            mock_registry=mock_registry_with_planner,
        )
        desc_lower = desc.lower()
        has_guardrail = "do not create" in desc_lower or "do NOT create" in desc or "br-11" in desc_lower
        assert has_guardrail, (
            f"Expected guardrail against ticket creation during survey mode. " f"Description snippet:\n{desc[:600]}"
        )

    def test_int_pi_09_type_a_instructions_still_present(
        self,
        mock_say: MagicMock,
        mock_registry_with_planner: MagicMock,
    ) -> None:
        """INT-PI-09: Type A epic-creation instructions are still present (not broken)."""
        desc = _capture_task_description(
            message="[planner] Build an invoice export PDF feature",
            mock_say=mock_say,
            mock_registry=mock_registry_with_planner,
        )
        assert "jira/create_issue" in desc, (
            f"Expected 'jira/create_issue' to still be in task description (Type A). "
            f"Description snippet:\n{desc[:500]}"
        )
        assert "clickup/create_task" in desc, (
            f"Expected 'clickup/create_task' to still be in task description (Type A). "
            f"Description snippet:\n{desc[:500]}"
        )

    def test_int_pi_09b_all_three_request_types_present(
        self,
        mock_say: MagicMock,
        mock_registry_with_planner: MagicMock,
    ) -> None:
        """INT-PI-09b: Task description has all three request type labels."""
        desc = _capture_task_description(
            message="[planner] some message",
            mock_say=mock_say,
            mock_registry=mock_registry_with_planner,
        )
        assert "Type A" in desc or "Request Type A" in desc, "Missing Type A block"
        assert "Type B" in desc or "Request Type B" in desc, "Missing Type B block"
        assert "Type C" in desc or "Request Type C" in desc, "Missing Type C block"

    def test_int_pi_09c_thread_ts_injected_into_description(
        self,
        mock_say: MagicMock,
        mock_registry_with_planner: MagicMock,
    ) -> None:
        """INT-PI-09c: thread_ts value is embedded in the task description."""
        desc = _capture_task_description(
            message="[planner] test message",
            mock_say=mock_say,
            mock_registry=mock_registry_with_planner,
            thread_ts="999.111",
        )
        assert "999.111" in desc, f"Expected thread_ts '999.111' to appear in task description. " f"Got:\n{desc[:400]}"

    def test_int_pi_09d_idea_survey_plan_heading_present(
        self,
        mock_say: MagicMock,
        mock_registry_with_planner: MagicMock,
    ) -> None:
        """INT-PI-09d: Task description contains the '📋 Idea Survey Plan' heading."""
        desc = _capture_task_description(
            message="[planner] explore new market opportunity",
            mock_say=mock_say,
            mock_registry=mock_registry_with_planner,
        )
        assert "Idea Survey Plan" in desc, (
            f"Expected '📋 Idea Survey Plan' heading in task description. " f"Got:\n{desc[:500]}"
        )


# ---------------------------------------------------------------------------
# INT-PI-10 through INT-PI-15 — Task description: Conclusion Mode (Type C)
# ---------------------------------------------------------------------------


class TestPlannerConclusionTaskDescription:
    def test_int_pi_10_conclusion_instructions_present(
        self,
        mock_say: MagicMock,
        mock_registry_with_planner: MagicMock,
    ) -> None:
        """INT-PI-10: Task description contains Type C Conclusion instructions."""
        desc = _capture_task_description(
            message="[planner] let's proceed with the MVP",
            mock_say=mock_say,
            mock_registry=mock_registry_with_planner,
        )
        desc_lower = desc.lower()
        assert "conclusion" in desc_lower, f"Expected 'conclusion' in task description. Got:\n{desc[:500]}"

    def test_int_pi_11_accept_path_has_open_status_and_dev_lead(
        self,
        mock_say: MagicMock,
        mock_registry_with_planner: MagicMock,
    ) -> None:
        """INT-PI-11: Accept path instructs OPEN-status ticket + Dev Lead hand-off."""
        desc = _capture_task_description(
            message="[planner] LGTM, let's go ahead",
            mock_say=mock_say,
            mock_registry=mock_registry_with_planner,
        )
        assert '"OPEN"' in desc or "'OPEN'" in desc or "OPEN" in desc, (
            f"Expected 'OPEN' status in task description accept path. " f"Got:\n{desc[:500]}"
        )
        desc_lower = desc.lower()
        assert (
            "dev lead" in desc_lower or "[dev lead]" in desc_lower
        ), f"Expected 'dev lead' mention in accept path. Got:\n{desc[:500]}"

    def test_int_pi_12_reject_path_has_rejected_status(
        self,
        mock_say: MagicMock,
        mock_registry_with_planner: MagicMock,
    ) -> None:
        """INT-PI-12: Reject path instructs creation of REJECTED-status ticket."""
        desc = _capture_task_description(
            message="[planner] let's drop this idea",
            mock_say=mock_say,
            mock_registry=mock_registry_with_planner,
        )
        assert "REJECTED" in desc, f"Expected 'REJECTED' status in task description reject path. " f"Got:\n{desc[:500]}"

    def test_int_pi_13_reject_path_guardrail_no_dev_lead(
        self,
        mock_say: MagicMock,
        mock_registry_with_planner: MagicMock,
    ) -> None:
        """INT-PI-13: Reject path guardrail: Dev Lead NOT mentioned in reject path."""
        desc = _capture_task_description(
            message="[planner] cancel this",
            mock_say=mock_say,
            mock_registry=mock_registry_with_planner,
        )
        # The description should contain a guardrail that says NOT to notify Dev Lead on reject
        desc_lower = desc.lower()
        has_reject_guard = (
            "do not mention" in desc_lower
            or "br-12" in desc_lower
            or "not mention" in desc_lower
            or ("reject" in desc_lower and "dev lead" in desc_lower and "not" in desc_lower)
        )
        assert has_reject_guard, (
            f"Expected guardrail preventing Dev Lead mention on reject path. " f"Description snippet:\n{desc[:800]}"
        )

    def test_int_pi_14_accept_path_uses_send_message_not_reply_to_thread(
        self,
        mock_say: MagicMock,
        mock_registry_with_planner: MagicMock,
    ) -> None:
        """INT-PI-14: Accept path uses slack/send_message for Dev Lead hand-off (not reply_to_thread)."""
        desc = _capture_task_description(
            message="[planner] approved! go ahead",
            mock_say=mock_say,
            mock_registry=mock_registry_with_planner,
        )
        assert "slack/send_message" in desc or "send_message" in desc, (
            f"Expected 'slack/send_message' in accept path for Dev Lead hand-off (BR-13). "
            f"Description snippet:\n{desc[:600]}"
        )

    def test_int_pi_15_conclusion_message_before_ticket_creation(
        self,
        mock_say: MagicMock,
        mock_registry_with_planner: MagicMock,
    ) -> None:
        """INT-PI-15: In the Type C block, conclusion message step appears before ticket creation."""
        desc = _capture_task_description(
            message="[planner] let's proceed with this",
            mock_say=mock_say,
            mock_registry=mock_registry_with_planner,
        )
        # Locate the start of the Type C section to do relative ordering within it.
        type_c_idx = desc.find("Request Type C")
        assert type_c_idx != -1, "Expected 'Request Type C' block in task description"

        type_c_section = desc[type_c_idx:]

        # "conclusion" (the Post conclusion step) must appear before "create_issue"
        # within the Type C section.
        conclusion_idx = type_c_section.lower().find("conclusion")
        create_issue_idx = type_c_section.find("create_issue")

        assert conclusion_idx != -1, "Expected 'conclusion' step in Type C section"
        assert create_issue_idx != -1, "Expected 'create_issue' step in Type C section"
        assert conclusion_idx < create_issue_idx, (
            f"Expected conclusion message step (idx={conclusion_idx}) to appear "
            f"before create_issue (idx={create_issue_idx}) within the Type C section"
        )

    def test_int_pi_15b_br1_guardrail_never_set_accepted(
        self,
        mock_say: MagicMock,
        mock_registry_with_planner: MagicMock,
    ) -> None:
        """INT-PI-15b: Task description contains BR-1 guardrail (never set ACCEPTED)."""
        desc = _capture_task_description(
            message="[planner] proceed with MVP",
            mock_say=mock_say,
            mock_registry=mock_registry_with_planner,
        )
        desc_lower = desc.lower()
        has_br1 = (
            "br-1" in desc_lower
            or ("never set" in desc_lower and "accepted" in desc_lower)
            or ("accepted" in desc_lower and "human-only" in desc_lower)
            or ("accepted" in desc_lower and "human only" in desc_lower)
        )
        assert has_br1, (
            f"Expected BR-1 guardrail (never set ACCEPTED) in task description. " f"Description snippet:\n{desc[:600]}"
        )

    def test_int_pi_15c_reject_path_has_reply_to_thread_step(
        self,
        mock_say: MagicMock,
        mock_registry_with_planner: MagicMock,
    ) -> None:
        """INT-PI-15c: Reject path includes reply_to_thread for conclusion message."""
        desc = _capture_task_description(
            message="[planner] drop the idea",
            mock_say=mock_say,
            mock_registry=mock_registry_with_planner,
        )
        assert "reply_to_thread" in desc, (
            f"Expected 'reply_to_thread' in task description for conclusion message. "
            f"Description snippet:\n{desc[:500]}"
        )
