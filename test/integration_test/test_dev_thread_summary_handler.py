"""
Integration tests for the [dev] Slack handler (INT-DH-01 through INT-DH-12).

Tests cover:
- Routing: [dev] tag dispatches to dev_handler; no call to planner/dev_lead handlers
- Acknowledgement message posted before executor.submit()
- Top-level mentions (no thread_ts) are rejected with a hint
- Task description content assertions (channel_id, thread_ts, tool names, guardrails)
- Error handling: crew exception → say() error message
- Error handling: dev_agent missing from registry → say() config error
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, call, patch

import pytest

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

def _make_in_thread_event(
    channel_id: str = "C001",
    thread_ts: str = "111.222",
    ts: str = "111.222",
) -> dict:
    return {
        "text": "<@UBOT> [dev] update the ticket",
        "channel": channel_id,
        "thread_ts": thread_ts,
        "ts": ts,
    }


def _make_top_level_event(channel_id: str = "C001", ts: str = "999.000") -> dict:
    return {
        "text": "<@UBOT> [dev] update the ticket",
        "channel": channel_id,
        # No "thread_ts" key — this is a top-level message
        "ts": ts,
    }


@pytest.fixture
def mock_say() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_executor() -> MagicMock:
    executor = MagicMock(spec=ThreadPoolExecutor)
    executor.submit = MagicMock()
    return executor


@pytest.fixture
def mock_registry_with_dev_agent() -> MagicMock:
    agent = MagicMock()
    agent.name = "dev_agent"
    registry = MagicMock()
    registry.__getitem__ = MagicMock(return_value=agent)
    return registry


@pytest.fixture
def mock_registry_missing_dev_agent() -> MagicMock:
    registry = MagicMock()
    registry.__getitem__ = MagicMock(side_effect=KeyError("dev_agent"))
    return registry


# ---------------------------------------------------------------------------
# INT-DH-01 — ACK message posted before executor.submit()
# ---------------------------------------------------------------------------

class TestDevHandlerAcknowledgement:
    def test_posts_ack_before_submit(
        self,
        mock_say: MagicMock,
        mock_executor: MagicMock,
        mock_registry_with_dev_agent: MagicMock,
    ) -> None:
        """INT-DH-01: say('⏳ Reading …') is called before executor.submit()."""
        from src.slack_app.handlers.dev import dev_handler

        event = _make_in_thread_event()
        dev_handler(
            text="[dev] update ticket",
            event=event,
            say=mock_say,
            registry=mock_registry_with_dev_agent,
            executor=mock_executor,
        )

        # say() must have been called at least once for the ACK.
        mock_say.assert_called()
        ack_call = mock_say.call_args_list[0]
        assert "⏳" in str(ack_call) or "Reading" in str(ack_call)

        # executor.submit() must also have been called.
        mock_executor.submit.assert_called_once()

    def test_ack_call_precedes_submit(
        self,
        mock_say: MagicMock,
        mock_executor: MagicMock,
        mock_registry_with_dev_agent: MagicMock,
    ) -> None:
        """The say() ACK call is tracked before executor.submit() is tracked."""
        from src.slack_app.handlers.dev import dev_handler

        call_order: list[str] = []
        mock_say.side_effect = lambda **kwargs: call_order.append("say")
        mock_executor.submit.side_effect = lambda *a, **kw: call_order.append("submit")

        event = _make_in_thread_event()
        dev_handler(
            text="[dev]",
            event=event,
            say=mock_say,
            registry=mock_registry_with_dev_agent,
            executor=mock_executor,
        )

        assert call_order.index("say") < call_order.index("submit")


# ---------------------------------------------------------------------------
# INT-DH-02 — Top-level mention (no thread_ts) → no executor.submit()
# ---------------------------------------------------------------------------

class TestDevHandlerTopLevelRejection:
    def test_rejects_top_level_mention(
        self,
        mock_say: MagicMock,
        mock_executor: MagicMock,
        mock_registry_with_dev_agent: MagicMock,
    ) -> None:
        """INT-DH-02: Top-level mention (no thread_ts) → executor.submit NOT called."""
        from src.slack_app.handlers.dev import dev_handler

        event = _make_top_level_event()
        dev_handler(
            text="[dev] update ticket",
            event=event,
            say=mock_say,
            registry=mock_registry_with_dev_agent,
            executor=mock_executor,
        )

        mock_executor.submit.assert_not_called()
        mock_say.assert_called_once()
        # The hint message should reference existing thread / thread usage.
        hint_text = str(mock_say.call_args)
        assert "thread" in hint_text.lower()


# ---------------------------------------------------------------------------
# INT-DH-03 — [dev] tag routing (via router)
# ---------------------------------------------------------------------------

class TestDevTagRouting:
    def test_dev_tag_routed_to_dev_handler(
        self,
        mock_say: MagicMock,
        mock_executor: MagicMock,
        mock_registry_with_dev_agent: MagicMock,
    ) -> None:
        """INT-DH-03: [dev] in message → dev_handler called; others NOT called."""
        from src.slack_app import router as router_mod

        event = _make_in_thread_event()
        event["text"] = "<@UBOT> [dev] update the ticket"

        with (
            patch.object(router_mod, "dev_handler") as mock_dev,
            patch.object(router_mod, "planner_handler") as mock_planner,
            patch.object(router_mod, "dev_lead_handler") as mock_dev_lead,
        ):
            router_mod.role_router(
                event=event,
                say=mock_say,
                registry=mock_registry_with_dev_agent,
                executor=mock_executor,
            )
            mock_dev.assert_called_once()
            mock_planner.assert_not_called()
            mock_dev_lead.assert_not_called()

    def test_dev_lead_tag_not_matched_by_dev_tag_re(
        self,
        mock_say: MagicMock,
        mock_executor: MagicMock,
        mock_registry_with_dev_agent: MagicMock,
    ) -> None:
        """[dev lead] tag should NOT trigger dev_handler (only dev_lead_handler)."""
        from src.slack_app import router as router_mod

        event = _make_in_thread_event()
        event["text"] = "<@UBOT> [dev lead] assign the task"
        event.pop("thread_ts", None)  # top-level for simplicity

        with (
            patch.object(router_mod, "dev_handler") as mock_dev,
            patch.object(router_mod, "dev_lead_handler") as mock_dev_lead,
        ):
            router_mod.role_router(
                event=event,
                say=mock_say,
                registry=mock_registry_with_dev_agent,
                executor=mock_executor,
            )
            mock_dev.assert_not_called()
            mock_dev_lead.assert_called_once()


# ---------------------------------------------------------------------------
# Task description content assertions (INT-DH-04 through INT-DH-10)
# We capture the Task() that gets constructed inside _run_dev_thread_summary_crew.
# ---------------------------------------------------------------------------

class TestDevTaskDescriptionContent:
    """
    Exercises the task description built by _run_dev_thread_summary_crew.
    We mock crew.kickoff() so no real LLM calls are made.
    """

    def _invoke_background_fn(
        self,
        channel_id: str,
        thread_ts: str,
        mock_registry: MagicMock,
    ) -> "list[str]":
        """Run _run_dev_thread_summary_crew and return all task descriptions seen."""
        from src.slack_app.handlers.dev import _run_dev_thread_summary_crew

        captured_descriptions: list[str] = []

        def fake_kickoff() -> str:
            return "done"

        with patch("src.slack_app.handlers.dev.CrewBuilder") as mock_cb:
            mock_crew = MagicMock()
            mock_crew.kickoff = fake_kickoff
            mock_cb.build.return_value = mock_crew

            with patch("src.slack_app.handlers.dev.Task") as mock_task_cls:
                def capture_task(**kwargs: object) -> MagicMock:
                    captured_descriptions.append(str(kwargs.get("description", "")))
                    t = MagicMock()
                    return t

                mock_task_cls.side_effect = capture_task
                thread_ts_nodot = thread_ts.replace(".", "")
                _run_dev_thread_summary_crew(
                    channel_id=channel_id,
                    thread_ts=thread_ts,
                    thread_ts_nodot=thread_ts_nodot,
                    say=MagicMock(),
                    registry=mock_registry,
                )

        return captured_descriptions

    def test_task_contains_channel_and_thread_ts(
        self, mock_registry_with_dev_agent: MagicMock
    ) -> None:
        """INT-DH-04: Task description contains channel_id and thread_ts."""
        descs = self._invoke_background_fn("C999", "111.222", mock_registry_with_dev_agent)
        assert any("C999" in d for d in descs), "channel_id 'C999' not in task description"
        assert any("111.222" in d for d in descs), "thread_ts '111.222' not in task description"

    def test_task_instructs_get_messages(
        self, mock_registry_with_dev_agent: MagicMock
    ) -> None:
        """INT-DH-05: Task description instructs slack/get_messages."""
        descs = self._invoke_background_fn("C001", "100.200", mock_registry_with_dev_agent)
        assert any("slack/get_messages" in d for d in descs), "slack/get_messages not in task description"

    def test_task_instructs_add_comment(
        self, mock_registry_with_dev_agent: MagicMock
    ) -> None:
        """INT-DH-06: Task description references jira/add_comment or clickup/add_comment."""
        descs = self._invoke_background_fn("C001", "100.200", mock_registry_with_dev_agent)
        assert any(
            "jira/add_comment" in d or "clickup/add_comment" in d
            for d in descs
        ), "Neither jira/add_comment nor clickup/add_comment in task description"

    def test_task_contains_slack_permalink_format(
        self, mock_registry_with_dev_agent: MagicMock
    ) -> None:
        """INT-DH-07: Task description includes slack.com/archives permalink format."""
        descs = self._invoke_background_fn("C123", "555.666", mock_registry_with_dev_agent)
        assert any("slack.com/archives" in d for d in descs), "Slack permalink format not in task description"

    def test_task_instructs_ask_and_stop_when_no_ticket_id(
        self, mock_registry_with_dev_agent: MagicMock
    ) -> None:
        """INT-DH-08: Task description instructs ask + STOP if no ticket ID (BR-6)."""
        descs = self._invoke_background_fn("C001", "100.200", mock_registry_with_dev_agent)
        full_text = " ".join(descs).upper()
        assert "STOP" in full_text, "Task description must instruct STOP when no ticket ID found (BR-6)"

    def test_task_forbids_creating_new_ticket(
        self, mock_registry_with_dev_agent: MagicMock
    ) -> None:
        """INT-DH-09: Task description contains 'Do NOT create any new ticket'."""
        descs = self._invoke_background_fn("C001", "100.200", mock_registry_with_dev_agent)
        full_text = " ".join(descs)
        assert "Do NOT create" in full_text or "do not create" in full_text.lower(), (
            "Task description must forbid ticket creation"
        )

    def test_task_forbids_state_transition(
        self, mock_registry_with_dev_agent: MagicMock
    ) -> None:
        """INT-DH-10: Task description contains 'Do NOT transition the ticket state'."""
        descs = self._invoke_background_fn("C001", "100.200", mock_registry_with_dev_agent)
        full_text = " ".join(descs)
        assert "transition" in full_text.lower() and (
            "do not" in full_text.lower() or "NOT" in full_text
        ), "Task description must forbid ticket state transitions"


# ---------------------------------------------------------------------------
# INT-DH-11 — Crew exception → say() error message
# ---------------------------------------------------------------------------

class TestDevHandlerErrorHandling:
    def test_posts_error_on_crew_exception(
        self,
        mock_registry_with_dev_agent: MagicMock,
    ) -> None:
        """INT-DH-11: crew.kickoff() raises → say() called with ❌ error message."""
        from src.slack_app.handlers.dev import _run_dev_thread_summary_crew

        mock_say = MagicMock()

        with (
            patch("src.slack_app.handlers.dev.Task") as mock_task_cls,
            patch("src.slack_app.handlers.dev.CrewBuilder") as mock_cb,
        ):
            mock_task_cls.return_value = MagicMock()
            mock_crew = MagicMock()
            mock_crew.kickoff.side_effect = RuntimeError("LLM timeout")
            mock_cb.build.return_value = mock_crew

            _run_dev_thread_summary_crew(
                channel_id="C001",
                thread_ts="100.200",
                thread_ts_nodot="100200",
                say=mock_say,
                registry=mock_registry_with_dev_agent,
            )

        mock_say.assert_called_once()
        call_text = str(mock_say.call_args)
        assert "❌" in call_text or "error" in call_text.lower()

    def test_posts_error_when_dev_agent_not_in_registry(
        self,
        mock_registry_missing_dev_agent: MagicMock,
    ) -> None:
        """INT-DH-12: dev_agent not in registry → say() called with config error."""
        from src.slack_app.handlers.dev import _run_dev_thread_summary_crew

        mock_say = MagicMock()

        _run_dev_thread_summary_crew(
            channel_id="C001",
            thread_ts="100.200",
            thread_ts_nodot="100200",
            say=mock_say,
            registry=mock_registry_missing_dev_agent,
        )

        mock_say.assert_called_once()
        call_text = str(mock_say.call_args)
        assert "❌" in call_text or "error" in call_text.lower() or "Configuration" in call_text


