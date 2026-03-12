"""
Unit tests for :func:`src.slack_app.handlers.planner.planner_handler`
and :func:`src.slack_app.handlers.dev_lead.dev_lead_handler`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

from src.slack_app.handlers.dev_lead import dev_lead_handler
from src.slack_app.handlers.planner import planner_handler


def _make_event(ts: str = "111.000", thread_ts: str | None = None) -> dict:
    evt: dict = {"ts": ts}
    if thread_ts is not None:
        evt["thread_ts"] = thread_ts
    return evt


def _make_registry(
    agent: MagicMock | None = None,
    raise_key_error: bool = False,
) -> MagicMock:
    reg = MagicMock()
    if raise_key_error:
        reg.__getitem__.side_effect = KeyError("agent")
        reg.agent_ids.return_value = []
    else:
        reg.__getitem__.return_value = agent or MagicMock()
    return reg


# =============================================================================
# Planner handler tests
# =============================================================================


class TestPlannerHandler:
    """Tests for planner_handler."""

    def test_posts_thinking_acknowledgement_immediately(self) -> None:
        """planner_handler must post the 'thinking…' message before submitting to executor."""
        say = MagicMock()
        registry = _make_registry()
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit.return_value = MagicMock()

        planner_handler(
            text="[planner] Build a checkout flow",
            event=_make_event(),
            say=say,
            registry=registry,
            executor=executor,
        )

        assert say.called
        first_call_text = say.call_args_list[0].kwargs.get("text", "")
        assert "⏳" in first_call_text or "thinking" in first_call_text.lower() or "on it" in first_call_text.lower()

    def test_submits_crew_to_executor(self) -> None:
        """planner_handler must submit the crew execution to the executor."""
        say = MagicMock()
        registry = _make_registry()
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit.return_value = MagicMock()

        planner_handler(
            text="[planner] Login feature",
            event=_make_event(),
            say=say,
            registry=registry,
            executor=executor,
        )

        executor.submit.assert_called_once()

    def test_replies_with_hint_when_message_is_empty(self) -> None:
        """planner_handler must post a hint and NOT submit to executor when body is empty."""
        say = MagicMock()
        registry = _make_registry()
        executor = MagicMock(spec=ThreadPoolExecutor)

        planner_handler(
            text="[planner]",  # no body after the tag
            event=_make_event(),
            say=say,
            registry=registry,
            executor=executor,
        )

        executor.submit.assert_not_called()
        say.assert_called_once()
        assert "planner" in say.call_args.kwargs.get("text", "").lower() or "[planner]" in say.call_args.kwargs.get(
            "text", ""
        )

    def test_strips_planner_tag_before_passing_to_crew(self) -> None:
        """The [planner] tag must be stripped from the message body sent to the crew."""
        say = MagicMock()
        registry = _make_registry()
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit.return_value = MagicMock()

        planner_handler(
            text="[planner] Build the order feature",
            event=_make_event(),
            say=say,
            registry=registry,
            executor=executor,
        )

        # The first positional arg after the callable is the human_message
        submitted_args = executor.submit.call_args.args
        human_message = submitted_args[1]  # second arg after the function
        assert "[planner]" not in human_message
        assert "Build the order feature" in human_message

    def test_thread_ts_from_event_ts_when_no_thread_ts(self) -> None:
        """planner_handler must use event['ts'] as thread_ts when 'thread_ts' is absent."""
        say = MagicMock()
        registry = _make_registry()
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit.return_value = MagicMock()

        planner_handler(
            text="[planner] Feature",
            event=_make_event(ts="999.111"),
            say=say,
            registry=registry,
            executor=executor,
        )

        first_say_kwargs = say.call_args_list[0].kwargs
        assert first_say_kwargs.get("thread_ts") == "999.111"

    def test_thread_ts_from_thread_ts_field_takes_priority(self) -> None:
        """planner_handler must use event['thread_ts'] when it is set."""
        say = MagicMock()
        registry = _make_registry()
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit.return_value = MagicMock()

        planner_handler(
            text="[planner] Feature",
            event=_make_event(ts="999.111", thread_ts="888.000"),
            say=say,
            registry=registry,
            executor=executor,
        )

        first_say_kwargs = say.call_args_list[0].kwargs
        assert first_say_kwargs.get("thread_ts") == "888.000"


# =============================================================================
# Dev Lead handler tests
# =============================================================================


class TestDevLeadHandler:
    """Tests for dev_lead_handler."""

    def test_posts_thinking_acknowledgement_immediately(self) -> None:
        """dev_lead_handler must post the 'thinking…' message before submitting to executor."""
        say = MagicMock()
        registry = _make_registry()
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit.return_value = MagicMock()

        dev_lead_handler(
            text="[dev lead] Break down PROJ-42",
            event=_make_event(),
            say=say,
            registry=registry,
            executor=executor,
        )

        assert say.called
        first_call_text = say.call_args_list[0].kwargs.get("text", "")
        assert "⏳" in first_call_text or "on it" in first_call_text.lower()

    def test_submits_crew_to_executor(self) -> None:
        """dev_lead_handler must submit the crew execution to the executor."""
        say = MagicMock()
        registry = _make_registry()
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit.return_value = MagicMock()

        dev_lead_handler(
            text="[dev lead] Break down PROJ-42",
            event=_make_event(),
            say=say,
            registry=registry,
            executor=executor,
        )

        executor.submit.assert_called_once()

    def test_replies_with_hint_when_message_is_empty(self) -> None:
        """dev_lead_handler must post a hint and NOT submit to executor when body is empty."""
        say = MagicMock()
        registry = _make_registry()
        executor = MagicMock(spec=ThreadPoolExecutor)

        dev_lead_handler(
            text="[dev lead]",
            event=_make_event(),
            say=say,
            registry=registry,
            executor=executor,
        )

        executor.submit.assert_not_called()
        say.assert_called_once()
        assert "dev lead" in say.call_args.kwargs.get("text", "").lower()

    def test_strips_dev_lead_tag_before_passing_to_crew(self) -> None:
        """The [dev lead] tag must be stripped from the message body sent to the crew."""
        say = MagicMock()
        registry = _make_registry()
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit.return_value = MagicMock()

        dev_lead_handler(
            text="[dev lead] Break down PROJ-42 into sub-tasks",
            event=_make_event(),
            say=say,
            registry=registry,
            executor=executor,
        )

        submitted_args = executor.submit.call_args.args
        human_message = submitted_args[1]
        assert "[dev lead]" not in human_message
        assert "Break down PROJ-42 into sub-tasks" in human_message

    def test_thread_ts_from_event_ts_when_no_thread_ts(self) -> None:
        """dev_lead_handler must use event['ts'] as thread_ts when 'thread_ts' is absent."""
        say = MagicMock()
        registry = _make_registry()
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit.return_value = MagicMock()

        dev_lead_handler(
            text="[dev lead] Directive",
            event=_make_event(ts="555.222"),
            say=say,
            registry=registry,
            executor=executor,
        )

        first_say_kwargs = say.call_args_list[0].kwargs
        assert first_say_kwargs.get("thread_ts") == "555.222"

    def test_thread_ts_from_thread_ts_field_takes_priority(self) -> None:
        """dev_lead_handler must use event['thread_ts'] when it is set."""
        say = MagicMock()
        registry = _make_registry()
        executor = MagicMock(spec=ThreadPoolExecutor)
        executor.submit.return_value = MagicMock()

        dev_lead_handler(
            text="[dev lead] Directive",
            event=_make_event(ts="555.222", thread_ts="444.000"),
            say=say,
            registry=registry,
            executor=executor,
        )

        first_say_kwargs = say.call_args_list[0].kwargs
        assert first_say_kwargs.get("thread_ts") == "444.000"


# =============================================================================
# Background crew execution tests (_run_planner_crew, _run_dev_lead_crew)
# =============================================================================


class TestRunPlannerCrew:
    """Tests for _run_planner_crew internal function."""

    def test_posts_error_to_thread_when_agent_missing(self) -> None:
        """_run_planner_crew must say an error message when 'planner' not in registry."""
        from src.slack_app.handlers.planner import _run_planner_crew

        say = MagicMock()
        registry = _make_registry(raise_key_error=True)

        _run_planner_crew(
            human_message="Build feature",
            thread_ts="111.000",
            say=say,
            registry=registry,
        )

        say.assert_called_once()
        assert "❌" in say.call_args.kwargs.get("text", "")

    def test_posts_error_to_thread_on_crew_exception(self) -> None:
        """_run_planner_crew must say an error message when crew.kickoff() raises."""
        from src.slack_app.handlers.planner import _run_planner_crew

        mock_crew = MagicMock()
        mock_crew.kickoff.side_effect = RuntimeError("LLM down")

        say = MagicMock()
        registry = _make_registry()

        with (
            patch("src.slack_app.handlers.planner.Task"),
            patch("src.slack_app.handlers.planner.CrewBuilder") as MockBuilder,
        ):
            MockBuilder.build.return_value = mock_crew
            _run_planner_crew(
                human_message="Build feature",
                thread_ts="111.000",
                say=say,
                registry=registry,
            )

        say.assert_called_once()
        error_text = say.call_args.kwargs.get("text", "")
        assert "❌" in error_text

    def test_does_not_call_say_on_success(self) -> None:
        """_run_planner_crew must NOT call say() on successful crew completion."""
        from src.slack_app.handlers.planner import _run_planner_crew

        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "Crew result"

        say = MagicMock()
        registry = _make_registry()

        with (
            patch("src.slack_app.handlers.planner.Task"),
            patch("src.slack_app.handlers.planner.CrewBuilder") as MockBuilder,
        ):
            MockBuilder.build.return_value = mock_crew
            _run_planner_crew(
                human_message="Build feature",
                thread_ts="111.000",
                say=say,
                registry=registry,
            )

        say.assert_not_called()


class TestRunDevLeadCrew:
    """Tests for _run_dev_lead_crew internal function."""

    def test_posts_error_to_thread_when_agent_missing(self) -> None:
        """_run_dev_lead_crew must say an error message when 'dev_lead' not in registry."""
        from src.slack_app.handlers.dev_lead import _run_dev_lead_crew

        say = MagicMock()
        registry = _make_registry(raise_key_error=True)

        _run_dev_lead_crew(
            human_message="Break down ticket",
            thread_ts="222.000",
            say=say,
            registry=registry,
        )

        say.assert_called_once()
        assert "❌" in say.call_args.kwargs.get("text", "")

    def test_posts_error_to_thread_on_crew_exception(self) -> None:
        """_run_dev_lead_crew must say an error message when crew.kickoff() raises."""
        from src.slack_app.handlers.dev_lead import _run_dev_lead_crew

        mock_crew = MagicMock()
        mock_crew.kickoff.side_effect = RuntimeError("API down")

        say = MagicMock()
        registry = _make_registry()

        with (
            patch("src.slack_app.handlers.dev_lead.Task"),
            patch("src.slack_app.handlers.dev_lead.CrewBuilder") as MockBuilder,
        ):
            MockBuilder.build.return_value = mock_crew
            _run_dev_lead_crew(
                human_message="Break down ticket",
                thread_ts="222.000",
                say=say,
                registry=registry,
            )

        say.assert_called_once()
        error_text = say.call_args.kwargs.get("text", "")
        assert "❌" in error_text

    def test_does_not_call_say_on_success(self) -> None:
        """_run_dev_lead_crew must NOT call say() on successful crew completion."""
        from src.slack_app.handlers.dev_lead import _run_dev_lead_crew

        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "Sub-tasks created."

        say = MagicMock()
        registry = _make_registry()

        with (
            patch("src.slack_app.handlers.dev_lead.Task"),
            patch("src.slack_app.handlers.dev_lead.CrewBuilder") as MockBuilder,
        ):
            MockBuilder.build.return_value = mock_crew
            _run_dev_lead_crew(
                human_message="Break down ticket",
                thread_ts="222.000",
                say=say,
                registry=registry,
            )

        say.assert_not_called()
