"""
Unit tests for :func:`src.slack_app.router.role_router`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, call, patch

import pytest

from src.slack_app.router import role_router


def _make_event(text: str, ts: str = "1234567890.000001", channel_type: str = "channel") -> dict:
    return {"text": text, "ts": ts, "channel_type": channel_type}


def _make_registry() -> MagicMock:
    reg = MagicMock()
    reg.__getitem__.return_value = MagicMock()
    reg.agent_ids.return_value = ["planner", "dev_lead", "dev_agent"]
    return reg


class TestRoleRouter:
    """Tests for role_router."""

    def test_dispatches_planner_tag_to_planner_handler(self) -> None:
        """[planner] tag must call planner_handler, not dev_lead_handler."""
        event = _make_event("[planner] Build a login feature")
        say = MagicMock()
        registry = _make_registry()
        executor = MagicMock(spec=ThreadPoolExecutor)

        with (
            patch("src.slack_app.router.planner_handler") as mock_planner,
            patch("src.slack_app.router.dev_lead_handler") as mock_dev_lead,
        ):
            role_router(event=event, say=say, registry=registry, executor=executor)

        mock_planner.assert_called_once()
        mock_dev_lead.assert_not_called()

    def test_dispatches_dev_lead_tag_to_dev_lead_handler(self) -> None:
        """[dev lead] tag must call dev_lead_handler, not planner_handler."""
        event = _make_event("[dev lead] Break down PROJ-42")
        say = MagicMock()
        registry = _make_registry()
        executor = MagicMock(spec=ThreadPoolExecutor)

        with (
            patch("src.slack_app.router.planner_handler") as mock_planner,
            patch("src.slack_app.router.dev_lead_handler") as mock_dev_lead,
        ):
            role_router(event=event, say=say, registry=registry, executor=executor)

        mock_dev_lead.assert_called_once()
        mock_planner.assert_not_called()

    def test_sends_usage_hint_when_no_tag_present(self) -> None:
        """No recognised tag must result in a usage hint via say()."""
        event = _make_event("Hello, are you there?")
        say = MagicMock()
        registry = _make_registry()
        executor = MagicMock(spec=ThreadPoolExecutor)

        with (
            patch("src.slack_app.router.planner_handler") as mock_planner,
            patch("src.slack_app.router.dev_lead_handler") as mock_dev_lead,
        ):
            role_router(event=event, say=say, registry=registry, executor=executor)

        say.assert_called_once()
        mock_planner.assert_not_called()
        mock_dev_lead.assert_not_called()

    def test_tag_matching_is_case_insensitive(self) -> None:
        """[PLANNER] and [DEV LEAD] (uppercase) must be matched the same way."""
        for text, expected_handler in [
            ("[PLANNER] Feature request", "planner"),
            ("[DEV LEAD] Task decomposition", "dev_lead"),
        ]:
            event = _make_event(text)
            say = MagicMock()
            registry = _make_registry()
            executor = MagicMock(spec=ThreadPoolExecutor)

            with (
                patch("src.slack_app.router.planner_handler") as mock_planner,
                patch("src.slack_app.router.dev_lead_handler") as mock_dev_lead,
            ):
                role_router(event=event, say=say, registry=registry, executor=executor)

            if expected_handler == "planner":
                mock_planner.assert_called_once()
            else:
                mock_dev_lead.assert_called_once()

    def test_mention_token_stripped_before_tag_detection(self) -> None:
        """<@BOTID> mention tokens must be stripped before checking for role tags."""
        event = _make_event("<@U12345678> [planner] Build the checkout flow")
        say = MagicMock()
        registry = _make_registry()
        executor = MagicMock(spec=ThreadPoolExecutor)

        with (
            patch("src.slack_app.router.planner_handler") as mock_planner,
            patch("src.slack_app.router.dev_lead_handler"),
        ):
            role_router(event=event, say=say, registry=registry, executor=executor)

        mock_planner.assert_called_once()

    def test_planner_handler_receives_correct_kwargs(self) -> None:
        """planner_handler must be called with text, event, say, registry, executor."""
        event = _make_event("[planner] Build feature X", ts="111.111")
        say = MagicMock()
        registry = _make_registry()
        executor = MagicMock(spec=ThreadPoolExecutor)

        with patch("src.slack_app.router.planner_handler") as mock_planner:
            role_router(event=event, say=say, registry=registry, executor=executor)

        call_kwargs = mock_planner.call_args.kwargs
        assert "text" in call_kwargs
        assert "event" in call_kwargs
        assert call_kwargs["say"] is say
        assert call_kwargs["registry"] is registry
        assert call_kwargs["executor"] is executor

    def test_dev_lead_handler_receives_correct_kwargs(self) -> None:
        """dev_lead_handler must be called with text, event, say, registry, executor."""
        event = _make_event("[dev lead] Break down PROJ-10", ts="222.222")
        say = MagicMock()
        registry = _make_registry()
        executor = MagicMock(spec=ThreadPoolExecutor)

        with patch("src.slack_app.router.dev_lead_handler") as mock_dev_lead:
            role_router(event=event, say=say, registry=registry, executor=executor)

        call_kwargs = mock_dev_lead.call_args.kwargs
        assert "text" in call_kwargs
        assert "event" in call_kwargs
        assert call_kwargs["say"] is say
        assert call_kwargs["registry"] is registry
        assert call_kwargs["executor"] is executor

    def test_usage_hint_posted_with_thread_ts(self) -> None:
        """Usage hint must be posted as thread_ts=event['ts'] so it appears as a reply."""
        event = _make_event("random message", ts="999.000")
        say = MagicMock()
        registry = _make_registry()
        executor = MagicMock(spec=ThreadPoolExecutor)

        with (
            patch("src.slack_app.router.planner_handler"),
            patch("src.slack_app.router.dev_lead_handler"),
        ):
            role_router(event=event, say=say, registry=registry, executor=executor)

        say_kwargs = say.call_args.kwargs
        assert say_kwargs.get("thread_ts") == "999.000"

    def test_empty_text_field_sends_usage_hint(self) -> None:
        """An event with empty text must trigger the usage hint, not a crash."""
        event = _make_event("")
        say = MagicMock()
        registry = _make_registry()
        executor = MagicMock(spec=ThreadPoolExecutor)

        with (
            patch("src.slack_app.router.planner_handler"),
            patch("src.slack_app.router.dev_lead_handler"),
        ):
            role_router(event=event, say=say, registry=registry, executor=executor)

        say.assert_called_once()

