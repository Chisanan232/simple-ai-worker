"""
Unit tests for :func:`src.slack_app.app.create_bolt_app`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from src.slack_app.app import create_bolt_app


def _make_settings(
    bot_token: str | None = "xoxb-test",
    signing_secret: str | None = "secret-test",
) -> MagicMock:
    s = MagicMock()

    def _make_secret(value: str | None) -> MagicMock | None:
        if value is None:
            return None
        m = MagicMock()
        m.get_secret_value.return_value = value
        return m

    s.SLACK_BOT_TOKEN = _make_secret(bot_token)
    s.SLACK_SIGNING_SECRET = _make_secret(signing_secret)
    return s


class TestCreateBoltApp:
    """Tests for create_bolt_app factory."""

    def test_raises_value_error_when_bot_token_missing(self) -> None:
        """create_bolt_app must raise ValueError when SLACK_BOT_TOKEN is None."""
        settings = _make_settings(bot_token=None)
        registry = MagicMock()
        executor = MagicMock(spec=ThreadPoolExecutor)

        with pytest.raises(ValueError, match="SLACK_BOT_TOKEN"):
            create_bolt_app(settings=settings, registry=registry, executor=executor)

    def test_raises_value_error_when_signing_secret_missing(self) -> None:
        """create_bolt_app must raise ValueError when SLACK_SIGNING_SECRET is None."""
        settings = _make_settings(signing_secret=None)
        registry = MagicMock()
        executor = MagicMock(spec=ThreadPoolExecutor)

        with pytest.raises(ValueError, match="SLACK_SIGNING_SECRET"):
            create_bolt_app(settings=settings, registry=registry, executor=executor)

    def test_returns_async_app(self) -> None:
        """create_bolt_app must return an AsyncApp instance (not a tuple)."""

        settings = _make_settings()
        registry = MagicMock()
        executor = MagicMock(spec=ThreadPoolExecutor)

        with patch("src.slack_app.app.AsyncApp") as MockAsyncApp:
            result = create_bolt_app(settings=settings, registry=registry, executor=executor)

        assert result is MockAsyncApp.return_value

    def test_app_created_with_correct_credentials(self) -> None:
        """AsyncApp must be constructed with bot_token and signing_secret from settings."""
        settings = _make_settings(bot_token="xoxb-abc", signing_secret="sig-xyz")
        registry = MagicMock()
        executor = MagicMock(spec=ThreadPoolExecutor)

        with patch("src.slack_app.app.AsyncApp") as MockAsyncApp:
            create_bolt_app(settings=settings, registry=registry, executor=executor)

        call_kwargs = MockAsyncApp.call_args.kwargs
        assert call_kwargs["token"] == "xoxb-abc"
        assert call_kwargs["signing_secret"] == "sig-xyz"

    def test_app_mention_event_handler_registered(self) -> None:
        """create_bolt_app must register an app_mention event handler."""
        settings = _make_settings()
        registry = MagicMock()
        executor = MagicMock(spec=ThreadPoolExecutor)

        with patch("src.slack_app.app.AsyncApp") as MockAsyncApp:
            create_bolt_app(settings=settings, registry=registry, executor=executor)

        mock_app = MockAsyncApp.return_value
        event_calls = [c.args[0] for c in mock_app.event.call_args_list]
        assert "app_mention" in event_calls

    def test_message_event_handler_registered(self) -> None:
        """create_bolt_app must register a message event handler."""
        settings = _make_settings()
        registry = MagicMock()
        executor = MagicMock(spec=ThreadPoolExecutor)

        with patch("src.slack_app.app.AsyncApp") as MockAsyncApp:
            create_bolt_app(settings=settings, registry=registry, executor=executor)

        mock_app = MockAsyncApp.return_value
        event_calls = [c.args[0] for c in mock_app.event.call_args_list]
        assert "message" in event_calls
