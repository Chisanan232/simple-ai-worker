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
    app_token: str | None = "xapp-test",
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
    s.SLACK_APP_TOKEN = _make_secret(app_token)
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

    def test_raises_value_error_when_app_token_missing(self) -> None:
        """create_bolt_app must raise ValueError when SLACK_APP_TOKEN is None."""
        settings = _make_settings(app_token=None)
        registry = MagicMock()
        executor = MagicMock(spec=ThreadPoolExecutor)

        with pytest.raises(ValueError, match="SLACK_APP_TOKEN"):
            create_bolt_app(settings=settings, registry=registry, executor=executor)

    def test_raises_value_error_when_signing_secret_missing(self) -> None:
        """create_bolt_app must raise ValueError when SLACK_SIGNING_SECRET is None."""
        settings = _make_settings(signing_secret=None)
        registry = MagicMock()
        executor = MagicMock(spec=ThreadPoolExecutor)

        with pytest.raises(ValueError, match="SLACK_SIGNING_SECRET"):
            create_bolt_app(settings=settings, registry=registry, executor=executor)

    def test_returns_bolt_app_and_socket_mode_handler(self) -> None:
        """create_bolt_app must return a (App, SocketModeHandler) tuple."""
        from slack_bolt import App
        from slack_bolt.adapter.socket_mode import SocketModeHandler

        settings = _make_settings()
        registry = MagicMock()
        executor = MagicMock(spec=ThreadPoolExecutor)

        with (
            patch("src.slack_app.app.App") as MockApp,
            patch("src.slack_app.app.SocketModeHandler") as MockHandler,
        ):
            bolt_app, socket_handler = create_bolt_app(
                settings=settings, registry=registry, executor=executor
            )

        assert bolt_app is MockApp.return_value
        assert socket_handler is MockHandler.return_value

    def test_app_created_with_correct_credentials(self) -> None:
        """App must be constructed with bot_token and signing_secret from settings."""
        settings = _make_settings(bot_token="xoxb-abc", signing_secret="sig-xyz")
        registry = MagicMock()
        executor = MagicMock(spec=ThreadPoolExecutor)

        with (
            patch("src.slack_app.app.App") as MockApp,
            patch("src.slack_app.app.SocketModeHandler"),
        ):
            create_bolt_app(settings=settings, registry=registry, executor=executor)

        call_kwargs = MockApp.call_args.kwargs
        assert call_kwargs["token"] == "xoxb-abc"
        assert call_kwargs["signing_secret"] == "sig-xyz"

    def test_socket_handler_created_with_app_token(self) -> None:
        """SocketModeHandler must be constructed with the app-level token."""
        settings = _make_settings(app_token="xapp-mytoken")
        registry = MagicMock()
        executor = MagicMock(spec=ThreadPoolExecutor)

        with (
            patch("src.slack_app.app.App"),
            patch("src.slack_app.app.SocketModeHandler") as MockHandler,
        ):
            create_bolt_app(settings=settings, registry=registry, executor=executor)

        call_kwargs = MockHandler.call_args.kwargs
        assert call_kwargs["app_token"] == "xapp-mytoken"

    def test_app_mention_event_handler_registered(self) -> None:
        """create_bolt_app must register an app_mention event handler."""
        settings = _make_settings()
        registry = MagicMock()
        executor = MagicMock(spec=ThreadPoolExecutor)

        with (
            patch("src.slack_app.app.App") as MockApp,
            patch("src.slack_app.app.SocketModeHandler"),
        ):
            create_bolt_app(settings=settings, registry=registry, executor=executor)

        mock_bolt_app = MockApp.return_value
        # Verify the app's .event() decorator was called with "app_mention"
        event_calls = [
            c.args[0] for c in mock_bolt_app.event.call_args_list
        ]
        assert "app_mention" in event_calls

    def test_message_event_handler_registered(self) -> None:
        """create_bolt_app must register a message event handler."""
        settings = _make_settings()
        registry = MagicMock()
        executor = MagicMock(spec=ThreadPoolExecutor)

        with (
            patch("src.slack_app.app.App") as MockApp,
            patch("src.slack_app.app.SocketModeHandler"),
        ):
            create_bolt_app(settings=settings, registry=registry, executor=executor)

        mock_bolt_app = MockApp.return_value
        event_calls = [
            c.args[0] for c in mock_bolt_app.event.call_args_list
        ]
        assert "message" in event_calls

