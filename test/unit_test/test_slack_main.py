"""Unit tests for :func:`src.slack_main.main` and :func:`src.slack_main._run`."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_fake_settings(
    slack_port: int = 3000,
    max_concurrent: int = 3,
    agent_config_path: str = "config/agents.yaml",
) -> MagicMock:
    """Return a minimal mock AppSettings for slack_main tests."""
    s = MagicMock()
    s.SLACK_PORT = slack_port
    s.MAX_CONCURRENT_DEV_AGENTS = max_concurrent
    s.AGENT_CONFIG_PATH = agent_config_path
    return s


def _make_fake_registry(agent_ids: list[str] | None = None) -> MagicMock:
    """Return a minimal mock AgentRegistry."""
    ids = agent_ids or ["planner", "dev_lead"]
    r = MagicMock()
    r.__len__ = MagicMock(return_value=len(ids))
    r.agent_ids = MagicMock(return_value=ids)
    return r


def _run_async(coro: object) -> None:
    """Helper to run a coroutine in a fresh event loop."""
    asyncio.run(coro)  # type: ignore[arg-type]


class TestSlackMainRun:
    """Tests for the _run() coroutine inside slack_main."""

    def test_get_settings_called(self) -> None:
        """_run() must call get_settings() exactly once."""
        fake_settings = _make_fake_settings()
        fake_registry = _make_fake_registry()
        mock_app = MagicMock()
        mock_app.start = AsyncMock()

        with (
            patch("src.slack_main.get_settings", return_value=fake_settings) as mock_gs,
            patch("src.slack_main.load_agent_config", return_value=MagicMock()),
            patch("src.slack_main.build_registry", return_value=fake_registry),
            patch("src.slack_main.create_bolt_app", return_value=mock_app),
        ):
            _run_async(__import__("src.slack_main", fromlist=["_run"])._run())

        mock_gs.assert_called_once()

    def test_load_agent_config_called_with_path(self) -> None:
        """_run() must call load_agent_config with AGENT_CONFIG_PATH from settings."""
        fake_settings = _make_fake_settings(agent_config_path="custom/path.yaml")
        fake_registry = _make_fake_registry()
        mock_app = MagicMock()
        mock_app.start = AsyncMock()

        with (
            patch("src.slack_main.get_settings", return_value=fake_settings),
            patch("src.slack_main.load_agent_config", return_value=MagicMock()) as mock_load,
            patch("src.slack_main.build_registry", return_value=fake_registry),
            patch("src.slack_main.create_bolt_app", return_value=mock_app),
        ):
            _run_async(__import__("src.slack_main", fromlist=["_run"])._run())

        mock_load.assert_called_once_with("custom/path.yaml")

    def test_build_registry_called(self) -> None:
        """_run() must call build_registry with the loaded team config and settings."""
        fake_settings = _make_fake_settings()
        fake_team_config = MagicMock()
        fake_registry = _make_fake_registry()
        mock_app = MagicMock()
        mock_app.start = AsyncMock()

        with (
            patch("src.slack_main.get_settings", return_value=fake_settings),
            patch("src.slack_main.load_agent_config", return_value=fake_team_config),
            patch("src.slack_main.build_registry", return_value=fake_registry) as mock_br,
            patch("src.slack_main.create_bolt_app", return_value=mock_app),
        ):
            _run_async(__import__("src.slack_main", fromlist=["_run"])._run())

        mock_br.assert_called_once_with(fake_team_config, fake_settings)

    def test_create_bolt_app_called_with_correct_args(self) -> None:
        """_run() must call create_bolt_app with settings, registry, and executor."""
        from concurrent.futures import ThreadPoolExecutor

        fake_settings = _make_fake_settings(max_concurrent=4)
        fake_registry = _make_fake_registry()
        mock_app = MagicMock()
        mock_app.start = AsyncMock()
        mock_executor = MagicMock(spec=ThreadPoolExecutor)

        with (
            patch("src.slack_main.get_settings", return_value=fake_settings),
            patch("src.slack_main.load_agent_config", return_value=MagicMock()),
            patch("src.slack_main.build_registry", return_value=fake_registry),
            patch("src.slack_main.ThreadPoolExecutor", return_value=mock_executor),
            patch("src.slack_main.create_bolt_app", return_value=mock_app) as mock_cba,
        ):
            _run_async(__import__("src.slack_main", fromlist=["_run"])._run())

        mock_cba.assert_called_once_with(
            settings=fake_settings,
            registry=fake_registry,
            executor=mock_executor,
        )

    def test_app_start_called_with_slack_port(self) -> None:
        """_run() must call app.start(port=settings.SLACK_PORT)."""
        fake_settings = _make_fake_settings(slack_port=4000)
        fake_registry = _make_fake_registry()
        mock_app = MagicMock()
        mock_app.start = AsyncMock()

        with (
            patch("src.slack_main.get_settings", return_value=fake_settings),
            patch("src.slack_main.load_agent_config", return_value=MagicMock()),
            patch("src.slack_main.build_registry", return_value=fake_registry),
            patch("src.slack_main.create_bolt_app", return_value=mock_app),
        ):
            _run_async(__import__("src.slack_main", fromlist=["_run"])._run())

        mock_app.start.assert_called_once_with(port=4000)

    def test_executor_created_with_correct_max_workers(self) -> None:
        """_run() must create ThreadPoolExecutor with MAX_CONCURRENT_DEV_AGENTS workers."""
        fake_settings = _make_fake_settings(max_concurrent=6)
        fake_registry = _make_fake_registry()
        mock_app = MagicMock()
        mock_app.start = AsyncMock()

        with (
            patch("src.slack_main.get_settings", return_value=fake_settings),
            patch("src.slack_main.load_agent_config", return_value=MagicMock()),
            patch("src.slack_main.build_registry", return_value=fake_registry),
            patch("src.slack_main.ThreadPoolExecutor") as MockExecutor,
            patch("src.slack_main.create_bolt_app", return_value=mock_app),
        ):
            _run_async(__import__("src.slack_main", fromlist=["_run"])._run())

        MockExecutor.assert_called_once_with(
            max_workers=6,
            thread_name_prefix="ai-slack-worker",
        )

    def test_raises_if_create_bolt_app_raises_value_error(self) -> None:
        """_run() must propagate ValueError raised by create_bolt_app."""
        fake_settings = _make_fake_settings()
        fake_registry = _make_fake_registry()

        with (
            patch("src.slack_main.get_settings", return_value=fake_settings),
            patch("src.slack_main.load_agent_config", return_value=MagicMock()),
            patch("src.slack_main.build_registry", return_value=fake_registry),
            patch(
                "src.slack_main.create_bolt_app",
                side_effect=ValueError("SLACK_BOT_TOKEN missing"),
            ),
        ):
            with pytest.raises(ValueError, match="SLACK_BOT_TOKEN"):
                _run_async(__import__("src.slack_main", fromlist=["_run"])._run())


class TestSlackMainEntry:
    """Tests for the main() entry-point wrapper."""

    def test_main_calls_asyncio_run(self) -> None:
        """main() must invoke asyncio.run() with the _run coroutine."""
        import src.slack_main as slack_main_module

        with patch("src.slack_main.asyncio.run") as mock_run:
            slack_main_module.main()

        mock_run.assert_called_once()
