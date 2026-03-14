"""Unit tests for :func:`src.slack_main.main`."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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


class TestSlackMainRun:
    """Tests for the logic inside main() (previously _run()).

    After the asyncio refactor, all setup and startup logic lives directly
    in ``main()``.  ``app.start()`` is now a synchronous blocking call, so
    tests use plain ``MagicMock`` (not ``AsyncMock``).
    """

    def test_get_settings_called(self) -> None:
        """main() must call get_settings() exactly once."""
        import src.slack_main as slack_main_module

        fake_settings = _make_fake_settings()
        fake_registry = _make_fake_registry()
        mock_app = MagicMock()

        with (
            patch("src.slack_main.get_settings", return_value=fake_settings) as mock_gs,
            patch("src.slack_main.load_agent_config", return_value=MagicMock()),
            patch("src.slack_main.build_registry", return_value=fake_registry),
            patch("src.slack_main.create_bolt_app", return_value=mock_app),
        ):
            slack_main_module.main()

        mock_gs.assert_called_once()

    def test_load_agent_config_called_with_path(self) -> None:
        """main() must call load_agent_config with AGENT_CONFIG_PATH from settings."""
        import src.slack_main as slack_main_module

        fake_settings = _make_fake_settings(agent_config_path="custom/path.yaml")
        fake_registry = _make_fake_registry()
        mock_app = MagicMock()

        with (
            patch("src.slack_main.get_settings", return_value=fake_settings),
            patch("src.slack_main.load_agent_config", return_value=MagicMock()) as mock_load,
            patch("src.slack_main.build_registry", return_value=fake_registry),
            patch("src.slack_main.create_bolt_app", return_value=mock_app),
        ):
            slack_main_module.main()

        mock_load.assert_called_once_with("custom/path.yaml", fake_settings)

    def test_build_registry_called(self) -> None:
        """main() must call build_registry with the loaded team config and settings."""
        import src.slack_main as slack_main_module

        fake_settings = _make_fake_settings()
        fake_team_config = MagicMock()
        fake_registry = _make_fake_registry()
        mock_app = MagicMock()

        with (
            patch("src.slack_main.get_settings", return_value=fake_settings),
            patch("src.slack_main.load_agent_config", return_value=fake_team_config),
            patch("src.slack_main.build_registry", return_value=fake_registry) as mock_br,
            patch("src.slack_main.create_bolt_app", return_value=mock_app),
        ):
            slack_main_module.main()

        mock_br.assert_called_once_with(fake_team_config, fake_settings)

    def test_create_bolt_app_called_with_correct_args(self) -> None:
        """main() must call create_bolt_app with settings, registry, and executor."""
        import src.slack_main as slack_main_module
        from concurrent.futures import ThreadPoolExecutor

        fake_settings = _make_fake_settings(max_concurrent=4)
        fake_registry = _make_fake_registry()
        mock_app = MagicMock()
        mock_executor = MagicMock(spec=ThreadPoolExecutor)

        with (
            patch("src.slack_main.get_settings", return_value=fake_settings),
            patch("src.slack_main.load_agent_config", return_value=MagicMock()),
            patch("src.slack_main.build_registry", return_value=fake_registry),
            patch("src.slack_main.ThreadPoolExecutor", return_value=mock_executor),
            patch("src.slack_main.create_bolt_app", return_value=mock_app) as mock_cba,
        ):
            slack_main_module.main()

        mock_cba.assert_called_once_with(
            settings=fake_settings,
            registry=fake_registry,
            executor=mock_executor,
        )

    def test_app_start_called_with_slack_port(self) -> None:
        """main() must call app.start(port=settings.SLACK_PORT).

        ``AsyncApp.start()`` is a synchronous blocking call — it owns the
        aiohttp event loop internally.  The mock is therefore a plain
        ``MagicMock``, not an ``AsyncMock``.
        """
        import src.slack_main as slack_main_module

        fake_settings = _make_fake_settings(slack_port=4000)
        fake_registry = _make_fake_registry()
        mock_app = MagicMock()

        with (
            patch("src.slack_main.get_settings", return_value=fake_settings),
            patch("src.slack_main.load_agent_config", return_value=MagicMock()),
            patch("src.slack_main.build_registry", return_value=fake_registry),
            patch("src.slack_main.create_bolt_app", return_value=mock_app),
        ):
            slack_main_module.main()

        mock_app.start.assert_called_once_with(port=4000)

    def test_executor_created_with_correct_max_workers(self) -> None:
        """main() must create ThreadPoolExecutor with MAX_CONCURRENT_DEV_AGENTS workers."""
        import src.slack_main as slack_main_module

        fake_settings = _make_fake_settings(max_concurrent=6)
        fake_registry = _make_fake_registry()
        mock_app = MagicMock()

        with (
            patch("src.slack_main.get_settings", return_value=fake_settings),
            patch("src.slack_main.load_agent_config", return_value=MagicMock()),
            patch("src.slack_main.build_registry", return_value=fake_registry),
            patch("src.slack_main.ThreadPoolExecutor") as MockExecutor,
            patch("src.slack_main.create_bolt_app", return_value=mock_app),
        ):
            slack_main_module.main()

        MockExecutor.assert_called_once_with(
            max_workers=6,
            thread_name_prefix="ai-slack-worker",
        )

    def test_raises_if_create_bolt_app_raises_value_error(self) -> None:
        """main() must propagate ValueError raised by create_bolt_app."""
        import src.slack_main as slack_main_module

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
                slack_main_module.main()


class TestSlackMainEntry:
    """Tests for the main() entry-point wrapper."""

    def test_main_calls_app_start(self) -> None:
        """main() must call app.start() with the configured SLACK_PORT.

        The old design called ``asyncio.run(_run())``.  After the refactor,
        ``main()`` calls ``app.start(port=...)`` directly — ``AsyncApp.start()``
        is a synchronous blocking method that owns its own aiohttp event loop.
        """
        import src.slack_main as slack_main_module

        fake_settings = _make_fake_settings(slack_port=3000)
        fake_registry = _make_fake_registry()
        mock_app = MagicMock()

        with (
            patch("src.slack_main.get_settings", return_value=fake_settings),
            patch("src.slack_main.load_agent_config", return_value=MagicMock()),
            patch("src.slack_main.build_registry", return_value=fake_registry),
            patch("src.slack_main.create_bolt_app", return_value=mock_app),
        ):
            slack_main_module.main()

        mock_app.start.assert_called_once_with(port=3000)
