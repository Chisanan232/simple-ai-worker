"""Unit tests for :func:`src.main.main` and its signal-handling helpers."""

from __future__ import annotations

import signal
from unittest.mock import MagicMock, patch

import src.main as main_module
from src.main import _handle_signal


class TestHandleSignal:
    """Tests for the _handle_signal() shutdown flag setter."""

    def setup_method(self) -> None:
        """Reset the global shutdown flag before each test."""
        main_module._shutdown_requested = False

    def test_sets_shutdown_flag_on_sigint(self) -> None:
        """_handle_signal must set _shutdown_requested to True for SIGINT."""
        _handle_signal(signal.SIGINT, None)
        assert main_module._shutdown_requested is True

    def test_sets_shutdown_flag_on_sigterm(self) -> None:
        """_handle_signal must set _shutdown_requested to True for SIGTERM."""
        _handle_signal(signal.SIGTERM, None)
        assert main_module._shutdown_requested is True

    def teardown_method(self) -> None:
        """Reset the global shutdown flag after each test."""
        main_module._shutdown_requested = False


class TestMain:
    """Tests for the main() entry-point function."""

    def setup_method(self) -> None:
        """Reset shared globals before each test."""
        main_module._shutdown_requested = False
        main_module._runner = None
        main_module._registry = None

    def teardown_method(self) -> None:
        """Reset shared globals after each test."""
        main_module._shutdown_requested = False
        main_module._runner = None
        main_module._registry = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_fake_settings(
        self,
        interval_seconds: int = 60,
        timezone: str = "UTC",
        max_concurrent: int = 3,
        agent_config_path: str = "config/agents.yaml",
    ) -> MagicMock:
        """Return a mock AppSettings object with sensible defaults."""
        mock_settings = MagicMock()
        mock_settings.SCHEDULER_INTERVAL_SECONDS = interval_seconds
        mock_settings.SCHEDULER_TIMEZONE = timezone
        mock_settings.MAX_CONCURRENT_DEV_AGENTS = max_concurrent
        mock_settings.AGENT_CONFIG_PATH = agent_config_path
        return mock_settings

    def _make_fake_registry(self, agent_ids: list = None) -> MagicMock:  # type: ignore[assignment]
        """Return a mock AgentRegistry."""
        ids = agent_ids or ["planner", "dev_lead", "dev_agent"]
        mock_registry = MagicMock()
        mock_registry.__len__ = MagicMock(return_value=len(ids))
        mock_registry.agent_ids = MagicMock(return_value=ids)
        return mock_registry

    def _all_patches(
        self,
        fake_settings: MagicMock,
        fake_registry: MagicMock,
        sleep_side_effect: object = None,
    ) -> tuple:
        """Return a tuple of context-manager patches used in every main() test."""
        return (
            patch("src.main.get_settings", return_value=fake_settings),
            patch("src.main.load_agent_config", return_value=MagicMock()),
            patch("src.main.build_registry", return_value=fake_registry),
            patch("src.main.SchedulerRunner"),
            patch("src.main.time.sleep", side_effect=sleep_side_effect),
            patch("src.main.signal.signal"),
        )

    # ------------------------------------------------------------------
    # Scheduler lifecycle
    # ------------------------------------------------------------------

    def test_main_starts_and_stops_scheduler(self) -> None:
        """main() must start a SchedulerRunner and stop it on shutdown."""
        call_count = 0

        def fake_sleep(seconds: float) -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                main_module._shutdown_requested = True

        fake_settings = self._make_fake_settings(interval_seconds=60, timezone="UTC")
        fake_registry = self._make_fake_registry()

        with (
            patch("src.main.get_settings", return_value=fake_settings),
            patch("src.main.load_agent_config", return_value=MagicMock()),
            patch("src.main.build_registry", return_value=fake_registry),
            patch("src.main.SchedulerRunner") as MockRunner,
            patch("src.main.time.sleep", side_effect=fake_sleep),
            patch("src.main.signal.signal"),
        ):
            mock_instance = MockRunner.return_value
            mock_instance.running = False

            main_module.main()

            MockRunner.assert_called_once_with(interval_seconds=60, timezone="UTC")
            mock_instance.start.assert_called_once()
            mock_instance.stop.assert_called_once_with(wait=True)

    def test_main_uses_settings_interval(self) -> None:
        """main() must pass SCHEDULER_INTERVAL_SECONDS from settings to SchedulerRunner."""
        fake_settings = self._make_fake_settings(interval_seconds=120, timezone="Asia/Taipei")
        fake_registry = self._make_fake_registry()

        def fast_exit(seconds: float) -> None:
            main_module._shutdown_requested = True

        with (
            patch("src.main.get_settings", return_value=fake_settings),
            patch("src.main.load_agent_config", return_value=MagicMock()),
            patch("src.main.build_registry", return_value=fake_registry),
            patch("src.main.SchedulerRunner") as MockRunner,
            patch("src.main.time.sleep", side_effect=fast_exit),
            patch("src.main.signal.signal"),
        ):
            main_module.main()
            MockRunner.assert_called_once_with(interval_seconds=120, timezone="Asia/Taipei")

    def test_main_registers_signal_handlers(self) -> None:
        """main() must register handlers for SIGINT and SIGTERM."""
        fake_settings = self._make_fake_settings()
        fake_registry = self._make_fake_registry()

        def fast_exit(seconds: float) -> None:
            main_module._shutdown_requested = True

        with (
            patch("src.main.get_settings", return_value=fake_settings),
            patch("src.main.load_agent_config", return_value=MagicMock()),
            patch("src.main.build_registry", return_value=fake_registry),
            patch("src.main.SchedulerRunner"),
            patch("src.main.time.sleep", side_effect=fast_exit),
            patch("src.main.signal.signal") as mock_signal,
        ):
            main_module.main()

            registered_signals = [c.args[0] for c in mock_signal.call_args_list]
            assert signal.SIGINT in registered_signals
            assert signal.SIGTERM in registered_signals

    # ------------------------------------------------------------------
    # Phase 4 — agent config + registry
    # ------------------------------------------------------------------

    def test_main_loads_agent_config_from_settings_path(self) -> None:
        """main() must call load_agent_config with AGENT_CONFIG_PATH from settings."""
        fake_settings = self._make_fake_settings(agent_config_path="custom/agents.yaml")
        fake_registry = self._make_fake_registry()

        def fast_exit(seconds: float) -> None:
            main_module._shutdown_requested = True

        with (
            patch("src.main.get_settings", return_value=fake_settings),
            patch("src.main.load_agent_config", return_value=MagicMock()) as mock_load,
            patch("src.main.build_registry", return_value=fake_registry),
            patch("src.main.SchedulerRunner"),
            patch("src.main.time.sleep", side_effect=fast_exit),
            patch("src.main.signal.signal"),
        ):
            main_module.main()

        mock_load.assert_called_once_with("custom/agents.yaml")

    def test_main_builds_registry_from_team_config_and_settings(self) -> None:
        """main() must call build_registry with the loaded team config and settings."""
        fake_settings = self._make_fake_settings()
        fake_team_config = MagicMock()
        fake_registry = self._make_fake_registry()

        def fast_exit(seconds: float) -> None:
            main_module._shutdown_requested = True

        with (
            patch("src.main.get_settings", return_value=fake_settings),
            patch("src.main.load_agent_config", return_value=fake_team_config),
            patch("src.main.build_registry", return_value=fake_registry) as mock_build,
            patch("src.main.SchedulerRunner"),
            patch("src.main.time.sleep", side_effect=fast_exit),
            patch("src.main.signal.signal"),
        ):
            main_module.main()

        mock_build.assert_called_once_with(fake_team_config, fake_settings)

    def test_main_assigns_registry_global(self) -> None:
        """main() must assign the built registry to the module-level _registry."""
        fake_settings = self._make_fake_settings()
        fake_registry = self._make_fake_registry()

        def fast_exit(seconds: float) -> None:
            main_module._shutdown_requested = True

        with (
            patch("src.main.get_settings", return_value=fake_settings),
            patch("src.main.load_agent_config", return_value=MagicMock()),
            patch("src.main.build_registry", return_value=fake_registry),
            patch("src.main.SchedulerRunner"),
            patch("src.main.time.sleep", side_effect=fast_exit),
            patch("src.main.signal.signal"),
        ):
            main_module.main()

        assert main_module._registry is fake_registry
