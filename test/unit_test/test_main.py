"""Unit tests for :func:`src.main.main` and its signal-handling helpers."""

from __future__ import annotations

import signal
from unittest.mock import patch

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

    def teardown_method(self) -> None:
        """Reset shared globals after each test."""
        main_module._shutdown_requested = False
        main_module._runner = None

    def test_main_starts_and_stops_scheduler(self) -> None:
        """main() must start a SchedulerRunner and stop it on shutdown."""
        call_count = 0

        def fake_sleep(seconds: float) -> None:
            """After the first sleep tick, inject a shutdown signal."""
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                main_module._shutdown_requested = True

        with (
            patch("src.main.SchedulerRunner") as MockRunner,
            patch("src.main.time.sleep", side_effect=fake_sleep),
            patch("src.main.signal.signal"),
        ):
            mock_instance = MockRunner.return_value
            mock_instance.running = False

            main_module.main()

            MockRunner.assert_called_once_with(interval_seconds=30, timezone="UTC")
            mock_instance.start.assert_called_once()
            mock_instance.stop.assert_called_once_with(wait=True)

    def test_main_registers_signal_handlers(self) -> None:
        """main() must register handlers for SIGINT and SIGTERM."""

        def fast_exit(seconds: float) -> None:
            main_module._shutdown_requested = True

        with (
            patch("src.main.SchedulerRunner"),
            patch("src.main.time.sleep", side_effect=fast_exit),
            patch("src.main.signal.signal") as mock_signal,
        ):
            main_module.main()

            registered_signals = [c.args[0] for c in mock_signal.call_args_list]
            assert signal.SIGINT in registered_signals
            assert signal.SIGTERM in registered_signals


