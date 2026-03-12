"""
Unit tests for :class:`src.crew.builder.CrewBuilder`.

crewai.Crew is patched so no real execution occurs.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from crewai import Process

from src.crew.builder import CrewBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_agent(role: str = "Planner") -> MagicMock:
    a = MagicMock()
    a.role = role
    return a


def _mock_task() -> MagicMock:
    return MagicMock()


# ===========================================================================
# CrewBuilder._resolve_process
# ===========================================================================

class TestResolveProcess:
    def test_sequential_maps_to_process_enum(self) -> None:
        result = CrewBuilder._resolve_process("sequential")
        assert result is Process.sequential

    def test_hierarchical_maps_to_process_enum(self) -> None:
        result = CrewBuilder._resolve_process("hierarchical")
        assert result is Process.hierarchical

    def test_unknown_process_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="unknown process"):
            CrewBuilder._resolve_process("parallel")

    def test_error_message_contains_valid_values(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            CrewBuilder._resolve_process("bad")
        msg = str(exc_info.value)
        assert "sequential" in msg
        assert "hierarchical" in msg


# ===========================================================================
# CrewBuilder.build — guard clauses
# ===========================================================================

class TestCrewBuilderGuards:
    def test_empty_agents_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="at least one agent"):
            CrewBuilder.build(agents=[], tasks=[_mock_task()])

    def test_empty_tasks_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="at least one task"):
            CrewBuilder.build(agents=[_mock_agent()], tasks=[])

    def test_invalid_process_raises_value_error(self) -> None:
        with (
            patch("src.crew.builder.Crew"),
            pytest.raises(ValueError),
        ):
            CrewBuilder.build(
                agents=[_mock_agent()],
                tasks=[_mock_task()],
                process="parallel",  # type: ignore[arg-type]
            )


# ===========================================================================
# CrewBuilder.build — happy paths (crewai.Crew patched)
# ===========================================================================

class TestCrewBuilderBuild:
    def test_build_returns_crew_instance(self) -> None:
        """build() must return whatever crewai.Crew() returns."""
        mock_crew = MagicMock()
        with patch("src.crew.builder.Crew", return_value=mock_crew) as MockCrew:
            result = CrewBuilder.build(
                agents=[_mock_agent()],
                tasks=[_mock_task()],
            )
        assert result is mock_crew
        MockCrew.assert_called_once()

    def test_build_passes_agents(self) -> None:
        """build() must forward the agents list to Crew()."""
        agents = [_mock_agent("Planner"), _mock_agent("Dev Lead")]
        with patch("src.crew.builder.Crew") as MockCrew:
            CrewBuilder.build(agents=agents, tasks=[_mock_task()])
        kwargs = MockCrew.call_args.kwargs
        assert kwargs["agents"] is agents

    def test_build_passes_tasks(self) -> None:
        """build() must forward the tasks list to Crew()."""
        tasks = [_mock_task(), _mock_task()]
        with patch("src.crew.builder.Crew") as MockCrew:
            CrewBuilder.build(agents=[_mock_agent()], tasks=tasks)
        kwargs = MockCrew.call_args.kwargs
        assert kwargs["tasks"] is tasks

    def test_build_default_process_is_sequential(self) -> None:
        """build() without explicit process must use Process.sequential."""
        with patch("src.crew.builder.Crew") as MockCrew:
            CrewBuilder.build(agents=[_mock_agent()], tasks=[_mock_task()])
        kwargs = MockCrew.call_args.kwargs
        assert kwargs["process"] is Process.sequential

    def test_build_hierarchical_process(self) -> None:
        """build() must map 'hierarchical' to Process.hierarchical."""
        with patch("src.crew.builder.Crew") as MockCrew:
            CrewBuilder.build(
                agents=[_mock_agent()],
                tasks=[_mock_task()],
                process="hierarchical",
            )
        kwargs = MockCrew.call_args.kwargs
        assert kwargs["process"] is Process.hierarchical

    def test_build_default_verbose_is_false(self) -> None:
        """build() must pass verbose=False by default."""
        with patch("src.crew.builder.Crew") as MockCrew:
            CrewBuilder.build(agents=[_mock_agent()], tasks=[_mock_task()])
        kwargs = MockCrew.call_args.kwargs
        assert kwargs["verbose"] is False

    def test_build_verbose_true_forwarded(self) -> None:
        """build() must forward verbose=True to Crew()."""
        with patch("src.crew.builder.Crew") as MockCrew:
            CrewBuilder.build(
                agents=[_mock_agent()],
                tasks=[_mock_task()],
                verbose=True,
            )
        kwargs = MockCrew.call_args.kwargs
        assert kwargs["verbose"] is True

    def test_build_multiple_agents_and_tasks(self) -> None:
        """build() must handle multiple agents and tasks."""
        agents = [_mock_agent(f"Agent {i}") for i in range(3)]
        tasks = [_mock_task() for _ in range(5)]
        with patch("src.crew.builder.Crew") as MockCrew:
            CrewBuilder.build(agents=agents, tasks=tasks)
        kwargs = MockCrew.call_args.kwargs
        assert len(kwargs["agents"]) == 3
        assert len(kwargs["tasks"]) == 5

