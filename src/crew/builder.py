"""
Crew builder for simple-ai-worker.

Provides :class:`CrewBuilder` — a stateless factory that assembles a
``crewai.Crew`` from a list of agents, a list of tasks, and a process
string sourced from :class:`~src.config.agent_config.AgentTeamConfig`.

Design decisions
----------------
- ``process`` is accepted as a plain ``str`` (``"sequential"`` or
  ``"hierarchical"``) matching the ``AgentTeamConfig.process`` field,
  and mapped to ``crewai.Process`` enum values internally.  Callers
  never need to import ``crewai.Process`` themselves.
- ``verbose`` defaults to ``False`` — individual agent verbosity is
  controlled per-agent in the YAML, not at crew level.
- The builder is intentionally **stateless**; each call to
  :meth:`CrewBuilder.build` returns a new, independent ``crewai.Crew``
  instance.  This matches the design of short-lived Crews described in
  Phase 6 (one Crew per Slack message, one Crew per ticket dispatch).

Usage::

    from crewai import Task
    from src.crew.builder import CrewBuilder
    from src.agents.registry import AgentRegistry

    registry = ...  # populated AgentRegistry
    planner = registry["planner"]

    task = Task(
        description="Plan the epic.",
        expected_output="JIRA epic created.",
        agent=planner,
    )
    crew = CrewBuilder.build(
        agents=[planner],
        tasks=[task],
        process="sequential",
    )
    result = crew.kickoff()
"""

from __future__ import annotations

import logging
from typing import List, Literal

from crewai import Agent, Crew, Process, Task

__all__: List[str] = ["CrewBuilder"]

logger = logging.getLogger(__name__)

# Map the YAML string values to crewai.Process enum members.
_PROCESS_MAP: dict[str, Process] = {
    "sequential": Process.sequential,
    "hierarchical": Process.hierarchical,
}


class CrewBuilder:
    """Stateless factory that builds a ``crewai.Crew``.

    All methods are class methods so the builder never needs to be
    instantiated.
    """

    @classmethod
    def build(
        cls,
        agents: List[Agent],
        tasks: List[Task],
        process: Literal["sequential", "hierarchical"] = "sequential",
        verbose: bool = False,
    ) -> Crew:
        """Build and return a ``crewai.Crew`` instance.

        Args:
            agents: The list of ``crewai.Agent`` objects participating in
                this crew.  Must contain at least one agent.
            tasks: The list of ``crewai.Task`` objects the crew will
                execute.  Must contain at least one task.
            process: Execution process — ``"sequential"`` (default) runs
                tasks one after another; ``"hierarchical"`` uses a manager
                agent.  Must match a value in
                :attr:`~src.config.agent_config.AgentTeamConfig.process`.
            verbose: Whether to enable verbose output for the crew.
                Individual agent verbosity is set per-agent in the YAML.
                Defaults to ``False``.

        Returns:
            A freshly constructed ``crewai.Crew`` instance.

        Raises:
            ValueError: If *agents* or *tasks* is empty, or if *process*
                is not one of the recognised values.
        """
        if not agents:
            raise ValueError("CrewBuilder.build() requires at least one agent.")
        if not tasks:
            raise ValueError("CrewBuilder.build() requires at least one task.")

        crew_process = cls._resolve_process(process)

        logger.debug(
            "Building Crew: agents=%s, tasks=%d, process=%s.",
            [a.role for a in agents],
            len(tasks),
            process,
        )

        return Crew(
            agents=agents,
            tasks=tasks,
            process=crew_process,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_process(process: str) -> Process:
        """Map the YAML *process* string to a ``crewai.Process`` enum value.

        Args:
            process: One of ``"sequential"`` or ``"hierarchical"``.

        Returns:
            The corresponding ``crewai.Process`` enum member.

        Raises:
            ValueError: If *process* is not a recognised value.
        """
        resolved = _PROCESS_MAP.get(process)
        if resolved is None:
            valid = list(_PROCESS_MAP)
            raise ValueError(
                f"CrewBuilder: unknown process {process!r}. "
                f"Valid values: {valid}"
            )
        return resolved

