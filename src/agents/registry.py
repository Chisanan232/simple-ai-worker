"""
Agent registry for simple-ai-worker.

Provides :class:`AgentRegistry` — a typed container that maps agent ``id``
strings to their corresponding ``crewai.Agent`` instances — and
:func:`build_registry`, a convenience function that populates a registry
from a :class:`~src.config.agent_config.AgentTeamConfig` in one call.

The registry is built once at application startup and then injected into
scheduler jobs and Slack Bolt handlers by reference.  Because all CrewAI
agent objects are stateless between ``kickoff()`` calls, sharing the same
registry instance across jobs is safe.

Usage::

    from src.agents.registry import build_registry
    from src.config import get_settings, load_agent_config

    settings = get_settings()
    team_config = load_agent_config(settings.AGENT_CONFIG_PATH)
    registry = build_registry(team_config, settings)

    planner = registry["planner"]
    dev_lead = registry.get("dev_lead")

    # Iterate all agents
    for agent_id, agent in registry.items():
        print(agent_id, agent.role)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterator, List, Optional

from crewai import Agent

from .factory import AgentFactory

if TYPE_CHECKING:
    from src.config.agent_config import AgentTeamConfig
    from src.config.settings import AppSettings

__all__: List[str] = ["AgentRegistry", "build_registry"]

logger = logging.getLogger(__name__)


class AgentRegistry:
    """A typed, dict-like container of ``crewai.Agent`` instances.

    Agents are keyed by their string ``id`` (as defined in the agent YAML).
    The registry exposes a minimal mapping interface — ``__getitem__``,
    ``get``, ``__contains__``, ``__len__``, ``__iter__``, and ``items`` —
    so callers can look up and iterate agents without depending on the
    internal ``dict`` directly.

    Attributes:
        _agents: Internal ``dict[str, Agent]`` store.
    """

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def register(self, agent_id: str, agent: Agent) -> None:
        """Register *agent* under *agent_id*.

        If an agent with the same *agent_id* already exists it is silently
        replaced (callers should avoid duplicate IDs — the YAML validator
        already enforces uniqueness at config-load time).

        Args:
            agent_id: The unique string identifier for this agent.
            agent: The ``crewai.Agent`` instance to store.
        """
        if agent_id in self._agents:
            logger.warning(
                "AgentRegistry: overwriting existing agent with id=%r.", agent_id
            )
        self._agents[agent_id] = agent
        logger.debug("AgentRegistry: registered agent id=%r, role=%r.", agent_id, agent.role)

    # ------------------------------------------------------------------
    # Mapping interface
    # ------------------------------------------------------------------

    def __getitem__(self, agent_id: str) -> Agent:
        """Return the agent registered under *agent_id*.

        Raises:
            KeyError: If no agent with *agent_id* has been registered.
        """
        try:
            return self._agents[agent_id]
        except KeyError:
            raise KeyError(
                f"AgentRegistry: no agent registered with id={agent_id!r}. "
                f"Available ids: {sorted(self._agents)}"
            ) from None

    def get(self, agent_id: str, default: Optional[Agent] = None) -> Optional[Agent]:
        """Return the agent registered under *agent_id*, or *default*.

        Args:
            agent_id: The agent identifier to look up.
            default: Value to return if *agent_id* is not found.
                Defaults to ``None``.

        Returns:
            The registered ``crewai.Agent``, or *default*.
        """
        return self._agents.get(agent_id, default)

    def __contains__(self, agent_id: object) -> bool:
        """Return ``True`` if an agent with *agent_id* is registered."""
        return agent_id in self._agents

    def __len__(self) -> int:
        """Return the number of registered agents."""
        return len(self._agents)

    def __iter__(self) -> Iterator[str]:
        """Iterate over registered agent IDs."""
        return iter(self._agents)

    def items(self) -> Iterator[tuple[str, Agent]]:
        """Iterate over ``(agent_id, agent)`` pairs.

        Returns:
            An iterator of ``(str, crewai.Agent)`` tuples.
        """
        return iter(self._agents.items())  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def agent_ids(self) -> List[str]:
        """Return a sorted list of all registered agent IDs.

        Returns:
            Sorted ``list[str]`` of agent identifier strings.
        """
        return sorted(self._agents)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"agents={self.agent_ids()!r})"
        )


def build_registry(
    team_config: "AgentTeamConfig",
    settings: "AppSettings",
) -> AgentRegistry:
    """Build and return a populated :class:`AgentRegistry`.

    Iterates over every :class:`~src.config.agent_config.AgentConfig` in
    *team_config*, builds a ``crewai.Agent`` via
    :class:`~src.agents.factory.AgentFactory`, and registers it.

    This function is the recommended way to initialise the registry at
    application startup:

    .. code-block:: python

        settings = get_settings()
        team_config = load_agent_config(settings.AGENT_CONFIG_PATH)
        registry = build_registry(team_config, settings)

    Args:
        team_config: The validated :class:`~src.config.agent_config.AgentTeamConfig`
            loaded from ``config/agents.yaml``.
        settings: The application :class:`~src.config.settings.AppSettings`
            singleton.

    Returns:
        A fully populated :class:`AgentRegistry`.
    """
    registry = AgentRegistry()
    for agent_cfg in team_config.agents:
        agent = AgentFactory.build(agent_cfg, settings)
        registry.register(agent_cfg.id, agent)

    logger.info(
        "AgentRegistry built: %d agent(s) registered (%s).",
        len(registry),
        ", ".join(registry.agent_ids()),
    )
    return registry

