"""
Agent factory for simple-ai-worker.

Provides :class:`AgentFactory` — a stateless factory that converts a
:class:`~src.config.agent_config.AgentConfig` and application
:class:`~src.config.settings.AppSettings` into a fully initialised
``crewai.Agent`` instance.

The factory delegates LLM construction to :class:`~src.agents.llm_factory.LLMFactory`
and passes the ``apps`` list directly to ``crewai.Agent(apps=[...])`` so
that CrewAI Enterprise native tool integrations are resolved by the
platform — no custom Python tool code is required here.

Usage::

    from src.agents.factory import AgentFactory
    from src.config import get_settings, load_agent_config

    settings = get_settings()
    team_config = load_agent_config(settings.AGENT_CONFIG_PATH)
    agent = AgentFactory.build(team_config.agents[0], settings)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

from crewai import Agent

from .llm_factory import LLMFactory

if TYPE_CHECKING:
    from src.config.agent_config import AgentConfig
    from src.config.settings import AppSettings

__all__: List[str] = ["AgentFactory"]

logger = logging.getLogger(__name__)


class AgentFactory:
    """Stateless factory that builds a ``crewai.Agent`` from Pydantic config.

    All methods are class methods so the factory never needs to be
    instantiated.
    """

    @classmethod
    def build(cls, agent_config: "AgentConfig", settings: "AppSettings") -> Agent:
        """Build and return a ``crewai.Agent`` instance.

        Delegates LLM construction to :class:`~src.agents.llm_factory.LLMFactory`
        and passes the ``apps``, ``allow_delegation``, and ``verbose`` fields
        directly to the ``crewai.Agent`` constructor.

        Args:
            agent_config: The :class:`~src.config.agent_config.AgentConfig`
                for this agent, parsed from the YAML config file.
            settings: The application :class:`~src.config.settings.AppSettings`
                singleton — forwarded to :class:`LLMFactory` for API-key
                resolution.

        Returns:
            A fully initialised ``crewai.Agent`` ready to be registered in
            :class:`~src.agents.registry.AgentRegistry` and used in a Crew.
        """
        llm = LLMFactory.build(agent_config.llm, settings)

        logger.debug(
            "Building Agent: id=%s, role=%s, provider=%s, model=%s, apps=%s.",
            agent_config.id,
            agent_config.role,
            agent_config.llm.provider,
            agent_config.llm.model,
            agent_config.apps,
        )

        return Agent(
            role=agent_config.role,
            goal=agent_config.goal,
            backstory=agent_config.backstory,
            llm=llm,
            apps=agent_config.apps,
            allow_delegation=agent_config.allow_delegation,
            verbose=agent_config.verbose,
        )
