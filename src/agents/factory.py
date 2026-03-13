"""
Agent factory for simple-ai-worker.

Provides :class:`AgentFactory` — a stateless factory that converts a
:class:`~src.config.agent_config.AgentConfig` and application
:class:`~src.config.settings.AppSettings` into a fully initialised
``crewai.Agent`` instance.

The factory delegates LLM construction to :class:`~src.agents.llm_factory.LLMFactory`
and resolves MCP server references from :attr:`AgentTeamConfig.mcp_servers`
into ``crewai.mcp.MCPServerHTTP`` / ``MCPServerSSE`` / ``MCPServerStdio``
objects that are passed to ``crewai.Agent(mcps=[...])``.

The legacy ``apps`` field is still forwarded to ``crewai.Agent(apps=[...])``
for backward compatibility during the MCP migration period.

Usage::

    from src.agents.factory import AgentFactory
    from src.config import get_settings, load_agent_config

    settings = get_settings()
    team_config = load_agent_config(settings.AGENT_CONFIG_PATH, settings)
    agent = AgentFactory.build(
        team_config.agents[0],
        settings,
        mcp_servers=team_config.mcp_servers,
    )
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from crewai import Agent
from crewai.mcp import (
    MCPServerHTTP,
    MCPServerSSE,
    MCPServerStdio,
    create_static_tool_filter,
)

from .llm_factory import LLMFactory

if TYPE_CHECKING:
    from crewai.mcp import MCPServerConfig

    from src.config.agent_config import AgentConfig, MCPRef, MCPServerDefinition
    from src.config.settings import AppSettings

__all__: List[str] = ["AgentFactory"]

logger = logging.getLogger(__name__)

_ENV_PLACEHOLDER_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")


class AgentFactory:
    """Stateless factory that builds a ``crewai.Agent`` from Pydantic config.

    All methods are class methods so the factory never needs to be
    instantiated.
    """

    @classmethod
    def build(
        cls,
        agent_config: "AgentConfig",
        settings: "AppSettings",
        mcp_servers: Optional[Dict[str, "MCPServerDefinition"]] = None,
    ) -> Agent:
        """Build and return a ``crewai.Agent`` instance.

        Delegates LLM construction to :class:`~src.agents.llm_factory.LLMFactory`,
        resolves MCP server references into ``crewai.mcp`` config objects via
        :meth:`_resolve_mcp_configs`, and forwards ``apps``,
        ``allow_delegation``, and ``verbose`` directly to ``crewai.Agent``.

        Args:
            agent_config: The :class:`~src.config.agent_config.AgentConfig`
                for this agent, parsed from the YAML config file.
            settings: The application :class:`~src.config.settings.AppSettings`
                singleton — forwarded to :class:`LLMFactory` for API-key
                resolution and used for ``${VAR}`` placeholder resolution in
                MCP server headers.
            mcp_servers: The MCP server registry from
                :attr:`~src.config.agent_config.AgentTeamConfig.mcp_servers`.
                When provided, each entry in ``agent_config.mcps`` is resolved
                to the matching ``crewai.mcp`` config object.
                When ``None`` or empty, ``mcps=None`` is passed to
                ``crewai.Agent``.

        Returns:
            A fully initialised ``crewai.Agent`` ready to be registered in
            :class:`~src.agents.registry.AgentRegistry` and used in a Crew.
        """
        llm = LLMFactory.build(agent_config.llm, settings)

        crewai_mcps: Optional[List["MCPServerConfig"]] = None
        if agent_config.mcps and mcp_servers:
            resolved = cls._resolve_mcp_configs(agent_config.mcps, mcp_servers, settings)
            crewai_mcps = resolved if resolved else None

        logger.debug(
            "Building Agent: id=%s, role=%s, provider=%s, model=%s, "
            "apps=%s, mcp_servers=%s.",
            agent_config.id,
            agent_config.role,
            agent_config.llm.provider,
            agent_config.llm.model,
            agent_config.apps,
            [r if isinstance(r, str) else r.server for r in agent_config.mcps],
        )

        return Agent(
            role=agent_config.role,
            goal=agent_config.goal,
            backstory=agent_config.backstory,
            llm=llm,
            apps=agent_config.apps or None,
            mcps=crewai_mcps,
            allow_delegation=agent_config.allow_delegation,
            verbose=agent_config.verbose,
        )

    # ------------------------------------------------------------------
    # MCP resolution helpers
    # ------------------------------------------------------------------

    @classmethod
    def _resolve_mcp_configs(
        cls,
        refs: List["MCPRef"],
        registry: Dict[str, "MCPServerDefinition"],
        settings: "AppSettings",
    ) -> List[Union[MCPServerHTTP, MCPServerSSE, MCPServerStdio]]:
        """Resolve agent MCP ID references to ``crewai.mcp`` config objects.

        For each entry in *refs*:
        - A plain ``str`` → looks up the server by ID and uses the registry's
          ``tool_filter`` unchanged.
        - An ``MCPServerRef`` mapping → looks up the server by ID but applies
          the override's ``tool_filter`` (if set) in place of the registry's.

        Args:
            refs: The list of MCP references from ``AgentConfig.mcps``.
            registry: The ``mcp_servers`` dict from ``AgentTeamConfig``.
            settings: Used for ``${VAR}`` header placeholder resolution.

        Returns:
            List of ``crewai.mcp`` server config objects ready to be passed
            to ``crewai.Agent(mcps=[...])``.
        """
        from src.config.agent_config import MCPServerRef  # local import avoids circular dep

        result: List[Union[MCPServerHTTP, MCPServerSSE, MCPServerStdio]] = []
        for ref in refs:
            if isinstance(ref, str):
                server_id = ref
                tool_filter_override = None
            else:  # MCPServerRef
                server_id = ref.server
                tool_filter_override = ref.tool_filter

            defn = registry[server_id]  # already validated at load time
            effective_filter = (
                tool_filter_override
                if tool_filter_override is not None
                else defn.tool_filter
            )
            crewai_cfg = cls._definition_to_crewai(defn, effective_filter, settings)
            result.append(crewai_cfg)
            logger.debug(
                "Resolved MCP server '%s' → %s (tools=%s).",
                server_id,
                type(crewai_cfg).__name__,
                effective_filter or "all",
            )
        return result

    @classmethod
    def _definition_to_crewai(
        cls,
        defn: "MCPServerDefinition",
        tool_filter: Optional[List[str]],
        settings: "AppSettings",
    ) -> Union[MCPServerHTTP, MCPServerSSE, MCPServerStdio]:
        """Convert one :class:`~src.config.agent_config.MCPServerDefinition`
        into the matching ``crewai.mcp`` config object.

        Args:
            defn: The registry entry to convert.
            tool_filter: Effective allow-list of tool names (may be ``None``
                meaning all tools are allowed).
            settings: Used for header placeholder resolution.

        Returns:
            A ``crewai.mcp.MCPServerHTTP``, ``MCPServerSSE``, or
            ``MCPServerStdio`` instance.
        """
        filter_obj = (
            create_static_tool_filter(allowed_tool_names=tool_filter)
            if tool_filter
            else None
        )
        headers = cls._resolve_headers(defn.headers, settings) if defn.headers else None

        if defn.type == "http":
            return MCPServerHTTP(
                url=defn.url,
                headers=headers,
                streamable=defn.streamable,
                tool_filter=filter_obj,
                cache_tools_list=defn.cache_tools_list,
            )
        elif defn.type == "sse":
            return MCPServerSSE(
                url=defn.url,
                headers=headers,
                tool_filter=filter_obj,
                cache_tools_list=defn.cache_tools_list,
            )
        else:  # stdio
            return MCPServerStdio(
                command=defn.command,
                args=defn.args,
                env=defn.env,
                tool_filter=filter_obj,
                cache_tools_list=defn.cache_tools_list,
            )

    @staticmethod
    def _resolve_headers(
        headers: Optional[Dict[str, str]],
        settings: "AppSettings",
    ) -> Optional[Dict[str, str]]:
        """Resolve ``${VAR_NAME}`` placeholders in MCP server header values.

        Placeholders that cannot be resolved are left unchanged so that
        downstream validation can detect them.

        Args:
            headers: Raw header dict that may contain ``${…}`` tokens.
            settings: :class:`~src.config.settings.AppSettings` used as the
                primary resolution source.

        Returns:
            A new dict with placeholder tokens replaced, or ``None`` if
            *headers* was ``None`` or empty.
        """
        if not headers:
            return headers

        def _sub(match: re.Match[str]) -> str:
            var_name = match.group(1)
            from pydantic import SecretStr
            field_val = getattr(settings, var_name, None)
            if field_val is not None:
                return (
                    field_val.get_secret_value()
                    if isinstance(field_val, SecretStr)
                    else str(field_val)
                )
            env_val = os.environ.get(var_name)
            if env_val is not None:
                return env_val
            logger.warning(
                "MCP header placeholder '${%s}' could not be resolved in "
                "AgentFactory — variable not found in AppSettings or environment.",
                var_name,
            )
            return match.group(0)

        return {k: _ENV_PLACEHOLDER_RE.sub(_sub, v) for k, v in headers.items()}
