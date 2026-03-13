"""
Pydantic models for the YAML-based agent configuration file.

These models define the schema for ``config/agents.yaml`` — the declarative
agent-team definition loaded at application startup.  All values are validated
by Pydantic at parse time, so the application fails fast with a clear error
message if the file is malformed or incomplete.

Schema hierarchy
----------------

.. code-block:: text

    AgentTeamConfig
    ├── mcp_servers: dict[str, MCPServerDefinition]   ← MCP registry
    │   └── MCPServerDefinition
    │       ├── type: "http" | "sse" | "stdio"
    │       ├── url, headers, streamable, tool_filter, cache_tools_list
    │       └── command, args, env  (stdio only)
    └── agents: list[AgentConfig]
        ├── id, role, goal, backstory, allow_delegation, verbose, apps
        ├── mcps: list[str | MCPServerRef]            ← MCP references
        │   └── MCPServerRef
        │       ├── server: str  (key in mcp_servers)
        │       └── tool_filter: list[str] | None
        └── llm: LLMConfig
            ├── provider, model
            └── options: LLMOptions
                    temperature, max_tokens, top_p, timeout

Usage::

    from src.config.agent_config import AgentTeamConfig

    raw: dict = yaml.safe_load(open("config/agents.yaml"))
    config = AgentTeamConfig.model_validate(raw)
    for agent in config.agents:
        print(agent.id, agent.llm.model, agent.mcps)

Design decisions
----------------
- ``LLMOptions`` fields all have sensible defaults so that an agent YAML entry
  only needs to specify the values it wants to override.
- ``AgentConfig.apps`` is intentionally ``list[str]`` — the strings are CrewAI
  Enterprise action IDs (e.g. ``"jira/create_issue"``) passed directly to
  ``crewai.Agent(apps=[...])``.  Kept for backward compatibility; will be
  deprecated once MCP migration is complete.
- ``AgentConfig.mcps`` is a list of MCP server ID strings (plain references)
  or :class:`MCPServerRef` override mappings.  IDs must match keys in
  ``AgentTeamConfig.mcp_servers``.
- ``AgentTeamConfig.mcp_servers`` is the single-source-of-truth registry for
  all MCP server connection details.  Agents reference servers by ID so that
  connection config is never duplicated across agents.
- ``AgentTeamConfig`` uses ``model_validator`` to enforce unique agent IDs and
  that every ID in every agent's ``mcps`` list resolves to a known
  ``mcp_servers`` entry (fail-fast at load time).
"""

from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

__all__: List[str] = [
    "LLMOptions",
    "LLMConfig",
    "MCPServerDefinition",
    "MCPServerRef",
    "MCPRef",
    "AgentConfig",
    "AgentTeamConfig",
]


class LLMOptions(BaseModel):
    """Fine-grained generation parameters for a language model.

    All fields have defaults so that partial overrides are allowed in YAML.

    Attributes:
        temperature: Sampling temperature.  Higher values produce more
            creative/random output; lower values are more deterministic.
            Range ``[0.0, 2.0]``.  Defaults to ``0.7``.
        max_tokens: Maximum number of tokens the model may generate in a
            single response.  Defaults to ``4096``.
        top_p: Nucleus sampling parameter.  Only the smallest set of tokens
            whose cumulative probability exceeds ``top_p`` are considered.
            Range ``(0.0, 1.0]``.  Defaults to ``1.0`` (disabled).
        timeout: HTTP timeout in seconds for a single model call.
            Defaults to ``120``.
    """

    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    timeout: int = Field(default=120, gt=0)


class LLMConfig(BaseModel):
    """Configuration for the language model used by an agent.

    Attributes:
        provider: The LLM provider to use.  Must be one of ``"openai"`` or
            ``"anthropic"``.
        model: The model identifier string passed to the CrewAI ``LLM``
            constructor (e.g. ``"gpt-4o"``, ``"claude-3-5-sonnet-latest"``).
        options: Fine-grained generation parameters.  All sub-fields have
            defaults, so this block is optional in YAML.
    """

    provider: Literal["openai", "anthropic"]
    model: str = Field(min_length=1)
    options: LLMOptions = Field(default_factory=LLMOptions)

    @field_validator("model")
    @classmethod
    def model_name_must_not_be_blank(cls, value: str) -> str:
        """Reject whitespace-only model name strings."""
        if not value.strip():
            raise ValueError("LLMConfig.model must not be blank.")
        return value


# ---------------------------------------------------------------------------
# MCP Server Models
# ---------------------------------------------------------------------------


class MCPServerDefinition(BaseModel):
    """Definition of one MCP server in the top-level ``mcp_servers`` registry.

    Identified by its dict key in :attr:`AgentTeamConfig.mcp_servers`.
    The ``type`` field acts as a discriminator selecting which transport the
    server uses; only the fields relevant to that transport need to be set.

    Attributes:
        type: Transport type.  One of ``"http"``, ``"sse"``, or ``"stdio"``.
        url: Base URL of the MCP server endpoint.
            Required for ``"http"`` and ``"sse"`` transports.
            Example: ``"http://127.0.0.1:8100/mcp"``.
        headers: Optional HTTP headers sent with every request.
            Useful for ``Authorization: Bearer <token>`` patterns.
            ``${VAR_NAME}`` placeholders are resolved from ``AppSettings``
            / environment variables at agent-build time.
            Only used for ``"http"`` and ``"sse"`` transports.
        streamable: Whether to use streamable HTTP transport (chunked
            streaming POST).  Defaults to ``True``.  Only meaningful for
            the ``"http"`` transport.
        tool_filter: Optional allow-list of tool names to expose from this
            server.  When ``None`` (the default), all tools advertised by the
            server are available.  Agents can further narrow this list via an
            :class:`MCPServerRef` override.
        cache_tools_list: Whether to cache the tool schema list returned by
            ``list_tools()`` after the first connection.  Useful for stable
            servers (e.g. JIRA, GitHub) to avoid repeated round-trips.
            Defaults to ``False``.
        command: Executable to launch for ``"stdio"`` servers
            (e.g. ``"python"``, ``"npx"``).
            Required when ``type == "stdio"``.
        args: Arguments to pass to the ``command``.
            Only used for ``"stdio"`` transport.
        env: Environment variables to inject into the subprocess.
            Only used for ``"stdio"`` transport.
    """

    type: Literal["http", "sse", "stdio"]
    # http / sse fields
    url: str | None = None
    headers: Dict[str, str] | None = None
    streamable: bool = True
    # all transports
    tool_filter: List[str] | None = None
    cache_tools_list: bool = False
    # stdio fields
    command: str | None = None
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] | None = None

    @model_validator(mode="after")
    def validate_required_fields_per_type(self) -> "MCPServerDefinition":
        """Ensure transport-specific required fields are present."""
        if self.type in ("http", "sse"):
            if not self.url:
                raise ValueError(f"MCPServerDefinition with type='{self.type}' requires a non-empty 'url'.")
        if self.type == "stdio":
            if not self.command:
                raise ValueError("MCPServerDefinition with type='stdio' requires a non-empty 'command'.")
        return self


class MCPServerRef(BaseModel):
    """Per-agent override for a registry MCP server.

    Allows an agent to reference a server from ``mcp_servers`` while
    overriding only the ``tool_filter`` — keeping the registry as the
    single source of truth for URL, transport, and authentication.

    Attributes:
        server: The ID (dict key) of the target entry in
            :attr:`AgentTeamConfig.mcp_servers`.
        tool_filter: Optional allow-list of tool names for *this agent only*.
            Overrides the ``tool_filter`` defined in the registry entry.
            When ``None``, the registry's ``tool_filter`` is used as-is.
    """

    server: str = Field(min_length=1)
    tool_filter: List[str] | None = None


# Type alias: a plain server-ID string  OR  an MCPServerRef override mapping.
MCPRef = str | MCPServerRef


class AgentConfig(BaseModel):
    """Configuration for a single AI agent.

    Attributes:
        id: Unique identifier for this agent within the team.  Used as a
            dictionary key in :class:`~src.agents.registry.AgentRegistry`.
            Examples: ``"planner"``, ``"dev_lead"``, ``"dev_agent"``.
        role: Short role title passed to the CrewAI ``Agent`` constructor
            (e.g. ``"Product Planner"``).
        goal: One-sentence goal statement that tells the agent what it is
            trying to achieve.
        backstory: Multi-sentence persona / background story fed to the LLM
            as part of the system prompt to shape the agent's behaviour.
        llm: Language model configuration (provider, model name, options).
        apps: List of CrewAI Enterprise native action IDs to attach to this
            agent (e.g. ``["jira/create_issue", "slack/reply_to_thread"]``).
            Passed directly to ``crewai.Agent(apps=[...])``.
            Kept for backward compatibility — prefer ``mcps`` for new
            integrations.
        mcps: List of MCP server references.  Each entry is either a plain
            string (the ID of an entry in :attr:`AgentTeamConfig.mcp_servers`)
            or an :class:`MCPServerRef` mapping that references a server ID
            and optionally overrides its ``tool_filter`` for this agent.
            Empty list is valid for agents that need no MCP tools.
        allow_delegation: Whether this agent may delegate tasks to other
            agents in the crew.  Defaults to ``False``.
        verbose: Whether the CrewAI agent prints verbose reasoning output.
            Defaults to ``False``.
    """

    id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    backstory: str = Field(min_length=1)
    llm: LLMConfig
    apps: List[str] = Field(default_factory=list)
    mcps: List["MCPRef"] = Field(default_factory=list)
    allow_delegation: bool = False
    verbose: bool = False

    @field_validator("id")
    @classmethod
    def id_must_not_be_blank(cls, value: str) -> str:
        """Reject whitespace-only id strings."""
        if not value.strip():
            raise ValueError("AgentConfig.id must not be blank.")
        return value

    @field_validator("apps", mode="before")
    @classmethod
    def apps_items_must_not_be_blank(cls, value: object) -> object:
        """Reject lists that contain blank or whitespace-only action IDs."""
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and not item.strip():
                    raise ValueError("AgentConfig.apps must not contain blank action ID strings.")
        return value


class AgentTeamConfig(BaseModel):
    """Top-level configuration for the full agent team.

    Loaded from ``config/agents.yaml`` (or the path set by
    ``AppSettings.AGENT_CONFIG_PATH``) at application startup.

    Attributes:
        process: CrewAI crew execution process.  ``"sequential"`` runs
            tasks one after another; ``"hierarchical"`` uses a manager
            agent to delegate sub-tasks.  Defaults to ``"sequential"``.
        mcp_servers: Registry of MCP server definitions, keyed by a short
            unique ID (e.g. ``"jira"``, ``"clickup"``).  Agents reference
            these IDs in their ``mcps`` list.  Defaults to an empty dict so
            that configs without MCP servers remain valid.
        agents: Ordered list of agent configurations.  At least one agent
            is required.  All ``id`` values must be unique.
    """

    process: Literal["sequential", "hierarchical"] = "sequential"
    mcp_servers: Dict[str, MCPServerDefinition] = Field(default_factory=dict)
    agents: List[AgentConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def agent_ids_must_be_unique(self) -> "AgentTeamConfig":
        """Raise ``ValueError`` if any two agents share the same ``id``."""
        seen: List[str] = []
        duplicates: List[str] = []
        for agent in self.agents:
            if agent.id in seen:
                duplicates.append(agent.id)
            else:
                seen.append(agent.id)
        if duplicates:
            raise ValueError(
                f"AgentTeamConfig.agents contains duplicate id(s): "
                f"{sorted(set(duplicates))}.  Every agent id must be unique."
            )
        return self

    @model_validator(mode="after")
    def mcp_refs_must_resolve(self) -> "AgentTeamConfig":
        """Raise ``ValueError`` if any agent references an unknown MCP server ID.

        Validates that every plain-string entry and every ``MCPServerRef.server``
        in every agent's ``mcps`` list is present as a key in ``mcp_servers``.
        Runs at load time so typos fail immediately with a clear message.
        """
        known = set(self.mcp_servers.keys())
        for agent in self.agents:
            for ref in agent.mcps:
                server_id = ref if isinstance(ref, str) else ref.server
                if server_id not in known:
                    raise ValueError(
                        f"Agent '{agent.id}' references unknown MCP server "
                        f"'{server_id}'.  "
                        f"Available servers: {sorted(known) if known else '(none defined)'}."
                    )
        return self
