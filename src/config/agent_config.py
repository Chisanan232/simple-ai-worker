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
    └── agents: list[AgentConfig]
        ├── id, role, goal, backstory, allow_delegation, verbose, apps
        └── llm: LLMConfig
            ├── provider, model
            └── options: LLMOptions
                    temperature, max_tokens, top_p, timeout

Usage::

    from src.config.agent_config import AgentTeamConfig

    raw: dict = yaml.safe_load(open("config/agents.yaml"))
    config = AgentTeamConfig.model_validate(raw)
    for agent in config.agents:
        print(agent.id, agent.llm.model)

Design decisions
----------------
- ``LLMOptions`` fields all have sensible defaults so that an agent YAML entry
  only needs to specify the values it wants to override.
- ``AgentConfig.apps`` is intentionally ``list[str]`` — the strings are CrewAI
  Enterprise action IDs (e.g. ``"jira/create_issue"``) passed directly to
  ``crewai.Agent(apps=[...])``.  No enum is used so that new action IDs can be
  added to the YAML without touching Python code.
- ``AgentTeamConfig`` uses a ``model_validator`` to enforce that every ``id``
  in the ``agents`` list is unique.
- ``LLMConfig`` uses a ``model_validator`` to enforce that ``provider`` is
  consistent with the credentials available in ``AppSettings`` (checked lazily
  at agent-build time rather than at YAML-load time to keep this module
  dependency-free from ``settings``).
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

__all__: List[str] = [
    "LLMOptions",
    "LLMConfig",
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
            Empty list is valid for agents that need no external tools.
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
                    raise ValueError(
                        "AgentConfig.apps must not contain blank action ID strings."
                    )
        return value


class AgentTeamConfig(BaseModel):
    """Top-level configuration for the full agent team.

    Loaded from ``config/agents.yaml`` (or the path set by
    ``AppSettings.AGENT_CONFIG_PATH``) at application startup.

    Attributes:
        process: CrewAI crew execution process.  ``"sequential"`` runs
            tasks one after another; ``"hierarchical"`` uses a manager
            agent to delegate sub-tasks.  Defaults to ``"sequential"``.
        agents: Ordered list of agent configurations.  At least one agent
            is required.  All ``id`` values must be unique.
    """

    process: Literal["sequential", "hierarchical"] = "sequential"
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
