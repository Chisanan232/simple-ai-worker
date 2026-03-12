"""
AI agents package for simple-ai-worker.

Public API
----------
- :class:`~src.agents.llm_factory.LLMFactory`   — builds ``crewai.LLM`` from config.
- :class:`~src.agents.factory.AgentFactory`      — builds ``crewai.Agent`` from config.
- :class:`~src.agents.registry.AgentRegistry`    — typed dict of built agents keyed by id.
- :func:`~src.agents.registry.build_registry`    — convenience builder from team config.
"""

from __future__ import annotations

from typing import List

from .factory import AgentFactory
from .llm_factory import LLMFactory
from .registry import AgentRegistry, build_registry

__all__: List[str] = [
    "LLMFactory",
    "AgentFactory",
    "AgentRegistry",
    "build_registry",
]
