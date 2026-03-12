"""
Crew package for simple-ai-worker.

Public API
----------
- :class:`~src.crew.builder.CrewBuilder` — builds ``crewai.Crew`` from
  agents, tasks, and a process string.
"""

from __future__ import annotations

from typing import List

from .builder import CrewBuilder

__all__: List[str] = ["CrewBuilder"]
