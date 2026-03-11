"""
Scheduler jobs package.

Each module in this package contains exactly one top-level job function
that APScheduler can invoke directly.  Job functions must be importable
at the module level (no closures) so APScheduler can serialise their
reference when needed.

Phase mapping:
    Phase 1 — :mod:`src.scheduler.jobs.hello_world`
    Phase 6 — ``scan_tickets``, ``planner_listener``, ``dev_lead_listener``
               (added in Phase 6)
"""

from __future__ import annotations

__all__: list[str] = []

