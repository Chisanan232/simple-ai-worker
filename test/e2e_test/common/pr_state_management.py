"""PR state management utilities for E2E tests.

Provides helpers for seeding and managing the shared mutable state
in src.scheduler.jobs.scan_tickets (_open_prs, _prs_under_review).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def populate_pr_state(
    ticket_id: str,
    pr_url: str,
    age_seconds: float = 310.0,
) -> None:
    """Seed both _open_prs and _prs_under_review as if a PR was already opened.

    Used to simulate the state after scan_and_dispatch_job has executed
    and opened a PR, allowing subsequent jobs (pr_merge_watcher,
    pr_review_comment_handler) to operate on pre-existing PRs.

    Args:
        ticket_id: The ticket ID (e.g., 'PROJ-10', 'cu-001')
        pr_url: The GitHub PR URL
        age_seconds: How long ago the PR was opened (default 310s, past timeout)
    """
    import src.scheduler.jobs.scan_tickets as scan_mod
    from src.ticket.models import PRRecord

    scan_mod._open_prs[ticket_id] = PRRecord(
        ticket_id=ticket_id,
        pr_url=pr_url,
        opened_at_utc=time.time() - age_seconds,
    )
    scan_mod._prs_under_review[ticket_id] = pr_url


def populate_under_review(ticket_id: str, pr_url: str) -> None:
    """Seed only _prs_under_review without creating a PRRecord.

    Used when a PR is already in review state but we don't need to
    track its opening time (e.g., for review comment handler tests).

    Args:
        ticket_id: The ticket ID (e.g., 'PROJ-30', 'cu-022')
        pr_url: The GitHub PR URL
    """
    import src.scheduler.jobs.scan_tickets as scan_mod

    scan_mod._prs_under_review[ticket_id] = pr_url


def reset_pr_state() -> None:
    """Clear all PR state from scan_tickets module.

    Useful for test cleanup to ensure no state leaks between tests.
    """
    import src.scheduler.jobs.scan_tickets as scan_mod

    scan_mod._open_prs.clear()
    scan_mod._prs_under_review.clear()
