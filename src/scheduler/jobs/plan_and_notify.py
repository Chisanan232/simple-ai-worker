"""
Plan-and-notify scheduler job (Phase 9).

Provides :func:`plan_and_notify_job`, an APScheduler interval job that
implements the **Dev Agent planning loop** for the Dev Lead planning workflow.

The job runs in two modes on every tick:

**Mode 1 — Initial Planning** (tickets in ``OPEN_FOR_DEV`` status)
    1. Fetch all tickets in the ``OPEN_FOR_DEV`` status from JIRA/ClickUp.
    2. For each new ticket, dispatch a Dev Agent crew to:
       a. Read the ticket description and acceptance criteria.
       b. Design a detailed development plan (Markdown format).
       c. Post the plan as a rich Markdown comment on the ticket.
       d. Post a second comment notifying the human engineer to review the plan.
    3. No code is written. No PRs are opened. No ticket status transition occurs
       (the human must set the ticket to ``IN PLANNING`` after reviewing).

**Mode 2 — Plan Revision** (tickets in ``IN_PLANNING`` status)
    1. Fetch all tickets in the ``IN_PLANNING`` status from JIRA/ClickUp.
    2. For each ticket, fetch its comments via REST API.
    3. If new human comments exist since the last plan-revision watermark,
       dispatch a Dev Agent crew to:
       a. Read the new comments.
       b. Revise the development plan accordingly.
       c. Re-post the updated plan as a new Markdown comment.
       d. Notify the human that the plan has been revised and is ready for
          re-review.

Business Rules Enforced
-----------------------
- **BR-8:**  Dev Agent must NOT write any code, open a PR, or transition the
  ticket to ``IN PROGRESS`` / ``ACCEPTED`` while in planning mode.
- **BR-10:** ``IN_PLANNING`` is ``human_only: true`` — the AI never transitions
  a ticket to this status.
- **BR-1:**  ``SCAN_FOR_WORK`` (``ACCEPTED``) is never written by the AI.

Shared State
------------
``_in_planning_tickets``: Set[str]
    Dispatch guard — prevents a second planning crew from being submitted for
    the same ticket while one is already running.  Cleared in the ``finally``
    block of the crew worker functions.

``_plan_comment_watermarks``: Dict[str, float]
    Per-ticket Unix timestamp of the most recent comment that was processed
    by the plan-revision crew.  Used to detect genuinely new human comments.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Dict, List, Optional, Set

from crewai import Agent, Task

from src.crew.builder import CrewBuilder
from src.ticket.models import TicketRecord
from src.ticket.rest_client import TicketFetchError
from src.ticket.workflow import WorkflowOperation

if TYPE_CHECKING:
    from src.agents.registry import AgentRegistry
    from src.config.settings import AppSettings
    from src.ticket.registry import TrackerRegistry
    from src.ticket.workflow import WorkflowConfig

__all__: List[str] = ["plan_and_notify_job"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

# Dispatch guard: ticket IDs currently being planned / revised.
_in_planning_tickets: Set[str] = set()

# Per-ticket watermark: Unix timestamp of the last comment processed.
_plan_comment_watermarks: Dict[str, float] = {}


# ---------------------------------------------------------------------------
# Task prompt templates
# ---------------------------------------------------------------------------

_INITIAL_PLAN_TASK_TEMPLATE: str = """
You are a senior software developer tasked with creating a development plan
for the following ticket.

Ticket ID:     {ticket_id}
Ticket Source: {source}

Steps:
1. Fetch the full ticket details:
   - JIRA: use jira/get_issue for ticket '{ticket_id}'
   - ClickUp: use clickup/get_task for ticket '{ticket_id}'

2. Read the title, description, and acceptance criteria thoroughly.

3. Design a comprehensive **development plan** in Markdown format.
   The plan must include:
   - ## Overview
     Brief summary of what will be built and why.
   - ## Technical Approach
     How you will implement the feature (architecture decisions, patterns,
     libraries/frameworks, APIs to call or modify).
   - ## Implementation Steps
     Numbered list of concrete implementation steps in order.
   - ## File Changes
     List of files to create / modify / delete.
   - ## Testing Strategy
     Unit tests, integration tests, or E2E tests that will be added.
   - ## Risks & Open Questions
     Any technical risks or questions that need clarification before coding.
   - ## Estimated Effort
     S / M / L / XL (with brief justification).

4. Post the development plan as a comment on the ticket:
   - JIRA: use jira/add_comment on ticket '{ticket_id}' with the full
     Markdown plan (begin the comment with "## 📋 Development Plan (v1)").
   - ClickUp: use clickup/add_comment on task '{ticket_id}'.

5. Post a second notification comment on the ticket asking the assigned
   engineer (or any reviewer) to review the plan.  Example:
   "👋 @team — I have drafted a development plan for this ticket.
    Please review the plan above and leave your feedback as comments.
    Once you are satisfied, set this ticket to **IN PLANNING** status.
    When the plan is approved, set the ticket to **ACCEPTED** and I will
    begin development."

IMPORTANT GUARDRAILS:
  - Do NOT write any code.
  - Do NOT open a GitHub Pull Request.
  - Do NOT transition the ticket to IN PROGRESS, ACCEPTED, or any other status.
  - Do NOT set any ticket to the ACCEPTED/scan_for_work status (BR-1).
  - Only post comments — the human will set the ticket state.
"""

_INITIAL_PLAN_TASK_EXPECTED_OUTPUT: str = (
    "Confirmation that the development plan was posted as a comment on ticket "
    "{ticket_id} and a notification comment was also posted asking the engineer "
    "to review the plan.  No code was written.  No ticket status was changed."
)

_REVISE_PLAN_TASK_TEMPLATE: str = """
You are a senior software developer.  A human engineer has reviewed your
development plan for ticket {ticket_id} and left feedback comments.

Ticket ID:     {ticket_id}
Ticket Source: {source}

New human feedback comments (posted since your last plan version):
---
{comments_text}
---

Steps:
1. Fetch the full ticket details to refresh context:
   - JIRA: use jira/get_issue for ticket '{ticket_id}'
   - ClickUp: use clickup/get_task for ticket '{ticket_id}'

2. Read each feedback comment carefully and understand what changes the
   reviewer is requesting.

3. Revise your development plan to address all the feedback.  Produce an
   updated Markdown plan using the same structure as before:
   ## 📋 Development Plan (v{version})
   - ## Overview
   - ## Technical Approach
   - ## Implementation Steps
   - ## File Changes
   - ## Testing Strategy
   - ## Risks & Open Questions
   - ## Estimated Effort
   Include a "## Revision Notes" section at the top that summarises what
   changed from the previous version in response to the reviewer's comments.

4. Post the revised plan as a new comment on the ticket:
   - JIRA: use jira/add_comment on ticket '{ticket_id}'.
   - ClickUp: use clickup/add_comment on task '{ticket_id}'.

5. Post a follow-up notification comment:
   "✅ Plan updated (v{version}) in response to your feedback.
    Please review the revised plan above and let me know if there are
    any remaining concerns.  Once satisfied, set the ticket to ACCEPTED
    and I will begin development."

IMPORTANT GUARDRAILS:
  - Do NOT write any code.
  - Do NOT open a GitHub Pull Request.
  - Do NOT transition the ticket to IN PROGRESS, ACCEPTED, or any other status.
  - Do NOT set any ticket to the ACCEPTED/scan_for_work status (BR-1).
  - Only post comments — the human will set the ticket state.
"""

_REVISE_PLAN_TASK_EXPECTED_OUTPUT: str = (
    "Confirmation that the revised development plan (with revision notes) was "
    "posted as a comment on ticket {ticket_id} and a follow-up notification "
    "comment was posted.  No code was written.  No ticket status was changed."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_agent_state(agent: "object") -> None:
    """Clear the agent executor's message history before a new crew run.

    When the same :class:`~crewai.Agent` object is reused across multiple
    sequential ``crew.kickoff()`` calls (e.g. planning three tickets one after
    another in a single ``ThreadPoolExecutor`` worker thread), CrewAI's
    ``CrewAgentExecutor._setup_messages()`` **appends** new messages to the
    existing list rather than clearing it.  This causes the conversation
    history to grow across runs, which makes the LLM perceive it is partway
    through a long conversation rather than starting fresh.

    Clearing ``agent_executor.messages`` before each new crew run ensures
    each ticket gets an independent, clean planning session.

    Args:
        agent: A ``crewai.Agent`` (typed as ``object`` to avoid a hard import
               that would create a circular dependency).  The attribute access
               is guarded so that non-Agent objects are silently ignored.
    """
    executor = getattr(agent, "agent_executor", None)
    if executor is not None and hasattr(executor, "messages"):
        try:
            executor.messages.clear()
            logger.debug(
                "_reset_agent_state: cleared agent_executor.messages for agent %r.",
                getattr(agent, "role", repr(agent)),
            )
        except Exception:  # noqa: BLE001
            logger.debug(
                "_reset_agent_state: could not clear messages for agent %r — skipping.",
                getattr(agent, "role", repr(agent)),
            )


# ---------------------------------------------------------------------------
# Worker: initial plan
# ---------------------------------------------------------------------------


def _create_initial_plan(
    ticket_id: str,
    source: str,
    title: str,
    registry: "AgentRegistry",
) -> None:
    """Create and post an initial development plan for *ticket_id*.

    Runs inside a ``ThreadPoolExecutor`` worker thread.

    Args:
        ticket_id: The ticket identifier.
        source:    ``"jira"`` or ``"clickup"``.
        title:     Human-readable ticket title for logging.
        registry:  The shared :class:`~src.agents.registry.AgentRegistry`.
    """
    try:
        dev_agent: Agent = registry["dev_agent"]
    except KeyError:
        logger.error(
            "_create_initial_plan: 'dev_agent' not found in registry for ticket %s.",
            ticket_id,
        )
        _in_planning_tickets.discard(ticket_id)
        return

    task = Task(
        description=_INITIAL_PLAN_TASK_TEMPLATE.format(
            ticket_id=ticket_id,
            source=source,
        ),
        expected_output=_INITIAL_PLAN_TASK_EXPECTED_OUTPUT.format(
            ticket_id=ticket_id,
        ),
        agent=dev_agent,
    )
    # Clear stale conversation history from any previous crew run on this agent.
    # CrewAgentExecutor._setup_messages() appends (not replaces) messages, so
    # without this reset the agent would start each new ticket mid-conversation.
    _reset_agent_state(dev_agent)
    crew = CrewBuilder.build(
        agents=[dev_agent],
        tasks=[task],
        process="sequential",
    )

    try:
        logger.info(
            "_create_initial_plan: planning crew starting for ticket %s (%s) — '%s'.",
            ticket_id,
            source,
            title,
        )
        result = crew.kickoff()
        logger.info(
            "_create_initial_plan: completed for ticket %s. Preview: %.300s",
            ticket_id,
            str(result),
        )
    except Exception:  # noqa: BLE001
        logger.exception("_create_initial_plan: crew failed for ticket %s.", ticket_id)
    finally:
        _in_planning_tickets.discard(ticket_id)
        logger.debug("_create_initial_plan: dispatch guard released for ticket %s.", ticket_id)


# ---------------------------------------------------------------------------
# Worker: plan revision
# ---------------------------------------------------------------------------


def _revise_plan(
    ticket_id: str,
    source: str,
    comments_text: str,
    version: int,
    registry: "AgentRegistry",
    latest_comment_ts: float,
) -> None:
    """Revise the development plan for *ticket_id* based on human comments.

    Runs inside a ``ThreadPoolExecutor`` worker thread.

    Args:
        ticket_id:          The ticket identifier.
        source:             ``"jira"`` or ``"clickup"``.
        comments_text:      Formatted string of new human comments.
        version:            Plan version number (for the revision heading).
        registry:           The shared :class:`~src.agents.registry.AgentRegistry`.
        latest_comment_ts:  Timestamp of the most recent comment in *comments_text*.
                            Stored in ``_plan_comment_watermarks`` on success.
    """
    try:
        dev_agent = registry["dev_agent"]
    except KeyError:
        logger.error(
            "_revise_plan: 'dev_agent' not found in registry for ticket %s.",
            ticket_id,
        )
        _in_planning_tickets.discard(ticket_id)
        return

    task = Task(
        description=_REVISE_PLAN_TASK_TEMPLATE.format(
            ticket_id=ticket_id,
            source=source,
            comments_text=comments_text,
            version=version,
        ),
        expected_output=_REVISE_PLAN_TASK_EXPECTED_OUTPUT.format(
            ticket_id=ticket_id,
        ),
        agent=dev_agent,
    )
    # Clear stale conversation history from any previous crew run on this agent.
    _reset_agent_state(dev_agent)
    crew = CrewBuilder.build(
        agents=[dev_agent],
        tasks=[task],
        process="sequential",
    )

    try:
        logger.info(
            "_revise_plan: revision crew (v%d) starting for ticket %s (%s).",
            version,
            ticket_id,
            source,
        )
        result = crew.kickoff()
        logger.info(
            "_revise_plan: completed for ticket %s v%d. Preview: %.300s",
            ticket_id,
            version,
            str(result),
        )
        # Advance the watermark so we don't re-process these comments.
        _plan_comment_watermarks[ticket_id] = latest_comment_ts

    except Exception:  # noqa: BLE001
        logger.exception("_revise_plan: crew failed for ticket %s.", ticket_id)
    finally:
        _in_planning_tickets.discard(ticket_id)
        logger.debug("_revise_plan: dispatch guard released for ticket %s.", ticket_id)


# ---------------------------------------------------------------------------
# Main job
# ---------------------------------------------------------------------------


def plan_and_notify_job(
    registry: "AgentRegistry",
    settings: "AppSettings",  # noqa: ARG001  (reserved for future use)
    executor: ThreadPoolExecutor,
    workflow: Optional["WorkflowConfig"] = None,
    tracker_registry: Optional["TrackerRegistry"] = None,
) -> None:
    """APScheduler job: plan development for OPEN tickets and revise plans for IN PLANNING tickets.

    Runs in two modes on every tick:

    **Mode 1** — Initial planning for tickets in ``OPEN_FOR_DEV`` status.
    **Mode 2** — Plan revision for tickets in ``IN_PLANNING`` status when new
    human comments are detected.

    Args:
        registry:         The shared :class:`~src.agents.registry.AgentRegistry`.
        settings:         The application :class:`~src.config.settings.AppSettings`
                          singleton (reserved for future use).
        executor:         The bounded :class:`~concurrent.futures.ThreadPoolExecutor`
                          shared across all agent dispatches.
        workflow:         Optional :class:`~src.ticket.workflow.WorkflowConfig`.
                          When ``None``, skips all planning (logs a warning).
        tracker_registry: Optional :class:`~src.ticket.registry.TrackerRegistry`.
                          When ``None``, skips all planning (logs a warning).
    """
    if workflow is None:
        logger.warning(
            "plan_and_notify_job: no WorkflowConfig provided — skipping. "
            "Add a 'workflow:' block with 'open_for_dev' and 'in_planning' to agents.yaml."
        )
        return

    if tracker_registry is None:
        logger.warning("plan_and_notify_job: no TrackerRegistry provided — skipping.")
        return

    # Verify dev_agent is accessible before doing any REST calls.
    try:
        registry["dev_agent"]
    except KeyError:
        logger.error(
            "plan_and_notify_job: 'dev_agent' not found in registry — " "available ids: %s. Skipping run.",
            registry.agent_ids(),
        )
        return

    open_for_dev_status = workflow.status_for(WorkflowOperation.OPEN_FOR_DEV)
    in_planning_status = workflow.status_for(WorkflowOperation.IN_PLANNING)

    # ------------------------------------------------------------------
    # Mode 1: Initial planning for OPEN_FOR_DEV tickets
    # ------------------------------------------------------------------
    if open_for_dev_status:
        logger.info("plan_and_notify_job: scanning for OPEN_FOR_DEV tickets (status=%r).", open_for_dev_status)
        open_tickets: List[TicketRecord] = []
        for source in ("jira", "clickup"):
            try:
                tracker = tracker_registry.get(source)
                fetched = tracker.fetch_tickets_for_operation(WorkflowOperation.OPEN_FOR_DEV)
                logger.info(
                    "plan_and_notify_job: %s returned %d OPEN_FOR_DEV ticket(s).",
                    source,
                    len(fetched),
                )
                open_tickets.extend(fetched)
            except TicketFetchError:
                logger.exception(
                    "plan_and_notify_job: REST fetch failed for source '%s' (OPEN_FOR_DEV) — skipping.",
                    source,
                )
            except ValueError:
                logger.warning(
                    "plan_and_notify_job: source '%s' is not configured — skipping.",
                    source,
                )

        for ticket in open_tickets:
            tid = ticket.id
            if tid in _in_planning_tickets:
                logger.debug(
                    "plan_and_notify_job: ticket %s already in planning — skipped.",
                    tid,
                )
                continue
            _in_planning_tickets.add(tid)
            logger.info(
                "plan_and_notify_job: dispatching initial plan crew for ticket %s (%s).",
                tid,
                ticket.source,
            )
            future: Future[None] = executor.submit(
                _create_initial_plan,
                tid,
                ticket.source,
                ticket.title,
                registry,
            )

            def _on_done(fut: Future[None], t: str = tid) -> None:
                exc = fut.exception()
                if exc:
                    logger.error(
                        "plan_and_notify_job: initial plan future for ticket %s raised: %s",
                        t,
                        exc,
                    )

            future.add_done_callback(_on_done)
    else:
        logger.debug("plan_and_notify_job: OPEN_FOR_DEV status not configured — skipping Mode 1.")

    # ------------------------------------------------------------------
    # Mode 2: Plan revision for IN_PLANNING tickets with new comments
    # ------------------------------------------------------------------
    if in_planning_status:
        logger.info("plan_and_notify_job: scanning for IN_PLANNING tickets (status=%r).", in_planning_status)
        in_planning_tickets: List[TicketRecord] = []
        for source in ("jira", "clickup"):
            try:
                tracker = tracker_registry.get(source)
                fetched = tracker.fetch_tickets_for_operation(WorkflowOperation.IN_PLANNING)
                logger.info(
                    "plan_and_notify_job: %s returned %d IN_PLANNING ticket(s).",
                    source,
                    len(fetched),
                )
                in_planning_tickets.extend(fetched)
            except TicketFetchError:
                logger.exception(
                    "plan_and_notify_job: REST fetch failed for source '%s' (IN_PLANNING) — skipping.",
                    source,
                )
            except ValueError:
                logger.warning(
                    "plan_and_notify_job: source '%s' not configured (IN_PLANNING) — skipping.",
                    source,
                )

        for ticket in in_planning_tickets:
            tid = ticket.id
            if tid in _in_planning_tickets:
                logger.debug(
                    "plan_and_notify_job: ticket %s already being processed — skipped.",
                    tid,
                )
                continue

            # Fetch comments via REST API (no LLM).
            try:
                tracker = tracker_registry.get(ticket.source)
                comments = tracker.fetch_ticket_comments(tid)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "plan_and_notify_job: failed to fetch comments for ticket %s — skipping.",
                    tid,
                )
                continue

            if not comments:
                logger.debug(
                    "plan_and_notify_job: no comments on IN_PLANNING ticket %s — skipped.",
                    tid,
                )
                continue

            # Determine new comments since last watermark.
            watermark: float = _plan_comment_watermarks.get(tid, 0.0)
            new_comments = [c for c in comments if c.created_at > watermark]

            if not new_comments:
                logger.debug(
                    "plan_and_notify_job: no new comments since watermark=%.0f for ticket %s — skipped.",
                    watermark,
                    tid,
                )
                continue

            # Count existing plan versions by counting AI revision comments
            # (rough heuristic: one revision per batch of new comments).
            existing_revisions = len([c for c in comments if c.created_at <= watermark])
            version = existing_revisions + 2  # v1 = initial, v2+ = revisions

            comments_text = "\n".join(
                f"[{i + 1}] {c.author} ({time.strftime('%Y-%m-%d %H:%M', time.gmtime(c.created_at))}):\n"
                f"    {c.body}"
                for i, c in enumerate(new_comments)
            )
            latest_ts = max(c.created_at for c in new_comments)

            _in_planning_tickets.add(tid)
            logger.info(
                "plan_and_notify_job: dispatching plan-revision crew (v%d) for ticket %s (%s) " "— %d new comment(s).",
                version,
                tid,
                ticket.source,
                len(new_comments),
            )
            rev_future: Future[None] = executor.submit(
                _revise_plan,
                tid,
                ticket.source,
                comments_text,
                version,
                registry,
                latest_ts,
            )

            def _on_rev_done(fut: Future[None], t: str = tid) -> None:
                exc = fut.exception()
                if exc:
                    logger.error(
                        "plan_and_notify_job: revision future for ticket %s raised: %s",
                        t,
                        exc,
                    )

            rev_future.add_done_callback(_on_rev_done)
    else:
        logger.debug("plan_and_notify_job: IN_PLANNING status not configured — skipping Mode 2.")
