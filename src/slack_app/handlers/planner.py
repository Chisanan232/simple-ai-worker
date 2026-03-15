"""
Planner Slack Bolt handler (Phase 6 + Phase 10 — Idea-Discussion Workflow).

Provides :func:`planner_handler`, called by
:func:`~src.slack_app.router.role_router` when a Slack message contains the
``[planner]`` tag.

Execution flow
--------------
1. Strip the ``[planner]`` prefix from the message text.
2. Extract ``thread_ts`` from the event so replies land in the same thread.
3. Post an immediate "thinking…" acknowledgement so the human knows the bot
   is working (Slack requires ACK within 3 s; the real Crew runs async).
4. Submit Crew execution to the bounded :class:`~concurrent.futures.ThreadPoolExecutor`.
5. Crew result (or error message) is posted back to the thread via
   ``say(thread_ts=...)``.

The Planner handles three request types:

**Type A — Actionable Requirement:**
  A clear, well-defined requirement that the Planner converts immediately into
  a JIRA epic and ClickUp task.

**Type B — Idea Survey & Discussion:**
  A product idea at an early / exploratory stage.  The Planner reads the full
  Slack thread history, conducts a comprehensive business survey (marketing
  value, business model, MVP features, budget, etc.), and posts the plan as a
  rich Markdown message.  **No tickets are created until the human decides.**

**Type C — Idea Discussion Conclusion:**
  The human has made a final accept/reject decision.  The Planner posts a
  conclusion message, creates a REJECTED or OPEN ticket, and (on accept only)
  mentions the Dev Lead in a *new* Slack message to start the development
  planning discussion.

Business Rules
--------------
- **BR-11:** No tickets during the survey/discussion phase (Type B).
- **BR-12:** Reject path must NOT mention or notify the Dev Lead.
- **BR-13:** Dev Lead hand-off uses ``slack/send_message`` (new message), NOT
  ``reply_to_thread`` — to start a clean new discussion context.
- **BR-14:** A REJECTED ticket must have status ``"REJECTED"``; it must never
  be set to ``"ACCEPTED"`` or ``"OPEN"``.
- **BR-1:**  ``ACCEPTED`` status must never be written by any AI agent.

Note: The Planner Agent also uses ``slack/reply_to_thread`` and
``slack/send_message`` as part of its native MCP integrations, but the
Python-level ``say()`` call here is used for acknowledgement and error
recovery only.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, List

from crewai import Task

from src.crew.builder import CrewBuilder

if TYPE_CHECKING:
    from src.agents.registry import AgentRegistry

__all__: List[str] = ["planner_handler"]

logger = logging.getLogger(__name__)

# Strip the role tag prefix (and any leading/trailing whitespace) from the
# message body.  Handles variations like [Planner], [ planner ], etc.
_PLANNER_TAG_RE: re.Pattern[str] = re.compile(r"\[\s*planner\s*\]", re.IGNORECASE)

_THINKING_MSG: str = "⏳ On it! Let me think about your product idea or requirement …"

_PLANNER_TASK_DESCRIPTION_TEMPLATE: str = """
A human stakeholder has sent you the following message via Slack:

---
{human_message}
---

Slack thread timestamp (if this message is part of an ongoing thread): {thread_ts}

You are the Planner Agent.  Read the message and determine which of the three
request types below applies, then execute the corresponding steps.

════════════════════════════════════════════════════════════════════════
**Request Type A — Actionable Requirement**
════════════════════════════════════════════════════════════════════════
Applies when: the human is giving you a clear, specific, immediately
actionable product requirement (e.g. "Build a feature to export invoices as
PDF"), NOT an exploratory idea discussion.  This is typically short, scoped,
and ready to be turned into a development ticket without further discussion.

Steps:
1. Carefully read and understand the requirement.
2. If the requirement is **ambiguous or incomplete**, ask targeted clarifying
   questions via slack/reply_to_thread.
3. If the requirement is **clear and actionable**:
   a. Create a JIRA epic via jira/create_issue with a descriptive title,
      acceptance criteria, and priority.
   b. Create a matching ClickUp task via clickup/create_task that mirrors
      the JIRA epic.
4. Reply in the Slack thread via slack/reply_to_thread with:
   - A brief summary of your understanding of the requirement.
   - Either: the JIRA epic key + ClickUp task ID that were created, OR
   - The clarifying questions you need answered before proceeding.

════════════════════════════════════════════════════════════════════════
**Request Type B — Idea Survey & Discussion**
════════════════════════════════════════════════════════════════════════
Applies when: the human is describing a product idea or concept at an
exploratory stage — not yet a concrete requirement, but something they want
to evaluate, discuss, and get a deeper plan for.

Steps:
1. Use slack/get_messages to fetch the full thread history (use thread_ts
   if available) so you have full context of the ongoing discussion.
2. Assess the current discussion stage:
   - **Initial** (first message): Begin with 3-5 targeted questions to
     understand the idea better.  Produce an initial survey outline.
   - **Mid-discussion** (thread has exchanges): Respond to the human's
     latest message in context, continuing to refine the plan.
   - **Survey ready** (all key dimensions are clear): Post the complete
     Markdown survey plan via slack/reply_to_thread.

3. When posting the complete survey plan, use the following structure and
   heading "### 📋 Idea Survey Plan" followed by all 8 required dimensions:

   ### 📋 Idea Survey Plan
   **1. Marketing Value** — what is the business/market opportunity?
   **2. Market Scope** — what is the addressable market size and geography?
   **3. Business Model** — how does this generate revenue or reduce costs?
   **4. Target Audience** — who are the primary users/customers?
   **5. Customer Pain Points Resolved** — what specific problems does this solve?
   **6. MVP Features** — the minimal feature set for the first release.
   **7. Quick Implementation Path** — how to build MVP fast (tech stack,
      effort estimate, key libraries or services).
   **8. Budget Estimation** — rough cost to reach MVP
      (development time + infrastructure + tooling).

IMPORTANT GUARDRAIL (BR-11):
  Do NOT create any JIRA issues or ClickUp tasks during Type B mode.
  Tickets are created ONLY when the human makes a final accept/reject
  decision (see Type C below).

════════════════════════════════════════════════════════════════════════
**Request Type C — Idea Discussion Conclusion**
════════════════════════════════════════════════════════════════════════
Applies when: the human has made a **final decision** — explicitly accepting
or rejecting the idea in their latest message.

Accept signals: phrases like "let's do it", "approved", "accepted",
"proceed", "go ahead", "LGTM", "let's proceed with the MVP", "we're doing
this", "greenlight".

Reject signals: phrases like "rejected", "not now", "cancel this",
"drop it", "we won't do this", "let's not pursue it", "too risky".

─────────────────────────────────────────────────
Steps for the **REJECT** path:
─────────────────────────────────────────────────
1. Use slack/get_messages to read the full thread history for context.
2. Post a conclusion message in the Slack thread via slack/reply_to_thread:
   "📋 Conclusion: This idea has been rejected. [1–2 sentence reason summary]"
3. Create 1 JIRA issue via jira/create_issue:
   - Type: Story or Task
   - Title: "[REJECTED] <idea name>"
   - Status: "REJECTED"
   - Description: Full discussion summary + reason for rejection + date.
4. Create 1 ClickUp task via clickup/create_task mirroring the above with
   status "REJECTED".
5. Reply in the Slack thread with the JIRA key and ClickUp task ID as a
   permanent record of the decision.

GUARDRAIL (BR-12): Do NOT mention, notify, or hand off to the Dev Lead in
the reject path.  No slack/send_message call should reference "[dev lead]".

─────────────────────────────────────────────────
Steps for the **ACCEPT** path:
─────────────────────────────────────────────────
1. Use slack/get_messages to read the full thread history for context.
2. Post a conclusion message in the Slack thread via slack/reply_to_thread:
   "✅ Conclusion: This idea has been accepted! [1–2 sentence summary]"
3. Create 1+ JIRA issue(s) via jira/create_issue:
   - Type: Epic or Story
   - Title: "<idea name>"
   - Status: "OPEN"
   - Description: Full survey plan + all discussion details + acceptance date.
4. Create matching ClickUp task(s) via clickup/create_task with status "OPEN".
5. Reply in the Slack thread with the created ticket URLs as a record.
6. Send a NEW Slack message (NOT a thread reply) via slack/send_message to
   the same channel with the following content (BR-13):
   "[dev lead] A new product idea has been accepted and is ready for
    development planning: <idea name>.
    Tickets created: <ticket URLs>.
    Please start the development feasibility and planning discussion."

GUARDRAIL (BR-1):  Never set any ticket to "ACCEPTED" status — that is a
human-only gate.
GUARDRAIL (BR-14): REJECTED tickets must use status "REJECTED", never "OPEN"
or "ACCEPTED".

════════════════════════════════════════════════════════════════════════
Determine which request type applies and execute the appropriate steps.
Be professional, thorough, and always confirm your actions back to the user.
"""

_PLANNER_TASK_EXPECTED_OUTPUT: str = (
    "A structured reply posted in the Slack thread. Outcome depends on request type: "
    "(A) clarifying questions OR confirmation of JIRA epic key + ClickUp task ID created; "
    "(B) survey questions OR the complete 8-dimension Idea Survey Plan in Markdown; "
    "(C-reject) conclusion message + REJECTED ticket keys/URLs; "
    "(C-accept) conclusion message + OPEN ticket keys/URLs + a new Slack message "
    "mentioning [dev lead] to start the development planning discussion."
)


def _run_planner_crew(
    human_message: str,
    thread_ts: str | None,
    say: Any,
    registry: "AgentRegistry",
) -> None:
    """Execute the Planner Crew in a background thread.

    Args:
        human_message: The clean message body (role tag already stripped).
        thread_ts:     Slack thread timestamp for reply threading.
        say:           Slack Bolt ``say()`` callable for error fallback.
        registry:      The shared :class:`~src.agents.registry.AgentRegistry`.
    """
    try:
        planner_agent = registry["planner"]
    except KeyError:
        logger.error("planner_handler: 'planner' agent not found in registry.")
        say(
            text="❌ Configuration error: the Planner agent is not available. Please contact the admin.",
            thread_ts=thread_ts,
        )
        return

    task = Task(
        description=_PLANNER_TASK_DESCRIPTION_TEMPLATE.format(
            human_message=human_message,
            thread_ts=thread_ts or "N/A",
        ),
        expected_output=_PLANNER_TASK_EXPECTED_OUTPUT,
        agent=planner_agent,
    )

    crew = CrewBuilder.build(
        agents=[planner_agent],
        tasks=[task],
        process="sequential",
    )

    try:
        result = crew.kickoff()
        logger.info(
            "planner_handler: crew completed. Result preview: %.300s",
            str(result),
        )
        # The agent posts the full reply via slack/reply_to_thread or
        # slack/send_message natively.  We log the result but do not post
        # it again through say() to avoid duplicate messages in the thread.

    except Exception as exc:  # noqa: BLE001
        logger.exception("planner_handler: crew raised an exception.")
        say(
            text=(
                f"❌ Something went wrong while the Planner was processing your request:\n"
                f"```{exc}```\nPlease try again or contact the admin."
            ),
            thread_ts=thread_ts,
        )


def planner_handler(
    text: str,
    event: dict[str, Any],
    say: Any,
    registry: "AgentRegistry",
    executor: ThreadPoolExecutor,
) -> None:
    """Handle an ``@ai-worker [planner]`` Slack message.

    Strips the role tag, posts an acknowledgement, then submits the Planner
    Crew to the background executor so Bolt ACKs within the 3-second window.

    Supports three modes via the task description template:
    - **Type A**: Actionable requirement → create JIRA epic + ClickUp task.
    - **Type B**: Idea survey → multi-turn discussion + Markdown plan.
    - **Type C**: Conclusion → REJECTED or OPEN ticket + Dev Lead hand-off.

    Args:
        text:     The cleaned message text (``<@BOTID>`` tokens already removed
                  by the router).
        event:    The full Slack event payload dict.
        say:      Slack Bolt ``say()`` callable.
        registry: The shared :class:`~src.agents.registry.AgentRegistry`.
        executor: The bounded :class:`~concurrent.futures.ThreadPoolExecutor`.
    """
    thread_ts: str | None = event.get("thread_ts") or event.get("ts")

    # Strip the [planner] tag to get the clean human message.
    human_message: str = _PLANNER_TAG_RE.sub("", text).strip()

    if not human_message:
        say(
            text="👋 You mentioned *[planner]* but didn't include a message. What would you like to plan?",
            thread_ts=thread_ts,
        )
        return

    logger.info(
        "planner_handler: dispatching crew for thread_ts=%s, message=%.100s …",
        thread_ts,
        human_message,
    )

    # Post immediate acknowledgement before submitting to executor.
    say(text=_THINKING_MSG, thread_ts=thread_ts)

    executor.submit(
        _run_planner_crew,
        human_message,
        thread_ts,
        say,
        registry,
    )
