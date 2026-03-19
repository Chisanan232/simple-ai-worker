"""
Deterministic fake LLM for offline E2E testing.

``FakeLLM`` extends ``crewai.BaseLLM`` and satisfies the ``crewai.Agent(llm=...)``
contract without making any real AI network calls.

How it drives MCP tool calls
-----------------------------
CrewAI offers two agent execution paths depending on whether the LLM reports
native function-calling support:

* **Native path** (``supports_function_calling() → True``): the executor passes
  ``tools=[...]`` to ``LLM.call()``.  The LLM must return a *list* of OpenAI-style
  tool-call dicts on the first turn; the executor dispatches the tools, appends
  their results to the message history, then calls the LLM again expecting a
  plain-string final answer.

* **ReAct path** (``supports_function_calling() → False``): the executor embeds
  tool schemas in the system prompt and expects the LLM to reply with::

      Thought: …
      Action: <tool_name>
      Action Input: {…}

  followed later by::

      Final Answer: …

``FakeLLM`` uses the **native path** so that real MCP tool calls are dispatched
through the ``MCPServerHTTP`` transport to the ``MCPStubServer`` — enabling
tests to assert ``stub.was_called("search_issues")``.

On the **first** call per agent task, it picks the first tool from the provided
``tools`` list (or from its own ``_tool_order`` registry) and emits a tool-call
JSON.  On **every subsequent** call it returns ``_default_response`` as the
final answer, ending the agent loop.

Activation
----------
``FakeLLM`` is injected globally via the ``_maybe_patch_llm_factory`` session
fixture in ``test/e2e_test/conftest.py`` when ``E2E_USE_FAKE_LLM=true``.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from typing import Any

from crewai import BaseLLM

__all__ = ["FakeLLM"]

logger = logging.getLogger(__name__)


class FakeLLM(BaseLLM):
    """Deterministic fake LLM that drives real MCP stub tool dispatch.

    Parameters
    ----------
    default_response:
        Final-answer string returned after all tool turns complete.
    responses:
        Optional ``{keyword: response}`` mapping for the *final* answer.
        The first keyword found in the concatenated prompt wins.
    tool_order:
        Optional list of tool names to call in sequence before finishing.
        If empty, the first tool from the ``tools`` argument is called once.
    model:
        Model identifier string (read by CrewAI internals for logging).
    """

    def __init__(
        self,
        default_response: str = "I have completed the task successfully.",
        responses: dict[str, str] | None = None,
        tool_order: list[str] | None = None,
        model: str = "fake/fake-model",
    ) -> None:
        try:
            super().__init__(model=model)
        except Exception:
            object.__init__(self)

        self.model = model
        self._default_response = default_response
        self._responses: dict[str, str] = dict(responses or {})
        self._tool_order: list[str] = list(tool_order or [])
        self._tool_args_overrides: dict[str, dict[str, Any]] = {}
        self._call_log: list[dict[str, Any]] = []
        # Thread-local storage for per-thread state:
        #   turn                  – turn counter (existing)
        #   _tool_invocation_counts – dict[bare_name, int] tracking how many times
        #     each tool has been called within this thread's execution.  Used by
        #     _tool_args to generate unique argument dicts for repeated tool calls
        #     (e.g. two consecutive create_task calls) so they produce different
        #     cache keys and bypass CrewAI's tool-result cache.
        self._local = threading.local()

    # ------------------------------------------------------------------
    # crewai.BaseLLM interface
    # ------------------------------------------------------------------

    def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str | list[dict[str, Any]]:
        """Return tool-call JSON on tool turns, final answer once all tools are done.

        When ``tools`` is provided (native-tool-call path):
        - If ``_tool_order`` has N tools: turns 0 … N-1 each call the next
          tool in the sequence; turn N returns the plain-text final answer.
        - If ``_tool_order`` is empty: turn 0 calls the first matching tool;
          turn 1+ returns the plain-text final answer (original behaviour).

        When ``tools`` is absent/empty (ReAct path or no-tool task):
        - Always return the plain-text final answer.
        """
        # Flatten all message content into a single searchable string.
        prompt = " ".join(
            str(m.get("content", ""))
            for m in messages
            if isinstance(m, dict) and m.get("content")
        )

        self._call_log.append({"messages": messages, "tools": tools, "prompt": prompt})
        logger.debug(
            "FakeLLM.call #%d — prompt_len=%d tools=%s",
            len(self._call_log),
            len(prompt),
            [t.get("function", {}).get("name") for t in (tools or [])],
        )
        # Log message roles for debugging tool result injection
        if logger.isEnabledFor(logging.DEBUG):
            for idx, m in enumerate(messages):
                if isinstance(m, dict):
                    role = m.get("role", "?")
                    content = m.get("content", "")
                    if role in ("tool", "function"):
                        logger.debug(
                            "FakeLLM.call msg[%d] role=%r content=%r",
                            idx, role, str(content)[:200],
                        )

        # ── per-thread turn counter (thread-local for parallel task isolation) ──
        # Reset to 0 when a new conversation starts (≤2 messages means the crew
        # just started; longer histories belong to ongoing tool-call exchanges).
        current_turn: int = self._local.turn or 0
        if len(messages) <= 2:
            current_turn = 0
            self._local._tool_invocation_counts = {}
        turn = current_turn
        logger.debug(
            "FakeLLM turn=%d messages_len=%d tool_order=%s",
            turn, len(messages), self._tool_order,
        )

        if tools:
            # Build list of available tool names from the provided tools schema
            available_names: list[str] = []
            for t in tools:
                name = t.get("function", {}).get("name") or t.get("name", "")
                if name:
                    available_names.append(name)

            # Determine which tool to call on this turn
            next_tool: str | None = None
            if self._tool_order:
                # Multi-step mode: iterate through _tool_order until exhausted
                tool_call_turns = len(self._tool_order)
                if turn < tool_call_turns:
                    # Pick the tool for this turn from the ordered sequence.
                    # MCP tools are namespaced (e.g. "localhost_8080_mcp_add_comment"),
                    # so match by exact name OR by suffix after the last underscore
                    # segment that equals the preferred name.
                    preferred = self._tool_order[turn]
                    next_tool = self._resolve_tool_name(preferred, available_names)
                    if next_tool is None:
                        # Fall back to first available tool for this turn slot
                        next_tool = available_names[0] if available_names else None
            else:
                # Single-step mode (legacy): call one tool on turn 0 only
                if turn == 0:
                    next_tool = self._pick_tool(prompt, tools)

            if next_tool is not None:
                self._local.turn = turn + 1
                call_id = f"call_{uuid.uuid4().hex[:8]}"
                tool_args = self._tool_args(next_tool, prompt, messages)
                tool_call_response = [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": next_tool,
                            "arguments": json.dumps(tool_args),
                        },
                    }
                ]
                logger.debug("FakeLLM → tool call [turn %d]: %s(%s)", turn, next_tool, tool_args)
                return tool_call_response

        # All tool turns done (or no tools available): return the final answer
        self._local.turn = turn + 1
        return self._choose_final_answer(prompt, messages)

    def supports_function_calling(self) -> bool:
        """Always True — use native tool-call path so MCP tools are dispatched."""
        return True

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 128_000

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_tool(
        self,
        prompt: str,
        tools: list[dict[str, Any]] | None,
    ) -> str | None:
        """Choose which tool to call.

        Priority:
        1. First name in ``_tool_order`` that is available in ``tools``
           (exact match or suffix match for namespaced MCP tool names).
        2. First keyword from ``_responses`` that appears in the prompt AND
           matches an available tool name.
        3. First tool in the provided ``tools`` list.
        """
        available_names: list[str] = []
        if tools:
            for t in tools:
                name = t.get("function", {}).get("name") or t.get("name", "")
                if name:
                    available_names.append(name)

        if not available_names:
            return None

        # 1. Explicit ordering — with suffix-aware matching
        for name in self._tool_order:
            resolved = self._resolve_tool_name(name, available_names)
            if resolved:
                return resolved

        # 2. Keyword match against registered responses
        for keyword in self._responses:
            if keyword in prompt:
                resolved = self._resolve_tool_name(keyword, available_names)
                if resolved:
                    return resolved
                # Broader substring match
                for n in available_names:
                    if keyword in n or n in keyword:
                        return n

        # 3. First available tool
        return available_names[0]

    @staticmethod
    def _resolve_tool_name(preferred: str, available_names: list[str]) -> str | None:
        """Resolve a preferred tool name to an actual available tool name.

        MCP tools served over HTTP are namespaced by the server address, e.g.
        ``localhost_8080_mcp_add_comment``.  This method matches *preferred*
        against *available_names* using:

        1. Exact match.
        2. Suffix match — available name ends with ``_<preferred>`` (handles
           any namespace depth, e.g. ``localhost_55572_mcp_add_comment``
           matches ``add_comment``).
        3. Substring match — preferred appears anywhere in the available name.

        Returns the first matching available name, or ``None``.
        """
        # 1. Exact match
        if preferred in available_names:
            return preferred
        # 2. Suffix match (namespaced MCP tool names)
        suffix = f"_{preferred}"
        for n in available_names:
            if n.endswith(suffix):
                return n
        # 3. Substring match (broader fallback)
        for n in available_names:
            if preferred in n:
                return n
        return None

    @staticmethod
    def _context_key(messages: list[dict[str, Any]]) -> str:
        """Kept for backward compatibility — no longer used for turn tracking."""
        if not messages:
            return "empty"
        last = messages[-1]
        content = str(last.get("content", ""))[:80]
        role = last.get("role", "")
        return f"{role}:{content}"

    def reset_turns(self) -> None:
        """Reset the calling thread's turn counter and tool invocation counts to 0.

        Call this between job runs within a single test to ensure the second
        run starts fresh (e.g. in E2E-PN-07 between run 1 and run 2).
        Since turn state is thread-local, this only affects the current thread
        (the test thread).  Worker threads reset automatically when they exit
        and are re-created for the next ThreadPoolExecutor.
        """
        self._local.turn = 0
        self._local._tool_invocation_counts = {}

    def _choose_final_answer(
        self,
        prompt: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> str:
        """Pick the appropriate final-answer string.

        Special case: if the prompt asks for a JSON PR-status object (contains
        "is_merged" in the expected-output instructions), return a valid JSON
        stub so that ``_run_pr_status_check`` can parse it without error.

        We inspect the ``messages`` list (not just the concatenated prompt) to
        find actual tool-result messages (``role == "tool"`` or ``"function"``)
        and parse their content to derive:
        - ``is_merged``: True when a ``get_pull_request`` result contains
          ``"merged": true`` or ``"is_merged": true``.
        - ``approval_count``: counted from ``get_pull_request_reviews`` results;
          defaults to ``1`` when no reviews tool result is found in the message
          history (so that tests which only call ``merge_pull_request`` still
          trigger the merge path).

        IMPORTANT: We do NOT scan the concatenated *prompt* string for approval
        counts because the task-description template itself contains example JSON
        like ``{"is_merged": false, "approval_count": 1}`` which would give false
        positives.
        """
        # Detect the PR status-check task specifically.
        # _PR_STATUS_TASK_TEMPLATE uniquely contains "Return ONLY the JSON object"
        # and "approval_count" in its task description and expected-output text.
        # _PR_MERGE_TASK_TEMPLATE does NOT — it asks for "MERGED: <url>" confirmation.
        # Using "Return ONLY the JSON object" prevents the merge crew from accidentally
        # returning PR status JSON when the get_pull_request tool result contains
        # "is_merged" / "approval_count" keys.
        _is_pr_status_task = (
            "Return ONLY the JSON object" in prompt
            or "JSON with is_merged" in prompt
        )
        if _is_pr_status_task:
            import re as _re

            pr_url_match = _re.search(r"https?://\S+/pull/\d+", prompt)
            pr_url = pr_url_match.group(0) if pr_url_match else "https://github.com/org/repo/pull/0"

            logger.debug(
                "FakeLLM._choose_final_answer: PR status check. prompt_len=%d, snippet=%r",
                len(prompt),
                prompt[-300:],
            )

            # ------------------------------------------------------------------
            # Extract text content from tool-result messages.
            # CrewAI injects tool results as messages with role="tool" or
            # role="function".  The content is a plain string extracted from
            # the MCP response's text field (e.g. "[]", '{"merged": false}').
            # ------------------------------------------------------------------
            tool_result_texts: list[str] = []
            for msg in (messages or []):
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role", "")
                if role not in ("tool", "function"):
                    continue
                content = msg.get("content", "")
                if isinstance(content, str):
                    tool_result_texts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            tool_result_texts.append(
                                str(block.get("text", block.get("content", "")))
                            )
                        else:
                            tool_result_texts.append(str(block))

            logger.debug(
                "FakeLLM._choose_final_answer: found %d tool-result message(s): %r",
                len(tool_result_texts),
                [t[:120] for t in tool_result_texts],
            )

            is_merged = False
            approval_count = 0
            has_reviews_result = False

            for text in tool_result_texts:
                text_stripped = text.strip()
                # Try to parse as JSON first
                try:
                    data = json.loads(text_stripped)
                except (json.JSONDecodeError, ValueError):
                    data = None

                if isinstance(data, dict):
                    # get_pull_request result
                    if data.get("merged") is True or data.get("is_merged") is True:
                        is_merged = True
                elif isinstance(data, list):
                    # get_pull_request_reviews returns a list (possibly empty)
                    has_reviews_result = True
                    for review in data:
                        if isinstance(review, dict) and str(review.get("state", "")).upper() == "APPROVED":
                            approval_count += 1
                else:
                    # Plain-text fallback (shouldn't normally happen)
                    if _re.search(r'"(?:merged|is_merged)"\s*:\s*true', text, _re.IGNORECASE):
                        is_merged = True
                    if _re.search(r'"state"\s*:', text):
                        has_reviews_result = True
                    approval_count += len(
                        _re.findall(r'"state"\s*:\s*"APPROVED"', text, _re.IGNORECASE)
                    )
                    if text_stripped in ("[]", "[ ]"):
                        has_reviews_result = True

            # If no tool-result messages were found at all, fall back to prompt
            # scan ONLY for the merged flag (not approval_count — see docstring).
            if not tool_result_texts:
                logger.debug(
                    "FakeLLM._choose_final_answer: no tool-result messages found, "
                    "falling back to prompt scan (merged only)"
                )
                if _re.search(r'"(?:merged|is_merged)"\s*:\s*true', prompt, _re.IGNORECASE):
                    is_merged = True
                # has_reviews_result stays False → approval_count defaults to 1 below

            # If we never saw a get_pull_request_reviews result, default to 1
            # so that tests using only merge_tool_order still trigger the merge path.
            if not has_reviews_result:
                approval_count = 1

            logger.debug(
                "FakeLLM._choose_final_answer: is_merged=%s approval_count=%d pr_url=%s",
                is_merged,
                approval_count,
                pr_url,
            )

            return json.dumps({
                "is_merged": is_merged,
                "approval_count": approval_count,
                "pr_url": pr_url,
            })

        for keyword, response in self._responses.items():
            if keyword in prompt:
                logger.debug("FakeLLM final answer matched keyword %r", keyword)
                return response

        # If any tool-result message contains a PR html_url (from create_pull_request),
        # emit "PR_URL: <url>" so _extract_pr_url can register it in the watcher dicts.
        import re as _re2
        for msg in (messages or []):
            if not isinstance(msg, dict):
                continue
            if msg.get("role", "") not in ("tool", "function"):
                continue
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            try:
                data = json.loads(content)
            except (json.JSONDecodeError, ValueError):
                data = None
            if isinstance(data, dict) and data.get("html_url"):
                pr_url = str(data["html_url"])
                if "/pull/" in pr_url:
                    logger.debug("FakeLLM: emitting PR_URL from tool result: %s", pr_url)
                    return (
                        f"I have completed the task successfully.\n"
                        f"PR_URL: {pr_url}"
                    )

        return self._default_response

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------

    def register(self, keyword: str, response: str) -> None:
        """Register a ``keyword → final_answer`` mapping."""
        self._responses[keyword] = response

    def set_tool_order(self, *tool_names: str) -> None:
        """Set the sequence of tools to call before finishing."""
        self._tool_order = list(tool_names)

    def clear_responses(self) -> None:
        """Remove all registered keyword→response mappings."""
        self._responses.clear()

    def reset(self) -> None:
        """Clear call log, responses, tool overrides, and the current thread's turn counter."""
        self._call_log.clear()
        self._responses.clear()
        self._tool_args_overrides.clear()
        self._local.turn = 0
        self._local._tool_invocation_counts = {}

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    @property
    def call_log(self) -> list[dict[str, Any]]:
        """Snapshot of all recorded calls."""
        return list(self._call_log)

    def was_called(self) -> bool:
        return len(self._call_log) > 0

    def call_count(self) -> int:
        return len(self._call_log)

    def set_tool_args(self, tool_name: str, **args: Any) -> None:
        """Override the args dict emitted when FakeLLM calls *tool_name*.

        Use this in fixtures/tests to inject specific field values (e.g. a
        status field) that the default ``_tool_args`` heuristic cannot infer
        reliably from the prompt.

        Example::

            fake_llm_session.set_tool_args("create_task", status="REJECTED")

        The override is merged on top of the default-generated args, so only
        the specified keys are replaced.
        """
        bare = tool_name.split("_mcp_", 1)[-1] if "_mcp_" in tool_name else tool_name
        self._tool_args_overrides.setdefault(bare, {}).update(args)

    def _tool_args(
        self,
        tool_name: str,
        prompt: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return minimal valid arguments for a tool call.

        - ``add_comment`` / ``add_comment_2``: returns a deterministic plan body.
          If the prompt contains a Slack permalink URL (from a prior
          ``get_thread_permalink`` tool result), it is appended to the body so
          that permalink-assertion tests pass.
        - ``get_task`` / ``get_issue``: extracts the task/issue ID from the prompt
          using common patterns (``cu-xxx``, ``PROJ-xxx``) so that test assertions
          on the fetched task ID pass.
        - ``update_task`` / ``transition_issue``: returns ``status: IN PROGRESS``.
        - ``create_task`` / ``create_issue``: returns status REJECTED or OPEN based
          on rejection keywords found in the first user message.
        - ``reply_to_thread`` / ``send_message``: returns a rich survey-plan text
          so that planner tests that inspect the reply body can find key dimensions.
        - All other tools receive an empty dict.
        - ``_tool_args_overrides``: any per-bare-name overrides are merged last.
        """
        import re as _re

        bare = tool_name.split("_mcp_", 1)[-1] if "_mcp_" in tool_name else tool_name

        base: dict[str, Any] = {}

        if bare in ("add_comment", "add_comment_2"):
            permalink_match = _re.search(
                r"https?://[a-zA-Z0-9._-]*slack\.com/[^\s\"']+", prompt
            )
            permalink_line = (
                f"\n\n🔗 Slack thread: {permalink_match.group(0)}\n"
                if permalink_match else ""
            )
            body = (
                "## 📋 Development Plan (v1)\n\n"
                "## Overview\n"
                "Implementation plan generated by the Dev Agent.\n\n"
                "## Technical Approach\n"
                "Design and architecture for the requested feature.\n\n"
                "## Implementation Steps\n"
                "1. Analyse requirements\n"
                "2. Implement solution\n"
                "3. Add tests\n\n"
                "## Risks & Open Questions\n"
                "- Rollback strategy for migration failures\n"
                "- Retry logic and failure handling\n\n"
                "## Revision Notes\n"
                "Addressed reviewer feedback on rollback and migration handling.\n\n"
                "👋 @team — Please review the plan above and leave your feedback. "
                f"Once satisfied, set the ticket to **IN PLANNING**.{permalink_line}\n"
            )
            base = {"comment": body, "body": body, "text": body}

        elif bare in ("get_task", "get_issue"):
            task_match = _re.search(r"\bcu-[a-zA-Z0-9]+\b", prompt)
            issue_match = _re.search(r"\b([A-Z]+-\d+)\b", prompt)
            if task_match:
                base = {"task_id": task_match.group(0), "id": task_match.group(0)}
            elif issue_match:
                base = {"issue_key": issue_match.group(1), "key": issue_match.group(1)}

        elif bare in ("update_task", "transition_issue", "update_issue"):
            task_match = _re.search(r"\bcu-[a-zA-Z0-9]+\b", prompt)
            issue_match = _re.search(r"\b([A-Z]+-\d+)\b", prompt)
            base = {"status": "IN PROGRESS"}
            if task_match:
                base["task_id"] = task_match.group(0)
            if issue_match:
                base["issue_key"] = issue_match.group(1)

        elif bare in ("create_task", "create_issue"):
            # Scan the FIRST user-role message (the task description) for reject/accept
            # signals, since the human intent appears at the top of the task description.
            # Fall back to the prompt head if no user message found.
            scan_text = ""
            for msg in (messages or []):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    scan_text = str(msg.get("content", ""))[:1000].lower()
                    break
            if not scan_text:
                scan_text = prompt[:1000].lower()
            reject_kw = ("let's drop", "drop this", "cancel this", "not pursue",
                         "too competitive", "won't do this", "rejected", "not now",
                         "drop it", "drop the", "drop this idea")
            status = "REJECTED" if any(k in scan_text for k in reject_kw) else "OPEN"

            # Use a per-thread invocation counter to generate a unique task name
            # for each call.  This ensures that repeated create_task/create_issue
            # calls (e.g. two sub-tasks in the breakdown_tool_order scenario)
            # produce DIFFERENT argument dicts, giving them distinct cache keys
            # in CrewAI's tool-result cache and forcing real MCP calls each time.
            counts: dict[str, int] = getattr(self._local, "_tool_invocation_counts", {})
            count = counts.get(bare, 0) + 1
            counts[bare] = count
            self._local._tool_invocation_counts = counts

            subtask_labels = [
                "WebSocket backend service",
                "Redis pub/sub integration",
                "Frontend notification subscriber",
                "API gateway integration",
                "Database schema migration",
            ]
            name = subtask_labels[count - 1] if count <= len(subtask_labels) else f"Sub-task {count}"
            base = {"name": name, "status": status}

        elif bare in ("reply_to_thread", "send_message"):
            # Determine if this is an acceptance send_message (needs [dev lead])
            scan_text = ""
            for msg in (messages or []):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    scan_text = str(msg.get("content", ""))[:1000].lower()
                    break
            if not scan_text:
                scan_text = prompt[:1000].lower()
            accept_kw = ("let's do it", "approved", "go ahead", "proceed",
                         "lgtm", "greenlight", "approve this", "i approve",
                         "we're doing", "accepted")
            is_accepted = any(k in scan_text for k in accept_kw)
            if bare == "send_message" and is_accepted:
                dev_lead_text = (
                    "[dev lead] The idea has been approved. "
                    "Please proceed with feasibility assessment and task breakdown. "
                    "Task URL: https://app.clickup.com/t/new-task"
                )
                base = {"text": dev_lead_text, "message": dev_lead_text}
            else:
                survey_text = (
                    "### 📋 Idea Survey Plan\n\n"
                    "**1. Market Opportunity** — Large addressable market.\n"
                    "**2. Business Model / Revenue** — SaaS subscription.\n"
                    "**3. Target Audience / Customer** — SMB owners.\n"
                    "**4. Pain Points / Problem** — Manual processes causing waste.\n"
                    "**5. MVP Features** — Core workflow, dashboard, alerts.\n"
                    "**6. Technical Implementation** — Cloud-native, REST API.\n"
                    "**7. Budget & Cost Estimation** — $100K–$200K for MVP.\n"
                    "**8. Risk Assessment** — Market competition, tech debt.\n\n"
                    "Please review and let me know if you approve or reject the idea."
                )
                base = {"text": survey_text, "message": survey_text}

        # Merge any per-tool overrides set via set_tool_args()
        overrides = self._tool_args_overrides.get(bare, {})
        if overrides:
            base = {**base, **overrides}
        return base

