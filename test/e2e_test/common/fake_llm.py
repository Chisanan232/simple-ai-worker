"""
Deterministic fake LLM for offline E2E testing.

``FakeLLM`` extends ``crewai.BaseLLM`` and satisfies the ``crewai.Agent(llm=...)``
contract without making any network calls.  Responses are returned from a
keyword-lookup table; unmatched prompts receive a configurable default answer.

Activation
----------
``FakeLLM`` is never used directly by production code.  In E2E tests it is
injected via the ``_maybe_patch_llm_factory`` session fixture defined in
``test/e2e_test/conftest.py``, which replaces ``LLMFactory.build`` for the
entire pytest session when ``E2E_USE_FAKE_LLM=true`` is set in
``test/e2e_test/.env.e2e``.

Usage::

    from test.e2e_test.common.fake_llm import FakeLLM

    # Plain default — every prompt returns the same answer
    llm = FakeLLM()

    # Keyword responses — first matching key wins
    llm = FakeLLM(
        default_response="Task completed successfully.",
        responses={
            "search_issues": "No open tickets found.",
            "create_pull_request": "Pull request #42 created.",
        },
    )

    # Dynamic registration after construction
    llm.register("transition_issue", "Status updated to IN PROGRESS.")

    # Inspection
    assert llm.was_called()
    assert llm.call_count() == 3
    assert llm.call_log[0]["messages"][0]["content"] == "..."

Design note — tool-calling
--------------------------
CrewAI agents that use MCP tools expect the LLM to emit JSON tool-call objects
on intermediate reasoning turns.  ``FakeLLM`` currently returns plain-text
final answers on *every* call, which causes the agent to skip MCP tool
dispatch entirely and short-circuit straight to the final answer.

This is intentional for Phase 1:
- Orchestration / handler logic can be tested (does the crew run? does it
  complete without error?).
- MCP tool-call sequence assertions (``stub.was_called("search_issues")``)
  will not fire because the agent never reaches tool dispatch.

A future Phase 2 enhancement can add a ``script`` parameter that accepts a
list of alternating tool-call JSON / final-answer strings to enable full
round-trip MCP stub verification without a real LLM.
"""

from __future__ import annotations

import logging
from typing import Any

from crewai import BaseLLM

__all__ = ["FakeLLM"]

logger = logging.getLogger(__name__)


class FakeLLM(BaseLLM):
    """Deterministic fake LLM — returns fixed responses without any network call.

    Parameters
    ----------
    default_response:
        Returned when no registered keyword matches the incoming prompt.
        Defaults to a generic task-completion message.
    responses:
        Optional pre-seeded ``{keyword: response}`` mapping.  The *first*
        keyword that appears in the concatenated prompt text wins.
    model:
        Model identifier string.  ``crewai.Agent`` / ``Crew`` may read
        ``.model`` for logging or routing purposes.  Defaults to
        ``"fake/fake-model"`` which deliberately does not match any real
        LiteLLM provider.
    """

    def __init__(
        self,
        default_response: str = "I have completed the task successfully.",
        responses: dict[str, str] | None = None,
        model: str = "fake/fake-model",
    ) -> None:
        # BaseLLM.__init__ may require a model kwarg depending on the version.
        # Call super with a safe no-op model string.
        try:
            super().__init__(model=model)
        except Exception:
            # Older / future versions may differ — fall back to object init.
            object.__init__(self)

        self.model = model
        self._default_response = default_response
        self._responses: dict[str, str] = dict(responses or {})
        self._call_log: list[dict[str, Any]] = []

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
    ) -> str:
        """Return a deterministic response without any network call.

        The concatenated content of all *messages* is searched for each
        registered keyword in insertion order.  The response mapped to the
        first matching keyword is returned.  If no keyword matches,
        ``default_response`` is returned.

        All calls are appended to ``_call_log`` for post-test inspection.
        """
        # Flatten all message content into a single searchable string.
        prompt = " ".join(
            str(m.get("content", ""))
            for m in messages
            if isinstance(m, dict)
        )

        self._call_log.append({"messages": messages, "tools": tools, "prompt": prompt})
        logger.debug("FakeLLM.call — prompt_len=%d tools=%s", len(prompt), tools)

        for keyword, response in self._responses.items():
            if keyword in prompt:
                logger.debug("FakeLLM matched keyword %r → returning registered response", keyword)
                return response

        logger.debug("FakeLLM no keyword matched → returning default response")
        return self._default_response

    def supports_function_calling(self) -> bool:
        """Always False — FakeLLM returns plain-text final answers only.

        Returning ``False`` tells CrewAI's internal router not to attempt
        JSON tool-call parsing on the output, which avoids spurious errors
        when the agent framework checks LLM capabilities.
        """
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 128_000

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------

    def register(self, keyword: str, response: str) -> None:
        """Register (or overwrite) a ``keyword → response`` mapping.

        When the *keyword* string appears anywhere in a prompt, the
        corresponding *response* is returned instead of the default.

        Args:
            keyword: A substring to search for in the incoming prompt.
            response: The string to return when *keyword* is matched.
        """
        self._responses[keyword] = response

    def clear_responses(self) -> None:
        """Remove all registered keyword→response mappings."""
        self._responses.clear()

    def reset(self) -> None:
        """Clear call log and all registered responses.

        Useful between test cases when the same ``FakeLLM`` instance is
        shared across multiple tests (e.g. via a session-scoped fixture).
        """
        self._call_log.clear()
        self._responses.clear()

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    @property
    def call_log(self) -> list[dict[str, Any]]:
        """Snapshot of all recorded calls.

        Each entry is a dict with keys: ``"messages"``, ``"tools"``,
        ``"prompt"`` (flattened message text).
        """
        return list(self._call_log)

    def was_called(self) -> bool:
        """Return ``True`` if ``call()`` was invoked at least once."""
        return len(self._call_log) > 0

    def call_count(self) -> int:
        """Return the total number of ``call()`` invocations."""
        return len(self._call_log)

