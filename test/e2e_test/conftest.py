"""
E2E test configuration and shared fixtures — Layer 2.

Architecture (Four-Mode E2E Strategy)
--------------------------------------
E2E tests can run in four modes controlled by ``test/e2e_test/.env.e2e``:

  Mode 1 — Stub + Fake LLM (fully offline, no credentials needed):
    E2E_USE_FAKE_LLM=true
    MCP layer: MCPStubServer (pytest-httpserver)
    LLM layer: FakeLLM — deterministic, no network calls

  Mode 2 — Stub + Real LLM (default):
    No extra flags needed; only an LLM API key required.
    MCP layer: MCPStubServer (pytest-httpserver)
    LLM layer: real OpenAI / Anthropic call

  Mode 3 — Testcontainers + Real LLM (auto-managed Docker Compose):
    E2E_USE_TESTCONTAINERS=true
    MCP layer: Docker Compose stack started/stopped by pytest automatically
    LLM layer: real OpenAI / Anthropic call

  Mode 3b — Testcontainers + Fake LLM (containers, no LLM cost):
    E2E_USE_TESTCONTAINERS=true  E2E_USE_FAKE_LLM=true
    MCP layer: Docker Compose stack started/stopped by pytest automatically
    LLM layer: FakeLLM — drives real tool calls into the running containers

  Mode 4 — Manual Live + Real LLM (pre-started Docker Compose):
    Set E2E_MCP_*_URL values in .env.e2e after starting services manually.
    MCP layer: pre-started Docker Compose containers
    LLM layer: real OpenAI / Anthropic call

The two flags are **orthogonal**:
  - ``E2E_USE_TESTCONTAINERS`` controls the MCP layer (containers vs stub).
  - ``E2E_USE_FAKE_LLM``       controls the LLM layer (FakeLLM vs real API).

The ``mcp_urls`` fixture selects the MCP tier automatically.
The ``_maybe_patch_llm_factory`` autouse fixture handles the LLM tier.
Both are controlled by flags in `test/e2e_test/.env.e2e`.

Run commands (uv-managed project)
-----------------------------------
  # Mode 1 — fully offline
  E2E_USE_FAKE_LLM=true uv run pytest -m "e2e" test/e2e_test/

  # Mode 2 — stub + real LLM (default)
  uv run pytest -m "e2e" test/e2e_test/

  # Mode 3 — testcontainers + real LLM (auto Docker Compose)
  E2E_USE_TESTCONTAINERS=true uv run pytest -m "e2e" test/e2e_test/

  # Mode 3b — testcontainers + FakeLLM (containers, no LLM cost)
  E2E_USE_TESTCONTAINERS=true E2E_USE_FAKE_LLM=true uv run pytest -m "e2e" test/e2e_test/

  # Mode 4 — manual live
  uv run pytest -m "e2e" test/e2e_test/  # after docker compose up
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Generator, Optional

# Absolute path to the project root — used for Docker Compose paths so that
# testcontainers' subprocess always resolves files correctly, regardless of the
# working directory from which pytest is invoked.
_PROJECT_ROOT: Path = Path(__file__).parents[2]

import pytest
from pytest_httpserver import HTTPServer
from pydantic import SecretStr

from test.e2e_test.common.e2e_settings import E2ESettings, get_e2e_settings
from test.e2e_test.common.fake_llm import FakeLLM  # noqa: F401 — re-exported for sub-conftest use
from test.e2e_test.common.mcp_stub import MCPStubServer, RecordingStub  # noqa: F401
from src.ticket.workflow import WorkflowConfig

# Load E2E settings once at module level — before any monkeypatching.
_e2e_settings: E2ESettings = get_e2e_settings()
_LLM_KEY_AVAILABLE: bool = _e2e_settings.has_llm_key

# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

# NOTE: pytest.mark.skipif evaluates its condition expression at the time the
# decorator is applied (module import).  To make the LLM-key check truly
# dynamic we use a custom pytest mark backed by the _e2e_require_llm_key
# autouse fixture below, which re-calls get_e2e_settings().has_llm_key at
# test execution time.
#
# skip_without_llm is kept as a convenience decorator for class-level skips;
# _e2e_require_llm_key provides the runtime guard for tests that are already
# collected (e.g. when E2E_USE_FAKE_LLM is set only in .env.e2e, not in the
# shell environment at collection time).
skip_without_llm = pytest.mark.skipif(
    not get_e2e_settings().has_llm_key,
    reason=(
        "E2E tests require OPENAI_API_KEY, ANTHROPIC_API_KEY, "
        "or E2E_USE_FAKE_LLM=true in test/e2e_test/.env.e2e"
    ),
)

skip_without_live_services = pytest.mark.skipif(
    not _e2e_settings.is_live_mode,
    reason=(
        "Live E2E tests require E2E_MCP_*_URL values set "
        "in test/e2e_test/.env.e2e"
    ),
)

# ---------------------------------------------------------------------------
# Standard 8-operation workflow config for all E2E tests
# ---------------------------------------------------------------------------

E2E_WORKFLOW_CONFIG: dict = {
    "scan_for_work":       {"status_value": "ACCEPTED",    "human_only": True},
    "skip_rejected":       {"status_value": "REJECTED"},
    "open_for_dev":        {"status_value": "OPEN",        "human_only": False},
    "in_planning":         {"status_value": "IN PLANNING", "human_only": True},
    "start_development":   {"status_value": "IN PROGRESS"},
    "open_for_review":     {"status_value": "IN REVIEW"},
    "mark_complete":       {"status_value": "COMPLETE"},
    "update_with_context": {"status_value": ""},
}


# ---------------------------------------------------------------------------
# Autouse fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _e2e_require_llm_key(request: pytest.FixtureRequest) -> None:
    """Skip any e2e-marked test at runtime if no LLM key (or fake LLM) is available.

    Re-evaluates ``has_llm_key`` at fixture execution time so that
    ``E2E_USE_FAKE_LLM=true`` set via an inline env var or ``.env.e2e`` is
    respected even when it was not present at module import time.
    """
    if request.node.get_closest_marker("e2e") and not get_e2e_settings().has_llm_key:
        pytest.skip(
            "E2E tests require OPENAI_API_KEY, ANTHROPIC_API_KEY, "
            "or E2E_USE_FAKE_LLM=true in test/e2e_test/.env.e2e"
        )


@pytest.fixture(autouse=True)
def e2e_state_reset() -> Generator[None, None, None]:
    """Clear all shared in-memory scheduler state before and after each test."""

    def _clear() -> None:
        try:
            import src.scheduler.jobs.scan_tickets as m
            m._in_progress_tickets.clear()
            m._open_prs.clear()
            m._prs_under_review.clear()
        except (ImportError, AttributeError):
            pass

        try:
            import src.scheduler.jobs.pr_review_comment_handler as m
            m._in_progress_comment_fixes.clear()
        except (ImportError, AttributeError):
            pass

        try:
            import src.scheduler.jobs.plan_and_notify as m
            m._in_planning_tickets.clear()
            m._plan_comment_watermarks.clear()
        except (ImportError, AttributeError):
            pass

    _clear()
    yield
    _clear()


@pytest.fixture(autouse=True)
def _fake_llm_reset() -> Generator[None, None, None]:
    """Reset FakeLLM turn counter and tool order before and after each test.

    This prevents turn-counter state from leaking between tests when the
    session-scoped FakeLLM is shared.  The ``_tool_order`` is also restored
    to its pre-test value so tests that configure it (via ``set_tool_order``)
    do not affect subsequent tests.
    """
    import test.e2e_test.conftest as _self
    llm = _self._session_fake_llm
    if llm is not None:
        saved_tool_order = list(llm._tool_order)
        saved_tool_args_overrides = dict(llm._tool_args_overrides)
        llm.reset_turns()
    yield
    if llm is not None:
        llm.reset_turns()
        llm._tool_order = saved_tool_order  # type: ignore[possibly-undefined]
        llm._tool_args_overrides = saved_tool_args_overrides  # type: ignore[possibly-undefined]


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def e2e_settings() -> E2ESettings:
    """Return the cached E2ESettings singleton."""
    return _e2e_settings


@pytest.fixture(scope="session")
def e2e_workflow_config() -> WorkflowConfig:
    """Standard 8-operation WorkflowConfig for all E2E tests."""
    return WorkflowConfig(E2E_WORKFLOW_CONFIG)


# ---------------------------------------------------------------------------
# Session-scoped: testcontainers Docker Compose stack  (Mode 3)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def live_mcp_stack(e2e_settings: E2ESettings) -> Generator[dict[str, str], None, None]:
    """Start the Docker Compose MCP stack via testcontainers (Mode 3 only).

    Activated when ``E2E_USE_TESTCONTAINERS=true`` is set in
    ``test/e2e_test/.env.e2e``.

    ``E2E_USE_FAKE_LLM`` is **orthogonal** to this fixture: it controls the
    LLM layer only (FakeLLM vs real provider), not the MCP layer.  When both
    flags are ``true``, real Docker containers are started and the tests run
    against them with FakeLLM driving the tool-call sequence.  This is the
    intended way to verify real container connectivity without incurring LLM
    API costs.

    ``autouse=True`` ensures the Docker Compose stack is started exactly once
    at session start, without requiring every test to request this fixture.

    The fixture:
    - Derives the set of services to start from ``e2e_settings.configured_tc_services``
      (only services whose credentials are present in ``.env.e2e``).  This
      prevents ``docker compose up --wait`` from failing due to unhealthy
      containers caused by missing credentials.
    - Uses ``testcontainers.compose.DockerCompose`` to start only those services.
    - Blocks until all started healthchecks pass (``wait=True``).
    - Yields a ``{service_key: url}`` dict derived from the container-mapped ports.
    - Stops and removes all containers + network on session teardown via a full
      ``docker compose down --volumes`` (without service filter) so the shared
      network is also cleaned up.
    """
    # Only start containers when USE_TESTCONTAINERS=true.  USE_FAKE_LLM is
    # irrelevant here — it controls the LLM tier, not the MCP tier.
    if not e2e_settings.USE_TESTCONTAINERS:
        yield {}
        return

    try:
        from testcontainers.compose import DockerCompose
    except ImportError as exc:
        pytest.skip(
            f"testcontainers[compose] is not installed — "
            f"run `uv add --group dev 'testcontainers[compose]>=4.9.0'` to enable "
            f"Mode 3.  Original error: {exc}"
        )

    # Determine which services have credentials configured.  Starting only
    # those prevents ``docker compose up --wait`` from exiting non-zero when
    # a service (e.g. mcp-jira-e2e) cannot authenticate due to missing creds.
    services_to_start = e2e_settings.configured_tc_services
    if not services_to_start:
        pytest.skip(
            "E2E_USE_TESTCONTAINERS=true but no service credentials are configured "
            "in test/e2e_test/.env.e2e.  Set at least one of: "
            "E2E_MCP_CLICKUP_TOKEN, E2E_MCP_GITHUB_TOKEN, E2E_MCP_SLACK_TOKEN, "
            "or (E2E_ATLASSIAN_URL + E2E_ATLASSIAN_EMAIL + E2E_MCP_JIRA_TOKEN)."
        )

    import warnings
    warnings.warn(
        f"[E2E] Testcontainers mode: starting services {services_to_start}",
        stacklevel=2,
    )

    # ``context`` is the CWD passed to subprocess for every docker compose
    # invocation.  ``compose_file_name`` and ``env_file`` are forwarded as
    # ``-f`` / ``--env-file`` flags and are therefore resolved *relative to
    # that CWD*.
    #
    # Fix: use the absolute project root as context so that all paths are
    # stable regardless of the directory from which pytest is invoked.
    compose = DockerCompose(  # type: ignore[possibly-undefined]
        context=str(_PROJECT_ROOT),
        compose_file_name="test/e2e_test/docker-compose.e2e.yml",
        env_file="test/e2e_test/.env.e2e",
        pull=False,
        wait=True,
        services=services_to_start,
    )

    try:
        compose.start()
    except Exception as exc:
        # Attempt a best-effort teardown before propagating the error so that
        # partially-started containers and networks are not left behind.
        try:
            _full_compose_down(_PROJECT_ROOT)
        except Exception:
            pass
        pytest.fail(
            f"[E2E] docker compose up failed for services {services_to_start}: {exc}\n"
            "Check that Docker daemon is running and all credentials in "
            "test/e2e_test/.env.e2e are correct."
        )

    # Map service name → (logical key used by tests, container port, URL path)
    # Transport types (used by agent builders and smoke tests):
    #   GitHub   → Streamable HTTP  → path=/mcp, type=http
    #   JIRA     → Legacy SSE       → path=/sse, type=sse
    #   ClickUp  → Legacy SSE       → path=/sse, type=sse
    #   Slack    → Legacy SSE       → path=/sse, type=sse
    _SERVICE_MAP = {
        "mcp-jira-e2e":    ("jira",    18100, "/sse"),
        "mcp-clickup-e2e": ("clickup", 18101, "/sse/sse"),
        "mcp-github-e2e":  ("github",  18102, "/mcp"),
        "mcp-slack-e2e":   ("slack",   18103, "/sse"),
    }

    urls: dict[str, str] = {}
    for svc in services_to_start:
        if svc not in _SERVICE_MAP:
            continue
        key, port, path = _SERVICE_MAP[svc]
        host = compose.get_service_host(service_name=svc, port=port)
        mapped_port = compose.get_service_port(service_name=svc, port=port)
        urls[key] = f"http://{host}:{mapped_port}{path}"

    try:
        yield urls
    finally:
        # Full ``down --volumes`` (no services filter) removes containers,
        # volumes *and* the shared network so nothing is left behind.
        try:
            _full_compose_down(_PROJECT_ROOT)
        except Exception as teardown_exc:
            warnings.warn(
                f"[E2E] docker compose down failed during teardown: {teardown_exc}",
                stacklevel=2,
            )


def _full_compose_down(project_root: Path) -> None:
    """Run ``docker compose down --volumes`` for the E2E stack (no service filter).

    Called by ``live_mcp_stack`` both on normal teardown and on error so that
    partially-started containers and the shared network are always cleaned up.
    """
    import subprocess
    subprocess.run(
        [
            "docker", "compose",
            "-f", "test/e2e_test/docker-compose.e2e.yml",
            "--env-file", "test/e2e_test/.env.e2e",
            "down", "--volumes",
        ],
        cwd=str(project_root),
        capture_output=True,
        check=False,  # best-effort — don't raise even if already stopped
    )


# ---------------------------------------------------------------------------
# Session-scoped autouse: FakeLLM injection  (Mode 1)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _maybe_patch_llm_factory(
    e2e_settings: E2ESettings,
) -> Generator[None, None, None]:
    """Optionally replace ``LLMFactory.build`` with a ``FakeLLM`` for the session.

    Activated when ``E2E_USE_FAKE_LLM=true`` is set in
    ``test/e2e_test/.env.e2e``.  No action is taken when the flag is off
    (the fixture yields immediately).

    Implementation notes
    --------------------
    - ``unittest.mock.patch`` (not ``monkeypatch``) is used here because
      pytest's ``monkeypatch`` fixture is function-scoped and cannot be used
      inside session-scoped fixtures.
    - The patch target is ``src.agents.llm_factory.LLMFactory.build`` —
      exactly where ``AgentFactory`` calls it — so every agent constructed
      during the session receives a ``FakeLLM`` instance.
    - The shared ``FakeLLM`` instance is stored on the module so that the
      ``fake_llm_session`` fixture can expose it for inspection.
    """
    if not e2e_settings.USE_FAKE_LLM:
        yield
        return

    from unittest.mock import patch

    _shared_fake_llm = FakeLLM()

    # Store on module so fake_llm_session can retrieve it.
    import test.e2e_test.conftest as _self
    _self._session_fake_llm = _shared_fake_llm

    with patch(
        "src.agents.llm_factory.LLMFactory.build",
        return_value=_shared_fake_llm,
    ):
        yield

    # Clean up the module-level reference after session teardown.
    _self._session_fake_llm = None


# Module-level slot for the shared FakeLLM instance (set by _maybe_patch_llm_factory).
_session_fake_llm: Optional[FakeLLM] = None


# ---------------------------------------------------------------------------
# Session-scoped: shared FakeLLM instance accessor
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fake_llm_session(e2e_settings: E2ESettings) -> Optional[FakeLLM]:
    """Return the shared ``FakeLLM`` instance active for this session, or ``None``.

    Use this when tests in the same session need to inspect the global call
    log (e.g. asserting that ``LLMFactory.build`` was called N times across
    multiple agents).

    Returns ``None`` when ``E2E_USE_FAKE_LLM=false`` — callers should guard::

        def test_something(fake_llm_session):
            if fake_llm_session is None:
                pytest.skip("fake LLM not active")
    """
    import test.e2e_test.conftest as _self
    return _self._session_fake_llm


# ---------------------------------------------------------------------------
# Function-scoped: per-test FakeLLM instance
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_llm(e2e_settings: E2ESettings) -> Optional[FakeLLM]:
    """Return a fresh ``FakeLLM`` instance for this test, or ``None``.

    Use this fixture when a test needs to register custom keyword→response
    mappings before constructing an agent.  The fresh instance is independent
    of the session-level patch — the test is responsible for passing it to
    the agent builder or using it with ``monkeypatch`` to override the factory.

    Returns ``None`` when ``E2E_USE_FAKE_LLM=false``.

    Example::

        def test_custom_response(fake_llm, monkeypatch):
            if fake_llm is None:
                pytest.skip("fake LLM not active")
            fake_llm.register("search_issues", "No open tickets found.")
            monkeypatch.setattr(
                "src.agents.llm_factory.LLMFactory.build", lambda *a, **kw: fake_llm
            )
            # ... run agent code ...
            assert fake_llm.was_called()
    """
    if not e2e_settings.USE_FAKE_LLM:
        return None
    return FakeLLM()


# ---------------------------------------------------------------------------
# Function-scoped: three-mode URL selector
# ---------------------------------------------------------------------------


@pytest.fixture
def mcp_urls(
    e2e_settings: E2ESettings,
    httpserver: HTTPServer,
    live_mcp_stack: dict[str, str],
) -> dict[str, str]:
    """Return MCP server URLs for the active test mode.

    Decision logic
    --------------
    - ``USE_TESTCONTAINERS=true``       → live container URLs from ``live_mcp_stack``
                                          (falls back to stub URL for unconfigured services)
    - ``USE_FAKE_LLM=true`` only        → stub URLs (no containers, FakeLLM drives calls)
    - neither flag                      → stub URLs (in-process stub)

    Note: ``USE_TESTCONTAINERS`` takes priority over ``USE_FAKE_LLM`` for MCP
    URL selection.  When both are ``true``, real containers are running and
    FakeLLM is used to drive the tool-call sequence against them — this is a
    valid and intentional combination for verifying container connectivity
    without incurring LLM API costs.
    """
    stub_url = httpserver.url_for("/mcp")

    if e2e_settings.USE_TESTCONTAINERS:
        # Real containers are running (started by live_mcp_stack).
        # For services not configured (absent from live_mcp_stack), fall back
        # to the stub URL so agents can still be constructed without error.
        return {
            "jira":    live_mcp_stack.get("jira",    stub_url),
            "clickup": live_mcp_stack.get("clickup", stub_url),
            "github":  live_mcp_stack.get("github",  stub_url),
            "slack":   live_mcp_stack.get("slack",   stub_url),
        }

    # Stub mode: FakeLLM-only or no flags set.
    return {"jira": stub_url, "clickup": stub_url, "github": stub_url, "slack": stub_url}


# ---------------------------------------------------------------------------
# Function-scoped: MCPStubServer factory — stub mode only
# ---------------------------------------------------------------------------


@pytest.fixture
def mcp_stub(
    e2e_settings: E2ESettings,
    httpserver: HTTPServer,
) -> "MCPStubServer":
    """Return a ready-to-use ``MCPStubServer`` backed by ``pytest-httpserver``.

    This fixture provides a stub server that works in both modes:

    - Stub mode (``USE_TESTCONTAINERS=false``): Agent connects to stub, 
      all tool calls are recorded for assertions

    - Live mode (``USE_TESTCONTAINERS=true``): Agent connects to live containers,
      but stub is still available for compatibility. Tests should use 
      ``e2e_settings.USE_TESTCONTAINERS`` to conditionally run stub-specific
      assertions.

    The fixture never skips tests - it's up to individual tests to handle
    the different modes appropriately.
    """
    # Always return a stub server - tests can decide how to use it
    return MCPStubServer(httpserver)


# ---------------------------------------------------------------------------
# Internal LLM key injection helpers
# ---------------------------------------------------------------------------


def _inject_llm_key(settings: E2ESettings) -> dict[str, str]:
    """Temporarily inject the LLM API key into os.environ for CrewAI validators."""
    llm_key = settings.llm_key_value
    llm_env_var = settings.llm_key_env_var
    injected: dict[str, str] = {}
    if llm_key and llm_env_var not in os.environ:
        os.environ[llm_env_var] = llm_key
        injected[llm_env_var] = llm_key
    return injected


def _restore_env(injected: dict[str, str]) -> None:
    for key in injected:
        os.environ.pop(key, None)


def _mcp_credentials_kwargs(settings: E2ESettings) -> dict[str, str]:
    """Return AppSettings kwargs carrying MCP auth secrets."""

    def _secret_value(secret: Optional[SecretStr]) -> Optional[str]:
        return secret.get_secret_value() if secret else None

    creds: dict[str, str] = {}
    for attr in ("MCP_JIRA_TOKEN", "MCP_CLICKUP_TOKEN", "MCP_GITHUB_TOKEN", "MCP_SLACK_TOKEN"):
        value = _secret_value(getattr(settings, attr, None))
        if value:
            creds[attr] = value
    return creds


# ---------------------------------------------------------------------------
# Transport-type helper
# ---------------------------------------------------------------------------


def _mcp_type_for_url(url: str) -> str:
    """Return the correct MCP transport type string for *url*.

    Decision logic:
    - URLs ending in ``/sse`` (or containing ``/sse?``) → ``"sse"``
      (legacy SSE transport — JIRA, ClickUp, Slack containers).
    - All other URLs → ``"http"``
      (Streamable HTTP transport — GitHub container, all stub URLs ending in ``/mcp``).

    This matters because CrewAI / the MCP Python client will use different
    connection logic depending on ``type``.  Connecting with ``"http"`` to an
    SSE-only server returns 405 Method Not Allowed; connecting with ``"sse"``
    to an HTTP server returns a non-event-stream response.
    """
    path = url.split("?")[0].rstrip("/")
    if path.endswith("/sse"):
        return "sse"
    return "http"


# ---------------------------------------------------------------------------
# Centralised agent builders — all three roles
# ---------------------------------------------------------------------------


def build_dev_agent_against_stubs(
    jira_url: str,
    clickup_url: str,
    github_url: str,
    slack_url: str,
    e2e_settings: Optional[E2ESettings] = None,
) -> Any:
    """Build a real dev_agent CrewAI Agent pointing at stub or live MCP servers.

    ``e2e_settings`` defaults to the module-level singleton so call sites that
    omit it (e.g. tests that build agents inline without the fixture) still
    receive a correctly configured LLM — including ``FakeLLM`` when
    ``E2E_USE_FAKE_LLM=true``.
    """
    settings_obj: E2ESettings = e2e_settings or _e2e_settings
    from src.agents.factory import AgentFactory
    from src.config.agent_config import AgentTeamConfig
    from src.config.settings import AppSettings

    injected = _inject_llm_key(settings_obj)
    try:
        settings_kwargs: dict = dict(
            MCP_JIRA_URL=jira_url, MCP_CLICKUP_URL=clickup_url,
            MCP_GITHUB_URL=github_url, MCP_SLACK_URL=slack_url,
        )
        settings_kwargs.update(_mcp_credentials_kwargs(settings_obj))
        llm_key = settings_obj.llm_key_value
        if llm_key:
            settings_kwargs[settings_obj.llm_key_env_var] = llm_key

        settings = AppSettings(**settings_kwargs)
        raw_config = {
            "process": "sequential",
            "mcp_servers": {
                "jira": {
                    "type": _mcp_type_for_url(jira_url), "url": jira_url,
                    "tool_filter": [
                        "search_issues", "get_issue", "transition_issue",
                        "add_comment", "update_issue",
                    ],
                    "cache_tools_list": False,
                },
                "clickup": {
                    "type": _mcp_type_for_url(clickup_url), "url": clickup_url,
                    "tool_filter": [
                        "search_tasks", "get_task", "update_task", "add_comment",
                    ],
                    "cache_tools_list": False,
                },
                "github": {
                    "type": _mcp_type_for_url(github_url), "url": github_url,
                    "tool_filter": [
                        "create_pull_request", "get_pull_request", "merge_pull_request",
                        "get_pull_request_reviews", "get_pull_request_comments",
                        "reply_to_review_comment",
                    ],
                    "cache_tools_list": False,
                },
                "slack": {
                    "type": _mcp_type_for_url(slack_url), "url": slack_url,
                    "tool_filter": [
                        "send_message", "reply_to_thread", "get_messages",
                        "get_thread_permalink",
                    ],
                    "cache_tools_list": False,
                },
            },
            "agents": [{
                "id": "dev_agent",
                "role": "Software Developer",
                "goal": (
                    "Scan for ready development tickets, implement them, open GitHub pull "
                    "requests, and update ticket status for human review."
                ),
                "backstory": (
                    "You are a skilled software developer who thrives in an "
                    "autonomous, pull-based workflow."
                ),
                "llm": {
                    "provider": settings_obj.LLM_PROVIDER,
                    "model": settings_obj.LLM_MODEL,
                },
                "mcps": ["jira", "clickup", "github", "slack"],
                "apps": [], "allow_delegation": False, "verbose": False,
            }],
        }
        team_config = AgentTeamConfig.model_validate(raw_config)
        return AgentFactory.build(
            team_config.agents[0], settings, mcp_servers=team_config.mcp_servers
        )
    finally:
        _restore_env(injected)


def build_planner_agent_against_stubs(url: str, e2e_settings: Optional[E2ESettings] = None) -> Any:
    """Build a real planner CrewAI Agent pointing at stub or live MCP servers."""
    from src.agents.factory import AgentFactory
    from src.config.agent_config import AgentTeamConfig
    from src.config.settings import AppSettings

    settings_obj: E2ESettings = e2e_settings or _e2e_settings
    injected = _inject_llm_key(settings_obj)
    try:
        settings_kwargs: dict = dict(
            MCP_JIRA_URL=url, MCP_CLICKUP_URL=url,
            MCP_GITHUB_URL=url, MCP_SLACK_URL=url,
        )
        settings_kwargs.update(_mcp_credentials_kwargs(settings_obj))
        settings_kwargs.update(_mcp_credentials_kwargs(settings_obj))
        llm_key = settings_obj.llm_key_value
        if llm_key:
            settings_kwargs[settings_obj.llm_key_env_var] = llm_key

        settings = AppSettings(**settings_kwargs)
        raw_config = {
            "process": "sequential",
            "mcp_servers": {
                "jira": {
                    "type": _mcp_type_for_url(url), "url": url,
                    "tool_filter": [
                        "create_issue", "search_issues", "update_issue",
                        "transition_issue", "add_comment",
                    ],
                    "cache_tools_list": False,
                },
                "clickup": {
                    "type": _mcp_type_for_url(url), "url": url,
                    "tool_filter": [
                        "create_task", "search_tasks", "update_task", "add_comment",
                    ],
                    "cache_tools_list": False,
                },
                "slack": {
                    "type": _mcp_type_for_url(url), "url": url,
                    "tool_filter": ["send_message", "reply_to_thread", "get_messages"],
                    "cache_tools_list": False,
                },
            },
            "agents": [{
                "id": "planner",
                "role": "Product Planner",
                "goal": (
                    "Survey and discuss product ideas with human stakeholders; "
                    "create detailed planning documents; manage accept/reject outcomes."
                ),
                "backstory": (
                    "You are a seasoned product planner with 10+ years of experience "
                    "in agile software development, market research, and business modelling."
                ),
                "llm": {
                    "provider": settings_obj.LLM_PROVIDER,
                    "model": settings_obj.LLM_MODEL,
                },
                "mcps": ["jira", "clickup", "slack"],
                "apps": [], "allow_delegation": False, "verbose": False,
            }],
        }
        team_config = AgentTeamConfig.model_validate(raw_config)
        return AgentFactory.build(
            team_config.agents[0], settings, mcp_servers=team_config.mcp_servers
        )
    finally:
        _restore_env(injected)


def build_dev_lead_agent_against_stubs(url: str, e2e_settings: Optional[E2ESettings] = None) -> Any:
    """Build a real dev_lead CrewAI Agent pointing at stub or live MCP servers."""
    from src.agents.factory import AgentFactory
    from src.config.agent_config import AgentTeamConfig
    from src.config.settings import AppSettings

    settings_obj: E2ESettings = e2e_settings or _e2e_settings
    injected = _inject_llm_key(settings_obj)
    try:
        settings_kwargs: dict = dict(
            MCP_JIRA_URL=url, MCP_CLICKUP_URL=url,
            MCP_GITHUB_URL=url, MCP_SLACK_URL=url,
        )
        llm_key = settings_obj.llm_key_value
        if llm_key:
            settings_kwargs[settings_obj.llm_key_env_var] = llm_key

        settings = AppSettings(**settings_kwargs)
        raw_config = {
            "process": "sequential",
            "mcp_servers": {
                "jira": {
                    "type": _mcp_type_for_url(url), "url": url,
                    "tool_filter": [
                        "get_issue", "search_issues", "create_issue",
                        "update_issue", "link_issues", "transition_issue", "add_comment",
                    ],
                    "cache_tools_list": False,
                },
                "clickup": {
                    "type": _mcp_type_for_url(url), "url": url,
                    "tool_filter": [
                        "get_task", "search_tasks", "create_task",
                        "update_task", "add_comment",
                    ],
                    "cache_tools_list": False,
                },
                "slack": {
                    "type": _mcp_type_for_url(url), "url": url,
                    "tool_filter": ["send_message", "reply_to_thread", "get_messages"],
                    "cache_tools_list": False,
                },
            },
            "agents": [{
                "id": "dev_lead",
                "role": "Technical Dev Lead",
                "goal": (
                    "Assess feasibility of requirements, ask clarifying questions, "
                    "break down epics into sub-tasks with dependencies."
                ),
                "backstory": (
                    "You are a senior software engineer and tech lead specialising "
                    "in system design, task decomposition, and agile estimation."
                ),
                "llm": {
                    "provider": settings_obj.LLM_PROVIDER,
                    "model": settings_obj.LLM_MODEL,
                },
                "mcps": ["jira", "clickup", "slack"],
                "apps": [], "allow_delegation": False, "verbose": False,
            }],
        }
        team_config = AgentTeamConfig.model_validate(raw_config)
        return AgentFactory.build(
            team_config.agents[0], settings, mcp_servers=team_config.mcp_servers
        )
    finally:
        _restore_env(injected)


def build_e2e_registry(agent: Any, role_id: str = "dev_agent") -> Any:
    """Wrap *agent* in an AgentRegistry under *role_id*."""
    from src.agents.registry import AgentRegistry

    registry = AgentRegistry()
    registry.register(role_id, agent)
    return registry

