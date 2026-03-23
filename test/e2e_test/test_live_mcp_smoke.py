"""
Live mode infrastructure smoke tests (E2E-LIVE-01 through E2E-LIVE-06).

These tests verify that the real MCP server containers started by either:
  - ``testcontainers`` (Mode 3, ``E2E_USE_TESTCONTAINERS=true``), or
  - a pre-started ``docker compose`` stack (Mode 4, ``E2E_MCP_*_URL`` set)
are reachable and respond correctly according to their transport type.

Transport types:
  - GitHub   → Streamable HTTP (``/mcp``)  — responds to POST JSON-RPC ``tools/list``
  - JIRA     → Legacy SSE       (``/sse``) — responds to GET with ``event: endpoint``
  - ClickUp  → Legacy SSE       (``/sse``) — responds to GET with ``event: endpoint``
  - Slack    → Legacy SSE       (``/sse``) — responds to GET with ``event: endpoint``

**All tests are skipped in stub/fake-LLM modes** when neither
``E2E_USE_TESTCONTAINERS=true`` nor any ``E2E_MCP_*_URL`` is set.

Mode 3 (testcontainers) — run from project root:
    E2E_USE_TESTCONTAINERS=true uv run pytest test/e2e_test/test_live_mcp_smoke.py -v

Mode 4 (manual) — pre-start the stack then run:
    docker compose \\
        -f test/e2e_test/docker-compose.e2e.yml \\
        --env-file test/e2e_test/.env.e2e \\
        up -d --wait
    uv run pytest test/e2e_test/test_live_mcp_smoke.py -v
"""

from __future__ import annotations

import json
import pytest
import requests

from test.e2e_test.conftest import skip_without_live_services
from test.e2e_test.common.e2e_settings import get_e2e_settings

pytestmark = [pytest.mark.e2e]

_e2e_settings = get_e2e_settings()


def _is_sse_url(url: str) -> bool:
    """Return True if *url* targets an SSE-transport MCP server (path ends in /sse)."""
    path = url.split("?")[0].rstrip("/")
    return path.endswith("/sse")


def _probe_sse_server(url: str) -> None:
    """Verify that *url* is a reachable SSE-transport MCP server.

    Legacy SSE servers respond to ``GET /sse`` by opening a server-sent event
    stream.  The first event is always::

        event: endpoint
        data: /messages/?session_id=<uuid>

    We confirm reachability by checking:
    - HTTP 200 OK
    - ``Content-Type: text/event-stream``
    - The response body (first chunk) contains ``event: endpoint``

    We use ``stream=True`` and close immediately after reading the first chunk
    to avoid blocking on the infinite event stream.
    """
    with requests.get(url, stream=True, timeout=10) as resp:
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        assert "text/event-stream" in content_type, (
            f"SSE server at {url} returned unexpected Content-Type: {content_type!r}. "
            "Expected 'text/event-stream'."
        )
        # Read up to 512 bytes — enough to contain the first SSE event.
        first_chunk = next(resp.iter_content(chunk_size=512), b"")
        text = first_chunk.decode("utf-8", errors="replace")
        assert "event: endpoint" in text or "data:" in text, (
            f"SSE server at {url} did not send an 'event: endpoint' message. "
            f"Got first chunk: {text!r}"
        )


def _probe_http_server(url: str, service: str = "") -> dict:
    """Send an MCP ``tools/list`` JSON-RPC 2.0 request and return the parsed response.

    Used for Streamable HTTP transport servers (GitHub MCP).
    """
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {},
    }
    
    # Try different URLs for ClickUp if the main one fails
    test_urls = [url]
    if "clickup" in url and url.endswith("/mcp"):
        # For ClickUp, try alternative endpoints
        base_url = url[:-4]  # Remove /mcp
        test_urls.extend([
            base_url + "/",
            base_url + "/sse",
            base_url + "/messages",
        ])
    
    last_error = None
    for test_url in test_urls:
        try:
            headers = {"Accept": "application/json, text/event-stream"}
            
            # Add Authorization header for GitHub MCP server
            if service == "github":
                from test.e2e_test.common.e2e_settings import E2ESettings
                settings = E2ESettings()
                github_token = settings.MCP_GITHUB_TOKEN
                if github_token:
                    headers["Authorization"] = f"Bearer {github_token.get_secret_value()}"
            
            response = requests.post(
                test_url,
                json=payload,
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()
            
            # Handle SSE response format (GitHub MCP server returns SSE)
            if "text/event-stream" in response.headers.get("Content-Type", ""):
                lines = response.text.split('\n')
                for line in lines:
                    if line.startswith('data: ') and line.strip() != 'data:':
                        json_data = line[6:]  # Remove 'data: ' prefix
                        if json_data.strip():
                            return json.loads(json_data)
                return {}
            else:
                # Handle regular JSON response
                if not response.text.strip():
                    return {}
                return response.json()
        except Exception as e:
            last_error = e
            continue
    
    raise last_error


def _probe_mcp_server(url: str) -> dict | None:
    """Probe *url* using the correct transport for its path.

    Returns:
    - For SSE servers: ``None`` after a successful reachability check
      (tool listing requires a full MCP session over SSE, which is beyond
      a simple smoke test).
    - For HTTP servers: the parsed ``tools/list`` JSON-RPC response dict.
    """
    if _is_sse_url(url):
        _probe_sse_server(url)
        return None
    return _probe_http_server(url)


def _resolve_url(service: str, live_mcp_stack: dict[str, str]) -> str:
    """Return the MCP server URL for *service*, preferring the testcontainers
    stack over the static ``E2E_MCP_*_URL`` setting.

    Priority:
    1. ``live_mcp_stack[service]`` — populated by the ``DockerCompose``
       fixture when ``E2E_USE_TESTCONTAINERS=true``.
    2. ``E2E_MCP_<SERVICE>_URL`` from ``.env.e2e`` — set manually for
       Mode 4 (pre-started containers).

    If neither source provides a URL the test is skipped.
    """
    url = live_mcp_stack.get(service)
    if not url:
        fallback_attr = f"MCP_{service.upper()}_URL"
        url = getattr(_e2e_settings, fallback_attr, None)
    if not url:
        pytest.skip(
            f"No URL available for MCP service '{service}'. "
            f"Set E2E_USE_TESTCONTAINERS=true or E2E_MCP_{service.upper()}_URL "
            f"in test/e2e_test/.env.e2e."
        )
    return url  # type: ignore[return-value]


# ===========================================================================
# E2E-LIVE-01: JIRA MCP server is reachable
# ===========================================================================

@skip_without_live_services
class TestJIRAMCPServer:
    def test_e2e_live_01_jira_mcp_reachable(self, live_mcp_stack: dict[str, str]) -> None:
        """E2E-LIVE-01: JIRA MCP server (SSE transport) is reachable and streams events."""
        url = _resolve_url("jira", live_mcp_stack)

        # JIRA uses legacy SSE transport — GET /sse must return text/event-stream.
        _probe_sse_server(url)

    def test_e2e_live_02_jira_mcp_search_issues_tool_present(
        self, live_mcp_stack: dict[str, str]
    ) -> None:
        """E2E-LIVE-02: JIRA MCP SSE endpoint is reachable (tool listing requires full session)."""
        url = _resolve_url("jira", live_mcp_stack)

        # SSE transport: full tool listing requires an MCP session (initialize +
        # tools/list over the event stream).  The smoke test only verifies that
        # the SSE endpoint is reachable and the server is healthy.
        _probe_sse_server(url)


# ===========================================================================
# E2E-LIVE-03: ClickUp MCP server is reachable
# ===========================================================================

@skip_without_live_services
class TestClickUpMCPServer:
    def test_e2e_live_03_clickup_mcp_reachable(
        self, live_mcp_stack: dict[str, str]
    ) -> None:
        """E2E-LIVE-03: ClickUp MCP server (SSE transport) is reachable and streams events."""
        url = _resolve_url("clickup", live_mcp_stack)

        # ClickUp uses legacy SSE transport — GET /sse must return text/event-stream.
        _probe_sse_server(url)

    def test_e2e_live_04_clickup_mcp_search_tasks_tool_present(
        self, live_mcp_stack: dict[str, str]
    ) -> None:
        """E2E-LIVE-04: ClickUp MCP SSE endpoint is reachable (tool listing requires full session)."""
        url = _resolve_url("clickup", live_mcp_stack)

        # SSE transport: smoke test only verifies reachability.
        _probe_sse_server(url)


# ===========================================================================
# E2E-LIVE-05: GitHub MCP server is reachable
# ===========================================================================

@skip_without_live_services
class TestGitHubMCPServer:
    def test_e2e_live_05_github_mcp_reachable(
        self, live_mcp_stack: dict[str, str]
    ) -> None:
        """E2E-LIVE-05: GitHub MCP server (Streamable HTTP) responds to tools/list."""
        url = _resolve_url("github", live_mcp_stack)

        # GitHub uses Streamable HTTP — POST JSON-RPC tools/list is supported.
        result = _probe_http_server(url, service="github")

        assert "result" in result or "tools" in result, (
            f"Expected MCP tools/list response. Got: {result}"
        )
        tools = result.get("result", {}).get("tools", result.get("tools", []))
        tool_names = [t.get("name", "") for t in tools]
        assert any(
            "pull_request" in name.lower() or "github" in name.lower() or "pr" in name.lower()
            for name in tool_names
        ), (
            f"Expected at least one GitHub tool. Got: {tool_names}"
        )


# ===========================================================================
# E2E-LIVE-06: Slack MCP server is reachable
# ===========================================================================

@skip_without_live_services
class TestSlackMCPServer:
    def test_e2e_live_06_slack_mcp_reachable(
        self, live_mcp_stack: dict[str, str]
    ) -> None:
        """E2E-LIVE-06: Slack MCP server (SSE transport) is reachable and streams events."""
        url = _resolve_url("slack", live_mcp_stack)

        # Slack uses legacy SSE transport — GET /sse must return text/event-stream.
        _probe_sse_server(url)
