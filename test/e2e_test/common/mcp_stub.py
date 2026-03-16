"""
MCP JSON-RPC 2.0 stub server for E2E tests.

``MCPStubServer`` wraps a ``pytest-httpserver`` instance and speaks the
MCP JSON-RPC 2.0 protocol so that CrewAI's ``MCPServerHTTP`` transport
can connect, discover tools, and call them normally.

Usage::

    stub = MCPStubServer(httpserver)
    stub.register_tool("search_issues", lambda args: [{"key": "PROJ-1"}])
    # ... run agent code ...
    calls = stub.calls_to("search_issues")
    assert len(calls) > 0
    assert stub.was_called("add_comment")
"""
from __future__ import annotations

import json
from typing import Any, List

from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Request, Response


def _make_jsonrpc_response(request_id: Any, result: Any) -> dict:
    """Build a JSON-RPC 2.0 success response."""
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _make_jsonrpc_error(request_id: Any, code: int, message: str) -> dict:
    """Build a JSON-RPC 2.0 error response."""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def _tool_result(content: Any) -> dict:
    """Wrap a result value in the MCP tools/call response structure."""
    return {
        "content": [{"type": "text", "text": json.dumps(content)}],
        "isError": False,
    }


class MCPStubServer:
    """A pytest-httpserver-backed stub that speaks MCP JSON-RPC 2.0 protocol.

    Advertises a full set of default tools (JIRA + ClickUp + GitHub + Slack)
    via ``tools/list``.  Individual handlers are registered per tool name and
    invoked when the agent calls ``tools/call``.

    Usage::

        stub = MCPStubServer(httpserver)
        stub.register_tool("search_issues", lambda args: [{"key": "PROJ-1"}])
        # ... run agent code ...
        calls = stub.calls_to("search_issues")
    """

    DEFAULT_TOOLS = [
        # JIRA
        {"name": "search_issues", "description": "Search JIRA issues",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": "get_issue", "description": "Get JIRA issue",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": "transition_issue", "description": "Transition JIRA issue",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": "add_comment", "description": "Add comment to ticket or task",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": "update_issue", "description": "Update JIRA issue",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": "create_issue", "description": "Create JIRA issue",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": "link_issues", "description": "Link JIRA issues",
         "inputSchema": {"type": "object", "properties": {}}},
        # ClickUp
        {"name": "search_tasks", "description": "Search ClickUp tasks",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": "get_task", "description": "Get ClickUp task",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": "update_task", "description": "Update ClickUp task",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": "create_task", "description": "Create ClickUp task",
         "inputSchema": {"type": "object", "properties": {}}},
        # GitHub
        {"name": "create_pull_request", "description": "Create GitHub PR",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": "get_pull_request", "description": "Get GitHub PR",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": "merge_pull_request", "description": "Merge GitHub PR",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": "get_pull_request_reviews", "description": "Get PR reviews",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": "get_pull_request_comments", "description": "Get PR inline comments",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": "reply_to_review_comment", "description": "Reply to a PR inline comment",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": "approve_pull_request", "description": "Approve PR",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": "submit_review", "description": "Submit PR review",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": "create_commit", "description": "Create a commit",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": "push_commits", "description": "Push commits",
         "inputSchema": {"type": "object", "properties": {}}},
        # Slack
        {"name": "get_messages", "description": "Get Slack channel/thread messages",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": "reply_to_thread", "description": "Reply in a Slack thread",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": "send_message", "description": "Send a Slack message",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": "get_thread_permalink", "description": "Get Slack thread permalink",
         "inputSchema": {"type": "object", "properties": {}}},
    ]

    def __init__(self, server: HTTPServer) -> None:
        self._server = server
        self._tool_handlers: dict[str, Any] = {}
        self._calls: list[dict] = []
        self._server.expect_request("/mcp").respond_with_handler(self._dispatch)

    def register_tool(self, tool_name: str, handler: Any) -> None:
        """Register a callable that handles ``tools/call`` for *tool_name*.

        The callable receives the ``arguments`` dict and returns a JSON-serialisable result.
        """
        self._tool_handlers[tool_name] = handler

    def _dispatch(self, req: Request) -> Response:
        """Handle all MCP JSON-RPC 2.0 requests."""
        try:
            body = json.loads(req.data.decode())
        except Exception:
            return Response(
                json.dumps(_make_jsonrpc_error(None, -32700, "Parse error")),
                status=200,
                content_type="application/json",
            )

        req_id = body.get("id")
        method = body.get("method", "")

        if method == "initialize":
            return self._ok(req_id, {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "e2e-stub", "version": "1.0.0"},
            })

        if method == "notifications/initialized":
            return Response("", status=200)

        if method == "tools/list":
            return self._ok(req_id, {"tools": self.DEFAULT_TOOLS})

        if method == "tools/call":
            params = body.get("params", {})
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            self._calls.append({"tool": tool_name, "arguments": arguments})

            handler = self._tool_handlers.get(tool_name)
            if handler is not None:
                try:
                    result = handler(arguments)
                except Exception as exc:
                    return self._ok(req_id, {
                        "content": [{"type": "text", "text": str(exc)}],
                        "isError": True,
                    })
                return self._ok(req_id, _tool_result(result))

            # Default: return empty success for unregistered tools.
            return self._ok(req_id, _tool_result({"ok": True}))

        return self._ok(req_id, {})

    def _ok(self, req_id: Any, result: Any) -> Response:
        return Response(
            json.dumps(_make_jsonrpc_response(req_id, result)),
            status=200,
            content_type="application/json",
        )

    def calls_to(self, tool_name: str) -> List[dict]:
        """Return all recorded argument dicts for calls to *tool_name*."""
        return [c["arguments"] for c in self._calls if c["tool"] == tool_name]

    def was_called(self, tool_name: str) -> bool:
        """True if *tool_name* was called at least once."""
        return any(c["tool"] == tool_name for c in self._calls)

    def was_never_called(self, tool_name: str) -> bool:
        """True if *tool_name* was never called."""
        return not self.was_called(tool_name)

    @property
    def all_calls(self) -> List[dict]:
        """All recorded tool calls as ``{"tool": ..., "arguments": ...}`` dicts."""
        return list(self._calls)

    @property
    def url(self) -> str:
        """The full URL of the stub's MCP endpoint."""
        return self._server.url_for("/mcp")

    def reset(self) -> None:
        """Clear all recorded calls (useful between steps in compound tests)."""
        self._calls.clear()


# Backward-compatibility alias used by some existing test files.
RecordingStub = MCPStubServer

