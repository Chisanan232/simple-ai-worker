"""
Direct REST API clients for ClickUp and JIRA ticket queries.

These clients replace the LLM-driven crew path for
:meth:`~src.ticket.tracker.TicketTracker.fetch_tickets_for_operation`.
Fetching "which tasks are ACCEPTED?" is a fully deterministic read with a
fixed input/output schema — no LLM reasoning is required.  Using a direct
HTTP call removes the LLM token cost and latency from every scheduler poll.

Classes
-------
TicketFetchError
    Raised by both clients on non-2xx HTTP responses or network errors.
ClickUpRestClient
    Queries the ClickUp REST API v2 for tasks in a given list by status.
    Handles pagination internally.
JiraRestClient
    Queries the JIRA REST API v3 (Atlassian Cloud / Server) using JQL.
    Handles pagination internally.

Auth details
------------
ClickUp
    ``Authorization: {api_token}`` — the personal API token (``pk_…``) is
    used *directly* in the header without a ``Bearer`` prefix.
JIRA Cloud
    ``Authorization: Basic base64("{email}:{api_token}")`` — uses
    ``ATLASSIAN_EMAIL`` and ``MCP_JIRA_TOKEN`` from ``AppSettings``.
"""

from __future__ import annotations

import base64
import logging
from typing import List, Optional

import httpx

__all__: List[str] = [
    "TicketFetchError",
    "ClickUpRestClient",
    "JiraRestClient",
]

logger = logging.getLogger(__name__)

# ClickUp v2 API returns a maximum of 100 tasks per page.
_CLICKUP_PAGE_SIZE: int = 100
# JIRA REST API: sensible page size that keeps payloads small.
_JIRA_PAGE_SIZE: int = 50
# Shared HTTP timeout for all REST calls (seconds).
_HTTP_TIMEOUT: float = 30.0


class TicketFetchError(Exception):
    """Raised when a REST API call fails.

    Wraps both network-level errors (``httpx.TransportError``) and
    non-2xx HTTP responses so callers only need to handle one exception
    type.

    Attributes:
        source: ``"clickup"`` or ``"jira"`` — which client raised this.
        status_code: HTTP status code if available, otherwise ``None``.
    """

    def __init__(
        self,
        message: str,
        source: str,
        status_code: Optional[int] = None,
    ) -> None:
        self.source = source
        self.status_code = status_code
        super().__init__(message)


# ---------------------------------------------------------------------------
# ClickUp REST Client
# ---------------------------------------------------------------------------


class ClickUpRestClient:
    """Direct ClickUp REST API v2 client for ticket queries.

    Calls ``GET /api/v2/list/{list_id}/task`` to find all tasks matching
    a given status.  Pagination is handled internally — callers receive a
    flat list regardless of how many pages were fetched.

    Parameters
    ----------
    api_token:
        The ClickUp personal API token (``pk_…``).  Stored as
        ``MCP_CLICKUP_TOKEN`` in ``AppSettings``.  Used directly in the
        ``Authorization`` header **without** a ``Bearer`` prefix.
    list_id:
        The ClickUp list ID to scope the query.  The v2 API does not
        expose a workspace-wide "search by status" endpoint — a list ID
        is required.  Stored as ``CLICKUP_LIST_ID`` in ``AppSettings``.

    Raises
    ------
    ValueError
        At construction time if *api_token* or *list_id* is empty.
    """

    _BASE_URL: str = "https://api.clickup.com/api/v2"

    def __init__(self, api_token: str, list_id: str) -> None:
        if not api_token:
            raise ValueError(
                "ClickUpRestClient: api_token must not be empty. " "Set MCP_CLICKUP_TOKEN in your .env file."
            )
        if not list_id:
            raise ValueError("ClickUpRestClient: list_id must not be empty. " "Set CLICKUP_LIST_ID in your .env file.")
        self._api_token = api_token
        self._list_id = list_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_tasks(self, status: str, exclude_status: str) -> List[dict]:
        """Return all tasks in the list whose status equals *status*.

        Fetches all pages and returns a flat list.  The *exclude_status*
        filter is applied as a Python-side post-filter after fetching
        (ClickUp's ``statuses[]`` param is an include filter, not an
        exclude filter).

        Args:
            status:         Status string to match (e.g. ``"ACCEPTED"``).
            exclude_status: Status string to exclude (e.g. ``"REJECTED"``).
                            Applied as a Python-side guard.

        Returns:
            A list of normalised dicts with keys
            ``id``, ``title``, ``url``, ``status``.

        Raises:
            TicketFetchError: On any non-2xx response or network error.
        """
        url = f"{self._BASE_URL}/list/{self._list_id}/task"
        headers = {
            # ClickUp: token directly in Authorization — no "Bearer" prefix.
            "Authorization": self._api_token,
            "Content-Type": "application/json",
        }

        results: List[dict] = []
        page = 0

        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            while True:
                params = {
                    "statuses[]": status,
                    "page": str(page),
                    "include_closed": "false",
                }
                logger.debug("ClickUpRestClient: GET %s page=%d status=%r", url, page, status)
                try:
                    response = client.get(url, headers=headers, params=params)
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise TicketFetchError(
                        f"ClickUp API returned {exc.response.status_code} for list "
                        f"'{self._list_id}': {exc.response.text[:200]}",
                        source="clickup",
                        status_code=exc.response.status_code,
                    ) from exc
                except httpx.TransportError as exc:
                    raise TicketFetchError(
                        f"ClickUp API network error: {exc}",
                        source="clickup",
                    ) from exc

                data = response.json()
                tasks: list = data.get("tasks", [])
                logger.debug("ClickUpRestClient: page=%d returned %d task(s).", page, len(tasks))

                for task in tasks:
                    raw_status = self._extract_status(task)
                    # Python-side exclude guard (belt-and-suspenders).
                    if raw_status.strip().lower() == exclude_status.strip().lower():
                        continue
                    results.append(self._normalise(task))

                # ClickUp returns up to 100 items per page.  Fewer means last page.
                if len(tasks) < _CLICKUP_PAGE_SIZE:
                    break
                page += 1

        logger.info(
            "ClickUpRestClient: found %d task(s) with status=%r in list '%s'.",
            len(results),
            status,
            self._list_id,
        )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_status(task: dict) -> str:
        """Extract the status string from a raw ClickUp task dict."""
        status_field = task.get("status", {})
        if isinstance(status_field, dict):
            return str(status_field.get("status", ""))
        return str(status_field)

    @staticmethod
    def _normalise(task: dict) -> dict:
        """Map a raw ClickUp task dict to the normalised schema.

        Returns a dict with keys ``id``, ``title``, ``url``, ``status``.
        """
        return {
            "id": str(task.get("id", "")),
            "title": str(task.get("name", "")),
            "url": str(task.get("url", "")),
            "status": ClickUpRestClient._extract_status(task),
        }

    def get_task_comments(self, task_id: str) -> List[dict]:
        """Return all comments on ClickUp task *task_id*, ordered oldest-first.

        Calls ``GET /api/v2/task/{task_id}/comment``.

        Args:
            task_id: The ClickUp task ID.

        Returns:
            A list of normalised dicts with keys
            ``id``, ``author``, ``body``, ``created_at``.

        Raises:
            TicketFetchError: On any non-2xx response or network error.
        """
        url = f"{self._BASE_URL}/task/{task_id}/comment"
        headers = {
            "Authorization": self._api_token,
            "Content-Type": "application/json",
        }
        logger.debug("ClickUpRestClient: GET comments for task %s", task_id)
        try:
            with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
                response = client.get(url, headers=headers)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise TicketFetchError(
                f"ClickUp API returned {exc.response.status_code} fetching comments "
                f"for task '{task_id}': {exc.response.text[:200]}",
                source="clickup",
                status_code=exc.response.status_code,
            ) from exc
        except httpx.TransportError as exc:
            raise TicketFetchError(
                f"ClickUp API network error fetching comments for task '{task_id}': {exc}",
                source="clickup",
            ) from exc

        data = response.json()
        raw_comments: list = data.get("comments", [])
        results: List[dict] = []
        for c in raw_comments:
            user = c.get("user", {})
            author = user.get("username", "") if isinstance(user, dict) else str(user)
            # ClickUp returns date as milliseconds epoch string
            date_str = str(c.get("date", "0"))
            try:
                created_at = float(date_str) / 1000.0
            except (ValueError, TypeError):
                created_at = 0.0
            comment_text = c.get("comment_text", "")
            if not comment_text:
                # Rich comments store text in comment[].text
                parts = c.get("comment", [])
                comment_text = " ".join(p.get("text", "") for p in parts if isinstance(p, dict))
            results.append(
                {
                    "id": str(c.get("id", "")),
                    "author": author,
                    "body": comment_text,
                    "created_at": created_at,
                }
            )
        logger.info(
            "ClickUpRestClient: fetched %d comment(s) for task '%s'.",
            len(results),
            task_id,
        )
        return results


# ---------------------------------------------------------------------------
# JIRA REST Client
# ---------------------------------------------------------------------------


class JiraRestClient:
    """Direct JIRA REST API v3 client for ticket queries.

    Calls ``GET /rest/api/3/search`` with a JQL query constructed from the
    provided status strings.  Pagination is handled internally.

    Parameters
    ----------
    base_url:
        Root URL of the JIRA instance (e.g. ``"https://yourteam.atlassian.net"``).
        Stored as ``ATLASSIAN_URL`` in ``AppSettings``.  Must **not** include
        a trailing slash or the ``/rest/api/3`` path.
    api_token:
        Atlassian API token.  Stored as ``MCP_JIRA_TOKEN`` in ``AppSettings``.
    email:
        Email address of the Atlassian account that owns *api_token*.
        Stored as ``ATLASSIAN_EMAIL`` in ``AppSettings``.
        Used together with *api_token* to build the Basic Auth header::

            Authorization: Basic base64("{email}:{api_token}")

    Raises
    ------
    ValueError
        At construction time if any of *base_url*, *api_token*, or *email*
        is empty.
    """

    def __init__(self, base_url: str, api_token: str, email: str) -> None:
        if not base_url:
            raise ValueError("JiraRestClient: base_url must not be empty. " "Set ATLASSIAN_URL in your .env file.")
        if not api_token:
            raise ValueError("JiraRestClient: api_token must not be empty. " "Set MCP_JIRA_TOKEN in your .env file.")
        if not email:
            raise ValueError("JiraRestClient: email must not be empty. " "Set ATLASSIAN_EMAIL in your .env file.")
        self._base_url = base_url.rstrip("/")
        self._email = email
        self._api_token = api_token
        # Pre-compute the Basic Auth header value once at construction.
        raw = f"{email}:{api_token}"
        self._auth_header = "Basic " + base64.b64encode(raw.encode()).decode()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_issues(
        self,
        status: str,
        exclude_status: str,
        project_key: Optional[str] = None,
    ) -> List[dict]:
        """Return all JIRA issues matching *status* via JQL.

        Fetches all pages and returns a flat list.

        Args:
            status:         Status string to match (e.g. ``"ACCEPTED"``).
            exclude_status: Status string to exclude via JQL
                            (e.g. ``"REJECTED"``).
            project_key:    Optional JIRA project key to scope the search
                            (e.g. ``"PROJ"``).  When ``None``, searches the
                            entire workspace.

        Returns:
            A list of normalised dicts with keys
            ``id``, ``title``, ``url``, ``status``.

        Raises:
            TicketFetchError: On any non-2xx response or network error.
        """
        jql = f'status = "{status}" AND status != "{exclude_status}"'
        if project_key:
            jql += f' AND project = "{project_key}"'
        jql += " ORDER BY created ASC"

        url = f"{self._base_url}/rest/api/3/search"
        headers = {
            "Authorization": self._auth_header,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        results: List[dict] = []
        start_at = 0

        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            while True:
                params = {
                    "jql": jql,
                    "startAt": str(start_at),
                    "maxResults": str(_JIRA_PAGE_SIZE),
                    "fields": "summary,status",
                }
                logger.debug("JiraRestClient: GET %s startAt=%d jql=%r", url, start_at, jql)
                try:
                    response = client.get(url, headers=headers, params=params)
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise TicketFetchError(
                        f"JIRA API returned {exc.response.status_code}: " f"{exc.response.text[:200]}",
                        source="jira",
                        status_code=exc.response.status_code,
                    ) from exc
                except httpx.TransportError as exc:
                    raise TicketFetchError(
                        f"JIRA API network error: {exc}",
                        source="jira",
                    ) from exc

                data = response.json()
                issues: list = data.get("issues", [])
                total: int = data.get("total", 0)
                logger.debug(
                    "JiraRestClient: startAt=%d returned %d issue(s) of %d total.",
                    start_at,
                    len(issues),
                    total,
                )

                for issue in issues:
                    results.append(self._normalise(issue, self._base_url))

                start_at += len(issues)
                # Stop when we've fetched all pages.
                if start_at >= total or not issues:
                    break

        logger.info(
            "JiraRestClient: found %d issue(s) with status=%r%s.",
            len(results),
            status,
            f" in project '{project_key}'" if project_key else "",
        )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(issue: dict, base_url: str) -> dict:
        """Map a raw JIRA issue dict to the normalised schema.

        Returns a dict with keys ``id``, ``title``, ``url``, ``status``.

        The ``id`` field uses ``issue["key"]`` (e.g. ``"PROJ-42"``) rather
        than the internal numeric ``issue["id"]``, because the key is the
        human-visible identifier used everywhere else in the system.
        """
        key: str = str(issue.get("key", ""))
        fields: dict = issue.get("fields", {})
        status_field = fields.get("status", {})
        raw_status = str(status_field.get("name", "")) if isinstance(status_field, dict) else ""
        return {
            "id": key,
            "title": str(fields.get("summary", key)),
            "url": f"{base_url}/browse/{key}" if key else "",
            "status": raw_status,
        }

    def get_issue_comments(self, ticket_id: str) -> List[dict]:
        """Return all comments on JIRA issue *ticket_id*, ordered oldest-first.

        Calls ``GET /rest/api/3/issue/{ticket_id}/comment?orderBy=created``.

        Args:
            ticket_id: The JIRA issue key (e.g. ``"PROJ-42"``).

        Returns:
            A list of normalised dicts with keys
            ``id``, ``author``, ``body``, ``created_at``.

        Raises:
            TicketFetchError: On any non-2xx response or network error.
        """
        import datetime

        url = f"{self._base_url}/rest/api/3/issue/{ticket_id}/comment"
        headers = {
            "Authorization": self._auth_header,
            "Accept": "application/json",
        }
        params = {"orderBy": "created", "maxResults": "500"}
        logger.debug("JiraRestClient: GET comments for issue %s", ticket_id)
        try:
            with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
                response = client.get(url, headers=headers, params=params)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise TicketFetchError(
                f"JIRA API returned {exc.response.status_code} fetching comments "
                f"for issue '{ticket_id}': {exc.response.text[:200]}",
                source="jira",
                status_code=exc.response.status_code,
            ) from exc
        except httpx.TransportError as exc:
            raise TicketFetchError(
                f"JIRA API network error fetching comments for issue '{ticket_id}': {exc}",
                source="jira",
            ) from exc

        data = response.json()
        raw_comments: list = data.get("comments", [])
        results: List[dict] = []
        for c in raw_comments:
            author_field = c.get("author", {})
            author = author_field.get("displayName", "") if isinstance(author_field, dict) else str(author_field)
            # JIRA returns ISO-8601 datetime strings, e.g. "2024-01-15T10:30:00.000+0000"
            created_str = c.get("created", "")
            try:
                dt = datetime.datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                created_at = dt.timestamp()
            except (ValueError, AttributeError):
                created_at = 0.0
            # Body may be in Atlassian Document Format (ADF) or plain text.
            body_field = c.get("body", "")
            if isinstance(body_field, dict):
                # ADF: extract plain text from content nodes
                body = self._extract_adf_text(body_field)
            else:
                body = str(body_field)
            results.append(
                {
                    "id": str(c.get("id", "")),
                    "author": author,
                    "body": body,
                    "created_at": created_at,
                }
            )
        logger.info(
            "JiraRestClient: fetched %d comment(s) for issue '%s'.",
            len(results),
            ticket_id,
        )
        return results

    @staticmethod
    def _extract_adf_text(adf: dict) -> str:
        """Extract plain text from an Atlassian Document Format (ADF) dict.

        Recursively walks the ``content`` tree and joins all ``text`` leaf nodes.

        Args:
            adf: The ADF document dict (``{"type": "doc", "content": [...]}``)

        Returns:
            Plain-text representation of the document.
        """
        parts: List[str] = []

        def _walk(node: dict) -> None:
            if node.get("type") == "text":
                parts.append(node.get("text", ""))
            for child in node.get("content", []):
                if isinstance(child, dict):
                    _walk(child)

        _walk(adf)
        return " ".join(parts).strip()
