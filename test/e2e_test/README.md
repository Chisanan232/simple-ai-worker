# E2E Test Suite — Developer Guide

## Overview

End-to-end (E2E) tests verify the complete path from a human trigger (Slack
message or scheduler tick) through the CrewAI agent's LLM reasoning and MCP
tool calls, to the final output (ticket update, PR creation, Slack reply).

Unlike unit and integration tests, E2E tests support four operating modes
ranging from fully-offline (no credentials) to fully-live (real services).

---

## Architecture

### Four-Mode Strategy

| Mode | Name | LLM Layer | MCP Layer | Credentials needed |
|------|------|-----------|-----------|--------------------|
| **1** | **Stub + Fake LLM** (offline) | `FakeLLM` (deterministic) | `MCPStubServer` (in-process) | None |
| **2** | **Stub + Real LLM** (default) | Real OpenAI / Anthropic | `MCPStubServer` (in-process) | LLM API key |
| **3** | **Testcontainers + Real LLM** | Real OpenAI / Anthropic | Docker Compose (auto-managed) | LLM key + all service credentials |
| **4** | **Manual Live + Real LLM** | Real OpenAI / Anthropic | Pre-started Docker Compose | All credentials |

The `mcp_urls` fixture selects the MCP tier automatically.
The `_maybe_patch_llm_factory` autouse fixture handles the LLM tier.
Both are controlled by flags in `test/e2e_test/.env.e2e`.

### Mode Selection Logic

```
E2E_USE_FAKE_LLM=true          → Mode 1 (LLM: FakeLLM)
E2E_USE_TESTCONTAINERS=true    → Mode 3 (MCP: auto Docker Compose)
E2E_MCP_*_URL set              → Mode 4 (MCP: manual Docker Compose)
(nothing set)                  → Mode 2 (default)
```

### Folder Structure

```
test/e2e_test/
├── .env.e2e.example        # Template — copy to .env.e2e and fill in secrets
├── .env.e2e                # GITIGNORED — your local secrets
├── docker-compose.e2e.yml  # Modes 3 & 4: starts MCP server containers
├── README.md               # This file
├── conftest.py             # E2E-wide fixtures (E2ESettings, mcp_urls, builders)
├── test_live_mcp_smoke.py  # Live mode infrastructure smoke tests
│
├── common/                 # Shared modules (not test files)
│   ├── e2e_settings.py     # E2ESettings Pydantic model (reads .env.e2e)
│   ├── fake_llm.py         # FakeLLM(BaseLLM) — offline deterministic LLM
│   └── mcp_stub.py         # MCPStubServer — MCP JSON-RPC 2.0 stub server
│
├── dev/                    # Dev Agent E2E tests
│   ├── conftest.py         # Layer 3: dev_agent, dev_registry fixtures
│   ├── test_dev_slack_thread_summary_clickup_e2e.py  # S3a/S3b — ClickUp ✅
│   ├── test_dev_slack_thread_summary_jira_e2e.py     # S3a/S3b — JIRA ⏸ skipped
│   ├── test_dev_agent_workflow_clickup_e2e.py        # S4–S6 — ClickUp ✅
│   ├── test_dev_agent_workflow_jira_e2e.py           # S4–S6 — JIRA ⏸ skipped
│   ├── test_pr_auto_merge_e2e.py                     # S7 — GitHub (no split) ✅
│   ├── test_pr_review_comment_e2e.py                 # S6b — GitHub (no split) ✅
│   ├── test_full_dev_lifecycle_clickup_e2e.py        # S1→COMPLETE — ClickUp ✅
│   ├── test_full_dev_lifecycle_jira_e2e.py           # S1→COMPLETE — JIRA ⏸ skipped
│   ├── test_dev_plan_and_notify_clickup_e2e.py       # S-PL-5–7 — ClickUp ✅
│   └── test_dev_plan_and_notify_jira_e2e.py          # S-PL-5–7 — JIRA ⏸ skipped
│
├── dev_lead/               # Dev Lead Agent E2E tests
│   ├── conftest.py         # Layer 3: dev_lead_agent, dev_lead_registry fixtures
│   ├── test_dev_lead_planning_lifecycle_clickup_e2e.py  # S-PL-1–8 — ClickUp ✅
│   └── test_dev_lead_planning_lifecycle_jira_e2e.py     # S-PL-1–8 — JIRA ⏸ skipped
│
└── planner/                # Planner Agent E2E tests
    ├── conftest.py         # Layer 3: planner_agent, planner_registry fixtures
    ├── test_planner_idea_discussion_clickup_e2e.py   # Idea survey — ClickUp ✅
    └── test_planner_idea_discussion_jira_e2e.py      # Idea survey — JIRA ⏸ skipped
```

### Three-Layer conftest Discovery Chain

```
test/conftest.py                     # Layer 1: env isolation
test/e2e_test/conftest.py            # Layer 2: E2E-wide fixtures
test/e2e_test/{role}/conftest.py     # Layer 3: role-specific agent fixtures
```

`common/` is a pure Python import package — no conftest lives there.

### Ticket Backend Split

Every test module that exercises ticket operations is split into two files:
- `*_clickup_e2e.py` — **Active** — runs in standard CI E2E stage
- `*_jira_e2e.py` — **Skipped** (`pytest.mark.skip`) — JIRA tooling not yet
  configured; will be activated once JIRA credentials are set up

Tests that only touch GitHub (PR auto-merge, PR review comments) are NOT split.

---

## Mode 1 — Stub + Fake LLM (Fully Offline)

No credentials, no Docker, no network calls. Fastest mode for CI pipelines
and development iteration on orchestration logic.

**Limitation:** `FakeLLM` returns plain-text final answers only — it does NOT
emit JSON tool-call objects. MCP stub call assertions
(`stub.was_called("search_issues")`) will not fire because the agent bypasses
tool dispatch. Use Mode 2 to verify tool-call sequences.

**Step 1: Set up `.env.e2e`**

```bash
cp test/e2e_test/.env.e2e.example test/e2e_test/.env.e2e
```

Edit `test/e2e_test/.env.e2e`:
```dotenv
E2E_USE_FAKE_LLM=true
```

**Step 2: Run E2E tests**

```bash
uv run pytest -m "e2e" test/e2e_test/
```

Or pass the flag inline without editing `.env.e2e`:
```bash
E2E_USE_FAKE_LLM=true uv run pytest -m "e2e" test/e2e_test/
```

---

## Mode 2 — Stub + Real LLM (Default)

Stub MCP mode requires only an LLM API key. No real services are needed.
This is the default mode and the one used in standard CI.

**Step 1: Set up `.env.e2e`**

```bash
cp test/e2e_test/.env.e2e.example test/e2e_test/.env.e2e
```

Edit `test/e2e_test/.env.e2e` and set at minimum:

```dotenv
OPENAI_API_KEY=sk-...
```

**Step 2: Run E2E tests**

```bash
# All E2E tests (all roles, ClickUp active + JIRA skipped)
uv run pytest -m "e2e" test/e2e_test/

# Specific role
uv run pytest -m "e2e" test/e2e_test/dev/
uv run pytest -m "e2e" test/e2e_test/dev_lead/
uv run pytest -m "e2e" test/e2e_test/planner/

# ClickUp only (all active)
uv run pytest -m "e2e" -k "clickup" test/e2e_test/

# JIRA only (all skipped — useful to verify collection)
uv run pytest -m "e2e" -k "jira" test/e2e_test/
```

---

## Mode 3 — Testcontainers + Real LLM (Auto-Managed Docker Compose)

pytest automatically starts, waits for, and tears down the Docker Compose
MCP stack. No manual `docker compose` commands needed.

**Prerequisites:**
- Docker daemon running locally or in CI.
- All service credentials filled in sections 4–7 of `.env.e2e`.
- `testcontainers[compose]` installed (already in `dev` dependency group):
  ```bash
  uv sync --group dev
  ```

**Step 1: Complete `.env.e2e`** with credentials + enable the flag:

```dotenv
OPENAI_API_KEY=sk-...
E2E_USE_TESTCONTAINERS=true

# Sections 4–7: all service credentials
E2E_ATLASSIAN_URL=https://your-org.atlassian.net
E2E_MCP_JIRA_TOKEN=...
# etc.
```

**Step 2: Run tests** — containers start and stop automatically:

```bash
uv run pytest -m "e2e" test/e2e_test/
```

Or inline:
```bash
E2E_USE_TESTCONTAINERS=true uv run pytest -m "e2e" test/e2e_test/
```

The `live_mcp_stack` session fixture manages the container lifecycle:
1. Calls `docker compose up --wait` (blocks until all healthchecks pass).
2. Derives mapped host/port for each container.
3. Yields the URL dict to `mcp_urls`.
4. Calls `docker compose down --volumes` on session teardown.

---

## Mode 4 — Manual Live + Real LLM (Pre-Started Docker Compose)

Classic live mode. Start containers manually, run tests, tear down manually.
Use this when you need fine-grained control over the container lifecycle
(e.g. debugging a specific MCP server).

**Step 1: Complete `.env.e2e`** with all credentials in sections 3–7.

**Step 2: Start MCP server containers**

```bash
docker compose \
  -f test/e2e_test/docker-compose.e2e.yml \
  --env-file test/e2e_test/.env.e2e \
  up -d --wait
```

**Step 3: Run tests** (mode is auto-detected from `E2E_MCP_*_URL` values)

```bash
uv run pytest -m "e2e" test/e2e_test/
```

**Step 4: Tear down**

```bash
docker compose -f test/e2e_test/docker-compose.e2e.yml down
```

### Live Mode Port Convention

| Service | Host Port | Container Port |
|---------|-----------|---------------|
| JIRA MCP | 18100 | 18100 |
| ClickUp MCP | 18101 | 18101 |
| GitHub MCP | 18102 | 18102 |
| Slack MCP | 18103 | 18103 |

Ports 18100–18103 are chosen to avoid conflicts with the main development
stack (8100–8103). Both stacks can run simultaneously.

### Infrastructure Smoke Tests

Verify the MCP containers are running and responding before running the full suite:

```bash
uv run pytest test/e2e_test/test_live_mcp_smoke.py -v
```

---

## `FakeLLM` — Using Custom Responses in Tests

For tests that need to register specific per-test responses, use the
`fake_llm` fixture (returns `None` when fake mode is off):

```python
def test_custom_response(fake_llm, monkeypatch):
    if fake_llm is None:
        pytest.skip("fake LLM not active — set E2E_USE_FAKE_LLM=true")

    fake_llm.register("search_issues", "No open tickets found.")
    monkeypatch.setattr(
        "src.agents.llm_factory.LLMFactory.build", lambda *a, **kw: fake_llm
    )
    # ... run agent code ...
    assert fake_llm.was_called()
    assert fake_llm.call_count() >= 1
```

Use the session-scoped `fake_llm_session` fixture to inspect the shared instance:

```python
def test_session_calls(fake_llm_session):
    if fake_llm_session is None:
        pytest.skip("fake LLM not active")
    assert fake_llm_session.call_count() > 0
```

---

## Activating JIRA Tests

All `*_jira_e2e.py` files are currently skipped. To activate them:

1. Configure JIRA credentials in `test/e2e_test/.env.e2e`:
   ```dotenv
   E2E_ATLASSIAN_URL=https://your-org.atlassian.net
   E2E_ATLASSIAN_EMAIL=you@example.com
   E2E_MCP_JIRA_TOKEN=your-api-token
   E2E_JIRA_PROJECT_KEY=TEST
   E2E_MCP_JIRA_URL=http://127.0.0.1:18100/mcp
   ```
2. Start the JIRA MCP container (Mode 4) or use Mode 3 with `E2E_USE_TESTCONTAINERS=true`.
3. In each `*_jira_e2e.py` file, remove `pytest.mark.skip` from `pytestmark`.

---

## Run Command Quick Reference

| Goal | Command |
|------|---------|
| All unit tests | `uv run pytest -m "unit"` |
| All integration tests | `uv run pytest -m "integration"` |
| Standard CI (no E2E) | `uv run pytest -m "unit or integration"` |
| **Mode 1** — fully offline E2E | `E2E_USE_FAKE_LLM=true uv run pytest -m "e2e" test/e2e_test/` |
| **Mode 2** — stub + real LLM (default) | `uv run pytest -m "e2e" test/e2e_test/` |
| **Mode 3** — testcontainers auto Docker Compose | `E2E_USE_TESTCONTAINERS=true uv run pytest -m "e2e" test/e2e_test/` |
| **Mode 4** — manual live (pre-started containers) | `uv run pytest -m "e2e" test/e2e_test/` (after `docker compose up`) |
| Dev E2E only | `uv run pytest -m "e2e" test/e2e_test/dev/` |
| Dev Lead E2E only | `uv run pytest -m "e2e" test/e2e_test/dev_lead/` |
| Planner E2E only | `uv run pytest -m "e2e" test/e2e_test/planner/` |
| Plan-and-notify feature | `uv run pytest -m "e2e" test/e2e_test/dev/test_dev_plan_and_notify_clickup_e2e.py` |
| ClickUp tests only | `uv run pytest -m "e2e" -k "clickup" test/e2e_test/` |
| JIRA tests only | `uv run pytest -m "e2e" -k "jira" test/e2e_test/` |
| Live mode smoke tests | `uv run pytest test/e2e_test/test_live_mcp_smoke.py` |
| Full suite incl. E2E stubs | `uv run pytest` |
| Start live mode services (Mode 4) | `docker compose -f test/e2e_test/docker-compose.e2e.yml --env-file test/e2e_test/.env.e2e up -d --wait` |
| Tear down live mode (Mode 4) | `docker compose -f test/e2e_test/docker-compose.e2e.yml down` |

---

## Credential Security

- `test/e2e_test/.env.e2e` is in `.gitignore` and must **never** be committed.
- Use **dedicated test accounts** for all live mode services (test JIRA project,
  test ClickUp list, test GitHub repo, test Slack channel).
- Never point live mode at production accounts — E2E tests create, modify, and
  potentially delete real data.

---

## Business Rules Verified

| Rule | Verification |
|------|-------------|
| BR-1: AI never writes `ACCEPTED` | Asserted in all scan/dispatch + full lifecycle tests |
| BR-2: Auto-merge requires ≥1 approval | Asserted in `dev/test_pr_auto_merge_e2e.py` |
| BR-3: REJECTED tickets silently skipped | Asserted in `dev/test_dev_agent_workflow_*_e2e.py` |
| BR-6: Ask supervisor when no ticket ID | Asserted in `dev/test_dev_slack_thread_summary_*_e2e.py` |
| BR-7: Dev Lead must not create sub-tasks prematurely | Asserted in `dev_lead/test_dev_lead_planning_lifecycle_*_e2e.py` |
| BR-8: Dev Agent must not code during planning mode | Asserted in `dev/test_dev_plan_and_notify_*_e2e.py` |
| BR-10: `IN PLANNING` never written by AI | Asserted in `dev/test_dev_plan_and_notify_*_e2e.py` |
| BR-12: No Dev Lead mention on reject path | Asserted in `planner/test_planner_idea_discussion_*_e2e.py` |
| BR-13: Dev Lead hand-off via `send_message` (not reply) | Asserted in `planner/test_planner_idea_discussion_*_e2e.py` |
| No AI self-approval of PRs | Asserted in `dev/test_pr_review_comment_e2e.py` |

---

## Test Count Summary

| Role | Active Tests | Skipped (JIRA) | Total Collected |
|------|-------------|----------------|-----------------|
| Dev | 30 | 26 | 56 |
| Dev Lead | 6 | 6 | 12 |
| Planner | 5 | 5 | 10 |
| Infrastructure | 6 | 0 | 6 |
| **Total** | **47** | **37** | **84** |

> Active tests run automatically in stub mode with only an LLM API key (Mode 2),
> or fully offline with no credentials (Mode 1).
> Skipped (JIRA) tests are collected by pytest and visible in `--collect-only`,
> ensuring they are discoverable even before JIRA credentials are configured.
