# 🦾 simple-ai-worker

[![CI](https://github.com/Chisanan232/simple-ai-worker/actions/workflows/ci.yaml/badge.svg)](https://github.com/Chisanan232/simple-ai-worker/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/Chisanan232/simple-ai-worker/branch/master/graph/badge.svg)](https://codecov.io/gh/Chisanan232/simple-ai-worker)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Chisanan232_simple-ai-worker&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=Chisanan232_simple-ai-worker)
[![Docker Pulls](https://img.shields.io/docker/pulls/chisanan232/simple-ai-worker)](https://hub.docker.com/r/chisanan232/simple-ai-worker)
[![Docker Image Size](https://img.shields.io/docker/image-size/chisanan232/simple-ai-worker/latest)](https://hub.docker.com/r/chisanan232/simple-ai-worker)
[![Docker Stars](https://img.shields.io/docker/stars/chisanan232/simple-ai-worker)](https://hub.docker.com/r/chisanan232/simple-ai-worker)
[![Python Versions](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue?logo=python&logoColor=FBE072)](https://github.com/Chisanan232/simple-ai-worker)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Chisanan232/simple-ai-worker/blob/master/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)

A production-ready, AI-powered worker that runs autonomous [CrewAI](https://crewai.com) agents on a schedule and responds to Slack events.  
The image serves **two service modes** selectable via the `SERVICE_TYPE` environment variable:

| Mode      | Entry-point         | Description                                                                                                |
|-----------|---------------------|------------------------------------------------------------------------------------------------------------|
| `worker`  | `src/main.py`       | APScheduler-based AI agent runner — polls tasks and dispatches Crews                                       |
| `webhook` | `src/slack_main.py` | Slack Bolt Events API HTTP server — receives `app_mention` / `message.im` events and routes them to agents |

---

## 🐳 Quick Start

### Pull the image

```bash
docker pull chisanan232/simple-ai-worker:latest
```

### Run the AI agent scheduler

```bash
docker run -d \
  --name ai-scheduler \
  -e SERVICE_TYPE=worker \
  -e OPENAI_API_KEY=sk-... \
  -e MCP_GITHUB_TOKEN=ghp_... \
  -e MCP_CLICKUP_TOKEN=pk_... \
  -e MCP_SLACK_TOKEN=xoxp-... \
  -e MCP_JIRA_URL=http://mcp-jira:8100/mcp \
  -e MCP_CLICKUP_URL=http://mcp-clickup:8101/mcp \
  -e MCP_GITHUB_URL=http://mcp-github:8102/mcp \
  -e MCP_SLACK_URL=http://mcp-slack:8103/mcp \
  -v $(pwd)/config/agents.yaml:/app/config/agents.yaml:ro \
  chisanan232/simple-ai-worker:latest
```

### Run the Slack webhook server

```bash
docker run -d \
  --name ai-slack-webhook \
  -e SERVICE_TYPE=webhook \
  -e SLACK_BOT_TOKEN=xoxb-... \
  -e SLACK_SIGNING_SECRET=... \
  -e SLACK_PORT=3000 \
  -e OPENAI_API_KEY=sk-... \
  -e MCP_GITHUB_TOKEN=ghp_... \
  -e MCP_CLICKUP_TOKEN=pk_... \
  -e MCP_SLACK_TOKEN=xoxp-... \
  -e MCP_JIRA_URL=http://mcp-jira:8100/mcp \
  -e MCP_CLICKUP_URL=http://mcp-clickup:8101/mcp \
  -e MCP_GITHUB_URL=http://mcp-github:8102/mcp \
  -e MCP_SLACK_URL=http://mcp-slack:8103/mcp \
  -p 3000:3000 \
  -v $(pwd)/config/agents.yaml:/app/config/agents.yaml:ro \
  chisanan232/simple-ai-worker:latest
```

---

## ⚙️ Environment Variables

### Service selection

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SERVICE_TYPE` | ✅ | `worker` | `worker` → agent scheduler · `webhook` / `integrated-webhook` → Slack Bolt server |

### LLM / AI providers

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | ✅ (if using OpenAI) | — | OpenAI API key — [obtain here](https://platform.openai.com/account/api-keys) |
| `ANTHROPIC_API_KEY` | ✅ (if using Anthropic) | — | Anthropic API key — [obtain here](https://console.anthropic.com/account/keys) |
| `DEFAULT_LLM_PROVIDER` | ❌ | `openai` | `openai` or `anthropic` |

### Slack Bolt (webhook mode only)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SLACK_BOT_TOKEN` | ✅ | — | Bot User OAuth Token (`xoxb-…`) |
| `SLACK_SIGNING_SECRET` | ✅ | — | App signing secret for HMAC verification |
| `SLACK_PORT` | ❌ | `3000` | TCP port the Bolt HTTP server listens on |

### MCP server authentication tokens

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MCP_GITHUB_TOKEN` | ✅ | — | GitHub Personal Access Token — [obtain here](https://github.com/settings/tokens) |
| `MCP_CLICKUP_TOKEN` | ✅ | — | ClickUp API token — [obtain here](https://app.clickup.com/settings/apps) |
| `MCP_SLACK_TOKEN` | ✅ | — | Slack user OAuth token (`xoxp-…`) used by the Slack MCP server |
| `MCP_JIRA_TOKEN` | ❌ | — | Atlassian API token — [obtain here](https://id.atlassian.com/manage-profile/security/api-tokens) |

### MCP server base URLs

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MCP_GITHUB_URL` | ✅ | `http://127.0.0.1:8102/mcp` | GitHub MCP server endpoint |
| `MCP_CLICKUP_URL` | ✅ | `http://127.0.0.1:8101/mcp` | ClickUp MCP server endpoint |
| `MCP_SLACK_URL` | ✅ | `http://127.0.0.1:8103/mcp` | Slack MCP server endpoint |
| `MCP_JIRA_URL` | ❌ | `http://127.0.0.1:8100/mcp` | Jira MCP server endpoint |

### Atlassian / Jira (optional — only when `mcp-jira` is in use)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ATLASSIAN_URL` | ❌ | — | Your Atlassian Cloud site root, e.g. `https://your-org.atlassian.net` |
| `ATLASSIAN_EMAIL` | ❌ | — | Email of the Atlassian account that owns the API token |

### Scheduler

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SCHEDULER_INTERVAL_SECONDS` | ❌ | `60` | Polling interval for all interval-based jobs (seconds) |
| `SCHEDULER_TIMEZONE` | ❌ | `UTC` | Timezone string recognised by pytz, e.g. `Asia/Taipei` |
| `MAX_CONCURRENT_DEV_AGENTS` | ❌ | `3` | Max number of Crews that may run concurrently |

### Agent config

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AGENT_CONFIG_PATH` | ❌ | `config/agents.yaml` | Path to the YAML agent configuration file (relative to `/app`) |

### CrewAI Enterprise

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CREWAI_PLATFORM_INTEGRATION_TOKEN` | ❌ | — | CrewAI Enterprise Integration Token for native tool integrations |

---

## 📂 Volume Mounts

### Agent configuration (required)

The container reads agent definitions from `config/agents.yaml` at the path specified by `AGENT_CONFIG_PATH`.  
**Mount your local config file into the container so it is available at runtime without rebuilding the image:**

```bash
-v $(pwd)/config/agents.yaml:/app/config/agents.yaml:ro
```

Use `config/agents.example.yaml` from the [GitHub repository](https://github.com/Chisanan232/simple-ai-worker) as a starting template.

> ⚠️ **Important — Docker Compose list-merging behaviour**  
> If you use an override file alongside `docker-compose.yml`, Docker Compose **replaces** list-type keys (`volumes`, `ports`, …) rather than appending to them.  
> Always include **both** volume entries in your override file:
> ```yaml
> volumes:
>   - ./src:/app/src:ro                               # hot-reload source (dev only)
>   - ./config/agents.yaml:/app/config/agents.yaml:ro # agent config (always required)
> ```

### Source hot-reload (development only)

Mount `src/` read-only to pick up Python code changes without rebuilding the image.  
Restart the container after editing to reload the changes:

```bash
-v $(pwd)/src:/app/src:ro
```

---

## 🚀 Running the Full Stack with Docker Compose

The recommended way to run `simple-ai-worker` is with the bundled `docker-compose.yml`, which starts all four MCP servers (`mcp-github`, `mcp-slack`, `mcp-jira`, `mcp-clickup`) alongside both application services.

### 1 — Prepare configuration files

```bash
# Copy and fill in secrets
cp .env.example .env

# Copy and configure agent definitions
cp config/agents.example.yaml config/agents.yaml
```

### 2 — Start the stack

```bash
docker compose up -d
```

Docker Compose will:
1. Start all four MCP servers and wait for their health checks to pass
2. Start `ai-scheduler` (worker mode) and `ai-slack-webhook` (webhook mode) once the MCP servers are healthy

### 3 — Inspect logs

```bash
docker compose logs -f ai-scheduler
docker compose logs -f ai-slack-webhook
```

### 4 — Stop the stack

```bash
docker compose down
```

### Port conventions (local dev)

Override host port exposure is provided in `docker-compose.override.yml` (applied automatically by Docker Compose):

| Service | Container port | Host port | Health endpoint |
|---------|---------------|-----------|-----------------|
| `mcp-jira` | 8100 | 8100 | `GET /healthz` |
| `mcp-clickup` | 8101 | 8101 | `GET /health` |
| `mcp-github` | 8102 | 8102 | `GET /.well-known/oauth-protected-resource` |
| `mcp-slack` | 8103 | 8103 | `GET /sse` |
| `ai-slack-webhook` | 3000 | 3000 | — (Slack event delivery) |

---

## 📋 Jira / Atlassian Notes

`mcp-jira` uses [sooperset/mcp-atlassian](https://github.com/sooperset/mcp-atlassian) and **validates Atlassian credentials at startup** before binding its HTTP port.  
If `ATLASSIAN_URL`, `ATLASSIAN_EMAIL`, or `MCP_JIRA_TOKEN` are empty or placeholder values, the server will start but **never open port 8100** — the health check will report `Connection refused`.

To use Jira integration, set all three in `.env`:

```dotenv
ATLASSIAN_URL=https://your-org.atlassian.net
ATLASSIAN_EMAIL=your@email.com
MCP_JIRA_TOKEN=your_atlassian_api_token
```

Then change the `mcp-jira` dependency condition in `docker-compose.yml` from `service_started` back to `service_healthy`.

---

## 🔨 Building the Image Locally

```bash
docker build -t simple-ai-worker:local .
```

> **Note** — The Dockerfile uses a multi-stage build.  The final stage removes `/app/.venv/bin/uv` and `/app/.venv/bin/uvx` that `uv sync` plants inside the venv during the build stage, preventing an `Exec format error` caused by architecture-mismatched binaries on the `PATH`.  
> `UV_NO_CACHE=1` is set in the final stage to prevent `uv run` from attempting to write a package cache into the container's read-only overlay filesystem at runtime.

---

## 📖 Documentation

Full documentation, architecture diagrams, and development guides are available at the project's [documentation site](https://github.com/Chisanan232/simple-ai-worker).

## 🔨 Contributing

Contributions are welcome! See the [GitHub repository](https://github.com/Chisanan232/simple-ai-worker) for issue tracking and pull request guidelines.

## 📜 License

[MIT License](https://github.com/Chisanan232/simple-ai-worker/blob/master/LICENSE)
