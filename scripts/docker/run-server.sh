#!/bin/bash
set -e

#
# Entry-point router for the simple-ai-worker Docker image.
#
# Selects which application process to launch based on the SERVICE_TYPE
# environment variable.  Called by the Dockerfile CMD.
#
# Environment variables:
#
#   SERVICE_TYPE (required) — determines which process to start:
#     "worker"             → AI agent scheduler           (src/main.py)
#                            Entry-point: uv run simple-ai-worker
#     "webhook"            → Slack Events API HTTP server  (src/slack_main.py)
#                            Entry-point: uv run simple-ai-slack
#     "integrated-webhook" → Alias for "webhook" (legacy compatibility)
#
# All other environment variables (API keys, ports, MCP URLs, etc.) are
# passed through transparently — see .env.example for the full list.
#
# Example usage:
#
#   # Run the AI agent scheduler
#   SERVICE_TYPE=worker docker run simple-ai-worker
#
#   # Run the Slack webhook server
#   SERVICE_TYPE=webhook docker run -p 3000:3000 simple-ai-worker
#
#   # docker compose (SERVICE_TYPE set in docker-compose.yml)
#   docker compose up -d
#

# Default to worker if SERVICE_TYPE is not set
SERVICE_TYPE=${SERVICE_TYPE:-worker}

# Directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "SERVICE_TYPE is set to: ${SERVICE_TYPE}"

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "${SERVICE_TYPE}" in
  worker)
    echo "[run-server.sh] Starting AI agent scheduler (simple-ai-worker) ..."
    uv run simple-ai-worker
    ;;
  webhook | integrated-webhook)
    echo "[run-server.sh] Starting Slack Events API webhook server (simple-ai-slack) ..."
    uv run simple-ai-slack
    ;;
  *)
    echo "[run-server.sh] ERROR: Unknown SERVICE_TYPE='${SERVICE_TYPE}'." >&2
    echo "[run-server.sh]   Allowed values: worker | webhook | integrated-webhook" >&2
    exit 1
    ;;
esac
