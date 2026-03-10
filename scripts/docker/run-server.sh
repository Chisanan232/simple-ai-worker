#!/bin/bash
set -e

#
# This script is a router that runs either the MCP server or Slack webhook server
# based on the SERVICE_TYPE environment variable.
#
# Environment variables:
#
# SERVICE_TYPE → Determines which service to run
#    - "mcp": Runs the MCP server (run-slack-mcp-server.sh)
#    - "webhook": Runs the Slack webhook server (run-slack-webhook-server.sh)
#    - "integrated": Runs either server in integrated mode
#
# For all other environment variables, see the respective server scripts:
# - run-slack-mcp-server.sh
# - run-slack-webhook-server.sh
#
# Example usage:
# # Run MCP server
# SERVICE_TYPE=mcp ./run-server.sh
#
## Run webhook server
# SERVICE_TYPE=webhook ./run-server.sh
#
## Run integrated server via MCP entry point
# SERVICE_TYPE=integrated ./run-server.sh
#
## Run integrated server via webhook entry point
# SERVICE_TYPE=integrated-webhook ./run-server.sh
#

# Default to MCP server if SERVICE_TYPE is not set
SERVICE_TYPE=${SERVICE_TYPE:-mcp}

# Directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Print informational message
echo "SERVICE_TYPE is set to: ${SERVICE_TYPE}"

# Determine which server to run based on SERVICE_TYPE
# Your own docker script to run or set up something
