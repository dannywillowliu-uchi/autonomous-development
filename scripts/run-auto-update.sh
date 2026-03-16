#!/usr/bin/env bash
# Intel scan: scan sources, evaluate findings, inject proposals into running swarm.
# Launched by LaunchAgent every 24 hours at 6 AM.

set -euo pipefail

export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
export TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-}"
export TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID:-}"

PROJECT_DIR="/Users/dannyliu/personal_projects/autonomous-development"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE="$LOG_DIR/auto-update-$TIMESTAMP.log"

cd "$PROJECT_DIR"

echo "=== Auto-update starting at $(date) ===" >> "$LOG_FILE"

# Run the auto-update pipeline
"$PROJECT_DIR/.venv/bin/python" -m autodev auto-update \
	>> "$LOG_FILE" 2>&1

echo "=== Auto-update finished at $(date) ===" >> "$LOG_FILE"
