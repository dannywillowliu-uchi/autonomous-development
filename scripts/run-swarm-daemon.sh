#!/usr/bin/env bash
# Persistent swarm daemon. Runs 24/7, idles when tasks complete,
# picks up new directives from auto-update via swarm-inject.

set -euo pipefail

export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
export TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-}"
export TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID:-}"

PROJECT_DIR="/Users/dannyliu/personal_projects/autonomous-development"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

cd "$PROJECT_DIR"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE="$LOG_DIR/swarm-daemon-$TIMESTAMP.log"

echo "=== Swarm daemon starting at $(date) ===" >> "$LOG_FILE"

"$PROJECT_DIR/.venv/bin/python" -m autodev swarm --no-dashboard \
	>> "$LOG_FILE" 2>&1

echo "=== Swarm daemon exited at $(date) ===" >> "$LOG_FILE"
