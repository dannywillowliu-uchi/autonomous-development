#!/usr/bin/env bash
# Self-recursion loop: scan intel sources, generate specs, apply improvements
# Launched by LaunchAgent every 24 hours

set -euo pipefail

export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

PROJECT_DIR="/Users/dannyliu/personal_projects/autonomous-development"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE="$LOG_DIR/auto-update-$TIMESTAMP.log"

cd "$PROJECT_DIR"

echo "=== Auto-update starting at $(date) ===" >> "$LOG_FILE"

# Run the auto-update pipeline (single cycle, not daemon mode)
"$PROJECT_DIR/.venv/bin/python" -m autodev auto-update \
	>> "$LOG_FILE" 2>&1

echo "=== Auto-update finished at $(date) ===" >> "$LOG_FILE"
