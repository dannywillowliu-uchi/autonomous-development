#!/usr/bin/env bash
# Scheduled mission run for autonomous-dev-scheduler
# Launched by LaunchAgent at 4:00 AM CST

set -euo pipefail

PROJECT_DIR="/Users/dannyliu/personal_projects/autonomous-dev-scheduler"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE="$LOG_DIR/mission-$TIMESTAMP.log"

cd "$PROJECT_DIR"

echo "=== Mission starting at $(date) ===" >> "$LOG_FILE"
echo "Objective: web research capabilities for frontier lab research" >> "$LOG_FILE"

# Run the mission, logging stdout+stderr
"$PROJECT_DIR/.venv/bin/python" -m mission_control.cli mission \
	--config "$PROJECT_DIR/mission-control.toml" \
	>> "$LOG_FILE" 2>&1

echo "=== Mission finished at $(date) ===" >> "$LOG_FILE"

# Self-cleanup: unload the LaunchAgent after run (one-shot)
launchctl bootout "gui/$(id -u)/com.dannyliu.mc-mission-ads" 2>/dev/null || true
rm -f "$HOME/Library/LaunchAgents/com.dannyliu.mc-mission-ads.plist"
