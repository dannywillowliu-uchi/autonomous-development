"""Code-based task discovery -- find work items by running commands."""

from __future__ import annotations

import re

from mission_control.config import MissionConfig
from mission_control.models import Session, Snapshot, TaskRecord


def discover_from_snapshot(
	snapshot: Snapshot,
	config: MissionConfig,
	recent_sessions: list[Session] | None = None,
	previous_snapshot: Snapshot | None = None,
) -> list[TaskRecord]:
	"""Discover work items from a project health snapshot.

	Priority order:
	1. Regressions (tests that previously passed now fail)
	2. Test failures
	3. Lint errors
	4. Type errors
	5. TODOs (discovered separately)
	6. Coverage gaps (discovered separately)
	7. Objective-driven work
	"""
	tasks: list[TaskRecord] = []

	# Priority 1: Regressions
	if previous_snapshot and snapshot.test_failed > previous_snapshot.test_failed:
		new_failures = snapshot.test_failed - previous_snapshot.test_failed
		tasks.append(TaskRecord(
			source="regression",
			description=f"Fix {new_failures} test regression(s) -- tests that previously passed now fail",
			priority=1,
		))

	# Priority 2: Test failures
	if snapshot.test_failed > 0:
		# Don't duplicate if already covered by regression
		if not any(t.source == "regression" for t in tasks):
			tasks.append(TaskRecord(
				source="test_failure",
				description=f"Fix {snapshot.test_failed} failing test(s)",
				priority=2,
			))

	# Priority 3: Lint errors
	if snapshot.lint_errors > 0:
		tasks.append(TaskRecord(
			source="lint",
			description=f"Fix {snapshot.lint_errors} lint error(s)",
			priority=3,
		))

	# Priority 4: Type errors
	if snapshot.type_errors > 0:
		tasks.append(TaskRecord(
			source="type_error",
			description=f"Fix {snapshot.type_errors} type error(s)",
			priority=4,
		))

	# Priority 5: Security findings
	if snapshot.security_findings > 0:
		tasks.append(TaskRecord(
			source="security",
			description=f"Address {snapshot.security_findings} security finding(s)",
			priority=5,
		))

	# Priority 7: Objective-driven (only if no higher-priority work and objective exists)
	if not tasks and config.target.objective:
		recently_attempted = set()
		if recent_sessions:
			recently_attempted = {s.task_description for s in recent_sessions[-5:]}

		objective_desc = f"Work toward objective: {config.target.objective}"
		if objective_desc not in recently_attempted:
			tasks.append(TaskRecord(
				source="objective",
				description=objective_desc,
				priority=7,
			))

	tasks.sort(key=lambda t: t.priority)
	return tasks


def discover_todos_from_output(grep_output: str) -> list[TaskRecord]:
	"""Parse grep output for TODO/FIXME items."""
	tasks: list[TaskRecord] = []
	seen: set[str] = set()

	for line in grep_output.strip().splitlines():
		if not line.strip():
			continue
		# Format: "file.py:line: ... TODO: message" or "file.py:line: ... FIXME: message"
		match = re.search(r"(TODO|FIXME):?\s*(.+)", line)
		if match:
			desc = match.group(2).strip()
			if desc not in seen:
				seen.add(desc)
				tasks.append(TaskRecord(
					source="todo",
					description=f"TODO: {desc}",
					priority=5,
				))

	return tasks
