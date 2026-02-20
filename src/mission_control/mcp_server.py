"""MCP server for programmatic mission-control from Claude Code."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from mission_control.db import Database
from mission_control.launcher import MissionLauncher
from mission_control.registry import ProjectRegistry

logger = logging.getLogger(__name__)

server = Server("mission-control")


def _get_registry() -> ProjectRegistry:
	return ProjectRegistry()


def _get_launcher(registry: ProjectRegistry) -> MissionLauncher:
	return MissionLauncher(registry)


# -- Tool definitions --

TOOLS = [
	Tool(
		name="list_projects",
		description="List all registered projects with their latest mission status.",
		inputSchema={
			"type": "object",
			"properties": {},
		},
	),
	Tool(
		name="get_project_status",
		description="Get detailed mission state for a project (rounds, units, scores).",
		inputSchema={
			"type": "object",
			"properties": {
				"project_name": {"type": "string", "description": "Name of the registered project"},
			},
			"required": ["project_name"],
		},
	),
	Tool(
		name="launch_mission",
		description="Launch a new mission for a registered project.",
		inputSchema={
			"type": "object",
			"properties": {
				"project_name": {"type": "string", "description": "Name of the registered project"},
				"max_rounds": {"type": "integer", "description": "Maximum rounds to run"},
				"num_workers": {"type": "integer", "description": "Number of parallel workers"},
			},
			"required": ["project_name"],
		},
	),
	Tool(
		name="stop_mission",
		description="Stop a running mission by sending a stop signal.",
		inputSchema={
			"type": "object",
			"properties": {
				"project_name": {"type": "string", "description": "Name of the registered project"},
			},
			"required": ["project_name"],
		},
	),
	Tool(
		name="retry_unit",
		description="Retry a failed work unit within the current mission.",
		inputSchema={
			"type": "object",
			"properties": {
				"project_name": {"type": "string", "description": "Name of the registered project"},
				"unit_id": {"type": "string", "description": "ID of the work unit to retry"},
			},
			"required": ["project_name", "unit_id"],
		},
	),
	Tool(
		name="adjust_mission",
		description="Adjust runtime parameters of a running mission.",
		inputSchema={
			"type": "object",
			"properties": {
				"project_name": {"type": "string", "description": "Name of the registered project"},
				"max_rounds": {"type": "integer", "description": "New max rounds value"},
				"num_workers": {"type": "integer", "description": "New worker count"},
			},
			"required": ["project_name"],
		},
	),
	Tool(
		name="register_project",
		description="Register a project in the central registry.",
		inputSchema={
			"type": "object",
			"properties": {
				"config_path": {"type": "string", "description": "Absolute path to mission-control.toml"},
				"name": {"type": "string", "description": "Project alias (optional, defaults to target.name)"},
				"description": {"type": "string", "description": "Project description"},
			},
			"required": ["config_path"],
		},
	),
	Tool(
		name="get_round_details",
		description="Get work units and handoffs for a specific round.",
		inputSchema={
			"type": "object",
			"properties": {
				"project_name": {"type": "string", "description": "Name of the registered project"},
				"round_id": {"type": "string", "description": "Round ID (optional, defaults to latest)"},
			},
			"required": ["project_name"],
		},
	),
]


@server.list_tools()
async def list_tools() -> list[Tool]:
	return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
	registry = _get_registry()
	try:
		result = _dispatch(name, arguments, registry)
		return [TextContent(type="text", text=json.dumps(result, indent=2))]
	except Exception as e:
		return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
	finally:
		registry.close()


def _dispatch(name: str, args: dict, registry: ProjectRegistry) -> dict:
	launcher = _get_launcher(registry)

	if name == "list_projects":
		return _tool_list_projects(registry)
	elif name == "get_project_status":
		return _tool_get_project_status(registry, args["project_name"])
	elif name == "launch_mission":
		return _tool_launch_mission(launcher, args)
	elif name == "stop_mission":
		return _tool_stop_mission(launcher, args["project_name"])
	elif name == "retry_unit":
		return _tool_retry_unit(launcher, args["project_name"], args["unit_id"])
	elif name == "adjust_mission":
		return _tool_adjust_mission(launcher, args)
	elif name == "register_project":
		return _tool_register_project(registry, args)
	elif name == "get_round_details":
		return _tool_get_round_details(registry, args)
	else:
		return {"error": f"Unknown tool: {name}"}


def _tool_list_projects(registry: ProjectRegistry) -> dict:
	projects = []
	for p in registry.list_projects():
		status = registry.get_project_status(p.name)
		projects.append({
			"name": p.name,
			"config_path": p.config_path,
			"description": p.description,
			"active_pid": p.active_pid,
			"mission_status": status.mission_status if status else "idle",
			"mission_score": status.mission_score if status else 0.0,
			"mission_rounds": status.mission_rounds if status else 0,
			"mission_cost": status.mission_cost if status else 0.0,
		})
	return {"projects": projects, "count": len(projects)}


def _tool_get_project_status(registry: ProjectRegistry, project_name: str) -> dict:
	status = registry.get_project_status(project_name)
	if status is None:
		return {"error": f"Project '{project_name}' not found"}

	result = {
		"name": status.project.name,
		"config_path": status.project.config_path,
		"mission_status": status.mission_status,
		"mission_objective": status.mission_objective,
		"mission_score": status.mission_score,
		"mission_rounds": status.mission_rounds,
		"mission_cost": status.mission_cost,
		"active_pid": status.project.active_pid,
	}

	# Enrich with round details if DB exists
	db_path = Path(status.project.db_path)
	if db_path.exists():
		try:
			db = Database(db_path)
			try:
				mission = db.get_latest_mission()
				if mission:
					rounds = db.get_rounds_for_mission(mission.id)
					result["rounds"] = [
						{
							"number": r.number,
							"status": r.status,
							"score": r.objective_score,
							"units_total": r.total_units,
							"units_completed": r.completed_units,
							"units_failed": r.failed_units,
							"cost": r.cost_usd,
						}
						for r in rounds
					]
					result["score_history"] = [r.objective_score for r in rounds]
			finally:
				db.close()
		except Exception:
			pass

	return result


def _tool_launch_mission(launcher: MissionLauncher, args: dict) -> dict:
	project_name = args["project_name"]
	overrides: dict[str, str] = {}
	if "max_rounds" in args:
		overrides["max_rounds"] = str(args["max_rounds"])
	if "num_workers" in args:
		overrides["workers"] = str(args["num_workers"])

	try:
		pid = launcher.launch(project_name, config_overrides=overrides)
		return {"status": "launched", "project": project_name, "pid": pid}
	except (ValueError, RuntimeError, FileNotFoundError) as e:
		return {"error": str(e)}


def _tool_stop_mission(launcher: MissionLauncher, project_name: str) -> dict:
	try:
		result = launcher.stop(project_name)
		if result:
			return {"status": "stop_signal_sent", "project": project_name}
		return {"status": "no_running_mission", "project": project_name}
	except ValueError as e:
		return {"error": str(e)}


def _tool_retry_unit(launcher: MissionLauncher, project_name: str, unit_id: str) -> dict:
	try:
		result = launcher.retry_unit(project_name, unit_id)
		if result:
			return {"status": "retry_signal_sent", "project": project_name, "unit_id": unit_id}
		return {"status": "could_not_retry", "project": project_name, "unit_id": unit_id}
	except ValueError as e:
		return {"error": str(e)}


def _tool_adjust_mission(launcher: MissionLauncher, args: dict) -> dict:
	project_name = args["project_name"]
	params: dict[str, int | float | str] = {}
	if "max_rounds" in args:
		params["max_rounds"] = args["max_rounds"]
	if "num_workers" in args:
		params["num_workers"] = args["num_workers"]

	if not params:
		return {"error": "No parameters to adjust"}

	try:
		result = launcher.adjust(project_name, params)
		if result:
			return {"status": "adjusted", "project": project_name, "params": params}
		return {"status": "no_running_mission", "project": project_name}
	except ValueError as e:
		return {"error": str(e)}


def _tool_register_project(registry: ProjectRegistry, args: dict) -> dict:
	config_path = args["config_path"]
	name = args.get("name")
	description = args.get("description", "")

	if not name:
		try:
			from mission_control.config import load_config
			config = load_config(config_path)
			name = config.target.name
		except Exception:
			name = Path(config_path).parent.name

	try:
		project = registry.register(
			name=name,
			config_path=config_path,
			description=description,
		)
		return {"status": "registered", "name": project.name, "config_path": project.config_path}
	except ValueError as e:
		return {"error": str(e)}


def _tool_get_round_details(registry: ProjectRegistry, args: dict) -> dict:
	project_name = args["project_name"]
	round_id = args.get("round_id")

	project = registry.get_project(project_name)
	if project is None:
		return {"error": f"Project '{project_name}' not found"}

	db_path = Path(project.db_path)
	if not db_path.exists():
		return {"error": "No database found for project"}

	db = Database(db_path)
	try:
		mission = db.get_latest_mission()
		if not mission:
			return {"error": "No mission found"}

		if round_id:
			rnd = db.get_round(round_id)
		else:
			rounds = db.get_rounds_for_mission(mission.id)
			rnd = rounds[-1] if rounds else None

		if not rnd:
			return {"error": "No round found"}

		# Get work units
		units = []
		if rnd.plan_id:
			work_units = db.get_work_units_for_plan(rnd.plan_id)
			units = [
				{
					"id": u.id,
					"title": u.title,
					"status": u.status,
					"output_summary": u.output_summary[:200] if u.output_summary else "",
					"attempt": u.attempt,
					"worker_id": u.worker_id,
				}
				for u in work_units
			]

		# Get handoffs
		handoffs = db.get_handoffs_for_round(rnd.id)
		handoff_data = [
			{
				"work_unit_id": h.work_unit_id,
				"status": h.status,
				"summary": h.summary[:200] if h.summary else "",
				"discoveries_count": len(h.discoveries),
			}
			for h in handoffs
		]

		return {
			"round_id": rnd.id,
			"number": rnd.number,
			"status": rnd.status,
			"score": rnd.objective_score,
			"units": units,
			"handoffs": handoff_data,
		}
	finally:
		db.close()


def run_mcp_server() -> None:
	"""Entry point for `mc mcp` CLI command."""
	import asyncio

	async def _run():
		async with stdio_server() as (read_stream, write_stream):
			await server.run(read_stream, write_stream, server.create_initialization_options())

	asyncio.run(_run())
