"""TOML configuration loader for mission-control."""

from __future__ import annotations

import os
import re
import shutil
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VerificationNodeConfig:
	"""Configuration for a single verification node."""

	kind: str = "custom"  # maps to VerificationNodeKind
	command: str = ""
	weight: float = 1.0
	timeout: int = 300
	required: bool = True


@dataclass
class VerificationConfig:
	"""Target project verification settings."""

	command: str = "pytest -q"
	timeout: int = 300
	setup_command: str = ""
	setup_timeout: int = 120
	nodes: list[VerificationNodeConfig] = field(default_factory=list)


@dataclass
class TargetConfig:
	"""Target project configuration."""

	name: str = ""
	path: str = ""
	branch: str = "main"
	objective: str = ""
	verification: VerificationConfig = field(default_factory=VerificationConfig)

	@property
	def resolved_path(self) -> Path:
		return Path(os.path.expanduser(self.path))


@dataclass
class GitConfig:
	"""Git strategy settings."""

	strategy: str = "branch-per-session"
	auto_merge: bool = False


@dataclass
class BudgetConfig:
	"""Cost budget settings."""

	max_per_session_usd: float = 5.0
	max_per_run_usd: float = 50.0
	evaluator_budget_usd: float = 0.50
	fixup_budget_usd: float = 2.0
	ema_alpha: float = 0.3
	outlier_multiplier: float = 3.0
	conservatism_base: float = 0.5


@dataclass
class ParallelConfig:
	"""Parallel execution settings."""

	num_workers: int = 4
	pool_dir: str = ""
	heartbeat_timeout: int = 600
	max_rebase_attempts: int = 3
	warm_clones: int = 0


@dataclass
class SchedulerConfig:
	"""Scheduler settings."""

	session_timeout: int = 2700
	cooldown: int = 60
	max_sessions_per_run: int = 10
	model: str = "opus"
	monitor_interval: int = 5
	output_summary_max_chars: int = 500
	polling_interval: int = 5  # seconds between status checks during execution
	raw_output_max_chars: int = 4000  # truncation limit for raw verification output
	session_lookback: int = 5  # recent sessions to check for dedup in discovery
	git: GitConfig = field(default_factory=GitConfig)
	budget: BudgetConfig = field(default_factory=BudgetConfig)
	parallel: ParallelConfig = field(default_factory=ParallelConfig)


@dataclass
class RoundsConfig:
	"""Outer loop round settings for mission mode."""

	max_rounds: int = 20
	stall_threshold: int = 3  # rounds with no improvement before stopping
	cooldown_between_rounds: int = 30  # seconds
	stall_score_epsilon: float = 0.01  # score delta below which rounds count as stalled
	max_discoveries_per_round: int = 20
	max_discovery_chars: int = 4000
	max_summary_items: int = 10
	timeout_multiplier: float = 1.2  # applied to session_timeout for outer deadline


@dataclass
class PlannerConfig:
	"""Recursive planner settings."""

	max_depth: int = 3  # max recursion depth (hard capped at absolute_max_depth)
	absolute_max_depth: int = 4  # safety cap to prevent runaway recursion
	max_children_per_node: int = 5
	budget_per_call_usd: float = 1.0
	max_file_tree_chars: int = 2000


@dataclass
class ContinuousConfig:
	"""Continuous (event-driven) mission mode settings."""

	max_wall_time_seconds: int = 7200  # 2 hour default
	stall_threshold_units: int = 10  # N units with no improvement -> stop
	stall_score_epsilon: float = 0.01
	replan_interval_units: int = 5  # replan after N completions
	verify_before_merge: bool = True
	backlog_min_size: int = 2  # replan when backlog drops below this
	cooldown_between_units: int = 0
	timeout_multiplier: float = 1.2  # applied to session_timeout for poll deadline
	retry_base_delay_seconds: int = 30
	retry_max_delay_seconds: int = 300
	chain_max_depth: int = 3
	max_consecutive_failures: int = 3
	failure_backoff_seconds: int = 60
	min_ambition_score: int = 4  # replan if ambition below this
	max_replan_attempts: int = 2  # max replans before proceeding anyway
	verify_objective_completion: bool = False  # LLM check before declaring mission done
	max_objective_checks: int = 2  # max verification attempts before accepting
	cleanup_enabled: bool = True  # run periodic cleanup missions
	cleanup_interval: int = 3  # run cleanup mission every N missions
	backlog_max_age_seconds: int = 1800
	circuit_breaker_enabled: bool = True
	circuit_breaker_max_failures: int = 3
	circuit_breaker_cooldown_seconds: int = 120


@dataclass
class DiscoveryConfig:
	"""Auto-discovery settings."""

	enabled: bool = True
	tracks: list[str] = field(default_factory=lambda: ["feature", "quality", "security"])
	max_items_per_track: int = 3
	min_priority_score: float = 3.0
	model: str = "opus"
	budget_per_call_usd: float = 2.0
	research_enabled: bool = True
	research_model: str = "sonnet"
	research_parallel_queries: int = 3
	effort_weight: float = 0.3  # how much effort penalizes priority (0=ignore, 1=old formula)


@dataclass
class GreenBranchConfig:
	"""Green branch pattern settings."""

	working_branch: str = "mc/working"
	green_branch: str = "mc/green"
	fixup_max_attempts: int = 3
	fixup_candidates: int = 3
	reset_on_init: bool = True
	auto_push: bool = False
	push_branch: str = "main"


@dataclass
class SSHHostConfig:
	"""A single SSH host for distributed workers."""

	hostname: str = ""
	user: str = ""
	max_workers: int = 4


@dataclass
class PricingConfig:
	"""Token pricing for cost estimation ($/MTok)."""

	input_per_million: float = 3.0  # Sonnet default
	output_per_million: float = 15.0
	cache_write_per_million: float = 3.75
	cache_read_per_million: float = 0.30


@dataclass
class ContainerConfig:
	"""Docker container backend settings."""

	image: str = "mission-control-worker:latest"
	docker_executable: str = "docker"
	workspace_mount: str = "/workspace"
	claude_config_dir: str = ""  # host path, fallback to CLAUDE_CONFIG_DIR env
	extra_volumes: list[str] = field(default_factory=list)
	cap_drop: list[str] = field(default_factory=lambda: ["ALL"])
	security_opt: list[str] = field(default_factory=lambda: ["no-new-privileges:true"])
	network: str = "bridge"
	run_as_user: str = "10000:10000"
	startup_timeout: int = 60


@dataclass
class BackendConfig:
	"""Worker backend settings."""

	type: str = "local"  # local/ssh/container
	max_output_mb: int = 50  # max stdout size per worker in MB
	ssh_hosts: list[SSHHostConfig] = field(default_factory=list)
	container: ContainerConfig = field(default_factory=ContainerConfig)


@dataclass
class HeartbeatConfig:
	"""Time-based progress monitoring settings."""

	interval: int = 300  # seconds between checks
	idle_threshold: int = 3  # consecutive idle checks before stall
	enable_recovery: bool = True


@dataclass
class TelegramConfig:
	"""Telegram notification settings."""

	bot_token: str = ""
	chat_id: str = ""
	on_heartbeat: bool = True
	on_merge_fail: bool = True
	on_mission_end: bool = True


@dataclass
class NotificationConfig:
	"""Notification settings."""

	telegram: TelegramConfig = field(default_factory=TelegramConfig)


@dataclass
class DashboardConfig:
	"""Live dashboard settings."""

	host: str = "127.0.0.1"
	port: int = 8080


@dataclass
class DeployConfig:
	"""Deployment settings."""

	enabled: bool = False
	command: str = ""
	health_check_url: str = ""
	health_check_timeout: int = 60
	timeout: int = 300
	on_auto_push: bool = False
	on_mission_end: bool = True


@dataclass
class ReviewConfig:
	"""LLM diff review settings."""

	enabled: bool = True
	model: str = "sonnet"
	budget_per_review_usd: float = 0.10
	gate_completion: bool = False  # when True, review blocks unit completion
	min_review_score: float = 0.0  # minimum avg_score to pass gating


@dataclass
class ModelsConfig:
	"""Per-component model overrides."""

	planner_model: str = "opus"
	worker_model: str = "opus"
	fixup_model: str = "opus"
	architect_editor_mode: bool = False


@dataclass
class SpecialistConfig:
	"""Specialist template settings."""

	templates_dir: str = str(Path(__file__).parent / "specialist_templates")


@dataclass
class ToolSynthesisConfig:
	"""Runtime tool synthesis settings."""

	enabled: bool = False
	tools_dir: str = ".mc-tools"


@dataclass
class TracingConfig:
	"""OpenTelemetry tracing settings."""

	enabled: bool = False
	service_name: str = "mission-control"
	exporter: str = "console"  # console | otlp | none
	otlp_endpoint: str = "http://localhost:4317"


@dataclass
class HITLGateConfig:
	"""Configuration for a single HITL approval gate."""

	enabled: bool = False
	timeout_seconds: int = 300
	timeout_action: str = "approve"  # approve | deny
	large_merge_threshold_lines: int = 500
	large_merge_threshold_files: int = 20


@dataclass
class HITLConfig:
	"""Human-in-the-Loop interrupt gate settings."""

	approvals_dir: str = ".mc-approvals"
	telegram_poll_interval: float = 5.0
	push_gate: HITLGateConfig = field(default_factory=HITLGateConfig)
	large_merge_gate: HITLGateConfig = field(default_factory=HITLGateConfig)


@dataclass
class A2AConfig:
	"""Agent-to-Agent protocol settings."""

	enabled: bool = False
	host: str = "0.0.0.0"
	port: int = 8420
	agent_name: str = "mission-control"
	agent_version: str = "0.1.0"
	agent_capabilities: list[str] = field(default_factory=lambda: ["code_edit", "test_write", "refactor"])


@dataclass
class PromptEvolutionConfig:
	"""Prompt A/B testing with UCB1 multi-armed bandit."""

	enabled: bool = False
	mutation_model: str = "sonnet"
	exploration_factor: float = 1.4
	min_samples_before_mutation: int = 5


@dataclass
class EpisodicMemoryConfig:
	"""Episodic-to-semantic memory system settings."""

	enabled: bool = False
	default_ttl_days: int = 30
	decay_alpha: float = 0.1
	access_boost_days: int = 5
	distill_model: str = "sonnet"
	distill_budget_usd: float = 0.30
	min_episodes_for_distill: int = 3
	max_semantic_per_query: int = 5


@dataclass
class SpeculationConfig:
	"""Speculation branching settings for high-uncertainty work units."""

	enabled: bool = False
	uncertainty_threshold: float = 0.7
	branch_count: int = 2
	selection_metric: str = "review_score"  # review_score | cost | status
	cost_limit_multiplier: float = 1.5


@dataclass
class MCPRegistryConfig:
	"""MCP tool registry settings."""

	enabled: bool = False
	promotion_threshold: float = 0.7
	ema_alpha: float = 0.3


@dataclass
class DegradationConfig:
	"""Tiered degradation settings."""

	budget_fraction_threshold: float = 0.8
	conflict_rate_threshold: float = 0.5
	reduced_worker_fraction: float = 0.5
	db_error_threshold: int = 5
	rate_limit_window_seconds: float = 60.0
	rate_limit_threshold: int = 3
	verification_failure_threshold: int = 3
	safe_stop_timeout_seconds: float = 300.0
	recovery_success_threshold: int = 3


@dataclass
class ZFCConfig:
	"""ZFC migration settings -- LLM-backed replacements for hardcoded heuristics."""

	zfc_ambition_scoring: bool = False
	zfc_fixup_prompts: bool = False
	zfc_propose_objective: bool = False
	llm_timeout: int = 60
	llm_budget_usd: float = 0.10
	model: str = ""  # empty = inherit scheduler.model


@dataclass
class SecurityConfig:
	"""Security settings for worker subprocess isolation."""

	extra_env_keys: list[str] = field(default_factory=list)


@dataclass
class MissionConfig:
	"""Top-level mission-control configuration."""

	target: TargetConfig = field(default_factory=TargetConfig)
	scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
	review: ReviewConfig = field(default_factory=ReviewConfig)
	rounds: RoundsConfig = field(default_factory=RoundsConfig)
	continuous: ContinuousConfig = field(default_factory=ContinuousConfig)
	planner: PlannerConfig = field(default_factory=PlannerConfig)
	green_branch: GreenBranchConfig = field(default_factory=GreenBranchConfig)
	backend: BackendConfig = field(default_factory=BackendConfig)
	pricing: PricingConfig = field(default_factory=PricingConfig)
	discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
	heartbeat: HeartbeatConfig = field(default_factory=HeartbeatConfig)
	notifications: NotificationConfig = field(default_factory=NotificationConfig)
	dashboard: DashboardConfig = field(default_factory=DashboardConfig)
	deploy: DeployConfig = field(default_factory=DeployConfig)
	models: ModelsConfig = field(default_factory=ModelsConfig)
	specialist: SpecialistConfig = field(default_factory=SpecialistConfig)
	tool_synthesis: ToolSynthesisConfig = field(default_factory=ToolSynthesisConfig)
	security: SecurityConfig = field(default_factory=SecurityConfig)
	tracing: TracingConfig = field(default_factory=TracingConfig)
	hitl: HITLConfig = field(default_factory=HITLConfig)
	zfc: ZFCConfig = field(default_factory=ZFCConfig)
	degradation: DegradationConfig = field(default_factory=DegradationConfig)
	a2a: A2AConfig = field(default_factory=A2AConfig)
	mcp_registry: MCPRegistryConfig = field(default_factory=MCPRegistryConfig)
	prompt_evolution: PromptEvolutionConfig = field(default_factory=PromptEvolutionConfig)
	episodic_memory: EpisodicMemoryConfig = field(default_factory=EpisodicMemoryConfig)
	speculation: SpeculationConfig = field(default_factory=SpeculationConfig)


def _build_dashboard(data: dict[str, Any]) -> DashboardConfig:
	dc = DashboardConfig()
	if "host" in data:
		dc.host = str(data["host"])
	if "port" in data:
		dc.port = int(data["port"])
	return dc


def _build_verification_nodes(data: list[dict[str, Any]]) -> list[VerificationNodeConfig]:
	nodes: list[VerificationNodeConfig] = []
	for item in data:
		nc = VerificationNodeConfig()
		if "kind" in item:
			nc.kind = str(item["kind"])
		if "command" in item:
			nc.command = str(item["command"])
		if "weight" in item:
			nc.weight = float(item["weight"])
		if "timeout" in item:
			nc.timeout = int(item["timeout"])
		if "required" in item:
			nc.required = bool(item["required"])
		nodes.append(nc)
	return nodes


def _build_verification(data: dict[str, Any]) -> VerificationConfig:
	vc = VerificationConfig()
	if "command" in data:
		vc.command = str(data["command"])
	if "timeout" in data:
		vc.timeout = int(data["timeout"])
	if "setup_command" in data:
		vc.setup_command = str(data["setup_command"])
	if "setup_timeout" in data:
		vc.setup_timeout = int(data["setup_timeout"])
	if "nodes" in data:
		vc.nodes = _build_verification_nodes(data["nodes"])
	return vc


def _build_target(data: dict[str, Any]) -> TargetConfig:
	tc = TargetConfig()
	for key in ("name", "path", "branch", "objective"):
		if key in data:
			setattr(tc, key, str(data[key]))
	if "verification" in data:
		tc.verification = _build_verification(data["verification"])
	return tc


def _build_git(data: dict[str, Any]) -> GitConfig:
	gc = GitConfig()
	if "strategy" in data:
		gc.strategy = str(data["strategy"])
	if "auto_merge" in data:
		gc.auto_merge = bool(data["auto_merge"])
	return gc


def _build_budget(data: dict[str, Any]) -> BudgetConfig:
	bc = BudgetConfig()
	for key in (
		"max_per_session_usd", "max_per_run_usd", "evaluator_budget_usd", "fixup_budget_usd",
		"ema_alpha", "outlier_multiplier", "conservatism_base",
	):
		if key in data:
			setattr(bc, key, float(data[key]))
	return bc


def _build_parallel(data: dict[str, Any]) -> ParallelConfig:
	pc = ParallelConfig()
	for key in ("num_workers", "heartbeat_timeout", "max_rebase_attempts", "warm_clones"):
		if key in data:
			setattr(pc, key, int(data[key]))
	if "pool_dir" in data:
		pc.pool_dir = str(data["pool_dir"])
	return pc


def _build_scheduler(data: dict[str, Any]) -> SchedulerConfig:
	sc = SchedulerConfig()
	for key in (
		"session_timeout", "cooldown", "max_sessions_per_run", "monitor_interval",
		"output_summary_max_chars", "polling_interval", "raw_output_max_chars", "session_lookback",
	):
		if key in data:
			setattr(sc, key, int(data[key]))
	if "model" in data:
		sc.model = str(data["model"])
	if "git" in data:
		sc.git = _build_git(data["git"])
	if "budget" in data:
		sc.budget = _build_budget(data["budget"])
	if "parallel" in data:
		sc.parallel = _build_parallel(data["parallel"])
	return sc


def _build_rounds(data: dict[str, Any]) -> RoundsConfig:
	rc = RoundsConfig()
	for key in (
		"max_rounds", "stall_threshold", "cooldown_between_rounds",
		"max_discoveries_per_round", "max_discovery_chars", "max_summary_items",
	):
		if key in data:
			setattr(rc, key, int(data[key]))
	for key in ("stall_score_epsilon", "timeout_multiplier"):
		if key in data:
			setattr(rc, key, float(data[key]))
	return rc


def _build_planner_config(data: dict[str, Any]) -> PlannerConfig:
	pc = PlannerConfig()
	for key in ("max_depth", "absolute_max_depth", "max_children_per_node", "max_file_tree_chars"):
		if key in data:
			setattr(pc, key, int(data[key]))
	if "budget_per_call_usd" in data:
		pc.budget_per_call_usd = float(data["budget_per_call_usd"])
	# Hard cap max_depth at absolute_max_depth to prevent runaway recursion
	pc.max_depth = min(pc.max_depth, pc.absolute_max_depth)
	return pc


def _build_continuous(data: dict[str, Any]) -> ContinuousConfig:
	cc = ContinuousConfig()
	int_keys = (
		"max_wall_time_seconds", "stall_threshold_units",
		"replan_interval_units", "backlog_min_size", "cooldown_between_units",
	)
	for key in int_keys:
		if key in data:
			setattr(cc, key, int(data[key]))
	if "stall_score_epsilon" in data:
		cc.stall_score_epsilon = float(data["stall_score_epsilon"])
	if "verify_before_merge" in data:
		cc.verify_before_merge = bool(data["verify_before_merge"])
	if "timeout_multiplier" in data:
		cc.timeout_multiplier = float(data["timeout_multiplier"])
	if "backlog_max_age_seconds" in data:
		cc.backlog_max_age_seconds = int(data["backlog_max_age_seconds"])
	for key in (
		"retry_base_delay_seconds", "retry_max_delay_seconds", "chain_max_depth",
		"max_consecutive_failures", "failure_backoff_seconds",
		"min_ambition_score", "max_replan_attempts", "max_objective_checks",
	):
		if key in data:
			setattr(cc, key, int(data[key]))
	if "verify_objective_completion" in data:
		cc.verify_objective_completion = bool(data["verify_objective_completion"])
	if "cleanup_enabled" in data:
		cc.cleanup_enabled = bool(data["cleanup_enabled"])
	if "cleanup_interval" in data:
		cc.cleanup_interval = int(data["cleanup_interval"])
	if "circuit_breaker_enabled" in data:
		cc.circuit_breaker_enabled = bool(data["circuit_breaker_enabled"])
	if "circuit_breaker_max_failures" in data:
		cc.circuit_breaker_max_failures = int(data["circuit_breaker_max_failures"])
	if "circuit_breaker_cooldown_seconds" in data:
		cc.circuit_breaker_cooldown_seconds = int(data["circuit_breaker_cooldown_seconds"])
	return cc


def _build_green_branch(data: dict[str, Any]) -> GreenBranchConfig:
	gc = GreenBranchConfig()
	for key in ("working_branch", "green_branch"):
		if key in data:
			setattr(gc, key, str(data[key]))
	if "fixup_max_attempts" in data:
		gc.fixup_max_attempts = int(data["fixup_max_attempts"])
	if "fixup_candidates" in data:
		gc.fixup_candidates = int(data["fixup_candidates"])
	if "reset_on_init" in data:
		gc.reset_on_init = bool(data["reset_on_init"])
	if "auto_push" in data:
		gc.auto_push = bool(data["auto_push"])
	if "push_branch" in data:
		gc.push_branch = str(data["push_branch"])
	return gc


def _build_discovery(data: dict[str, Any]) -> DiscoveryConfig:
	dc = DiscoveryConfig()
	if "enabled" in data:
		dc.enabled = bool(data["enabled"])
	if "tracks" in data:
		dc.tracks = list(data["tracks"])
	for key in ("max_items_per_track",):
		if key in data:
			setattr(dc, key, int(data[key]))
	if "min_priority_score" in data:
		dc.min_priority_score = float(data["min_priority_score"])
	if "model" in data:
		dc.model = str(data["model"])
	if "budget_per_call_usd" in data:
		dc.budget_per_call_usd = float(data["budget_per_call_usd"])
	if "research_enabled" in data:
		dc.research_enabled = bool(data["research_enabled"])
	if "research_model" in data:
		dc.research_model = str(data["research_model"])
	if "research_parallel_queries" in data:
		dc.research_parallel_queries = int(data["research_parallel_queries"])
	if "effort_weight" in data:
		dc.effort_weight = float(data["effort_weight"])
	return dc


def _build_pricing(data: dict[str, Any]) -> PricingConfig:
	pc = PricingConfig()
	for key in ("input_per_million", "output_per_million", "cache_write_per_million", "cache_read_per_million"):
		if key in data:
			setattr(pc, key, float(data[key]))
	return pc


def _build_container(data: dict[str, Any]) -> ContainerConfig:
	cc = ContainerConfig()
	for key in ("image", "docker_executable", "workspace_mount", "claude_config_dir", "network", "run_as_user"):
		if key in data:
			setattr(cc, key, str(data[key]))
	for key in ("extra_volumes", "cap_drop", "security_opt"):
		if key in data:
			setattr(cc, key, [str(v) for v in data[key]])
	if "startup_timeout" in data:
		cc.startup_timeout = int(data["startup_timeout"])
	return cc


def _build_backend(data: dict[str, Any]) -> BackendConfig:
	bc = BackendConfig()
	if "type" in data:
		bc.type = str(data["type"])
	if "max_output_mb" in data:
		bc.max_output_mb = int(data["max_output_mb"])
	if "ssh" in data:
		ssh_data = data["ssh"]
		hosts = ssh_data.get("hosts", [])
		for h in hosts:
			host = SSHHostConfig()
			if "hostname" in h:
				host.hostname = str(h["hostname"])
			if "user" in h:
				host.user = str(h["user"])
			if "max_workers" in h:
				host.max_workers = int(h["max_workers"])
			bc.ssh_hosts.append(host)
	if "container" in data:
		bc.container = _build_container(data["container"])
	return bc


def _build_heartbeat(data: dict[str, Any]) -> HeartbeatConfig:
	hc = HeartbeatConfig()
	if "interval" in data:
		hc.interval = int(data["interval"])
	if "idle_threshold" in data:
		hc.idle_threshold = int(data["idle_threshold"])
	if "enable_recovery" in data:
		hc.enable_recovery = bool(data["enable_recovery"])
	return hc


def _build_notifications(data: dict[str, Any]) -> NotificationConfig:
	nc = NotificationConfig()
	if "telegram" in data:
		tc = TelegramConfig()
		tg = data["telegram"]
		if "bot_token" in tg:
			tc.bot_token = str(tg["bot_token"])
		if "chat_id" in tg:
			tc.chat_id = str(tg["chat_id"])
		if "on_heartbeat" in tg:
			tc.on_heartbeat = bool(tg["on_heartbeat"])
		if "on_merge_fail" in tg:
			tc.on_merge_fail = bool(tg["on_merge_fail"])
		if "on_mission_end" in tg:
			tc.on_mission_end = bool(tg["on_mission_end"])
		nc.telegram = tc
	return nc


def _build_review(data: dict[str, Any]) -> ReviewConfig:
	rc = ReviewConfig()
	if "enabled" in data:
		rc.enabled = bool(data["enabled"])
	if "model" in data:
		rc.model = str(data["model"])
	if "budget_per_review_usd" in data:
		rc.budget_per_review_usd = float(data["budget_per_review_usd"])
	if "gate_completion" in data:
		rc.gate_completion = bool(data["gate_completion"])
	if "min_review_score" in data:
		rc.min_review_score = float(data["min_review_score"])
	return rc


def _build_models(data: dict[str, Any]) -> ModelsConfig:
	mc = ModelsConfig()
	for key in ("planner_model", "worker_model", "fixup_model"):
		if key in data:
			setattr(mc, key, str(data[key]))
	if "architect_editor_mode" in data:
		mc.architect_editor_mode = bool(data["architect_editor_mode"])
	return mc


def _build_specialist(data: dict[str, Any]) -> SpecialistConfig:
	sc = SpecialistConfig()
	if "templates_dir" in data:
		sc.templates_dir = str(data["templates_dir"])
	return sc


def _build_tool_synthesis(data: dict[str, Any]) -> ToolSynthesisConfig:
	tc = ToolSynthesisConfig()
	if "enabled" in data:
		tc.enabled = bool(data["enabled"])
	if "tools_dir" in data:
		tc.tools_dir = str(data["tools_dir"])
	return tc


def _build_tracing(data: dict[str, Any]) -> TracingConfig:
	tc = TracingConfig()
	if "enabled" in data:
		tc.enabled = bool(data["enabled"])
	if "service_name" in data:
		tc.service_name = str(data["service_name"])
	if "exporter" in data:
		tc.exporter = str(data["exporter"])
	if "otlp_endpoint" in data:
		tc.otlp_endpoint = str(data["otlp_endpoint"])
	return tc


def _build_security(data: dict[str, Any]) -> SecurityConfig:
	sc = SecurityConfig()
	if "extra_env_keys" in data:
		sc.extra_env_keys = [str(k) for k in data["extra_env_keys"]]
	return sc


def _build_hitl_gate(data: dict[str, Any]) -> HITLGateConfig:
	gc = HITLGateConfig()
	if "enabled" in data:
		gc.enabled = bool(data["enabled"])
	if "timeout_seconds" in data:
		gc.timeout_seconds = int(data["timeout_seconds"])
	if "timeout_action" in data:
		gc.timeout_action = str(data["timeout_action"])
	if "large_merge_threshold_lines" in data:
		gc.large_merge_threshold_lines = int(data["large_merge_threshold_lines"])
	if "large_merge_threshold_files" in data:
		gc.large_merge_threshold_files = int(data["large_merge_threshold_files"])
	return gc


def _build_hitl(data: dict[str, Any]) -> HITLConfig:
	hc = HITLConfig()
	if "approvals_dir" in data:
		hc.approvals_dir = str(data["approvals_dir"])
	if "telegram_poll_interval" in data:
		hc.telegram_poll_interval = float(data["telegram_poll_interval"])
	if "push_gate" in data:
		hc.push_gate = _build_hitl_gate(data["push_gate"])
	if "large_merge_gate" in data:
		hc.large_merge_gate = _build_hitl_gate(data["large_merge_gate"])
	return hc


def _build_zfc(data: dict[str, Any]) -> ZFCConfig:
	zc = ZFCConfig()
	for key in ("zfc_ambition_scoring", "zfc_fixup_prompts", "zfc_propose_objective"):
		if key in data:
			setattr(zc, key, bool(data[key]))
	if "llm_timeout" in data:
		zc.llm_timeout = int(data["llm_timeout"])
	if "llm_budget_usd" in data:
		zc.llm_budget_usd = float(data["llm_budget_usd"])
	if "model" in data:
		zc.model = str(data["model"])
	return zc


def _build_a2a(data: dict[str, Any]) -> A2AConfig:
	ac = A2AConfig()
	if "enabled" in data:
		ac.enabled = bool(data["enabled"])
	if "host" in data:
		ac.host = str(data["host"])
	if "port" in data:
		ac.port = int(data["port"])
	if "agent_name" in data:
		ac.agent_name = str(data["agent_name"])
	if "agent_version" in data:
		ac.agent_version = str(data["agent_version"])
	if "agent_capabilities" in data:
		ac.agent_capabilities = [str(c) for c in data["agent_capabilities"]]
	return ac


def _build_mcp_registry(data: dict[str, Any]) -> MCPRegistryConfig:
	mc = MCPRegistryConfig()
	if "enabled" in data:
		mc.enabled = bool(data["enabled"])
	if "promotion_threshold" in data:
		mc.promotion_threshold = float(data["promotion_threshold"])
	if "ema_alpha" in data:
		mc.ema_alpha = float(data["ema_alpha"])
	return mc


def _build_prompt_evolution(data: dict[str, Any]) -> PromptEvolutionConfig:
	pc = PromptEvolutionConfig()
	if "enabled" in data:
		pc.enabled = bool(data["enabled"])
	if "mutation_model" in data:
		pc.mutation_model = str(data["mutation_model"])
	if "exploration_factor" in data:
		pc.exploration_factor = float(data["exploration_factor"])
	if "min_samples_before_mutation" in data:
		pc.min_samples_before_mutation = int(data["min_samples_before_mutation"])
	return pc


def _build_speculation(data: dict[str, Any]) -> SpeculationConfig:
	sc = SpeculationConfig()
	if "enabled" in data:
		sc.enabled = bool(data["enabled"])
	if "uncertainty_threshold" in data:
		sc.uncertainty_threshold = float(data["uncertainty_threshold"])
	if "branch_count" in data:
		sc.branch_count = int(data["branch_count"])
	if "selection_metric" in data:
		sc.selection_metric = str(data["selection_metric"])
	if "cost_limit_multiplier" in data:
		sc.cost_limit_multiplier = float(data["cost_limit_multiplier"])
	return sc


def _build_episodic_memory(data: dict[str, Any]) -> EpisodicMemoryConfig:
	ec = EpisodicMemoryConfig()
	if "enabled" in data:
		ec.enabled = bool(data["enabled"])
	if "default_ttl_days" in data:
		ec.default_ttl_days = int(data["default_ttl_days"])
	if "decay_alpha" in data:
		ec.decay_alpha = float(data["decay_alpha"])
	if "access_boost_days" in data:
		ec.access_boost_days = int(data["access_boost_days"])
	if "distill_model" in data:
		ec.distill_model = str(data["distill_model"])
	if "distill_budget_usd" in data:
		ec.distill_budget_usd = float(data["distill_budget_usd"])
	if "min_episodes_for_distill" in data:
		ec.min_episodes_for_distill = int(data["min_episodes_for_distill"])
	if "max_semantic_per_query" in data:
		ec.max_semantic_per_query = int(data["max_semantic_per_query"])
	return ec


def _build_degradation(data: dict[str, Any]) -> DegradationConfig:
	dc = DegradationConfig()
	float_keys = (
		"budget_fraction_threshold", "conflict_rate_threshold", "reduced_worker_fraction",
		"rate_limit_window_seconds", "safe_stop_timeout_seconds",
	)
	for key in float_keys:
		if key in data:
			setattr(dc, key, float(data[key]))
	int_keys = (
		"db_error_threshold", "rate_limit_threshold",
		"verification_failure_threshold", "recovery_success_threshold",
	)
	for key in int_keys:
		if key in data:
			setattr(dc, key, int(data[key]))
	return dc


def _build_deploy(data: dict[str, Any]) -> DeployConfig:
	dc = DeployConfig()
	if "enabled" in data:
		dc.enabled = bool(data["enabled"])
	if "command" in data:
		dc.command = str(data["command"])
	if "health_check_url" in data:
		dc.health_check_url = str(data["health_check_url"])
	for key in ("health_check_timeout", "timeout"):
		if key in data:
			setattr(dc, key, int(data[key]))
	if "on_auto_push" in data:
		dc.on_auto_push = bool(data["on_auto_push"])
	if "on_mission_end" in data:
		dc.on_mission_end = bool(data["on_mission_end"])
	return dc


_ENV_ALLOWLIST = {
	# System essentials
	"HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "LC_CTYPE",
	"TERM", "TERM_PROGRAM", "TMPDIR", "TMP", "TEMP", "XDG_RUNTIME_DIR",
	"XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME",
	# Required for subprocess execution
	"PATH", "PWD", "OLDPWD", "SHLVL",
	# Claude Code auth (OAuth/Max -- NOT raw API keys)
	"CLAUDE_CONFIG_DIR",
	# Python / Node toolchain
	"VIRTUAL_ENV", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE",
	"NODE_PATH", "NPM_CONFIG_PREFIX",
	# Git (needed for worker commits)
	"GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL",
	"GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL",
	# Encoding
	"PYTHONIOENCODING", "PYTHONUTF8",
}

# Keys that must NEVER reach workers, even if added to extra_env_keys
_ENV_DENYLIST = {
	"ANTHROPIC_API_KEY", "CLAUDECODE",
	"AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
	"GITHUB_TOKEN", "GH_TOKEN", "GITLAB_TOKEN",
	"NPM_TOKEN", "PYPI_TOKEN",
	"DATABASE_URL", "REDIS_URL",
	"TELEGRAM_BOT_TOKEN",
}

# Module-level extra keys, populated by load_config from [security] section
_extra_env_keys: set[str] = set()


def claude_subprocess_env() -> dict[str, str]:
	"""Build a restricted environment for claude subprocess calls.

	Uses an allowlist approach: only safe system vars plus any explicitly
	configured extras are passed through. Secrets and API keys are stripped.
	"""
	allowed = _ENV_ALLOWLIST | _extra_env_keys
	return {
		k: v for k, v in os.environ.items()
		if k in allowed and k not in _ENV_DENYLIST
	}


def load_config(path: str | Path) -> MissionConfig:
	"""Load a mission-control.toml config file.

	Args:
		path: Path to the TOML config file.

	Returns:
		Parsed MissionConfig.

	Raises:
		FileNotFoundError: If config file doesn't exist.
		tomllib.TOMLDecodeError: If config is invalid TOML.
	"""
	config_path = Path(path)
	if not config_path.exists():
		raise FileNotFoundError(f"Config file not found: {config_path}")

	with open(config_path, "rb") as f:
		data = tomllib.load(f)

	mc = MissionConfig()
	if "target" in data:
		mc.target = _build_target(data["target"])
	if "scheduler" in data:
		mc.scheduler = _build_scheduler(data["scheduler"])
	if "rounds" in data:
		mc.rounds = _build_rounds(data["rounds"])
	if "continuous" in data:
		mc.continuous = _build_continuous(data["continuous"])
	if "planner" in data:
		mc.planner = _build_planner_config(data["planner"])
	if "green_branch" in data:
		mc.green_branch = _build_green_branch(data["green_branch"])
	if "backend" in data:
		mc.backend = _build_backend(data["backend"])
	if "discovery" in data:
		mc.discovery = _build_discovery(data["discovery"])
	if "pricing" in data:
		mc.pricing = _build_pricing(data["pricing"])
	if "heartbeat" in data:
		mc.heartbeat = _build_heartbeat(data["heartbeat"])
	if "notifications" in data:
		mc.notifications = _build_notifications(data["notifications"])
	if "dashboard" in data:
		mc.dashboard = _build_dashboard(data["dashboard"])
	if "review" in data:
		mc.review = _build_review(data["review"])
	if "deploy" in data:
		mc.deploy = _build_deploy(data["deploy"])
	if "models" in data:
		mc.models = _build_models(data["models"])
	if "specialist" in data:
		mc.specialist = _build_specialist(data["specialist"])
	if "tool_synthesis" in data:
		mc.tool_synthesis = _build_tool_synthesis(data["tool_synthesis"])
	if "security" in data:
		mc.security = _build_security(data["security"])
	if "tracing" in data:
		mc.tracing = _build_tracing(data["tracing"])
	if "hitl" in data:
		mc.hitl = _build_hitl(data["hitl"])
	if "zfc" in data:
		mc.zfc = _build_zfc(data["zfc"])
	if "degradation" in data:
		mc.degradation = _build_degradation(data["degradation"])
	if "a2a" in data:
		mc.a2a = _build_a2a(data["a2a"])
	if "mcp_registry" in data:
		mc.mcp_registry = _build_mcp_registry(data["mcp_registry"])
	if "prompt_evolution" in data:
		mc.prompt_evolution = _build_prompt_evolution(data["prompt_evolution"])
	if "episodic_memory" in data:
		mc.episodic_memory = _build_episodic_memory(data["episodic_memory"])
	if "speculation" in data:
		mc.speculation = _build_speculation(data["speculation"])
	# Populate module-level extra env keys for claude_subprocess_env()
	global _extra_env_keys
	_extra_env_keys = set(mc.security.extra_env_keys) - _ENV_DENYLIST
	# Allow env vars as fallback for Telegram credentials
	tg = mc.notifications.telegram
	if not tg.bot_token:
		tg.bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
	if not tg.chat_id:
		tg.chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
	return mc


_TELEGRAM_TOKEN_RE = re.compile(r"^\d+:[A-Za-z0-9_-]+$")


def validate_config(config: MissionConfig) -> list[tuple[str, str]]:
	"""Perform semantic validation of a loaded MissionConfig.

	Returns a list of (level, message) tuples where level is 'error' or 'warning'.
	"""
	issues: list[tuple[str, str]] = []

	# 1. target.path exists and is a git repo
	target_path = config.target.resolved_path
	if not target_path.exists():
		issues.append(("error", f"target.path does not exist: {target_path}"))
	elif not (target_path / ".git").is_dir():
		issues.append(("error", f"target.path is not a git repository (no .git dir): {target_path}"))

	# 2. pool_dir is writable if set
	pool_dir = config.scheduler.parallel.pool_dir
	if pool_dir:
		pool_path = Path(os.path.expanduser(pool_dir))
		if pool_path.exists() and not os.access(pool_path, os.W_OK):
			issues.append(("error", f"pool_dir is not writable: {pool_path}"))
		elif not pool_path.exists():
			issues.append(("error", f"pool_dir does not exist: {pool_path}"))

	# 3. verification command first token is executable
	verify_cmd = config.target.verification.command
	if verify_cmd:
		first_token = verify_cmd.split()[0]
		if shutil.which(first_token) is None:
			issues.append(("error", f"verification command not found: {first_token}"))

	# 4. Telegram bot_token format if notifications enabled
	tg = config.notifications.telegram
	notifications_enabled = tg.on_heartbeat or tg.on_merge_fail or tg.on_mission_end
	if notifications_enabled and tg.bot_token and not _TELEGRAM_TOKEN_RE.match(tg.bot_token):
		issues.append(("error", "telegram bot_token format invalid (expected digits:alphanumeric)"))

	# 5. Container backend checks
	if config.backend.type == "container":
		cc = config.backend.container
		if not cc.image:
			issues.append(("error", "backend.container.image must not be empty"))
		docker_exe = shutil.which(cc.docker_executable)
		if docker_exe is None:
			issues.append(("error", f"docker executable not found on PATH: {cc.docker_executable}"))
		if cc.claude_config_dir:
			ccd = Path(os.path.expanduser(cc.claude_config_dir))
			if not ccd.exists():
				issues.append(("error", f"backend.container.claude_config_dir does not exist: {ccd}"))

	# 6. Suspicious values
	if config.scheduler.session_timeout < 60:
		issues.append(("warning", f"session_timeout is very low: {config.scheduler.session_timeout}s"))
	if config.scheduler.parallel.num_workers > 8:
		issues.append(("warning", f"num_workers is high: {config.scheduler.parallel.num_workers}"))
	if config.scheduler.session_timeout < 0:
		issues.append(("warning", f"session_timeout is negative: {config.scheduler.session_timeout}"))
	if config.scheduler.parallel.num_workers == 0:
		issues.append(("warning", "num_workers is zero"))

	return issues
