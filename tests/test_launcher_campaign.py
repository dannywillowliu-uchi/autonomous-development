"""Tests for campaign-mode mission chaining in MissionLauncher."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mission_control.db import Database
from mission_control.launcher import MissionLauncher
from mission_control.models import Campaign, CampaignObjective, Mission
from mission_control.registry import ProjectRegistry


@pytest.fixture
def registry(tmp_path):
	db_path = tmp_path / "test_registry.db"
	reg = ProjectRegistry(db_path=db_path, allowed_bases=[tmp_path, Path.home()])
	yield reg
	reg.close()


@pytest.fixture
def sample_config(tmp_path):
	config = tmp_path / "mission-control.toml"
	config.write_text(
		'[target]\nname = "test"\npath = "."\nbranch = "main"\nobjective = "test"\n'
		'[target.verification]\ncommand = "echo ok"\ntimeout = 60\n'
		"[scheduler]\nmodel = \"sonnet\"\n"
		"[scheduler.git]\nstrategy = \"branch-per-session\"\n"
		"[scheduler.budget]\nmax_per_session_usd = 1.0\nmax_per_run_usd = 10.0\n"
		"[scheduler.parallel]\nnum_workers = 2\n"
		"[rounds]\nmax_rounds = 5\nstall_threshold = 3\n"
		"[planner]\n"
		"[green_branch]\nworking_branch = \"mc/working\"\ngreen_branch = \"mc/green\"\n"
		'[backend]\ntype = "local"\n'
	)
	return config


@pytest.fixture
def project_db(tmp_path):
	db_path = tmp_path / "mission-control.db"
	db = Database(db_path)
	yield db
	db.close()


@pytest.fixture
def launcher(registry):
	return MissionLauncher(registry)


def _register_project(registry, sample_config, tmp_path):
	"""Helper to register a test project with a DB path."""
	db_path = tmp_path / "mission-control.db"
	registry.register(name="test", config_path=str(sample_config), db_path=str(db_path))
	return db_path


class TestLinearChainExecution:
	"""Test objectives that form a linear dependency chain: 0 -> 1 -> 2."""

	@patch("mission_control.launcher.subprocess.Popen")
	@patch("mission_control.launcher._pid_alive", return_value=False)
	def test_linear_chain_all_succeed(
		self, mock_alive, mock_popen, launcher, registry, sample_config, tmp_path
	):
		db_path = _register_project(registry, sample_config, tmp_path)

		# Track launch count
		launch_count = 0

		def fake_popen(*args, **kwargs):
			nonlocal launch_count
			launch_count += 1
			# Simulate a completed mission in the DB each time
			db = Database(db_path)
			mission = Mission(objective=f"obj-{launch_count}", status="completed")
			db.insert_mission(mission)
			db.close()

			mock_proc = MagicMock()
			mock_proc.pid = 10000 + launch_count
			return mock_proc

		mock_popen.side_effect = fake_popen

		campaign = Campaign(
			name="linear-test",
			objectives=[
				CampaignObjective(objective="step 1", depends_on_indices=[]),
				CampaignObjective(objective="step 2", depends_on_indices=[0]),
				CampaignObjective(objective="step 3", depends_on_indices=[1]),
			],
		)

		result = launcher.run_campaign("test", campaign)

		assert result.status == "completed"
		assert all(obj.status == "completed" for obj in result.objectives)
		assert launch_count == 3
		assert result.started_at is not None
		assert result.finished_at is not None

	@patch("mission_control.launcher.subprocess.Popen")
	@patch("mission_control.launcher._pid_alive", return_value=False)
	def test_linear_chain_middle_fails(
		self, mock_alive, mock_popen, launcher, registry, sample_config, tmp_path
	):
		db_path = _register_project(registry, sample_config, tmp_path)

		launch_count = 0

		def fake_popen(*args, **kwargs):
			nonlocal launch_count
			launch_count += 1
			db = Database(db_path)
			# Second mission fails
			status = "completed" if launch_count != 2 else "failed"
			mission = Mission(objective=f"obj-{launch_count}", status=status)
			db.insert_mission(mission)
			db.close()

			mock_proc = MagicMock()
			mock_proc.pid = 10000 + launch_count
			return mock_proc

		mock_popen.side_effect = fake_popen

		campaign = Campaign(
			name="linear-fail",
			objectives=[
				CampaignObjective(objective="step 1", depends_on_indices=[]),
				CampaignObjective(objective="step 2", depends_on_indices=[0]),
				CampaignObjective(objective="step 3", depends_on_indices=[1]),
			],
		)

		result = launcher.run_campaign("test", campaign)

		assert result.status == "failed"
		assert result.objectives[0].status == "completed"
		assert result.objectives[1].status == "failed"
		assert result.objectives[2].status == "skipped"
		assert launch_count == 2  # third never launched


class TestParallelIndependentObjectives:
	"""Test objectives with no dependencies that can all be runnable."""

	@patch("mission_control.launcher.subprocess.Popen")
	@patch("mission_control.launcher._pid_alive", return_value=False)
	def test_independent_objectives_all_succeed(
		self, mock_alive, mock_popen, launcher, registry, sample_config, tmp_path
	):
		db_path = _register_project(registry, sample_config, tmp_path)

		launch_count = 0

		def fake_popen(*args, **kwargs):
			nonlocal launch_count
			launch_count += 1
			db = Database(db_path)
			mission = Mission(objective=f"obj-{launch_count}", status="completed")
			db.insert_mission(mission)
			db.close()

			mock_proc = MagicMock()
			mock_proc.pid = 10000 + launch_count
			return mock_proc

		mock_popen.side_effect = fake_popen

		campaign = Campaign(
			name="parallel-test",
			objectives=[
				CampaignObjective(objective="independent A", depends_on_indices=[]),
				CampaignObjective(objective="independent B", depends_on_indices=[]),
				CampaignObjective(objective="independent C", depends_on_indices=[]),
			],
		)

		result = launcher.run_campaign("test", campaign)

		assert result.status == "completed"
		assert all(obj.status == "completed" for obj in result.objectives)
		assert launch_count == 3

	@patch("mission_control.launcher.subprocess.Popen")
	@patch("mission_control.launcher._pid_alive", return_value=False)
	def test_diamond_dependency(
		self, mock_alive, mock_popen, launcher, registry, sample_config, tmp_path
	):
		"""Test diamond: 0 -> 1, 0 -> 2, 1+2 -> 3."""
		db_path = _register_project(registry, sample_config, tmp_path)

		launch_count = 0

		def fake_popen(*args, **kwargs):
			nonlocal launch_count
			launch_count += 1
			db = Database(db_path)
			mission = Mission(objective=f"obj-{launch_count}", status="completed")
			db.insert_mission(mission)
			db.close()

			mock_proc = MagicMock()
			mock_proc.pid = 10000 + launch_count
			return mock_proc

		mock_popen.side_effect = fake_popen

		campaign = Campaign(
			name="diamond-test",
			objectives=[
				CampaignObjective(objective="base", depends_on_indices=[]),
				CampaignObjective(objective="left", depends_on_indices=[0]),
				CampaignObjective(objective="right", depends_on_indices=[0]),
				CampaignObjective(objective="join", depends_on_indices=[1, 2]),
			],
		)

		result = launcher.run_campaign("test", campaign)

		assert result.status == "completed"
		assert all(obj.status == "completed" for obj in result.objectives)
		assert launch_count == 4


class TestFailurePropagation:
	"""Test that failure correctly skips dependent objectives."""

	@patch("mission_control.launcher.subprocess.Popen")
	@patch("mission_control.launcher._pid_alive", return_value=False)
	def test_failure_skips_all_dependents(
		self, mock_alive, mock_popen, launcher, registry, sample_config, tmp_path
	):
		db_path = _register_project(registry, sample_config, tmp_path)

		launch_count = 0

		def fake_popen(*args, **kwargs):
			nonlocal launch_count
			launch_count += 1
			db = Database(db_path)
			# First objective fails
			status = "failed" if launch_count == 1 else "completed"
			mission = Mission(objective=f"obj-{launch_count}", status=status)
			db.insert_mission(mission)
			db.close()

			mock_proc = MagicMock()
			mock_proc.pid = 10000 + launch_count
			return mock_proc

		mock_popen.side_effect = fake_popen

		campaign = Campaign(
			name="failure-test",
			objectives=[
				CampaignObjective(objective="root", depends_on_indices=[]),
				CampaignObjective(objective="child A", depends_on_indices=[0]),
				CampaignObjective(objective="child B", depends_on_indices=[0]),
				CampaignObjective(objective="grandchild", depends_on_indices=[1]),
			],
		)

		result = launcher.run_campaign("test", campaign)

		assert result.status == "failed"
		assert result.objectives[0].status == "failed"
		assert result.objectives[1].status == "skipped"
		assert result.objectives[2].status == "skipped"
		assert result.objectives[3].status == "skipped"
		assert launch_count == 1

	@patch("mission_control.launcher.subprocess.Popen")
	@patch("mission_control.launcher._pid_alive", return_value=False)
	def test_partial_failure_independent_branch_continues(
		self, mock_alive, mock_popen, launcher, registry, sample_config, tmp_path
	):
		"""If one branch fails, independent branches still execute."""
		db_path = _register_project(registry, sample_config, tmp_path)

		launch_count = 0

		def fake_popen(*args, **kwargs):
			nonlocal launch_count
			launch_count += 1
			db = Database(db_path)
			# Objective index 1 (second launch) fails
			status = "failed" if launch_count == 2 else "completed"
			mission = Mission(objective=f"obj-{launch_count}", status=status)
			db.insert_mission(mission)
			db.close()

			mock_proc = MagicMock()
			mock_proc.pid = 10000 + launch_count
			return mock_proc

		mock_popen.side_effect = fake_popen

		campaign = Campaign(
			name="partial-fail",
			objectives=[
				CampaignObjective(objective="base", depends_on_indices=[]),
				CampaignObjective(objective="branch A (fails)", depends_on_indices=[0]),
				CampaignObjective(objective="branch B (succeeds)", depends_on_indices=[0]),
				CampaignObjective(objective="depends on A (skipped)", depends_on_indices=[1]),
			],
		)

		result = launcher.run_campaign("test", campaign)

		assert result.status == "failed"
		assert result.objectives[0].status == "completed"
		assert result.objectives[1].status == "failed"
		assert result.objectives[2].status == "completed"
		assert result.objectives[3].status == "skipped"
		assert launch_count == 3

	def test_run_campaign_unregistered_project(self, launcher):
		campaign = Campaign(name="test", objectives=[])
		with pytest.raises(ValueError, match="not registered"):
			launcher.run_campaign("nonexistent", campaign)

	@patch("mission_control.launcher.subprocess.Popen")
	@patch("mission_control.launcher._pid_alive", return_value=False)
	def test_empty_campaign(
		self, mock_alive, mock_popen, launcher, registry, sample_config, tmp_path
	):
		_register_project(registry, sample_config, tmp_path)
		campaign = Campaign(name="empty", objectives=[])
		result = launcher.run_campaign("test", campaign)
		assert result.status == "completed"
		assert result.finished_at is not None

	@patch("mission_control.launcher.subprocess.Popen")
	@patch("mission_control.launcher._pid_alive", return_value=False)
	def test_strategic_context_shared(
		self, mock_alive, mock_popen, launcher, registry, sample_config, tmp_path
	):
		db_path = _register_project(registry, sample_config, tmp_path)

		launch_count = 0

		def fake_popen(*args, **kwargs):
			nonlocal launch_count
			launch_count += 1
			db = Database(db_path)
			mission = Mission(objective=f"obj-{launch_count}", status="completed")
			db.insert_mission(mission)
			db.close()

			mock_proc = MagicMock()
			mock_proc.pid = 10000 + launch_count
			return mock_proc

		mock_popen.side_effect = fake_popen

		campaign = Campaign(
			name="context-test",
			objectives=[
				CampaignObjective(objective="step 1"),
				CampaignObjective(objective="step 2", depends_on_indices=[0]),
			],
		)

		launcher.run_campaign("test", campaign)

		# Verify strategic context was written for completed objectives
		db = Database(db_path)
		contexts = db.get_strategic_context(limit=10)
		db.close()
		assert len(contexts) == 2
		assert "Campaign objective 0" in contexts[1].what_attempted
		assert "Campaign objective 1" in contexts[0].what_attempted


class TestCampaignFileParsing:
	"""Test TOML campaign file parsing."""

	def test_parse_basic_campaign(self, tmp_path):
		toml_path = tmp_path / "campaign.toml"
		toml_path.write_text(
			'[campaign]\n'
			'name = "refactor-flow"\n'
			'\n'
			'[[campaign.objectives]]\n'
			'objective = "refactor auth module"\n'
			'depends_on = []\n'
			'\n'
			'[[campaign.objectives]]\n'
			'objective = "add OAuth support"\n'
			'depends_on = [0]\n'
			'\n'
			'[[campaign.objectives]]\n'
			'objective = "write integration tests"\n'
			'depends_on = [0, 1]\n'
		)

		campaign = MissionLauncher.parse_campaign_file(toml_path)

		assert campaign.name == "refactor-flow"
		assert len(campaign.objectives) == 3
		assert campaign.objectives[0].objective == "refactor auth module"
		assert campaign.objectives[0].depends_on_indices == []
		assert campaign.objectives[1].objective == "add OAuth support"
		assert campaign.objectives[1].depends_on_indices == [0]
		assert campaign.objectives[2].depends_on_indices == [0, 1]
		assert campaign.status == "pending"

	def test_parse_with_config_overrides(self, tmp_path):
		toml_path = tmp_path / "campaign.toml"
		toml_path.write_text(
			'[campaign]\n'
			'name = "override-test"\n'
			'\n'
			'[[campaign.objectives]]\n'
			'objective = "fast step"\n'
			'depends_on = []\n'
			'[campaign.objectives.config_overrides]\n'
			'mode = "continuous"\n'
			'max_rounds = "10"\n'
		)

		campaign = MissionLauncher.parse_campaign_file(toml_path)

		assert campaign.objectives[0].config_overrides == {
			"mode": "continuous",
			"max_rounds": "10",
		}

	def test_parse_defaults_name_to_stem(self, tmp_path):
		toml_path = tmp_path / "my-workflow.toml"
		toml_path.write_text(
			'[campaign]\n'
			'[[campaign.objectives]]\n'
			'objective = "only step"\n'
			'depends_on = []\n'
		)

		campaign = MissionLauncher.parse_campaign_file(toml_path)
		assert campaign.name == "my-workflow"

	def test_parse_empty_objectives(self, tmp_path):
		toml_path = tmp_path / "empty.toml"
		toml_path.write_text('[campaign]\nname = "empty"\n')

		campaign = MissionLauncher.parse_campaign_file(toml_path)
		assert campaign.name == "empty"
		assert campaign.objectives == []

	def test_parse_nonexistent_file(self):
		with pytest.raises(FileNotFoundError):
			MissionLauncher.parse_campaign_file(Path("/nonexistent/campaign.toml"))
