"""Tests for multi-contributor coordination protocol."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from autodev.contrib import ContributorProtocol


@pytest.fixture
def repo(tmp_path):
	return tmp_path


@pytest.fixture
def proto(repo):
	return ContributorProtocol(repo, "alice")


def _make_git_mock(returncode=0, stdout="", stderr=""):
	"""Create a mock for asyncio.create_subprocess_exec."""
	mock_proc = AsyncMock()
	mock_proc.communicate = AsyncMock(return_value=(stdout.encode(), stderr.encode()))
	mock_proc.returncode = returncode
	return mock_proc


def _patch_git(returncode=0, stdout="", stderr=""):
	mock_proc = _make_git_mock(returncode, stdout, stderr)
	return patch("asyncio.create_subprocess_exec", return_value=mock_proc)


@pytest.mark.asyncio
async def test_register_new_contributor(repo, proto):
	with _patch_git() as mock_exec:
		info = await proto.register()

	assert info.username == "alice"
	assert info.joined_at != ""
	assert info.agent_count == 0

	registry = json.loads((repo / ContributorProtocol.REGISTRY_FILE).read_text())
	assert "alice" in registry
	assert registry["alice"]["username"] == "alice"

	calls = [c[0] for c in mock_exec.call_args_list]
	commit_calls = [c for c in calls if "commit" in c]
	assert len(commit_calls) > 0


@pytest.mark.asyncio
async def test_register_existing_contributor(repo, proto):
	registry = {
		"alice": {
			"username": "alice", "joined_at": "2026-01-01T00:00:00+00:00",
			"agent_count": 3, "proposals_completed": 5,
		},
	}
	(repo / ContributorProtocol.REGISTRY_FILE).write_text(json.dumps(registry))

	with _patch_git():
		info = await proto.register()

	assert info.username == "alice"
	assert info.joined_at == "2026-01-01T00:00:00+00:00"
	assert info.agent_count == 3
	assert info.proposals_completed == 5


@pytest.mark.asyncio
async def test_claim_proposal_success(repo, proto):
	(repo / ContributorProtocol.CLAIMS_FILE).write_text("{}")

	with _patch_git():
		result = await proto.claim_proposal("prop-1")

	assert result is True
	claims = json.loads((repo / ContributorProtocol.CLAIMS_FILE).read_text())
	assert "prop-1" in claims
	assert claims["prop-1"]["user"] == "alice"
	assert claims["prop-1"]["status"] == "claimed"


@pytest.mark.asyncio
async def test_claim_proposal_already_claimed(repo, proto):
	claims = {"prop-1": {"user": "bob", "claimed_at": "2026-01-01T00:00:00+00:00", "status": "claimed"}}
	(repo / ContributorProtocol.CLAIMS_FILE).write_text(json.dumps(claims))

	with _patch_git():
		result = await proto.claim_proposal("prop-1")

	assert result is False


@pytest.mark.asyncio
async def test_claim_proposal_push_fails(repo, proto):
	(repo / ContributorProtocol.CLAIMS_FILE).write_text("{}")

	async def side_effect(*args, **kwargs):
		cmd_args = args
		if "push" in cmd_args:
			return _make_git_mock(returncode=1, stderr="conflict")
		return _make_git_mock(returncode=0)

	with patch("asyncio.create_subprocess_exec", side_effect=side_effect):
		result = await proto.claim_proposal("prop-1")

	assert result is False


@pytest.mark.asyncio
async def test_claim_abandoned_proposal(repo, proto):
	claims = {"prop-1": {"user": "bob", "claimed_at": "2026-01-01T00:00:00+00:00", "status": "abandoned"}}
	(repo / ContributorProtocol.CLAIMS_FILE).write_text(json.dumps(claims))

	with _patch_git():
		result = await proto.claim_proposal("prop-1")

	assert result is True
	updated = json.loads((repo / ContributorProtocol.CLAIMS_FILE).read_text())
	assert updated["prop-1"]["user"] == "alice"
	assert updated["prop-1"]["status"] == "claimed"


@pytest.mark.asyncio
async def test_publish_result(repo, proto):
	claims = {"prop-1": {"user": "alice", "claimed_at": "2026-01-01T00:00:00+00:00", "status": "claimed"}}
	(repo / ContributorProtocol.CLAIMS_FILE).write_text(json.dumps(claims))

	result = {
		"commit": "abc1234",
		"tests_before": 100,
		"tests_after": 105,
		"outcome": "keep",
		"proposal_title": "Add feature X",
		"duration_s": 300.5,
		"cost_usd": 1.50,
		"timestamp": "2026-03-12T10:00:00Z",
	}

	with _patch_git():
		await proto.publish_result("prop-1", result)

	log_path = repo / ContributorProtocol.EXPERIMENT_LOG
	assert log_path.exists()
	lines = log_path.read_text().strip().split("\n")
	assert len(lines) == 2  # header + 1 row
	assert "abc1234" in lines[1]
	assert "Add feature X" in lines[1]

	updated_claims = json.loads((repo / ContributorProtocol.CLAIMS_FILE).read_text())
	assert updated_claims["prop-1"]["status"] == "completed"


@pytest.mark.asyncio
async def test_sync_learnings(repo, proto):
	content = "# Learnings\n\n- Lesson 1\n- Lesson 2\n"
	(repo / ContributorProtocol.LEARNINGS_FILE).write_text(content)

	with _patch_git():
		result = await proto.sync_learnings()

	assert result == content


@pytest.mark.asyncio
async def test_sync_learnings_no_file(repo, proto):
	with _patch_git():
		result = await proto.sync_learnings()

	assert result == ""


@pytest.mark.asyncio
async def test_list_proposals_empty(repo, proto):
	(repo / ContributorProtocol.CLAIMS_FILE).write_text("{}")

	with _patch_git():
		result = await proto.list_proposals()

	assert result == []


@pytest.mark.asyncio
async def test_list_proposals_no_file(repo, proto):
	with _patch_git():
		result = await proto.list_proposals()

	assert result == []


@pytest.mark.asyncio
async def test_list_proposals_filters_claimed(repo, proto):
	claims = {
		"prop-1": {"user": "bob", "claimed_at": "2026-01-01T00:00:00+00:00", "status": "claimed"},
		"prop-2": {"user": "carol", "claimed_at": "2026-01-01T00:00:00+00:00", "status": "abandoned"},
		"prop-3": {"user": "dave", "claimed_at": "2026-01-01T00:00:00+00:00", "status": "completed"},
	}
	(repo / ContributorProtocol.CLAIMS_FILE).write_text(json.dumps(claims))

	with _patch_git():
		result = await proto.list_proposals()

	assert len(result) == 1
	assert result[0]["proposal_id"] == "prop-2"
