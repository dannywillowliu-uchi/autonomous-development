"""Tests for token parser, cost calculation, and pricing config."""

from __future__ import annotations

import json
from pathlib import Path

from mission_control.config import PricingConfig, load_config
from mission_control.token_parser import TokenUsage, compute_token_cost, parse_stream_json


class TestParseStreamJson:
	def test_empty_input(self) -> None:
		result = parse_stream_json("")
		assert result.usage.input_tokens == 0
		assert result.usage.output_tokens == 0
		assert result.text_content == ""
		assert result.mc_result is None

	def test_whitespace_only(self) -> None:
		result = parse_stream_json("  \n  \n  ")
		assert result.usage.input_tokens == 0
		assert result.text_content == ""

	def test_single_result_event(self) -> None:
		event = {
			"type": "result",
			"usage": {
				"input_tokens": 1000,
				"output_tokens": 500,
				"cache_creation_input_tokens": 200,
				"cache_read_input_tokens": 100,
			},
			"content": [
				{"type": "text", "text": "Hello world"},
			],
		}
		result = parse_stream_json(json.dumps(event))
		assert result.usage.input_tokens == 1000
		assert result.usage.output_tokens == 500
		assert result.usage.cache_creation_tokens == 200
		assert result.usage.cache_read_tokens == 100
		assert result.text_content == "Hello world"

	def test_multi_event_accumulation(self) -> None:
		events = [
			{
				"type": "assistant",
				"message": {
					"usage": {"input_tokens": 500, "output_tokens": 200},
					"content": [{"type": "text", "text": "Part 1. "}],
				},
			},
			{
				"type": "result",
				"usage": {"input_tokens": 600, "output_tokens": 300},
				"content": [{"type": "text", "text": "Part 2."}],
			},
		]
		ndjson = "\n".join(json.dumps(e) for e in events)
		result = parse_stream_json(ndjson)
		assert result.usage.input_tokens == 1100  # 500 + 600
		assert result.usage.output_tokens == 500  # 200 + 300
		assert result.text_content == "Part 1. Part 2."

	def test_mc_result_extraction(self) -> None:
		event = {
			"type": "result",
			"usage": {"input_tokens": 100, "output_tokens": 50},
			"content": [
				{
					"type": "text",
					"text": 'Done.\nMC_RESULT:{"status":"completed","summary":"all good","commits":["abc123"]}',
				},
			],
		}
		result = parse_stream_json(json.dumps(event))
		assert result.mc_result is not None
		assert result.mc_result["status"] == "completed"
		assert result.mc_result["summary"] == "all good"

	def test_malformed_lines_skipped(self) -> None:
		ndjson = "not json at all\n{invalid json}\n" + json.dumps({
			"type": "result",
			"usage": {"input_tokens": 42, "output_tokens": 7},
			"content": [],
		})
		result = parse_stream_json(ndjson)
		assert result.usage.input_tokens == 42
		assert result.usage.output_tokens == 7

	def test_content_block_delta(self) -> None:
		events = [
			{"type": "content_block_delta", "delta": {"type": "text_delta", "text": "chunk1"}},
			{"type": "content_block_delta", "delta": {"type": "text_delta", "text": "chunk2"}},
		]
		ndjson = "\n".join(json.dumps(e) for e in events)
		result = parse_stream_json(ndjson)
		assert result.text_content == "chunk1chunk2"

	def test_no_mc_result_returns_none(self) -> None:
		event = {
			"type": "result",
			"usage": {"input_tokens": 10, "output_tokens": 5},
			"content": [{"type": "text", "text": "Just some output with no marker"}],
		}
		result = parse_stream_json(json.dumps(event))
		assert result.mc_result is None


class TestComputeTokenCost:
	def test_zero_usage(self) -> None:
		usage = TokenUsage()
		pricing = PricingConfig()
		assert compute_token_cost(usage, pricing) == 0.0

	def test_known_pricing(self) -> None:
		usage = TokenUsage(
			input_tokens=1_000_000,
			output_tokens=1_000_000,
			cache_creation_tokens=1_000_000,
			cache_read_tokens=1_000_000,
		)
		pricing = PricingConfig(
			input_per_million=3.0,
			output_per_million=15.0,
			cache_write_per_million=3.75,
			cache_read_per_million=0.30,
		)
		cost = compute_token_cost(usage, pricing)
		expected = 3.0 + 15.0 + 3.75 + 0.30
		assert abs(cost - expected) < 0.001

	def test_partial_usage(self) -> None:
		usage = TokenUsage(input_tokens=500_000, output_tokens=100_000)
		pricing = PricingConfig()  # defaults: 3.0, 15.0
		cost = compute_token_cost(usage, pricing)
		expected = (500_000 * 3.0 / 1_000_000) + (100_000 * 15.0 / 1_000_000)
		assert abs(cost - expected) < 0.001


class TestPricingConfig:
	def test_toml_parsing(self, tmp_path: Path) -> None:
		toml = tmp_path / "mission-control.toml"
		toml.write_text("""\
[target]
name = "test"
path = "/tmp/test"

[pricing]
input_per_million = 6.0
output_per_million = 30.0
cache_write_per_million = 7.5
cache_read_per_million = 0.6
""")
		config = load_config(toml)
		assert config.pricing.input_per_million == 6.0
		assert config.pricing.output_per_million == 30.0
		assert config.pricing.cache_write_per_million == 7.5
		assert config.pricing.cache_read_per_million == 0.6


class TestTokenColumnMigration:
	def test_insert_and_read_tokens(self) -> None:
		"""Token fields survive insert/read round-trip."""
		from mission_control.db import Database
		from mission_control.models import Plan, WorkUnit
		db = Database(":memory:")
		plan = Plan(objective="test")
		db.insert_plan(plan)
		wu = WorkUnit(plan_id=plan.id, input_tokens=1234, output_tokens=567, cost_usd=0.42)
		db.insert_work_unit(wu)
		loaded = db.get_work_unit(wu.id)
		assert loaded is not None
		assert loaded.input_tokens == 1234
		assert loaded.output_tokens == 567
		assert abs(loaded.cost_usd - 0.42) < 0.001
		db.close()

	def test_update_tokens(self) -> None:
		"""Token fields are updated correctly."""
		from mission_control.db import Database
		from mission_control.models import Plan, WorkUnit
		db = Database(":memory:")
		plan = Plan(objective="test")
		db.insert_plan(plan)
		wu = WorkUnit(plan_id=plan.id)
		db.insert_work_unit(wu)
		wu.input_tokens = 999
		wu.output_tokens = 111
		wu.cost_usd = 1.23
		db.update_work_unit(wu)
		loaded = db.get_work_unit(wu.id)
		assert loaded is not None
		assert loaded.input_tokens == 999
		assert loaded.output_tokens == 111
		assert abs(loaded.cost_usd - 1.23) < 0.001
		db.close()

	def test_unit_event_tokens_round_trip(self) -> None:
		"""UnitEvent token fields survive insert/read."""
		from mission_control.db import Database
		from mission_control.models import Epoch, Mission, Plan, UnitEvent, WorkUnit
		db = Database(":memory:")
		mission = Mission(objective="test")
		db.insert_mission(mission)
		epoch = Epoch(mission_id=mission.id, number=1)
		db.insert_epoch(epoch)
		plan = Plan(objective="test")
		db.insert_plan(plan)
		wu = WorkUnit(plan_id=plan.id)
		db.insert_work_unit(wu)
		evt = UnitEvent(
			mission_id=mission.id,
			epoch_id=epoch.id,
			work_unit_id=wu.id,
			event_type="completed",
			input_tokens=5000,
			output_tokens=2000,
		)
		db.insert_unit_event(evt)
		events = db.get_unit_events_for_mission(mission.id)
		assert len(events) == 1
		assert events[0].input_tokens == 5000
		assert events[0].output_tokens == 2000
		db.close()


class TestTokenUsageByEpoch:
	def test_aggregation(self) -> None:
		from mission_control.db import Database
		from mission_control.models import Epoch, Mission, WorkUnit
		db = Database(":memory:")
		mission = Mission(objective="test")
		db.insert_mission(mission)
		epoch = Epoch(mission_id=mission.id, number=1)
		db.insert_epoch(epoch)
		# Insert work units with tokens linked to epoch
		from mission_control.models import Plan
		plan = Plan(objective="test")
		db.insert_plan(plan)
		wu1 = WorkUnit(plan_id=plan.id, epoch_id=epoch.id, input_tokens=1000, output_tokens=500)
		wu2 = WorkUnit(plan_id=plan.id, epoch_id=epoch.id, input_tokens=2000, output_tokens=800)
		db.insert_work_unit(wu1)
		db.insert_work_unit(wu2)
		result = db.get_token_usage_by_epoch(mission.id)
		assert len(result) == 1
		assert result[0]["epoch"] == 1
		assert result[0]["input_tokens"] == 3000
		assert result[0]["output_tokens"] == 1300
		db.close()
