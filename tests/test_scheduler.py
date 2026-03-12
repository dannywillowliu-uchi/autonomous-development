"""Tests for AutoUpdateScheduler and CLI daemon flags."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autodev.scheduler import AutoUpdateScheduler


@pytest.fixture
def mock_config():
	return MagicMock()


@pytest.fixture
def mock_db():
	return MagicMock()


@pytest.fixture
def scheduler(mock_config, mock_db):
	return AutoUpdateScheduler(mock_config, mock_db, interval_hours=1.0)


class TestAutoUpdateScheduler:
	@pytest.mark.asyncio
	async def test_run_forever_calls_pipeline(self, scheduler):
		"""run_forever should create and run the pipeline."""
		mock_result = MagicMock(title="test", action="launched")
		mock_pipeline = AsyncMock()
		mock_pipeline.run.return_value = [mock_result]

		async def stop_after_one_cycle(*args, **kwargs):
			scheduler.stop()

		with (
			patch("autodev.scheduler.AutoUpdatePipeline", return_value=mock_pipeline) as mock_cls,
			patch("asyncio.sleep", new_callable=AsyncMock, side_effect=stop_after_one_cycle),
		):
			await scheduler.run_forever()

		mock_cls.assert_called_once_with(scheduler._config, scheduler._db)
		mock_pipeline.run.assert_called_once()

	@pytest.mark.asyncio
	async def test_stop_exits_loop(self, scheduler):
		"""stop() should cause run_forever() to exit after current cycle."""
		mock_pipeline = AsyncMock()
		mock_pipeline.run.return_value = []

		call_count = 0

		async def stop_on_second(*args, **kwargs):
			nonlocal call_count
			call_count += 1
			if call_count >= 1:
				scheduler.stop()

		with (
			patch("autodev.scheduler.AutoUpdatePipeline", return_value=mock_pipeline),
			patch("asyncio.sleep", new_callable=AsyncMock, side_effect=stop_on_second),
		):
			await scheduler.run_forever()

		assert not scheduler._running

	@pytest.mark.asyncio
	async def test_exception_does_not_crash(self, scheduler):
		"""Pipeline exceptions should be logged, not crash the loop."""
		mock_pipeline = AsyncMock()
		mock_pipeline.run.side_effect = RuntimeError("boom")

		async def stop_after_one(*args, **kwargs):
			scheduler.stop()

		with (
			patch("autodev.scheduler.AutoUpdatePipeline", return_value=mock_pipeline),
			patch("asyncio.sleep", new_callable=AsyncMock, side_effect=stop_after_one),
		):
			await scheduler.run_forever()

		mock_pipeline.run.assert_called_once()

	@pytest.mark.asyncio
	async def test_interval_passed_to_sleep(self, mock_config, mock_db):
		"""Sleep duration should match interval_hours * 3600."""
		sched = AutoUpdateScheduler(mock_config, mock_db, interval_hours=2.5)
		mock_pipeline = AsyncMock()
		mock_pipeline.run.return_value = []

		async def stop_after_one(*args, **kwargs):
			sched.stop()

		with (
			patch("autodev.scheduler.AutoUpdatePipeline", return_value=mock_pipeline),
			patch("asyncio.sleep", new_callable=AsyncMock, side_effect=stop_after_one) as mock_sleep,
		):
			await sched.run_forever()

		mock_sleep.assert_called_once_with(2.5 * 3600)

	def test_default_interval(self, mock_config, mock_db):
		"""Default interval should be 24 hours."""
		sched = AutoUpdateScheduler(mock_config, mock_db)
		assert sched._interval_hours == 24.0


class TestCLIDaemonFlags:
	def test_daemon_flag_parsed(self):
		"""--daemon flag should be parsed correctly."""
		from autodev.cli import build_parser

		parser = build_parser()
		args = parser.parse_args(["auto-update", "--daemon"])
		assert args.daemon is True

	def test_daemon_flag_default(self):
		"""--daemon should default to False."""
		from autodev.cli import build_parser

		parser = build_parser()
		args = parser.parse_args(["auto-update"])
		assert args.daemon is False

	def test_interval_flag_parsed(self):
		"""--interval flag should be parsed correctly."""
		from autodev.cli import build_parser

		parser = build_parser()
		args = parser.parse_args(["auto-update", "--interval", "12.0"])
		assert args.interval == 12.0

	def test_interval_flag_default(self):
		"""--interval should default to 24.0."""
		from autodev.cli import build_parser

		parser = build_parser()
		args = parser.parse_args(["auto-update"])
		assert args.interval == 24.0

	def test_daemon_with_interval(self):
		"""--daemon and --interval should work together."""
		from autodev.cli import build_parser

		parser = build_parser()
		args = parser.parse_args(["auto-update", "--daemon", "--interval", "6.0"])
		assert args.daemon is True
		assert args.interval == 6.0
