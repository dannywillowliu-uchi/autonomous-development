"""Abstract base class for worker execution backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class WorkerHandle:
	worker_id: str
	pid: int | None = None
	workspace_path: str = ""
	backend_metadata: str = ""


class WorkerBackend(ABC):
	"""Abstract base for worker execution backends."""

	@abstractmethod
	async def provision_workspace(
		self, worker_id: str, source_repo: str, base_branch: str
	) -> str:
		"""Create a workspace for a worker. Returns workspace path."""

	@abstractmethod
	async def spawn(
		self, worker_id: str, workspace_path: str, command: list[str], timeout: int
	) -> WorkerHandle:
		"""Spawn a worker process. Returns a handle for monitoring."""

	@abstractmethod
	async def check_status(self, handle: WorkerHandle) -> str:
		"""Check worker status: running/completed/failed."""

	@abstractmethod
	async def get_output(self, handle: WorkerHandle) -> str:
		"""Get worker stdout output."""

	@abstractmethod
	async def kill(self, handle: WorkerHandle) -> None:
		"""Kill a running worker."""

	@abstractmethod
	async def release_workspace(self, workspace_path: str) -> None:
		"""Release a workspace back to the pool."""

	@abstractmethod
	async def cleanup(self) -> None:
		"""Clean up all resources."""
