"""Container backend -- stub for future Docker/Podman support."""

from __future__ import annotations

from mission_control.backends.base import WorkerBackend, WorkerHandle


class ContainerBackend(WorkerBackend):
	"""Container-based worker execution (not yet implemented)."""

	async def provision_workspace(
		self, worker_id: str, source_repo: str, base_branch: str
	) -> str:
		raise NotImplementedError("Container backend not yet implemented")

	async def spawn(
		self, worker_id: str, workspace_path: str, command: list[str], timeout: int
	) -> WorkerHandle:
		raise NotImplementedError("Container backend not yet implemented")

	async def check_status(self, handle: WorkerHandle) -> str:
		raise NotImplementedError("Container backend not yet implemented")

	async def get_output(self, handle: WorkerHandle) -> str:
		raise NotImplementedError("Container backend not yet implemented")

	async def kill(self, handle: WorkerHandle) -> None:
		raise NotImplementedError("Container backend not yet implemented")

	async def release_workspace(self, workspace_path: str) -> None:
		raise NotImplementedError("Container backend not yet implemented")

	async def cleanup(self) -> None:
		raise NotImplementedError("Container backend not yet implemented")
