"""Regression tests: no direct self.db.conn.execute() outside db.py.

All SQL execution must go through db.py's locked_call() to avoid
data races on the single sqlite3 connection under asyncio concurrency.
"""

import ast
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src" / "mission_control"

# Files with known pre-existing violations that are tracked separately.
# Remove entries as they get migrated to locked_call().
ALLOWLIST = {
	"auto_discovery.py",
}


def _find_conn_execute_calls(filepath: Path) -> list[int]:
	"""Parse *filepath* with ast and return line numbers of self.db.conn.execute/executemany calls.

	Matches attribute chains like:
		self.db.conn.execute(...)
		self.db.conn.executemany(...)
	"""
	source = filepath.read_text()
	try:
		tree = ast.parse(source, filename=str(filepath))
	except SyntaxError:
		return []

	violations: list[int] = []
	for node in ast.walk(tree):
		if not isinstance(node, ast.Call):
			continue
		func = node.func
		# Match: self.db.conn.execute / self.db.conn.executemany
		if not isinstance(func, ast.Attribute):
			continue
		if func.attr not in ("execute", "executemany"):
			continue
		# func.value should be self.db.conn (Attribute chain)
		conn = func.value
		if not isinstance(conn, ast.Attribute) or conn.attr != "conn":
			continue
		db = conn.value
		if not isinstance(db, ast.Attribute) or db.attr != "db":
			continue
		self_node = db.value
		if not isinstance(self_node, ast.Name) or self_node.id != "self":
			continue
		violations.append(node.lineno)
	return violations


def test_no_direct_conn_execute_outside_db():
	"""No file outside db.py should call self.db.conn.execute() directly."""
	py_files = sorted(SRC_DIR.glob("**/*.py"))
	assert py_files, f"No .py files found under {SRC_DIR}"

	all_violations: list[str] = []
	for filepath in py_files:
		if filepath.name == "db.py":
			continue
		if filepath.name in ALLOWLIST:
			continue
		lines = _find_conn_execute_calls(filepath)
		for lineno in lines:
			rel = filepath.relative_to(SRC_DIR)
			all_violations.append(f"{rel}:{lineno}")

	assert not all_violations, (
		"Direct self.db.conn.execute() calls found outside db.py "
		"(use db.locked_call() instead):\n  " + "\n  ".join(all_violations)
	)


def test_continuous_controller_has_zero_conn_execute():
	"""Regression for commit 17e4398: continuous_controller.py must have zero direct conn.execute calls."""
	cc_path = SRC_DIR / "continuous_controller.py"
	assert cc_path.exists(), f"{cc_path} not found"

	violations = _find_conn_execute_calls(cc_path)
	assert not violations, (
		f"continuous_controller.py still has direct self.db.conn.execute() at lines {violations} "
		"(regression of commit 17e4398 fix)"
	)
