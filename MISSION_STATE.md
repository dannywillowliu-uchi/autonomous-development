# Mission State
Objective: Implement the top-priority improvements discovered via codebase analysis:

1. DEPENDENCY-AWARE DISPATCH: The dispatch loop in continuous_controller.py dispatches all units in parallel without checking depends_on. Modify the dispatch logic to respect dependency ordering -- units with unresolved dependencies must wait until their blockers complete. Add tests.

2. SMOKE-TEST VERIFICATION BEFORE MERGE: GreenBranchManager.merge_unit() merges directly to mc/green without verification. Add a lightweight smoke-test step (run the verification command) BEFORE merging to mc/green. Only merge if verification passes. Add tests.

3. SANDBOX TOOL SYNTHESIS: tool_synthesis.py writes arbitrary Python scripts from worker output directly to disk with no validation. Add content validation (no imports of dangerous modules, no file system access outside project), and restrict execution to a sandboxed subprocess. Add tests.

Each feature: implement, add tests, ensure all existing tests pass.

## Remaining
The planner should focus on what hasn't been done yet.
Do NOT re-target files in the 'Files Modified' list unless fixing a failure.
