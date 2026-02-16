# Mission State
Objective: Replace the LLM-based evaluator with deterministic objective signals, and add N-of-M candidate selection for fixup. Both changes improve mission reliability and reduce cost.

1. DETERMINISTIC EVALUATOR: Rewrite evaluator.py to replace the LLM call with a deterministic scoring function. The current evaluator spawns a full Claude session per round to assign a subjective 0.0-1.0 score -- this is expensive and noisy. Replace with: score = 0.4 * test_improvement + 0.2 * lint_improvement + 0.2 * completion_rate + 0.2 * no_regression. Each component is 0.0-1.0 based on objective deltas from SnapshotDelta. test_improvement = (tests_after - tests_before) / max(tests_before, 1), clamped to [0,1]. lint_improvement = max(0, (errors_before - errors_after) / max(errors_before, 1)). completion_rate = units_completed / units_planned. no_regression = 1.0 if no test regressions else 0.0. Remove the LLM call entirely. Our own feedback.py already computes objective rewards -- use the same pattern.

2. SIMPLIFY ROUND CONTROLLER: Update round_controller.py to use the new deterministic evaluator. Remove any async LLM evaluation logic. The evaluate step should be a simple synchronous function call now. Update tests/test_evaluator.py with comprehensive tests for the new scoring function -- edge cases like zero tests, all tests failing, partial improvements.

3. N-OF-M FIXUP CANDIDATES: In green_branch.py run_fixup(), instead of one attempt, spawn N candidate fix patches (N=3 default, configurable via fixup_candidates in config.py). Each candidate runs on a temporary branch with a slightly different prompt (vary the approach: "fix the failing test by modifying the implementation", "fix by adjusting the test expectations", "fix by refactoring the surrounding code"). Run verification on each candidate. Select the best-scoring one (most tests passing, fewest lint errors). If multiple pass, pick the one with the smallest diff. Merge the winner.

4. FIXUP CONFIG AND TESTS: Add fixup_candidates field to config.py (default 3). Add comprehensive tests for the N-of-M selection logic -- mock multiple candidates, verify the best one is selected. Test edge cases: all candidates fail, only one passes, tie-breaking by diff size.

Each unit: implement the feature, add tests, ensure all existing tests pass. Read BACKLOG.md for the full spec (P1 entries).

## Completed
- [x] c8848f9f (2026-02-16T16:01:54.604392+00:00) -- Added fixup_candidates: int = 3 field to GreenBranchConfig, updated _build_green_branch() to parse i (files: src/mission_control/config.py, tests/test_config.py)

## Files Modified
src/mission_control/config.py, tests/test_config.py

## Remaining
The planner should focus on what hasn't been done yet.
Do NOT re-target files in the 'Files Modified' list unless fixing a failure.

## Changelog
- 2026-02-16T16:01:54.604392+00:00 | c8848f9f merged (commit: 0c48d4b) -- Added fixup_candidates: int = 3 field to GreenBranchConfig, updated _build_green
