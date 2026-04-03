It seems the file write is being blocked by permissions. Let me output the spec directly instead -- the full spec is ready. Here's a summary of the 9 changes:

**Target files**: `swarm/prompts.py`, `swarm/worker_prompt.py`, `specialist_templates/*.md`

### Changes Overview

| # | File | Change | Pattern Applied |
|---|------|--------|----------------|
| 1 | `swarm/prompts.py` SYSTEM_PROMPT | Add orchestrator-workers framing, delegation quality heuristics, result evaluation checklist | Orchestrator pattern |
| 2 | `swarm/prompts.py` CYCLE_PROMPT | Replace loose "Consider:" with 5-step reasoning chain (What changed -> Evaluate -> Bottleneck -> Anti-patterns -> Decide) | Prompt chaining |
| 3 | `swarm/worker_prompt.py` | New `_self_evaluation_section()` -- 5-point pre-completion checklist | Self-correction |
| 4 | `swarm/worker_prompt.py` | New `_error_recovery_section()` -- diagnose-before-fix, track attempts, report blocked after 3 tries | Error recovery |
| 5 | `swarm/worker_prompt.py` | Add good/bad examples to `_inbox_section()` and `_result_protocol_section()` | Tool use clarity |
| 6 | `specialist_templates/*.md` | Add numbered `## Workflow` sections to all 4 templates (debugger, test-writer, refactorer, simplifier) | Sequential steps |
| 7 | `swarm/prompts.py` INITIAL_PLANNING | Add task scoping guidance (1-3 files, WHAT/WHERE/WHY/DONE WHEN checklist) | Task decomposition |
| 8 | `specialist_templates/researcher.md` | New template -- scope, survey, hypothesize, validate, synthesize, report | Research-before-implement |
| 9 | `swarm/prompts.py` ANALYSIS_PROMPT | Add delegation quality review + output quality check to analysis schema | Evaluation criteria |

### Key design decisions

- **All changes are prompt-only** -- no logic changes to planner.py, controller.py, or context.py. Pure prompt engineering, minimal blast radius.
- **Additive, not rewriting** -- existing heuristics, decision types, and examples are preserved. New sections are appended.
- **~1500 tokens added to planner, ~800 to worker** -- well within budget, with a test to guard against future bloat.

Should I write the file with different permissions, or would you like me to proceed directly to implementation?