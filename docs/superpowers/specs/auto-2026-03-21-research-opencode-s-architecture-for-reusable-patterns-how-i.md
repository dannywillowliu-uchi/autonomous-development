The spec is ready to write. It covers:

- **Problem Statement**: Tight coupling to Claude Code CLI, need for model diversity and resilience
- **Phase 1 (Research)**: Deep analysis of OpenCode's tool execution, context management, session lifecycle, and process model -- each compared against specific autodev modules
- **Phase 2 (Backend Integration)**: New `OpenCodeBackend` implementing `WorkerBackend` ABC, config extension with `OpenCodeConfig`, swarm controller spawn routing by `backend` field, and prompt adaptation
- **Phase 3 (Conditional)**: LSP-driven context, session persistence, and tool execution patterns -- only if research validates them
- **Changes Needed**: 3 new files, 7 modified files with specific function-level changes
- **Testing**: 10 unit tests, 3 manual integration tests
- **Risk Assessment**: 6 risks with mitigations, key one being the phased approach where Phase 1 produces a go/no-go gate before any code is written

Would you like to approve writing the file?