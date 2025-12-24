# ARC SAGA

**AI-Powered Code Quality Enforcement with Intelligent Memory Management**

## Overview

ARC SAGA is a native desktop application that ensures AI coding assistants generate production-ready code while optimizing token usage through configurable memory tiers.

## Key Features

- **Quality Enforcement**: Automatic mypy, pytest, pylint, bandit verification
- **Token Budget Dashboard**: Know costs before spending
- **Tiered Memory**: 5 configurable levels (Minimal to Unlimited)
- **Intelligent Orchestrator**: Coordinates multiple AI agents
- **Desktop-Native**: Local-first, works offline, professional UX
- **Cross-IDE Compatible**: Cursor now, VS Code + more via MCP (Phase 4)
- **Workflow Persistence**: Resume work after crashes or restarts

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# API mode (Phase 1-3)
python -m arc_saga.api.server

# Desktop mode (Phase 5+)
python -m arc_saga --mode desktop
```

## Resume Workflow Example

SAGA persists workflow state, allowing you to resume after crashes:

```python
from saga.core.warden import Warden

async def main():
    # Initialize Warden with project root
    warden = Warden(project_root="/path/to/project")
    await warden.initialize()

    # Resume an interrupted workflow
    graph = await warden.resume_work("req-auth-feature-001")

    if graph:
        print(f"Resuming {len(graph.get_all_tasks())} tasks...")

        # Get tasks ready to execute
        for task in graph.get_ready_tasks():
            print(f"Executing: {task.description}")
            await warden.execute_task(task)
    else:
        print("No saved workflow found for this request.")
```

## Architecture

- **Phase 1** (✅ Complete): Foundation, API, monitoring
- **Phase 2** (✅ Complete): Orchestrator + SAGA Constitution
- **Phase 3** (✅ Complete): Agent Execution Framework, LoreBook
- **Phase 4** (✅ Complete): Workflow Persistence, TaskStore
- **Phase 5**: Desktop UI with PyQt6
- **Phase 6**: Team collaboration features

## Performance Benchmarks

| Operation      | Size           | Time  | Threshold |
| -------------- | -------------- | ----- | --------- |
| Save TaskGraph | 100 tasks      | <1.5s | <5s       |
| Load TaskGraph | 100 tasks      | <0.5s | <2s       |
| Batch Updates  | 50 tasks       | <1.0s | <3s       |
| Pending Query  | 100 tasks      | <0.1s | <0.5s     |
| Vector Search  | 1000 decisions | <50ms | <100ms    |

## Test Coverage

```
182 tests passing
Coverage: 75.73%
```

| Component               | Coverage |
| ----------------------- | -------- |
| `task_store.py`         | 92.21%   |
| `task_verifier.py`      | 89.47%   |
| `injection_detector.py` | 90.48%   |
| `secrets_scanner.py`    | 86.21%   |

## Documentation

- `docs/arc_saga_master_index.md` - Complete system guide
- `docs/decision_catalog.md` - Architecture decisions
- `docs/prompts_library.md` - Token-optimized prompts
- `docs/ROADMAP.md` - Development plan
- `.cursorrules` - Quality enforcement rules

## Token Optimization

Memory tiers adjust context based on task:

- **Minimal** (~2k tokens): Simple refactoring
- **Standard** (~5k tokens): Most features
- **Enhanced** (~8k tokens): Complex debugging
- **Complete** (~10k tokens): Full context
- **Unlimited** (~15k tokens): Learning/research

## Quality Standards

Every generated code must pass:

- `mypy --strict` (type safety)
- `pytest` with 95%+ coverage
- `pylint` >= 8.0
- `bandit` 0 security issues

## Status

- ✅ Phase 1 Complete: Foundation & API
- ✅ Phase 2 Complete: Orchestrator & SAGA Constitution
- ✅ Phase 3 Complete: Agent Execution & LoreBook
- ✅ Phase 4 Complete: Workflow Persistence (182 tests passing)
- 📅 Phase 5: Desktop UI
- 📅 Phase 6: Team Collaboration

## License

[Your License]

## Contributing

See CONTRIBUTING.md

---

**Built for developers who need production-grade AI-generated code with predictable costs.**
