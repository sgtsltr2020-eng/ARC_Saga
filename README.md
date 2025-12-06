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

## Quick Start

```bash
# Phase 1-3: API mode
pip install -r requirements.txt
python -m arc_saga.api.server

# Phase 5+: Desktop mode
python -m arc_saga --mode desktop
```

## Architecture

- **Phase 1** (Complete): Foundation, API, monitoring
- **Phase 2** (Current): Orchestrator + your stack MVP
- **Phase 3**: Memory improvements (knowledge graph, reasoning traces)
- **Phase 4**: MCP integration for multi-IDE support
- **Phase 5**: Desktop UI with PyQt6
- **Phase 6**: Team collaboration features

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

- ✅ Phase 1 Complete: 104 tests passing, production-ready
- 🔨 Phase 2 In Progress: Orchestrator foundation
- 📅 Phases 3-6: See ROADMAP.md

## License

[Your License]

## Contributing

See CONTRIBUTING.md

---

**Built for developers who need production-grade AI-generated code with predictable costs.**
