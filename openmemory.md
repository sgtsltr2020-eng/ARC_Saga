# ARC SAGA - OpenMemory Project Index

## Project Overview

**ARC Saga** (Advanced Reasoning and Context - Saga) is an enterprise-grade persistent memory layer for AI conversations with full-text search, auto-tagging, and multi-provider support.

**Current Phase:** Phase 1b (API & Services) - In Progress  
**Quality Standard:** FAANG-level (95%+ coverage, mypy --strict, comprehensive logging)

---

## User Defined Namespaces

- `core` - Core models, exceptions, configuration
- `storage` - Data persistence layer (SQLite, future: PostgreSQL)
- `api` - FastAPI REST endpoints
- `services` - Business logic services (AutoTagger, FileProcessor)
- `integrations` - External provider clients (Perplexity, future: OpenAI)
- `monitors` - Health checks and monitoring (planned)
- `tests` - Test infrastructure

---

## Architecture

### Key Patterns
- **Repository Pattern** - StorageBackend interface for data access
- **Dependency Injection** - Services receive storage as constructor argument
- **Async/Await** - All I/O operations are async
- **FTS5 Full-Text Search** - SQLite virtual tables for fast search

### Data Flow
```
Client → FastAPI → Services → StorageBackend → SQLite
```

### Database Location
- Production: `~/.arc-saga/memory.db`
- Tests: Temporary directory with auto-cleanup

---

## Components Status

### ✅ Complete
- Data Models (Message, File, Provider, MessageRole)
- SQLite Storage with FTS5
- Storage Interface (StorageBackend ABC)
- Custom Exceptions
- JSON Structured Logging
- Shared Configuration
- FastAPI REST API
- AutoTagger Service
- FileProcessor Service
- Error Instrumentation

### 🔄 In Progress
- Perplexity Integration (storage method mismatch)
- Test Coverage (currently ~80%, target 95%)
- Integration Tests

### ⏳ Planned
- Circuit Breaker Pattern
- Rate Limiting
- Vector/Semantic Search
- Event Sourcing
- CQRS Pattern
- OAuth Authentication

---

## Critical Files

| Purpose | File |
|---------|------|
| Data Models | `arc_saga/models/message.py` |
| Storage Interface | `arc_saga/storage/base.py` |
| SQLite Implementation | `arc_saga/storage/sqlite.py` |
| API Server | `arc_saga/api/server.py` |
| Configuration | `shared/config.py` |
| Cursor Rules | `.cursorrules` |

---

## Known Issues

1. **Perplexity client** uses `store_message` instead of `save_message` - Location: `arc_saga/integrations/perplexity_client.py`
2. **Perplexity client** uses `thread_id` field instead of `session_id` - Location: `arc_saga/integrations/perplexity_client.py`
3. **Duplicate test files** in `tests/` root and `tests/unit/` directories
4. **File storage** not fully integrated with database - Location: `arc_saga/api/server.py` `attach_file()` endpoint

---

## Commands

```bash
# Start server
python -m arc_saga.api.server

# Run tests
pytest --cov=arc_saga

# Type check
mypy arc_saga --strict

# System audit
python audit_system.py
```

---

## Documentation

- `docs/AGENT_ONBOARDING.md` - Complete agent onboarding guide
- `docs/arc_saga_master_index.md` - System overview
- `docs/decision_catalog.md` - Architectural decisions
- `docs/error_catalog.md` - Error solutions
- `docs/prompts_library.md` - Cursor prompts
- `docs/verification_checklist.md` - Quality gates
