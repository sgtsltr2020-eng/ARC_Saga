# ARC SAGA - Complete Agent Onboarding Guide

## 🎯 Executive Summary for Steering Agent

**Last Updated:** 2025-12-02  
**Document Purpose:** Comprehensive onboarding for development-steering agents  
**Version:** 1.0.0  

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Vision and Goals](#2-vision-and-goals)
3. [System Architecture](#3-system-architecture)
4. [Current Implementation Status](#4-current-implementation-status)
5. [Component Deep Dive](#5-component-deep-dive)
6. [Development Conventions](#6-development-conventions)
7. [Testing Infrastructure](#7-testing-infrastructure)
8. [Planned Features & Roadmap](#8-planned-features--roadmap)
   - 8.1 Feature Ideas Backlog (living)
9. [Known Issues & Technical Debt](#9-known-issues--technical-debt)
10. [Critical Files Reference](#10-critical-files-reference)
11. [Development Workflows](#11-development-workflows)
12. [Integration Points](#12-integration-points)

---

## 1. Project Overview

### What is ARC Saga?

ARC Saga (Advanced Reasoning and Context - Saga) is an **enterprise-grade persistent memory layer for AI conversations**. It enables multi-provider AI assistants (Perplexity, OpenAI, Anthropic, Google, Copilot, etc.) to maintain context across sessions, search historical conversations, and share knowledge.

### Core Value Proposition

1. **Unified Memory Layer**: Single source of truth for all AI conversations
2. **Full-Text Search**: FTS5-powered search across all stored content
3. **Auto-Tagging**: TF-IDF based automatic keyword extraction
4. **Multi-Provider Support**: Works with Perplexity, OpenAI, Anthropic, Google, Groq, Antigravity
5. **File Processing**: Extract and index content from PDFs, DOCX, code files
6. **Session Grouping**: Thread-based conversation organization
7. **API-First Design**: FastAPI REST API for integration

### Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **API** | FastAPI 0.104.1 | REST API server |
| **Database** | SQLite + FTS5 | Persistent storage with full-text search |
| **Validation** | Pydantic 2.5.0 | Data validation and serialization |
| **Server** | Uvicorn 0.24.0 | ASGI server |
| **File Processing** | PyMuPDF, python-docx | Text extraction |
| **Auto-Tagging** | scikit-learn TF-IDF | Keyword extraction |
| **Monitoring** | watchdog | File system monitoring |
| **Logging** | Python logging + JSON | Structured logging |

---

## 2. Vision and Goals

### Project Mission

Create a world-class, FAANG-quality persistent memory system that:
- Never loses conversation context
- Makes searching past conversations trivial
- Enables AI assistants to learn from historical interactions
- Provides enterprise-grade reliability and security

### Quality Standards (Non-Negotiable)

| Standard | Target | Current Status |
|----------|--------|----------------|
| Type Safety | mypy --strict | ✅ Implemented |
| Test Coverage | 95%+ | 🔄 ~80% (improving) |
| Error Handling | Complete | ✅ Implemented |
| Logging | Comprehensive JSON | ✅ Implemented |
| Security | 0 vulnerabilities | ✅ Passing |
| Performance | Benchmarks met | 🔄 Needs verification |

### Design Principles

1. **Repository Pattern** - All data access through abstract interfaces
2. **Event-Driven CQRS** - Planned for Phase 2 (audit trails, consistency)
3. **Circuit Breaker** - For external service calls
4. **Retry with Exponential Backoff** - For transient failures
5. **Dependency Injection** - For testability and flexibility

---

## 3. System Architecture

### Directory Structure

```
ARC_Saga/
├── arc_saga/                    # Main application package
│   ├── __init__.py             # Package initialization
│   ├── api/                    # FastAPI REST API
│   │   ├── __init__.py
│   │   └── server.py           # Main API server (8421 port)
│   ├── core/                   # Core business logic (placeholder)
│   │   └── __init__.py
│   ├── exceptions/             # Custom exception classes
│   │   ├── __init__.py
│   │   └── storage_exceptions.py
│   ├── integrations/           # External provider integrations
│   │   ├── __init__.py
│   │   └── perplexity_client.py
│   ├── models/                 # Data models (Pydantic + dataclasses)
│   │   ├── __init__.py
│   │   └── message.py
│   ├── monitors/               # Monitoring services (placeholder)
│   │   └── __init__.py
│   ├── services/               # Business services
│   │   ├── __init__.py
│   │   ├── auto_tagger.py      # TF-IDF keyword extraction
│   │   └── file_processor.py   # File text extraction
│   ├── storage/                # Data persistence layer
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract StorageBackend interface
│   │   └── sqlite.py           # SQLite implementation
│   ├── error_instrumentation.py # Comprehensive error tracking
│   └── logging_config.py       # JSON structured logging
├── shared/                     # Shared configuration
│   ├── __init__.py
│   └── config.py               # SharedConfig class
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── integration/            # Integration tests
│   │   └── __init__.py
│   ├── unit/                   # Unit tests
│   │   ├── __init__.py
│   │   ├── test_models.py
│   │   └── test_storage.py
│   ├── test_models.py          # Root-level model tests
│   └── test_storage.py         # Root-level storage tests
├── docs/                       # Documentation
│   ├── arc_saga_master_index.md
│   ├── architecture_decision_records/
│   ├── cursorrules-master-reference.md
│   ├── decision_catalog.md
│   ├── deployment_guide.md
│   ├── error_catalog.md
│   ├── prompts_library.md
│   ├── system-delivered.md
│   └── verification_checklist.md
├── .cursorrules               # Cursor AI configuration
├── requirements.txt           # Python dependencies
├── setup.py                   # Package installation
├── audit_system.py            # System health audit tool
├── diagnose.py                # Diagnostic utilities
└── test_server.py             # Integration test runner
```

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          CLIENT LAYER                                │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐        │
│  │ Perplexity│  │   Copilot │  │ VSCode Ext│  │    CLI    │        │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘        │
└────────┼──────────────┼──────────────┼──────────────┼───────────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       FastAPI REST API (:8421)                       │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │/capture │ │/search  │ │/context │ │/thread  │ │/health  │       │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘       │
└───────┼───────────┼───────────┼───────────┼───────────┼─────────────┘
        │           │           │           │           │
        ▼           ▼           ▼           ▼           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         SERVICE LAYER                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐         │
│  │  AutoTagger │  │FileProcessor│  │PerplexityClient     │         │
│  │  (TF-IDF)   │  │ (PDF/DOCX)  │  │  (Streaming API)    │         │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘         │
└─────────┼────────────────┼────────────────────┼─────────────────────┘
          │                │                    │
          ▼                ▼                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        STORAGE LAYER                                 │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │              StorageBackend (Abstract Interface)          │      │
│  └────────────────────────────┬─────────────────────────────┘      │
│                               │                                     │
│  ┌────────────────────────────┼────────────────────────────┐       │
│  │              SQLiteStorage Implementation                │       │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │       │
│  │  │  messages   │  │   files     │  │messages_fts │     │       │
│  │  │   table     │  │   table     │  │ (FTS5 index)│     │       │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │       │
│  └──────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PERSISTENCE LAYER                                │
│                ~/.arc-saga/memory.db (SQLite)                        │
└─────────────────────────────────────────────────────────────────────┘
```

### Database Schema

#### Messages Table
```sql
CREATE TABLE messages (
    id TEXT PRIMARY KEY,           -- UUID
    provider TEXT NOT NULL,        -- openai, anthropic, perplexity, etc.
    role TEXT NOT NULL,            -- user, assistant, system
    content TEXT NOT NULL,         -- Message content (max 100KB)
    tags TEXT,                     -- JSON array of tags
    timestamp DATETIME NOT NULL,   -- ISO 8601 timestamp
    metadata TEXT,                 -- JSON object for extensibility
    session_id TEXT,               -- Thread/session grouping
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_messages_provider ON messages(provider);
CREATE INDEX idx_messages_timestamp ON messages(timestamp DESC);
CREATE INDEX idx_messages_session ON messages(session_id);
```

#### Files Table
```sql
CREATE TABLE files (
    id TEXT PRIMARY KEY,           -- UUID
    filename TEXT NOT NULL,        -- Original filename
    filepath TEXT NOT NULL,        -- Storage path
    file_type TEXT NOT NULL,       -- pdf, docx, code, image, etc.
    extracted_text TEXT,           -- Extracted text content
    tags TEXT,                     -- JSON array of tags
    file_size INTEGER,             -- Size in bytes (max 100MB)
    uploaded_at DATETIME NOT NULL, -- Upload timestamp
    metadata TEXT,                 -- JSON object
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_files_type ON files(file_type);
CREATE INDEX idx_files_uploaded ON files(uploaded_at DESC);
```

#### FTS5 Virtual Tables
```sql
-- Full-text search for messages
CREATE VIRTUAL TABLE messages_fts USING fts5(
    id UNINDEXED,
    content,
    tags,
    content='messages',
    content_rowid='rowid'
);

-- Full-text search for files
CREATE VIRTUAL TABLE files_fts USING fts5(
    id UNINDEXED,
    filename,
    extracted_text,
    tags,
    content='files',
    content_rowid='rowid'
);
```

---

## 4. Current Implementation Status

### Phase 1a: Foundation ✅ COMPLETE

| Component | Status | Location | Description |
|-----------|--------|----------|-------------|
| Data Models | ✅ Complete | `arc_saga/models/message.py` | Message, File, SearchResult, ValidationResult |
| SQLite Storage | ✅ Complete | `arc_saga/storage/sqlite.py` | Full CRUD + FTS5 search |
| Storage Interface | ✅ Complete | `arc_saga/storage/base.py` | Abstract StorageBackend |
| Exceptions | ✅ Complete | `arc_saga/exceptions/` | StorageError, ValidationError, etc. |
| Logging | ✅ Complete | `arc_saga/logging_config.py` | JSON structured logging |
| Shared Config | ✅ Complete | `shared/config.py` | SharedConfig class |
| Unit Tests | ✅ Complete | `tests/unit/` | Model and storage tests |

### Phase 1b: API & Services 🔄 IN PROGRESS

| Component | Status | Location | Description |
|-----------|--------|----------|-------------|
| FastAPI Server | ✅ Complete | `arc_saga/api/server.py` | REST API on port 8421 |
| AutoTagger | ✅ Complete | `arc_saga/services/auto_tagger.py` | TF-IDF keyword extraction |
| FileProcessor | ✅ Complete | `arc_saga/services/file_processor.py` | PDF/DOCX text extraction |
| Perplexity Client | 🔄 Partial | `arc_saga/integrations/perplexity_client.py` | Needs storage method alignment |
| Error Instrumentation | ✅ Complete | `arc_saga/error_instrumentation.py` | Comprehensive error tracking |
| CORS Configuration | ✅ Complete | `arc_saga/api/server.py` | VSCode extension support |

### Phase 1c: Monitoring & Validators ⏳ NOT STARTED

| Component | Status | Location | Description |
|-----------|--------|----------|-------------|
| Monitor Services | ⏳ Planned | `arc_saga/monitors/` | Health checks, metrics |
| Validator Integration | ⏳ Planned | TBD | AI config validation |
| Circuit Breaker | ⏳ Planned | TBD | External call resilience |
| Rate Limiting | ⏳ Planned | TBD | API rate limits |

### Phase 2: Advanced Features ⏳ NOT STARTED

| Component | Status | Description |
|-----------|--------|-------------|
| Event Sourcing | ⏳ Planned | Complete audit trail |
| CQRS Implementation | ⏳ Planned | Separated read/write models |
| Vector Search | ⏳ Planned | Semantic similarity search |
| Multi-Agent Sync | ⏳ Planned | Cross-agent memory sharing |
| OAuth Authentication | ⏳ Planned | Google/GitHub/Microsoft SSO |

---

## 5. Component Deep Dive

### 5.1 Data Models (`arc_saga/models/message.py`)

#### Enums

```python
class Provider(str, Enum):
    """Supported AI providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    ANTIGRAVITY = "antigravity"
    PERPLEXITY = "perplexity"
    GROQ = "groq"

class MessageRole(str, Enum):
    """Message sender role."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class FileType(str, Enum):
    """Supported file types."""
    PDF = "pdf"
    DOCX = "docx"
    CODE = "code"
    IMAGE = "image"
    MARKDOWN = "markdown"
    TEXT = "text"
    DOCUMENT = "document"
```

#### Message Dataclass

```python
@dataclass
class Message:
    """Represents a single message in a conversation."""
    provider: Provider              # Required: AI provider
    role: MessageRole               # Required: user/assistant/system
    content: str                    # Required: Message text (max 100KB)
    tags: list[str]                 # Default: []
    id: str                         # Default: UUID4
    timestamp: datetime             # Default: UTC now
    metadata: dict[str, Any]        # Default: {}
    session_id: Optional[str]       # Default: None

    # Validation: Empty content raises ValueError
    # Validation: Content > 100KB raises ValueError
```

#### Key Validation Rules

1. **Message content** cannot be empty or whitespace-only
2. **Message content** cannot exceed 100,000 characters (100KB)
3. **File filename** cannot be empty
4. **File size** cannot exceed 100MB (100,000,000 bytes)

### 5.2 Storage Layer (`arc_saga/storage/`)

#### Abstract Interface (`base.py`)

```python
class StorageBackend(ABC):
    """Abstract interface for data persistence."""
    
    @abstractmethod
    async def initialize(self) -> None: ...
    
    @abstractmethod
    async def save_message(self, message: Message) -> str: ...
    
    @abstractmethod
    async def save_file(self, file: File) -> str: ...
    
    @abstractmethod
    async def search_messages(
        self, query: str, tags: Optional[list[str]] = None, limit: int = 50
    ) -> list[SearchResult]: ...
    
    @abstractmethod
    async def get_message_by_id(self, message_id: str) -> Optional[Message]: ...
    
    @abstractmethod
    async def get_file_by_id(self, file_id: str) -> Optional[File]: ...
    
    @abstractmethod
    async def get_by_session(self, session_id: str) -> list[Message]: ...
    
    @abstractmethod
    async def health_check(self) -> bool: ...
```

#### SQLite Implementation (`sqlite.py`)

**Key Features:**
- FTS5 full-text search indexes
- JSON serialization for tags and metadata
- Session-based message grouping
- Automatic timestamp handling
- Connection pooling with configurable timeout
- Windows-compatible path handling

**Default Database Location:** `~/.arc-saga/memory.db`

**FTS5 Search Syntax:**
- Simple word search: `"Python"`
- Phrase search: `"machine learning"`
- Boolean operators: `Python AND Flask`
- Negation: `Python NOT Django`

### 5.3 API Server (`arc_saga/api/server.py`)

#### Endpoints

| Method | Path | Description | Status |
|--------|------|-------------|--------|
| POST | `/capture` | Store a conversation message | ✅ Working |
| GET | `/context/recent` | Get recent context | ✅ Working |
| GET | `/thread/{thread_id}` | Get complete thread history | ✅ Working |
| POST | `/search` | Search across all conversations | ✅ Working |
| POST | `/attach/file` | Attach file to thread | ✅ Working |
| POST | `/perplexity/ask` | Ask Perplexity with context | 🔄 Requires API key |
| GET | `/health` | Server health check | ✅ Working |
| GET | `/threads` | List all threads | ✅ Working |

#### Request/Response Models

**CaptureRequest:**
```python
class CaptureRequest(BaseModel):
    source: str              # "perplexity", "copilot", etc.
    role: str                # "user" or "assistant"
    content: str             # Message content
    thread_id: Optional[str] # Session/thread ID
    metadata: Optional[dict] # Additional metadata
```

**SearchRequest:**
```python
class SearchRequest(BaseModel):
    query: str                       # Search query
    search_type: str = "keyword"     # Future: "semantic"
    sources: Optional[List[str]]     # Filter by provider
    date_from: Optional[str]         # Date range start
    date_to: Optional[str]           # Date range end
    limit: int = 20                  # Max results
```

#### Server Configuration

- **Port:** 8421
- **Host:** 127.0.0.1 (localhost only)
- **CORS Origins:** `vscode-webview://*`, `http://localhost:*`
- **Lifespan:** Uses modern FastAPI lifespan context manager

### 5.4 Services

#### AutoTagger (`arc_saga/services/auto_tagger.py`)

**Purpose:** Extract keywords from message content using TF-IDF

**Configuration:**
- Max features: 10
- Stop words: English
- N-gram range: (1, 2) - unigrams and bigrams
- Minimum score threshold: 0.1

**Usage:**
```python
tagger = AutoTagger()
tags = tagger.extract_tags("Python programming with machine learning", max_tags=5)
# Returns: ["python", "machine learning", "programming", ...]
```

#### FileProcessor (`arc_saga/services/file_processor.py`)

**Purpose:** Extract text from uploaded files for indexing

**Supported Formats:**
| Extension | Method | Library |
|-----------|--------|---------|
| .pdf | `_extract_pdf_text()` | PyMuPDF (fitz) |
| .docx, .doc | `_extract_docx_text()` | python-docx |
| .txt, .md, .py, .js, .ts, .json | `_extract_plain_text()` | Built-in |

**Storage Location:** `~/.arc_saga/files/`

### 5.5 Error Instrumentation (`arc_saga/error_instrumentation.py`)

**Components:**

1. **Request Context** - Thread-safe context variables for correlation IDs
2. **LatencyMetrics** - Track p50, p95, p99 latency
3. **ErrorContext** - Capture complete error information
4. **CircuitBreakerMetrics** - Track circuit breaker state

**Key Functions:**

```python
# Create request context at start of operation
ctx = create_request_context(user_id="user123", service_name="arc_saga")
request_context.set(ctx)

# Log with full context
log_with_context(
    "info",
    "operation_complete",
    duration_ms=145,
    items_processed=100
)

# Get correlation ID for tracing
correlation_id = get_correlation_id()
```

---

## 6. Development Conventions

### 6.1 Code Quality Standards

| Tool | Purpose | Target |
|------|---------|--------|
| `mypy --strict` | Type checking | 0 errors |
| `pylint` | Linting | >= 8.0 |
| `black` | Code formatting | Compliant |
| `isort` | Import sorting | Compliant |
| `bandit` | Security scanning | 0 issues |

### 6.2 Type Hints

**Mandatory on:**
- All function parameters
- All function return types
- All class attributes

**Prohibited:**
- Bare `Any` types without justification comment
- Untyped function signatures

**Example:**
```python
from typing import Optional, List, Dict, Any

async def search_messages(
    self,
    query: str,
    tags: Optional[list[str]] = None,
    limit: int = 50
) -> list[SearchResult]:
    """Search messages using FTS5."""
    ...
```

### 6.3 Docstring Style

Use **Google-style docstrings**:

```python
def process_item(item_id: str, config: Optional[Dict[str, Any]] = None) -> Result:
    """
    Process a single item.

    Args:
        item_id: Unique identifier for the item
        config: Optional configuration overrides

    Returns:
        Result object with processed item or error

    Raises:
        ValidationError: If item_id is invalid
        ProcessingError: If processing fails

    Example:
        >>> result = process_item("abc123")
        >>> print(result.success)
        True
    """
```

### 6.4 Error Handling Pattern

```python
try:
    result = await external_service.call()
    log_with_context("info", "operation_success", duration_ms=elapsed)
    return result

except RateLimitError:
    # Transient - retry with backoff
    log_with_context("warning", "rate_limited", attempt=attempt)
    await asyncio.sleep(min(2 ** attempt, 60))

except ConnectionError:
    # Transient - retry
    log_with_context("warning", "connection_failed")

except TimeoutError:
    # Transient - retry
    log_with_context("warning", "timeout")

except NotFoundError:
    # Permanent - don't retry
    log_with_context("error", "resource_not_found")
    raise

except Exception as e:
    # Unexpected - log fully and re-raise
    log_with_context("error", "unexpected_error", error=str(e), exc_info=True)
    raise
```

### 6.5 Logging Requirements

**Every major operation must log:**

1. **Operation start** with parameters (sanitized)
2. **Milestones** during operation
3. **Success** with duration and metrics
4. **Failure** with full error context

**All logs include:**
- `request_id` - Correlation ID
- `trace_id` - Distributed tracing
- `timestamp` - ISO 8601
- `level` - DEBUG/INFO/WARNING/ERROR/CRITICAL

### 6.6 Git Commit Messages

Format: `<type>: <description>`

Types:
- `feat:` - New feature
- `fix:` - Bug fix
- `refactor:` - Code refactoring
- `docs:` - Documentation
- `test:` - Tests
- `chore:` - Maintenance

---

## 7. Testing Infrastructure

### 7.1 Test Structure

```
tests/
├── __init__.py
├── unit/                       # Fast, isolated tests
│   ├── __init__.py
│   ├── test_models.py         # Data model tests
│   └── test_storage.py        # Storage tests (with temp DB)
├── integration/               # Tests with real dependencies
│   └── __init__.py
├── test_models.py             # Root-level duplicates
└── test_storage.py            # Root-level duplicates
```

### 7.2 Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=arc_saga --cov-report=html

# Run specific test file
pytest tests/unit/test_models.py

# Run tests matching pattern
pytest -k "test_message"

# Run async tests
pytest -v --asyncio-mode=auto
```

### 7.3 Test Fixtures

**Temporary Storage Fixture:**
```python
@pytest_asyncio.fixture
async def storage():
    """Create temporary storage for testing."""
    tmpdir = tempfile.mkdtemp()
    try:
        db_path = Path(tmpdir) / "test.db"
        stor = SQLiteStorage(str(db_path))
        await stor.initialize()
        yield stor
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
```

### 7.4 Coverage Targets

| Category | Target | Current |
|----------|--------|---------|
| Unit Tests | 95%+ | ~80% |
| Integration Tests | 85%+ | ~50% |
| Error Paths | 100% | ~70% |
| Edge Cases | 100% | ~60% |

---

## 8. Planned Features & Roadmap

### Phase 1c: Monitoring & Resilience (Next)

1. **Health Check System**
   - Database health
   - Storage space monitoring
   - API latency tracking

2. **Circuit Breaker Implementation**
   - For Perplexity API calls
   - For file processing
   - Configurable thresholds

3. **Rate Limiting**
   - Per-client limits
   - Per-endpoint limits
   - Graceful degradation

4. **Validator Integration**
   - AI config validation
   - Antigravity log monitoring

### Feature Ideas Backlog (living)

A concise, living backlog of feature and control-plane ideas lives in docs/feature_ideas_backlog.md. This file is intended to capture brainstormed features, orchestration patterns, UX ideas (Audit tab, Agent Activity, Command Console), provider adaptors, and stability primitives so they are not lost. The backlog items are grouped by their rough phase alignment below for discoverability; consult the standalone file for full details.

- Phase 1c (near-term): Audit & UX (Audit Dashboard, WebSocket events, Actionable Audit Notes), Capture API idempotency/validation, Perplexity integration fixes, ProviderAdapter refactor (foundations for replaceability).
- Phase 2: Orchestrator & Control Plane (Policy Engine, Intent Classification, CommandEnvelope), Provider registry, Vector/Semantic RetrievalStrategy.
- Phase 3: Enterprise features (Multi-tenancy, Analytics Dashboard, Event Sourcing / CQRS, OAuth, backups).

Refer to docs/feature_ideas_backlog.md for the full backlog and concise one-line descriptions.

### Phase 2: Advanced Features

1. **Event Sourcing**
   - Immutable event log
   - Event replay capability
   - Audit trail

2. **CQRS Pattern**
   - Separated read/write models
   - Optimized read projections
   - Eventually consistent updates

3. **Vector Search (Semantic)**
   - Embedding generation
   - Qdrant/Pinecone integration
   - Similarity search

4. **Multi-Agent Memory Sync**
   - Cross-agent context sharing
   - Conflict resolution
   - Privacy controls

5. **OAuth Authentication**
   - Google SSO
   - GitHub SSO
   - Microsoft SSO
   - JWT token management

### Phase 3: Enterprise Features

1. **Multi-Tenancy**
   - User isolation
   - Team/org support
   - Role-based access

2. **Analytics Dashboard**
   - Usage metrics
   - Search analytics
   - Performance monitoring

3. **Export/Import**
   - Backup/restore
   - Data portability
   - Migration tools

---

## 9. Known Issues & Technical Debt

### 9.1 Critical Issues

| Issue | Location | Impact | Priority |
|-------|----------|--------|----------|
| Perplexity client storage method mismatch | `arc_saga/integrations/perplexity_client.py` - `ask_streaming()` method | Uses `store_message` instead of `save_message` | 🔴 High |
| Message model field mismatch | `arc_saga/integrations/perplexity_client.py` - Message creation | Uses `thread_id` instead of `session_id` | 🔴 High |

### 9.2 Warnings

| Issue | Location | Impact | Priority |
|-------|----------|--------|----------|
| Duplicate test files | `tests/test_*.py` and `tests/unit/test_*.py` | Root and unit/ have duplicates | 🟡 Medium |
| Missing integration tests | `tests/integration/` | Only `__init__.py` present | 🟡 Medium |
| File storage not fully integrated | `arc_saga/api/server.py` - `attach_file()` endpoint | Files saved to disk but not stored in DB | 🟡 Medium |

### 9.3 Technical Debt

1. **Search query workaround** - Using "a" as default query to avoid FTS5 empty query error. Location: `arc_saga/api/server.py` in `get_recent_context()` and `list_threads()` endpoints
2. **Hardcoded port** - Server port 8421 should be configurable. Location: `arc_saga/api/server.py` at bottom
3. **No connection pooling** - SQLite connection created per request. Location: `arc_saga/storage/sqlite.py` - `_get_connection()` method
4. **Missing retry logic** - External calls don't have retry with backoff. Location: `arc_saga/integrations/perplexity_client.py`
5. **No request validation** - API requests not fully validated. Location: Various endpoints in `arc_saga/api/server.py`

### 9.4 Missing Features (Documented but Not Implemented)

- Circuit breaker pattern
- Rate limiting
- OAuth authentication
- Vector/semantic search
- Event sourcing
- CQRS implementation
- Health check dashboard

---

## 10. Critical Files Reference

### Configuration Files

| File | Purpose | Key Contents |
|------|---------|--------------|
| `.cursorrules` | Cursor AI configuration | Quality standards, patterns, checklists |
| `shared/config.py` | Shared configuration | Paths, limits, settings |
| `requirements.txt` | Python dependencies | All package requirements |
| `setup.py` | Package installation | Install configuration |

### Documentation Files

| File | Purpose | When to Reference |
|------|---------|-------------------|
| `docs/arc_saga_master_index.md` | System overview | Understanding the system |
| `docs/decision_catalog.md` | Architectural decisions | Making design choices |
| `docs/error_catalog.md` | Error solutions | Debugging errors |
| `docs/prompts_library.md` | Cursor prompts | Generating code |
| `docs/verification_checklist.md` | Quality gates | Before deploying |

### Source Files

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `arc_saga/models/message.py` | Data models | Message, File, Provider, MessageRole |
| `arc_saga/storage/base.py` | Storage interface | StorageBackend (ABC) |
| `arc_saga/storage/sqlite.py` | SQLite implementation | SQLiteStorage |
| `arc_saga/api/server.py` | REST API | FastAPI app, all endpoints |
| `arc_saga/services/auto_tagger.py` | Auto-tagging | AutoTagger |
| `arc_saga/services/file_processor.py` | File processing | FileProcessor |
| `arc_saga/error_instrumentation.py` | Error tracking | All instrumentation classes |
| `arc_saga/logging_config.py` | Logging setup | setup_logging, get_logger |
| `arc_saga/exceptions/storage_exceptions.py` | Custom exceptions | All exception classes |

---

## 11. Development Workflows

### 11.1 Adding a New Feature

1. **Check `decision_catalog.md`** for existing patterns
2. **Find matching prompt** in `prompts_library.md`
3. **Implement with type hints** and error handling
4. **Add logging** with correlation IDs
5. **Write tests** (unit + integration)
6. **Run verification checklist**
7. **Update documentation** if needed

### 11.2 Fixing a Bug

1. **Check `error_catalog.md`** for known solutions
2. **Reproduce with test** (failing test first)
3. **Add logging** to trace the issue
4. **Fix with minimal changes**
5. **Verify test passes**
6. **Update `error_catalog.md`** if new error type

### 11.3 Running the Server

```bash
# Install dependencies
pip install -e .

# Run development server
python -m arc_saga.api.server

# Or with uvicorn
uvicorn arc_saga.api.server:app --host 127.0.0.1 --port 8421 --reload
```

### 11.4 Health Check

```bash
# Check server health
curl http://localhost:8421/health

# Expected response:
# {"status": "healthy", "database": "~/.arc-saga/memory.db", "timestamp": "..."}
```

### 11.5 Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=arc_saga --cov-report=term-missing

# Specific module
pytest tests/unit/test_storage.py -v
```

---

## 12. Integration Points

### 12.1 Perplexity Integration

**Status:** 🔄 Partial (requires API key)

**Configuration:**
```bash
export PPLX_API_KEY="your-api-key"
```

**Endpoint:** `POST /perplexity/ask`

**Features:**
- Streaming responses
- Context injection from stored messages
- Automatic conversation storage

### 12.2 VSCode Extension Integration

**CORS Configuration:** Allows `vscode-webview://*`

**Typical Usage:**
1. Extension captures user query
2. Sends to `/capture` endpoint
3. Retrieves context via `/context/recent`
4. Shows results in webview

### 12.3 File System Monitoring (Planned)

**Purpose:** Watch for Antigravity/Copilot log files

**Location:** `arc_saga/monitors/`

**Planned Features:**
- Watchdog file monitoring
- Log parsing
- Automatic ingestion

---

## Appendix A: Environment Variables

| Variable | Purpose | Required | Default |
|----------|---------|----------|---------|
| `PPLX_API_KEY` | Perplexity API key | Optional | None |
| `ARC_SAGA_LOG_LEVEL` | Logging level | Optional | INFO |
| `ANTIGRAVITY_LOGS` | Antigravity log path | Optional | `~/AppData/Roaming/Antigravity/logs` |

## Appendix B: Port Assignments

| Service | Port | Purpose |
|---------|------|---------|
| ARC Saga API | 8421 | Main REST API |

## Appendix C: Quick Commands

```bash
# Start server
python -m arc_saga.api.server

# Run tests
pytest --cov=arc_saga

# Type check
mypy arc_saga --strict

# Lint
pylint arc_saga

# Format
black arc_saga

# Sort imports
isort arc_saga

# Security scan
bandit -r arc_saga

# Run system audit
python audit_system.py
```

---

**Document maintained by:** ARC Saga Development Team  
**For questions:** Reference `docs/` directory or check `error_catalog.md`  
**Last verification:** 2025-12-02
