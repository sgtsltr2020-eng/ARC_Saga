# SAGACODEX: Python/FastAPI Expert Standards

**Version**: 1.0.0  
**Date**: December 14, 2025  
**Status**: MVP Foundation - Phase 2 Week 1  
**Authority**: BINDING for ARC SAGA project

---

## INTRODUCTION

SAGA MVP is built with Python/FastAPI. These standards are **non-negotiable** for the ARC SAGA codebase.

Future versions will support other languages/frameworks, but MVP must demonstrate Python/FastAPI excellence.

These standards are derived from:

- **Phase 1 achievements**: 104 tests, 99% coverage, 0 security issues, mypy --strict compliance
- **FAANG operational contracts**: Patterns from Google, Meta, Amazon engineering practices
- **FastAPI best practices**: Async patterns, dependency injection, Pydantic validation
- **Elite app patterns**: Linear, Figma, Stripe engineering principles

---

## AUTHORITY HIERARCHY

1. **SagaConstitution (Meta-Rules)**: How SAGA operates → Cannot be overridden
2. **SagaCodex (This Document)**: Project quality standards → User can override with protest
3. **LoreBook**: Project-specific learned patterns → Advisory, not enforced

**Mimiry** (advisory subagent) references this document before providing guidance.  
**The Warden** (delegation agent) consults this document before creating tasks.  
**All subagents** must check relevant sections before generating code.

---

## CORE PYTHON STANDARDS

### 1. Type Safety (MANDATORY)

**Tool**: `mypy --strict`  
**Coverage**: 100% of public functions and methods  
**Why**: Python's dynamic typing causes runtime errors. Static typing catches bugs at development time.

**Rule**: Every function signature must have complete type hints.

✅ CORRECT

```python
from typing import Dict, Any, Optional
from datetime import datetime

def process_request(
    user_id: str,
    data: Dict[str, Any],
    timestamp: Optional[datetime] = None
) -> Dict[str, Any]:
    """Process user request with validation."""
    ...
```

❌ WRONG - No type hints

```python
def process_request(user_id, data, timestamp=None):
    ...
```

❌ WRONG - Incomplete type hints

```python
def process_request(user_id: str, data, timestamp) -> dict:
    ...
```

**Enforcement**:

- CI blocks PRs with mypy errors
- Pre-commit hook runs mypy --strict
- IDE (Cursor) shows type errors in real-time

---

### 2. Async/Await Best Practices

**Rule**: Use `async/await` for I/O-bound operations (database, API calls, file I/O)  
**Rule**: Use synchronous code for CPU-bound operations (computation, parsing, validation)  
**Why**: FastAPI is async-first. Blocking the event loop kills performance.

✅ CORRECT - Async for I/O

```python
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

async def get_user(db: AsyncSession, user_id: str) -> Optional[User]:
    """Fetch user from database asynchronously."""
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    return result.scalar_one_or_none()
```

✅ CORRECT - Sync for CPU-bound

```python
def calculate_hash(password: str) -> str:
    """CPU-intensive hashing is synchronous."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
```

❌ WRONG - Sync database call blocks event loop

```python
def get_user(db: Session, user_id: str) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()
```

❌ WRONG - Unnecessary async for CPU-bound

```python
async def calculate_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
```

**Pattern**: If it waits for external systems → async. If it computes → sync.

---

### 3. Error Handling (Explicit & Contextual)

**Rule**: Custom exceptions with context  
**Rule**: Never silent failures, never bare `except:`  
**Rule**: Log before raising, include trace_id

✅ CORRECT - Custom exception with context

```python
class UserNotFoundError(Exception):
    """User does not exist in database."""
    def __init__(self, user_id: str):
        self.user_id = user_id
        super().__init__(f"User {user_id} not found")

async def get_user_or_raise(db: AsyncSession, user_id: str) -> User:
    """Fetch user or raise descriptive error."""
    user = await get_user(db, user_id)

    if user is None:
        logger.error(
            "User not found",
            extra={"user_id": user_id, "trace_id": get_trace_id()}
        )
        raise UserNotFoundError(user_id)

    return user
```

❌ WRONG - Silent failure

```python
async def get_user_silent(db: AsyncSession, user_id: str) -> Optional[User]:
    try:
        return await get_user(db, user_id)
    except Exception:
        return None  # Swallowed error, no logging
```

❌ WRONG - Bare except catches everything (including KeyboardInterrupt!)

```python
try:
    user = await get_user(db, user_id)
except:  # Bad: catches SystemExit, KeyboardInterrupt, etc.
    pass
```

**Exception Hierarchy** (for ARC SAGA):

```text
SagaException (base)
├── ConfigurationError
├── ValidationError
├── ExecutionError
│   ├── SubagentError
│   ├── WardenError
│   └── MimiryError
├── ResourceError
│   ├── BudgetExceededError
│   ├── RateLimitError
│   └── TimeoutError
└── SecurityError
    ├── SecretsDetectedError
    ├── UnauthorizedAccessError
    └── PromptInjectionError
```

---

### 4. Structured Logging (No Print Statements)

**Tool**: Python `logging` module with JSON formatter  
**Rule**: All logs include `trace_id`, `user_id`, `operation`, `outcome`  
**Why**: Grep-able, parseable, traceable logs enable debugging production issues

✅ CORRECT - Structured logging

```python
import logging

logger = logging.getLogger(name)

async def create_user(db: AsyncSession, user_data: UserCreate) -> User:
    """Create new user with full audit trail."""
    logger.info(
        "Creating user",
        extra={
            "email": user_data.email,
            "trace_id": get_trace_id(),
            "operation": "user_create",
        }
    )

    try:
        user = User(**user_data.dict())
        db.add(user)
        await db.commit()

        logger.info(
            "User created successfully",
            extra={
                "user_id": user.id,
                "trace_id": get_trace_id(),
                "operation": "user_create",
                "outcome": "success",
            }
        )
        return user

    except Exception as e:
        logger.error(
            "User creation failed",
            extra={
                "error": str(e),
                "trace_id": get_trace_id(),
                "operation": "user_create",
                "outcome": "failure",
            },
            exc_info=True
        )
        raise
```

❌ WRONG - Print statements

```python
def create_user(db: Session, user_data: dict):
    print(f"Creating user: {user_data['email']}")  # Not parseable
    user = User(**user_data)
    db.add(user)
    db.commit()
    print("User created!")  # No context, no trace_id
    return user
```

**Log Levels** (strict usage):

- `DEBUG`: Internal state, flow control (disabled in production)
- `INFO`: Normal operations, state changes, successful completions
- `WARNING`: Recoverable errors, degraded performance, approaching limits
- `ERROR`: Operation failed, user-impacting, requires investigation
- `CRITICAL`: System failure, data loss risk, immediate action required

---

### 5. Input Validation at Boundaries

**Rule**: Validate all external inputs before use  
**Tool**: Pydantic models for API inputs, custom validators for complex logic  
**Why**: Prevents injection, type errors, malformed data crashes

✅ CORRECT - Pydantic validation

```python
from pydantic import BaseModel, EmailStr, Field, validator

class UserCreate(BaseModel):
    """Validated user creation input."""
    email: EmailStr  # Automatic email validation
    password: str = Field(min_length=8, max_length=100)
    age: int = Field(ge=18, le=120)

    @validator("password")
    def password_strength(cls, v: str) -> str:
        """Ensure password meets security requirements."""
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain digit")
        return v

@app.post("/users/")
async def create_user_endpoint(
    user: UserCreate,  # Automatic validation
    db: AsyncSession = Depends(get_db)
) -> UserResponse:
    """Create user with validated input."""
    # By the time we reach here, user is guaranteed valid
    return await create_user(db, user)
```

❌ WRONG - No validation

```python
@app.post("/users/")
async def create_user_endpoint(request: dict):
    email = request.get("email")  # Could be None, malformed, SQL injection
    password = request.get("password")  # Could be empty, too weak
    user = User(email=email, password=password)  # Crashes if invalid
    ...
```

---

## FASTAPI STANDARDS

### 6. Dependency Injection (Always)

**Rule**: Use FastAPI's `Depends()` for all external dependencies  
**Why**: Testable, mockable, clean separation of concerns, automatic lifecycle management

✅ CORRECT - Dependency injection

```python
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import AsyncGenerator

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Database session dependency with automatic cleanup."""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    """Authentication dependency."""
    user_id = decode_token(token)
    user = await get_user(db, user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid authentication")
    return user

@app.post("/users/")
async def create_user(
    user: UserCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),  # Automatic auth check
) -> UserResponse:
    """Dependencies injected automatically."""
    ...
```

❌ WRONG - Global state, untestable

```python
db_engine = create_engine(DATABASE_URL)

@app.post("/users/")
async def create_user(user: dict):
    session = db_engine.Session()  # Hard to mock in tests
    # No automatic cleanup
    # No transaction management
    ...
```

**Testing with Dependency Injection**:
Easy to override dependencies in tests

```python
def override_get_db():
    return test_db

app.dependency_overrides[get_db] = override_get_db
```

---

### 7. Pydantic Models for Request/Response

**Rule**: All API inputs and outputs use Pydantic models  
**Why**: Automatic validation, OpenAPI docs generation, type safety, serialization

✅ CORRECT - Pydantic models

```python
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class UserBase(BaseModel):
    """Shared user fields."""
    email: EmailStr
    full_name: str

class UserCreate(UserBase):
    """User creation input."""
    password: str

class UserUpdate(BaseModel):
    """User update input (all fields optional)."""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = None

class UserResponse(UserBase):
    """User API response."""
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True  # Allows ORM model conversion

@app.post("/users/", response_model=UserResponse, status_code=201)
async def create_user(
    user: UserCreate,
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Pydantic validates input, serializes output.
    OpenAPI docs auto-generated from models.
    """
    db_user = User(
        email=user.email,
        full_name=user.full_name,
        hashed_password=hash_password(user.password)
    )
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user  # Automatically serialized to UserResponse
```

❌ WRONG - Dict-based, no validation

```python
@app.post("/users/")
async def create_user(request: dict) -> dict:
    # No validation
    # No docs
    # No type safety
    # Manual serialization
    ...
```

---

### 8. Background Tasks for Long Operations

**Rule**: Operations >100ms use `BackgroundTasks`  
**Why**: Don't block HTTP response waiting for emails, webhooks, cleanup

✅ CORRECT - Background task

```python
from fastapi import BackgroundTasks

async def send_welcome_email(user_email: str, user_name: str) -> None:
    """Send email asynchronously (takes 2 seconds)."""
    await email_service.send(
        to=user_email,
        subject="Welcome!",
        body=f"Hello {user_name}, welcome to SAGA!"
    )

@app.post("/users/", response_model=UserResponse)
async def create_user(
    user: UserCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> User:
    """User created, email sent in background."""
    db_user = await user_service.create(db, user)

    # Runs after response sent to user
    background_tasks.add_task(send_welcome_email, db_user.email, db_user.full_name)

    return db_user  # Returns immediately (< 50ms)
```

❌ WRONG - Blocks response

```python
@app.post("/users/")
async def create_user(user: UserCreate, db: AsyncSession = Depends(get_db)):
    db_user = await user_service.create(db, user)
    await send_welcome_email(db_user.email, db_user.full_name)  # User waits 2s
    return db_user
```

---

## SQLALCHEMY ASYNC PATTERNS

### 9. Session Management (Context Manager)

**Rule**: Use async context manager for session lifecycle  
**Rule**: Never hold sessions across requests  
**Why**: Prevents connection leaks, ensures proper cleanup

✅ CORRECT - Async context manager

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

engine = create_async_engine(DATABASE_URL, echo=False)
async_session_maker = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Properly managed database session."""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()  # Always closes

# Usage in endpoint
@app.get("/users/{user_id}")
async def get_user_endpoint(
    user_id: str,
    db: AsyncSession = Depends(get_db)
) -> UserResponse:
    user = await get_user(db, user_id)
    return user  # Session automatically closed after response
```

❌ WRONG - Session leak

```python
db = AsyncSession(engine)

@app.get("/users/{user_id}")
async def get_user_endpoint(user_id: str):
    user = await get_user(db, user_id)  # Session never closes
    return user  # Connection leak
```

---

### 10. Explicit Queries (No Lazy Loading)

**Rule**: Use `selectinload()` or `joinedload()` for relationships  
**Why**: Async SQLAlchemy doesn't support lazy loading. N+1 query problem kills performance.

✅ CORRECT - Eager loading

```python
from sqlalchemy.orm import selectinload

async def get_user_with_posts(db: AsyncSession, user_id: str) -> User:
    """Fetch user and posts in 2 queries (not N+1)."""
    result = await db.execute(
        select(User)
        .where(User.id == user_id)
        .options(selectinload(User.posts))  # Explicit eager load
    )
    return result.scalar_one_or_none()
```

❌ WRONG - Lazy loading doesn't work with async

```python
async def get_user_with_posts_lazy(db: AsyncSession, user_id: str) -> User:
    user = await get_user(db, user_id)
    # This will FAIL or cause N+1 queries:
    for post in user.posts:  # Tries to lazy load, doesn't work
        print(post.title)
    return user
```

---

## TESTING STANDARDS

### 11. Pytest + Async

**Tool**: `pytest-asyncio`  
**Coverage Target**: 99% (as achieved in Phase 1)  
**Pattern**: Fixtures for dependencies

✅ CORRECT - Async test with fixtures

```python
import pytest
from httpx import AsyncClient

@pytest.fixture
async def client(app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Test client with automatic cleanup."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
async def test_user(db: AsyncSession) -> User:
    """Create test user for tests."""
    user = User(email="test@example.com", full_name="Test User")
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user

@pytest.mark.asyncio
async def test_create_user(client: AsyncClient):
    """Test user creation endpoint."""
    response = await client.post("/users/", json={
        "email": "new@example.com",
        "full_name": "New User",
        "password": "SecurePass123",
    })
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "new@example.com"
    assert "id" in data
    assert "password" not in data  # Never return passwords

@pytest.mark.asyncio
async def test_get_user(client: AsyncClient, test_user: User):
    """Test user retrieval."""
    response = await client.get(f"/users/{test_user.id}")
    assert response.status_code == 200
    assert response.json()["email"] == test_user.email
```

---

## SAGA MVP SPECIFIC RULES

### 12. Event Sourcing Pattern (Phase 1 Architecture)

**Rule**: Use existing event-sourced architecture  
**Why**: Proven pattern from Phase 1, enables audit trail and rollback

✅ CORRECT - Event-sourced operation

```python
from saga.core.events import Event, EventType

async def create_user_with_events(
    db: AsyncSession,
    user_data: UserCreate,
    trace_id: str
) -> User:
    """Create user with full event trail."""

    # Emit start event
    start_event = Event(
        type=EventType.USER_CREATE_STARTED,
        trace_id=trace_id,
        payload={"email": user_data.email},
    )
    await event_store.append(start_event)

    try:
        user = User(**user_data.dict(exclude={"password"}))
        user.hashed_password = hash_password(user_data.password)
        db.add(user)
        await db.commit()

        # Emit success event
        success_event = Event(
            type=EventType.USER_CREATE_COMPLETED,
            trace_id=trace_id,
            payload={"user_id": user.id},
        )
        await event_store.append(success_event)

        return user

    except Exception as e:
        # Emit failure event
        failure_event = Event(
            type=EventType.USER_CREATE_FAILED,
            trace_id=trace_id,
            payload={"error": str(e)},
        )
        await event_store.append(failure_event)
        raise
```

---

### 13. Circuit Breaker for External APIs (Phase 1 Pattern)

**Rule**: All external API calls use circuit breaker  
**Why**: Graceful degradation, prevents cascade failures

✅ CORRECT - Circuit breaker pattern

```python
from saga.resilience.circuit_breaker import CircuitBreaker

perplexity_breaker = CircuitBreaker(
    failure_threshold=5,
    timeout=30,
    recovery_timeout=60
)

@perplexity_breaker
async def search_perplexity(query: str) -> SearchResult:
    """Protected external API call."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.perplexity.ai/search",
            json={"query": query},
            timeout=10.0
        )
        response.raise_for_status()
        return SearchResult(**response.json())

# Usage handles circuit breaker states
try:
    result = await search_perplexity("Python best practices")
except CircuitBreakerOpen:
    logger.warning("Perplexity circuit breaker open, using fallback")
    result = await search_fallback(query)
```

---

### 14. Observability (trace_id Required)

**Rule**: Every operation has trace_id  
**Rule**: Emit events for all state changes  
**Why**: Enables debugging, audit trail, LoreBook learning

✅ CORRECT - Full observability

```python
import uuid
from contextvars import ContextVar

trace_id_var: ContextVar[str] = ContextVar("trace_id")

def get_trace_id() -> str:
    """Get current trace_id from context."""
    return trace_id_var.get()

async def operation_with_tracing(db: AsyncSession, data: SomeData) -> Result:
    """Every operation is fully traceable."""
    trace_id = str(uuid.uuid4())
    trace_id_var.set(trace_id)

    logger.info(
        "Operation started",
        extra={"trace_id": trace_id, "operation": "some_operation"}
    )

    try:
        result = await perform_operation(db, data)

        logger.info(
            "Operation completed",
            extra={
                "trace_id": trace_id,
                "operation": "some_operation",
                "outcome": "success"
            }
        )
        return result

    except Exception as e:
        logger.error(
            "Operation failed",
            extra={
                "trace_id": trace_id,
                "operation": "some_operation",
                "outcome": "failure",
                "error": str(e)
            },
            exc_info=True
        )
        raise
```

---

## ENFORCEMENT

**Mimiry** (advisory subagent) will:

- Reference these standards before providing guidance
- Flag violations with rule numbers
- Suggest fixes with code examples

**The Warden** (delegation agent) will:

- Consult SagaCodex before creating subagent tasks
- Include relevant rule numbers in task specifications
- Verify subagent outputs against standards

**All Subagents** will:

- Check relevant sections before generating code
- Self-verify against checklists
- Report compliance in task completion

---

## VERSION HISTORY

- **v1.0.0** (Dec 14, 2025): Initial MVP standards for ARC SAGA Python/FastAPI project

---

**Next**: Language-specific profiles will be added in Phase 3 (TypeScript/React, Go, etc.)

---

## PROCESS STANDARDS

### 45. Minimize Diff Surface for Safe Fixes

**Rule**: Minimal changes for static-analysis-only fixes (mypy, lint)  
**Why**: Large diffs in tests mask regressions and make review impossible.
**Tags**: refactoring, tests, mypy, diff-size

✅ CORRECT - Targeted annotation

```python
# Only change the signature, leave the body alone
def test_user_creation(client) -> None:  # Added -> None
    response = client.post("/users/", json={...})
    assert response.status_code == 201
```

❌ WRONG - Full rewrite (Anti-Pattern)

```python
# Rewrote entire test file to "fix types"
# Changed logic, variable names, or formatting unnecessarily
class TestUser:
    def test_create(self, api_client):
        # ... completely new implementation ...
```

**Checklist Item**: "When fixing lint/mypy-only issues, prefer adding annotations or small edits instead of rewriting entire files—especially tests."
**Detection Hint**: Large diff in a test file where commit message mentions 'type hints' or 'mypy'.
