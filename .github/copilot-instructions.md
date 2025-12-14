# GitHub Copilot Instructions for ARC SAGA

## Project Overview

ARC SAGA is a Python-based AI-powered code quality enforcement system with intelligent memory management. It is built using FastAPI and follows FAANG-level coding standards.

### Key Technologies
- **Language**: Python 3.11 (enforced in mypy.ini)
- **Framework**: FastAPI with Pydantic v2
- **Database**: SQLite with SQLAlchemy (async)
- **Testing**: pytest with pytest-asyncio
- **Quality Tools**: mypy (strict), pylint, ruff, black, bandit

## Directory Structure

```
arc_saga/
├── api/           # FastAPI endpoints and server
├── core/          # Core business logic and utilities
├── exceptions/    # Custom exception classes
├── integrations/  # External service integrations
├── models/        # Pydantic models and data structures
├── monitors/      # Monitoring and observability
├── orchestrator/  # AI agent orchestration
├── services/      # Business services
└── storage/       # Database and persistence layer

tests/             # Test suite (mirrors arc_saga structure)
docs/              # Documentation and decision records
```

## Code Quality Standards

### Non-Negotiable Requirements (Enforced by CI)
- **Type Safety**: All code must pass `mypy --strict`
- **Test Coverage**: Minimum 98% coverage (enforced by CI, see `.github/workflows/ci.yml`)
- **Linting**: pylint score must be >= 9.0 (enforced by CI)
- **Security**: Zero bandit issues allowed
- **No TODOs**: No TODO/FIXME comments in production code

### Type Hints
- Use type hints for all function parameters and return values
- Avoid `Any` type without explicit justification in comments
- Use `Optional[]` for nullable values
- Use `Protocol` for interfaces

```python
# Good
async def process_item(item_id: str, config: Optional[Dict[str, str]] = None) -> Result[ProcessedItem]:
    ...

# Bad
def process_item(item_id, config=None):
    ...
```

### Error Handling
- Handle all failure modes explicitly
- Use structured logging with correlation IDs
- Distinguish between transient and permanent errors
- Implement retry with exponential backoff for transient failures

```python
try:
    result = await external_service.call()
    log_with_context("info", "operation_success", duration_ms=elapsed)
    return result
except RateLimitError:
    # Transient - retry
    log_and_retry()
except NotFoundError:
    # Permanent - don't retry
    log_with_context("error", "resource_not_found")
    raise
```

### Logging
- Use structured JSON logging with correlation IDs
- Include: request_id, trace_id, user_id, timestamp
- Never log secrets, passwords, or API keys
- Log operation start, milestones, success, and failures

## Testing Guidelines

### Test Organization
- Unit tests: `tests/unit/`
- Integration tests: `tests/integration/`
- Use `pytest.mark` decorators: `@pytest.mark.unit`, `@pytest.mark.integration`

### Test Requirements
- Test happy paths and error paths
- Test edge cases (empty, null, max, min values)
- Use `pytest.raises` for exception testing
- Use `pytest.mark.asyncio` for async tests

```python
@pytest.mark.unit
def test_validate_input_with_missing_field():
    """Test validation fails when required field is missing."""
    with pytest.raises(ValueError, match="required"):
        Entity(name="test")  # Missing id

@pytest.mark.asyncio
async def test_async_operation():
    """Test async database operation."""
    result = await repo.save(entity)
    assert result.id is not None
```

## Build and Quality Commands

### Formatting
```bash
python -m black arc_saga tests
isort --profile black arc_saga tests
```

### Linting
```bash
ruff check arc_saga tests
pylint arc_saga tests --fail-under=9.0
```

### Type Checking
```bash
mypy --strict arc_saga tests
```

### Security Scanning
```bash
bandit -r arc_saga -x tests
```

### Testing
```bash
pytest --cov=arc_saga --cov-fail-under=98 tests/
```

### Full Quality Check
```bash
python -m black . && ruff check --fix . && mypy --strict arc_saga tests && pylint arc_saga tests --fail-under=9.0 && bandit -r arc_saga -x tests && pytest --cov=arc_saga --cov-fail-under=98
```

## Architecture Patterns

### Repository Pattern
Use for all data access operations:
```python
class IRepository(Protocol[T]):
    async def get_by_id(self, id: str) -> Optional[T]: ...
    async def save(self, entity: T) -> T: ...
    async def delete(self, id: str) -> bool: ...
```

### Circuit Breaker
Use for external service calls to prevent cascading failures:
```python
async with circuit_breaker.call(external_service):
    result = await external_service.fetch()
```

### Retry with Exponential Backoff
Use for transient failures:
```python
delay = min(base_delay * (2 ** attempt), max_delay) + jitter
```

## Security Guidelines

- Input validation on all external data using Pydantic validators
- Parameterized SQL queries (never string concatenation)
- Secrets from environment variables (never hardcoded)
- HTTPS/TLS for all external calls
- Rate limiting on API endpoints

## Dependencies

Core dependencies are managed in `requirements.txt`. Key packages:
- `fastapi==0.104.1` - Web framework
- `pydantic==2.5.0` - Data validation
- `sqlalchemy==2.0.23` - Database ORM
- `aiosqlite==0.19.0` - Async SQLite
- `pytest==7.4.3` - Testing framework

## Documentation

- `docs/decision_catalog.md` - Architecture decisions
- `docs/error_catalog.md` - Error handling patterns
- `docs/prompts_library.md` - Token-optimized prompts
- `docs/ROADMAP.md` - Development plan
- `.cursorrules` - Detailed quality enforcement rules

## Important Notes

1. **Do not use pre-commit hooks** - Run quality tools manually or via GitHub Actions CI
2. **GitHub Actions CI** is the source of truth for quality gates
3. **Review existing patterns** in `docs/decision_catalog.md` before making architectural decisions
4. **Update documentation** when adding new patterns or making significant changes
