# Cursor Prompts Library

# Pre-Optimized, Surgical Prompts for Maximum Token Efficiency

# Copy, paste, customize, done.

---

## 🎯 HOW TO USE THIS LIBRARY

1. **Find the prompt** that matches your need
2. **Copy the entire prompt**
3. **Replace {placeholders} with actual values**
4. **Paste into Cursor**
5. **Get production-grade code in 1-2 tokens**

---

## Architecture & Design Prompts

### PROMPT-ARCH-001: Create Repository Interface

**Use When:** Need to create a new data access interface

**Prompt:**

```
@codebase Create a repository interface for {EntityName} following our Repository pattern in decision_catalog.md.

Requirements:
- Use generic IRepository[T] pattern from .cursorrules
- Type hints on all methods
- Methods: get_by_id(id) -> Optional[T], save(entity) -> T, delete(id) -> bool, find_by_criteria(**criteria) -> List[T]
- Include transaction context manager for multi-operation transactions
- Google-style docstrings with examples
- Full error handling for all database operations
- Comprehensive logging with correlation IDs
- Tests with 95%+ coverage including edge cases

File location: arc_saga/repositories/{entity_name}_repository.py

Reference: See database connection retry pattern in error_catalog.md for external service calls.
```

**Token Cost:** ~2-3 requests (good return)

**What Cursor Will Generate:**

- Type-safe interface
- Error handling with Circuit Breaker
- Logging with correlation IDs
- Tests for happy path + errors
- 95%+ coverage

---

### PROMPT-ARCH-002: Implement Event Sourcing Pattern

**Use When:** Need to capture events for audit trail or temporal tracking

**Prompt:**

```
@codebase Implement Event Sourcing for {BoundedContext} following the Event-Driven CQRS pattern in architecture.mdc.

Requirements:
- Create BaseEvent class with Pydantic validation
- Specific event types for this bounded context (List: {event1}, {event2}, {event3})
- Event Store interface: append_event(event), get_events(aggregate_id), replay(aggregate_id)
- Immutable events (frozen dataclass)
- Correlation IDs on all events
- Timestamps on all events
- Version tracking for event schema migration
- Full type hints
- Comprehensive error handling
- Tests with 95%+ coverage

File location: arc_saga/{bounded_context}/events.py

Reference: See CircuitBreaker pattern for external API calls in .cursorrules
```

**Token Cost:** ~3-4 requests (complex, worth it)

**Success Rate:** 95% (clear pattern)

---

### PROMPT-ARCH-003: Add Health Check System

**Use When:** Need to monitor service health

**Prompt:**

```
@codebase Add health check system for {ServiceName}.

Requirements:
- HealthCheck interface: async check() -> bool, name() -> str
- Specific checks for {service}: [{check1}, {check2}, {check3}]
- Timeout protection (5000ms max)
- Circuit breaker for health checks (don't spam checks)
- Logging all check results
- Correlation IDs on all logs
- Structured JSON output: {status: healthy|degraded|unhealthy, checks: [...], timestamp: ...}
- Tests including timeout scenarios
- 95%+ coverage

Reference: See health check implementation in .cursorrules section on Self-Healing
```

**Token Cost:** ~2 requests

---

## Error Handling Prompts

### PROMPT-ERROR-001: Fix Database Timeout

**Use When:** Getting database timeout errors

**Prompt:**

```
@codebase Database timeouts occurring. Fix using decision_catalog.md ERROR-001.

Current error: [Paste error message and stack trace]

Requirements:
- Diagnose the root cause (pool size? slow query? overload?)
- Implement appropriate fix from error_catalog.md (retry, pool increase, circuit breaker)
- Add comprehensive logging with context
- Add metrics: query duration, pool utilization
- Update SYSTEM_STATUS.md with findings
- Add tests verifying fix works
- 95%+ coverage

Reference: Database timeout decision in decision_catalog.md
```

**Token Cost:** ~2-3 requests

---

### PROMPT-ERROR-002: Null Pointer Exception

**Use When:** Getting "object has no attribute" errors

**Prompt:**

```
@codebase Fix NullPointer error at {file}:{line}.

Error: {exact error message}

Requirements:
- Add None checks with meaningful error messages
- Add Optional type hints
- Add error handling with specific exception types
- Add logging with full context
- Add test reproducing error
- Add test verifying fix
- Run mypy --strict verification

Reference: ERROR-004 in error_catalog.md
```

**Token Cost:** ~1-2 requests

---

## Feature Implementation Prompts

### PROMPT-FEAT-001: Create API Endpoint

**Use When:** Need to create a new FastAPI endpoint

**Prompt:**

```
@codebase Create FastAPI endpoint for {operation_name}.

Specification:
- Path: {HTTP_METHOD} /api/{path}
- Input: {InputModel}
- Output: {OutputModel}
- Authentication: {auth_strategy}
- Authorization: {auth_check}

Requirements:
- Request validation using Pydantic
- Response validation using Pydantic
- Complete error handling (400, 401, 403, 404, 500)
- Structured logging with correlation IDs
- Performance metric logging (duration_ms)
- Circuit breaker for external calls
- Tests for: happy path, validation failure, auth failure, external service failure
- 95%+ coverage
- Type hints on all parameters and returns

Reference: See error handling patterns in .cursorrules
```

**Token Cost:** ~2-3 requests

---

### PROMPT-FEAT-002: Implement Conversation Capture

**Use When:** Need to capture conversations from AI provider

**Prompt:**

```
@codebase Implement conversation capture for {Provider} following Event-Driven CQRS pattern.

Provider Details:
- API endpoint: {endpoint}
- Authentication: {auth_type}
- Rate limit: {rate_limit}
- Failure modes: {failure_modes}

Requirements:
- ConversationCapturedEvent with full schema
- Retry with exponential backoff (base: 1s, max: 60s)
- Circuit breaker (threshold: 5 failures, timeout: 60s)
- Rate limiter respect provider limits
- Correlation IDs for tracing
- Comprehensive logging (attempt N of M, duration, result)
- Error categorization (transient vs permanent)
- Tests: success, rate limit, timeout, connection error
- 95%+ coverage
- Performance benchmark: < 5s for capture

Reference: See ERROR-001 (timeout) and Retry pattern in .cursorrules, look up provider API in decision_catalog.md
```

**Token Cost:** ~4-5 requests (complex integration)

---

## Testing Prompts

### PROMPT-TEST-001: Generate Unit Tests

**Use When:** Need comprehensive tests for a function

**Prompt:**

```
@codebase Generate comprehensive unit tests for {FunctionName} in {file}.

Function Purpose: {brief_description}

Test Cases Needed:
- Happy path: {scenario1}
- Error case: {error_type1}
- Edge case: {boundary1}
- Edge case: {boundary2}

Requirements:
- Use pytest with Arrange-Act-Assert pattern
- Mock external dependencies
- Test success path
- Test all error paths
- Test edge cases (empty, null, max, min)
- Parametrize similar test cases
- 95%+ coverage verification
- Clear test names explaining what's tested
- Docstrings explaining why each test matters

File: tests/test_{function_name}.py

Reference: See testing requirements in testing.mdc
```

**Token Cost:** ~1-2 requests

---

### PROMPT-TEST-002: Generate Integration Tests

**Use When:** Need to test with real dependencies

**Prompt:**

```
@codebase Generate integration tests for {ComponentName}.

Integration Points:
- Database: {connection}
- External API: {endpoint}
- Message queue: {queue_name}

Requirements:
- Use async pytest fixtures for setup/teardown
- Test with actual database (or test DB)
- Mock external APIs with responses
- Test happy path end-to-end
- Test error path (external service fails)
- Test timeout handling
- Test retry logic
- Verify logging output
- 85%+ coverage (lower than unit because of setup cost)

File: tests/integration/test_{component_name}.py

Reference: See integration test patterns in testing.mdc
```

**Token Cost:** ~2-3 requests

---

### PROMPT-TEST-003: Generate Performance Tests

**Use When:** Need to verify performance benchmarks

**Prompt:**

```
@codebase Generate performance tests for {FunctionName}.

Performance Requirements:
- Latency p95: < {target_ms}ms
- Throughput: > {target_req/s} requests/second
- Memory: < {target_mb}mb

Requirements:
- Measure latency (p50, p95, p99)
- Measure throughput
- Run with realistic data volume ({data_size} items)
- Generate performance report
- Fail test if benchmarks not met
- Use pytest-benchmark or similar
- Log results to SYSTEM_STATUS.md

File: tests/performance/test_{function_name}_perf.py

Reference: See performance benchmarks in testing.md
```

**Token Cost:** ~2 requests

---

## Debugging Prompts

### PROMPT-DEBUG-001: Diagnose Error

**Use When:** Need help figuring out what's wrong

**Prompt:**

```
@codebase Help diagnose this error.

Error Message:
{paste full error text and stack trace}

Context:
- What was the user trying to do? {context}
- What's the expected behavior? {expected}
- What actually happened? {actual}

Requirements:
- Read error_catalog.md for similar errors
- Identify root cause (not just symptom)
- Provide specific diagnostic steps
- Suggest fix with success probability
- Provide test to verify fix
- Update error_catalog.md if new error type

Reference: See debugging decision tree in error_catalog.md
```

**Token Cost:** ~3-4 requests (complex diagnosis)

**Note:** This prompt is expensive but worth it because Cursor needs deep analysis.

---

## Refactoring Prompts

### PROMPT-REFACTOR-001: Improve Code Quality

**Use When:** Code works but could be cleaner

**Prompt:**

```
@codebase Refactor {FunctionName} for better code quality.

Current Issues:
- {issue1}
- {issue2}
- {issue3}

Requirements:
- Improve readability
- Reduce complexity (target: cyclomatic complexity < 10)
- Add type hints where missing
- Add docstrings where missing
- Extract magic numbers to constants
- Break into smaller functions if > 50 lines
- Update tests if behavior changes
- Maintain 95%+ coverage
- Verify functionality unchanged

File: {file}

Reference: See code quality standards in python_standards.mdc
```

**Token Cost:** ~2 requests

---

### PROMPT-REFACTOR-002: Reduce Technical Debt

**Use When:** Code has shortcuts that need fixing

**Prompt:**

```
@codebase Eliminate technical debt in {ComponentName}.

Known Issues:
- {issue1}
- {issue2}
- {issue3}

Requirements:
- Identify all technical debt
- Fix in priority order (breaking > security > performance > cleanup)
- Add type hints to untyped code
- Add error handling to bare exceptions
- Replace manual retry with circuit breaker
- Add logging where missing
- Add tests for unclear code
- Update documentation
- Maintain 95%+ coverage
- No breaking API changes

Reference: Technical debt cleanup checklist in .cursorrules
```

**Token Cost:** ~3-5 requests (might need multiple changes)

---

## Optimization Prompts

### PROMPT-OPT-001: Optimize Slow Operation

**Use When:** Something is too slow

**Prompt:**

```
@codebase Optimize {FunctionName} - currently taking {current_time}ms, target: {target_time}ms.

Current Implementation: [See {file}:{line}]

Requirements:
- Analyze current complexity (is it O(n²)? O(n*m)?)
- Identify bottleneck (is it DB query? loops? calculation?)
- Implement fix (add index? add caching? optimize algorithm?)
- Verify improvement with benchmark
- Maintain correctness (no behavior changes)
- Update SYSTEM_STATUS.md with performance improvement
- Add performance test to prevent regression

Reference: Performance optimization patterns in decision_catalog.md
```

**Token Cost:** ~2-3 requests

---

### PROMPT-OPT-002: Add Caching

**Use When:** Operation hits database repeatedly

**Prompt:**

```
@codebase Add caching to {FunctionName} for {data_type}.

Current: Queries database on every call

Requirements:
- Implement Redis caching (TTL: {ttl} seconds)
- Invalidate cache on write operations
- Publish event when cache invalidated
- Log cache hits/misses for metrics
- Add tests verifying cache behavior
- Benchmark showing improvement
- Handle cache miss gracefully

Caching strategy: See caching decisions in decision_catalog.md

File: {file}
```

**Token Cost:** ~2-3 requests

---

## Quick Reference Prompts (1 Token Wonders)

### PROMPT-QUICK-001: Add Type Hints

**Prompt:**

```
@codebase Add type hints to {FunctionName} in {file}. Use strict typing, no Any without justification.
```

### PROMPT-QUICK-002: Add Docstring

**Prompt:**

```
@codebase Add Google-style docstring to {FunctionName} in {file}. Include Args, Returns, Raises, Examples.
```

### PROMPT-QUICK-003: Add Error Handling

**Prompt:**

```
@codebase Add error handling to {FunctionName}. Catch specific exceptions, log with context, raise meaningful errors.
```

### PROMPT-QUICK-004: Add Logging

**Prompt:**

```
@codebase Add structured logging to {FunctionName}. Log operation start/end, errors, duration, metrics.
```

---

## Meta-Prompts (Strategic)

### PROMPT-META-001: Review Code for Issues

**Prompt:**

```
@codebase Review {FunctionName} for architectural issues, security problems, or performance bugs. Use code_review checklist in .cursorrules.
```

### PROMPT-META-002: Prepare for Production

**Prompt:**

```
@codebase Prepare {ComponentName} for production. Verify: type checking, tests (95%+ coverage), linting (8.0+), security scan (0 issues), performance (benchmarks met).
```

### PROMPT-META-003: Document Architecture

**Prompt:**

```
@codebase Document the architecture of {ComponentName}. Include: problem statement, solution chosen, tradeoffs, failure modes, testing strategy. Add to docs/architecture_decision_records/.
```

---

## Real-World Example Workflow

**Scenario:** You need to create a new feature that captures conversations from Perplexity API.

**Workflow:**

1. **Brainstorm** (use Perplexity)

   - "What's the best way to capture conversations from Perplexity?"
   - Get concepts, approaches

2. **Plan** (use Perplexity)

   - "Walk me through implementing this step-by-step"
   - Get understanding

3. **Create** (use Cursor + PROMPT-FEAT-002)

   ```
   @codebase Implement conversation capture for Perplexity following Event-Driven CQRS pattern.
   [Fill in details from Perplexity research]
   ```

4. **Test** (use PROMPT-TEST-001 + PROMPT-TEST-002)

   ```
   @codebase Generate comprehensive unit tests and integration tests for PerplexityConversationCapture.
   ```

5. **Review** (use PROMPT-META-001)

   ```
   @codebase Review PerplexityConversationCapture for architectural issues, security, performance.
   ```

6. **Deploy** (use PROMPT-META-002)
   ```
   @codebase Prepare PerplexityConversationCapture for production.
   ```

**Total Cursor Tokens:** ~6-8 requests = production-grade feature

---

## Version History

- v1.0 (2024-12-01): Initial prompt library with 20+ optimized prompts
