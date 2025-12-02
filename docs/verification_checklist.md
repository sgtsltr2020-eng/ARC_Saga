# Verification Checklist

# The Quality Gate That Never Lets Bad Code Through

---

## 🎯 PRE-GENERATION CHECKLIST (Use Before Asking Cursor to Generate)

### Requirements Clarity

- [ ] Problem statement is crystal clear
- [ ] Constraints are documented (time, memory, tokens, etc.)
- [ ] Success criteria are specific and measurable
- [ ] Failure modes are identified
- [ ] Scale requirements documented (how many users/requests/records?)
- [ ] Performance requirements explicit (latency target, throughput target)
- [ ] Security requirements explicit (what data needs protection?)
- [ ] I've consulted decision_catalog.md for similar problems

### Edge Cases Identified

- [ ] What happens with empty/null input?
- [ ] What happens with maximum input?
- [ ] What happens with minimum input?
- [ ] What happens on concurrent access?
- [ ] What happens on rate limiting?
- [ ] What happens on timeout?
- [ ] What happens on external service failure?

### Testing Strategy Planned

- [ ] Unit tests for happy path
- [ ] Unit tests for all error paths
- [ ] Integration tests with real dependencies
- [ ] Edge case tests
- [ ] Performance tests with benchmarks
- [ ] Security tests (if applicable)
- [ ] Target coverage: 95%+

### Error Handling Planned

- [ ] How will this fail?
- [ ] How will we detect the failure?
- [ ] How will we handle the failure?
- [ ] How will we recover?
- [ ] What will we log?
- [ ] How will future devs debug this?

---

## 🔍 POST-GENERATION CHECKLIST (After Cursor Generates Code)

### Type Safety

- [ ] `mypy --strict` passes (0 errors)
- [ ] All function parameters have type hints
- [ ] All function returns have type hints
- [ ] No `Any` types without justification comment
- [ ] Generic types used where appropriate
- [ ] Optional types used correctly

### Error Handling

- [ ] All external calls wrapped in try-except
- [ ] Specific exceptions caught (not bare `Exception`)
- [ ] Meaningful error messages provided
- [ ] Errors logged with full context
- [ ] Circuit breaker for external calls
- [ ] Retry logic for transient failures
- [ ] Graceful degradation when possible

### Logging

- [ ] Operation start logged
- [ ] Operation end logged
- [ ] Errors logged with stack trace
- [ ] Performance metrics logged (duration)
- [ ] Correlation IDs included
- [ ] Structured JSON logging format
- [ ] No secrets in logs
- [ ] Appropriate log levels (INFO, WARNING, ERROR)

### Documentation

- [ ] Google-style docstrings present
- [ ] Type hints as documentation
- [ ] Comments for non-obvious logic
- [ ] Examples in docstrings (where applicable)
- [ ] Architectural decisions documented
- [ ] Failure modes documented

### Testing

- [ ] Unit tests written
- [ ] Integration tests written
- [ ] Edge case tests written
- [ ] Error path tests written
- [ ] All tests passing
- [ ] Coverage >= 95%
- [ ] Tests are maintainable (not brittle)

### Code Quality

- [ ] `black --check` passes (formatting)
- [ ] `isort --check` passes (imports organized)
- [ ] `pylint` score >= 8.0
- [ ] No unused imports
- [ ] No unused variables
- [ ] Functions <= 50 lines (or well-justified)
- [ ] Cyclomatic complexity < 10

### Security

- [ ] `bandit` scan passes (0 issues)
- [ ] All external input validated
- [ ] SQL queries parameterized
- [ ] Credentials from environment (never hardcoded)
- [ ] Secrets never logged
- [ ] Rate limiting in place (if applicable)
- [ ] OWASP top 10 considered

### Performance

- [ ] Latency target met (p95)
- [ ] No N+1 queries
- [ ] Caching where appropriate
- [ ] Database queries indexed
- [ ] Async/await used for I/O
- [ ] Performance test passes
- [ ] Metrics logged for monitoring

### Integration

- [ ] Works with existing architecture
- [ ] Follows established patterns (CQRS, Repository, etc.)
- [ ] Uses existing utilities and helpers
- [ ] No duplicate code
- [ ] Clean interfaces (not tightly coupled)
- [ ] No breaking API changes

### Production Readiness

- [ ] All checks above pass
- [ ] Code review approved
- [ ] Documentation complete
- [ ] No technical debt introduced
- [ ] No shortcuts taken
- [ ] Deployable immediately

---

## 🚨 DEPLOYMENT BLOCKERS (Never Deploy If ANY of These)

- [ ] Any test failing
- [ ] Coverage below 95%
- [ ] Type checking shows errors
- [ ] Linting score below 8.0
- [ ] Security scan shows issues
- [ ] Performance benchmarks not met
- [ ] Code review has blockers
- [ ] Documentation incomplete
- [ ] Unhandled error paths
- [ ] No logging for debugging
- [ ] External calls without circuit breaker
- [ ] Secrets in code or logs

---

## ✅ DEPLOYMENT READY (All of These Must Be True)

- [ ] All tests passing (100%)
- [ ] Coverage >= 95%
- [ ] Type checking passes (mypy --strict)
- [ ] Linting passes (pylint >= 8.0)
- [ ] Security scan clean (bandit 0 issues)
- [ ] Performance benchmarks met
- [ ] Code review approved
- [ ] Documentation complete
- [ ] Error handling comprehensive
- [ ] Logging adequate
- [ ] Architecture patterns followed
- [ ] No technical debt
- [ ] Production-ready quality

---

## 🎯 CODE REVIEW CHECKLIST

When reviewing code generated by Cursor:

### Architecture

- [ ] Follows chosen pattern (CQRS, Repository, Circuit Breaker, etc.)
- [ ] Bounded contexts clear
- [ ] Dependencies flow correctly
- [ ] No circular dependencies
- [ ] Interfaces clean and focused

### Logic

- [ ] Algorithm correct
- [ ] Edge cases handled
- [ ] Boundary conditions checked
- [ ] Off-by-one errors prevented
- [ ] No infinite loops

### Error Handling

- [ ] All failure modes covered
- [ ] Recovery strategies in place
- [ ] Errors propagate correctly
- [ ] Meaningful error messages
- [ ] Logging helps debugging

### Testing

- [ ] Happy path tested
- [ ] Error paths tested
- [ ] Edge cases tested
- [ ] Performance tested
- [ ] Security tested

### Performance

- [ ] No O(n²) algorithms where O(n) is possible
- [ ] No N+1 database queries
- [ ] Caching used appropriately
- [ ] Async/await for I/O
- [ ] No unnecessary allocations

### Security

- [ ] Input validated
- [ ] Credentials secure
- [ ] No SQL injection
- [ ] Rate limiting in place
- [ ] OWASP compliance

### Maintainability

- [ ] Code is clear and readable
- [ ] Functions have single responsibility
- [ ] Names are descriptive
- [ ] Comments explain why, not what
- [ ] Tests document behavior

---

## Version History

- v1.0 (2024-12-01): Initial comprehensive verification checklist
