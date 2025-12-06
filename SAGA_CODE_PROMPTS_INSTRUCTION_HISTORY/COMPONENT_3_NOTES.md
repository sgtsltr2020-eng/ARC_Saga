# ARC SAGA – Component 3 ProviderRouter Notes

**Date:** December 5, 2025  
**Status:** Ready for Implementation  
**Based On:** Component 2 ✅ Complete (88/88 tests passing)

---

## 1. Context and Goals

This document captures the planning, design discussion, and concrete instructions for implementing **Phase 2.4 Component 3: ProviderRouter** on top of the existing ARC SAGA orchestration stack (Phase 2.3 + Components 1–2 of Phase 2.4).

### Goals

- Integrate a **ProviderRouter** that:
  - Uses the existing `IReasoningEngine` protocol (no fork).
  - Uses `ReasoningEngineRegistry` for provider → engine lookup.
  - Implements deterministic fallback and clear error taxonomy.
  - Emits structured logging suitable for debugging and observability.
- Maintain all Phase 2.3 and Components 1–2 guarantees:
  - No protocol breaking changes.
  - All tests stay green.
  - Type safety and coverage meet the existing quality gates.

---

## 2. Integration Scope and Files

### 2.1 Files to Create

| File                                               | Purpose                                                     | Lines (est) |
| -------------------------------------------------- | ----------------------------------------------------------- | ----------- |
| `arc_saga/orchestrator/errors.py`                  | ProviderError, TransientError, PermanentError               | ~30         |
| `arc_saga/orchestrator/provider_router.py`         | ProviderRouter, RoutingRule, AttemptRecord, RouteProvenance | ~250        |
| `tests/unit/orchestration/test_provider_router.py` | Full test matrix (20+ tests)                                | ~300        |

### 2.2 Files NOT to Modify

- Phase 2.3 files (no changes)
- Component 1 files (no changes)
- Component 2 files (no changes)
- `arc_saga/orchestrator/protocols.py` (import only, no changes)
- `arc_saga/orchestrator/engine_registry.py` (use only, no changes)

---

## 3. Key Design Constraints

### 3.1 Protocol Alignment

`IReasoningEngine` is defined in `arc_saga.orchestrator.protocols` (import from this module; do not re-export via the registry). ProviderRouter must:

- Import from `protocols`, not redefine.
- Rely on mypy `--strict` for conformance enforcement.
- Optionally validate at runtime via `supports()` checks.

### 3.2 Registry Alignment

`ReasoningEngineRegistry` is a singleton with class-level state and `AIProvider` keys:

```python
# Correct usage in ProviderRouter
from arc_saga.orchestrator.engine_registry import ReasoningEngineRegistry
from arc_saga.orchestrator.types import AIProvider

engine = ReasoningEngineRegistry.get(AIProvider.COPILOT_CHAT)
```

Do NOT instantiate the registry (`ReasoningEngineRegistry()`) in production code—use classmethods only. Tests may rely on a clean registry fixture to isolate state but should still avoid constructing instances directly.

### 3.3 Async-First, No Shared Mutable State

- Router must be safe in async contexts.
- May hold routing rules and configuration.
- Must not mutate shared global data structures.

### 3.4 Fallback Separation of Concerns

- Registry: provider → engine mapping only.
- Router: fallback chain, retry behavior, error classification.

---

## 4. Error Taxonomy

### 4.1 Error Classes (`errors.py`)

```python
# arc_saga/orchestrator/errors.py
from __future__ import annotations


class ProviderError(Exception):
    """Base exception for provider routing and engine execution errors."""


class TransientError(ProviderError):
    """Retryable errors: timeouts, rate limits, temporary network issues."""


class PermanentError(ProviderError):
    """Non-retryable errors: misconfiguration, unsupported ops, hard failures."""
```

### 4.2 Router Behavior by Error Type

| Error Type          | Router Action                                                      |
| ------------------- | ------------------------------------------------------------------ |
| `TransientError`    | Retry with backoff up to `max_retries`, then move to next provider |
| `PermanentError`    | Log at ERROR, move immediately to next provider                    |
| Unknown `Exception` | Configurable: treat as transient (default) or permanent            |

---

## 5. Data Structures

### 5.1 RoutingRule

```python
@dataclass(frozen=True)
class RoutingRule:
    task_types: Set[str]
    ordered_providers: List[AIProvider]
    max_retries: int = 2
    base_backoff_seconds: float = 0.2
    max_backoff_seconds: float = 1.0
```

### 5.2 AttemptRecord

```python
@dataclass
class AttemptRecord:
    provider: AIProvider
    attempt_index: int
    started_at: float
    finished_at: float
    outcome: str  # "success" | "transient_error" | "permanent_error" | "unsupported" | "exception"
    error_type: Optional[str] = None
    error_message: Optional[str] = None
```

### 5.3 RouteProvenance

```python
@dataclass
class RouteProvenance:
    task_type: str
    selected_rule: Optional[RoutingRule]
    candidate_providers: List[AIProvider]
    attempts: List[AttemptRecord]
    outcome: str  # "success" | "failed"
    chosen_provider: Optional[AIProvider]
    total_duration_seconds: float
    final_error_type: Optional[str] = None
    final_error_message: Optional[str] = None
```

---

## 6. ProviderRouter API

### 6.1 Constructor

```python
def __init__(
    self,
    rules: List[RoutingRule],
    default_order: Optional[List[AIProvider]] = None,
    classify_unknown_exceptions_as_transient: bool = True,
) -> None:
```

### 6.2 Public Methods

```python
def get_candidate_providers(self, task_type: str) -> List[AIProvider]:
    """Return ordered providers for task type based on rules or default."""

async def route(self, task: Mapping[str, Any], context: Mapping[str, Any]) -> Any:
    """Route task and return engine result. Raises ProviderError on failure."""

async def route_with_provenance(
    self, task: Mapping[str, Any], context: Mapping[str, Any]
) -> RouteProvenance:
    """Route task and return full provenance (all attempts, outcomes)."""
```

### 6.3 Private Helpers

- `_find_rule(task_type: str) -> Optional[RoutingRule]`
- `_retry_policy(rule: Optional[RoutingRule]) -> Tuple[int, float, float]`
- `_compute_backoff(attempt_index: int, base: float, max_backoff: float) -> float`
- `_get_engine(provider: AIProvider) -> IReasoningEngine`
- `_supports(engine: IReasoningEngine, task: Mapping[str, Any]) -> bool`

---

## 7. Routing Algorithm

```
1. Determine task_type from task
2. Find matching RoutingRule or use default_order
3. If no candidates: return failed RouteProvenance

4. For each provider in candidate list:
   a. Get engine from ReasoningEngineRegistry
   b. Check engine.supports(task)
      - If False/error: record "unsupported", continue to next provider

   c. For attempt_index in 1..max_retries+1:
      - Try engine.execute(task, context)
      - On success: record success, return RouteProvenance(outcome="success")
      - On PermanentError: record, break to next provider
      - On TransientError: record, sleep backoff, retry
      - On unknown Exception:
        - If classify_as_transient: retry like TransientError
        - Else: treat like PermanentError

5. If all providers exhausted: return RouteProvenance(outcome="failed")
```

---

## 8. Logging Specification

### 8.1 Required Log Events

| Level   | Event                      | Fields                                                                   |
| ------- | -------------------------- | ------------------------------------------------------------------------ |
| INFO    | `routing_start`            | task_type, correlation_id, candidate_providers, rule                     |
| INFO    | `routing_success`          | task_type, correlation_id, chosen_provider, attempts, latency_ms         |
| WARNING | `engine_transient_error`   | task_type, correlation_id, provider, attempt, message                    |
| WARNING | `engine_unknown_exception` | task_type, correlation_id, provider, attempt, transient, message         |
| ERROR   | `engine_permanent_error`   | task_type, correlation_id, provider, attempt, message                    |
| ERROR   | `engine_missing`           | provider, message                                                        |
| ERROR   | `routing_failed`           | task_type, correlation_id, tried, attempts, latency_ms, final_error_type |

### 8.2 Logging Rules

- Use `logging.getLogger(__name__)`
- Never log secrets, tokens, or credentials
- Include correlation_id for traceability; expected in `context["correlation_id"]`, fallback to `task["id"]` if missing
- Use structured fields (parseable)

---

## 9. Test Matrix

### 9.1 Test Scenarios (20+ tests)

| #   | Scenario                                                           | Expected Outcome                        |
| --- | ------------------------------------------------------------------ | --------------------------------------- |
| 1   | Primary provider succeeds first attempt                            | success, 1 attempt                      |
| 2   | Primary provider transient fails once, then succeeds               | success, 2 attempts same provider       |
| 3   | Primary provider permanent fails, fallback succeeds                | success, 2 providers                    |
| 4   | All providers transient fail exhaustively                          | failed                                  |
| 5   | All providers permanent fail                                       | failed                                  |
| 6   | Unknown exception as transient (retry)                             | success after retry or fallback         |
| 7   | Unknown exception as permanent (no retry)                          | fallback to next provider               |
| 8   | No matching rule, uses default_order                               | success via default                     |
| 9   | No rule and no default_order                                       | failed immediately                      |
| 10  | Engine unsupported (supports=False)                                | skipped, next provider tried            |
| 11  | Engine supports() raises                                           | skipped, next provider tried            |
| 12  | route() returns result on success                                  | dict with engine output                 |
| 13  | route() raises ProviderError on failure                            | ProviderError raised                    |
| 14  | Non-conforming engine registered                                   | handled gracefully, failed or logged    |
| 15  | Empty candidate list                                               | failed, no attempts                     |
| 16  | Correlation ID flows through logs                                  | caplog contains correlation_id          |
| 17  | Backoff timing is exponential                                      | delays increase correctly               |
| 18  | Max backoff is capped                                              | delay never exceeds max_backoff_seconds |
| 19  | Multiple rules, correct rule selected                              | task_type matches rule                  |
| 20  | Provenance contains all attempt records                            | attempts list complete                  |
| 21  | ProviderRouter instantiates registry instead of using classmethods | misuse detected (raise or log error)    |

### 9.2 Test Fixtures

```python
@pytest.fixture(autouse=True)
def clean_registry():
    """Singleton discipline: clear before and after each test."""
    ReasoningEngineRegistry.clear()
    yield
    ReasoningEngineRegistry.clear()

@pytest.fixture
def mock_engines():
    """Register mock engines for testing."""
    # SuccessEngine, TransientThenSuccessEngine, PermanentFailEngine, etc.
    # Register with AIProvider keys
```

---

## 10. Quality Gates

### 10.1 Commands

```bash
# Formatting
isort arc_saga tests
black arc_saga tests

# Type checking
mypy --strict arc_saga/orchestrator/errors.py arc_saga/orchestrator/provider_router.py
mypy --strict arc_saga

# Tests
pytest tests/unit/orchestration/test_provider_router.py -v
pytest tests/unit/orchestration/ -v
pytest tests/unit/ -v
```

### 10.2 Success Criteria

| Gate                   | Target                                                                           |
| ---------------------- | -------------------------------------------------------------------------------- |
| Type Safety            | mypy --strict: 0 errors                                                          |
| Component 3 Tests      | 20+/20+ passing                                                                  |
| Component 2 Regression | 16/16 still passing                                                              |
| Component 1 Regression | 11/11 still passing                                                              |
| Phase 2.3 Regression   | 61/61 still passing                                                              |
| Combined Tests         | 108+/108+ all passing                                                            |
| Coverage               | ≥ 95% for router logic (measured in CI via `pytest --cov=arc_saga/orchestrator`) |
| Formatting             | isort + black clean                                                              |

---

## 11. Watchpoints

### 11.1 Thread Safety

Current design is async-first. If multi-threading is introduced later:

- Guard registry access with locks
- Ensure router state is thread-safe

### 11.2 Engine Conformity

- Only route to engines implementing `IReasoningEngine`
- Include negative test for non-conforming engine
- mypy --strict enforces at compile time

### 11.3 route() Execution Semantics

Current: `route()` re-executes chosen engine to return result.

Future option: extend `RouteProvenance` to carry the result from `route_with_provenance()`, avoiding double execution.

### 11.4 Failure Mode Coverage

Tests must cover:

- Empty fallback list
- All providers fail (transient)
- All providers fail (permanent)
- Mixed transient/permanent failures
- Unknown exception classification both ways

### 11.5 Backoff Non-Blocking

- Router backoff must not block the event loop; use `asyncio.sleep` during retries to avoid blocking.

---

## 12. Out of Scope (Future Work)

Keep these OUT of the initial Component 3 implementation:

- Weighted routing and health signals
- Per-engine retry policies (beyond rule-level)
- Provenance persistence (to datastore)
- Circuit breakers for flapping engines
- Dynamic provider health checks

These can be added in Component 4 (CostOptimizer) or a future enhancement phase.

---

## 13. Implementation Checklist

### Pre-Implementation

- [ ] Component 2 complete (88/88 tests passing)
- [ ] Review this document
- [ ] Understand registry singleton pattern
- [ ] Understand IReasoningEngine protocol location

### File Creation

- [ ] Create `arc_saga/orchestrator/errors.py`
- [ ] Create `arc_saga/orchestrator/provider_router.py`
- [ ] Create `tests/unit/orchestration/test_provider_router.py`

### Implementation

- [ ] Implement error classes (3)
- [ ] Implement RoutingRule dataclass
- [ ] Implement AttemptRecord dataclass
- [ ] Implement RouteProvenance dataclass
- [ ] Implement ProviderRouter class
- [ ] Implement all public methods
- [ ] Implement all private helpers
- [ ] Add comprehensive logging
- [ ] Add docstrings to all public APIs (Google or NumPy style, consistent with Component 2)

### Testing

- [ ] Implement test fixtures
- [ ] Implement all 20+ test scenarios
- [ ] Verify logging with caplog
- [ ] Verify error handling paths

### Quality Gates

- [ ] mypy --strict: 0 errors
- [ ] All Component 3 tests passing
- [ ] All Component 2 tests still passing
- [ ] All Component 1 tests still passing
- [ ] All Phase 2.3 tests still passing
- [ ] isort + black clean
- [ ] Coverage ≥ 95%

### Post-Implementation

- [ ] All 108+ tests passing
- [ ] No regressions detected
- [ ] Ready for Component 4

---

## 14. Cursor Instruction Block

Copy this block for direct use in Cursor:

```
# CURSOR INSTRUCTION: PHASE 2.4 COMPONENT 3

## Scope
Implement ProviderRouter with fallback routing using ReasoningEngineRegistry.
No modifications to Phase 2.3 or Components 1-2.

## Files to Create
1. arc_saga/orchestrator/errors.py
   - ProviderError, TransientError, PermanentError

2. arc_saga/orchestrator/provider_router.py
   - RoutingRule, AttemptRecord, RouteProvenance, ProviderRouter
   - Import IReasoningEngine from protocols (not engine_registry)
   - Use ReasoningEngineRegistry classmethods with AIProvider keys

3. tests/unit/orchestration/test_provider_router.py
   - 20+ tests covering all scenarios
   - Use clean_registry fixture (clear before/after)
   - Use caplog for logging verification

## Commands
isort arc_saga tests
black arc_saga tests
mypy --strict arc_saga/orchestrator/errors.py arc_saga/orchestrator/provider_router.py
mypy --strict arc_saga
pytest tests/unit/orchestration/test_provider_router.py -v
pytest tests/unit/ -v

## Success Criteria
- 0 mypy errors
- 108+/108+ tests passing
- Coverage ≥ 95%
- No regressions
```

---

**Component 2: ✅ Complete (Dec 5)**  
**Component 3: 🔜 Ready for Implementation**  
**Phase 2.4 Progress:** 2/5 components complete (40%)

**Let's build Component 3.** 🚀
