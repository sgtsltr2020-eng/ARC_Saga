# Release Notes: Post-Regression Fix (Dec 24, 2025)

**Tag:** `recovery/post-regression-fix-dec24`
**Branch:** `wip/save-dec21`

## Summary

Stabilized the development and test environment, fixed regressions in the Orchestrator executor, and resolved a flaky E2E test caused by `HealthMonitor` state leakage.
This update provides a pinned `requirements-dev.txt` for reproducible test runs and fixes API misalignments in the executor.

## Changes

### Environment & Dependencies

- **[NEW]** `requirements-dev.txt`: Added pinned dependencies to create an isolated test environment.
  - `httpx==0.23.3`: Fixes `AsyncClient` compatibility issues.
  - Runtime deps: `aiosqlite`, `scikit-learn`, `python-docx`, `python-multipart`, `pymupdf`, `cryptography`, `greenlet`, `aiohttp`, `google-generativeai`.

### Orchestrator

- **[FIX]** `saga/orchestrator/executor.py`:
  - Implemented `__call__` interface.
  - Updated `TokenBudgetManager` usage to `check_budget()`.
  - Fixed `AIResult` serialization bug affecting tests.

### Tests

- **[FIX]** `tests/integration/test_orchestration_e2e.py`: Added mechanism to reset `HealthMonitor` state between tests.
- **[NEW]** `tests/conftest.py`: Added `reset_health_monitor` autouse fixture for global test isolation.

## Verification

- Unit tests (`test_registry_integration.py`, `test_response_mode.py`) passed.
- Integration tests (`test_orchestration_e2e.py`) passed after fix.

## Known Issues

- `httpx` is pinned to `0.23.3` in the test environment. Upgrade to latest version is tracked as a follow-up item.
