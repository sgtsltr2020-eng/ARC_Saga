# Follow-up: Upgrade httpx and remove dev pin

**Type:** Maintenance / Technical Debt
**Priority:** P2
**Owner:** Unassigned

## Context

During the "Dec 24 Regression" triage, we pinned `httpx==0.23.3` in `requirements-dev.txt` to resolve specific `AsyncClient` compatibility issues with the current test suite. We need to upgrade the codebase to support modern `httpx` versions (currently 0.27+) to ensure security updates and compatibility with other libraries.

## Acceptance Criteria

- [ ] Remove `httpx` pin from `requirements-dev.txt`.
- [ ] Upgrade `httpx` to latest stable version (`pip install -U httpx`).
- [ ] Fix `AsyncClient` usage in `tests/integration/test_orchestration_e2e.py` (likely requires `transport` argument changes or lifecycle updates).
- [ ] Verify all tests pass: `pytest tests/integration/test_orchestration_e2e.py`.

## Estimated Effort

Small (2-4 hours)
