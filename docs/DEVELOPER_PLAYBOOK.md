# Developer Playbook: Recovering from Test Regressions

## Context

This guide documents the steps taken to recover from the "Dec 24 Regression" involving strict dependency conflicts (`httpx` versioning) and state pollution in E2E tests. Use these steps if similar regressions occur.

## 1. Stabilize Environment

If tests fail due to dependency version mismatches (e.g., `AsyncClient` errors):

1. **Create Isolated Environment**:

   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. **Pin Dependencies**:
   Use `requirements-dev.txt` to strictly pin problematic versions while keeping `pyproject.toml` loose for library consumers.

   ```text
   # requirements-dev.txt
   httpx==0.23.3
   httpcore==0.16.3
   pytest
   pytest-asyncio
   # ... add runtime deps
   ```

3. **Install**:
   ```powershell
   pip install -r requirements-dev.txt
   pip install -e .
   ```

## 2. Verify Fixes (Smoke Test)

Run these critical tests to verify stability:

```powershell
# 1. Unit Test for Registry (Executors)
pytest tests/unit/orchestrator/test_registry_integration.py -vv

# 2. Unit Test for Response Mode
pytest tests/unit/orchestration/test_response_mode.py -vv

# 3. E2E Test (Stateful)
pytest tests/integration/test_orchestration_e2e.py -vv
```

## 3. Handle State Pollution

If E2E tests are flaky or see "ghost" metrics:

- Ensure the singleton `HealthMonitor` is reset between tests.
- Check `tests/conftest.py` for the `reset_health_monitor` autouse fixture.
- If debugging, compare object IDs:
  ```python
  from saga.api import server, health_monitor
  print(id(server.health_monitor), id(health_monitor.health_monitor))
  ```

## 4. Snapshot & Tag

When stabilized:

```powershell
git add .
git commit -m "Fix(regression): <summary>"
git tag recovery/<date>
```
