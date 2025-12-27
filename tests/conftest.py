"""
Pytest Configuration and Shared Fixtures
=========================================

Provides reusable fixtures for all test modules.

Author: ARC SAGA Development Team
Date: December 14, 2025
Status: Phase 2 Week 1 - Testing Foundation
"""

import asyncio
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from saga.config.sagacodex_profiles import LanguageProfile, SagaCodexManager
from saga.config.sagarules_embedded import EscalationContext, SagaConstitution
from saga.core.mimiry import Mimiry
from saga.core.task import Task
from saga.core.task_graph import TaskGraph
from saga.core.warden.agent import Warden

# ============================================================
# ASYNC SUPPORT
# ============================================================

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================
# CONSTITUTION FIXTURES
# ============================================================

@pytest.fixture
def constitution() -> SagaConstitution:
    """Provide SagaConstitution instance."""
    return SagaConstitution()


@pytest.fixture
def escalation_context_safe() -> EscalationContext:
    """
    Provide safe EscalationContext (SAGA can act autonomously).

    All conditions met for autonomous action:
    - All agents agree
    - SagaCodex aligned
    - Warden approves
    - No conflicts
    - High confidence (80%)
    - Budget OK
    - No secrets
    """
    return EscalationContext(
        conflict_detected=False,
        saga_confidence=80.0,
        llm_disagreement=False,
        user_requested_escalation=False,
        affects_multiple_systems=False,
        all_agents_agree=True,
        sagacodex_aligned=True,
        warden_approves=True,
        no_conflicts=True,
        budget_exceeded=False,
        secrets_detected=False,
    )


@pytest.fixture
def escalation_context_unsafe() -> EscalationContext:
    """
    Provide unsafe EscalationContext (SAGA must escalate).

    Multiple red flags:
    - Low confidence (60%)
    - Conflict detected
    - Budget exceeded
    """
    return EscalationContext(
        conflict_detected=True,
        saga_confidence=60.0,
        budget_exceeded=True,
    )


@pytest.fixture
def escalation_context_secrets() -> EscalationContext:
    """Provide CRITICAL context with secrets detected."""
    return EscalationContext(
        secrets_detected=True,
    )


# ============================================================
# SAGACODEX FIXTURES
# ============================================================

@pytest.fixture
def codex_manager() -> SagaCodexManager:
    """Provide SagaCodexManager instance."""
    return SagaCodexManager()


@pytest.fixture
def python_fastapi_codex(codex_manager: SagaCodexManager):
    """Provide Python/FastAPI SagaCodex profile."""
    return codex_manager.get_profile(LanguageProfile.PYTHON_FASTAPI)


# ============================================================
# MIMIRY FIXTURES
# ============================================================

@pytest.fixture
def mimiry() -> Mimiry:
    """Provide Mimiry oracle instance."""
    return Mimiry()


@pytest.fixture
def sample_code_good() -> str:
    """Provide well-formed code sample."""
    return '''
async def get_user(
    db: AsyncSession,
    user_id: str
) -> Optional[User]:
    """Fetch user by ID."""
    try:
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()
    except SQLAlchemyError as e:
        logger.error(
            "Failed to fetch user",
            extra={"user_id": user_id, "error": str(e)}
        )
        raise
'''


@pytest.fixture
def sample_code_bad() -> str:
    """Provide poorly-formed code sample with violations."""
    return '''
def get_user(db, user_id):
    print("Fetching user:", user_id)
    try:
        return db.query(User).filter(User.id == user_id).first()
    except:
        pass
'''


# ============================================================
# WARDEN FIXTURES
# ============================================================

@pytest.fixture
def warden() -> Warden:
    """Provide Warden instance."""
    return Warden()


@pytest.fixture
def simple_task() -> Task:
    """Provide simple task example."""
    return Task(
        id="task-simple-001",
        description="Create GET /users/{id} endpoint",
        weight="simple",
        budget_allocation=15.0,
        checklist=[
            "Pydantic model for response",
            "Async database query",
            "404 handling",
            "Structured logging",
        ],
        trace_id="trace-test-001"
    )


@pytest.fixture
def complex_task() -> Task:
    """Provide complex task example."""
    return Task(
        id="task-complex-001",
        description="Create POST /users/ endpoint with authentication",
        weight="complex",
        budget_allocation=50.0,
        checklist=[
            "Pydantic validation",
            "Password hashing",
            "JWT token generation",
            "Error handling",
            "Structured logging",
        ],
        vetting_criteria={
            "mypy_strict": True,
            "test_coverage": 99,
            "security_scan": True,
            "api_documented": True,
        },
        trace_id="trace-test-002"
    )


@pytest.fixture
def task_graph() -> TaskGraph:
    """Provide TaskGraph with sample tasks."""
    graph = TaskGraph()

    t1 = Task(
        id="task-001",
        description="Task 1",
        weight="simple",
        budget_allocation=10.0,
    )
    graph.add_task(t1)

    t2 = Task(
        id="task-002",
        description="Task 2",
        weight="complex",
        budget_allocation=20.0,
        dependencies=["task-001"],
    )
    graph.add_task(t2)

    return graph


@pytest.fixture
def agent_outputs_conflicting() -> list[dict[str, Any]]:
    """Provide conflicting agent outputs for testing."""
    return [
        {
            "agent": "AgentA",
            "approach": "Use print() for logging",
            "code": "print('User created:', user_id)"
        },
        {
            "agent": "AgentB",
            "approach": "Use structured logging",
            "code": "logger.info('User created', extra={'user_id': user_id})"
        }
    ]


@pytest.fixture
def agent_outputs_aligned() -> list[dict[str, Any]]:
    """Provide aligned agent outputs (both correct)."""
    return [
        {
            "agent": "AgentA",
            "approach": "Use structured logging",
            "code": "logger.info('User created', extra={'user_id': user_id})"
        },
        {
            "agent": "AgentB",
            "approach": "Use structured logging with trace_id",
            "code": "logger.info('User created', extra={'user_id': user_id, 'trace_id': trace_id})"
        }
    ]


# ============================================================
# MOCK FIXTURES
# ============================================================

@pytest.fixture
def mock_llm_response() -> dict[str, Any]:
    """Provide mock LLM API response."""
    return {
        "choices": [
            {
                "message": {
                    "content": "This is a test response from the LLM."
                }
            }
        ]
    }


@pytest.fixture
def mock_mimiry(monkeypatch: Any) -> AsyncMock:
    """Provide mocked Mimiry for testing Warden without real oracle calls."""
    mock = AsyncMock(spec=Mimiry)

    # Mock consult_on_discrepancy
    async def mock_consult(*args: Any, **kwargs: Any):
        from saga.core.mimiry import OracleResponse
        return OracleResponse(
            question=kwargs.get("question", "test question"),
            canonical_answer="Mock canonical answer from SagaCodex.",
            cited_rules=[1, 2, 3],
            severity="ACCEPTABLE",
            oracle_confidence=95.0,
        )

    mock.consult_on_discrepancy = mock_consult

    # Mock resolve_conflict
    async def mock_resolve(*args: Any, **kwargs: Any):
        from saga.core.mimiry import ConflictResolution
        return ConflictResolution(
            conflict_description="Mock conflict",
            conflicting_approaches=["A", "B"],
            canonical_approach="B is correct",
            rationale="Mock rationale",
            cited_rules=[4],
            agents_in_alignment=["AgentB"],
            agents_in_violation=["AgentA"],
        )

    mock.resolve_conflict = mock_resolve

    return mock


# ============================================================
# TEMP DIRECTORY FIXTURES
# ============================================================

@pytest.fixture
def temp_project_dir(tmp_path: Path) -> Path:
    """Provide temporary project directory for file operations."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()

    # Create basic structure
    (project_dir / "src").mkdir()
    (project_dir / "tests").mkdir()
    (project_dir / "docs").mkdir()

    return project_dir


# ============================================================
# CLEANUP
# ============================================================

@pytest.fixture(autouse=False)
def reset_singletons() -> Generator[None, None, None]:
    """Reset singleton instances between tests."""
    # Reset SagaCodexManager singleton
    import saga.config.sagacodex_profiles as profiles_module
    # Access private _manager if it exists/exposed, checking implementation first
    if hasattr(profiles_module, '_manager'):
        profiles_module._manager = None # type: ignore

    yield

    # Cleanup after test
    if hasattr(profiles_module, '_manager'):
        profiles_module._manager = None # type: ignore


@pytest.fixture(autouse=True)
def reset_health_monitor() -> Generator[None, None, None]:
    """
    Reset HealthMonitor state between tests.

    Ensures that metrics, circuit breakers, and caches from one test
    do not leak into others.
    """
    from saga.api import health_monitor as hm_module

    # Reset if method exists (added in fix)
    if hasattr(hm_module.health_monitor, "reset"):
        hm_module.health_monitor.reset()
    else:
        # Fallback for older versions or if method removed
        hm_module.health_monitor.clear_cache()
        if hasattr(hm_module.health_monitor, "_circuit_breakers"):
             hm_module.health_monitor._circuit_breakers.clear()
        if hasattr(hm_module.health_monitor, "_endpoint_latencies"):
             hm_module.health_monitor._endpoint_latencies.clear()

    yield

    # Optional: cleanup after test too
    if hasattr(hm_module.health_monitor, "reset"):
        hm_module.health_monitor.reset()

