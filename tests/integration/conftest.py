"""
Integration Test Fixtures (Mocked Mimiry)
"""

from unittest.mock import AsyncMock

import pytest

from saga.core.mimiry import ConflictResolution, OracleResponse
from saga.core.warden import Warden


@pytest.fixture
def integration_warden(monkeypatch):
    """Warden with mocked Mimiry for integration."""
    warden = Warden()

    # Mock Mimiry for deterministic tests
    mock_mimiry = AsyncMock()
    mock_consult = AsyncMock(return_value=OracleResponse(
        question="mock", canonical_answer="Approved", severity="ACCEPTABLE", cited_rules=[1]
    ))
    mock_resolve = AsyncMock(return_value=ConflictResolution(
        conflict_description="mock",
        conflicting_approaches=["Code A", "Code B"],
        canonical_approach="AgentB (structured logging)",
        rationale="Structured logging is required.",
        cited_rules=[4],
        agents_in_alignment=["AgentB"],
        agents_in_violation=["AgentA"]
    ))
    mock_mimiry.consult_on_discrepancy = mock_consult
    mock_mimiry.resolve_conflict = mock_resolve
    mock_mimiry.measure_against_ideal = AsyncMock(return_value=OracleResponse(
        question="mock measurement",
        canonical_answer="Perfection",
        cited_rules=[],
        severity="ACCEPTABLE"
    ))

    monkeypatch.setattr(warden, 'mimiry', mock_mimiry)
    return warden

@pytest.fixture
def conflicting_agents():
    """Mock agent outputs for conflict test."""
    return [
        {"agent": "AgentA", "code": "print('User created')"},
        {"agent": "AgentB", "code": "logger.info('User created', extra={'trace_id': trace_id})"}
    ]
