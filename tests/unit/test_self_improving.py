"""
Tests for Self-Improving Agent Loop (SICA-Style)

Author: ARC SAGA Development Team (AntiGravity)
Date: December 26, 2025
"""

from unittest.mock import AsyncMock, Mock

import pytest

from saga.core.memory.self_improving import (
    EditType,
    ScaffoldingEdit,
    SelfImprovingReflector,
)

# ═══════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def mock_optimizer():
    return Mock()

@pytest.fixture
def mock_graph():
    return Mock()

@pytest.fixture
def mock_ripple():
    return AsyncMock()

@pytest.fixture
def reflector(mock_optimizer, mock_graph, mock_ripple):
    reflector = SelfImprovingReflector(
        optimizer=mock_optimizer,
        graph=mock_graph,
        ripple_simulator=mock_ripple,
        enable_reflection=True  # Important: Enable for testing
    )
    # Default behavior: mock sandbox utility delta
    # Since we can't easily mock inner async calls without heavy patching,
    # we'll patch _test_proposal_in_sandbox instead in specific tests
    return reflector

# ═══════════════════════════════════════════════════════════════
# REFLECTION TRIGGER TESTS
# ═══════════════════════════════════════════════════════════════

def test_trigger_disabled_by_default(mock_optimizer, mock_graph):
    """Test global enable flag works."""
    rf = SelfImprovingReflector(mock_optimizer, mock_graph, enable_reflection=False)
    assert rf.should_trigger_reflection({"user_feedback": 1}) is False

def test_trigger_high_signal_only(reflector):
    """Test only high-signal events trigger reflection."""
    # Low signal
    assert reflector.should_trigger_reflection({"type": "task_complete"}) is False

    # High signal (User feedback)
    assert reflector.should_trigger_reflection({"user_feedback": 1}) is True

    # High signal (Token inefficiency)
    assert reflector.should_trigger_reflection({"token_inefficiency": 0.35}) is True

def test_session_limit(reflector):
    """Test session proposal limit."""
    reflector.proposals_this_session = 3  # Max
    assert reflector.should_trigger_reflection({"force_reflection": True}) is False

# ═══════════════════════════════════════════════════════════════
# PROPOSAL SCOPE TESTS
# ═══════════════════════════════════════════════════════════════

def test_proposal_generation_scope(reflector):
    """Test generated proposals are within allowed scope."""
    context = {"missing_context_type": "some_context"}
    proposals = reflector._generate_proposals(None, context)

    assert len(proposals) > 0
    for p in proposals:
        assert isinstance(p.edit_type, EditType)
        # Should be one of the allowed types
        assert p.edit_type in [EditType.NEW_EDGE_TYPE, EditType.WEIGHT_TWEAK, EditType.CACHE_POLICY]

# ═══════════════════════════════════════════════════════════════
# REFLECTION LOOP TESTS (ASYNC)
# ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_reflection_loop_veto_logic(reflector):
    """Test complete loop including mandatory veto."""

    # 1. Mock Proposal
    mock_proposal = ScaffoldingEdit(
        edit_id="test_edit",
        edit_type=EditType.CACHE_POLICY,
        target="ttl",
        old_value=1,
        new_value=2,
        rationale="test",
        confidence=0.9
    )

    reflector._generate_proposals = Mock(return_value=[mock_proposal])

    # 2. Mock Sandbox Test (High Utility)
    reflector._test_proposal_in_sandbox = AsyncMock(return_value=0.35)  # 35% > 30% threshold

    # 3. Mock Veto (REJECTED)
    reflector._request_human_veto = AsyncMock(return_value=False)

    # Run Reflection
    result = await reflector.reflect_on_outcome(None, {"force_reflection": True})

    # Assertions
    assert result.proposals_tested == 1
    assert result.proposals_applied == 0
    assert "test_edit" in result.veto_required  # Should be flagged as requiring/failing veto

@pytest.mark.asyncio
async def test_reflection_loop_approval_logic(reflector):
    """Test loop with approval."""

    mock_proposal = ScaffoldingEdit(
        edit_id="test_edit_approved",
        edit_type=EditType.CACHE_POLICY,
        target="ttl",
        old_value=1,
        new_value=2,
        rationale="test",
        confidence=0.9
    )

    reflector._generate_proposals = Mock(return_value=[mock_proposal])
    reflector._test_proposal_in_sandbox = AsyncMock(return_value=0.35)

    # APPROVED
    reflector._request_human_veto = AsyncMock(return_value=True)

    result = await reflector.reflect_on_outcome(None, {"force_reflection": True})

    assert result.proposals_applied == 1
    assert reflector.proposals_this_session == 1

@pytest.mark.asyncio
async def test_utility_threshold(reflector):
    """Test strict utility threshold (30%)."""

    mock_proposal = ScaffoldingEdit(
        edit_id="test_edit_mediocre",
        edit_type=EditType.CACHE_POLICY,
        target="ttl",
        old_value=1,
        new_value=2,
        rationale="test",
        confidence=0.9
    )

    reflector._generate_proposals = Mock(return_value=[mock_proposal])

    # 25% Utility (Below 30% threshold)
    reflector._test_proposal_in_sandbox = AsyncMock(return_value=0.25)

    result = await reflector.reflect_on_outcome(None, {"force_reflection": True})

    assert result.proposals_tested == 1
    assert result.proposals_applied == 0
    # Should NOT have even requested veto
    assert len(result.veto_required) == 0

# ═══════════════════════════════════════════════════════════════
# SAFETY BOUNDS TESTS
# ═══════════════════════════════════════════════════════════════

def test_benchmark_suite_integrity(reflector):
    """Test benchmark suite is loaded."""
    assert len(reflector._benchmark_queries) == 10
    assert "auth flow" in reflector._benchmark_queries
