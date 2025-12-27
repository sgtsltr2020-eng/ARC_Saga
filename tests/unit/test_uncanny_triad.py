"""
Tests for Uncanny Triad Upgrades
- Procedural Memory
- Information-Gain Rewards
- Multi-Agent Critique

Author: ARC SAGA Development Team (AntiGravity)
Date: December 26, 2025
"""

from unittest.mock import AsyncMock, Mock

import pytest

from saga.core.memory.evolutionary_oversight import SovereignOptimizer
from saga.core.memory.procedural_memory import ProceduralMemory
from saga.core.warden.consensus_protocol import ConsensusProtocol, CriticAgent, Critique, Vote

# ═══════════════════════════════════════════════════════════════
# PROCEDURAL MEMORY TESTS
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def mock_embedding_engine():
    m = Mock()
    # Return fake embedding [0.1, 0.2, ...]
    m.generate_embedding = AsyncMock(return_value=[0.1] * 64)
    return m

@pytest.fixture
def procedural_memory(mock_embedding_engine):
    return ProceduralMemory(embedding_engine=mock_embedding_engine)

@pytest.mark.asyncio
async def test_procedural_indexing_retrieval(procedural_memory, mock_embedding_engine):
    """Test standard index -> retrieve cycle."""
    # 1. Index an experience
    mid = await procedural_memory.index_experience(
        query="fix race condition",
        active_file="core.py",
        context_nodes=["func:lock"],
        outcome={"success": True, "summary": "Added mutex", "fix": "mutex.lock()"}
    )
    assert mid != ""
    assert len(procedural_memory.ids) == 1

    # 2. Retrieve with similar query
    # Mock embedding with slightly perturbed vector
    mock_embedding_engine.generate_embedding.return_value = [0.11] * 64

    results = await procedural_memory.retrieve_relevant(
        query="prevent race",
        active_file="core.py",
        context_nodes=["func:lock"],
        k=1
    )

    assert len(results) == 1
    memory, score = results[0]
    assert memory.memory_id == mid
    assert memory.applied_fix == "mutex.lock()"
    assert score > 0.95  # High similarity

@pytest.mark.asyncio
async def test_procedural_retrieval_empty(procedural_memory):
    """Test retrieval on empty index."""
    results = await procedural_memory.retrieve_relevant("test", "file.py", [])
    assert len(results) == 0

# ═══════════════════════════════════════════════════════════════
# INFORMATION-GAIN REWARDS TESTS
# ═══════════════════════════════════════════════════════════════

def test_novelty_calculation_bonus():
    """Test novelty bonus logic."""
    optimizer = SovereignOptimizer()

    # Case 1: No history -> Max Novelty
    bonus = optimizer.calculate_novelty_bonus([1.0], [])
    assert bonus == 2.0

    # Case 2: Identical history -> No Novelty
    hist = [[1.0, 0.0], [0.0, 1.0]]
    vec = [1.0, 0.0]  # Exact match with first item
    bonus = optimizer.calculate_novelty_bonus(vec, hist)
    assert bonus == pytest.approx(0.0, abs=1e-6)  # Distance is effectively 0

    # Case 3: Novel vector (Orthogonal)
    vec_ortho = [0.0, 0.0]  # Zero vector edge case or completely different
    # Let's use clean orthogonal examples
    h = [[1,0,0], [0,1,0]]
    v = [0,0,1]
    bonus = optimizer.calculate_novelty_bonus(v, h)
    # Distance of orthogonal vectors is 1.0
    # Logic: 2.0 + (1.0 - 0.6) = 2.4
    assert bonus == pytest.approx(2.4, 0.1)

def test_synthesis_utility_integration():
    """Test total utility calculation includes bonuses."""
    optimizer = SovereignOptimizer()

    # Mock predict_utility to return baseline 0.5
    optimizer.predict_utility = Mock(return_value=0.5)

    # Novelty bonus logic check
    # Vector [0,0,1] vs history [[1,0,0]] -> Distance 1.0 -> Bonus 2.4
    ctx = [0.0, 0.0, 1.0]
    hist = [[1.0, 0.0, 0.0]]

    # With LM Summary -> Adds heuristic bonus 0.5
    total = optimizer.synthesis_utility(ctx, hist, proposal_summary="Huge breakthrough")

    # Expected: 0.5 (Base) + 2.4 (Novelty) + 0.5 (LM) = 3.4
    assert total == pytest.approx(3.4, 0.1)

# ═══════════════════════════════════════════════════════════════
# MULTI-AGENT CONSENSUS TESTS
# ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_critic_intervention():
    """Test critic flags highly novel proposals without citations."""
    critic = CriticAgent()

    # Case 1: High novelty, no provenance -> REJECT
    ctx = {"novelty_score": 0.9}
    prop = {"provenance_citations": []}

    critique = await critic.evaluate_proposal(prop, ctx)
    assert critique.vote == Vote.REJECT
    assert "provenance" in critique.reasoning.lower()

    # Case 2: High novelty, with provenance -> APPROVE
    prop_good = {"provenance_citations": ["ref1"]}
    critique_good = await critic.evaluate_proposal(prop_good, ctx)
    assert critique_good.vote == Vote.APPROVE

@pytest.mark.asyncio
async def test_consensus_protocol_voting():
    """Test 2/3 majority requirement (triggered by high novelty)."""
    critic = CriticAgent()
    protocol = ConsensusProtocol(critic)

    # Mock Critic response to APPROVE
    critic.evaluate_proposal = AsyncMock(return_value=Critique("c1", Vote.APPROVE, "Good"))

    # Case 1: High Novelty -> Triggers Vote -> Planner=APPROVE, Coder=APPROVE, Critic=APPROVE -> Pass
    high_novelty_prop = {"context": {"novelty_score": 0.9}}
    passed = await protocol.request_consensus(high_novelty_prop, planner_vote=Vote.APPROVE)
    assert passed is True

    # Case 2: Mixed Vote -> Planner=APPROVE, Coder=APPROVE, Critic=REJECT -> 2/3 Pass
    critic.evaluate_proposal = AsyncMock(return_value=Critique("c1", Vote.REJECT, "Bad"))
    passed_mixed = await protocol.request_consensus(high_novelty_prop, planner_vote=Vote.APPROVE)
    assert passed_mixed is True

    # Case 3: Failed Vote -> Planner=REJECT, Coder=APPROVE, Critic=REJECT -> 1/3 Fail
    passed_fail = await protocol.request_consensus(high_novelty_prop, planner_vote=Vote.REJECT)
    assert passed_fail is False

@pytest.mark.asyncio
async def test_consensus_bypass_routine():
    """Test that routine tasks bypass consensus."""
    critic = CriticAgent()
    critic.evaluate_proposal = AsyncMock() # Should not be called
    protocol = ConsensusProtocol(critic)

    # Routine proposal (Low novelty)
    routine_prop = {"context": {"novelty_score": 0.1}}

    passed = await protocol.request_consensus(routine_prop)

    assert passed is True
    critic.evaluate_proposal.assert_not_called()
