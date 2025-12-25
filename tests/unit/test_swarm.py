"""
Unit Tests for Swarm
====================

Tests for the Swarm agent coordination system including
utility scoring, dropout, chameleon spawning, and coordination.

Author: ARC SAGA Development Team
Date: December 25, 2025
Status: Phase 8 - MAE Foundation
"""

import pytest

from saga.core.mae.governor import AgentMode, Governor
from saga.core.mae.swarm import (
    AgentDropout,
    AgentType,
    DropoutReason,
    SwarmCoordinator,
)

# ============================================================
# Test 1: Utility Score Calculation
# ============================================================

class TestUtilityScoreCalculation:
    """Test agent utility score calculation."""

    def test_utility_from_governor_metrics(self):
        """Test utility is calculated from Governor tracked metrics."""
        governor = Governor()
        dropout = AgentDropout(governor)

        # Track productive turns
        for _ in range(3):
            governor.track_turn("task-001", "agent-alpha", 1000, 50.0, 60.0)

        utility = dropout.calculate_utility("agent-alpha")

        # All turns were productive, should have high utility
        assert utility > 0.5

    def test_utility_zero_for_unknown_agent(self):
        """Test utility is 0 for unknown agent."""
        governor = Governor()
        dropout = AgentDropout(governor)

        utility = dropout.calculate_utility("unknown-agent")
        assert utility == 0.0

    def test_utility_from_provided_metrics(self):
        """Test utility calculated from provided metrics."""
        governor = Governor()
        dropout = AgentDropout(governor)

        from saga.core.mae.governor import TurnMetrics

        # Create metrics and properly complete them to set compliance_delta
        metrics = []

        # Productive turn 1
        m1 = TurnMetrics(compliance_before=50.0)
        m1.complete(tokens=1000, compliance_after=60.0)
        metrics.append(m1)

        # Productive turn 2
        m2 = TurnMetrics(compliance_before=60.0)
        m2.complete(tokens=1000, compliance_after=70.0)
        metrics.append(m2)

        # Non-productive turn
        m3 = TurnMetrics(compliance_before=70.0)
        m3.complete(tokens=1000, compliance_after=65.0)
        metrics.append(m3)

        utility = dropout.calculate_utility("new-agent", metrics=metrics)

        # 2/3 productive
        assert utility == pytest.approx(2/3, abs=0.01)


# ============================================================
# Test 2: Agent Dropout at Threshold (0.15)
# ============================================================

class TestAgentDropoutAtThreshold:
    """Test agent dropout at utility threshold 0.15."""

    def test_low_utility_triggers_dropout(self):
        """Test agent with utility < 0.15 triggers dropout."""
        governor = Governor()
        dropout = AgentDropout(governor)

        # All non-productive turns
        for _ in range(5):
            governor.track_turn("task-001", "agent-beta", 1000, 50.0, 49.0)

        should_drop, reason = dropout.should_dropout("agent-beta")

        assert should_drop == True
        assert reason == DropoutReason.LOW_UTILITY

    def test_high_utility_no_dropout(self):
        """Test agent with high utility doesn't trigger dropout."""
        governor = Governor()
        dropout = AgentDropout(governor)

        # Productive turns
        for _ in range(5):
            governor.track_turn("task-001", "agent-alpha", 1000, 50.0, 70.0)

        should_drop, reason = dropout.should_dropout("agent-alpha")

        assert should_drop == False
        assert reason is None

    def test_threshold_boundary(self):
        """Test behavior at exact threshold."""
        governor = Governor()
        dropout = AgentDropout(governor)

        # Create utility just above threshold
        # 1 productive out of 5 = 0.2 success rate
        # Utility = 0.2 * 0.6 = 0.12 (below threshold)
        governor.track_turn("task-001", "agent-gamma", 1000, 50.0, 60.0)
        for _ in range(4):
            governor.track_turn("task-001", "agent-gamma", 1000, 60.0, 55.0)

        should_drop, _ = dropout.should_dropout("agent-gamma")

        assert should_drop == True  # Should be below threshold

    def test_prune_agent_records_decision(self):
        """Test pruning records the dropout decision."""
        governor = Governor()
        dropout = AgentDropout(governor)

        for _ in range(5):
            governor.track_turn("task-001", "agent-beta", 1000, 50.0, 49.0)

        decision = dropout.prune_agent("agent-beta", DropoutReason.LOW_UTILITY)

        assert decision.agent_id == "agent-beta"
        assert decision.reason == DropoutReason.LOW_UTILITY
        assert decision.utility_at_dropout < 0.15

        history = dropout.get_dropout_history()
        assert len(history) == 1


# ============================================================
# Test 3: Chameleon Spawning
# ============================================================

class TestChameleonSpawning:
    """Test Refactoring Chameleon agent spawning."""

    def test_chameleon_spawn(self):
        """Test Refactoring Chameleon can be spawned."""
        governor = Governor()
        dropout = AgentDropout(governor)

        chameleon = dropout.spawn_chameleon({"task": "refactor authentication"})

        assert chameleon.agent_type == AgentType.REFACTORING_CHAMELEON
        assert chameleon.is_active == True

    def test_chameleon_registered(self):
        """Test spawned chameleon is registered."""
        governor = Governor()
        dropout = AgentDropout(governor)

        chameleon = dropout.spawn_chameleon({"task": "optimize database"})

        assert chameleon.agent_id in dropout._agent_type_registry
        assert dropout._agent_type_registry[chameleon.agent_id] == AgentType.REFACTORING_CHAMELEON

    def test_specialist_spawn(self):
        """Test specialist agent spawn."""
        governor = Governor()
        dropout = AgentDropout(governor)

        specialist = dropout.spawn_specialist(AgentType.SECURITY, {"task": "audit"})

        assert specialist.agent_type == AgentType.SECURITY
        assert specialist.is_active == True


# ============================================================
# Test 4: Swarm Coordination
# ============================================================

class TestSwarmCoordination:
    """Test SwarmCoordinator functionality."""

    def test_add_agent_to_swarm(self):
        """Test adding agent to swarm."""
        governor = Governor()
        coordinator = SwarmCoordinator(governor)

        agent = coordinator.add_agent("agent-001", AgentType.CODER)

        assert agent.agent_id == "agent-001"
        assert agent.agent_type == AgentType.CODER
        assert len(coordinator.get_active_agents()) == 1

    def test_remove_agent_from_swarm(self):
        """Test removing agent from swarm."""
        governor = Governor()
        coordinator = SwarmCoordinator(governor)

        coordinator.add_agent("agent-001")
        removed = coordinator.remove_agent("agent-001")

        assert removed == True
        assert len(coordinator.get_active_agents()) == 0

    def test_mode_switches_to_swarm(self):
        """Test mode switches to SWARM when multiple agents."""
        governor = Governor()
        coordinator = SwarmCoordinator(governor)

        coordinator.add_agent("agent-001")
        assert governor.current_mode == AgentMode.SOLO

        coordinator.add_agent("agent-002")
        assert governor.current_mode == AgentMode.SWARM

    def test_mode_switches_back_to_solo(self):
        """Test mode switches back to SOLO when agents removed."""
        governor = Governor()
        coordinator = SwarmCoordinator(governor)

        coordinator.add_agent("agent-001")
        coordinator.add_agent("agent-002")
        assert governor.current_mode == AgentMode.SWARM

        coordinator.remove_agent("agent-002")
        assert governor.current_mode == AgentMode.SOLO

    def test_scale_up_on_complexity(self):
        """Test scaling up based on complexity score."""
        governor = Governor()
        coordinator = SwarmCoordinator(governor)

        coordinator.add_agent("agent-001")

        should_scale = coordinator.should_scale_up(0.8)  # High complexity

        assert should_scale == True

    def test_no_scale_up_on_low_complexity(self):
        """Test no scaling on low complexity."""
        governor = Governor()
        coordinator = SwarmCoordinator(governor)

        coordinator.add_agent("agent-001")

        should_scale = coordinator.should_scale_up(0.3)  # Low complexity

        assert should_scale == False


# ============================================================
# Test 5: Redundancy Detection
# ============================================================

class TestRedundancyDetection:
    """Test redundancy detection between agents."""

    def test_no_redundancy_with_different_performance(self):
        """Test no redundancy when agents have different performance."""
        governor = Governor()
        dropout = AgentDropout(governor)

        # Agent 1 - high performance
        for _ in range(5):
            governor.track_turn("task-001", "agent-alpha", 1000, 50.0, 70.0)

        # Agent 2 - low performance
        for _ in range(5):
            governor.track_turn("task-001", "agent-beta", 1000, 50.0, 51.0)

        is_redundant, _ = dropout.detect_redundancy(
            "agent-alpha",
            ["agent-alpha", "agent-beta"]
        )

        assert is_redundant == False

    def test_redundancy_detection_keeps_higher_utility(self):
        """Test redundancy keeps higher utility agent."""
        governor = Governor()
        dropout = AgentDropout(governor)

        # Agent alpha - 80% success rate (4 productive, 1 non-productive)
        governor.track_turn("task-001", "agent-alpha", 1000, 50.0, 55.0)  # productive
        governor.track_turn("task-001", "agent-alpha", 1000, 55.0, 60.0)  # productive
        governor.track_turn("task-001", "agent-alpha", 1000, 60.0, 65.0)  # productive
        governor.track_turn("task-001", "agent-alpha", 1000, 65.0, 70.0)  # productive
        governor.track_turn("task-001", "agent-alpha", 1000, 70.0, 69.0)  # NOT productive

        # Agent beta - 100% success rate (5 productive)
        # Success rate diff = 0.20 which is > 0.1 threshold, so need closer rates
        # Make beta have 80% too but slightly higher total utility
        governor.track_turn("task-001", "agent-beta", 1000, 50.0, 60.0)  # productive (+10)
        governor.track_turn("task-001", "agent-beta", 1000, 60.0, 70.0)  # productive (+10)
        governor.track_turn("task-001", "agent-beta", 1000, 70.0, 80.0)  # productive (+10)
        governor.track_turn("task-001", "agent-beta", 1000, 80.0, 90.0)  # productive (+10)
        governor.track_turn("task-001", "agent-beta", 1000, 90.0, 89.0)  # NOT productive

        # Both have 80% success rate (0.8), so score_diff = 0 < 0.1
        # But beta has higher FQL contribution due to bigger gains
        # Actually FQL contribution isn't set, so utilities are equal
        # Need to verify the utility calculation

        # For now, let's verify that the function works when there IS a utility diff
        # We'll create one more productive turn for beta to get 83% vs 80%
        governor.track_turn("task-001", "agent-beta", 1000, 89.0, 95.0)  # productive

        # Now beta has 5/6 = 83% success rate, alpha has 4/5 = 80%
        # score_diff = 0.03 < 0.1, so they're "similar"
        # beta utility > alpha utility

        # Check from lower utility agent's perspective
        is_redundant, redundant_with = dropout.detect_redundancy(
            "agent-alpha",
            ["agent-alpha", "agent-beta"]
        )

        # Should detect alpha as redundant with beta (beta has higher utility)
        assert is_redundant is True
        assert redundant_with == "agent-beta"

    def test_no_redundancy_for_unknown_agent(self):
        """Test no redundancy for agent with no tracked metrics."""
        governor = Governor()
        dropout = AgentDropout(governor)

        is_redundant, _ = dropout.detect_redundancy("unknown", ["unknown", "other"])

        assert is_redundant == False


# ============================================================
# Additional Swarm Tests
# ============================================================

class TestSwarmRouting:
    """Test task routing to best agents."""

    def test_route_to_only_agent(self):
        """Test routing when only one agent."""
        governor = Governor()
        coordinator = SwarmCoordinator(governor)

        coordinator.add_agent("agent-001")

        best = coordinator.route_to_best_agent({"task": "write code"})

        assert best == "agent-001"

    def test_route_to_matching_specialist(self):
        """Test routing prefers matching specialist."""
        governor = Governor()
        coordinator = SwarmCoordinator(governor)

        coordinator.add_agent("coder-001", AgentType.CODER)
        coordinator.add_agent("security-001", AgentType.SECURITY)

        best = coordinator.route_to_best_agent({"task": "security audit"})

        assert best == "security-001"

    def test_route_empty_swarm(self):
        """Test routing returns None for empty swarm."""
        governor = Governor()
        coordinator = SwarmCoordinator(governor)

        best = coordinator.route_to_best_agent({"task": "anything"})

        assert best is None


class TestSwarmHeat:
    """Test Swarm Heat indicator."""

    def test_swarm_heat_empty(self):
        """Test swarm heat is 0 when empty."""
        governor = Governor()
        coordinator = SwarmCoordinator(governor)

        heat = coordinator.get_swarm_heat()

        assert heat == 0.0

    def test_swarm_heat_increases_with_agents(self):
        """Test swarm heat increases with agents."""
        governor = Governor()
        coordinator = SwarmCoordinator(governor)

        coordinator.add_agent("agent-001")
        heat1 = coordinator.get_swarm_heat()

        coordinator.add_agent("agent-002")
        heat2 = coordinator.get_swarm_heat()

        assert heat2 > heat1

    def test_swarm_heat_max_at_capacity(self):
        """Test swarm heat is 1.0 at max capacity."""
        governor = Governor()
        coordinator = SwarmCoordinator(governor)

        # Add max agents
        for i in range(coordinator.MAX_SWARM_SIZE):
            coordinator.add_agent(f"agent-{i}")

        heat = coordinator.get_swarm_heat()

        assert heat == 1.0


class TestSwarmPruning:
    """Test automatic swarm pruning."""

    def test_prune_low_utility_agents(self):
        """Test pruning removes low utility agents."""
        governor = Governor()
        coordinator = SwarmCoordinator(governor)

        coordinator.add_agent("good-agent")
        coordinator.add_agent("bad-agent")

        # Good agent - productive
        for _ in range(5):
            governor.track_turn("task-001", "good-agent", 1000, 50.0, 70.0)

        # Bad agent - non-productive
        for _ in range(5):
            governor.track_turn("task-001", "bad-agent", 1000, 50.0, 49.0)

        decisions = coordinator.prune_low_utility_agents()

        assert len(decisions) == 1
        assert decisions[0].agent_id == "bad-agent"
        assert len(coordinator.get_active_agents()) == 1

    def test_swarm_status(self):
        """Test swarm status includes all info."""
        governor = Governor()
        coordinator = SwarmCoordinator(governor)

        coordinator.add_agent("agent-001", AgentType.CODER)
        coordinator.add_agent("agent-002", AgentType.REVIEWER)

        status = coordinator.get_swarm_status()

        assert status["mode"] == "SWARM"
        assert status["active_agent_count"] == 2
        assert status["swarm_heat"] == 0.4  # 2/5
        assert AgentType.CODER.value in status["agent_types"]
