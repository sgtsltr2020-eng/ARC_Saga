"""
Unit Tests for Governor
=======================

Tests for the Turn-Control Governor including turn tracking,
MASS trigger conditions, burn rate calculation, and information-gain decay.

Author: ARC SAGA Development Team
Date: December 25, 2025
Status: Phase 8 - MAE Foundation
"""

import time

from saga.core.mae.governor import (
    AgentMode,
    Governor,
    MASSTriggeredReason,
    TurnMetrics,
)

# ============================================================
# Test 1: Turn Tracking and Limits
# ============================================================

class TestTurnTrackingAndLimits:
    """Test turn tracking functionality and turn limits."""

    def test_turn_metrics_creation(self):
        """Test TurnMetrics can be created."""
        turn = TurnMetrics(
            task_id="task-001",
            agent_id="agent-alpha",
            compliance_before=50.0
        )

        assert turn.task_id == "task-001"
        assert turn.agent_id == "agent-alpha"
        assert turn.compliance_before == 50.0
        assert turn.end_time is None

    def test_turn_completion(self):
        """Test turn can be completed with metrics."""
        turn = TurnMetrics(
            task_id="task-001",
            agent_id="agent-alpha",
            compliance_before=50.0
        )

        turn.complete(tokens=1500, compliance_after=75.0)

        assert turn.tokens_used == 1500
        assert turn.compliance_after == 75.0
        assert turn.compliance_delta == 25.0
        assert turn.end_time is not None
        assert turn.is_productive == True

    def test_non_productive_turn(self):
        """Test turn with no improvement is not productive."""
        turn = TurnMetrics(compliance_before=75.0)
        turn.complete(tokens=1000, compliance_after=70.0)

        assert turn.is_productive == False
        assert turn.compliance_delta == -5.0

    def test_governor_tracks_turns(self):
        """Test Governor tracks turns correctly."""
        governor = Governor()

        turn = governor.track_turn(
            task_id="task-001",
            agent_id="agent-alpha",
            tokens=1500,
            compliance_before=50.0,
            compliance_after=75.0
        )

        assert turn.tokens_used == 1500
        assert turn.compliance_delta == 25.0

        # Check task record exists
        summary = governor.get_task_summary("task-001")
        assert summary["turn_count"] == 1
        assert summary["total_tokens"] == 1500

    def test_turn_count_increments(self):
        """Test turn count increments correctly."""
        governor = Governor()

        for i in range(3):
            governor.track_turn(
                task_id="task-001",
                agent_id="agent-alpha",
                tokens=1000,
                compliance_before=50.0 + i*10,
                compliance_after=60.0 + i*10
            )

        summary = governor.get_task_summary("task-001")
        assert summary["turn_count"] == 3


# ============================================================
# Test 2: MASS Trigger Conditions
# ============================================================

class TestMASSTriggerConditions:
    """Test Multi-Agent System Search trigger conditions."""

    def test_mass_triggers_at_turn_limit(self):
        """Test MASS triggers when turn limit (5) is exceeded."""
        governor = Governor()

        # Execute 5 turns
        for i in range(5):
            governor.track_turn(
                task_id="task-001",
                agent_id="agent-alpha",
                tokens=1000,
                compliance_before=50.0,
                compliance_after=51.0  # Small improvement each turn
            )

        should_trigger, reason = governor.should_trigger_mass("task-001")

        assert should_trigger == True
        assert reason == MASSTriggeredReason.TURN_LIMIT_EXCEEDED

    def test_mass_not_triggered_under_limit(self):
        """Test MASS not triggered when under turn limit."""
        governor = Governor()

        for i in range(3):
            governor.track_turn(
                task_id="task-001",
                agent_id="agent-alpha",
                tokens=1000,
                compliance_before=50.0,
                compliance_after=60.0
            )

        should_trigger, reason = governor.should_trigger_mass("task-001")

        assert should_trigger == False
        assert reason is None

    def test_mass_triggers_on_stagnant_gain(self):
        """Test MASS triggers after 3 consecutive non-improving turns."""
        governor = Governor()

        # 3 turns with minimal improvement
        for i in range(3):
            governor.track_turn(
                task_id="task-001",
                agent_id="agent-alpha",
                tokens=1000,
                compliance_before=50.0,
                compliance_after=50.3  # Below MIN_COMPLIANCE_GAIN (0.5)
            )

        should_trigger, reason = governor.should_trigger_mass("task-001")

        assert should_trigger == True
        assert reason == MASSTriggeredReason.STAGNANT_GAIN

    def test_mass_triggers_on_regression(self):
        """Test MASS triggers on compliance regression."""
        governor = Governor()

        # First turn - some improvement
        governor.track_turn(
            task_id="task-001",
            agent_id="agent-alpha",
            tokens=1000,
            compliance_before=70.0,
            compliance_after=75.0
        )

        # Second turn - regression below initial
        governor.track_turn(
            task_id="task-001",
            agent_id="agent-alpha",
            tokens=1000,
            compliance_before=75.0,
            compliance_after=65.0  # Below initial 70.0
        )

        should_trigger, reason = governor.should_trigger_mass("task-001")

        assert should_trigger == True
        assert reason == MASSTriggeredReason.COMPLIANCE_REGRESSION

    def test_mass_only_triggers_once(self):
        """Test MASS only triggers once per task."""
        governor = Governor()

        for i in range(6):
            governor.track_turn(
                task_id="task-001",
                agent_id="agent-alpha",
                tokens=1000,
                compliance_before=50.0,
                compliance_after=51.0
            )

        # First check should trigger
        should_trigger1, reason1 = governor.should_trigger_mass("task-001")
        assert should_trigger1 == True

        # Second check should not trigger again
        should_trigger2, reason2 = governor.should_trigger_mass("task-001")
        assert should_trigger2 == False


# ============================================================
# Test 3: Burn Rate Calculation
# ============================================================

class TestBurnRateCalculation:
    """Test burn rate (tokens/time) calculation."""

    def test_burn_rate_calculation(self):
        """Test burn rate is calculated correctly."""
        governor = Governor()

        # First turn
        turn1 = governor.start_turn("task-001", "agent-alpha", 50.0)
        time.sleep(0.1)  # Small delay
        governor.complete_turn(turn1, tokens=1000, compliance_after=60.0)

        # Second turn
        turn2 = governor.start_turn("task-001", "agent-alpha", 60.0)
        time.sleep(0.1)
        governor.complete_turn(turn2, tokens=1500, compliance_after=70.0)

        burn_rate = governor.get_burn_rate("task-001")

        assert burn_rate["total_tokens"] == 2500
        assert burn_rate["tokens_per_turn"] == 1250.0
        assert burn_rate["total_seconds"] > 0.1
        assert burn_rate["tokens_per_second"] > 0

    def test_burn_rate_empty_task(self):
        """Test burn rate for non-existent task."""
        governor = Governor()

        burn_rate = governor.get_burn_rate("non-existent")

        assert burn_rate["total_tokens"] == 0
        assert burn_rate["tokens_per_second"] == 0.0

    def test_tokens_per_turn_accuracy(self):
        """Test tokens per turn is accurate."""
        governor = Governor()

        tokens_list = [1000, 2000, 1500]
        for tokens in tokens_list:
            governor.track_turn(
                task_id="task-001",
                agent_id="agent-alpha",
                tokens=tokens,
                compliance_before=50.0,
                compliance_after=60.0
            )

        burn_rate = governor.get_burn_rate("task-001")

        expected_avg = sum(tokens_list) / len(tokens_list)
        assert burn_rate["tokens_per_turn"] == expected_avg


# ============================================================
# Test 4: Information-Gain Decay
# ============================================================

class TestInformationGainDecay:
    """Test Information-Gain Decay detection."""

    def test_productive_turns_not_stagnant(self):
        """Test productive turns don't trigger stagnant detection."""
        governor = Governor()

        for i in range(3):
            governor.track_turn(
                task_id="task-001",
                agent_id="agent-alpha",
                tokens=1000,
                compliance_before=50.0 + i*10,
                compliance_after=60.0 + i*10  # 10 point improvement each
            )

        should_trigger, reason = governor.should_trigger_mass("task-001")

        assert should_trigger == False

    def test_mixed_turns_not_stagnant(self):
        """Test mixed productive/non-productive turns don't trigger."""
        governor = Governor()

        # Productive
        governor.track_turn("task-001", "agent-alpha", 1000, 50.0, 65.0)
        # Non-productive
        governor.track_turn("task-001", "agent-alpha", 1000, 65.0, 65.0)
        # Productive
        governor.track_turn("task-001", "agent-alpha", 1000, 65.0, 80.0)

        should_trigger, reason = governor.should_trigger_mass("task-001")

        assert should_trigger == False

    def test_three_stagnant_triggers_mass(self):
        """Test exactly 3 stagnant turns triggers MASS."""
        governor = Governor()

        # 2 stagnant turns - should not trigger yet
        for _ in range(2):
            governor.track_turn("task-001", "agent-alpha", 1000, 50.0, 50.1)

        should_trigger, _ = governor.should_trigger_mass("task-001")
        assert should_trigger == False

        # 3rd stagnant turn - should trigger
        governor.track_turn("task-001", "agent-alpha", 1000, 50.0, 50.2)

        should_trigger, reason = governor.should_trigger_mass("task-001")
        assert should_trigger == True
        assert reason == MASSTriggeredReason.STAGNANT_GAIN


# ============================================================
# Test 5: Escalation After 5 Turns
# ============================================================

class TestEscalationAfter5Turns:
    """Test escalation behavior after max turns."""

    def test_escalation_recommendation_logged(self):
        """Test that MASS trigger includes escalation recommendation."""
        governor = Governor()

        for i in range(5):
            governor.track_turn(
                task_id="task-001",
                agent_id="agent-alpha",
                tokens=1000,
                compliance_before=50.0,
                compliance_after=51.0
            )

        should_trigger, reason = governor.should_trigger_mass("task-001")

        assert should_trigger == True
        # Reason should indicate turn limit
        assert reason == MASSTriggeredReason.TURN_LIMIT_EXCEEDED

    def test_task_summary_shows_mass_status(self):
        """Test task summary includes MASS trigger info."""
        governor = Governor()

        for i in range(5):
            governor.track_turn(
                task_id="task-001",
                agent_id="agent-alpha",
                tokens=1000,
                compliance_before=50.0,
                compliance_after=51.0
            )

        governor.should_trigger_mass("task-001")  # Trigger MASS

        summary = governor.get_task_summary("task-001")

        assert summary["mass_triggered"] == True
        assert summary["mass_trigger_reason"] == "TURN_LIMIT_EXCEEDED"

    def test_task_reset_allows_new_turns(self):
        """Test resetting task allows fresh tracking."""
        governor = Governor()

        for i in range(5):
            governor.track_turn("task-001", "agent-alpha", 1000, 50.0, 51.0)

        governor.should_trigger_mass("task-001")
        governor.reset_task("task-001")

        # New turn should work
        governor.track_turn("task-001", "agent-alpha", 1000, 50.0, 60.0)

        summary = governor.get_task_summary("task-001")
        assert summary["turn_count"] == 1
        assert summary["mass_triggered"] == False


# ============================================================
# Test Agent Utility and Mode
# ============================================================

class TestAgentUtilityAndMode:
    """Test agent utility scoring and mode switching."""

    def test_agent_utility_calculation(self):
        """Test agent utility is calculated from turns."""
        governor = Governor()

        # Productive turns
        governor.track_turn("task-001", "agent-alpha", 1000, 50.0, 60.0)
        governor.track_turn("task-001", "agent-alpha", 1000, 60.0, 70.0)
        # Non-productive turn
        governor.track_turn("task-001", "agent-alpha", 1000, 70.0, 65.0)

        utility = governor.evaluate_efficiency("agent-alpha")

        # 2/3 productive = ~0.67 success rate
        # Utility = success_rate * 0.6 + fql_contribution * 0.4
        # With fql_contribution = 0, utility = 0.67 * 0.6 = ~0.4
        assert utility > 0.3
        assert utility < 0.6

    def test_low_utility_agents_detected(self):
        """Test low utility agents are detected."""
        governor = Governor()

        # Non-productive turns for agent
        for _ in range(5):
            governor.track_turn("task-001", "agent-beta", 1000, 50.0, 49.0)

        low_utility = governor.get_low_utility_agents()

        assert len(low_utility) == 1
        assert low_utility[0].agent_id == "agent-beta"

    def test_mode_starts_as_solo(self):
        """Test Governor starts in SOLO mode."""
        governor = Governor()
        assert governor.current_mode == AgentMode.SOLO

    def test_mode_can_be_changed(self):
        """Test mode can be changed to SWARM."""
        governor = Governor()
        governor.set_mode(AgentMode.SWARM)
        assert governor.current_mode == AgentMode.SWARM

    def test_governor_stats(self):
        """Test Governor stats are accurate."""
        governor = Governor()

        governor.track_turn("task-001", "agent-alpha", 1000, 50.0, 60.0)
        governor.track_turn("task-002", "agent-beta", 1000, 50.0, 55.0)

        stats = governor.get_stats()

        assert stats["total_tasks_tracked"] == 2
        assert stats["total_agents_tracked"] == 2
        assert stats["current_mode"] == "SOLO"
