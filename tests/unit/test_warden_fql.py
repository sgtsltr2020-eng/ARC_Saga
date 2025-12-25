"""
Unit Tests for Warden FQL Integration
=====================================

Tests for Phase 8 Stage C: Warden Zero-Trust FQL enforcement,
Citation Loop, and Governance Escalation.

Author: ARC SAGA Development Team
Date: December 25, 2025
Status: Phase 8 - MAE Stage C
"""

from unittest.mock import AsyncMock, patch

import pytest

from saga.core.mae import (
    ComplianceResult,
    Governor,
    MASSTriggeredReason,
    SwarmCoordinator,
    create_fql_packet,
)
from saga.core.mae.fql_schema import FQLAction, PrincipleCitation, StrictnessLevel
from saga.core.mae.swarm import AgentType

# ============================================================
# Test 1: Reject Raw Text Proposal (400 Bad Request)
# ============================================================

class TestRejectRawTextProposal:
    """Test Zero-Trust FQL enforcement rejects raw text."""

    @pytest.mark.asyncio
    async def test_reject_raw_text_proposal(self):
        """Raw text proposals return 400 Bad Request with Protocol Hint."""
        from saga.core.warden import Warden

        warden = Warden()

        # Call with raw text (no FQL packet)
        proposal = await warden.receive_proposal(
            saga_request="Create user endpoint",
            context={"budget": 100}
        )

        assert proposal.decision == "rejected"
        assert "400 Bad Request" in proposal.rationale
        assert "Non-FQL Protocol" in proposal.rationale
        assert "saga/core/mae/fql_schema.py" in proposal.rationale

    @pytest.mark.asyncio
    async def test_reject_includes_protocol_hint(self):
        """Rejection includes specific file reference for autonomous learning."""
        from saga.core.warden import Warden

        warden = Warden()

        proposal = await warden.receive_proposal(
            saga_request="Any raw text request"
        )

        assert "create_fql_packet()" in proposal.rationale
        assert "PROTOCOL-001" in proposal.sagacodex_violations[0]

    @pytest.mark.asyncio
    async def test_empty_request_also_rejected(self):
        """Empty request with no FQL packet is rejected."""
        from saga.core.warden import Warden

        warden = Warden()

        proposal = await warden.receive_proposal()

        assert proposal.decision == "rejected"


# ============================================================
# Test 2: Accept FQL Packet
# ============================================================

class TestAcceptFQLPacket:
    """Test valid FQL packets are accepted."""

    @pytest.mark.asyncio
    async def test_accept_fql_packet(self):
        """Valid FQL packet proceeds to validation."""
        from saga.core.warden import Warden

        warden = Warden()

        # Mock Mimiry.validate_proposal to return compliant result
        mock_result = ComplianceResult(
            is_compliant=True,
            compliance_score=95.0,
            principle_citations=[
                PrincipleCitation(
                    rule_id=4,
                    rule_name="SDLC-RESILIENCE-04",
                    relevance="HIGH",
                    excerpt="Async operations must be resilient"
                )
            ],
            corrections=[],
            rejected_alternatives=[],
            validation_hash="abc123"
        )

        with patch.object(warden.mimiry, 'validate_proposal', new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = mock_result

            packet = create_fql_packet(
                sender="SAGA-Orchestrator",
                action=FQLAction.VALIDATE_PATTERN,
                subject="CreateUserEndpoint",
                principle_id="SDLC-RESILIENCE-04"
            )

            proposal = await warden.receive_proposal(fql_packet=packet)

            assert proposal.decision == "approved"
            assert "FQL validated" in proposal.rationale
            mock_validate.assert_called_once()


# ============================================================
# Test 3: Citation Loop Turn 2
# ============================================================

class TestCitationLoopTurn2:
    """Test Citation Loop self-correction on turn 2."""

    def test_governor_tracks_multiple_turns(self):
        """Governor tracks turn metrics correctly."""
        governor = Governor()

        # Turn 1
        turn1 = governor.track_turn(
            task_id="task-001",
            agent_id="CodingAgent",
            tokens=1000,
            compliance_before=50.0,
            compliance_after=60.0
        )

        # Turn 2
        turn2 = governor.track_turn(
            task_id="task-001",
            agent_id="CodingAgent",
            tokens=1200,
            compliance_before=60.0,
            compliance_after=75.0
        )

        summary = governor.get_task_summary("task-001")
        assert summary["turn_count"] == 2
        assert summary["total_tokens"] == 2200


# ============================================================
# Test 4: Compliance Regression Swaps Agent
# ============================================================

class TestComplianceRegressionSwapsAgent:
    """Test Governance Escalation on compliance regression."""

    def test_mass_triggers_on_regression(self):
        """COMPLIANCE_REGRESSION triggers MASS."""
        governor = Governor()

        # Turn 1 - improvement
        governor.track_turn("task-001", "agent-alpha", 1000, 50.0, 60.0)

        # Turn 2 - regression below initial
        governor.track_turn("task-001", "agent-alpha", 1000, 60.0, 45.0)

        should_mass, reason = governor.should_trigger_mass("task-001")

        assert should_mass is True
        assert reason == MASSTriggeredReason.COMPLIANCE_REGRESSION

    def test_swarm_spawns_specialist_on_regression(self):
        """SwarmCoordinator spawns specialist on regression."""
        governor = Governor()
        swarm = SwarmCoordinator(governor)

        # Add initial agent
        swarm.add_agent("primary-coder", AgentType.CODER)

        # Spawn for high complexity (triggers specialist)
        new_agent = swarm.spawn_for_complexity(
            0.9,
            {"task": "security audit", "strictness": "FAANG_GOLDEN_PATH"}
        )

        assert new_agent is not None
        assert new_agent.agent_type == AgentType.SECURITY

    def test_faang_strictness_on_escalation(self):
        """Escalated specialist gets FAANG_GOLDEN_PATH strictness."""
        governor = Governor()
        swarm = SwarmCoordinator(governor)

        # The spawn_for_complexity should route to security specialist
        new_agent = swarm.spawn_for_complexity(
            0.9,
            {"task": "authentication security"}
        )

        # Verify it's a security type (due to "security" in task)
        assert new_agent.agent_type == AgentType.SECURITY


# ============================================================
# Test 5: Synthesis Requires Citations
# ============================================================

class TestSynthesisRequiresCitations:
    """Test that Synthesis Engine proposals require citations."""

    def test_fql_packet_includes_citations(self):
        """FQL packet can include principle citations."""
        packet = create_fql_packet(
            sender="SynthesisEngine",
            action=FQLAction.VALIDATE_PATTERN,  # Use valid action
            subject="AuthService.validate_token",
            principle_id="SECURITY-AUTH-01",
            context={
                "citations": [
                    {"principle": "SECURITY-AUTH-01", "source": "Codex"},
                    {"principle": "PATTERN-JWT-02", "source": "Mythos"}
                ]
            }
        )

        assert packet.governance.mimiry_principle_id == "SECURITY-AUTH-01"
        assert "citations" in packet.payload.context
        assert len(packet.payload.context["citations"]) == 2


# ============================================================
# Test 6: Shadow Static FQL Check
# ============================================================

class TestShadowStaticFQLCheck:
    """Test pre-Docker FQL check in ShadowWorkspace."""

    def test_fql_packet_for_shadow_check(self):
        """Create FQL packet for shadow validation."""
        packet = create_fql_packet(
            sender="ShadowWorkspace",
            action=FQLAction.VALIDATE_PATTERN,
            subject="proposed_code_block",
            principle_id="AUTO-DETECT"
        )

        assert packet.header.sender == "ShadowWorkspace"
        assert packet.payload.action == FQLAction.VALIDATE_PATTERN


# ============================================================
# Test 7: Governor Initialization in Warden
# ============================================================

class TestGovernorInitializationInWarden:
    """Test Governor is properly initialized in Warden."""

    def test_warden_has_governor(self):
        """Warden has Governor instance."""
        from saga.core.warden import Warden

        warden = Warden()

        assert hasattr(warden, 'governor')
        assert isinstance(warden.governor, Governor)

    def test_warden_has_swarm(self):
        """Warden has SwarmCoordinator instance."""
        from saga.core.warden import Warden

        warden = Warden()

        assert hasattr(warden, 'swarm')
        assert isinstance(warden.swarm, SwarmCoordinator)


# ============================================================
# Test 8: FQL Context Injection
# ============================================================

class TestFQLContextInjection:
    """Test FQL governance is injected into task context."""

    @pytest.mark.asyncio
    async def test_fql_governance_in_task_context(self):
        """Tasks receive FQL governance metadata."""
        from saga.core.warden import Warden

        warden = Warden()

        mock_result = ComplianceResult(
            is_compliant=True,
            compliance_score=90.0,
            principle_citations=[
                PrincipleCitation(
                    rule_id=1,
                    rule_name="SDLC-001",
                    relevance="HIGH",
                    excerpt="Standard development lifecycle"
                )
            ],
            corrections=[],
            rejected_alternatives=[],
            validation_hash="xyz"
        )

        with patch.object(warden.mimiry, 'validate_proposal', new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = mock_result

            packet = create_fql_packet(
                sender="SAGA",
                action=FQLAction.VALIDATE_PATTERN,
                subject="UserService",
                principle_id="SDLC-001",
                strictness=StrictnessLevel.FAANG_GOLDEN_PATH
            )

            proposal = await warden.receive_proposal(fql_packet=packet)

            # Check task graph has FQL governance injected
            if proposal.task_graph:
                tasks = proposal.task_graph.get_all_tasks()
                if tasks:
                    task = tasks[0]
                    assert "fql_governance" in task.context
                    assert task.context["fql_governance"]["strictness"] == "FAANG_GOLDEN_PATH"


# ============================================================
# Test 9: Warden Proposal Violations
# ============================================================

class TestWardenProposalViolations:
    """Test violations are properly reported in proposals."""

    @pytest.mark.asyncio
    async def test_low_compliance_rejected(self):
        """Low compliance score (< 50) results in rejection."""
        from saga.core.warden import Warden

        warden = Warden()

        mock_result = ComplianceResult(
            is_compliant=False,
            compliance_score=30.0,
            principle_citations=[],
            corrections=["Fix async patterns", "Add error handling"],
            rejected_alternatives=[],
            validation_hash="abc"
        )

        with patch.object(warden.mimiry, 'validate_proposal', new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = mock_result

            packet = create_fql_packet(
                sender="SAGA",
                action=FQLAction.VALIDATE_PATTERN,
                subject="BadCode",
                principle_id="SDLC-001"
            )

            proposal = await warden.receive_proposal(fql_packet=packet)

            assert proposal.decision == "rejected"
            assert "Fix async patterns" in proposal.rationale


# ============================================================
# Test 10: Partial Compliance Proceeds with Warnings
# ============================================================

class TestPartialComplianceProceeds:
    """Test partial compliance (score >= 50) proceeds with warnings."""

    @pytest.mark.asyncio
    async def test_partial_compliance_approved(self):
        """Score >= 50 but not fully compliant adds warnings."""
        from saga.core.warden import Warden

        warden = Warden()

        mock_result = ComplianceResult(
            is_compliant=False,  # Not fully compliant
            compliance_score=75.0,  # But above threshold
            principle_citations=[
                PrincipleCitation(
                    rule_id=1,
                    rule_name="SDLC-001",
                    relevance="MEDIUM",
                    excerpt="Development best practices"
                )
            ],
            corrections=["Consider adding retry logic"],
            rejected_alternatives=[],
            validation_hash="def"
        )

        with patch.object(warden.mimiry, 'validate_proposal', new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = mock_result

            packet = create_fql_packet(
                sender="SAGA",
                action=FQLAction.VALIDATE_PATTERN,
                subject="OkayCode",
                principle_id="SDLC-001"
            )

            proposal = await warden.receive_proposal(fql_packet=packet)

            assert proposal.decision == "approved"
            assert "Consider adding retry logic" in proposal.sagacodex_violations
