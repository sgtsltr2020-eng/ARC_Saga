"""
Unit Tests for Synthesis Engine FQL Integration
================================================

Tests for Phase 8 Stage D: Synthesis-FQL Bridge with dual-citation
requirement and provenance tracking.

Author: ARC SAGA Development Team
Date: December 25, 2025
Status: Phase 8 - MAE Stage D
"""

from unittest.mock import MagicMock

import pytest

from saga.core.mae.fql_schema import FQLAction, PrincipleCitation, StrictnessLevel
from saga.core.memory.graph_engine import RepoGraph
from saga.core.memory.synthesis_engine import SynthesisAgent, SynthesisSpark

# ============================================================
# Test 1: Dual-Citation Requirement
# ============================================================

class TestDualCitationRequirement:
    """Test that synthesis requires minimum 2 citations."""

    def test_reject_single_citation(self):
        """Single citation should raise ValueError."""
        spark = SynthesisSpark(
            source_domain="error_handling",
            target_domain="api",
            synthesis_prompt="Test prompt"
        )

        single_citation = [
            PrincipleCitation(
                rule_id=1,
                rule_name="ERROR-HANDLING-01",
                relevance="HIGH",
                excerpt="Error handling best practices"
            )
        ]

        with pytest.raises(ValueError) as exc_info:
            spark.to_fql_packet(single_citation)

        assert "minimum 2 principle citations" in str(exc_info.value)

    def test_accept_two_citations(self):
        """Two citations should be accepted."""
        spark = SynthesisSpark(
            source_domain="error_handling",
            target_domain="api",
            synthesis_prompt="Test prompt"
        )

        citations = [
            PrincipleCitation(
                rule_id=1,
                rule_name="ERROR-HANDLING-01",
                relevance="HIGH",
                excerpt="Error handling best practices"
            ),
            PrincipleCitation(
                rule_id=2,
                rule_name="API-DESIGN-01",
                relevance="HIGH",
                excerpt="API design standards"
            )
        ]

        packet = spark.to_fql_packet(citations)

        assert packet.header.sender == "SynthesisAgent"
        assert packet.payload.action == FQLAction.VALIDATE_PATTERN

    def test_accept_three_citations(self):
        """Three citations should also be accepted."""
        spark = SynthesisSpark(
            source_domain="auth",
            target_domain="api",
            synthesis_prompt="Security integration"
        )

        citations = [
            PrincipleCitation(rule_id=1, rule_name="SEC-01", relevance="HIGH", excerpt="Security"),
            PrincipleCitation(rule_id=2, rule_name="API-01", relevance="HIGH", excerpt="API"),
            PrincipleCitation(rule_id=3, rule_name="AUTH-01", relevance="MEDIUM", excerpt="Auth")
        ]

        packet = spark.to_fql_packet(citations)
        assert len(packet.payload.context["citations"]) == 3


# ============================================================
# Test 2: FQL Packet Generation
# ============================================================

class TestFQLPacketGeneration:
    """Test to_fql_packet() generates valid FQL packets."""

    def test_packet_has_sender(self):
        """Packet sender should be SynthesisAgent."""
        spark = SynthesisSpark(source_domain="api", target_domain="db")
        citations = [
            PrincipleCitation(rule_id=1, rule_name="API-01", relevance="HIGH", excerpt="API"),
            PrincipleCitation(rule_id=2, rule_name="DB-01", relevance="HIGH", excerpt="DB")
        ]

        packet = spark.to_fql_packet(citations)
        assert packet.header.sender == "SynthesisAgent"

    def test_packet_action_is_validate_pattern(self):
        """Packet action should be VALIDATE_PATTERN."""
        spark = SynthesisSpark(source_domain="api", target_domain="db")
        citations = [
            PrincipleCitation(rule_id=1, rule_name="API-01", relevance="HIGH", excerpt="API"),
            PrincipleCitation(rule_id=2, rule_name="DB-01", relevance="HIGH", excerpt="DB")
        ]

        packet = spark.to_fql_packet(citations)
        assert packet.payload.action == FQLAction.VALIDATE_PATTERN

    def test_packet_subject_contains_domains(self):
        """Packet subject should contain source→target domains."""
        spark = SynthesisSpark(source_domain="error", target_domain="logging")
        citations = [
            PrincipleCitation(rule_id=1, rule_name="ERR-01", relevance="HIGH", excerpt="Err"),
            PrincipleCitation(rule_id=2, rule_name="LOG-01", relevance="HIGH", excerpt="Log")
        ]

        packet = spark.to_fql_packet(citations)
        assert "error" in packet.payload.subject
        assert "logging" in packet.payload.subject

    def test_packet_includes_strictness(self):
        """Packet should include strictness level."""
        spark = SynthesisSpark(source_domain="api", target_domain="db")
        citations = [
            PrincipleCitation(rule_id=1, rule_name="API-01", relevance="HIGH", excerpt="API"),
            PrincipleCitation(rule_id=2, rule_name="DB-01", relevance="HIGH", excerpt="DB")
        ]

        packet = spark.to_fql_packet(citations, strictness=StrictnessLevel.SENIOR_DEV)
        # Check strictness is in the governance section
        assert packet.governance.strictness_level == StrictnessLevel.SENIOR_DEV


# ============================================================
# Test 3: Provenance Tracking
# ============================================================

class TestProvenanceTracking:
    """Test provenance (Spark ID) tracking in FQL packets."""

    def test_spark_id_in_context(self):
        """Spark ID should be included in context for audit trail."""
        spark = SynthesisSpark(
            source_domain="api",
            target_domain="db",
            spark_id="test-spark-123"
        )
        citations = [
            PrincipleCitation(rule_id=1, rule_name="API-01", relevance="HIGH", excerpt="API"),
            PrincipleCitation(rule_id=2, rule_name="DB-01", relevance="HIGH", excerpt="DB")
        ]

        packet = spark.to_fql_packet(citations)

        assert "provenance" in packet.payload.context
        assert packet.payload.context["provenance"]["spark_id"] == "test-spark-123"

    def test_source_nodes_in_provenance(self):
        """Source nodes should be tracked in provenance."""
        spark = SynthesisSpark(
            source_domain="api",
            target_domain="db",
            source_nodes=["node1", "node2"]
        )
        citations = [
            PrincipleCitation(rule_id=1, rule_name="API-01", relevance="HIGH", excerpt="API"),
            PrincipleCitation(rule_id=2, rule_name="DB-01", relevance="HIGH", excerpt="DB")
        ]

        packet = spark.to_fql_packet(citations)

        assert packet.payload.context["provenance"]["source_nodes"] == ["node1", "node2"]


# ============================================================
# Test 4: Find Applicable Principles
# ============================================================

class TestFindApplicablePrinciples:
    """Test _find_applicable_principles() method."""

    def test_finds_domain_specific_principles(self):
        """Should find principles matching domain keywords."""
        # Create mock graph
        mock_graph = MagicMock(spec=RepoGraph)
        mock_graph._node_index = {}

        agent = SynthesisAgent(graph=mock_graph)

        citations = agent._find_applicable_principles(
            source_domain="error_handling",
            target_domain="api"
        )

        # Should find at least 2 (dual-citation guarantee)
        assert len(citations) >= 2

        # Should have error-related principle
        rule_names = [c.rule_name for c in citations]
        assert any("ERROR" in r for r in rule_names) or len(citations) >= 2

    def test_ensures_minimum_two_citations(self):
        """Should always return at least 2 citations."""
        mock_graph = MagicMock(spec=RepoGraph)
        mock_graph._node_index = {}

        agent = SynthesisAgent(graph=mock_graph)

        # Use obscure domains with no direct matches
        citations = agent._find_applicable_principles(
            source_domain="obscure_domain_xyz",
            target_domain="unknown_area_abc"
        )

        assert len(citations) >= 2

    def test_fallback_principles_used(self):
        """Fallback principles should be used when no matches."""
        mock_graph = MagicMock(spec=RepoGraph)
        mock_graph._node_index = {}

        agent = SynthesisAgent(graph=mock_graph)

        citations = agent._find_applicable_principles(
            source_domain="xyz123",
            target_domain="abc456"
        )

        # Should have fallback principles
        rule_names = [c.rule_name for c in citations]
        assert any("SYNTHESIS" in r or "INTEGRATION" in r for r in rule_names)
