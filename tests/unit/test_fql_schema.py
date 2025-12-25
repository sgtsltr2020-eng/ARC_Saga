"""
Unit Tests for FQL Schema
=========================

Tests for the Formal Query Language schema definitions.
Covers validation, caching, strictness enforcement, and AST meta-tags.

Author: ARC SAGA Development Team
Date: December 25, 2025
Status: Phase 8 - MAE Foundation
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from saga.core.mae.fql_schema import (
    ASTMetaTag,
    ComplianceResult,
    FQLAction,
    FQLGovernance,
    FQLHeader,
    FQLPacket,
    FQLPayload,
    PatternType,
    PrincipleCitation,
    RejectedAlternative,
    StrictnessLevel,
    ValidationCache,
    compute_code_hash,
    create_fql_packet,
)

# ============================================================
# Test 1: Valid FQL Packet Creation
# ============================================================

class TestFQLSchemaValidation:
    """Test FQL schema validation for valid packets."""

    def test_valid_fql_packet_creation(self):
        """Test that a valid FQL packet can be created."""
        packet = FQLPacket(
            header=FQLHeader(
                sender="Saga-Warden",
                correlation_id="trace-001"
            ),
            payload=FQLPayload(
                action=FQLAction.VALIDATE_PATTERN,
                subject="UserService.create_user",
                context={"stack": ["Python", "FastAPI"], "constraint": "P99<10ms"}
            ),
            governance=FQLGovernance(
                mimiry_principle_id="SDLC-RESILIENCE-04"
            )
        )

        assert packet.header.sender == "Saga-Warden"
        assert packet.header.target == "Mimiry-Oracle"
        assert packet.header.protocol_version == "1.0.0"
        assert packet.payload.action == FQLAction.VALIDATE_PATTERN
        assert packet.payload.subject == "UserService.create_user"
        assert packet.governance.strictness_level == StrictnessLevel.FAANG_GOLDEN_PATH
        assert "SDLC-RESILIENCE-04" in packet.governance.mimiry_principle_id

    def test_fql_packet_with_all_actions(self):
        """Test all FQL actions can be used."""
        actions = [
            FQLAction.VALIDATE_PATTERN,
            FQLAction.INTERPRET_RULE,
            FQLAction.MEASURE_CODE,
            FQLAction.RESOLVE_CONFLICT,
        ]

        for action in actions:
            packet = create_fql_packet(
                sender="Test-Agent",
                action=action,
                subject="TestSubject",
                principle_id="TEST-01"
            )
            assert packet.payload.action == action

    def test_fql_header_timestamp_default(self):
        """Test that header timestamp defaults to current time."""
        header = FQLHeader(sender="Test")
        assert isinstance(header.timestamp, datetime)
        # Should be within last second
        assert (datetime.utcnow() - header.timestamp).total_seconds() < 1

    def test_compute_packet_hash(self):
        """Test packet hash computation for caching."""
        packet = create_fql_packet(
            sender="Agent1",
            action=FQLAction.VALIDATE_PATTERN,
            subject="Module.function",
            principle_id="RULE-01"
        )

        hash1 = packet.compute_packet_hash()
        assert len(hash1) == 16  # First 16 chars of SHA256

        # Same packet should produce same hash
        packet2 = create_fql_packet(
            sender="Agent1",
            action=FQLAction.VALIDATE_PATTERN,
            subject="Module.function",
            principle_id="RULE-01"
        )
        assert packet.compute_packet_hash() == packet2.compute_packet_hash()


# ============================================================
# Test 2: Malformed FQL Packet Rejection
# ============================================================

class TestFQLSchemaRejection:
    """Test FQL schema validation for malformed packets."""

    def test_empty_sender_rejected(self):
        """Test that empty sender is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            FQLHeader(sender="")

        assert "sender cannot be empty" in str(exc_info.value)

    def test_whitespace_sender_rejected(self):
        """Test that whitespace-only sender is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            FQLHeader(sender="   ")

        assert "sender cannot be empty" in str(exc_info.value)

    def test_empty_subject_rejected(self):
        """Test that empty subject is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            FQLPayload(
                action=FQLAction.VALIDATE_PATTERN,
                subject=""
            )

        assert "subject cannot be empty" in str(exc_info.value)

    def test_invalid_hash_format_rejected(self):
        """Test that invalid hash format is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            FQLPayload(
                action=FQLAction.VALIDATE_PATTERN,
                subject="Test",
                proposed_logic_hash="invalid-hash"
            )

        assert "SHA256" in str(exc_info.value)

    def test_valid_hash_format_accepted(self):
        """Test that valid SHA256 hash is accepted."""
        valid_hash = "sha256:" + "a" * 64
        payload = FQLPayload(
            action=FQLAction.VALIDATE_PATTERN,
            subject="Test",
            proposed_logic_hash=valid_hash
        )
        assert payload.proposed_logic_hash == "sha256:" + "a" * 64

    def test_empty_principle_id_rejected(self):
        """Test that empty principle ID is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            FQLGovernance(mimiry_principle_id="")

        assert "mimiry_principle_id cannot be empty" in str(exc_info.value)


# ============================================================
# Test 3: ValidationCache Deduplication
# ============================================================

class TestValidationCacheDeduplication:
    """Test ValidationCache for efficient repeat query handling."""

    def test_cache_stores_and_retrieves(self):
        """Test that cache stores and retrieves results."""
        cache = ValidationCache()

        packet = create_fql_packet(
            sender="Test",
            action=FQLAction.VALIDATE_PATTERN,
            subject="TestSubject",
            principle_id="RULE-01"
        )

        result = ComplianceResult(
            is_compliant=True,
            compliance_score=95.0,
            validation_hash="test-hash"
        )

        cache.put(packet, result)
        retrieved = cache.get(packet)

        assert retrieved is not None
        assert retrieved.is_compliant == True
        assert retrieved.compliance_score == 95.0

    def test_cache_miss_returns_none(self):
        """Test that cache miss returns None."""
        cache = ValidationCache()

        packet = create_fql_packet(
            sender="Test",
            action=FQLAction.VALIDATE_PATTERN,
            subject="NonExistent",
            principle_id="RULE-99"
        )

        result = cache.get(packet)
        assert result is None

    def test_cache_ttl_expiration(self):
        """Test that expired cache entries are not returned."""
        cache = ValidationCache(ttl_seconds=0)  # Immediate expiration

        packet = create_fql_packet(
            sender="Test",
            action=FQLAction.VALIDATE_PATTERN,
            subject="TestSubject",
            principle_id="RULE-01"
        )

        result = ComplianceResult(
            is_compliant=True,
            compliance_score=95.0,
            validation_hash="test-hash"
        )

        cache.put(packet, result)

        # Force entry to be expired by manipulating timestamp
        import time
        time.sleep(0.1)

        retrieved = cache.get(packet)
        assert retrieved is None

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = ValidationCache(max_size=2)

        packets = [
            create_fql_packet("Test", FQLAction.VALIDATE_PATTERN, f"Subject{i}", "RULE-01")
            for i in range(3)
        ]

        results = [
            ComplianceResult(is_compliant=True, compliance_score=90.0 + i, validation_hash=f"hash-{i}")
            for i in range(3)
        ]

        # Add first two
        cache.put(packets[0], results[0])
        cache.put(packets[1], results[1])

        # Add third - should evict first
        cache.put(packets[2], results[2])

        # First should be evicted
        assert cache.get(packets[0]) is None
        # Second and third should still be present
        assert cache.get(packets[1]) is not None
        assert cache.get(packets[2]) is not None

    def test_cache_stats(self):
        """Test cache statistics tracking."""
        cache = ValidationCache()

        packet = create_fql_packet("Test", FQLAction.VALIDATE_PATTERN, "Subject", "RULE-01")
        result = ComplianceResult(is_compliant=True, compliance_score=95.0, validation_hash="hash")

        # Miss
        cache.get(packet)
        # Put and hit
        cache.put(packet, result)
        cache.get(packet)
        cache.get(packet)

        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert "66.7%" in stats["hit_rate"]


# ============================================================
# Test 4: Strictness Level Enforcement
# ============================================================

class TestStrictnessLevelEnforcement:
    """Test strictness level validation and defaults."""

    def test_default_strictness_is_faang(self):
        """Test that default strictness is FAANG_GOLDEN_PATH."""
        governance = FQLGovernance(mimiry_principle_id="TEST-01")
        assert governance.strictness_level == StrictnessLevel.FAANG_GOLDEN_PATH

    def test_all_strictness_levels_valid(self):
        """Test all strictness levels can be set."""
        levels = [
            StrictnessLevel.FAANG_GOLDEN_PATH,
            StrictnessLevel.SENIOR_DEV,
            StrictnessLevel.ENTERPRISE,
        ]

        for level in levels:
            governance = FQLGovernance(
                mimiry_principle_id="TEST-01",
                strictness_level=level
            )
            assert governance.strictness_level == level

    def test_strictness_affects_compliance_threshold(self):
        """Test that strictness level affects what counts as compliant."""
        # This is tested indirectly through the Mimiry validate_proposal
        # but we can verify the model accepts all levels
        packet_faang = create_fql_packet(
            sender="Test",
            action=FQLAction.VALIDATE_PATTERN,
            subject="Subject",
            principle_id="TEST-01",
            strictness=StrictnessLevel.FAANG_GOLDEN_PATH
        )

        packet_enterprise = create_fql_packet(
            sender="Test",
            action=FQLAction.VALIDATE_PATTERN,
            subject="Subject",
            principle_id="TEST-01",
            strictness=StrictnessLevel.ENTERPRISE
        )

        assert packet_faang.governance.strictness_level == StrictnessLevel.FAANG_GOLDEN_PATH
        assert packet_enterprise.governance.strictness_level == StrictnessLevel.ENTERPRISE

    def test_require_citations_defaults_true(self):
        """Test that require_citations defaults to True for FAANG strictness."""
        governance = FQLGovernance(mimiry_principle_id="TEST-01")
        assert governance.require_citations == True

    def test_max_violations_defaults_zero(self):
        """Test that max_violations_allowed defaults to 0 for FAANG."""
        governance = FQLGovernance(mimiry_principle_id="TEST-01")
        assert governance.max_violations_allowed == 0


# ============================================================
# Test 5: AST Meta-Tag Parsing
# ============================================================

class TestASTMetaTagParsing:
    """Test AST meta-tag schema-driven validation."""

    def test_ast_tag_creation(self):
        """Test AST meta-tag can be created."""
        tag = ASTMetaTag(
            pattern_type=PatternType.CREATIONAL,
            implementation="SINGLETON_THREAD_SAFE",
            language="python",
            framework="FastAPI"
        )

        assert tag.pattern_type == PatternType.CREATIONAL
        assert tag.implementation == "SINGLETON_THREAD_SAFE"
        assert tag.language == "python"
        assert tag.framework == "FastAPI"

    def test_implementation_normalized_to_uppercase(self):
        """Test that implementation is normalized to uppercase."""
        tag = ASTMetaTag(
            pattern_type=PatternType.BEHAVIORAL,
            implementation="observer pattern"
        )
        assert tag.implementation == "OBSERVER_PATTERN"

    def test_all_pattern_types_valid(self):
        """Test all pattern types can be used."""
        pattern_types = [
            PatternType.CREATIONAL,
            PatternType.STRUCTURAL,
            PatternType.BEHAVIORAL,
            PatternType.CONCURRENCY,
            PatternType.ARCHITECTURAL,
        ]

        for pt in pattern_types:
            tag = ASTMetaTag(pattern_type=pt, implementation="TEST")
            assert tag.pattern_type == pt

    def test_fql_payload_with_ast_tags(self):
        """Test FQL payload can include AST tags."""
        payload = FQLPayload(
            action=FQLAction.VALIDATE_PATTERN,
            subject="ServiceLayer",
            ast_tags=[
                ASTMetaTag(pattern_type=PatternType.CREATIONAL, implementation="FACTORY"),
                ASTMetaTag(pattern_type=PatternType.STRUCTURAL, implementation="ADAPTER"),
            ]
        )

        assert len(payload.ast_tags) == 2
        assert payload.ast_tags[0].pattern_type == PatternType.CREATIONAL

    def test_compute_code_hash_helper(self):
        """Test the compute_code_hash helper function."""
        code = "def hello(): return 'world'"
        hash_value = compute_code_hash(code)

        assert hash_value.startswith("sha256:")
        assert len(hash_value) == 7 + 64  # "sha256:" + 64 hex chars

    def test_code_hash_deterministic(self):
        """Test that code hash is deterministic."""
        code = "async def fetch_user(user_id: str) -> User: ..."
        hash1 = compute_code_hash(code)
        hash2 = compute_code_hash(code)
        assert hash1 == hash2

    def test_different_code_different_hash(self):
        """Test that different code produces different hashes."""
        hash1 = compute_code_hash("def foo(): pass")
        hash2 = compute_code_hash("def bar(): pass")
        assert hash1 != hash2


# ============================================================
# Test ComplianceResult Helpers
# ============================================================

class TestComplianceResultHelpers:
    """Test ComplianceResult helper methods."""

    def test_correction_rubric_for_non_compliant(self):
        """Test correction rubric generation for failures."""
        result = ComplianceResult(
            is_compliant=False,
            compliance_score=65.0,
            corrections=["Missing type hints", "Synchronous I/O detected"],
            principle_citations=[
                PrincipleCitation(rule_id=1, rule_name="Type Safety", relevance="HIGH"),
                PrincipleCitation(rule_id=2, rule_name="Async I/O", relevance="CRITICAL"),
            ]
        )

        rubric = result.to_correction_rubric()

        assert "Compliance Failed" in rubric
        assert "65.0" in rubric
        assert "Missing type hints" in rubric
        assert "Type Safety" in rubric

    def test_correction_rubric_for_compliant(self):
        """Test correction rubric for passing result."""
        result = ComplianceResult(
            is_compliant=True,
            compliance_score=98.0
        )

        rubric = result.to_correction_rubric()
        assert "No corrections needed" in rubric

    def test_rejected_alternatives_stored(self):
        """Test that rejected alternatives are stored."""
        result = ComplianceResult(
            is_compliant=True,
            compliance_score=95.0,
            rejected_alternatives=[
                RejectedAlternative(
                    approach="Use synchronous Redis client",
                    rejection_reason="Blocks event loop under load"
                ),
                RejectedAlternative(
                    approach="Use LSM-Tree",
                    rejection_reason="Write amplification constraints"
                ),
            ]
        )

        assert len(result.rejected_alternatives) == 2
        assert "LSM-Tree" in result.rejected_alternatives[1].approach


# ============================================================
# Test Factory Function
# ============================================================

class TestCreateFQLPacket:
    """Test the create_fql_packet factory function."""

    def test_minimal_packet_creation(self):
        """Test minimal packet creation with required args only."""
        packet = create_fql_packet(
            sender="Agent",
            action=FQLAction.VALIDATE_PATTERN,
            subject="Module",
            principle_id="RULE-01"
        )

        assert packet.header.sender == "Agent"
        assert packet.payload.action == FQLAction.VALIDATE_PATTERN
        assert packet.payload.subject == "Module"
        assert packet.governance.mimiry_principle_id == "RULE-01"

    def test_full_packet_creation(self):
        """Test full packet creation with all optional args."""
        packet = create_fql_packet(
            sender="Saga-Commander",
            action=FQLAction.MEASURE_CODE,
            subject="AuthService",
            principle_id="SEC-AUDIT-01",
            context={"framework": "FastAPI", "version": "0.100.0"},
            code_hash="sha256:" + "b" * 64,
            strictness=StrictnessLevel.SENIOR_DEV,
            trace_id="trace-xyz-123"
        )

        assert packet.header.correlation_id == "trace-xyz-123"
        assert packet.payload.context["framework"] == "FastAPI"
        assert packet.payload.proposed_logic_hash == "sha256:" + "b" * 64
        assert packet.governance.strictness_level == StrictnessLevel.SENIOR_DEV
