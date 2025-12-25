"""
FQL Schema - Formal Query Language Definitions
===============================================

FQL is a Contractual JSON Schema that prevents "natural language drift"
between Saga and the Immutable Oracle (Mimiry). This is NOT a chat protocol.

The FQL Gateway acts as a "Software Engineering Firewall" - all Mimiry
interactions must go through validated FQL packets.

Design Principles:
- Schema-Driven Development: AST meta-tags instead of natural language
- Stateless Validation: Failed proposals return corrections without pollution
- Zero-Trust: Raw text queries are rejected at the interface level

Author: ARC SAGA Development Team
Date: December 25, 2025
Status: Phase 8 - MAE Foundation
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ============================================================
# ENUMERATIONS
# ============================================================

class FQLAction(str, Enum):
    """Valid FQL actions for Mimiry consultation."""
    VALIDATE_PATTERN = "VALIDATE_PATTERN"
    INTERPRET_RULE = "INTERPRET_RULE"
    MEASURE_CODE = "MEASURE_CODE"
    RESOLVE_CONFLICT = "RESOLVE_CONFLICT"


class StrictnessLevel(str, Enum):
    """Validation strictness levels."""
    FAANG_GOLDEN_PATH = "FAANG_GOLDEN_PATH"  # Google/Meta level
    SENIOR_DEV = "SENIOR_DEV"  # Experienced dev standards
    ENTERPRISE = "ENTERPRISE"  # Corporate acceptable


class PatternType(str, Enum):
    """AST pattern types for schema-driven validation."""
    CREATIONAL = "CREATIONAL"
    STRUCTURAL = "STRUCTURAL"
    BEHAVIORAL = "BEHAVIORAL"
    CONCURRENCY = "CONCURRENCY"
    ARCHITECTURAL = "ARCHITECTURAL"


# ============================================================
# FQL PACKET COMPONENTS
# ============================================================

class FQLHeader(BaseModel):
    """
    FQL packet header with sender/target identification.

    Attributes:
        sender: Identifier of the requesting agent (e.g., "Saga-Swarm-Commander")
        target: Always "Mimiry-Oracle" for now
        protocol_version: FQL protocol version for compatibility checking
        timestamp: When the packet was created
        correlation_id: Optional trace ID for request correlation
    """
    sender: str
    target: Literal["Mimiry-Oracle"] = "Mimiry-Oracle"
    protocol_version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: str | None = None

    @field_validator("sender")
    @classmethod
    def validate_sender(cls, v: str) -> str:
        """Ensure sender is not empty."""
        if not v or not v.strip():
            raise ValueError("sender cannot be empty")
        return v.strip()


class ASTMetaTag(BaseModel):
    """
    AST-based pattern descriptor for schema-driven validation.

    Instead of describing code in natural language, Saga sends
    structured AST meta-tags that Mimiry can verify at FAANG level.

    Example:
        {"pattern_type": "CREATIONAL", "implementation": "SINGLETON_THREAD_SAFE"}
    """
    pattern_type: PatternType
    implementation: str
    language: str = "python"
    framework: str | None = None

    @field_validator("implementation")
    @classmethod
    def validate_implementation(cls, v: str) -> str:
        """Ensure implementation is uppercase and valid."""
        return v.upper().replace(" ", "_")


class FQLPayload(BaseModel):
    """
    The action payload with AST meta-tags.

    Uses structured data instead of natural language to prevent
    hallucination and ensure precise Codex matching.

    Attributes:
        action: The FQL action to perform
        subject: Component or module being validated
        context: Structured context (stack, constraints, etc.)
        proposed_logic_hash: SHA256 of proposed code for change tracking
        ast_tags: Optional AST meta-tags for pattern validation
    """
    action: FQLAction
    subject: str
    context: dict[str, Any] = Field(default_factory=dict)
    proposed_logic_hash: str | None = None
    ast_tags: list[ASTMetaTag] = Field(default_factory=list)

    @field_validator("subject")
    @classmethod
    def validate_subject(cls, v: str) -> str:
        """Ensure subject is not empty."""
        if not v or not v.strip():
            raise ValueError("subject cannot be empty")
        return v.strip()

    @field_validator("proposed_logic_hash")
    @classmethod
    def validate_hash(cls, v: str | None) -> str | None:
        """Validate SHA256 hash format if provided."""
        if v is None:
            return None
        # Remove "sha256:" prefix if present
        clean = v.replace("sha256:", "")
        if len(clean) != 64 or not all(c in "0123456789abcdef" for c in clean.lower()):
            raise ValueError("proposed_logic_hash must be valid SHA256 (64 hex chars)")
        return f"sha256:{clean.lower()}"


class FQLGovernance(BaseModel):
    """
    Governance parameters for validation strictness.

    Controls how strictly Mimiry validates against the Codex.

    Attributes:
        mimiry_principle_id: Reference to specific Codex principle
        strictness_level: How strictly to enforce the principle
        require_citations: Whether to require explicit rule citations
        max_violations_allowed: Soft limit on acceptable violations
    """
    mimiry_principle_id: str
    strictness_level: StrictnessLevel = StrictnessLevel.FAANG_GOLDEN_PATH
    require_citations: bool = True
    max_violations_allowed: int = 0

    @field_validator("mimiry_principle_id")
    @classmethod
    def validate_principle_id(cls, v: str) -> str:
        """Ensure principle ID follows naming convention."""
        if not v or not v.strip():
            raise ValueError("mimiry_principle_id cannot be empty")
        # Format: CATEGORY-SUBCATEGORY-NN (e.g., SDLC-RESILIENCE-04)
        parts = v.split("-")
        if len(parts) < 2:
            logger.warning(f"Principle ID '{v}' doesn't follow expected format")
        return v.strip().upper()


class FQLPacket(BaseModel):
    """
    Complete FQL request packet.

    This is the ONLY valid way to query Mimiry. Raw text queries
    are rejected at the FQL Gateway level.

    Example:
        ```python
        packet = FQLPacket(
            header=FQLHeader(sender="Saga-Warden"),
            payload=FQLPayload(
                action=FQLAction.VALIDATE_PATTERN,
                subject="UserService.create_user",
                context={"stack": ["Python", "FastAPI"], "constraint": "P99<10ms"}
            ),
            governance=FQLGovernance(
                mimiry_principle_id="SDLC-RESILIENCE-04"
            )
        )
        ```
    """
    header: FQLHeader
    payload: FQLPayload
    governance: FQLGovernance

    def compute_packet_hash(self) -> str:
        """Compute unique hash for cache key."""
        content = f"{self.payload.action}:{self.payload.subject}:{self.governance.mimiry_principle_id}"
        if self.payload.proposed_logic_hash:
            content += f":{self.payload.proposed_logic_hash}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


# ============================================================
# FQL RESPONSE
# ============================================================

class PrincipleCitation(BaseModel):
    """
    Citation of a SagaCodex principle in the response.

    Provides traceability from validation result to specific rules.
    """
    rule_id: int
    rule_name: str
    relevance: Literal["CRITICAL", "HIGH", "MEDIUM"] = "HIGH"
    excerpt: str = ""


class RejectedAlternative(BaseModel):
    """
    An alternative approach that was considered but rejected.

    Feeds the Tauri UI "Why" panel with trade-off analysis.
    """
    approach: str
    rejection_reason: str
    cited_constraint: str | None = None


class ComplianceResult(BaseModel):
    """
    FQL validation response from Mimiry.

    Contains the compliance assessment, citations, and any
    corrections needed. This is a STATELESS response - failures
    are not recorded in Mimiry's long-term memory.

    Attributes:
        is_compliant: Whether the proposal meets the principle requirements
        compliance_score: 0.0 to 100.0 score
        principle_citations: Cited SagaCodex rules
        corrections: List of required corrections if non-compliant
        rejected_alternatives: Trade-off analysis for "Why" panel
        validation_hash: Unique hash for caching/deduplication
        validated_at: Timestamp of validation
    """
    is_compliant: bool
    compliance_score: float = Field(ge=0.0, le=100.0)
    principle_citations: list[PrincipleCitation] = Field(default_factory=list)
    corrections: list[str] = Field(default_factory=list)
    rejected_alternatives: list[RejectedAlternative] = Field(default_factory=list)
    validation_hash: str = ""
    validated_at: datetime = Field(default_factory=datetime.utcnow)

    # Trade-off analysis summary for UI
    validated_approach: str = ""
    rationale: str = ""

    def to_correction_rubric(self) -> str:
        """
        Generate human-readable correction rubric.

        Used when validation fails to guide the agent.
        """
        if self.is_compliant:
            return "No corrections needed."

        lines = [
            f"## Compliance Failed (Score: {self.compliance_score:.1f}/100)",
            "",
            "### Required Corrections:",
        ]
        for i, correction in enumerate(self.corrections, 1):
            lines.append(f"{i}. {correction}")

        if self.principle_citations:
            lines.extend(["", "### Cited Principles:"])
            for citation in self.principle_citations:
                lines.append(f"- Rule {citation.rule_id}: {citation.rule_name}")

        return "\n".join(lines)


# ============================================================
# VALIDATION CACHE
# ============================================================

class CacheEntry(BaseModel):
    """Single entry in the validation cache."""
    packet_hash: str
    result: ComplianceResult
    cached_at: datetime = Field(default_factory=datetime.utcnow)
    hit_count: int = 1


class ValidationCache:
    """
    Cache for FQL validation results.

    Implements efficient repeat query handling without taxing
    the core Oracle. Uses LRU eviction when cache is full.

    Advisory Tip: This prevents redundant validation of the same
    proposal during retry loops.
    """

    DEFAULT_MAX_SIZE = 100
    DEFAULT_TTL_SECONDS = 300  # 5 minutes

    def __init__(
        self,
        max_size: int = DEFAULT_MAX_SIZE,
        ttl_seconds: int = DEFAULT_TTL_SECONDS
    ) -> None:
        """Initialize the validation cache."""
        self._cache: dict[str, CacheEntry] = {}
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._hits = 0
        self._misses = 0

    def get(self, packet: FQLPacket) -> ComplianceResult | None:
        """
        Retrieve cached result for a packet.

        Returns:
            Cached ComplianceResult if found and not expired, else None.
        """
        packet_hash = packet.compute_packet_hash()
        entry = self._cache.get(packet_hash)

        if entry is None:
            self._misses += 1
            return None

        # Check TTL
        age = (datetime.utcnow() - entry.cached_at).total_seconds()
        if age > self._ttl_seconds:
            del self._cache[packet_hash]
            self._misses += 1
            logger.debug(f"Cache entry expired: {packet_hash}")
            return None

        # Cache hit
        entry.hit_count += 1
        self._hits += 1
        logger.debug(f"Cache hit: {packet_hash} (hits: {entry.hit_count})")
        return entry.result

    def put(self, packet: FQLPacket, result: ComplianceResult) -> None:
        """
        Store validation result in cache.

        Uses LRU eviction if cache is full.
        """
        packet_hash = packet.compute_packet_hash()

        # Evict oldest entries if at capacity
        while len(self._cache) >= self._max_size:
            oldest_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].cached_at
            )
            del self._cache[oldest_key]
            logger.debug(f"Cache evicted: {oldest_key}")

        self._cache[packet_hash] = CacheEntry(
            packet_hash=packet_hash,
            result=result
        )
        logger.debug(f"Cache stored: {packet_hash}")

    def invalidate(self, packet: FQLPacket) -> bool:
        """Invalidate a specific cache entry."""
        packet_hash = packet.compute_packet_hash()
        if packet_hash in self._cache:
            del self._cache[packet_hash]
            return True
        return False

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        logger.info("Validation cache cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0.0
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "ttl_seconds": self._ttl_seconds,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1f}%",
        }


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_fql_packet(
    sender: str,
    action: FQLAction,
    subject: str,
    principle_id: str,
    context: dict[str, Any] | None = None,
    code_hash: str | None = None,
    strictness: StrictnessLevel = StrictnessLevel.FAANG_GOLDEN_PATH,
    trace_id: str | None = None,
) -> FQLPacket:
    """
    Factory function to create FQL packets.

    Convenience method for common packet creation patterns.

    Example:
        ```python
        packet = create_fql_packet(
            sender="Saga-Warden",
            action=FQLAction.VALIDATE_PATTERN,
            subject="UserController",
            principle_id="SDLC-ASYNC-01",
            context={"framework": "FastAPI"},
        )
        ```
    """
    return FQLPacket(
        header=FQLHeader(
            sender=sender,
            correlation_id=trace_id,
        ),
        payload=FQLPayload(
            action=action,
            subject=subject,
            context=context or {},
            proposed_logic_hash=code_hash,
        ),
        governance=FQLGovernance(
            mimiry_principle_id=principle_id,
            strictness_level=strictness,
        ),
    )


def compute_code_hash(code: str) -> str:
    """
    Compute SHA256 hash of code for FQL packet.

    Returns:
        Hash in format "sha256:abc123..."
    """
    return f"sha256:{hashlib.sha256(code.encode()).hexdigest()}"
