"""
Mimiry Guardrail - Read-Only Principle Enforcement at Retrieval Time
======================================================================

Validates retrieval candidates against Mimiry principles before context assembly.
Enforces architectural integrity by down-ranking or excluding anti-patterns.

Author: ARC SAGA Development Team (AntiGravity)
Date: December 26, 2025
Status: Phase 2 - Memory Trust Enhancement
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════


@dataclass
class Principle:
    """A Mimiry principle applicable to code."""

    rule_number: int
    name: str
    description: str
    tags: list[str]  # Fast keyword matching
    embedding: np.ndarray | None = None  # Semantic matching
    severity: Literal["critical", "major", "minor"] = "major"


@dataclass
class Violation:
    """A detected principle violation."""

    principle: Principle
    severity: Literal["critical", "major", "minor"]
    reason: str
    node_id: str


@dataclass
class GuardrailResult:
    """Result of guardrail validation."""

    node_id: str
    passed: bool
    violations: list[Violation]
    penalty_multiplier: float  # 0.0-1.0, multiply with existing score
    action: Literal["allow", "downrank", "exclude"]
    principle_checks_count: int
    cache_hit: bool
    warden_veto_triggered: bool = False  # For critical violations
    veto_reason: str | None = None  # Explanation for veto
    check_duration_ms: float = 0.0  # Performance tracking


@dataclass
class ViolationPenalties:
    """Configurable penalty strengths."""

    # Structured as dict for future per-principle tuning
    _penalties: dict[int, float] = field(default_factory=dict)  # rule_number -> penalty

    # Tier defaults
    critical_default: float = 0.0  # Hard exclude
    major_default: float = 0.4  # Heavy down-rank
    minor_default: float = 0.8  # Light down-rank

    def get_penalty(self, rule_number: int, severity: str) -> float:
        """Get penalty for rule, falling back to tier default."""
        if rule_number in self._penalties:
            return self._penalties[rule_number]

        if severity == "critical":
            return self.critical_default
        elif severity == "major":
            return self.major_default
        else:
            return self.minor_default

    def set_penalty(self, rule_number: int, penalty: float) -> None:
        """Override penalty for specific rule."""
        self._penalties[rule_number] = penalty


@dataclass
class PrincipleMatch:
    """Cached principle match result."""

    node_id: str
    content_hash: str
    matched_principles: list[Principle]
    timestamp: float
    match_method: Literal["keyword", "embedding", "none"]


# ═══════════════════════════════════════════════════════════════
# MIMIRY GUARDRAIL
# ═══════════════════════════════════════════════════════════════


class MimiryGuardrail:
    """
    Read-only principle enforcer for retrieval candidates.

    Queries Mimiry for applicable principles and applies penalties
    to candidates that violate them, without modifying Mimiry itself.

    Features:
    - Two-stage matching: keyword → embedding fallback
    - Tiered violations: critical/major/minor
    - Per-node caching with TTL
    - Warden veto integration for critical violations
    - Persistent allow-list for false positives
    """

    def __init__(
        self,
        mimiry: Any,  # Mimiry instance
        warden: Any = None,  # For critical violation vetoes
        principle_cache_ttl: int = 3600,  # 1 hour
        enable_cache: bool = True,
        allowlist_path: Path | None = None,
        embedding_threshold: float = 0.7,
    ):
        """
        Initialize the Mimiry guardrail.

        Args:
            mimiry: Read-only Mimiry instance for principle queries
            warden: Optional Warden instance for veto escalation
            principle_cache_ttl: Cache TTL in seconds
            enable_cache: Enable caching
            allowlist_path: Path to persistent allow-list JSON
            embedding_threshold: Similarity threshold for embedding matches
        """
        self.mimiry = mimiry  # Read-only reference
        self.warden = warden  # For veto escalation

        # Caching
        self._principle_cache: dict[str, PrincipleMatch] = {}
        self._cache_timestamps: dict[str, float] = {}
        self.cache_ttl = principle_cache_ttl
        self.enable_cache = enable_cache

        # Persistent allow-list (survives sessions)
        self.allowlist_path = allowlist_path or (
            Path.home() / ".saga" / "memory" / "allowlist.json"
        )
        self._allowlist: set[str] = self._load_allowlist()

        # Configurable penalties
        self.penalties = ViolationPenalties()

        # Principle matching config
        self.embedding_threshold = embedding_threshold

        # Performance tracking
        self.stats = {
            "validations_total": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "keyword_matches": 0,
            "embedding_matches": 0,
            "violations_detected": 0,
            "vetoes_triggered": 0,
        }

        # Load principles from Mimiry
        self._principles: list[Principle] = self._load_principles()

        logger.info(
            f"MimiryGuardrail initialized: {len(self._principles)} principles loaded, "
            f"cache_ttl={principle_cache_ttl}s, allowlist={len(self._allowlist)} nodes"
        )

    # ─── Principle Loading ──────────────────────────────────────

    def _load_principles(self) -> list[Principle]:
        """Load principles from Mimiry (read-only)."""
        principles = []

        # Extract from Mimiry's current codex
        if hasattr(self.mimiry, "current_codex") and hasattr(
            self.mimiry.current_codex, "standards"
        ):
            for std in self.mimiry.current_codex.standards:
                # Extract tags from name and description
                tags = self._extract_tags(std.name, std.description)

                # Determine severity based on rule number
                severity = self._determine_severity(std.rule_number)

                principles.append(
                    Principle(
                        rule_number=std.rule_number,
                        name=std.name,
                        description=std.description,
                        tags=tags,
                        embedding=None,  # Lazy-loaded if needed
                        severity=severity,
                    )
                )

        logger.debug(f"Loaded {len(principles)} principles from Mimiry")
        return principles

    def _extract_tags(self, name: str, description: str) -> list[str]:
        """Extract keyword tags from principle name and description."""
        # Common patterns to extract
        keywords = set()

        text = f"{name} {description}".lower()

        # Add whole name words
        keywords.update(name.lower().split())

        # Detect common patterns
        if "type" in text and "hint" in text:
            keywords.update(["type", "hints", "annotation", "typing"])
        if "async" in text:
            keywords.update(["async", "await", "asyncio"])
        if "secret" in text or "password" in text or "key" in text:
            keywords.update(["secret", "password", "api_key", "token", "credential"])
        if "except" in text:
            keywords.update(["except", "exception", "error"])
        if "log" in text:
            keywords.update(["log", "logger", "logging", "print"])
        if "modular" in text or "separation" in text:
            keywords.update(["modular", "class", "function", "separation"])
        if "sql" in text or "inject" in text:
            keywords.update(["sql", "query", "injection", "execute"])

        return sorted(keywords)

    def _determine_severity(self, rule_number: int) -> Literal["critical", "major", "minor"]:
        """Determine severity tier for a rule number."""
        # Critical: Security/safety violations
        if rule_number in [15, 99]:  # Secrets, SQL injection
            return "critical"

        # Major: Architectural violations
        if rule_number in [1, 2, 3, 7]:  # Type hints, async, except, modularity
            return "major"

        # Minor: Style/convention
        return "minor"

    # ─── Allow-List Management ──────────────────────────────────

    def _load_allowlist(self) -> set[str]:
        """Load persistent allow-list from disk."""
        if not self.allowlist_path.exists():
            return set()

        try:
            data = json.loads(self.allowlist_path.read_text(encoding="utf-8"))
            allowlist = set(data.get("allowed_nodes", []))
            logger.debug(f"Loaded {len(allowlist)} nodes from allow-list")
            return allowlist
        except Exception as e:
            logger.warning(f"Failed to load allow-list: {e}")
            return set()

    def _save_allowlist(self) -> None:
        """Save allow-list to disk."""
        try:
            self.allowlist_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "allowed_nodes": sorted(self._allowlist),
                "updated_at": time.time(),
            }
            self.allowlist_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            logger.debug(f"Saved allow-list: {len(self._allowlist)} nodes")
        except Exception as e:
            logger.error(f"Failed to save allow-list: {e}")

    def add_to_allowlist(self, node_id: str, reason: str = "") -> None:
        """
        Add node to persistent allow-list.

        Args:
            node_id: Node to allow
            reason: Why this is a false positive
        """
        self._allowlist.add(node_id)
        self._save_allowlist()
        logger.info(f"Added to allow-list: {node_id} (reason: {reason})")

    def remove_from_allowlist(self, node_id: str) -> None:
        """Remove node from allow-list."""
        self._allowlist.discard(node_id)
        self._save_allowlist()
        logger.info(f"Removed from allow-list: {node_id}")

    # ─── Main Validation ────────────────────────────────────────

    def validate_candidate(
        self, node_id: str, node_content: str, query_context: dict[str, Any]
    ) -> GuardrailResult:
        """
        Validate a single retrieval candidate.

        Args:
            node_id: Unique node identifier
            node_content: Code content of the node
            query_context: Query and retrieval context

        Returns:
            GuardrailResult with penalty and action
        """
        start_time = time.time()
        self.stats["validations_total"] += 1

        # Check allow-list first (short-circuit)
        if node_id in self._allowlist:
            return GuardrailResult(
                node_id=node_id,
                passed=True,
                violations=[],
                penalty_multiplier=1.0,
                action="allow",
                principle_checks_count=0,
                cache_hit=True,
                check_duration_ms=(time.time() - start_time) * 1000,
            )

        # Get applicable principles
        principles, cache_hit = self._get_applicable_principles(node_id, node_content)

        if cache_hit:
            self.stats["cache_hits"] += 1
        else:
            self.stats["cache_misses"] += 1

        # Detect violations
        violations = self._detect_violations(node_id, node_content, principles)

        if violations:
            self.stats["violations_detected"] += len(violations)

        # Calculate penalty and action
        result = self._calculate_penalty(node_id, violations, query_context)
        result.principle_checks_count = len(principles)
        result.cache_hit = cache_hit
        result.check_duration_ms = (time.time() - start_time) * 1000

        return result

    def validate_batch(
        self, candidates: list[dict[str, Any]], query_context: dict[str, Any]
    ) -> list[GuardrailResult]:
        """
        Validate multiple candidates efficiently.

        Args:
            candidates: List of dicts with 'node_id', 'content', 'score'
            query_context: Query and retrieval context

        Returns:
            List of GuardrailResult in same order as input
        """
        results = []

        for candidate in candidates:
            result = self.validate_candidate(
                node_id=candidate["node_id"],
                node_content=candidate.get("content", ""),
                query_context=query_context,
            )
            results.append(result)

        return results

    # ─── Principle Matching ─────────────────────────────────────

    def _get_applicable_principles(
        self, node_id: str, node_content: str
    ) -> tuple[list[Principle], bool]:
        """
        Get principles applicable to this node.

        Returns:
            (matched_principles, cache_hit)
        """
        # Check cache
        content_hash = self._compute_content_hash(node_content)
        cache_key = f"{node_id}:{content_hash}"

        if self.enable_cache and cache_key in self._principle_cache:
            cached = self._principle_cache[cache_key]
            age = time.time() - cached.timestamp

            if age < self.cache_ttl:
                return cached.matched_principles, True

        # Cache miss - perform matching
        matched = self._match_principles(node_content)

        # Cache result
        if self.enable_cache:
            self._principle_cache[cache_key] = PrincipleMatch(
                node_id=node_id,
                content_hash=content_hash,
                matched_principles=matched,
                timestamp=time.time(),
                match_method="keyword" if matched else "none",
            )
            self._cache_timestamps[cache_key] = time.time()

        return matched, False

    def _match_principles(self, node_content: str) -> list[Principle]:
        """
        Match principles using keyword-first, embedding-fallback strategy.

        Returns:
            List of matched principles
        """
        # Stage 1: Fast keyword matching
        keyword_matches = self._keyword_match(node_content)

        if keyword_matches:
            self.stats["keyword_matches"] += len(keyword_matches)
            return keyword_matches

        # Stage 2: Embedding fallback (if no keyword matches)
        # For now, skip embedding fallback to keep it fast
        # This can be added later if needed

        return []

    def _keyword_match(self, node_content: str) -> list[Principle]:
        """Fast keyword-based principle matching."""
        matched = []
        content_lower = node_content.lower()

        for principle in self._principles:
            # Check if any tags appear in content
            if any(tag.lower() in content_lower for tag in principle.tags):
                matched.append(principle)

        return matched

    # ─── Violation Detection ────────────────────────────────────

    def _detect_violations(
        self, node_id: str, node_content: str, principles: list[Principle]
    ) -> list[Violation]:
        """
        Detect principle violations in node content.

        Args:
            node_id: Node identifier
            node_content: Code content
            principles: Applicable principles

        Returns:
            List of detected violations
        """
        violations = []

        for principle in principles:
            # Check specific principle constraints
            violation = self._check_principle(node_id, node_content, principle)
            if violation:
                violations.append(violation)

        return violations

    def _check_principle(
        self, node_id: str, node_content: str, principle: Principle
    ) -> Violation | None:
        """Check if node violates a specific principle."""

        # Rule 15: No hardcoded secrets
        if principle.rule_number == 15:
            if self._contains_hardcoded_secret(node_content):
                return Violation(
                    principle=principle,
                    severity="critical",
                    reason="Hardcoded secret/credential detected",
                    node_id=node_id,
                )

        # Rule 1: Type hints
        if principle.rule_number == 1:
            if self._lacks_type_hints(node_content):
                return Violation(
                    principle=principle,
                    severity="major",
                    reason="Missing type annotations",
                    node_id=node_id,
                )

        # Rule 3: No bare except
        if principle.rule_number == 3:
            if "except:" in node_content and "except " not in node_content:
                return Violation(
                    principle=principle,
                    severity="major",
                    reason="Bare except clause detected",
                    node_id=node_id,
                )

        # Rule 4: Structured logging
        if principle.rule_number == 4:
            if "print(" in node_content:
                return Violation(
                    principle=principle,
                    severity="minor",
                    reason="Print statement instead of logging",
                    node_id=node_id,
                )

        return None

    def _contains_hardcoded_secret(self, content: str) -> bool:
        """Detect hardcoded secrets (basic heuristic)."""
        secret_patterns = [
            "api_key =",
            "password =",
            "secret =",
            "token =",
            "bearer ",
            "authorization:",
        ]

        content_lower = content.lower()
        for pattern in secret_patterns:
            if pattern in content_lower:
                # Check if it's not a placeholder
                if not any(
                    placeholder in content_lower
                    for placeholder in ["env", "config", "settings", "none", '""', "''"]
                ):
                    return True

        return False

    def _lacks_type_hints(self, content: str) -> bool:
        """Check if code lacks type hints (simple heuristic)."""
        # Look for function definitions without type hints
        if "def " in content:
            # If we see "def foo(" but no "->" or ":" after params
            lines = content.split("\n")
            for line in lines:
                if line.strip().startswith("def "):
                    if ")" in line and "->" not in line:
                        # Function without return type hint
                        return True
        return False

    # ─── Penalty Calculation ────────────────────────────────────

    def _calculate_penalty(
        self, node_id: str, violations: list[Violation], query_context: dict[str, Any]
    ) -> GuardrailResult:
        """
        Calculate penalty based on violations.

        Returns:
            GuardrailResult with action and penalty
        """
        if not violations:
            return GuardrailResult(
                node_id=node_id,
                passed=True,
                violations=[],
                penalty_multiplier=1.0,
                action="allow",
                principle_checks_count=0,
                cache_hit=False,
            )

        # Check for critical violations
        critical_violations = [v for v in violations if v.severity == "critical"]

        if critical_violations:
            # Hard exclude + Warden veto
            self.stats["vetoes_triggered"] += 1

            veto_reason = f"Critical Mimiry violation (Rule #{critical_violations[0].principle.rule_number}): {critical_violations[0].reason}"

            # Trigger veto if Warden is available
            if self.warden:
                self._trigger_warden_veto(critical_violations[0], query_context)

            return GuardrailResult(
                node_id=node_id,
                passed=False,
                violations=violations,
                penalty_multiplier=0.0,
                action="exclude",
                principle_checks_count=len(violations),
                cache_hit=False,
                warden_veto_triggered=True,
                veto_reason=veto_reason,
            )

        # Calculate weighted penalty for non-critical violations
        total_penalty = 1.0

        for violation in violations:
            penalty = self.penalties.get_penalty(
                violation.principle.rule_number, violation.severity
            )
            total_penalty *= penalty

        # Determine action
        if total_penalty < 0.5:
            action = "downrank"
        else:
            action = "allow"

        return GuardrailResult(
            node_id=node_id,
            passed=total_penalty >= 0.8,
            violations=violations,
            penalty_multiplier=total_penalty,
            action=action,
            principle_checks_count=len(violations),
            cache_hit=False,
        )

    # ─── Warden Integration ─────────────────────────────────────

    def _trigger_warden_veto(
        self, violation: Violation, query_context: dict[str, Any]
    ) -> None:
        """
        Trigger Warden veto for critical violation.

        Args:
            violation: The critical violation
            query_context: Context for the veto
        """
        if not self.warden:
            logger.warning("Warden veto triggered but no Warden instance available")
            return

        veto_trace = (
            f"Retrieval blocked by Mimiry Guardrail:\n"
            f"  Rule: #{violation.principle.rule_number} - {violation.principle.name}\n"
            f"  Severity: {violation.severity.upper()}\n"
            f"  Reason: {violation.reason}\n"
            f"  Node: {violation.node_id}\n"
            f"  Query: {query_context.get('query', 'N/A')}\n\n"
            f"This node was excluded from context assembly to preserve architectural integrity."
        )

        logger.warning(f"Warden veto triggered: {veto_trace}")

        # TODO: Integrate with actual Warden veto API when available
        # For now, just log it

    # ─── Cache Management ───────────────────────────────────────

    def invalidate_cache(self, node_id: str | None = None, reason: str = "unknown") -> None:
        """
        Invalidate cache for specific node or all nodes.

        Args:
            node_id: Specific node to invalidate, or None for all
            reason: Why cache is being invalidated
        """
        if node_id:
            # Invalidate specific node
            keys_to_remove = [k for k in self._principle_cache if k.startswith(f"{node_id}:")]
            for key in keys_to_remove:
                self._principle_cache.pop(key, None)
                self._cache_timestamps.pop(key, None)
            logger.debug(f"Invalidated cache for node {node_id}: {reason}")
        else:
            # Full cache clear
            self._principle_cache.clear()
            self._cache_timestamps.clear()
            logger.info(f"Full cache invalidated: {reason}")

    def invalidate_on_mimiry_update(self) -> None:
        """Clear all cache when Mimiry principles change."""
        self._principle_cache.clear()
        self._cache_timestamps.clear()

        # Reload principles
        self._principles = self._load_principles()

        logger.info("Mimiry principles updated - full guardrail cache invalidated")

    # ─── Helpers ────────────────────────────────────────────────

    def _compute_content_hash(self, content: str) -> str:
        """Compute hash of content for caching."""
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def get_stats(self) -> dict[str, Any]:
        """Get guardrail statistics."""
        cache_hit_rate = (
            self.stats["cache_hits"] / max(self.stats["validations_total"], 1) * 100
        )

        return {
            **self.stats,
            "cache_hit_rate_pct": round(cache_hit_rate, 2),
            "principles_loaded": len(self._principles),
            "allowlist_size": len(self._allowlist),
            "cache_size": len(self._principle_cache),
        }
