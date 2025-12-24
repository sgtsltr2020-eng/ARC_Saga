"""
LoreBook - Empirical Memory System
===================================

Balances Mimiry's idealism with project-specific learned patterns.
Stores decisions, tracks outcomes, extracts patterns.

Author: ARC SAGA Development Team
Date: December 17, 2025
Status: Phase 3A - Core Memory System
"""

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from saga.resilience.async_utils import with_retry, with_timeout

logger = logging.getLogger(__name__)


@dataclass
class Decision:
    """A decision made by SAGA, stored for learning."""

    decision_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    trace_id: str = ""

    # Question context
    question: str = ""
    context: dict[str, Any] = field(default_factory=dict)

    # Rulings
    mimiry_ruling: str = ""           # What Mimiry said (ideal)
    lorebook_ruling: str = ""         # What we actually did
    divergence_rationale: str = ""    # WHY we diverged from ideal

    # Metadata
    confidence: float = 1.0           # Decays over time
    cited_rules: list[int] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    # Success tracking
    outcome_recorded: bool = False
    success: Optional[bool] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Decision":
        """Deserialize from dict."""
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class Outcome:
    """The result of a decision (success/failure + metrics)."""

    outcome_id: str = field(default_factory=lambda: str(uuid4()))
    decision_id: str = ""
    learned_at: datetime = field(default_factory=datetime.utcnow)

    # Results
    success: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)  # test_coverage, errors, etc.
    feedback: str = ""                                      # What we learned

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        data = asdict(self)
        data["learned_at"] = self.learned_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Outcome":
        """Deserialize from dict."""
        data["learned_at"] = datetime.fromisoformat(data["learned_at"])
        return cls(**data)


@dataclass
class Pattern:
    """An extracted pattern from multiple decisions."""

    pattern_id: str = field(default_factory=lambda: str(uuid4()))
    description: str = ""
    success_rate: float = 0.0
    examples: list[str] = field(default_factory=list)
    learned_from: list[str] = field(default_factory=list)  # decision_ids

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return asdict(self)


class LoreBook:
    """
    Empirical Memory System for SAGA.

    Balances Mimiry's idealistic SagaCodex with project-specific learnings.
    Consult LoreBook FIRST (fast, local), then Mimiry (slower, ideal).

    Architecture:
        - DecisionStore: Persistent SQLite storage
        - VectorSearch: Similarity matching for past decisions
        - Confidence decay: Old decisions lose relevance
        - Pattern extraction: Generalize from multiple outcomes

    Usage:
        lorebook = LoreBook(project_root=Path.cwd())
        await lorebook.initialize()

        # Check memory first
        past_decision = await lorebook.consult("Should I use async Redis?", context)

        if past_decision and past_decision.confidence > 0.8:
            # Use learned approach
            ...
        else:
            # Fall back to Mimiry
            ...
    """

    def __init__(self, project_root: str = ".") -> None:
        """Initialize LoreBook for a project."""
        self.project_root = project_root
        self.store: Any = None  # DecisionStore (injected)
        self.vector_search: Any = None  # VectorSearch (injected)

        # Configuration
        self.confidence_decay_days = 90  # Decisions decay after 90 days
        self.similarity_threshold = 0.7   # Min similarity for match
        self.min_confidence = 0.6         # Min confidence to use

    async def initialize(self) -> None:
        """
        Initialize storage backend.

        Creates .saga/lorebook.db if not exists.
        Loads vector embeddings for existing decisions.
        """
        from saga.storage.decision_store import DecisionStore
        from saga.storage.vector_search import VectorSearch

        self.store = DecisionStore(self.project_root)
        await self.store.initialize()

        self.vector_search = VectorSearch()

        # Load existing decisions into vector search
        decisions = await self.store.get_all_decisions()
        for decision in decisions:
            self.vector_search.add_decision(decision)

        logger.info(
            "LoreBook initialized",
            extra={
                "project_root": self.project_root,
                "decisions_count": len(decisions)
            }
        )

    @with_timeout(10.0)
    @with_retry(max_attempts=3, backoff=1.0)
    async def consult(
        self,
        question: str,
        context: dict[str, Any],
        trace_id: str = ""
    ) -> Optional[Decision]:
        """
        Find similar past decision in LoreBook.

        Args:
            question: Question being asked
            context: Current context (framework, domain, etc.)
            trace_id: Distributed tracing ID

        Returns:
            Past decision if found with high confidence, else None

        Algorithm:
            1. Vector search for similar questions
            2. Apply confidence decay (old decisions lose confidence)
            3. Filter by context similarity
            4. Return best match if confidence > threshold
        """
        if not self.store or not self.vector_search:
            logger.warning("LoreBook not initialized")
            return None

        # Find similar decisions
        similar: list[tuple[Decision, float]] = self.vector_search.find_similar(
            question,
            context,
            top_k=5,
            threshold=self.similarity_threshold
        )

        if not similar:
            logger.debug(
                "No similar decisions found",
                extra={"question": question, "trace_id": trace_id}
            )
            return None

        # Apply confidence decay and filter
        best_decision: Optional[Decision] = None
        best_confidence: float = 0.0

        for decision, similarity_score in similar:
            # Decay confidence based on age
            age_days = (datetime.utcnow() - decision.timestamp).days
            decay_factor = max(0.0, 1.0 - (age_days / self.confidence_decay_days))

            adjusted_confidence = decision.confidence * decay_factor * similarity_score

            if adjusted_confidence > best_confidence and adjusted_confidence >= self.min_confidence:
                best_confidence = adjusted_confidence
                best_decision = decision

        if best_decision:
            # Return a copy with the effective (decayed) confidence
            # so the caller knows the current weight of this memory
            result = Decision.from_dict(best_decision.to_dict())
            result.confidence = best_confidence

            logger.info(
                "LoreBook match found",
                extra={
                    "question": question,
                    "decision_id": best_decision.decision_id,
                    "confidence": best_confidence,
                    "age_days": (datetime.utcnow() - best_decision.timestamp).days,
                    "trace_id": trace_id
                }
            )
            return result

        return None

    @with_timeout(5.0)
    async def record_decision(self, decision: Decision) -> None:
        """
        Store a new decision in LoreBook.

        Args:
            decision: Decision to record

        Side effects:
            - Persists to SQLite
            - Adds to vector search index
        """
        if not self.store or not self.vector_search:
            logger.error("LoreBook not initialized")
            return

        await self.store.save_decision(decision)
        self.vector_search.add_decision(decision)

        logger.info(
            "Decision recorded",
            extra={
                "decision_id": decision.decision_id,
                "question": decision.question,
                "diverged_from_mimiry": decision.mimiry_ruling != decision.lorebook_ruling,
                "trace_id": decision.trace_id
            }
        )

    @with_timeout(5.0)
    async def record_outcome(self, outcome: Outcome) -> None:
        """
        Record the outcome of a decision (learn from results).

        Args:
            outcome: Outcome to record

        Side effects:
            - Updates decision.success and decision.confidence
            - If failure, reduces confidence
            - If success, increases confidence (up to 1.0)
        """
        if not self.store:
            logger.error("LoreBook not initialized")
            return

        await self.store.save_outcome(outcome)

        # Update decision based on outcome
        decision = await self.store.get_decision(outcome.decision_id)
        if decision:
            decision.outcome_recorded = True
            decision.success = outcome.success

            # Adjust confidence based on outcome
            if outcome.success:
                decision.confidence = min(1.0, decision.confidence * 1.1)  # Boost
            else:
                decision.confidence = max(0.0, decision.confidence * 0.7)  # Penalize

            await self.store.update_decision(decision)

            logger.info(
                "Outcome recorded",
                extra={
                    "outcome_id": outcome.outcome_id,
                    "decision_id": outcome.decision_id,
                    "success": outcome.success,
                    "new_confidence": decision.confidence
                }
            )

    async def get_project_patterns(self) -> list[Pattern]:
        """
        Extract learned patterns from multiple decisions.

        Returns:
            List of patterns discovered (e.g., "Always use async Redis")

        Algorithm:
            1. Group decisions by tags/context
            2. Calculate success rate per group
            3. Generalize into patterns
        """
        if not self.store:
            return []

        decisions = await self.store.get_all_decisions()
        outcomes = await self.store.get_all_outcomes()

        # Map outcomes to decisions only if needed, currently unused in this simplified logic
        # outcome_map = {o.decision_id: o for o in outcomes}

        # Group by tags
        tag_groups: dict[str, list[Decision]] = {}
        for decision in decisions:
            for tag in decision.tags:
                tag_groups.setdefault(tag, []).append(decision)

        # Extract patterns
        patterns: list[Pattern] = []

        for tag, decisions_with_tag in tag_groups.items():
            if len(decisions_with_tag) < 3:  # Need multiple examples
                continue

            successes = sum(
                1 for d in decisions_with_tag
                if d.outcome_recorded and d.success
            )
            total = sum(1 for d in decisions_with_tag if d.outcome_recorded)

            if total > 0:
                success_rate = successes / total

                if success_rate > 0.7:  # High success pattern
                    pattern = Pattern(
                        description=f"Pattern: {tag} (works well in this project)",
                        success_rate=success_rate,
                        examples=[d.lorebook_ruling for d in decisions_with_tag[:3]],
                        learned_from=[d.decision_id for d in decisions_with_tag]
                    )
                    patterns.append(pattern)

        logger.info(
            "Patterns extracted",
            extra={"patterns_count": len(patterns)}
        )

        return patterns

    async def get_decision_history(
        self,
        limit: int = 10,
        tag: Optional[str] = None
    ) -> list[Decision]:
        """Get recent decisions, optionally filtered by tag."""
        if not self.store:
            return []

        return await self.store.get_recent_decisions(limit=limit, tag=tag)  # type: ignore

    async def close(self) -> None:
        """Close storage connections."""
        if self.store:
            await self.store.close()
