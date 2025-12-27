"""
Training Signals - Structured Feature Extraction for RL
========================================================

Provides grounded, observable features for the SovereignOptimizer.
Replaces noise-based context_vectors with meaningful signals.

Author: ARC SAGA Development Team (AntiGravity)
Date: December 26, 2025
Status: USMA P1 Fix - Real Training Signals
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# FEATURE DEFINITIONS
# ═══════════════════════════════════════════════════════════════

@dataclass
class RetrievalFeatures:
    """
    Structured feature vector for retrieval/search events.

    All features are normalized to [0, 1] range for stable training.
    """
    # Core retrieval metrics
    retrieval_confidence: float = 0.0      # Top cosine similarity [0, 1]
    graph_distance: float = 0.0            # Normalized avg shortest path [0, 1]
    node_centrality: float = 0.0           # Avg degree centrality [0, 1]

    # Temporal signals
    recency_score: float = 0.0             # Time since last access [0, 1]

    # Query-content overlap
    co_occurrence: float = 0.0             # Term overlap score [0, 1]

    # Feedback signals (may be sparse)
    user_feedback: float = 0.5             # Normalized: -1→0, 0→0.5, +1→1

    # Outcome signals
    was_used: float = 0.0                  # Binary: result used in output [0, 1]
    token_savings: float = 0.0             # Estimated tokens saved, normalized [0, 1]
    simulation_success: float = 0.5        # Shadow trial pass rate [0, 1]

    def to_vector(self) -> np.ndarray:
        """Convert to numpy array for MLP input."""
        return np.array([
            self.retrieval_confidence,
            self.graph_distance,
            self.node_centrality,
            self.recency_score,
            self.co_occurrence,
            self.user_feedback,
            self.was_used,
            self.token_savings,
            self.simulation_success
        ], dtype=np.float32)

    def to_list(self) -> list[float]:
        """Convert to list for serialization."""
        return self.to_vector().tolist()

    @staticmethod
    def feature_dim() -> int:
        """Number of features in the vector."""
        return 9

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> "RetrievalFeatures":
        """Create from numpy array."""
        return cls(
            retrieval_confidence=float(vec[0]) if len(vec) > 0 else 0.0,
            graph_distance=float(vec[1]) if len(vec) > 1 else 0.0,
            node_centrality=float(vec[2]) if len(vec) > 2 else 0.0,
            recency_score=float(vec[3]) if len(vec) > 3 else 0.0,
            co_occurrence=float(vec[4]) if len(vec) > 4 else 0.0,
            user_feedback=float(vec[5]) if len(vec) > 5 else 0.5,
            was_used=float(vec[6]) if len(vec) > 6 else 0.0,
            token_savings=float(vec[7]) if len(vec) > 7 else 0.0,
            simulation_success=float(vec[8]) if len(vec) > 8 else 0.5
        )


# ═══════════════════════════════════════════════════════════════
# FEATURE EXTRACTOR
# ═══════════════════════════════════════════════════════════════

class RetrievalFeatureExtractor:
    """
    Extracts structured features from search results.

    Features are grounded in observable signals:
    - Cosine similarity from search
    - Graph structure metrics
    - Temporal patterns
    - Query-content overlap
    """

    def __init__(self, max_graph_distance: int = 10, max_tokens: int = 2000):
        """
        Initialize the feature extractor.

        Args:
            max_graph_distance: Max expected graph distance for normalization
            max_tokens: Max expected token savings for normalization
        """
        self.max_graph_distance = max_graph_distance
        self.max_tokens = max_tokens

    def extract_from_search(
        self,
        query: str,
        results: list[Any],  # list[SearchResult]
        graph: Any = None,  # RepoGraph
    ) -> RetrievalFeatures:
        """
        Extract features from search results.

        Args:
            query: The search query
            results: List of SearchResult from SemanticSearchEngine
            graph: Optional RepoGraph for structural metrics

        Returns:
            RetrievalFeatures with computed values
        """
        features = RetrievalFeatures()

        if not results:
            return features

        # Retrieval confidence (top similarity)
        features.retrieval_confidence = float(results[0].cosine_similarity)

        # Graph distance (avg of top-k, normalized)
        distances = [r.graph_distance for r in results[:5] if r.graph_distance >= 0]
        if distances:
            avg_distance = np.mean(distances)
            features.graph_distance = 1.0 - min(avg_distance / self.max_graph_distance, 1.0)

        # Node centrality (if graph available)
        if graph is not None:
            try:
                centralities = []
                for r in results[:5]:
                    node = graph.get_node(r.node_id)
                    if node:
                        # Degree centrality = degree / (n-1)
                        degree = graph.graph.degree(r.node_id)
                        n = graph.node_count
                        centrality = degree / max(n - 1, 1) if n > 1 else 0
                        centralities.append(centrality)
                if centralities:
                    features.node_centrality = float(np.mean(centralities))
            except Exception:
                pass

        # Recency score (from results)
        recency_scores = [r.recency_score for r in results[:5] if hasattr(r, 'recency_score')]
        if recency_scores:
            features.recency_score = float(np.mean(recency_scores))

        # Co-occurrence (query term overlap with results)
        features.co_occurrence = self._compute_co_occurrence(query, results)

        return features

    def _compute_co_occurrence(self, query: str, results: list[Any]) -> float:
        """
        Compute query-content term overlap.

        Returns:
            Score between 0 and 1
        """
        query_terms = set(query.lower().split())
        if not query_terms:
            return 0.0

        matches = 0
        total = 0

        for r in results[:5]:
            name_terms = set(r.name.lower().replace("_", " ").split())
            overlap = len(query_terms & name_terms)
            matches += overlap
            total += len(query_terms)

        return min(matches / max(total, 1), 1.0)

    def add_outcome_signals(
        self,
        features: RetrievalFeatures,
        user_feedback: int = 0,  # -1, 0, +1
        was_used: bool = False,
        token_savings: int = 0,
        simulation_passed: bool | None = None
    ) -> RetrievalFeatures:
        """
        Add outcome signals to features after task completion.

        Args:
            features: Base features from search
            user_feedback: Explicit user rating (-1, 0, +1)
            was_used: Whether result was used in output
            token_savings: Estimated tokens saved
            simulation_passed: Shadow trial result (None = not run)

        Returns:
            Updated features
        """
        # User feedback: normalize -1 to 0, 0 to 0.5, +1 to 1
        features.user_feedback = (user_feedback + 1) / 2.0

        # Was used: binary
        features.was_used = 1.0 if was_used else 0.0

        # Token savings: normalize
        features.token_savings = min(token_savings / self.max_tokens, 1.0)

        # Simulation success
        if simulation_passed is not None:
            features.simulation_success = 1.0 if simulation_passed else 0.0

        return features


# ═══════════════════════════════════════════════════════════════
# UTILITY CALCULATOR
# ═══════════════════════════════════════════════════════════════

@dataclass
class UtilityConfig:
    """Configuration for utility calculation weights."""
    user_feedback_weight: float = 0.5
    simulation_weight: float = 0.3
    token_efficiency_weight: float = 0.2

    # Proxy signal weights (when explicit feedback missing)
    was_used_proxy: float = 0.6  # Implicit positive if result was used
    not_used_penalty: float = -0.2  # Slight negative if not used


class UtilityCalculator:
    """
    Calculates target utility from features.

    Utility = weighted combination of:
    - User feedback (explicit thumbs up/down)
    - Simulation success (shadow trial pass rate)
    - Token efficiency (savings ratio)
    - Proxy signals (when explicit feedback missing)
    """

    def __init__(self, config: UtilityConfig | None = None):
        """Initialize the utility calculator."""
        self.config = config or UtilityConfig()

    def compute_utility(self, features: RetrievalFeatures) -> float:
        """
        Compute target utility from features.

        Args:
            features: Extracted retrieval features

        Returns:
            Utility score between 0 and 1
        """
        # Start with weighted explicit signals
        utility = 0.0
        total_weight = 0.0

        # User feedback (most important)
        if features.user_feedback != 0.5:  # Has explicit feedback
            utility += features.user_feedback * self.config.user_feedback_weight
            total_weight += self.config.user_feedback_weight

        # Simulation success
        if features.simulation_success != 0.5:  # Simulation was run
            utility += features.simulation_success * self.config.simulation_weight
            total_weight += self.config.simulation_weight

        # Token efficiency
        utility += features.token_savings * self.config.token_efficiency_weight
        total_weight += self.config.token_efficiency_weight

        # Proxy signals (when explicit feedback missing)
        if features.user_feedback == 0.5:  # No explicit feedback
            if features.was_used > 0.5:
                utility += self.config.was_used_proxy * 0.3
                total_weight += 0.3
            else:
                utility += (0.5 + self.config.not_used_penalty) * 0.2
                total_weight += 0.2

        # Add retrieval confidence as baseline
        utility += features.retrieval_confidence * 0.2
        total_weight += 0.2

        # Normalize by total weight
        if total_weight > 0:
            utility = utility / total_weight

        return np.clip(utility, 0.0, 1.0)

    def compute_batch_utility(self, feature_list: list[RetrievalFeatures]) -> list[float]:
        """Compute utility for a batch of features."""
        return [self.compute_utility(f) for f in feature_list]


# ═══════════════════════════════════════════════════════════════
# FEATURE NORMALIZER
# ═══════════════════════════════════════════════════════════════

class FeatureNormalizer:
    """
    MinMax normalization with running statistics.

    Maintains min/max per feature across sessions for stable training.
    """

    def __init__(self, feature_dim: int = 9):
        """Initialize the normalizer."""
        self.feature_dim = feature_dim

        # Running statistics
        self.feature_min = np.zeros(feature_dim, dtype=np.float32)
        self.feature_max = np.ones(feature_dim, dtype=np.float32)
        self.sample_count = 0

        # Small epsilon for numerical stability
        self.eps = 1e-8

    def update(self, features: np.ndarray) -> None:
        """
        Update running statistics with new features.

        Args:
            features: 1D or 2D array of features
        """
        features = np.atleast_2d(features)

        if features.shape[1] != self.feature_dim:
            logger.warning(f"Feature dim mismatch: {features.shape[1]} vs {self.feature_dim}")
            return

        batch_min = np.min(features, axis=0)
        batch_max = np.max(features, axis=0)

        if self.sample_count == 0:
            self.feature_min = batch_min
            self.feature_max = batch_max
        else:
            self.feature_min = np.minimum(self.feature_min, batch_min)
            self.feature_max = np.maximum(self.feature_max, batch_max)

        self.sample_count += features.shape[0]

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using MinMax scaling.

        Args:
            features: 1D or 2D array

        Returns:
            Normalized features in [0, 1] range
        """
        features = np.atleast_2d(features)

        range_vals = self.feature_max - self.feature_min + self.eps
        normalized = (features - self.feature_min) / range_vals

        return np.clip(normalized, 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        """Serialize normalization stats."""
        return {
            "feature_min": self.feature_min.tolist(),
            "feature_max": self.feature_max.tolist(),
            "sample_count": self.sample_count
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureNormalizer":
        """Deserialize normalization stats."""
        normalizer = cls(feature_dim=len(data.get("feature_min", [])) or 9)
        normalizer.feature_min = np.array(data.get("feature_min", [0] * 9), dtype=np.float32)
        normalizer.feature_max = np.array(data.get("feature_max", [1] * 9), dtype=np.float32)
        normalizer.sample_count = data.get("sample_count", 0)
        return normalizer


# ═══════════════════════════════════════════════════════════════
# TRAINING SIGNAL MANAGER
# ═══════════════════════════════════════════════════════════════

@dataclass
class TrainingEvent:
    """A complete training event with features and outcome."""
    event_id: str
    task_id: str
    features: RetrievalFeatures
    utility: float
    retrieval_path: list[str]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            "event_id": self.event_id,
            "task_id": self.task_id,
            "features": self.features.to_list(),
            "utility": self.utility,
            "retrieval_path": self.retrieval_path,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingEvent":
        """Deserialize from storage."""
        return cls(
            event_id=data["event_id"],
            task_id=data["task_id"],
            features=RetrievalFeatures.from_vector(np.array(data["features"])),
            utility=data["utility"],
            retrieval_path=data["retrieval_path"],
            timestamp=data.get("timestamp", time.time())
        )


class TrainingSignalManager:
    """
    Manages the full training signal pipeline.

    Handles:
    - Feature extraction from search results
    - Outcome signal collection
    - Utility calculation
    - Batch accumulation
    - Integration with SovereignOptimizer
    """

    def __init__(
        self,
        optimizer: Any = None,  # SovereignOptimizer
        batch_threshold: int = 10,
        normalizer: FeatureNormalizer | None = None
    ):
        """
        Initialize the training signal manager.

        Args:
            optimizer: SovereignOptimizer instance
            batch_threshold: Accumulate this many events before training
            normalizer: Optional pre-existing normalizer
        """
        self.optimizer = optimizer
        self.batch_threshold = batch_threshold

        self.extractor = RetrievalFeatureExtractor()
        self.calculator = UtilityCalculator()
        self.normalizer = normalizer or FeatureNormalizer()

        # Pending events (waiting for outcome signals)
        self.pending_events: dict[str, dict[str, Any]] = {}

        # Accumulated training events
        self.training_buffer: list[TrainingEvent] = []

        # Stats
        self.total_events = 0
        self.train_calls = 0

    def record_search(
        self,
        search_id: str,
        query: str,
        results: list[Any],
        graph: Any = None
    ) -> str:
        """
        Record a search event (first phase).

        Call complete_event() later with outcome signals.

        Args:
            search_id: Unique identifier for this search
            query: The search query
            results: Search results
            graph: Optional RepoGraph

        Returns:
            The search_id for later completion
        """
        # Extract base features
        features = self.extractor.extract_from_search(query, results, graph)

        self.pending_events[search_id] = {
            "query": query,
            "features": features,
            "retrieval_path": [r.node_id for r in results[:5]],
            "timestamp": time.time()
        }

        logger.debug(f"Recorded search: {search_id}")
        return search_id

    def complete_event(
        self,
        search_id: str,
        user_feedback: int = 0,
        was_used: bool = False,
        token_savings: int = 0,
        simulation_passed: bool | None = None,
        task_id: str = ""
    ) -> TrainingEvent | None:
        """
        Complete a search event with outcome signals.

        Args:
            search_id: The search to complete
            user_feedback: Explicit rating (-1, 0, +1)
            was_used: Whether result was used
            token_savings: Tokens saved
            simulation_passed: Shadow trial result
            task_id: Associated task ID

        Returns:
            TrainingEvent if successfully completed
        """
        if search_id not in self.pending_events:
            logger.warning(f"Unknown search_id: {search_id}")
            return None

        pending = self.pending_events.pop(search_id)
        features = pending["features"]

        # Add outcome signals
        self.extractor.add_outcome_signals(
            features,
            user_feedback=user_feedback,
            was_used=was_used,
            token_savings=token_savings,
            simulation_passed=simulation_passed
        )

        # Calculate utility
        utility = self.calculator.compute_utility(features)

        # Create training event
        event = TrainingEvent(
            event_id=f"train_{time.time()}",
            task_id=task_id or search_id,
            features=features,
            utility=utility,
            retrieval_path=pending["retrieval_path"]
        )

        # Add to buffer
        self.training_buffer.append(event)
        self.total_events += 1

        # Update normalizer
        self.normalizer.update(features.to_vector())

        logger.debug(f"Completed event: {search_id} → utility={utility:.3f}")

        # Check if batch ready
        if len(self.training_buffer) >= self.batch_threshold:
            self._train_batch()

        return event

    def _train_batch(self) -> dict[str, Any]:
        """Train optimizer on accumulated batch."""
        if not self.training_buffer or self.optimizer is None:
            return {"status": "no_data_or_optimizer"}

        # Prepare batch
        batch = self.training_buffer[:self.batch_threshold]
        self.training_buffer = self.training_buffer[self.batch_threshold:]

        # Record feedback events to optimizer
        for event in batch:
            # Normalize features
            normalized = self.normalizer.normalize(event.features.to_vector())

            self.optimizer.record_feedback(
                task_id=event.task_id,
                context_vector=normalized.flatten().tolist(),
                retrieval_path=event.retrieval_path,
                confidence=event.features.retrieval_confidence,
                success=event.utility > 0.5
            )

        self.train_calls += 1

        logger.info(f"Training batch: {len(batch)} events, total calls: {self.train_calls}")

        return {
            "status": "trained",
            "batch_size": len(batch),
            "total_events": self.total_events
        }

    def force_train(self) -> dict[str, Any]:
        """Force training on current buffer regardless of size."""
        if not self.training_buffer or self.optimizer is None:
            return {"status": "no_data"}

        # Temporarily lower threshold
        old_threshold = self.batch_threshold
        self.batch_threshold = 1
        result = self._train_batch()
        self.batch_threshold = old_threshold

        return result

    def get_stats(self) -> dict[str, Any]:
        """Get training signal statistics."""
        return {
            "total_events": self.total_events,
            "train_calls": self.train_calls,
            "pending_searches": len(self.pending_events),
            "buffer_size": len(self.training_buffer),
            "normalizer_samples": self.normalizer.sample_count
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence."""
        return {
            "normalizer": self.normalizer.to_dict(),
            "training_buffer": [e.to_dict() for e in self.training_buffer],
            "total_events": self.total_events,
            "train_calls": self.train_calls
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], optimizer: Any = None) -> "TrainingSignalManager":
        """Deserialize from persistence."""
        normalizer = FeatureNormalizer.from_dict(data.get("normalizer", {}))
        manager = cls(optimizer=optimizer, normalizer=normalizer)

        manager.training_buffer = [
            TrainingEvent.from_dict(e)
            for e in data.get("training_buffer", [])
        ]
        manager.total_events = data.get("total_events", 0)
        manager.train_calls = data.get("train_calls", 0)

        return manager
