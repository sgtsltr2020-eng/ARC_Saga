"""
Hybrid Scorer - Semantic-Structural-Provenance Retrieval Scoring
================================================================

Unified relevance scoring that fuses three signals:
- Semantic: Cosine similarity from embeddings
- Structural: Graph distance + centrality
- Provenance: Utility from linked LoreEntries/MythosChapters

Author: ARC SAGA Development Team (AntiGravity)
Date: December 26, 2025
Status: USMA - Memory Precision Enhancement
"""

import logging
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

@dataclass
class ScorerConfig:
    """Configuration for hybrid scoring weights."""

    # Primary signal weights (must sum to 1.0)
    semantic_weight: float = 0.5
    structural_weight: float = 0.3
    provenance_weight: float = 0.2

    # Secondary weights (applied within structural)
    path_weight: float = 0.6  # Within structural: path distance
    centrality_weight: float = 0.4  # Within structural: centrality

    # Normalization parameters
    max_path_distance: int = 5  # Cap for path normalization
    recency_decay_hours: float = 168.0  # 1 week decay

    # Provenance parameters
    min_provenance_confidence: float = 0.3  # Min confidence to count
    provenance_recency_boost: float = 0.2  # Boost for recent lore

    def validate(self) -> bool:
        """Validate that weights sum to 1.0."""
        total = self.semantic_weight + self.structural_weight + self.provenance_weight
        return abs(total - 1.0) < 0.01


# ═══════════════════════════════════════════════════════════════
# SCORE COMPONENTS
# ═══════════════════════════════════════════════════════════════

@dataclass
class ScoreComponents:
    """Breakdown of individual score components."""
    node_id: str

    # Raw signals
    semantic_raw: float = 0.0
    path_distance: int = -1
    centrality_raw: float = 0.0
    provenance_raw: float = 0.0

    # Normalized signals [0,1]
    semantic_normalized: float = 0.0
    structural_normalized: float = 0.0
    provenance_normalized: float = 0.0

    # Combined score
    combined_score: float = 0.0

    # Metadata
    linked_lore_count: int = 0
    linked_chapter_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "semantic": self.semantic_normalized,
            "structural": self.structural_normalized,
            "provenance": self.provenance_normalized,
            "combined": self.combined_score,
            "path_distance": self.path_distance,
            "lore_count": self.linked_lore_count
        }


# ═══════════════════════════════════════════════════════════════
# HYBRID SCORER
# ═══════════════════════════════════════════════════════════════

class HybridScorer:
    """
    Unified relevance scoring fusing semantic, structural, and provenance signals.

    Formula:
        combined = (
            semantic_weight × normalized_cosine +
            structural_weight × (path_score + centrality_score) +
            provenance_weight × avg_utility
        )

    All components normalized to [0,1] range.
    """

    def __init__(
        self,
        graph: Any = None,  # RepoGraph
        provenance_linker: Any = None,  # ProvenanceLinker
        lorebook: Any = None,  # LoreBook
        mythos_library: Any = None,  # MythosLibrary
        config: ScorerConfig | None = None
    ):
        """
        Initialize the hybrid scorer.

        Args:
            graph: RepoGraph instance for structural signals
            provenance_linker: ProvenanceLinker for bidirectional links
            lorebook: LoreBook for entry utilities
            mythos_library: MythosLibrary for chapter utilities
            config: Scoring configuration
        """
        self.graph = graph
        self.provenance_linker = provenance_linker
        self.lorebook = lorebook
        self.mythos_library = mythos_library
        self.config = config or ScorerConfig()

        # Cache for centrality (expensive to compute)
        self._centrality_cache: dict[str, float] = {}
        self._centrality_computed = False

        # Cache for provenance scores
        self._provenance_cache: dict[str, float] = {}
        self._provenance_dirty: set[str] = set()

        # Stats
        self.scores_computed = 0

    # ─── Main Scoring ──────────────────────────────────────────

    def score(
        self,
        node_id: str,
        query_embedding: np.ndarray | None = None,
        reference_node_id: str | None = None,
        semantic_similarity: float | None = None
    ) -> ScoreComponents:
        """
        Compute hybrid score for a node.

        Args:
            node_id: Target node to score
            query_embedding: Query embedding for semantic similarity
            reference_node_id: Reference node for structural distance
            semantic_similarity: Pre-computed semantic score (optional)

        Returns:
            ScoreComponents with all signals and combined score
        """
        components = ScoreComponents(node_id=node_id)

        # 1. Semantic signal
        if semantic_similarity is not None:
            components.semantic_raw = semantic_similarity
        elif query_embedding is not None:
            components.semantic_raw = self._compute_semantic(node_id, query_embedding)

        components.semantic_normalized = self._normalize_cosine(components.semantic_raw)

        # 2. Structural signal
        if reference_node_id and self.graph:
            path_score, centrality = self._compute_structural(node_id, reference_node_id)
            components.path_distance = int(1 / path_score - 1) if path_score > 0 else -1
            components.centrality_raw = centrality
            components.structural_normalized = (
                self.config.path_weight * path_score +
                self.config.centrality_weight * centrality
            )
        else:
            # Use centrality only if no reference
            centrality = self._get_centrality(node_id)
            components.centrality_raw = centrality
            components.structural_normalized = centrality

        # 3. Provenance signal
        prov_score, lore_count, chapter_count = self._compute_provenance(node_id)
        components.provenance_raw = prov_score
        components.provenance_normalized = prov_score  # Already [0,1]
        components.linked_lore_count = lore_count
        components.linked_chapter_count = chapter_count

        # 4. Combined score
        components.combined_score = (
            self.config.semantic_weight * components.semantic_normalized +
            self.config.structural_weight * components.structural_normalized +
            self.config.provenance_weight * components.provenance_normalized
        )

        self.scores_computed += 1
        return components

    def score_batch(
        self,
        node_ids: list[str],
        query_embedding: np.ndarray | None = None,
        reference_node_id: str | None = None,
        semantic_similarities: dict[str, float] | None = None
    ) -> list[ScoreComponents]:
        """
        Score multiple nodes efficiently.

        Args:
            node_ids: List of nodes to score
            query_embedding: Query embedding
            reference_node_id: Reference node for structural
            semantic_similarities: Pre-computed similarities {node_id: similarity}

        Returns:
            List of ScoreComponents sorted by combined_score descending
        """
        results = []

        for node_id in node_ids:
            sim = semantic_similarities.get(node_id, 0.0) if semantic_similarities else None
            components = self.score(
                node_id=node_id,
                query_embedding=query_embedding,
                reference_node_id=reference_node_id,
                semantic_similarity=sim
            )
            results.append(components)

        # Sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)
        return results

    # ─── Semantic Signal ───────────────────────────────────────

    def _compute_semantic(
        self,
        node_id: str,
        query_embedding: np.ndarray
    ) -> float:
        """Compute semantic similarity via cosine."""
        if not self.graph:
            return 0.0

        node = self.graph.get_node(node_id)
        if not node or node.embedding_vector is None:
            return 0.0

        return self._cosine_similarity(query_embedding, node.embedding_vector)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        a_flat = np.array(a).flatten()
        b_flat = np.array(b).flatten()

        norm_a = np.linalg.norm(a_flat)
        norm_b = np.linalg.norm(b_flat)

        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0

        return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))

    def _normalize_cosine(self, similarity: float) -> float:
        """Normalize cosine similarity to [0,1]."""
        # Cosine is [-1, 1], transform to [0, 1]
        return (similarity + 1.0) / 2.0

    # ─── Structural Signal ─────────────────────────────────────

    def _compute_structural(
        self,
        node_id: str,
        reference_node_id: str
    ) -> tuple[float, float]:
        """
        Compute structural signal: path distance + centrality.

        Returns:
            (path_score, centrality_score) both in [0,1]
        """
        path_score = 0.0
        centrality = 0.0

        if not self.graph:
            return path_score, centrality

        # Path distance
        try:
            distance = nx.shortest_path_length(
                self.graph._graph,
                source=reference_node_id,
                target=node_id
            )
            # Normalize: 1 / (1 + distance), capped at max_path_distance
            distance = min(distance, self.config.max_path_distance)
            path_score = 1.0 / (1.0 + distance)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            path_score = 0.0

        # Centrality
        centrality = self._get_centrality(node_id)

        return path_score, centrality

    def _get_centrality(self, node_id: str) -> float:
        """Get normalized centrality for a node."""
        if node_id in self._centrality_cache:
            return self._centrality_cache[node_id]

        if not self.graph:
            return 0.0

        # Compute degree centrality if not cached
        if not self._centrality_computed:
            self._compute_centrality()

        return self._centrality_cache.get(node_id, 0.0)

    def _compute_centrality(self) -> None:
        """Compute and cache centrality for all nodes."""
        if not self.graph:
            return

        try:
            # Use degree centrality (fast)
            centrality = nx.degree_centrality(self.graph._graph)
            self._centrality_cache = centrality
            self._centrality_computed = True
        except Exception as e:
            logger.warning(f"Failed to compute centrality: {e}")

    # ─── Provenance Signal ─────────────────────────────────────

    def _compute_provenance(self, node_id: str) -> tuple[float, int, int]:
        """
        Compute provenance signal from linked LoreEntries/Chapters.

        Returns:
            (provenance_score, lore_count, chapter_count)
        """
        if node_id not in self._provenance_dirty and node_id in self._provenance_cache:
            cached = self._provenance_cache[node_id]
            return cached, 0, 0  # Counts not cached

        lore_ids = []
        chapter_ids = []
        utilities = []

        # Get linked entries via ProvenanceLinker
        if self.provenance_linker:
            lore_ids = self.provenance_linker.get_lore_for_node(node_id)

        # Fallback to graph node
        if not lore_ids and self.graph:
            node = self.graph.get_node(node_id)
            if node:
                lore_ids = list(node.lore_entry_ids)
                chapter_ids = list(node.mythos_chapter_ids)

        # Aggregate utilities from LoreBook
        if self.lorebook and hasattr(self.lorebook, 'lore_entries'):
            for entry in self.lorebook.lore_entries:
                if getattr(entry, 'entry_id', None) in lore_ids:
                    # Use codex_status as proxy for utility
                    status = getattr(entry, 'codex_status', 'NEUTRAL')
                    if status == "COMPLIANT":
                        utilities.append(1.0)
                    elif status == "VIOLATION":
                        utilities.append(0.0)
                    else:
                        utilities.append(0.5)

        # Aggregate utilities from MythosLibrary
        if self.mythos_library and hasattr(self.mythos_library, 'chapters'):
            for chapter in self.mythos_library.chapters:
                if getattr(chapter, 'chapter_id', None) in chapter_ids:
                    # Use success rate if available
                    patterns = getattr(chapter, 'solved_patterns', [])
                    for pattern in patterns:
                        conf = getattr(pattern, 'confidence', 0.5)
                        utilities.append(conf)

        # Compute average utility
        if utilities:
            provenance_score = sum(utilities) / len(utilities)
        else:
            provenance_score = 0.0

        # Clip to [0,1]
        provenance_score = max(0.0, min(1.0, provenance_score))

        # Cache
        self._provenance_cache[node_id] = provenance_score
        self._provenance_dirty.discard(node_id)

        return provenance_score, len(lore_ids), len(chapter_ids)

    # ─── Weight Learning Hook ──────────────────────────────────

    def record_feedback(
        self,
        node_id: str,
        components: ScoreComponents,
        utility: float
    ) -> dict[str, float]:
        """
        Record feedback for weight learning.

        Returns signal contributions for optimizer training.

        Args:
            node_id: The scored node
            components: Score breakdown
            utility: Observed utility (0-1)

        Returns:
            Dict of signal contributions
        """
        # Compute contribution of each signal to success/failure
        contributions = {
            "semantic_contribution": components.semantic_normalized * utility,
            "structural_contribution": components.structural_normalized * utility,
            "provenance_contribution": components.provenance_normalized * utility,
            "combined_score": components.combined_score,
            "observed_utility": utility
        }

        logger.debug(f"Recorded feedback for {node_id}: {contributions}")
        return contributions

    def get_weight_recommendations(
        self,
        feedback_history: list[dict[str, float]]
    ) -> ScorerConfig:
        """
        Analyze feedback to recommend weight adjustments.

        Args:
            feedback_history: List of feedback dicts from record_feedback

        Returns:
            Recommended ScorerConfig
        """
        if not feedback_history:
            return self.config

        # Compute correlation between each signal and utility
        semantic_sum = sum(f["semantic_contribution"] for f in feedback_history)
        structural_sum = sum(f["structural_contribution"] for f in feedback_history)
        provenance_sum = sum(f["provenance_contribution"] for f in feedback_history)

        total = semantic_sum + structural_sum + provenance_sum
        if total < 0.01:
            return self.config

        # Normalize to weights
        new_semantic = semantic_sum / total
        new_structural = structural_sum / total
        new_provenance = provenance_sum / total

        return ScorerConfig(
            semantic_weight=new_semantic,
            structural_weight=new_structural,
            provenance_weight=new_provenance
        )

    # ─── Cache Management ──────────────────────────────────────

    def invalidate_provenance(self, node_id: str) -> None:
        """Mark provenance cache as dirty for a node."""
        self._provenance_dirty.add(node_id)

    def invalidate_all_provenance(self) -> None:
        """Clear all provenance cache."""
        self._provenance_cache.clear()
        self._provenance_dirty.clear()

    def invalidate_centrality(self) -> None:
        """Clear centrality cache (call after graph changes)."""
        self._centrality_cache.clear()
        self._centrality_computed = False

    # ─── Helper Methods ────────────────────────────────────────

    def get_top_provenance_paths(
        self,
        top_k: int = 10
    ) -> list[tuple[str, float]]:
        """
        Get nodes with highest provenance scores.

        Returns:
            List of (node_id, provenance_score) tuples
        """
        if not self.graph:
            return []

        results = []
        for node_id in self.graph._node_index.keys():
            prov_score, _, _ = self._compute_provenance(node_id)
            if prov_score > 0:
                results.append((node_id, prov_score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_stats(self) -> dict[str, Any]:
        """Get scorer statistics."""
        return {
            "scores_computed": self.scores_computed,
            "centrality_cached": len(self._centrality_cache),
            "provenance_cached": len(self._provenance_cache),
            "config": {
                "semantic_weight": self.config.semantic_weight,
                "structural_weight": self.config.structural_weight,
                "provenance_weight": self.config.provenance_weight
            }
        }

    # ─── Persistence ───────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize scorer state."""
        return {
            "centrality_cache": self._centrality_cache,
            "provenance_cache": self._provenance_cache,
            "scores_computed": self.scores_computed
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        **kwargs
    ) -> "HybridScorer":
        """Deserialize scorer state."""
        scorer = cls(**kwargs)

        scorer._centrality_cache = data.get("centrality_cache", {})
        scorer._centrality_computed = len(scorer._centrality_cache) > 0
        scorer._provenance_cache = data.get("provenance_cache", {})
        scorer.scores_computed = data.get("scores_computed", 0)

        return scorer
