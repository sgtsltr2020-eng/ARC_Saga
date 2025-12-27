"""
Predictive Prefetcher - Ego-Graph Expansion + Predictive Pre-Fetching
======================================================================

Proactive caching system that auto-expands weighted ego-graphs around
changed/accessed nodes and uses RL to predict next likely hotspots.

Author: ARC SAGA Development Team (AntiGravity)
Date: December 26, 2025
Status: Memory Efficiency Upgrade
"""

import heapq
import logging
import math
import pickle
import time
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════


@dataclass
class EgoCacheEntry:
    """A cached ego-graph subgraph."""

    center_node: str
    radius: int
    subgraph: nx.DiGraph
    node_ids: set[str]
    total_weight: float
    created_at: float
    last_accessed: float
    access_count: int = 0
    utility_score: float = 0.0
    retrieval_success_rate: float = 0.5  # NEW: Track retrieval success


@dataclass
class PredictionContext:
    """Context for hotspot prediction."""

    recent_touches: list[str]
    query_trajectory: list[str]
    session_phase: str = "exploration"  # "exploration", "refactor", "debugging"
    time_of_day: float = 0.0


@dataclass
class PrefetchStats:
    """Statistics for prefetcher performance."""

    cache_hits: int = 0
    cache_misses: int = 0
    expansions: int = 0
    predictions_made: int = 0
    predictions_correct: int = 0
    invalidations: int = 0
    evictions: int = 0
    total_prefetch_time_ms: float = 0.0
    prefetch_queue_processed: int = 0


@dataclass
class PrefetchQueueItem:
    """Item in the priority prefetch queue."""

    node_id: str
    confidence: float
    queued_at: float

    def __lt__(self, other: "PrefetchQueueItem") -> bool:
        # Higher confidence = higher priority
        return self.confidence > other.confidence


# ═══════════════════════════════════════════════════════════════
# PREDICTIVE PREFETCHER
# ═══════════════════════════════════════════════════════════════


class PredictivePrefetcher:
    """
    Proactive caching for ego-graph expansion + predictive pre-fetching.

    Features:
    - Auto-expand weighted ego-graphs on edit/access
    - Predict next hotspots using SovereignOptimizer RL policy
    - LRU + utility-based cache eviction
    - Priority prefetch queue (highest confidence first)
    - Persistence via SessionManager
    - FileWatcher integration
    - Chronicle emphasis hooks

    Senior Advisor Adjustments Incorporated:
    - enable_predictions defaults to False (opt-in)
    - Absolute hard cap of 10,000 nodes alongside ratio
    - Utility score includes retrieval success
    - Priority prefetch queue
    - Chronicle hook for accuracy logging
    """

    # Absolute hard cap to prevent OOM on pathological repos
    ABSOLUTE_MAX_CACHED_NODES = 10_000

    def __init__(
        self,
        graph: Any,  # RepoGraph
        optimizer: Any = None,  # SovereignOptimizer
        session_manager: Any = None,
        chronicler: Any = None,  # NEW: For Chronicle hook
        max_cache_ratio: float = 0.25,  # 25% of graph max
        default_radius: int = 3,
        extended_radius: int = 5,
        confidence_threshold: float = 0.8,
        enable_predictions: bool = False,  # NEW: Default OFF (opt-in)
        recency_halflife_hours: float = 24.0,
        max_recent_touches: int = 50,
        max_trajectory_length: int = 20,
    ):
        """
        Initialize the Predictive Prefetcher.

        Args:
            graph: RepoGraph instance
            optimizer: SovereignOptimizer for RL predictions
            session_manager: SessionManager for persistence
            chronicler: Chronicler for Story Time emphasis
            max_cache_ratio: Maximum cache size as ratio of graph (0.25 = 25%)
            default_radius: Default ego-graph expansion radius
            extended_radius: Extended radius for high-centrality nodes
            confidence_threshold: Min confidence to prefetch (0.8)
            enable_predictions: Enable RL predictions (default OFF)
            recency_halflife_hours: Recency decay half-life
            max_recent_touches: Max nodes in recent touch history
            max_trajectory_length: Max nodes in query trajectory
        """
        self.graph = graph
        self.optimizer = optimizer
        self.session_manager = session_manager
        self.chronicler = chronicler

        # Configuration
        self.max_cache_ratio = max_cache_ratio
        self.default_radius = default_radius
        self.extended_radius = extended_radius
        self.confidence_threshold = confidence_threshold
        self.enable_predictions = enable_predictions
        self.recency_halflife_hours = recency_halflife_hours
        self.max_recent_touches = max_recent_touches
        self.max_trajectory_length = max_trajectory_length

        # Cache storage
        self._ego_cache: dict[str, EgoCacheEntry] = {}
        self._cache_order: list[str] = []  # LRU order (oldest first)

        # Priority prefetch queue (max-heap by confidence)
        self._prefetch_queue: list[PrefetchQueueItem] = []

        # Tracking
        self._recent_touches: list[str] = []
        self._query_trajectory: list[str] = []
        self._session_patterns: dict[str, int] = {}
        self._prediction_outcomes: list[tuple[str, bool]] = []  # (node_id, was_accessed)

        # Stats
        self.stats = PrefetchStats()

        logger.info(
            f"PredictivePrefetcher initialized: "
            f"enable_predictions={enable_predictions}, "
            f"max_cache_ratio={max_cache_ratio}, "
            f"confidence_threshold={confidence_threshold}, "
            f"absolute_max={self.ABSOLUTE_MAX_CACHED_NODES}"
        )

    # ─── Cache Size Management ──────────────────────────────────

    def _get_max_cached_nodes(self) -> int:
        """
        Get maximum allowed cached nodes.

        Uses whichever is smaller:
        - Ratio-based limit (max_cache_ratio × graph.node_count)
        - Absolute hard cap (ABSOLUTE_MAX_CACHED_NODES)
        """
        ratio_limit = int(self.graph.node_count * self.max_cache_ratio)
        return min(ratio_limit, self.ABSOLUTE_MAX_CACHED_NODES)

    def _get_current_cached_count(self) -> int:
        """Get total number of nodes currently cached."""
        return sum(len(entry.node_ids) for entry in self._ego_cache.values())

    # ─── Ego-Graph Expansion ────────────────────────────────────

    def expand_ego_graph(
        self,
        node_id: str,
        radius: int | None = None,
        force: bool = False,
    ) -> EgoCacheEntry | None:
        """
        Expand and cache weighted ego-graph around a node.

        Args:
            node_id: Center node for ego-graph
            radius: Expansion radius (None = auto-select based on centrality)
            force: Force re-expansion even if cached

        Returns:
            EgoCacheEntry on success, None on failure
        """
        # Check if already cached
        if not force and node_id in self._ego_cache:
            entry = self._ego_cache[node_id]
            entry.last_accessed = time.time()
            entry.access_count += 1
            self._update_lru(node_id)
            self.stats.cache_hits += 1
            return entry

        self.stats.cache_misses += 1

        # Check if node exists
        if not self.graph.has_node(node_id):
            logger.debug(f"Node not found for ego expansion: {node_id}")
            return None

        start_time = time.time()

        # Determine radius
        if radius is None:
            radius = self._determine_radius(node_id)

        # Build weighted ego-graph
        try:
            subgraph, node_ids, total_weight = self._build_weighted_ego_graph(
                node_id, radius
            )
        except Exception as e:
            logger.warning(f"Ego-graph expansion failed for {node_id}: {e}")
            return None

        # Create cache entry
        now = time.time()
        entry = EgoCacheEntry(
            center_node=node_id,
            radius=radius,
            subgraph=subgraph,
            node_ids=node_ids,
            total_weight=total_weight,
            created_at=now,
            last_accessed=now,
            access_count=1,
            utility_score=total_weight,
        )

        # Evict if needed before adding
        self._evict_if_needed(len(node_ids))

        # Add to cache
        self._ego_cache[node_id] = entry
        self._cache_order.append(node_id)
        self.stats.expansions += 1

        elapsed_ms = (time.time() - start_time) * 1000
        self.stats.total_prefetch_time_ms += elapsed_ms

        logger.debug(
            f"Expanded ego-graph for {node_id}: "
            f"radius={radius}, nodes={len(node_ids)}, "
            f"weight={total_weight:.2f}, time={elapsed_ms:.1f}ms"
        )

        return entry

    def _determine_radius(self, node_id: str) -> int:
        """
        Determine expansion radius based on node centrality.

        High-centrality nodes (core reducers) get extended radius.
        """
        nx_graph = self.graph.graph

        # Calculate degree centrality
        in_degree = nx_graph.in_degree(node_id)
        out_degree = nx_graph.out_degree(node_id)
        total_degree = in_degree + out_degree

        # Get max degree for normalization
        if nx_graph.number_of_nodes() > 0:
            max_degree = max(d for _, d in nx_graph.degree())
        else:
            max_degree = 1

        centrality = total_degree / max(max_degree, 1)

        # High centrality = extended radius
        if centrality > 0.5:
            return self.extended_radius
        return self.default_radius

    def _build_weighted_ego_graph(
        self,
        node_id: str,
        radius: int,
    ) -> tuple[nx.DiGraph, set[str], float]:
        """
        Build weighted ego-graph using NetworkX.

        Returns:
            (subgraph, node_ids, total_weight)
        """
        nx_graph = self.graph.graph.to_undirected()

        # Get ego graph
        try:
            ego = nx.ego_graph(nx_graph, node_id, radius=radius)
        except nx.NetworkXError:
            # Node might be isolated
            ego = nx.DiGraph()
            ego.add_node(node_id)

        node_ids = set(ego.nodes())

        # Compute total weight
        total_weight = 0.0
        for u, v, data in ego.edges(data=True):
            edge_weight = self._compute_edge_weight(u, v, data)
            total_weight += edge_weight

        # Convert to directed subgraph from original
        subgraph = self.graph.graph.subgraph(node_ids).copy()

        return subgraph, node_ids, total_weight

    def _compute_edge_weight(
        self,
        source_id: str,
        target_id: str,
        edge_data: dict,
    ) -> float:
        """
        Compute edge weight for ego-graph expansion.

        Formula: provenance × recency_boost × centrality_score
        """
        # Provenance utility (from edge data)
        provenance = edge_data.get("weight", 1.0)

        # Recency boost
        target_node = self.graph.get_node(target_id)
        if target_node and target_node.last_accessed:
            age_hours = (time.time() - target_node.last_accessed) / 3600
            recency_boost = math.exp(-age_hours / self.recency_halflife_hours)
        else:
            recency_boost = 0.5  # Default for unaccessed

        # Centrality score
        nx_graph = self.graph.graph
        if nx_graph.number_of_nodes() > 0:
            in_deg = nx_graph.in_degree(target_id)
            out_deg = nx_graph.out_degree(target_id)
            max_deg = max(d for _, d in nx_graph.degree()) or 1
            centrality = (in_deg + out_deg) / (2 * max_deg)
        else:
            centrality = 0.5

        # Combine multiplicatively
        raw_weight = provenance * recency_boost * (0.5 + 0.5 * centrality)

        # Normalize to [0, 1]
        return min(1.0, max(0.0, raw_weight))

    # ─── Prediction ─────────────────────────────────────────────

    def predict_next_hotspots(self, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Predict next likely accessed nodes using RL policy.

        Returns:
            List of (node_id, confidence) sorted by confidence descending
        """
        if not self.enable_predictions:
            return []

        if not self.optimizer:
            return self._predict_heuristic(top_k)

        self.stats.predictions_made += 1

        # Build prediction context
        context = PredictionContext(
            recent_touches=self._recent_touches[-self.max_recent_touches :],
            query_trajectory=self._query_trajectory[-self.max_trajectory_length :],
            session_phase=self._infer_session_phase(),
            time_of_day=time.time() % 86400,  # Seconds since midnight
        )

        predictions = []

        # Score candidate nodes
        candidates = self._get_prediction_candidates()

        for node_id in candidates:
            features = self._build_prediction_features(context, node_id)
            try:
                score = self.optimizer.predict_utility(features)
                if score >= self.confidence_threshold:
                    predictions.append((node_id, float(score)))
            except Exception as e:
                logger.debug(f"Prediction failed for {node_id}: {e}")

        # Sort by confidence
        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:top_k]

    def _predict_heuristic(self, top_k: int) -> list[tuple[str, float]]:
        """Fallback heuristic prediction based on patterns."""
        predictions = []

        # Score based on recent touches and patterns
        for node_id, count in self._session_patterns.items():
            if self.graph.has_node(node_id):
                # Simple heuristic: frequency × recency
                recency_idx = (
                    self._recent_touches.index(node_id)
                    if node_id in self._recent_touches
                    else len(self._recent_touches)
                )
                recency_factor = 1.0 / (1 + recency_idx)
                confidence = min(1.0, count * 0.1 * recency_factor)
                if confidence >= self.confidence_threshold:
                    predictions.append((node_id, confidence))

        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_k]

    def _get_prediction_candidates(self) -> list[str]:
        """Get candidate nodes for prediction."""
        candidates = set()

        # Neighbors of recently touched nodes
        for node_id in self._recent_touches[-10:]:
            if self.graph.has_node(node_id):
                for _, neighbor, _ in self.graph.graph.out_edges(node_id, data=True):
                    candidates.add(neighbor)
                for neighbor, _, _ in self.graph.graph.in_edges(node_id, data=True):
                    candidates.add(neighbor)

        # High-frequency pattern nodes
        for node_id in list(self._session_patterns.keys())[:20]:
            candidates.add(node_id)

        # Filter out already cached
        return [nid for nid in candidates if nid not in self._ego_cache]

    def _build_prediction_features(
        self,
        context: PredictionContext,
        node_id: str,
    ) -> np.ndarray:
        """
        Build 64-dim feature vector for RL prediction.

        Features:
        - Recent touch recency (16 dims)
        - Query trajectory similarity (16 dims)
        - Node centrality features (8 dims)
        - Session phase encoding (8 dims)
        - Provenance/pattern features (8 dims)
        - Temporal features (8 dims)
        """
        features = np.zeros(64)

        # Recent touch recency (dims 0-15)
        if node_id in context.recent_touches:
            idx = context.recent_touches.index(node_id)
            features[0] = 1.0 / (1 + idx)
        features[1] = float(node_id in context.recent_touches)

        # Query trajectory (dims 16-31)
        if node_id in context.query_trajectory:
            idx = context.query_trajectory.index(node_id)
            features[16] = 1.0 / (1 + idx)
        features[17] = len([t for t in context.query_trajectory if t == node_id]) / max(
            len(context.query_trajectory), 1
        )

        # Node centrality (dims 32-39)
        if self.graph.has_node(node_id):
            in_deg = self.graph.graph.in_degree(node_id)
            out_deg = self.graph.graph.out_degree(node_id)
            features[32] = min(in_deg / 50, 1.0)
            features[33] = min(out_deg / 50, 1.0)

        # Session phase (dims 40-47)
        phase_map = {"exploration": 0, "refactor": 1, "debugging": 2}
        phase_idx = phase_map.get(context.session_phase, 0)
        features[40 + phase_idx] = 1.0

        # Pattern frequency (dims 48-55)
        freq = self._session_patterns.get(node_id, 0)
        features[48] = min(freq / 10, 1.0)

        # Temporal (dims 56-63)
        hour = context.time_of_day / 3600
        features[56] = math.sin(2 * math.pi * hour / 24)
        features[57] = math.cos(2 * math.pi * hour / 24)

        return features

    def _infer_session_phase(self) -> str:
        """Infer current session phase from patterns."""
        if len(self._recent_touches) < 5:
            return "exploration"

        # Check for refactor pattern (many touches on same files)
        unique_ratio = len(set(self._recent_touches[-20:])) / max(
            len(self._recent_touches[-20:]), 1
        )
        if unique_ratio < 0.3:
            return "refactor"

        return "exploration"

    # ─── Prefetch Queue ─────────────────────────────────────────

    def queue_for_prefetch(self, node_id: str, confidence: float) -> None:
        """Add node to priority prefetch queue."""
        if confidence < self.confidence_threshold:
            return

        item = PrefetchQueueItem(
            node_id=node_id,
            confidence=confidence,
            queued_at=time.time(),
        )
        heapq.heappush(self._prefetch_queue, item)

    def process_prefetch_queue(self, max_items: int = 5) -> int:
        """
        Process prefetch queue (highest confidence first).

        Call during idle cycles.
        """
        processed = 0

        while self._prefetch_queue and processed < max_items:
            item = heapq.heappop(self._prefetch_queue)

            # Skip if already cached
            if item.node_id in self._ego_cache:
                continue

            # Skip stale items (>60s old)
            if time.time() - item.queued_at > 60:
                continue

            # Prefetch
            self.expand_ego_graph(item.node_id)
            processed += 1
            self.stats.prefetch_queue_processed += 1

        return processed

    def prefetch_anticipated(self) -> int:
        """
        Pre-fetch ego-graphs for predicted hotspots.

        Returns number of nodes prefetched.
        """
        if not self.enable_predictions:
            return 0

        predictions = self.predict_next_hotspots()

        # Queue predictions
        for node_id, confidence in predictions:
            self.queue_for_prefetch(node_id, confidence)

        # Process queue
        return self.process_prefetch_queue()

    # ─── Access Tracking ────────────────────────────────────────

    def on_node_accessed(self, node_id: str) -> None:
        """
        Called when a node is accessed during retrieval.

        Updates trajectory, patterns, and triggers prediction.
        """
        # Update trajectory
        self._query_trajectory.append(node_id)
        if len(self._query_trajectory) > self.max_trajectory_length:
            self._query_trajectory.pop(0)

        # Update recent touches
        if node_id in self._recent_touches:
            self._recent_touches.remove(node_id)
        self._recent_touches.append(node_id)
        if len(self._recent_touches) > self.max_recent_touches:
            self._recent_touches.pop(0)

        # Update patterns
        self._session_patterns[node_id] = self._session_patterns.get(node_id, 0) + 1

        # Check prediction accuracy
        self._check_prediction_outcome(node_id)

        # Update cache entry if exists
        if node_id in self._ego_cache:
            entry = self._ego_cache[node_id]
            entry.last_accessed = time.time()
            entry.access_count += 1
            self._update_lru(node_id)

    def update_retrieval_success(self, node_id: str, success: bool) -> None:
        """
        Update retrieval success rate for cached entry.

        Called after hybrid scorer feedback.
        """
        if node_id in self._ego_cache:
            entry = self._ego_cache[node_id]
            # Exponential moving average
            alpha = 0.3
            entry.retrieval_success_rate = (
                alpha * (1.0 if success else 0.0)
                + (1 - alpha) * entry.retrieval_success_rate
            )
            # Update utility score
            entry.utility_score = entry.total_weight * entry.retrieval_success_rate

    def _check_prediction_outcome(self, accessed_node: str) -> None:
        """Check if we correctly predicted this access."""
        # Check if this was in recent predictions
        for pred_node, _ in self._prediction_outcomes[-10:]:
            if pred_node == accessed_node:
                self.stats.predictions_correct += 1
                break

    # ─── Cache Invalidation ─────────────────────────────────────

    def on_graph_updated(self, changed_node_ids: list[str]) -> None:
        """
        Invalidate caches overlapping with changed nodes.

        Uses set intersection for O(1) per entry.
        """
        changed_set = set(changed_node_ids)
        to_invalidate = []

        for node_id, entry in self._ego_cache.items():
            # Check overlap
            if entry.node_ids & changed_set:
                to_invalidate.append(node_id)

        for node_id in to_invalidate:
            self._invalidate_entry(node_id)
            self.stats.invalidations += 1

        if to_invalidate:
            logger.debug(f"Invalidated {len(to_invalidate)} cache entries")

    def _invalidate_entry(self, node_id: str) -> None:
        """Remove a cache entry."""
        if node_id in self._ego_cache:
            del self._ego_cache[node_id]
        if node_id in self._cache_order:
            self._cache_order.remove(node_id)

    # ─── Cache Eviction ─────────────────────────────────────────

    def _evict_if_needed(self, adding_nodes: int) -> list[str]:
        """
        Evict cache entries to stay within memory budget.

        Strategy:
        1. Calculate if adding would exceed budget
        2. Evict lowest utility entries first
        3. Fallback to LRU for tie-breaking
        """
        max_nodes = self._get_max_cached_nodes()
        current = self._get_current_cached_count()

        if current + adding_nodes <= max_nodes:
            return []

        evicted = []
        needed_space = current + adding_nodes - max_nodes

        # Sort by utility (lowest first), then by last accessed (oldest first)
        candidates = sorted(
            self._ego_cache.items(),
            key=lambda x: (x[1].utility_score, x[1].last_accessed),
        )

        freed = 0
        for node_id, entry in candidates:
            if freed >= needed_space:
                break
            freed += len(entry.node_ids)
            evicted.append(node_id)
            self.stats.evictions += 1

        for node_id in evicted:
            self._invalidate_entry(node_id)

        return evicted

    def _update_lru(self, node_id: str) -> None:
        """Move node to end of LRU list (most recently used)."""
        if node_id in self._cache_order:
            self._cache_order.remove(node_id)
        self._cache_order.append(node_id)

    # ─── Cache Lookup ───────────────────────────────────────────

    def get_cached_subgraph(self, node_id: str) -> nx.DiGraph | None:
        """
        Get cached ego-graph if available.

        Returns None on cache miss.
        """
        if node_id in self._ego_cache:
            entry = self._ego_cache[node_id]
            entry.last_accessed = time.time()
            entry.access_count += 1
            self._update_lru(node_id)
            self.stats.cache_hits += 1
            return entry.subgraph

        self.stats.cache_misses += 1
        return None

    def is_cached(self, node_id: str) -> bool:
        """Check if node has cached ego-graph."""
        return node_id in self._ego_cache

    # ─── Chronicle Emphasis ─────────────────────────────────────

    def log_chronicle_emphasis(self) -> None:
        """
        Log prefetch accuracy to Chronicle for Story Time.

        Example: "Saga anticipated 8 of 10 next accesses—foresight sharpening."
        """
        if not self.chronicler:
            return

        if self.stats.predictions_made == 0:
            return

        accuracy = self.stats.predictions_correct / self.stats.predictions_made
        cache_hit_rate = self._get_cache_hit_rate()

        emphasis = (
            f"Saga anticipated {self.stats.predictions_correct} of "
            f"{self.stats.predictions_made} next accesses ({accuracy:.0%} accuracy). "
            f"Cache hit rate: {cache_hit_rate:.0%}. Foresight sharpening."
        )

        try:
            # Would call chronicler.add_emphasis() here
            logger.info(f"Chronicle emphasis: {emphasis}")
        except Exception as e:
            logger.debug(f"Chronicle emphasis failed: {e}")

    # ─── Persistence ────────────────────────────────────────────

    def save_to_session(self) -> bool:
        """Persist cache to SessionManager."""
        if not self.session_manager:
            return False

        try:
            # Serialize cache
            cache_data = {
                "ego_cache": {
                    node_id: {
                        "center_node": entry.center_node,
                        "radius": entry.radius,
                        "subgraph": pickle.dumps(entry.subgraph),
                        "node_ids": list(entry.node_ids),
                        "total_weight": entry.total_weight,
                        "created_at": entry.created_at,
                        "last_accessed": entry.last_accessed,
                        "access_count": entry.access_count,
                        "utility_score": entry.utility_score,
                        "retrieval_success_rate": entry.retrieval_success_rate,
                    }
                    for node_id, entry in self._ego_cache.items()
                },
                "cache_order": self._cache_order,
                "recent_touches": self._recent_touches,
                "query_trajectory": self._query_trajectory,
                "session_patterns": self._session_patterns,
            }

            # Would call session_manager.save("prefetcher_cache", cache_data)
            logger.debug(f"Saved {len(self._ego_cache)} cache entries to session")
            return True

        except Exception as e:
            logger.warning(f"Failed to save prefetcher cache: {e}")
            return False

    def load_from_session(self) -> bool:
        """Restore cache from SessionManager."""
        if not self.session_manager:
            return False

        try:
            # Would call session_manager.load("prefetcher_cache")
            # For now, return False (no data to load)
            return False

        except Exception as e:
            logger.debug(f"Failed to load prefetcher cache: {e}")
            return False

    # ─── Stats ──────────────────────────────────────────────────

    def _get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.stats.cache_hits + self.stats.cache_misses
        if total == 0:
            return 0.0
        return self.stats.cache_hits / total

    def get_stats(self) -> dict[str, Any]:
        """Get prefetcher statistics."""
        return {
            "cache_entries": len(self._ego_cache),
            "cached_nodes": self._get_current_cached_count(),
            "max_cached_nodes": self._get_max_cached_nodes(),
            "cache_hit_rate_pct": self._get_cache_hit_rate() * 100,
            "cache_hits": self.stats.cache_hits,
            "cache_misses": self.stats.cache_misses,
            "expansions": self.stats.expansions,
            "predictions_made": self.stats.predictions_made,
            "predictions_correct": self.stats.predictions_correct,
            "prediction_accuracy_pct": (
                self.stats.predictions_correct / max(self.stats.predictions_made, 1) * 100
            ),
            "invalidations": self.stats.invalidations,
            "evictions": self.stats.evictions,
            "avg_prefetch_time_ms": (
                self.stats.total_prefetch_time_ms / max(self.stats.expansions, 1)
            ),
            "prefetch_queue_size": len(self._prefetch_queue),
            "prefetch_queue_processed": self.stats.prefetch_queue_processed,
            "enable_predictions": self.enable_predictions,
        }

    def enable_predictions_mode(self) -> None:
        """Enable prediction mode (after warmup)."""
        self.enable_predictions = True
        logger.info("Prediction mode enabled")

    def disable_predictions_mode(self) -> None:
        """Disable prediction mode."""
        self.enable_predictions = False
        logger.info("Prediction mode disabled")
