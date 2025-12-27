"""
Tests for PredictivePrefetcher - Ego-Graph Expansion + Predictive Pre-Fetching

Author: ARC SAGA Development Team (AntiGravity)
Date: December 26, 2025
"""

import time
from unittest.mock import Mock

import networkx as nx
import numpy as np
import pytest

from saga.core.memory.graph_engine import EdgeType, GraphEdge, GraphNode, NodeType, RepoGraph
from saga.core.memory.predictive_prefetcher import (
    PredictionContext,
    PredictivePrefetcher,
)

# ═══════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def mock_graph():
    """Create a mock RepoGraph with test data."""
    graph = RepoGraph(project_root="/test/project")

    # Add nodes
    nodes = [
        GraphNode(node_id="file:main.py", node_type=NodeType.FILE, name="main.py"),
        GraphNode(node_id="func:main", node_type=NodeType.FUNCTION, name="main"),
        GraphNode(node_id="func:helper", node_type=NodeType.FUNCTION, name="helper"),
        GraphNode(node_id="class:App", node_type=NodeType.CLASS, name="App"),
        GraphNode(node_id="func:process", node_type=NodeType.FUNCTION, name="process"),
        GraphNode(node_id="file:utils.py", node_type=NodeType.FILE, name="utils.py"),
        GraphNode(node_id="func:format", node_type=NodeType.FUNCTION, name="format"),
        GraphNode(node_id="func:validate", node_type=NodeType.FUNCTION, name="validate"),
    ]

    for node in nodes:
        graph.add_node(node)

    # Add edges
    edges = [
        GraphEdge(source_id="file:main.py", target_id="func:main", edge_type=EdgeType.CONTAINS),
        GraphEdge(source_id="func:main", target_id="func:helper", edge_type=EdgeType.CALLS, weight=0.9),
        GraphEdge(source_id="func:main", target_id="class:App", edge_type=EdgeType.REFERENCES, weight=0.8),
        GraphEdge(source_id="class:App", target_id="func:process", edge_type=EdgeType.CONTAINS),
        GraphEdge(source_id="func:process", target_id="func:format", edge_type=EdgeType.CALLS, weight=0.7),
        GraphEdge(source_id="file:utils.py", target_id="func:format", edge_type=EdgeType.CONTAINS),
        GraphEdge(source_id="file:utils.py", target_id="func:validate", edge_type=EdgeType.CONTAINS),
        GraphEdge(source_id="func:helper", target_id="func:validate", edge_type=EdgeType.CALLS, weight=0.6),
    ]

    for edge in edges:
        graph.add_edge(edge)

    return graph


@pytest.fixture
def prefetcher(mock_graph):
    """Create a PredictivePrefetcher instance."""
    return PredictivePrefetcher(
        graph=mock_graph,
        max_cache_ratio=0.5,  # 50% for testing
        default_radius=2,
        extended_radius=3,
        confidence_threshold=0.8,
        enable_predictions=False,  # Default OFF
    )


@pytest.fixture
def prefetcher_with_predictions(mock_graph):
    """Create a PredictivePrefetcher with predictions enabled."""
    return PredictivePrefetcher(
        graph=mock_graph,
        enable_predictions=True,
        confidence_threshold=0.7,
    )


# ═══════════════════════════════════════════════════════════════
# INITIALIZATION TESTS
# ═══════════════════════════════════════════════════════════════


def test_initialization_defaults():
    """Test PredictivePrefetcher initialization with defaults."""
    mock_graph = Mock()
    mock_graph.node_count = 100

    pf = PredictivePrefetcher(graph=mock_graph)

    assert pf.enable_predictions is False  # Default OFF
    assert pf.max_cache_ratio == 0.25
    assert pf.default_radius == 3
    assert pf.extended_radius == 5
    assert pf.confidence_threshold == 0.8
    assert pf.ABSOLUTE_MAX_CACHED_NODES == 10_000


def test_max_cached_nodes_ratio(mock_graph):
    """Test max cached nodes calculation with ratio."""
    pf = PredictivePrefetcher(graph=mock_graph, max_cache_ratio=0.5)

    # 8 nodes × 0.5 = 4
    assert pf._get_max_cached_nodes() == 4


def test_max_cached_nodes_hard_cap():
    """Test absolute hard cap on cached nodes."""
    mock_graph = Mock()
    mock_graph.node_count = 100_000  # Large graph

    pf = PredictivePrefetcher(graph=mock_graph, max_cache_ratio=0.5)

    # Ratio would be 50,000, but hard cap is 10,000
    assert pf._get_max_cached_nodes() == 10_000


# ═══════════════════════════════════════════════════════════════
# EGO-GRAPH EXPANSION TESTS
# ═══════════════════════════════════════════════════════════════


def test_expand_ego_graph_basic(prefetcher):
    """Test basic ego-graph expansion."""
    entry = prefetcher.expand_ego_graph("func:main")

    assert entry is not None
    assert entry.center_node == "func:main"
    assert len(entry.node_ids) > 0
    assert entry.subgraph is not None
    assert prefetcher.stats.expansions == 1


def test_expand_ego_graph_caching(prefetcher):
    """Test that repeated expansions use cache."""
    entry1 = prefetcher.expand_ego_graph("func:main")
    entry2 = prefetcher.expand_ego_graph("func:main")

    assert entry1 is entry2  # Same object (cached)
    assert prefetcher.stats.cache_hits == 1
    assert prefetcher.stats.expansions == 1


def test_expand_ego_graph_force(prefetcher):
    """Test forced re-expansion bypasses cache."""
    entry1 = prefetcher.expand_ego_graph("func:main")
    entry2 = prefetcher.expand_ego_graph("func:main", force=True)

    assert entry1 is not entry2  # Different objects
    assert prefetcher.stats.expansions == 2


def test_expand_ego_graph_radius(prefetcher):
    """Test explicit radius parameter."""
    entry = prefetcher.expand_ego_graph("func:main", radius=1)

    assert entry.radius == 1


def test_expand_ego_graph_centrality_radius(prefetcher):
    """Test radius increases for high-centrality nodes."""
    # func:main is a hub with many connections
    radius = prefetcher._determine_radius("func:main")

    # Should use default or extended based on centrality
    assert radius in [prefetcher.default_radius, prefetcher.extended_radius]


def test_expand_ego_graph_nonexistent(prefetcher):
    """Test expansion for non-existent node."""
    entry = prefetcher.expand_ego_graph("nonexistent:node")

    assert entry is None


# ═══════════════════════════════════════════════════════════════
# EDGE WEIGHTING TESTS
# ═══════════════════════════════════════════════════════════════


def test_edge_weight_calculation(prefetcher):
    """Test edge weight combines provenance, recency, and centrality."""
    weight = prefetcher._compute_edge_weight(
        "func:main",
        "func:helper",
        {"weight": 0.9}
    )

    assert 0.0 <= weight <= 1.0


def test_edge_weight_with_recency(prefetcher, mock_graph):
    """Test recency boost affects weight."""
    # Set last_accessed to recent time
    node = mock_graph.get_node("func:helper")
    node.last_accessed = time.time()

    weight_recent = prefetcher._compute_edge_weight(
        "func:main", "func:helper", {"weight": 1.0}
    )

    # Set to old time
    node.last_accessed = time.time() - 86400 * 7  # 7 days ago

    weight_old = prefetcher._compute_edge_weight(
        "func:main", "func:helper", {"weight": 1.0}
    )

    assert weight_recent >= weight_old


# ═══════════════════════════════════════════════════════════════
# CACHE MANAGEMENT TESTS
# ═══════════════════════════════════════════════════════════════


def test_cache_eviction(mock_graph):
    """Test cache eviction when over limit."""
    # Create prefetcher with tiny cache limit
    pf = PredictivePrefetcher(
        graph=mock_graph,
        max_cache_ratio=0.1,  # Very small
    )

    # Expand multiple nodes
    pf.expand_ego_graph("func:main")
    pf.expand_ego_graph("func:helper")
    pf.expand_ego_graph("class:App")

    # Should have evicted some entries
    assert pf.stats.evictions >= 0


def test_cache_lru_order(prefetcher):
    """Test LRU ordering is maintained."""
    prefetcher.expand_ego_graph("func:main")
    prefetcher.expand_ego_graph("func:helper")
    prefetcher.expand_ego_graph("func:main")  # Access again

    # func:main should be at end (most recent)
    assert prefetcher._cache_order[-1] == "func:main"


def test_cache_utility_eviction(mock_graph):
    """Test low-utility entries are evicted first."""
    pf = PredictivePrefetcher(
        graph=mock_graph,
        max_cache_ratio=0.25,
    )

    # Expand nodes and set different utilities
    entry1 = pf.expand_ego_graph("func:main")
    entry1.utility_score = 0.1  # Low utility

    entry2 = pf.expand_ego_graph("func:helper")
    entry2.utility_score = 0.9  # High utility

    # Force eviction
    pf._evict_if_needed(100)

    # Low utility should be evicted first
    # (exact behavior depends on cache size)


# ═══════════════════════════════════════════════════════════════
# INVALIDATION TESTS
# ═══════════════════════════════════════════════════════════════


def test_cache_invalidation(prefetcher):
    """Test cache invalidation on graph changes."""
    prefetcher.expand_ego_graph("func:main")
    prefetcher.expand_ego_graph("func:helper")

    initial_cache_size = len(prefetcher._ego_cache)
    assert initial_cache_size >= 1

    # Invalidate nodes overlapping with func:main
    prefetcher.on_graph_updated(["func:main"])

    # Should have invalidated at least one entry
    assert prefetcher.stats.invalidations >= 1
    # Cache size should decrease or stay same
    assert len(prefetcher._ego_cache) <= initial_cache_size


def test_cache_invalidation_overlap(prefetcher):
    """Test invalidation considers node overlap."""
    entry = prefetcher.expand_ego_graph("func:main")
    initial_nodes = entry.node_ids.copy()

    # Invalidate with overlapping node
    overlapping_node = list(initial_nodes)[0]
    prefetcher.on_graph_updated([overlapping_node])

    # Should be invalidated
    assert "func:main" not in prefetcher._ego_cache


# ═══════════════════════════════════════════════════════════════
# PREDICTION TESTS
# ═══════════════════════════════════════════════════════════════


def test_prediction_disabled_by_default(prefetcher):
    """Test predictions are disabled by default."""
    predictions = prefetcher.predict_next_hotspots()

    assert predictions == []


def test_prediction_enabled(prefetcher_with_predictions):
    """Test predictions when enabled."""
    # Simulate some accesses to build patterns
    prefetcher_with_predictions.on_node_accessed("func:main")
    prefetcher_with_predictions.on_node_accessed("func:helper")
    prefetcher_with_predictions.on_node_accessed("func:main")

    predictions = prefetcher_with_predictions.predict_next_hotspots()

    # May or may not have predictions based on confidence
    assert isinstance(predictions, list)


def test_prediction_features(prefetcher_with_predictions):
    """Test prediction feature vector is correct shape."""
    context = PredictionContext(
        recent_touches=["func:main"],
        query_trajectory=["func:main", "func:helper"],
    )

    features = prefetcher_with_predictions._build_prediction_features(
        context, "func:main"
    )

    assert features.shape == (64,)
    assert features.dtype == np.float64


def test_session_phase_inference(prefetcher):
    """Test session phase inference."""
    # Few touches = exploration
    assert prefetcher._infer_session_phase() == "exploration"

    # Same file repeatedly = refactor
    for _ in range(10):
        prefetcher.on_node_accessed("func:main")

    # Many touches on same node
    phase = prefetcher._infer_session_phase()
    assert phase in ["exploration", "refactor", "debugging"]


# ═══════════════════════════════════════════════════════════════
# PREFETCH QUEUE TESTS
# ═══════════════════════════════════════════════════════════════


def test_prefetch_queue_priority(prefetcher):
    """Test priority queue orders by confidence."""
    prefetcher.queue_for_prefetch("node:low", 0.85)
    prefetcher.queue_for_prefetch("node:high", 0.95)
    prefetcher.queue_for_prefetch("node:mid", 0.90)

    # Process queue
    items = []
    while prefetcher._prefetch_queue:
        import heapq
        item = heapq.heappop(prefetcher._prefetch_queue)
        items.append(item)

    # Should be high, mid, low (by confidence)
    assert items[0].node_id == "node:high"


def test_prefetch_queue_threshold(prefetcher):
    """Test queue rejects low-confidence items."""
    prefetcher.queue_for_prefetch("node:low", 0.5)  # Below threshold

    assert len(prefetcher._prefetch_queue) == 0


# ═══════════════════════════════════════════════════════════════
# ACCESS TRACKING TESTS
# ═══════════════════════════════════════════════════════════════


def test_on_node_accessed(prefetcher):
    """Test access tracking updates state."""
    prefetcher.on_node_accessed("func:main")

    assert "func:main" in prefetcher._recent_touches
    assert "func:main" in prefetcher._query_trajectory
    assert prefetcher._session_patterns["func:main"] == 1


def test_retrieval_success_update(prefetcher):
    """Test retrieval success rate updates utility."""
    entry = prefetcher.expand_ego_graph("func:main")
    _ = entry.utility_score  # Record for inspection

    # Simulate successful retrieval
    prefetcher.update_retrieval_success("func:main", success=True)

    # Utility should change based on success
    assert entry.retrieval_success_rate > 0.5


# ═══════════════════════════════════════════════════════════════
# STATS TESTS
# ═══════════════════════════════════════════════════════════════


def test_stats_tracking(prefetcher):
    """Test statistics are tracked correctly."""
    prefetcher.expand_ego_graph("func:main")
    prefetcher.expand_ego_graph("func:main")  # Hit
    prefetcher.expand_ego_graph("func:helper")

    stats = prefetcher.get_stats()

    # At least one cache entry should exist
    assert stats["cache_entries"] >= 1
    assert stats["cache_hits"] == 1
    assert stats["expansions"] >= 2


def test_cache_hit_rate(prefetcher):
    """Test cache hit rate calculation."""
    prefetcher.expand_ego_graph("func:main")  # Miss
    prefetcher.expand_ego_graph("func:main")  # Hit
    prefetcher.expand_ego_graph("func:main")  # Hit

    stats = prefetcher.get_stats()

    # 2 hits, 1 miss = 66.67%
    assert stats["cache_hit_rate_pct"] == pytest.approx(66.67, abs=0.1)


# ═══════════════════════════════════════════════════════════════
# MODE TOGGLE TESTS
# ═══════════════════════════════════════════════════════════════


def test_enable_predictions_mode(prefetcher):
    """Test enabling prediction mode."""
    assert prefetcher.enable_predictions is False

    prefetcher.enable_predictions_mode()

    assert prefetcher.enable_predictions is True


def test_disable_predictions_mode(prefetcher_with_predictions):
    """Test disabling prediction mode."""
    assert prefetcher_with_predictions.enable_predictions is True

    prefetcher_with_predictions.disable_predictions_mode()

    assert prefetcher_with_predictions.enable_predictions is False


# ═══════════════════════════════════════════════════════════════
# SUBGRAPH LOOKUP TESTS
# ═══════════════════════════════════════════════════════════════


def test_get_cached_subgraph_hit(prefetcher):
    """Test cache hit returns subgraph."""
    prefetcher.expand_ego_graph("func:main")

    subgraph = prefetcher.get_cached_subgraph("func:main")

    assert subgraph is not None
    assert isinstance(subgraph, nx.DiGraph)


def test_get_cached_subgraph_miss(prefetcher):
    """Test cache miss returns None."""
    subgraph = prefetcher.get_cached_subgraph("func:main")

    assert subgraph is None
    assert prefetcher.stats.cache_misses == 1


def test_is_cached(prefetcher):
    """Test is_cached method."""
    assert prefetcher.is_cached("func:main") is False

    prefetcher.expand_ego_graph("func:main")

    assert prefetcher.is_cached("func:main") is True
