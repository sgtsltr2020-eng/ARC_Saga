"""
Unit Tests for Hybrid Scorer
============================

Tests for semantic-structural-provenance scoring.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

import numpy as np
import pytest

from saga.core.memory import (
    EdgeType,
    GraphEdge,
    GraphNode,
    HybridScorer,
    NodeType,
    RepoGraph,
    ScoreComponents,
    ScorerConfig,
)

# ═══════════════════════════════════════════════════════════════
# MOCK OBJECTS
# ═══════════════════════════════════════════════════════════════

@dataclass
class MockLoreEntry:
    """Mock LoreEntry for testing."""
    entry_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    codex_status: str = "NEUTRAL"
    related_entities: list[str] = field(default_factory=list)


class MockLoreBook:
    """Mock LoreBook for testing."""

    def __init__(self):
        self.lore_entries: list[MockLoreEntry] = []

    def add_entry(self, entry: MockLoreEntry) -> None:
        self.lore_entries.append(entry)


# ═══════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════

class TestScorerConfig:
    """Tests for ScorerConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ScorerConfig()

        assert config.semantic_weight == 0.5
        assert config.structural_weight == 0.3
        assert config.provenance_weight == 0.2
        assert config.validate() is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ScorerConfig(
            semantic_weight=0.4,
            structural_weight=0.4,
            provenance_weight=0.2
        )

        assert config.validate() is True

    def test_invalid_weights(self):
        """Test that invalid weights are detected."""
        config = ScorerConfig(
            semantic_weight=0.5,
            structural_weight=0.5,
            provenance_weight=0.5
        )

        assert config.validate() is False


class TestScoreComponents:
    """Tests for ScoreComponents."""

    def test_create_components(self):
        """Test creating score components."""
        comp = ScoreComponents(
            node_id="func:test",
            semantic_raw=0.8,
            semantic_normalized=0.9
        )

        assert comp.node_id == "func:test"
        assert comp.semantic_raw == 0.8

    def test_to_dict(self):
        """Test serialization."""
        comp = ScoreComponents(
            node_id="node_1",
            semantic_normalized=0.8,
            structural_normalized=0.5,
            provenance_normalized=0.3,
            combined_score=0.6
        )

        data = comp.to_dict()

        assert data["node_id"] == "node_1"
        assert data["semantic"] == 0.8
        assert data["combined"] == 0.6


class TestHybridScorer:
    """Tests for HybridScorer."""

    @pytest.fixture
    def graph(self):
        """Create a test graph with connected nodes."""
        g = RepoGraph()

        # Create a chain: n0 -> n1 -> n2 -> n3
        for i in range(4):
            node = GraphNode(
                node_id=f"func:n{i}",
                node_type=NodeType.FUNCTION,
                name=f"func_{i}",
                embedding_vector=np.random.randn(384).astype(np.float32)
            )
            g.add_node(node)

        # Add edges to create path distances
        for i in range(3):
            edge = GraphEdge(
                source_id=f"func:n{i}",
                target_id=f"func:n{i+1}",
                edge_type=EdgeType.CALLS
            )
            g.add_edge(edge)

        return g

    @pytest.fixture
    def lorebook(self):
        """Create a test lorebook."""
        lb = MockLoreBook()

        # Add entries with various statuses
        lb.add_entry(MockLoreEntry(entry_id="e_good", codex_status="COMPLIANT"))
        lb.add_entry(MockLoreEntry(entry_id="e_bad", codex_status="VIOLATION"))
        lb.add_entry(MockLoreEntry(entry_id="e_neutral", codex_status="NEUTRAL"))

        return lb

    @pytest.fixture
    def scorer(self, graph, lorebook):
        """Create scorer with graph and lorebook."""
        return HybridScorer(graph=graph, lorebook=lorebook)

    def test_basic_score(self, scorer):
        """Test basic scoring."""
        components = scorer.score("func:n0", semantic_similarity=0.8)

        assert components.node_id == "func:n0"
        assert components.semantic_normalized > 0
        assert 0 <= components.combined_score <= 1

    def test_semantic_normalization(self, scorer):
        """Test cosine normalization to [0,1]."""
        # Raw cosine of 1.0 should normalize to 1.0
        result = scorer._normalize_cosine(1.0)
        assert result == 1.0

        # Raw cosine of 0.0 should normalize to 0.5
        result = scorer._normalize_cosine(0.0)
        assert result == 0.5

        # Raw cosine of -1.0 should normalize to 0.0
        result = scorer._normalize_cosine(-1.0)
        assert result == 0.0

    def test_structural_scoring(self, graph, scorer):
        """Test structural scoring with path distance."""
        # Score n3 relative to n0 (distance = 3)
        components = scorer.score(
            "func:n3",
            reference_node_id="func:n0",
            semantic_similarity=0.5
        )

        assert components.path_distance == 3
        assert components.structural_normalized > 0

    def test_structural_no_path(self, graph, scorer):
        """Test structural scoring when no path exists."""
        # Add isolated node
        isolated = GraphNode("func:isolated", NodeType.FUNCTION, "isolated")
        graph.add_node(isolated)

        components = scorer.score(
            "func:isolated",
            reference_node_id="func:n0",
            semantic_similarity=0.5
        )

        assert components.path_distance == -1

    def test_provenance_scoring(self, graph, lorebook, scorer):
        """Test provenance scoring from linked entries."""
        # Link node to good entry
        node = graph.get_node("func:n0")
        node.lore_entry_ids = ["e_good"]

        components = scorer.score("func:n0", semantic_similarity=0.5)

        assert components.provenance_normalized == 1.0  # COMPLIANT = 1.0

    def test_provenance_mixed(self, graph, lorebook, scorer):
        """Test provenance with mixed entry statuses."""
        node = graph.get_node("func:n0")
        node.lore_entry_ids = ["e_good", "e_bad"]

        components = scorer.score("func:n0", semantic_similarity=0.5)

        # Average of 1.0 and 0.0
        assert components.provenance_normalized == 0.5

    def test_batch_scoring(self, scorer):
        """Test batch scoring."""
        results = scorer.score_batch(
            node_ids=["func:n0", "func:n1", "func:n2"],
            semantic_similarities={
                "func:n0": 0.9,
                "func:n1": 0.5,
                "func:n2": 0.3
            }
        )

        assert len(results) == 3
        # Should be sorted by combined score
        assert results[0].semantic_normalized >= results[1].semantic_normalized


class TestWeightLearning:
    """Tests for weight learning hooks."""

    def test_record_feedback(self):
        """Test recording feedback."""
        scorer = HybridScorer()

        components = ScoreComponents(
            node_id="func:test",
            semantic_normalized=0.8,
            structural_normalized=0.5,
            provenance_normalized=0.3,
            combined_score=0.6
        )

        contributions = scorer.record_feedback(
            node_id="func:test",
            components=components,
            utility=1.0
        )

        assert "semantic_contribution" in contributions
        assert contributions["observed_utility"] == 1.0

    def test_weight_recommendations(self):
        """Test weight recommendation from feedback."""
        scorer = HybridScorer()

        # Simulate feedback where semantic always predicts success
        feedback = [
            {"semantic_contribution": 0.8, "structural_contribution": 0.2,
             "provenance_contribution": 0.1, "observed_utility": 1.0}
            for _ in range(10)
        ]

        recommended = scorer.get_weight_recommendations(feedback)

        # Semantic should get highest recommended weight
        assert recommended.semantic_weight > recommended.structural_weight


class TestCaching:
    """Tests for cache management."""

    @pytest.fixture
    def scorer_with_graph(self):
        """Create scorer with small graph."""
        graph = RepoGraph()
        for i in range(5):
            graph.add_node(GraphNode(f"n{i}", NodeType.FUNCTION, f"func{i}"))

        return HybridScorer(graph=graph)

    def test_centrality_caching(self, scorer_with_graph):
        """Test that centrality is computed and cached."""
        scorer = scorer_with_graph

        # First access triggers computation
        centality = scorer._get_centrality("n0")
        assert scorer._centrality_computed is True

        # Second access uses cache
        centality2 = scorer._get_centrality("n0")
        assert centality == centality2

    def test_invalidate_centrality(self, scorer_with_graph):
        """Test centrality invalidation."""
        scorer = scorer_with_graph

        scorer._get_centrality("n0")
        assert len(scorer._centrality_cache) > 0

        scorer.invalidate_centrality()
        assert len(scorer._centrality_cache) == 0
        assert scorer._centrality_computed is False

    def test_provenance_caching(self, scorer_with_graph):
        """Test provenance caching."""
        scorer = scorer_with_graph

        scorer._compute_provenance("n0")
        assert "n0" in scorer._provenance_cache

        scorer.invalidate_provenance("n0")
        assert "n0" in scorer._provenance_dirty


class TestPerformance:
    """Performance tests."""

    def test_5k_node_scoring(self):
        """Test performance on 5000+ nodes."""
        graph = RepoGraph()

        # Create 5000 nodes
        for i in range(5000):
            node = GraphNode(
                node_id=f"n{i}",
                node_type=NodeType.FUNCTION,
                name=f"func_{i}",
                embedding_vector=np.random.randn(384).astype(np.float32)
            )
            graph.add_node(node)

        scorer = HybridScorer(graph=graph)

        # Pre-compute centrality
        scorer._compute_centrality()

        # Time batch scoring
        start = time.time()

        node_ids = [f"n{i}" for i in range(1000)]
        similarities = {f"n{i}": np.random.random() for i in range(1000)}

        results = scorer.score_batch(
            node_ids=node_ids,
            semantic_similarities=similarities
        )

        elapsed = time.time() - start

        assert len(results) == 1000
        assert elapsed < 2.0  # Should be fast

    def test_stats(self):
        """Test statistics reporting."""
        scorer = HybridScorer()

        stats = scorer.get_stats()

        assert "scores_computed" in stats
        assert "config" in stats
        assert stats["config"]["semantic_weight"] == 0.5


class TestPersistence:
    """Tests for scorer persistence."""

    def test_serialization_roundtrip(self):
        """Test save/load cycle."""
        graph = RepoGraph()
        for i in range(3):
            graph.add_node(GraphNode(f"n{i}", NodeType.FUNCTION, f"f{i}"))

        scorer = HybridScorer(graph=graph)

        # Populate caches
        scorer._compute_centrality()
        scorer._compute_provenance("n0")
        scorer.scores_computed = 42

        # Serialize
        data = scorer.to_dict()

        # Deserialize
        restored = HybridScorer.from_dict(data, graph=graph)

        assert restored._centrality_computed is True
        assert restored.scores_computed == 42


class TestEdgeCases:
    """Tests for edge cases."""

    def test_no_graph(self):
        """Test scoring without graph."""
        scorer = HybridScorer()

        components = scorer.score("unknown", semantic_similarity=0.5)

        assert components.semantic_normalized > 0
        assert components.structural_normalized == 0
        assert components.provenance_normalized == 0

    def test_no_embedding(self):
        """Test node without embedding."""
        graph = RepoGraph()
        graph.add_node(GraphNode("n1", NodeType.FUNCTION, "func1"))

        scorer = HybridScorer(graph=graph)
        query_emb = np.random.randn(384).astype(np.float32)

        components = scorer.score("n1", query_embedding=query_emb)

        assert components.semantic_raw == 0.0

    def test_isolated_node(self):
        """Test isolated node with no edges."""
        graph = RepoGraph()
        graph.add_node(GraphNode("isolated", NodeType.FUNCTION, "isolated"))

        scorer = HybridScorer(graph=graph)

        components = scorer.score(
            "isolated",
            reference_node_id="nonexistent",
            semantic_similarity=0.7
        )

        assert components.path_distance == -1
        assert components.combined_score > 0  # Still has semantic
