"""
Unit Tests for USMA Phase 3: Evolutionary Oversight
=====================================================

Tests for SovereignOptimizer, replay buffer, and graph weight evolution.
"""

import numpy as np

from saga.core.memory import (
    EdgeType,
    FeedbackEvent,
    GraphEdge,
    GraphNode,
    NodeType,
    RepoGraph,
    SovereignOptimizer,
)


class TestSovereignOptimizer:
    """Tests for SovereignOptimizer MLP."""

    def test_optimizer_initialization(self):
        """Test optimizer initializes with correct dimensions."""
        optimizer = SovereignOptimizer(feature_dim=32, hidden_dim=16)

        assert optimizer.W1.shape == (32, 16)
        assert optimizer.W2.shape == (16, 1)
        assert optimizer.training_count == 0

    def test_predict_utility(self):
        """Test utility prediction returns valid score."""
        optimizer = SovereignOptimizer(feature_dim=16, hidden_dim=8)

        context = np.random.randn(16)
        score = optimizer.predict_utility(context)

        assert 0.0 <= score <= 1.0

    def test_predict_utility_padding(self):
        """Test prediction handles smaller vectors via padding."""
        optimizer = SovereignOptimizer(feature_dim=32, hidden_dim=16)

        # Smaller vector
        context = np.random.randn(10)
        score = optimizer.predict_utility(context)

        assert 0.0 <= score <= 1.0

    def test_sigmoid_activation(self):
        """Test sigmoid function returns correct values."""
        assert abs(SovereignOptimizer.sigmoid(np.array([0.0]))[0] - 0.5) < 0.01
        assert SovereignOptimizer.sigmoid(np.array([10.0]))[0] > 0.99
        assert SovereignOptimizer.sigmoid(np.array([-10.0]))[0] < 0.01


class TestFeedbackAndTraining:
    """Tests for feedback recording and batch training."""

    def test_record_feedback(self, tmp_path):
        """Test recording feedback events."""
        buffer_path = tmp_path / "buffer.json"
        optimizer = SovereignOptimizer(
            feature_dim=8,
            hidden_dim=4,
            batch_size=5,
            buffer_path=buffer_path
        )

        optimizer.record_feedback(
            task_id="task_1",
            context_vector=[0.1] * 8,
            retrieval_path=["node_a", "node_b"],
            confidence=0.9,
            success=True
        )

        assert len(optimizer.replay_buffer) == 1
        assert optimizer.replay_buffer[0].reward == 1.0

    def test_batch_training_triggers(self, tmp_path):
        """Test that training triggers when buffer is full."""
        optimizer = SovereignOptimizer(
            feature_dim=4,
            hidden_dim=2,
            batch_size=3,
            learning_rate=0.1
        )

        initial_W1 = optimizer.W1.copy()

        # Record enough feedback to trigger training
        for i in range(3):
            optimizer.record_feedback(
                task_id=f"task_{i}",
                context_vector=list(np.random.randn(4)),
                retrieval_path=["a", "b"],
                confidence=0.8,
                success=i % 2 == 0
            )

        # Buffer should be empty after training
        assert len(optimizer.replay_buffer) == 0
        assert optimizer.training_count == 1
        # Weights should have changed
        assert not np.allclose(optimizer.W1, initial_W1)


class TestThreeSigmaValidation:
    """Tests for 3σ weight change validation."""

    def test_validation_accepts_normal_changes(self):
        """Test that normal weight changes are accepted."""
        optimizer = SovereignOptimizer(feature_dim=4, hidden_dim=2)

        # Add some history
        optimizer.weight_change_history = [0.1, 0.12, 0.11, 0.09, 0.13]

        # Normal change should pass
        assert optimizer._validate_weight_change(0.15) == True

    def test_validation_flags_anomalous_changes(self):
        """Test that anomalous changes are flagged."""
        optimizer = SovereignOptimizer(feature_dim=4, hidden_dim=2)

        # Small, consistent history
        optimizer.weight_change_history = [0.1, 0.11, 0.1, 0.09, 0.1]

        # Huge change should fail
        assert optimizer._validate_weight_change(10.0) == False


class TestGraphWeightEvolution:
    """Tests for graph edge weight updates."""

    def test_update_graph_weights_success(self):
        """Test weight updates for successful retrieval."""
        optimizer = SovereignOptimizer(learning_rate=0.1)
        graph = RepoGraph()

        graph.add_node(GraphNode("a", NodeType.FUNCTION, "a"))
        graph.add_node(GraphNode("b", NodeType.FUNCTION, "b"))
        graph.add_edge(GraphEdge("a", "b", EdgeType.CALLS, weight=1.0))

        initial_weight = graph.graph["a"]["b"]["weight"]

        optimizer.update_graph_weights(
            graph=graph,
            retrieval_path=["a", "b"],
            reward=1.0  # Success
        )

        new_weight = graph.graph["a"]["b"]["weight"]
        assert new_weight > initial_weight * 0.9  # Increased after L2

    def test_weight_decay_application(self):
        """Test weight decay is applied to all edges."""
        optimizer = SovereignOptimizer(weight_decay=0.9)
        graph = RepoGraph()

        graph.add_node(GraphNode("a", NodeType.FUNCTION, "a"))
        graph.add_node(GraphNode("b", NodeType.FUNCTION, "b"))
        graph.add_edge(GraphEdge("a", "b", EdgeType.CALLS, weight=1.0))

        optimizer.apply_weight_decay_to_graph(graph)

        assert graph.graph["a"]["b"]["weight"] == 0.9


class TestSelfAudit:
    """Tests for periodic self-audit functionality."""

    def test_audit_finds_stale_edges(self):
        """Test audit identifies low-weight edges."""
        optimizer = SovereignOptimizer()
        graph = RepoGraph()

        graph.add_node(GraphNode("a", NodeType.FUNCTION, "a"))
        graph.add_node(GraphNode("b", NodeType.FUNCTION, "b"))
        graph.add_node(GraphNode("c", NodeType.FUNCTION, "c"))

        graph.add_edge(GraphEdge("a", "b", EdgeType.CALLS, weight=0.2))  # Stale
        graph.add_edge(GraphEdge("b", "c", EdgeType.CALLS, weight=0.8))  # Active

        result = optimizer.periodic_self_audit(graph, stale_threshold=0.3)

        assert len(result["stale_edges"]) == 1
        assert result["stale_edges"][0][0] == "a"

    def test_soft_prune_to_cold_storage(self):
        """Test edges are moved to cold storage, not deleted."""
        optimizer = SovereignOptimizer()
        graph = RepoGraph()

        graph.add_node(GraphNode("a", NodeType.FUNCTION, "a"))
        graph.add_node(GraphNode("b", NodeType.FUNCTION, "b"))
        graph.add_edge(GraphEdge("a", "b", EdgeType.CALLS))

        optimizer.soft_prune_to_cold_storage(graph, [("a", "b")])

        assert len(optimizer.cold_storage) == 1
        assert not graph.graph.has_edge("a", "b")


class TestPersistence:
    """Tests for state save/load with ECC hashing."""

    def test_save_and_load_state(self, tmp_path):
        """Test optimizer state persistence."""
        state_path = tmp_path / "optimizer_state.json"

        # Create and train
        optimizer1 = SovereignOptimizer(
            feature_dim=4,
            hidden_dim=2,
            state_path=state_path
        )
        optimizer1.training_count = 5
        optimizer1.W1[0, 0] = 0.12345
        optimizer1._save_state()

        # Load into new instance
        optimizer2 = SovereignOptimizer(
            feature_dim=4,
            hidden_dim=2,
            state_path=state_path
        )
        success = optimizer2.load_state()

        assert success
        assert optimizer2.training_count == 5
        assert abs(optimizer2.W1[0, 0] - 0.12345) < 1e-6

    def test_hash_integrity_check(self, tmp_path):
        """Test that corrupted state is detected."""
        state_path = tmp_path / "optimizer_state.json"

        optimizer = SovereignOptimizer(state_path=state_path)
        optimizer._save_state()

        # Corrupt the file
        import json
        data = json.loads(state_path.read_text())
        data["W1"][0][0] = 999.0  # Corrupt weight
        # Don't update hash
        state_path.write_text(json.dumps(data))

        # Load should detect corruption
        optimizer2 = SovereignOptimizer(state_path=state_path)
        success = optimizer2.load_state()

        assert success is False  # Corruption detected


class TestFeedbackEvent:
    """Tests for FeedbackEvent dataclass."""

    def test_event_serialization(self):
        """Test event to_dict and from_dict."""
        event = FeedbackEvent(
            event_id="test_1",
            task_id="task_1",
            context_vector=[0.1, 0.2, 0.3],
            retrieval_path=["a", "b"],
            confidence=0.9,
            reward=1.0
        )

        data = event.to_dict()
        restored = FeedbackEvent.from_dict(data)

        assert restored.event_id == event.event_id
        assert restored.reward == event.reward
