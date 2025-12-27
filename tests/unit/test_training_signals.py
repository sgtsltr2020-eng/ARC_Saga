"""
Unit Tests for Training Signals
================================

Tests for feature extraction, utility calculation, and normalization.
"""


import numpy as np
import pytest

from saga.core.memory import (
    FeatureNormalizer,
    RetrievalFeatureExtractor,
    RetrievalFeatures,
    SearchResult,
    SovereignOptimizer,
    TrainingSignalManager,
    UtilityCalculator,
)


class TestRetrievalFeatures:
    """Tests for RetrievalFeatures dataclass."""

    def test_default_values(self):
        """Test default feature values."""
        features = RetrievalFeatures()

        assert features.retrieval_confidence == 0.0
        assert features.user_feedback == 0.5  # Neutral default
        assert features.simulation_success == 0.5  # Neutral default

    def test_to_vector(self):
        """Test conversion to numpy array."""
        features = RetrievalFeatures(
            retrieval_confidence=0.8,
            graph_distance=0.5,
            node_centrality=0.3
        )

        vec = features.to_vector()

        assert isinstance(vec, np.ndarray)
        assert len(vec) == 9
        assert vec[0] == 0.8  # retrieval_confidence
        assert vec[1] == 0.5  # graph_distance

    def test_from_vector(self):
        """Test creation from numpy array."""
        vec = np.array([0.9, 0.4, 0.3, 0.2, 0.1, 0.8, 1.0, 0.5, 0.7])

        features = RetrievalFeatures.from_vector(vec)

        assert features.retrieval_confidence == pytest.approx(0.9)
        assert features.was_used == pytest.approx(1.0)

    def test_feature_dim(self):
        """Test feature dimension constant."""
        assert RetrievalFeatures.feature_dim() == 9


class TestRetrievalFeatureExtractor:
    """Tests for feature extraction from search results."""

    @pytest.fixture
    def extractor(self):
        return RetrievalFeatureExtractor()

    @pytest.fixture
    def mock_results(self):
        """Create mock search results."""
        return [
            SearchResult(
                node_id="func:handler",
                name="async_handler",
                node_type="FUNCTION",
                file_path="api/handler.py",
                cosine_similarity=0.85,
                graph_distance=2,
                recency_score=0.9
            ),
            SearchResult(
                node_id="func:process",
                name="process_request",
                node_type="FUNCTION",
                file_path="api/processor.py",
                cosine_similarity=0.72,
                graph_distance=3,
                recency_score=0.5
            ),
        ]

    def test_extract_retrieval_confidence(self, extractor, mock_results):
        """Test retrieval confidence extraction."""
        features = extractor.extract_from_search("async request", mock_results)

        assert features.retrieval_confidence == 0.85  # Top result

    def test_extract_graph_distance(self, extractor, mock_results):
        """Test graph distance normalization."""
        features = extractor.extract_from_search("async request", mock_results)

        # Avg distance = (2+3)/2 = 2.5, normalized = 1 - 2.5/10 = 0.75
        assert features.graph_distance == pytest.approx(0.75)

    def test_extract_recency_score(self, extractor, mock_results):
        """Test recency score averaging."""
        features = extractor.extract_from_search("async request", mock_results)

        # Avg recency = (0.9 + 0.5) / 2 = 0.7
        assert features.recency_score == pytest.approx(0.7)

    def test_extract_empty_results(self, extractor):
        """Test extraction with no results."""
        features = extractor.extract_from_search("query", [])

        assert features.retrieval_confidence == 0.0

    def test_add_outcome_signals(self, extractor, mock_results):
        """Test adding outcome signals."""
        features = extractor.extract_from_search("async", mock_results)

        extractor.add_outcome_signals(
            features,
            user_feedback=1,  # Thumbs up
            was_used=True,
            token_savings=500,
            simulation_passed=True
        )

        assert features.user_feedback == 1.0  # +1 → 1.0
        assert features.was_used == 1.0
        assert features.token_savings == 0.25  # 500/2000
        assert features.simulation_success == 1.0

    def test_user_feedback_normalization(self, extractor, mock_results):
        """Test user feedback normalization."""
        features = RetrievalFeatures()

        # Test -1 → 0
        extractor.add_outcome_signals(features, user_feedback=-1)
        assert features.user_feedback == 0.0

        # Test 0 → 0.5
        features = RetrievalFeatures()
        extractor.add_outcome_signals(features, user_feedback=0)
        assert features.user_feedback == 0.5


class TestUtilityCalculator:
    """Tests for utility calculation."""

    @pytest.fixture
    def calculator(self):
        return UtilityCalculator()

    def test_high_utility_from_positive_feedback(self, calculator):
        """Test that positive feedback leads to high utility."""
        features = RetrievalFeatures(
            retrieval_confidence=0.8,
            user_feedback=1.0,  # Explicit positive
            simulation_success=0.9,
            token_savings=0.5
        )

        utility = calculator.compute_utility(features)

        assert utility > 0.7  # Should be high

    def test_low_utility_from_negative_feedback(self, calculator):
        """Test that negative feedback leads to low utility."""
        features = RetrievalFeatures(
            retrieval_confidence=0.8,
            user_feedback=0.0,  # Explicit negative
            simulation_success=0.2,
            token_savings=0.1
        )

        utility = calculator.compute_utility(features)

        assert utility < 0.4  # Should be low

    def test_neutral_without_feedback(self, calculator):
        """Test utility without explicit feedback."""
        features = RetrievalFeatures(
            retrieval_confidence=0.7,
            user_feedback=0.5,  # No feedback
            simulation_success=0.5,  # No simulation
            was_used=1.0  # Proxy signal
        )

        utility = calculator.compute_utility(features)

        # Should use proxy signals
        assert 0.4 < utility < 0.7

    def test_batch_utility(self, calculator):
        """Test batch utility calculation."""
        features_list = [
            RetrievalFeatures(retrieval_confidence=0.9, user_feedback=1.0),
            RetrievalFeatures(retrieval_confidence=0.5, user_feedback=0.0),
        ]

        utilities = calculator.compute_batch_utility(features_list)

        assert len(utilities) == 2
        assert utilities[0] > utilities[1]  # First should be higher


class TestFeatureNormalizer:
    """Tests for feature normalization."""

    def test_initial_state(self):
        """Test initial normalizer state."""
        normalizer = FeatureNormalizer(feature_dim=9)

        assert normalizer.sample_count == 0
        assert len(normalizer.feature_min) == 9

    def test_update_statistics(self):
        """Test updating running statistics."""
        normalizer = FeatureNormalizer(feature_dim=3)

        batch = np.array([[0.1, 0.5, 0.3], [0.2, 0.8, 0.4]])
        normalizer.update(batch)

        assert normalizer.sample_count == 2
        assert normalizer.feature_min[0] == pytest.approx(0.1)
        assert normalizer.feature_max[1] == pytest.approx(0.8)

    def test_normalize(self):
        """Test MinMax normalization."""
        normalizer = FeatureNormalizer(feature_dim=2)

        # Set known range
        normalizer.feature_min = np.array([0.0, 0.0])
        normalizer.feature_max = np.array([10.0, 20.0])

        features = np.array([[5.0, 10.0]])
        normalized = normalizer.normalize(features)

        assert normalized[0, 0] == pytest.approx(0.5)
        assert normalized[0, 1] == pytest.approx(0.5)

    def test_serialization(self):
        """Test serialization round-trip."""
        normalizer = FeatureNormalizer(feature_dim=9)
        normalizer.feature_min = np.array([0.1] * 9)
        normalizer.feature_max = np.array([0.9] * 9)
        normalizer.sample_count = 100

        # Serialize and deserialize
        data = normalizer.to_dict()
        restored = FeatureNormalizer.from_dict(data)

        assert restored.sample_count == 100
        assert np.allclose(restored.feature_min, normalizer.feature_min)


class TestTrainingSignalManager:
    """Tests for the full training signal pipeline."""

    @pytest.fixture
    def optimizer(self, tmp_path):
        """Create test optimizer."""
        return SovereignOptimizer(
            feature_dim=9,
            hidden_dim=16,
            batch_size=5,
            buffer_path=tmp_path / "buffer.json",
            state_path=tmp_path / "state.json"
        )

    @pytest.fixture
    def manager(self, optimizer):
        """Create training signal manager."""
        return TrainingSignalManager(
            optimizer=optimizer,
            batch_threshold=5
        )

    @pytest.fixture
    def mock_results(self):
        """Create mock search results."""
        return [
            SearchResult(
                node_id="func:a",
                name="test_function",
                node_type="FUNCTION",
                file_path="test.py",
                cosine_similarity=0.8,
                graph_distance=2,
                recency_score=0.5
            )
        ]

    def test_record_search(self, manager, mock_results):
        """Test recording a search event."""
        search_id = manager.record_search(
            search_id="search_1",
            query="test function",
            results=mock_results
        )

        assert search_id == "search_1"
        assert "search_1" in manager.pending_events

    def test_complete_event(self, manager, mock_results):
        """Test completing an event."""
        manager.record_search("search_2", "test", mock_results)

        event = manager.complete_event(
            search_id="search_2",
            user_feedback=1,
            was_used=True
        )

        assert event is not None
        assert event.utility > 0.5
        assert "search_2" not in manager.pending_events

    def test_batch_training_trigger(self, manager, mock_results):
        """Test that batch training triggers at threshold."""
        # Record and complete 5 events
        for i in range(5):
            manager.record_search(f"search_{i}", "test", mock_results)
            manager.complete_event(f"search_{i}", user_feedback=1)

        # Should have triggered training
        assert manager.train_calls >= 1

    def test_serialization(self, manager, mock_results):
        """Test serialization round-trip."""
        manager.record_search("search_x", "test", mock_results)
        manager.complete_event("search_x", user_feedback=1)

        # Serialize
        data = manager.to_dict()

        # Deserialize
        restored = TrainingSignalManager.from_dict(data)

        assert restored.total_events == 1
        assert restored.normalizer.sample_count == 1

    def test_get_stats(self, manager, mock_results):
        """Test stats reporting."""
        manager.record_search("search_stats", "test", mock_results)

        stats = manager.get_stats()

        assert stats["pending_searches"] == 1
        assert stats["total_events"] == 0


class TestMultiSessionTraining:
    """Tests for training across multiple sessions."""

    def test_weights_evolve_with_feedback(self, tmp_path):
        """Test that weights change in response to feedback."""
        # Create optimizer
        optimizer = SovereignOptimizer(
            feature_dim=9,
            hidden_dim=16,
            batch_size=5,
            state_path=tmp_path / "optimizer_state.json"
        )

        # Record initial weight norm
        initial_norm = np.sqrt(np.sum(optimizer.W1 ** 2) + np.sum(optimizer.W2 ** 2))

        # Create manager and run training
        manager = TrainingSignalManager(optimizer=optimizer, batch_threshold=5)

        mock_result = SearchResult(
            node_id="func:test",
            name="test",
            node_type="FUNCTION",
            file_path="test.py",
            cosine_similarity=0.9,
            graph_distance=1
        )

        # Generate consistent positive feedback
        for i in range(10):
            manager.record_search(f"s_{i}", "test", [mock_result])
            manager.complete_event(f"s_{i}", user_feedback=1, was_used=True)

        # Check weights changed
        final_norm = np.sqrt(np.sum(optimizer.W1 ** 2) + np.sum(optimizer.W2 ** 2))
        assert abs(final_norm - initial_norm) > 0.01  # Should have changed

    def test_persistence_across_sessions(self, tmp_path):
        """Test that training continues across sessions."""
        state_path = tmp_path / "session_state.json"
        buffer_path = tmp_path / "session_buffer.json"

        # Session 1: Train
        optimizer1 = SovereignOptimizer(
            feature_dim=9,
            hidden_dim=16,
            batch_size=5,
            state_path=state_path,
            buffer_path=buffer_path
        )

        manager1 = TrainingSignalManager(optimizer=optimizer1, batch_threshold=5)

        mock_result = SearchResult(
            node_id="func:persist",
            name="persist",
            node_type="FUNCTION",
            file_path="persist.py",
            cosine_similarity=0.8,
            graph_distance=2
        )

        for i in range(5):
            manager1.record_search(f"p_{i}", "persist", [mock_result])
            manager1.complete_event(f"p_{i}", user_feedback=1)

        training_count_1 = optimizer1.training_count

        # Session 2: Load and continue
        optimizer2 = SovereignOptimizer(
            feature_dim=9,
            hidden_dim=16,
            batch_size=5,
            state_path=state_path,
            buffer_path=buffer_path
        )
        optimizer2.load_state()

        # Should have loaded previous training count
        assert optimizer2.training_count >= training_count_1
