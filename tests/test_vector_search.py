"""
Tests for VectorSearch
======================

Author: ARC SAGA Development Team
Date: December 17, 2025
"""

import pytest

try:
    from saga.core.lorebook import Decision
    from saga.storage.vector_search import VectorSearch
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


@pytest.mark.skipif(not DEPS_AVAILABLE, reason="scikit-learn not installed")
class TestVectorSearchBasics:
    """Test basic vector search operations."""

    def test_initialization(self):
        """Test VectorSearch initializes."""
        vs = VectorSearch()
        assert vs.fitted is False
        assert len(vs.decisions) == 0

    def test_add_decision(self):
        """Test adding decision to index."""
        vs = VectorSearch()
        decision = Decision(
            question="Test question",
            mimiry_ruling="Test",
            lorebook_ruling="Test",
            context={}
        )

        vs.add_decision(decision)

        assert len(vs.decisions) == 1
        assert vs.fitted is True

    def test_find_similar_exact_match(self):
        """Test finding exact matching question."""
        vs = VectorSearch()
        decision = Decision(
            question="Should I use async Redis?",
            context={"framework": "FastAPI"},
            mimiry_ruling="Yes",
            lorebook_ruling="Yes"
        )
        vs.add_decision(decision)

        results = vs.find_similar(
            question="Should I use async Redis?",
            context={"framework": "FastAPI"}
        )

        assert len(results) > 0
        assert results[0][0].decision_id == decision.decision_id
        assert results[0][1] > 0.9  # High similarity

    def test_find_similar_returns_empty_when_no_matches(self):
        """Test no results for unrelated question."""
        vs = VectorSearch()
        decision = Decision(
            question="Use Redis?",
            context={},
            mimiry_ruling="Yes",
            lorebook_ruling="Yes"
        )
        vs.add_decision(decision)

        results = vs.find_similar(
            question="How to train neural networks?",
            context={},
            threshold=0.7
        )

        assert len(results) == 0

    def test_find_similar_respects_top_k(self):
        """Test top_k limit is respected."""
        vs = VectorSearch()

        # Add 10 similar decisions
        for i in range(10):
            decision = Decision(
                question=f"Redis question {i}",
                context={"framework": "FastAPI"},
                mimiry_ruling="Test",
                lorebook_ruling="Test",
                tags=["redis"]
            )
            vs.add_decision(decision)

        results = vs.find_similar(
            question="Redis question",
            context={"framework": "FastAPI"},
            top_k=3,
            threshold=0.0 # Relax threshold to ensure matches
        )

        assert len(results) <= 3
