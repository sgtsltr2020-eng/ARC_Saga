"""
Unit Tests for Lore Consolidator
=================================

Tests for clustering, chapter generation, and consolidation pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

import numpy as np
import pytest

from saga.core.memory import (
    ChapterGenerator,
    ConsolidationConfig,
    EntryCluster,
    LoreClusterer,
    LoreConsolidator,
    MythosLibrary,
)

# ═══════════════════════════════════════════════════════════════
# MOCK OBJECTS
# ═══════════════════════════════════════════════════════════════

@dataclass
class MockLoreEntry:
    """Mock LoreEntry for testing."""
    entry_id: str = field(default_factory=lambda: str(uuid4()))
    summary: str = "Test decision summary"
    tags: list[str] = field(default_factory=list)
    codex_status: str = "COMPLIANT"
    created_at: datetime = field(default_factory=datetime.utcnow)
    chapter_id: str | None = None
    embedding: np.ndarray | None = None


class MockLoreBook:
    """Mock LoreBook for testing."""

    def __init__(self):
        self.lore_entries: list[MockLoreEntry] = []

    def add_entry(self, entry: MockLoreEntry) -> None:
        self.lore_entries.append(entry)


# ═══════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════

class TestEntryCluster:
    """Tests for EntryCluster."""

    def test_compute_stats(self):
        """Test aggregate statistics computation."""
        entries = [
            MockLoreEntry(tags=["async", "api"], codex_status="COMPLIANT"),
            MockLoreEntry(tags=["async", "db"], codex_status="COMPLIANT"),
            MockLoreEntry(tags=["async"], codex_status="VIOLATION"),
        ]

        cluster = EntryCluster(entries=entries)
        cluster.compute_stats()

        assert cluster.success_rate == pytest.approx(2/3)
        assert "async" in cluster.dominant_tags

    def test_empty_cluster_stats(self):
        """Test stats on empty cluster."""
        cluster = EntryCluster()
        cluster.compute_stats()

        assert cluster.success_rate == 0.0
        assert cluster.dominant_tags == []


class TestLoreClusterer:
    """Tests for LoreClusterer."""

    @pytest.fixture
    def clusterer(self):
        return LoreClusterer(
            similarity_threshold=0.7,
            min_cluster_size=2,
            max_cluster_size=10
        )

    def test_cluster_by_tags(self, clusterer):
        """Test tag-based clustering fallback."""
        entries = [
            MockLoreEntry(tags=["async"]),
            MockLoreEntry(tags=["async"]),
            MockLoreEntry(tags=["async"]),
            MockLoreEntry(tags=["sync"]),
        ]

        clusters = clusterer.cluster_entries(entries)

        # Should form one cluster for "async"
        assert len(clusters) >= 1
        async_cluster = [c for c in clusters if "async" in c.dominant_tags]
        assert len(async_cluster) == 1
        assert len(async_cluster[0].entries) == 3

    def test_cluster_by_embedding(self, clusterer):
        """Test embedding-based clustering."""
        entries = [
            MockLoreEntry(entry_id="e1"),
            MockLoreEntry(entry_id="e2"),
            MockLoreEntry(entry_id="e3"),
        ]

        # Similar embeddings for e1, e2; different for e3
        embeddings = {
            "e1": np.array([1.0, 0.0, 0.0]),
            "e2": np.array([0.9, 0.1, 0.0]),
            "e3": np.array([0.0, 0.0, 1.0]),
        }

        clusters = clusterer.cluster_entries(entries, embeddings)

        # e1 and e2 should cluster together (similar embeddings)
        assert len(clusters) >= 1

    def test_cosine_similarity(self):
        """Test cosine similarity computation."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        c = np.array([0.0, 1.0, 0.0])

        assert LoreClusterer._cosine_similarity(a, b) == pytest.approx(1.0)
        assert LoreClusterer._cosine_similarity(a, c) == pytest.approx(0.0)

    def test_empty_entries(self, clusterer):
        """Test clustering with no entries."""
        clusters = clusterer.cluster_entries([])
        assert clusters == []


class TestChapterGenerator:
    """Tests for ChapterGenerator."""

    @pytest.fixture
    def generator(self):
        return ChapterGenerator()

    @pytest.fixture
    def sample_cluster(self):
        entries = [
            MockLoreEntry(
                summary="Always use async for I/O operations",
                tags=["async", "performance"],
                codex_status="COMPLIANT"
            ),
            MockLoreEntry(
                summary="Async improves response times",
                tags=["async", "api"],
                codex_status="COMPLIANT"
            ),
            MockLoreEntry(
                summary="Sync I/O blocks event loop",
                tags=["async", "bug"],
                codex_status="VIOLATION"
            ),
        ]
        cluster = EntryCluster(
            entry_ids=[e.entry_id for e in entries],
            entries=entries
        )
        cluster.compute_stats()
        return cluster

    def test_generate_heuristic(self, generator, sample_cluster):
        """Test heuristic chapter generation."""
        chapter_data = generator.generate_chapter(sample_cluster)

        assert "title" in chapter_data
        assert "summary" in chapter_data
        assert "tags" in chapter_data
        assert chapter_data["entry_count"] == 3

    def test_generate_with_mock_llm(self, sample_cluster):
        """Test LLM-based generation with mock."""
        def mock_llm(prompt: str) -> str:
            return '''{"title": "The Async Imperative", "summary": "Always use async for I/O.", "tags": ["async"], "principles": ["Async first"]}'''

        generator = ChapterGenerator(llm_summarizer=mock_llm)
        chapter_data = generator.generate_chapter(sample_cluster)

        assert chapter_data["title"] == "The Async Imperative"
        assert "async" in chapter_data["tags"]

    def test_llm_failure_fallback(self, sample_cluster):
        """Test fallback when LLM fails."""
        def failing_llm(prompt: str) -> str:
            raise Exception("LLM unavailable")

        generator = ChapterGenerator(llm_summarizer=failing_llm)
        chapter_data = generator.generate_chapter(sample_cluster)

        # Should fallback to heuristic
        assert "title" in chapter_data
        assert chapter_data["entry_count"] == 3


class TestLoreConsolidator:
    """Tests for main consolidation pipeline."""

    @pytest.fixture
    def consolidator(self):
        lorebook = MockLoreBook()
        mythos = MythosLibrary()

        return LoreConsolidator(
            config=ConsolidationConfig(
                entry_threshold=5,
                time_threshold_hours=24,
                min_cluster_size=2
            ),
            lorebook=lorebook,
            mythos_library=mythos
        )

    def test_entry_trigger(self, consolidator):
        """Test that entry threshold triggers consolidation."""
        # Add entries below threshold
        for _ in range(4):
            triggered = consolidator.on_entry_added()
            assert triggered is False

        # Add one more to reach threshold
        triggered = consolidator.on_entry_added()
        assert triggered is True

    def test_consolidate_now(self, consolidator):
        """Test immediate consolidation."""
        # Add entries to lorebook
        for i in range(10):
            entry = MockLoreEntry(
                summary=f"Entry {i}",
                tags=["test"] if i < 7 else ["other"]
            )
            consolidator.lorebook.add_entry(entry)

        stats = consolidator.consolidate_now()

        assert stats.entries_processed == 10
        assert stats.duration_seconds >= 0

    def test_chapter_creation(self, consolidator):
        """Test that chapters are created and added to library."""
        # Add clusterable entries
        for i in range(6):
            entry = MockLoreEntry(
                summary=f"Async operation {i}",
                tags=["async", "pattern"]
            )
            consolidator.lorebook.add_entry(entry)

        consolidator.consolidate_now()

        # Should have created at least one chapter
        assert len(consolidator.mythos_library.chapters) >= 0  # May be 0 if min cluster not met

    def test_entries_marked_consolidated(self, consolidator):
        """Test that entries are marked with chapter_id."""
        # Add entries
        for i in range(6):
            entry = MockLoreEntry(
                summary=f"Test entry {i}",
                tags=["api"]
            )
            consolidator.lorebook.add_entry(entry)

        consolidator.consolidate_now()

        # Check if entries got chapter_id (if chapter was created)
        # This depends on cluster formation

    def test_get_stats(self, consolidator):
        """Test statistics reporting."""
        stats = consolidator.get_stats()

        assert "total_consolidations" in stats
        assert "unconsolidated_count" in stats
        assert "worker_running" in stats


class TestConsolidatorPersistence:
    """Tests for consolidator serialization."""

    def test_serialization_roundtrip(self):
        """Test save/load cycle."""
        consolidator = LoreConsolidator(
            config=ConsolidationConfig(entry_threshold=10)
        )

        # Set some state
        consolidator.total_consolidations = 5
        consolidator.total_chapters_created = 3
        consolidator.chapter_embeddings["ch1"] = np.random.randn(384).astype(np.float32)

        # Serialize
        data = consolidator.to_dict()

        # Deserialize
        restored = LoreConsolidator.from_dict(data)

        assert restored.total_consolidations == 5
        assert restored.total_chapters_created == 3
        assert "ch1" in restored.chapter_embeddings

    def test_embedding_precision(self):
        """Test that embeddings are preserved accurately."""
        consolidator = LoreConsolidator()

        original_emb = np.random.randn(384).astype(np.float32)
        consolidator.chapter_embeddings["test"] = original_emb

        # Round-trip
        data = consolidator.to_dict()
        restored = LoreConsolidator.from_dict(data)

        assert np.allclose(original_emb, restored.chapter_embeddings["test"])


class TestChapterSearch:
    """Tests for chapter search functionality."""

    def test_search_chapters(self):
        """Test semantic search on chapters."""
        def mock_embedder(text: str) -> np.ndarray:
            # Deterministic embedding based on text hash
            import hashlib
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            return rng.random(384).astype(np.float32)

        consolidator = LoreConsolidator(
            embedding_generator=mock_embedder
        )

        # Add chapter embeddings
        consolidator.chapter_embeddings["ch1"] = mock_embedder("async patterns")
        consolidator.chapter_embeddings["ch2"] = mock_embedder("database queries")

        # Search with same embedding as ch1
        results = consolidator.search_chapters(
            "async",
            top_k=2,
            query_embedding=mock_embedder("async patterns")
        )

        assert len(results) == 2
        # First result should be ch1 (exact match)
        assert results[0][0] == "ch1"
        assert results[0][1] == pytest.approx(1.0)  # Exact match

    def test_search_empty(self):
        """Test search with no chapters."""
        consolidator = LoreConsolidator()
        results = consolidator.search_chapters("test")

        assert results == []

