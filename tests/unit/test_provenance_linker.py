"""
Unit Tests for Provenance Linker
=================================

Tests for bidirectional graph-LoreBook linking.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

import pytest

from saga.core.memory import (
    GraphNode,
    LinkerConfig,
    NodeType,
    ProvenanceLink,
    ProvenanceLinker,
    RepoGraph,
)

# ═══════════════════════════════════════════════════════════════
# MOCK OBJECTS
# ═══════════════════════════════════════════════════════════════

@dataclass
class MockLoreEntry:
    """Mock LoreEntry for testing."""
    entry_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    semantic_tags: list[str] = field(default_factory=list)
    summary: str = ""
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

class TestProvenanceLink:
    """Tests for ProvenanceLink dataclass."""

    def test_create_link(self):
        """Test creating a link."""
        link = ProvenanceLink(
            node_id="func:test",
            entry_id="entry_123"
        )

        assert link.node_id == "func:test"
        assert link.entry_id == "entry_123"
        assert link.link_type == "auto"
        assert link.confidence == 1.0

    def test_serialization(self):
        """Test link serialization."""
        link = ProvenanceLink(
            node_id="node_1",
            entry_id="entry_1",
            link_type="manual",
            confidence=0.8
        )

        data = link.to_dict()
        restored = ProvenanceLink.from_dict(data)

        assert restored.node_id == link.node_id
        assert restored.entry_id == link.entry_id
        assert restored.confidence == link.confidence


class TestLinkerConfig:
    """Tests for LinkerConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = LinkerConfig()

        assert config.max_links_per_node == 30
        assert config.similarity_threshold == 0.6
        assert config.recency_window_hours == 24

    def test_custom_config(self):
        """Test custom configuration."""
        config = LinkerConfig(
            max_links_per_node=50,
            similarity_threshold=0.8
        )

        assert config.max_links_per_node == 50
        assert config.similarity_threshold == 0.8


class TestProvenanceLinker:
    """Tests for ProvenanceLinker."""

    @pytest.fixture
    def graph(self):
        """Create a test graph."""
        g = RepoGraph()

        # Add some nodes
        for i in range(5):
            node = GraphNode(
                node_id=f"func:test_{i}",
                node_type=NodeType.FUNCTION,
                name=f"test_func_{i}",
                file_path=f"/src/module_{i}.py"
            )
            g.add_node(node)

        return g

    @pytest.fixture
    def lorebook(self):
        """Create a test lorebook."""
        lb = MockLoreBook()

        # Add some entries
        for i in range(3):
            entry = MockLoreEntry(
                entry_id=f"entry_{i}",
                summary=f"Decision about module {i}",
                semantic_tags=["test", f"module_{i}"]
            )
            lb.add_entry(entry)

        return lb

    @pytest.fixture
    def linker(self, graph, lorebook):
        """Create linker with graph and lorebook."""
        return ProvenanceLinker(graph=graph, lorebook=lorebook)

    def test_manual_link(self, linker):
        """Test creating a manual link."""
        result = linker.link("func:test_0", "entry_0", link_type="manual")

        assert result is True
        assert "entry_0" in linker.get_lore_for_node("func:test_0")
        assert "func:test_0" in linker.get_nodes_for_entry("entry_0")

    def test_duplicate_link(self, linker):
        """Test that duplicate links are rejected."""
        linker.link("func:test_0", "entry_0")
        result = linker.link("func:test_0", "entry_0")

        assert result is False

    def test_link_limit(self, linker):
        """Test that link limits are enforced."""
        linker.config.max_links_per_node = 3

        for i in range(5):
            linker.link("func:test_0", f"entry_{i}")

        # Should only have 3 links
        assert len(linker.get_lore_for_node("func:test_0")) == 3

    def test_unlink(self, linker):
        """Test removing a link."""
        linker.link("func:test_0", "entry_0")
        result = linker.unlink("func:test_0", "entry_0")

        assert result is True
        assert "entry_0" not in linker.get_lore_for_node("func:test_0")

    def test_on_entry_created_with_context(self, linker):
        """Test auto-linking on entry creation."""
        # Set active context
        linker.set_active_context(["func:test_0", "func:test_1"])

        # Create entry
        links = linker.on_entry_created("new_entry", context_nodes=["func:test_2"])

        assert links >= 1
        # Should link to context nodes
        nodes = linker.get_nodes_for_entry("new_entry")
        assert len(nodes) >= 1

    def test_context_tracking(self, linker):
        """Test context tracking."""
        linker.add_to_context("func:test_0")
        linker.add_to_context("func:test_1")

        assert "func:test_0" in linker._active_context_nodes
        assert "func:test_1" in linker._active_context_nodes

        linker.clear_context()
        assert len(linker._active_context_nodes) == 0


class TestQueryHelpers:
    """Tests for query helper methods."""

    @pytest.fixture
    def linked_setup(self):
        """Create a graph with linked entries."""
        graph = RepoGraph()
        lorebook = MockLoreBook()

        # Create nodes
        for i in range(3):
            node = GraphNode(
                node_id=f"func:f{i}",
                node_type=NodeType.FUNCTION,
                name=f"func_{i}"
            )
            graph.add_node(node)

        # Create entries
        for i in range(3):
            entry = MockLoreEntry(entry_id=f"e{i}")
            lorebook.add_entry(entry)

        linker = ProvenanceLinker(graph=graph, lorebook=lorebook)

        # Create links
        linker.link("func:f0", "e0")
        linker.link("func:f0", "e1")
        linker.link("func:f1", "e0")
        linker.link("func:f2", "e2")

        return graph, lorebook, linker

    def test_get_lore_for_node(self, linked_setup):
        """Test getting entries for a node."""
        _, _, linker = linked_setup

        entries = linker.get_lore_for_node("func:f0")

        assert len(entries) == 2
        assert "e0" in entries
        assert "e1" in entries

    def test_get_nodes_for_entry(self, linked_setup):
        """Test getting nodes for an entry."""
        _, _, linker = linked_setup

        nodes = linker.get_nodes_for_entry("e0")

        assert len(nodes) == 2
        assert "func:f0" in nodes
        assert "func:f1" in nodes

    def test_get_provenance_path(self, linked_setup):
        """Test getting provenance chain."""
        _, _, linker = linked_setup

        path = linker.get_provenance_path("func:f0")

        assert len(path) >= 1
        assert path[0]["id"] == "func:f0"
        assert len(path[0]["entries"]) == 2

    def test_get_link_metadata(self, linked_setup):
        """Test getting link metadata."""
        _, _, linker = linked_setup

        link = linker.get_link_metadata("func:f0", "e0")

        assert link is not None
        assert link.node_id == "func:f0"
        assert link.entry_id == "e0"


class TestPersistence:
    """Tests for linker persistence."""

    def test_serialization_roundtrip(self):
        """Test save/load cycle."""
        graph = RepoGraph()
        graph.add_node(GraphNode("n1", NodeType.FUNCTION, "func1"))
        graph.add_node(GraphNode("n2", NodeType.FUNCTION, "func2"))

        linker = ProvenanceLinker(graph=graph)

        # Create links
        linker.link("n1", "e1", confidence=0.9)
        linker.link("n1", "e2")
        linker.link("n2", "e1")

        # Serialize
        data = linker.to_dict()

        # Deserialize
        restored = ProvenanceLinker.from_dict(data, graph=graph)

        # Verify
        assert len(restored.get_lore_for_node("n1")) == 2
        assert len(restored.get_nodes_for_entry("e1")) == 2

        link = restored.get_link_metadata("n1", "e1")
        assert link.confidence == 0.9

    def test_sync_from_graph(self):
        """Test syncing links from graph node fields."""
        graph = RepoGraph()

        # Create node with existing lore_entry_ids
        node = GraphNode("n1", NodeType.FUNCTION, "func1")
        node.lore_entry_ids = ["e1", "e2"]
        graph.add_node(node)

        linker = ProvenanceLinker(graph=graph)
        synced = linker.sync_from_graph()

        assert synced == 2
        assert len(linker.get_lore_for_node("n1")) == 2


class TestPerformance:
    """Performance tests for large datasets."""

    def test_1k_nodes_entries(self):
        """Test performance with 1000+ nodes and entries."""
        graph = RepoGraph()
        lorebook = MockLoreBook()

        # Create 1000 nodes
        for i in range(1000):
            node = GraphNode(
                node_id=f"node_{i}",
                node_type=NodeType.FUNCTION,
                name=f"func_{i}"
            )
            graph.add_node(node)

        # Create 500 entries
        for i in range(500):
            entry = MockLoreEntry(entry_id=f"entry_{i}")
            lorebook.add_entry(entry)

        linker = ProvenanceLinker(graph=graph, lorebook=lorebook)

        # Time linking
        start = time.time()

        # Create 2000 random links
        for i in range(2000):
            linker.link(f"node_{i % 1000}", f"entry_{i % 500}")

        link_time = time.time() - start

        # Time queries
        start = time.time()

        for i in range(100):
            linker.get_lore_for_node(f"node_{i}")
            linker.get_nodes_for_entry(f"entry_{i}")

        query_time = time.time() - start

        # Assertions
        stats = linker.get_stats()
        assert stats["total_links"] > 0
        assert link_time < 2.0  # Should be fast
        assert query_time < 0.5  # Queries should be very fast

    def test_stats(self):
        """Test statistics reporting."""
        linker = ProvenanceLinker()

        stats = linker.get_stats()

        assert "total_links" in stats
        assert "nodes_with_links" in stats
        assert "entries_with_links" in stats
