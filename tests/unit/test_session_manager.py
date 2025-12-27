"""
Unit Tests for SessionManager
==============================

Tests for persistence, atomic writes, snapshots, and component serialization.
"""

import time

import numpy as np
import pytest

from saga.core.memory import (
    DirtyFlags,
    GraphNode,
    MythosChapter,
    MythosLibrary,
    NodeType,
    PersistenceConfig,
    RepoGraph,
    SessionManager,
    SolvedPattern,
)


class TestDirtyFlags:
    """Tests for DirtyFlags."""

    def test_initial_state(self):
        """Test flags start clean."""
        flags = DirtyFlags()

        assert flags.graph is False
        assert flags.mythos is False
        assert flags.optimizer is False
        assert flags.graph_changes == 0

    def test_mark_dirty(self):
        """Test marking components dirty."""
        flags = DirtyFlags()

        flags.mark_graph_dirty(5)

        assert flags.graph is True
        assert flags.graph_changes == 5

    def test_should_checkpoint(self):
        """Test checkpoint threshold."""
        flags = DirtyFlags()

        flags.mark_graph_dirty(30)
        flags.mark_mythos_dirty(25)

        assert flags.should_checkpoint(50) is True
        assert flags.should_checkpoint(100) is False

    def test_reset(self):
        """Test resetting flags."""
        flags = DirtyFlags()
        flags.mark_graph_dirty(100)
        flags.mark_mythos_dirty(50)

        flags.reset()

        assert flags.graph is False
        assert flags.graph_changes == 0


class TestPersistenceConfig:
    """Tests for PersistenceConfig."""

    def test_default_paths(self):
        """Test default configuration."""
        config = PersistenceConfig()

        assert config.graph_file == "graph.json"
        assert config.mythos_file == "mythos.json"
        assert config.checkpoint_interval_seconds == 900


class TestSessionManagerSingleton:
    """Tests for singleton behavior."""

    def test_singleton_returns_same_instance(self, tmp_path):
        """Test that SessionManager is a singleton."""
        # Reset any existing singleton
        SessionManager.reset_singleton()

        config = PersistenceConfig(base_path=tmp_path / "session1")
        session1 = SessionManager(config)
        session2 = SessionManager()

        assert session1 is session2

        # Cleanup
        SessionManager.reset_singleton()

    def test_reset_singleton(self, tmp_path):
        """Test singleton can be reset."""
        SessionManager.reset_singleton()

        config = PersistenceConfig(base_path=tmp_path / "session_reset")
        session1 = SessionManager(config)

        SessionManager.reset_singleton()

        config2 = PersistenceConfig(base_path=tmp_path / "session_reset2")
        session2 = SessionManager(config2)

        assert session1 is not session2

        SessionManager.reset_singleton()


class TestSessionManagerGraphPersistence:
    """Tests for graph save/load."""

    @pytest.fixture
    def session(self, tmp_path):
        """Create fresh session for each test."""
        SessionManager.reset_singleton()
        config = PersistenceConfig(base_path=tmp_path / "graph_test")
        session = SessionManager(config)
        yield session
        SessionManager.reset_singleton()

    def test_save_and_load_graph(self, session):
        """Test graph round-trip persistence."""
        # Create graph with nodes
        graph = RepoGraph()
        node1 = GraphNode("func:a", NodeType.FUNCTION, "test_function")
        node1.embedding_vector = np.random.randn(384).astype(np.float32)
        graph.add_node(node1)

        node2 = GraphNode("class:b", NodeType.CLASS, "TestClass")
        graph.add_node(node2)

        # Register and save
        session.register_graph(graph)
        result = session.save_all()

        assert result["graph"] is True

        # Create new graph and load
        new_graph = RepoGraph()
        SessionManager.reset_singleton()

        config = PersistenceConfig(base_path=session.config.base_path)
        new_session = SessionManager(config)
        new_session.register_graph(new_graph)
        new_session.load_all()

        # Verify
        assert len(new_graph._node_index) == 2
        assert new_graph.get_node("func:a") is not None
        assert new_graph.get_node("class:b") is not None

        # Verify embedding preserved
        loaded_node = new_graph.get_node("func:a")
        assert loaded_node.embedding_vector is not None
        assert np.allclose(node1.embedding_vector, loaded_node.embedding_vector)

        SessionManager.reset_singleton()

    def test_atomic_write(self, session):
        """Test that saves are atomic (temp file then rename)."""
        graph = RepoGraph()
        graph.add_node(GraphNode("func:a", NodeType.FUNCTION, "test"))

        session.register_graph(graph)
        session.save_all()

        # Verify no temp files left
        temp_files = list(session.config.base_path.glob("*.tmp"))
        assert len(temp_files) == 0


class TestSessionManagerMythosPersistence:
    """Tests for mythos save/load."""

    @pytest.fixture
    def session(self, tmp_path):
        """Create fresh session for each test."""
        SessionManager.reset_singleton()
        config = PersistenceConfig(base_path=tmp_path / "mythos_test")
        session = SessionManager(config)
        yield session
        SessionManager.reset_singleton()

    def test_save_and_load_mythos(self, session):
        """Test mythos round-trip persistence."""
        # Create mythos with chapter
        mythos = MythosLibrary()
        chapter = MythosChapter(
            title="Test Chapter",
            summary="A test chapter",
            universal_principles=["DRY", "KISS"],
            solved_patterns=[
                SolvedPattern(name="Singleton", description="One instance")
            ]
        )
        mythos.add_chapter(chapter)

        # Save
        session.register_mythos(mythos)
        session.save_all()

        # Load into new mythos
        new_mythos = MythosLibrary()
        SessionManager.reset_singleton()

        config = PersistenceConfig(base_path=session.config.base_path)
        new_session = SessionManager(config)
        new_session.register_mythos(new_mythos)
        new_session.load_all()

        # Verify
        assert len(new_mythos.chapters) == 1
        assert new_mythos.chapters[0].title == "Test Chapter"
        assert len(new_mythos.chapters[0].universal_principles) == 2

        SessionManager.reset_singleton()


class TestSessionManagerSnapshots:
    """Tests for snapshot system."""

    @pytest.fixture
    def session(self, tmp_path):
        """Create fresh session for each test."""
        SessionManager.reset_singleton()
        config = PersistenceConfig(base_path=tmp_path / "snapshot_test")
        session = SessionManager(config)
        yield session
        SessionManager.reset_singleton()

    def test_create_snapshot(self, session):
        """Test snapshot creation."""
        # Save something first
        graph = RepoGraph()
        graph.add_node(GraphNode("func:a", NodeType.FUNCTION, "test"))
        session.register_graph(graph)
        session.save_all()

        # Create snapshot
        snapshot_name = session._create_snapshot()

        assert snapshot_name is not None
        snapshot_path = session.config.base_path / session.config.snapshot_dir / snapshot_name
        assert snapshot_path.exists()

    def test_restore_from_snapshot(self, session):
        """Test restoring from snapshot."""
        # Create initial state
        graph = RepoGraph()
        graph.add_node(GraphNode("func:a", NodeType.FUNCTION, "original"))
        session.register_graph(graph)
        session.save_all()

        # Create snapshot
        snapshot_name = session._create_snapshot()

        # Modify and save
        graph.add_node(GraphNode("func:b", NodeType.FUNCTION, "added"))
        session.save_all()

        # Restore
        success = session.restore_from_snapshot(snapshot_name)
        assert success is True

    def test_cleanup_old_snapshots(self, session):
        """Test that old snapshots are cleaned up."""
        graph = RepoGraph()
        graph.add_node(GraphNode("func:a", NodeType.FUNCTION, "test"))
        session.register_graph(graph)
        session.save_all()

        # Create more snapshots than max
        for i in range(session.config.max_snapshots + 3):
            time.sleep(0.01)  # Ensure different timestamps
            session._create_snapshot()

        # Check only max kept
        snapshot_dir = session.config.base_path / session.config.snapshot_dir
        snapshots = list(snapshot_dir.iterdir())
        assert len(snapshots) <= session.config.max_snapshots


class TestSessionManagerCheckpoint:
    """Tests for checkpoint functionality."""

    @pytest.fixture
    def session(self, tmp_path):
        """Create fresh session for each test."""
        SessionManager.reset_singleton()
        config = PersistenceConfig(base_path=tmp_path / "checkpoint_test")
        session = SessionManager(config)
        yield session
        SessionManager.reset_singleton()

    def test_checkpoint_only_saves_dirty(self, session):
        """Test that checkpoint only saves dirty components."""
        graph = RepoGraph()
        mythos = MythosLibrary()

        session.register_graph(graph)
        session.register_mythos(mythos)

        # Mark only graph dirty
        session.dirty.mark_graph_dirty()

        result = session.checkpoint()

        assert "graph" in result
        assert "mythos" not in result  # Not dirty, not saved

    def test_get_status(self, session):
        """Test status reporting."""
        graph = RepoGraph()
        session.register_graph(graph)
        session.dirty.mark_graph_dirty(25)

        status = session.get_status()

        assert status["registered"]["graph"] is True
        assert status["dirty"]["graph"] is True
        assert status["dirty"]["total_changes"] == 25


class TestEncodingUtilities:
    """Tests for array encoding/decoding."""

    @pytest.fixture
    def session(self, tmp_path):
        """Create fresh session."""
        SessionManager.reset_singleton()
        config = PersistenceConfig(base_path=tmp_path / "encoding_test")
        session = SessionManager(config)
        yield session
        SessionManager.reset_singleton()

    def test_array_round_trip(self, session):
        """Test numpy array encoding/decoding preserves precision."""
        original = np.random.randn(384).astype(np.float32)

        encoded = session._encode_array(original)
        decoded = session._decode_array(encoded)

        assert np.allclose(original, decoded)

    def test_encoding_is_base64(self, session):
        """Test that encoding produces valid base64."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        encoded = session._encode_array(arr)

        assert isinstance(encoded, str)
        # Should be decodable
        import base64
        base64.b64decode(encoded)  # Should not raise
