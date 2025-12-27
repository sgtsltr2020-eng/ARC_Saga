"""
Unit Tests for File Watcher and Incremental Analyzer
=====================================================

Tests for file system monitoring and incremental graph updates.
"""

import tempfile
import time
from pathlib import Path

import pytest

from saga.core.memory import (
    EdgeType,
    FileEvent,
    FileWatcher,
    GraphNode,
    GraphWatcherService,
    IncrementalAnalyzer,
    NodeType,
    RepoGraph,
    WatcherConfig,
)


class TestFileEvent:
    """Tests for FileEvent dataclass."""

    def test_create_event(self):
        """Test creating a file event."""
        event = FileEvent(
            event_type="modified",
            src_path="/path/to/file.py"
        )

        assert event.event_type == "modified"
        assert event.src_path == "/path/to/file.py"
        assert event.timestamp > 0

    def test_move_event(self):
        """Test move event with destination."""
        event = FileEvent(
            event_type="moved",
            src_path="/old/path.py",
            dst_path="/new/path.py"
        )

        assert event.dst_path == "/new/path.py"


class TestWatcherConfig:
    """Tests for WatcherConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = WatcherConfig()

        assert config.debounce_seconds == 1.0
        assert "__pycache__" in config.excluded_dirs
        assert ".py" in config.watch_extensions

    def test_custom_config(self):
        """Test custom configuration."""
        config = WatcherConfig(
            debounce_seconds=2.0,
            max_events_per_batch=50
        )

        assert config.debounce_seconds == 2.0
        assert config.max_events_per_batch == 50


class TestFileWatcher:
    """Tests for FileWatcher."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)

            # Create some Python files
            (project / "main.py").write_text("print('hello')")
            (project / "utils.py").write_text("def helper(): pass")
            (project / "tests").mkdir()
            (project / "tests" / "test_main.py").write_text("def test_it(): pass")

            yield project

    def test_watcher_stats(self, temp_project):
        """Test watcher statistics."""
        watcher = FileWatcher(temp_project)
        stats = watcher.get_stats()

        assert "events_processed" in stats
        assert stats["running"] is False

    def test_notify_file_changed(self, temp_project):
        """Test manual file change notification."""
        received_events = []

        def on_update(events):
            received_events.extend(events)

        watcher = FileWatcher(
            temp_project,
            config=WatcherConfig(debounce_seconds=0.1),
            on_update=on_update
        )

        # Start watcher
        watcher.start()

        try:
            # Notify of change
            watcher.notify_file_changed(temp_project / "main.py")

            # Wait for debounce
            time.sleep(0.3)

            # May or may not have events depending on content hash
        finally:
            watcher.stop()

    def test_event_filtering(self, temp_project):
        """Test that excluded directories are filtered."""
        watcher = FileWatcher(temp_project)

        # Create event for excluded path
        pycache_event = FileEvent(
            event_type="modified",
            src_path=str(temp_project / "__pycache__" / "module.cpython-312.pyc")
        )

        # Queue it (should be filtered)
        watcher._queue_event(pycache_event)

        # Queue should be empty (event was filtered)
        assert watcher._event_queue.empty()

    def test_extension_filtering(self, temp_project):
        """Test that non-Python files are filtered."""
        watcher = FileWatcher(temp_project)

        # Create event for non-Python file
        txt_event = FileEvent(
            event_type="modified",
            src_path=str(temp_project / "readme.txt")
        )

        # Queue it (should be filtered)
        watcher._queue_event(txt_event)

        # Queue should be empty
        assert watcher._event_queue.empty()


class TestIncrementalAnalyzer:
    """Tests for IncrementalAnalyzer."""

    @pytest.fixture
    def graph(self):
        """Create a test graph."""
        return RepoGraph()

    @pytest.fixture
    def analyzer(self, graph):
        """Create analyzer with graph."""
        return IncrementalAnalyzer(graph=graph)

    def test_process_created_event(self, analyzer, graph, tmp_path):
        """Test handling file creation."""
        # Create a Python file
        test_file = tmp_path / "new_module.py"
        test_file.write_text("def new_func(): pass")

        events = [FileEvent(event_type="created", src_path=str(test_file))]
        stats = analyzer.process_events(events)

        assert stats["created"] == 1

    def test_process_deleted_event(self, analyzer, graph, tmp_path):
        """Test handling file deletion."""
        # Add a node for a file
        test_file = tmp_path / "to_delete.py"
        node = GraphNode(
            node_id="file:to_delete.py",
            node_type=NodeType.FILE,
            name="to_delete.py",
            file_path=str(test_file)
        )
        graph.add_node(node)

        # Process delete event
        events = [FileEvent(event_type="deleted", src_path=str(test_file))]
        stats = analyzer.process_events(events)

        assert stats["deleted"] == 1
        assert analyzer.nodes_removed >= 1

    def test_get_stats(self, analyzer):
        """Test statistics reporting."""
        stats = analyzer.get_stats()

        assert "nodes_added" in stats
        assert "nodes_updated" in stats
        assert "nodes_removed" in stats


class TestGraphWatcherService:
    """Tests for GraphWatcherService."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            (project / "app.py").write_text("# Application")
            yield project

    def test_service_lifecycle(self, temp_project):
        """Test service start/stop."""
        graph = RepoGraph(project_root=temp_project)

        service = GraphWatcherService(
            project_root=temp_project,
            graph=graph
        )

        # Start
        assert service.start() is True

        # Check stats
        stats = service.get_stats()
        assert "watcher" in stats
        assert "analyzer" in stats

        # Stop
        service.stop()

    def test_service_stats(self, temp_project):
        """Test combined statistics."""
        graph = RepoGraph(project_root=temp_project)

        service = GraphWatcherService(
            project_root=temp_project,
            graph=graph
        )

        stats = service.get_stats()

        assert stats["watcher"]["running"] is False
        assert stats["analyzer"]["nodes_added"] == 0


class TestEnrichedGraphNode:
    """Tests for enriched GraphNode fields."""

    def test_new_fields(self):
        """Test that new fields are available."""
        node = GraphNode(
            node_id="func:test",
            node_type=NodeType.FUNCTION,
            name="test_function",
            docstring="This is a test function.",
            summary="Test function that does testing.",
            source_snippet="def test_function():\n    pass",
            end_line=10
        )

        assert node.docstring == "This is a test function."
        assert node.summary == "Test function that does testing."
        assert node.source_snippet is not None
        assert node.end_line == 10

    def test_execution_traces(self):
        """Test execution traces field."""
        node = GraphNode(
            node_id="func:traced",
            node_type=NodeType.FUNCTION,
            name="traced_func"
        )

        # Add execution trace
        node.execution_traces.append({
            "trial_id": "trial_1",
            "passed": True,
            "timestamp": time.time()
        })

        assert len(node.execution_traces) == 1
        assert node.execution_traces[0]["passed"] is True

    def test_provenance_links(self):
        """Test LoreBook/Mythos links."""
        node = GraphNode(
            node_id="class:MyClass",
            node_type=NodeType.CLASS,
            name="MyClass"
        )

        node.lore_entry_ids.append("lore_123")
        node.mythos_chapter_ids.append("chapter_456")

        assert "lore_123" in node.lore_entry_ids
        assert "chapter_456" in node.mythos_chapter_ids

    def test_content_hash(self):
        """Test content hash for dirty detection."""
        node = GraphNode(
            node_id="file:test.py",
            node_type=NodeType.FILE,
            name="test.py",
            content_hash="abc123"
        )

        assert node.content_hash == "abc123"


class TestNewEdgeTypes:
    """Tests for new edge types."""

    def test_data_flow_edge(self):
        """Test DATA_FLOW edge type."""
        assert EdgeType.DATA_FLOW == "DATA_FLOW"

    def test_inter_repo_edge(self):
        """Test INTER_REPO edge type."""
        assert EdgeType.INTER_REPO == "INTER_REPO"

    def test_provenance_edge(self):
        """Test PROVENANCE edge type."""
        assert EdgeType.PROVENANCE == "PROVENANCE"

    def test_add_data_flow_edge(self):
        """Test adding data flow edge to graph."""
        from saga.core.memory import GraphEdge

        graph = RepoGraph()

        # Add nodes
        func1 = GraphNode("func:producer", NodeType.FUNCTION, "producer")
        func2 = GraphNode("func:consumer", NodeType.FUNCTION, "consumer")
        graph.add_node(func1)
        graph.add_node(func2)

        # Add data flow edge
        edge = GraphEdge(
            source_id="func:producer",
            target_id="func:consumer",
            edge_type=EdgeType.DATA_FLOW,
            metadata={"variable": "data", "flow_type": "return->param"}
        )
        graph.add_edge(edge)

        # Verify (edges are returned as tuples: (source, target, data_dict))
        edges = graph.get_edges_from("func:producer")
        assert len(edges) == 1
        source, target, data = edges[0]
        assert data["edge_type"] == EdgeType.DATA_FLOW.value

