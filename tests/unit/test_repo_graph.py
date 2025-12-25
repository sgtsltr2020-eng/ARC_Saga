"""
Unit Tests for USMA Repo-to-Graph Indexer
==========================================

Tests for RepoGraph, StaticAnalyzer, and impact analysis.
"""


from saga.core.memory import (
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
    RepoGraph,
    StaticAnalyzer,
)


class TestRepoGraph:
    """Tests for RepoGraph core operations."""

    def test_graph_initialization(self):
        """Test graph initializes with empty state."""
        graph = RepoGraph()
        assert graph.node_count == 0
        assert graph.edge_count == 0

    def test_add_node(self):
        """Test adding nodes to the graph."""
        graph = RepoGraph()
        node = GraphNode(
            node_id="file:test.py",
            node_type=NodeType.FILE,
            name="test.py",
            file_path="test.py"
        )
        graph.add_node(node)

        assert graph.node_count == 1
        assert graph.has_node("file:test.py")

        retrieved = graph.get_node("file:test.py")
        assert retrieved is not None
        assert retrieved.name == "test.py"
        assert retrieved.node_type == NodeType.FILE

    def test_add_edge(self):
        """Test adding edges between nodes."""
        graph = RepoGraph()

        # Add two nodes
        graph.add_node(GraphNode("file:a.py", NodeType.FILE, "a.py"))
        graph.add_node(GraphNode("file:b.py", NodeType.FILE, "b.py"))

        # Add edge
        graph.add_edge(GraphEdge(
            source_id="file:a.py",
            target_id="file:b.py",
            edge_type=EdgeType.IMPORTS
        ))

        assert graph.edge_count == 1
        edges = graph.get_edges_from("file:a.py")
        assert len(edges) == 1
        assert edges[0][1] == "file:b.py"

    def test_get_nodes_by_type(self):
        """Test filtering nodes by type."""
        graph = RepoGraph()
        graph.add_node(GraphNode("file:a.py", NodeType.FILE, "a.py"))
        graph.add_node(GraphNode("func:a.py:foo", NodeType.FUNCTION, "foo"))
        graph.add_node(GraphNode("class:a.py:Bar", NodeType.CLASS, "Bar"))

        files = graph.get_nodes_by_type(NodeType.FILE)
        functions = graph.get_nodes_by_type(NodeType.FUNCTION)
        classes = graph.get_nodes_by_type(NodeType.CLASS)

        assert len(files) == 1
        assert len(functions) == 1
        assert len(classes) == 1


class TestImpactAnalysis:
    """Tests for impact analysis functionality."""

    def test_analyze_impact_simple_chain(self):
        """Test impact analysis with simple dependency chain."""
        graph = RepoGraph()

        # Create chain: A -> B -> C
        graph.add_node(GraphNode("func:a", NodeType.FUNCTION, "a"))
        graph.add_node(GraphNode("func:b", NodeType.FUNCTION, "b"))
        graph.add_node(GraphNode("func:c", NodeType.FUNCTION, "c"))

        graph.add_edge(GraphEdge("func:a", "func:b", EdgeType.CALLS))
        graph.add_edge(GraphEdge("func:b", "func:c", EdgeType.CALLS))

        # Analyze impact of changing A
        result = graph.analyze_impact("func:a", max_depth=3)

        assert len(result["affected_nodes"]) == 2
        affected_ids = [n["node_id"] for n in result["affected_nodes"]]
        assert "func:b" in affected_ids
        assert "func:c" in affected_ids

    def test_analyze_impact_no_dependencies(self):
        """Test impact analysis on isolated node."""
        graph = RepoGraph()
        graph.add_node(GraphNode("func:isolated", NodeType.FUNCTION, "isolated"))

        result = graph.analyze_impact("func:isolated")

        assert len(result["affected_nodes"]) == 0
        assert result["impact_score"] == 0.0

    def test_analyze_impact_nonexistent_node(self):
        """Test impact analysis on nonexistent node."""
        graph = RepoGraph()

        result = graph.analyze_impact("nonexistent")

        assert len(result["affected_nodes"]) == 0


class TestLoreBridging:
    """Tests for LoreEntry to code bridging."""

    def test_link_lore_entry(self):
        """Test linking LoreEntry to code entities."""
        graph = RepoGraph()

        # Add code entities
        graph.add_node(GraphNode("file:main.py", NodeType.FILE, "main.py"))
        graph.add_node(GraphNode("func:main.py:run", NodeType.FUNCTION, "run"))

        # Link lore entry
        graph.link_lore_entry(
            lore_entry_id="lore-123",
            entity_ids=["file:main.py", "func:main.py:run"],
            summary="Refactored main entry point"
        )

        # Verify lore node created
        assert graph.has_node("lore-123")
        lore_node = graph.get_node("lore-123")
        assert lore_node.node_type == NodeType.LORE_ENTRY

        # Verify DESCRIBES edges
        edges = graph.get_edges_from("lore-123")
        assert len(edges) == 2
        for _, target, data in edges:
            assert data["edge_type"] == EdgeType.DESCRIBES.value


class TestSubgraph:
    """Tests for subgraph operations."""

    def test_get_related_subgraph(self):
        """Test extracting related nodes within depth."""
        graph = RepoGraph()

        # Create: A -> B -> C -> D
        for name in ["a", "b", "c", "d"]:
            graph.add_node(GraphNode(f"func:{name}", NodeType.FUNCTION, name))

        graph.add_edge(GraphEdge("func:a", "func:b", EdgeType.CALLS))
        graph.add_edge(GraphEdge("func:b", "func:c", EdgeType.CALLS))
        graph.add_edge(GraphEdge("func:c", "func:d", EdgeType.CALLS))

        # Get subgraph within depth 1 of B
        subgraph = graph.get_related_subgraph("func:b", depth=1)

        assert subgraph.node_count == 3  # A, B, C
        assert subgraph.has_node("func:a")
        assert subgraph.has_node("func:b")
        assert subgraph.has_node("func:c")
        assert not subgraph.has_node("func:d")


class TestSerialization:
    """Tests for graph serialization."""

    def test_to_dict_and_from_dict(self):
        """Test graph serialization round-trip."""
        graph = RepoGraph()
        graph.add_node(GraphNode("file:test.py", NodeType.FILE, "test.py", "test.py"))
        graph.add_node(GraphNode("func:test.py:foo", NodeType.FUNCTION, "foo", "test.py", 10))
        graph.add_edge(GraphEdge("file:test.py", "func:test.py:foo", EdgeType.CONTAINS))

        # Serialize
        data = graph.to_dict()

        # Deserialize
        restored = RepoGraph.from_dict(data)

        assert restored.node_count == 2
        assert restored.edge_count == 1
        assert restored.has_node("file:test.py")
        assert restored.has_node("func:test.py:foo")


class TestStaticAnalyzer:
    """Tests for StaticAnalyzer AST parsing."""

    def test_analyzer_initialization(self, tmp_path):
        """Test analyzer initializes correctly."""
        analyzer = StaticAnalyzer(tmp_path)
        assert analyzer.project_root == tmp_path
        assert analyzer.graph.node_count == 0

    def test_analyze_simple_file(self, tmp_path):
        """Test analyzing a simple Python file."""
        # Create test file
        test_file = tmp_path / "simple.py"
        test_file.write_text("""
def hello():
    print("Hello")

class Greeter:
    def greet(self):
        hello()
""")

        analyzer = StaticAnalyzer(tmp_path)
        graph = analyzer.analyze_project()

        # Should have: 1 file, 1 class, 2 functions
        files = graph.get_nodes_by_type(NodeType.FILE)
        classes = graph.get_nodes_by_type(NodeType.CLASS)
        functions = graph.get_nodes_by_type(NodeType.FUNCTION)

        assert len(files) == 1
        assert len(classes) == 1
        assert len(functions) >= 2  # hello + greet

    def test_analyze_imports(self, tmp_path):
        """Test analyzing import statements."""
        test_file = tmp_path / "imports.py"
        test_file.write_text("""
import os
from pathlib import Path
""")

        analyzer = StaticAnalyzer(tmp_path)
        graph = analyzer.analyze_project()

        # Should have import edges
        file_id = "file:imports.py"
        edges = graph.get_edges_from(file_id)
        import_edges = [e for e in edges if e[2].get("edge_type") == EdgeType.IMPORTS.value]

        assert len(import_edges) == 2

    def test_excludes_venv(self, tmp_path):
        """Test that venv directories are excluded."""
        # Create venv dir with Python file
        venv_dir = tmp_path / "venv"
        venv_dir.mkdir()
        (venv_dir / "ignored.py").write_text("x = 1")

        # Create normal file
        (tmp_path / "included.py").write_text("y = 2")

        analyzer = StaticAnalyzer(tmp_path)
        graph = analyzer.analyze_project()

        files = graph.get_nodes_by_type(NodeType.FILE)
        file_names = [f.name for f in files]

        assert "included.py" in file_names
        assert "ignored.py" not in file_names
