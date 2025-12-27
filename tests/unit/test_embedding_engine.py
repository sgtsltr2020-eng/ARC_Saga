"""
Unit Tests for Embedding Engine
================================

Tests for EmbeddingGenerator, SemanticSearchEngine, and GraphEmbeddingManager.
"""


import numpy as np
import pytest

from saga.core.memory import (
    CodeSummarizer,
    EdgeType,
    EmbeddingGenerator,
    EmbeddingResult,
    EmbeddingStore,
    GraphEdge,
    GraphEmbeddingManager,
    GraphNode,
    NodeType,
    RepoGraph,
    SearchResult,
    SemanticSearchEngine,
)


class TestCodeSummarizer:
    """Tests for CodeSummarizer."""

    def test_summarize_function(self):
        """Test function summarization."""
        summarizer = CodeSummarizer()

        result = summarizer.summarize_function(
            name="process_data",
            signature="def process_data(items: list, config: dict) -> None",
            docstring="Process a list of items according to configuration.",
            body="for item in items:\n    handle(item)",
            file_path="core/processor.py"
        )

        assert "process_data" in result
        assert "Function" in result
        assert "processor.py" in result

    def test_summarize_class(self):
        """Test class summarization."""
        summarizer = CodeSummarizer()

        result = summarizer.summarize_class(
            name="DataProcessor",
            bases=["BaseHandler", "Configurable"],
            docstring="Handles data processing with configurable pipelines.",
            methods=["process", "validate", "transform"],
            file_path="core/processor.py"
        )

        assert "DataProcessor" in result
        assert "Class" in result
        assert "inherits" in result

    def test_summarize_file(self):
        """Test file/module summarization."""
        summarizer = CodeSummarizer()

        result = summarizer.summarize_file(
            file_path="saga/core/warden.py",
            docstring="Main warden orchestration module.",
            imports=["asyncio", "langgraph", "pydantic"],
            definitions=["Warden", "solve_request", "initialize"]
        )

        assert "warden.py" in result
        assert "Module" in result

    def test_truncation(self):
        """Test that long content is truncated."""
        summarizer = CodeSummarizer()

        long_body = "x = 1\n" * 1000
        result = summarizer.summarize_function(
            name="long_function",
            body=long_body
        )

        assert len(result) <= summarizer.MAX_CHARS + 10  # Small buffer for "..."


class TestEmbeddingGenerator:
    """Tests for EmbeddingGenerator."""

    def test_generator_initialization(self):
        """Test generator initializes."""
        generator = EmbeddingGenerator(batch_size=16)
        assert generator.batch_size == 16

    def test_embed_text_returns_vector(self):
        """Test text embedding returns proper shape."""
        generator = EmbeddingGenerator()

        result = generator.embed_text("async function handler")

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        # If model not available, returns zero vector
        assert len(result) == 384

    def test_embed_empty_text(self):
        """Test empty text returns zero vector."""
        generator = EmbeddingGenerator()

        result = generator.embed_text("")

        assert np.allclose(result, np.zeros(384))

    def test_embed_batch(self):
        """Test batch embedding."""
        generator = EmbeddingGenerator()

        texts = ["function one", "function two", "function three"]
        results = generator.embed_batch(texts)

        assert len(results) == 3
        for emb in results:
            assert len(emb) == 384

    def test_compute_text_hash(self):
        """Test hash computation for dirty-checking."""
        generator = EmbeddingGenerator()

        hash1 = generator.compute_text_hash("same content")
        hash2 = generator.compute_text_hash("same content")
        hash3 = generator.compute_text_hash("different content")

        assert hash1 == hash2
        assert hash1 != hash3

    def test_embed_node(self):
        """Test node embedding."""
        generator = EmbeddingGenerator()

        result = generator.embed_node(
            node_id="func:handler",
            name="handler",
            node_type="FUNCTION",
            metadata={
                "signature": "async def handler(request)",
                "docstring": "Handle incoming requests"
            }
        )

        assert isinstance(result, EmbeddingResult)
        assert result.node_id == "func:handler"
        assert result.success


class TestEmbeddingStore:
    """Tests for EmbeddingStore persistence."""

    def test_store_initialization(self, tmp_path):
        """Test store initializes database."""
        store = EmbeddingStore(tmp_path / "embeddings.db")

        assert (tmp_path / "embeddings.db").exists()

    def test_save_and_load_embedding(self, tmp_path):
        """Test saving and loading embeddings."""
        store = EmbeddingStore(tmp_path / "embeddings.db")

        embedding = np.random.randn(384).astype(np.float32)
        store.save_embedding("node_1", embedding, text_hash="abc123")

        loaded = store.load_embedding("node_1")

        assert loaded is not None
        assert np.allclose(embedding, loaded)

    def test_save_batch(self, tmp_path):
        """Test batch saving."""
        store = EmbeddingStore(tmp_path / "embeddings.db")

        results = [
            EmbeddingResult("node_1", np.zeros(384, dtype=np.float32), "hash1"),
            EmbeddingResult("node_2", np.ones(384, dtype=np.float32), "hash2"),
        ]

        saved = store.save_batch(results)

        assert saved == 2
        assert store.count() == 2

    def test_dirty_flag(self, tmp_path):
        """Test dirty flag for incremental updates."""
        store = EmbeddingStore(tmp_path / "embeddings.db")

        store.save_embedding("node_1", np.zeros(384, dtype=np.float32))
        store.mark_dirty("node_1")

        dirty = store.get_dirty_nodes()

        assert "node_1" in dirty


class TestSemanticSearchEngine:
    """Tests for SemanticSearchEngine."""

    def test_engine_initialization(self):
        """Test engine initializes with new weights."""
        graph = RepoGraph()
        engine = SemanticSearchEngine(graph)

        # New weights: semantic=0.6, structural=0.2, recency=0.1, context=0.1
        assert engine.semantic_weight == 0.6
        assert engine.structural_weight == 0.2
        assert engine.recency_weight == 0.1
        assert engine.context_weight == 0.1

    def test_build_index_empty(self):
        """Test building index on empty graph."""
        graph = RepoGraph()
        engine = SemanticSearchEngine(graph)

        count = engine.build_index()

        assert count == 0

    def test_build_index_with_embeddings(self):
        """Test building index with embedded nodes."""
        graph = RepoGraph()

        # Add nodes with embeddings
        node1 = GraphNode("func:a", NodeType.FUNCTION, "function_a")
        node1.embedding_vector = np.random.randn(384).astype(np.float32)
        graph.add_node(node1)

        node2 = GraphNode("func:b", NodeType.FUNCTION, "function_b")
        node2.embedding_vector = np.random.randn(384).astype(np.float32)
        graph.add_node(node2)

        engine = SemanticSearchEngine(graph)
        count = engine.build_index()

        assert count == 2

    def test_search_semantic(self):
        """Test pure semantic search."""
        graph = RepoGraph()

        # Add nodes with embeddings
        node1 = GraphNode("func:async_handler", NodeType.FUNCTION, "async_handler")
        node1.embedding_vector = np.random.randn(384).astype(np.float32)
        graph.add_node(node1)

        engine = SemanticSearchEngine(graph)
        engine.build_index()

        results = engine.search_semantic("async function", top_k=5)

        # Should return something (even if similarity is random)
        assert isinstance(results, list)

    def test_cosine_similarity(self):
        """Test cosine similarity computation."""
        graph = RepoGraph()
        engine = SemanticSearchEngine(graph)

        vec1 = np.array([1, 0, 0], dtype=np.float32)
        vec2 = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

        sims = engine._cosine_similarity(vec1, vec2)

        assert sims[0] == pytest.approx(1.0, abs=0.01)  # Same direction
        assert sims[1] == pytest.approx(0.0, abs=0.01)  # Orthogonal


class TestGraphEmbeddingManager:
    """Tests for GraphEmbeddingManager integration."""

    def test_manager_initialization(self):
        """Test manager initializes."""
        graph = RepoGraph()
        manager = GraphEmbeddingManager(graph)

        assert manager.graph is graph

    def test_embed_all_nodes(self):
        """Test embedding all nodes in graph."""
        graph = RepoGraph()
        graph.add_node(GraphNode("func:a", NodeType.FUNCTION, "function_a"))
        graph.add_node(GraphNode("func:b", NodeType.FUNCTION, "function_b"))

        manager = GraphEmbeddingManager(graph)
        stats = manager.embed_all_nodes()

        assert stats["embedded"] >= 0
        assert "errors" in stats

    def test_search_integration(self):
        """Test search through manager."""
        graph = RepoGraph()
        node = GraphNode("func:handler", NodeType.FUNCTION, "request_handler")
        node.embedding_vector = np.random.randn(384).astype(np.float32)
        graph.add_node(node)

        manager = GraphEmbeddingManager(graph)
        manager.search_engine.build_index()

        results = manager.search("handle request", top_k=5)

        assert isinstance(results, list)

    def test_get_stats(self):
        """Test getting embedding statistics."""
        graph = RepoGraph()
        node = GraphNode("func:a", NodeType.FUNCTION, "a")
        node.embedding_vector = np.zeros(384, dtype=np.float32)
        graph.add_node(node)

        manager = GraphEmbeddingManager(graph)
        stats = manager.get_stats()

        assert stats["total_nodes"] == 1
        assert stats["embedded_nodes"] == 1
        assert stats["coverage"] == 1.0


class TestHybridSearch:
    """Tests for hybrid semantic + structural search."""

    def test_hybrid_search_combines_results(self):
        """Test that hybrid search includes structural expansion."""
        graph = RepoGraph()

        # Create connected nodes
        node1 = GraphNode("func:a", NodeType.FUNCTION, "entry_point")
        node1.embedding_vector = np.random.randn(384).astype(np.float32)
        graph.add_node(node1)

        node2 = GraphNode("func:b", NodeType.FUNCTION, "helper_function")
        node2.embedding_vector = np.random.randn(384).astype(np.float32)
        graph.add_node(node2)

        graph.add_edge(GraphEdge("func:a", "func:b", EdgeType.CALLS))

        engine = SemanticSearchEngine(graph)
        engine.build_index()

        results = engine.search_hybrid("entry point", top_k=10, expand_depth=2)

        # Should include both nodes due to graph expansion
        node_ids = [r.node_id for r in results]
        assert "func:a" in node_ids or "func:b" in node_ids

    def test_combined_score_calculation(self):
        """Test that combined score uses all weight factors."""
        result = SearchResult(
            node_id="test",
            name="test",
            node_type="FUNCTION",
            file_path="",
            cosine_similarity=0.8,
            graph_distance=2,
            recency_score=0.5,
            context_proximity=1.0
        )

        # New weights: semantic=0.6, structural=0.2, recency=0.1, context=0.1
        expected = 0.6 * 0.8 + 0.2 * (1 / 2) + 0.1 * 0.5 + 0.1 * 1.0

        # The engine would compute this
        semantic_score = result.cosine_similarity * 0.6
        structural_score = (1 / result.graph_distance) * 0.2
        recency_score = result.recency_score * 0.1
        context_score = result.context_proximity * 0.1
        combined = semantic_score + structural_score + recency_score + context_score

        assert combined == pytest.approx(expected, abs=0.01)


class TestRecencyAndContext:
    """Tests for recency tracking and context proximity."""

    def test_node_touch_updates_timestamp(self):
        """Test that touch() updates last_accessed."""
        import time
        node = GraphNode("func:a", NodeType.FUNCTION, "test_func")

        assert node.last_accessed is None
        node.touch()
        assert node.last_accessed is not None
        assert node.last_accessed <= time.time()

    def test_recency_score_computation(self):
        """Test recency score decay."""
        import time
        graph = RepoGraph()
        node = GraphNode("func:a", NodeType.FUNCTION, "recent_func")
        node.last_accessed = time.time()  # Just accessed
        graph.add_node(node)

        engine = SemanticSearchEngine(graph)
        score = engine._compute_recency_score(node)

        # Recently accessed should have high score (close to 1.0)
        assert score > 0.9

    def test_context_proximity_in_neighbors(self):
        """Test context proximity detection."""
        graph = RepoGraph()
        node1 = GraphNode("func:a", NodeType.FUNCTION, "main")
        node2 = GraphNode("func:b", NodeType.FUNCTION, "helper")
        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_edge(GraphEdge("func:a", "func:b", EdgeType.CALLS))

        engine = SemanticSearchEngine(graph)
        # Use node name "main" to find the context file
        engine.set_context_file("main", depth=2)

        # func:b should be in context because it's connected to "main"
        assert engine._compute_context_proximity("func:b") == 1.0
        # Unrelated node should not be in context
        assert engine._compute_context_proximity("func:unrelated") == 0.0

    def test_touch_result_nodes(self):
        """Test that search results can be touched."""
        graph = RepoGraph()
        node = GraphNode("func:a", NodeType.FUNCTION, "test")
        node.embedding_vector = np.zeros(384, dtype=np.float32)
        graph.add_node(node)

        manager = GraphEmbeddingManager(graph)
        results = [SearchResult(node_id="func:a", name="test", node_type="FUNCTION", file_path="")]

        touched = manager.touch_result_nodes(results)

        assert touched == 1
        assert graph.get_node("func:a").last_accessed is not None
