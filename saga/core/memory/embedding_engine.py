"""
Embedding Engine - Semantic Vector Generation & Retrieval
==========================================================

Generates embeddings for RepoGraph nodes using sentence-transformers
and enables hybrid semantic + structural retrieval.

Author: ARC SAGA Development Team (AntiGravity)
Date: December 26, 2025
Status: USMA P0 Fix - Semantic Retrieval Integration
"""

import base64
import hashlib
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Lazy import for sentence-transformers (heavy dependency)
_EMBEDDING_MODEL = None
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension


def get_embedding_model():
    """Lazy-load the embedding model (22MB, CPU-friendly)."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBEDDING_MODEL = SentenceTransformer(_MODEL_NAME)
            logger.info(f"Loaded embedding model: {_MODEL_NAME}")
        except ImportError:
            logger.warning("sentence-transformers not installed. Run: pip install sentence-transformers")
            return None
    return _EMBEDDING_MODEL


# ═══════════════════════════════════════════════════════════════
# INPUT SUMMARIZATION
# ═══════════════════════════════════════════════════════════════

class CodeSummarizer:
    """
    Summarizes code entities for embedding generation.

    Creates a semantic-rich text representation from:
    - Function/class signature
    - Docstring
    - Code body excerpt (first/last N chars)
    """

    MAX_TOKENS = 512  # Model input limit
    CHARS_PER_TOKEN = 4  # Rough estimate
    MAX_CHARS = MAX_TOKENS * CHARS_PER_TOKEN  # ~2048 chars

    def summarize_function(
        self,
        name: str,
        signature: str = "",
        docstring: str = "",
        body: str = "",
        file_path: str = ""
    ) -> str:
        """
        Create embedding-ready summary for a function.

        Format:
        [Function] name in file_path
        Signature: def name(args)
        Purpose: docstring
        Code: first_100 ... last_100
        """
        parts = [f"[Function] {name}"]

        if file_path:
            parts.append(f"in {Path(file_path).name}")

        if signature:
            parts.append(f"\nSignature: {signature[:200]}")

        if docstring:
            parts.append(f"\nPurpose: {docstring[:300]}")

        if body:
            body_excerpt = self._excerpt_body(body, max_chars=400)
            parts.append(f"\nCode: {body_excerpt}")

        summary = " ".join(parts)
        return self._truncate(summary)

    def summarize_class(
        self,
        name: str,
        bases: list[str] | None = None,
        docstring: str = "",
        methods: list[str] | None = None,
        file_path: str = ""
    ) -> str:
        """Create embedding-ready summary for a class."""
        parts = [f"[Class] {name}"]

        if bases:
            parts.append(f"inherits {', '.join(bases[:3])}")

        if file_path:
            parts.append(f"in {Path(file_path).name}")

        if docstring:
            parts.append(f"\nPurpose: {docstring[:300]}")

        if methods:
            parts.append(f"\nMethods: {', '.join(methods[:10])}")

        return self._truncate(" ".join(parts))

    def summarize_file(
        self,
        file_path: str,
        docstring: str = "",
        imports: list[str] | None = None,
        definitions: list[str] | None = None
    ) -> str:
        """Create embedding-ready summary for a file/module."""
        parts = [f"[Module] {Path(file_path).name}"]

        if docstring:
            parts.append(f"\nPurpose: {docstring[:300]}")

        if imports:
            parts.append(f"\nImports: {', '.join(imports[:10])}")

        if definitions:
            parts.append(f"\nDefines: {', '.join(definitions[:15])}")

        return self._truncate(" ".join(parts))

    def summarize_generic(self, name: str, node_type: str, metadata: dict[str, Any] | None = None) -> str:
        """Fallback summarizer for any node type."""
        parts = [f"[{node_type}] {name}"]

        if metadata:
            if "docstring" in metadata:
                parts.append(f"\n{metadata['docstring'][:300]}")
            if "signature" in metadata:
                parts.append(f"\n{metadata['signature'][:200]}")

        return self._truncate(" ".join(parts))

    def _excerpt_body(self, body: str, max_chars: int = 400) -> str:
        """Extract first/last portions of code body."""
        body = body.strip()
        if len(body) <= max_chars:
            return body

        half = max_chars // 2
        return f"{body[:half]} ... {body[-half:]}"

    def _truncate(self, text: str) -> str:
        """Truncate to max chars."""
        if len(text) <= self.MAX_CHARS:
            return text
        return text[:self.MAX_CHARS - 3] + "..."


# ═══════════════════════════════════════════════════════════════
# EMBEDDING GENERATOR
# ═══════════════════════════════════════════════════════════════

@dataclass
class EmbeddingResult:
    """Result of embedding a node."""
    node_id: str
    embedding: np.ndarray
    text_hash: str  # For dirty-checking
    success: bool = True
    error: str | None = None


class EmbeddingGenerator:
    """
    Generates embeddings for code entities using sentence-transformers.

    Features:
    - Batch processing for efficiency (32-64 nodes at a time)
    - Dirty-checking via content hash
    - Zero-vector fallback for empty content
    """

    def __init__(self, batch_size: int = 32):
        """Initialize the embedding generator."""
        self.batch_size = batch_size
        self.summarizer = CodeSummarizer()
        self._model = None

    @property
    def model(self):
        """Lazy-load model."""
        if self._model is None:
            self._model = get_embedding_model()
        return self._model

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Returns:
            384-dimensional vector as numpy array
        """
        if not text or not text.strip():
            return np.zeros(_EMBEDDING_DIM, dtype=np.float32)

        if self.model is None:
            logger.warning("Embedding model not available, returning zero vector")
            return np.zeros(_EMBEDDING_DIM, dtype=np.float32)

        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings

        Returns:
            List of 384-dimensional vectors
        """
        if not texts:
            return []

        if self.model is None:
            return [np.zeros(_EMBEDDING_DIM, dtype=np.float32) for _ in texts]

        # Handle empty texts
        processed = [t if t and t.strip() else "[empty]" for t in texts]

        embeddings = self.model.encode(
            processed,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100
        )

        # Zero out embeddings for empty texts
        result = []
        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            if not text or not text.strip():
                result.append(np.zeros(_EMBEDDING_DIM, dtype=np.float32))
            else:
                result.append(emb.astype(np.float32))

        return result

    def compute_text_hash(self, text: str) -> str:
        """Compute hash for dirty-checking."""
        return hashlib.md5(text.encode()).hexdigest()[:16]

    def embed_node(
        self,
        node_id: str,
        name: str,
        node_type: str,
        metadata: dict[str, Any] | None = None,
        file_path: str = ""
    ) -> EmbeddingResult:
        """
        Generate embedding for a single graph node.

        Args:
            node_id: Unique node identifier
            name: Node name (function/class/file name)
            node_type: Type (FUNCTION, CLASS, FILE, MODULE)
            metadata: Optional metadata dict
            file_path: Source file path

        Returns:
            EmbeddingResult with vector and hash
        """
        try:
            # Generate summary based on type
            metadata = metadata or {}

            if node_type == "FUNCTION":
                text = self.summarizer.summarize_function(
                    name=name,
                    signature=metadata.get("signature", ""),
                    docstring=metadata.get("docstring", ""),
                    body=metadata.get("body", ""),
                    file_path=file_path
                )
            elif node_type == "CLASS":
                text = self.summarizer.summarize_class(
                    name=name,
                    bases=metadata.get("bases", []),
                    docstring=metadata.get("docstring", ""),
                    methods=metadata.get("methods", []),
                    file_path=file_path
                )
            elif node_type in ("FILE", "MODULE"):
                text = self.summarizer.summarize_file(
                    file_path=file_path or name,
                    docstring=metadata.get("docstring", ""),
                    imports=metadata.get("imports", []),
                    definitions=metadata.get("definitions", [])
                )
            else:
                text = self.summarizer.summarize_generic(name, node_type, metadata)

            embedding = self.embed_text(text)
            text_hash = self.compute_text_hash(text)

            return EmbeddingResult(
                node_id=node_id,
                embedding=embedding,
                text_hash=text_hash,
                success=True
            )
        except Exception as e:
            logger.error(f"Failed to embed node {node_id}: {e}")
            return EmbeddingResult(
                node_id=node_id,
                embedding=np.zeros(_EMBEDDING_DIM, dtype=np.float32),
                text_hash="",
                success=False,
                error=str(e)
            )


# ═══════════════════════════════════════════════════════════════
# SEMANTIC SEARCH ENGINE
# ═══════════════════════════════════════════════════════════════

@dataclass
class SearchResult:
    """A single search result."""
    node_id: str
    name: str
    node_type: str
    file_path: str

    # Scores
    cosine_similarity: float = 0.0
    graph_distance: int = -1  # -1 = not reachable
    recency_score: float = 0.0  # Higher = more recently accessed
    context_proximity: float = 0.0  # Higher = closer to active file
    combined_score: float = 0.0

    # Provenance
    matched_via: str = "semantic"  # "semantic", "structural", "hybrid"

    # Guardrail transparency (NEW)
    guardrail_violations: list[Any] | None = None  # list[Violation] if guardrails enabled
    guardrail_penalty_applied: float | None = None  # Penalty multiplier applied


class SemanticSearchEngine:
    """
    Hybrid semantic + structural search on RepoGraph.

    Algorithm:
    1. Embed query text
    2. Compute cosine similarity to all node embeddings
    3. Get top-k semantic matches
    4. Expand via graph traversal (ego-graph or shortest paths)
    5. Fuse scores: 0.7 * cosine + 0.3 * (1 / graph_distance)
    """

    def __init__(
        self,
        graph: Any,  # RepoGraph
        semantic_weight: float = 0.6,
        structural_weight: float = 0.2,
        recency_weight: float = 0.1,
        context_weight: float = 0.1,
        guardrail: Any = None  # MimiryGuardrail (NEW)
    ):
        """Initialize the search engine."""
        self.graph = graph
        self.semantic_weight = semantic_weight
        self.structural_weight = structural_weight
        self.recency_weight = recency_weight
        self.context_weight = context_weight
        self.generator = EmbeddingGenerator()
        self.guardrail = guardrail  # NEW: Guardrail instance

        # Cache embeddings as numpy matrix for fast search
        self._embedding_matrix: np.ndarray | None = None
        self._node_ids: list[str] = []
        self._is_indexed = False

        # Context for current search session
        self._context_neighbors: set[str] = set()

    def build_index(self) -> int:
        """
        Build the embedding index from graph nodes.

        Returns:
            Number of nodes indexed
        """
        embeddings = []
        node_ids = []

        for node_id, node in self.graph._node_index.items():
            if hasattr(node, 'embedding_vector') and node.embedding_vector is not None:
                if isinstance(node.embedding_vector, np.ndarray):
                    emb = node.embedding_vector
                elif isinstance(node.embedding_vector, list):
                    emb = np.array(node.embedding_vector, dtype=np.float32)
                else:
                    continue

                if len(emb) == _EMBEDDING_DIM:
                    embeddings.append(emb)
                    node_ids.append(node_id)

        if embeddings:
            self._embedding_matrix = np.vstack(embeddings)
            self._node_ids = node_ids
            self._is_indexed = True
            logger.info(f"Built search index with {len(node_ids)} nodes")
        else:
            self._embedding_matrix = np.zeros((0, _EMBEDDING_DIM), dtype=np.float32)
            self._node_ids = []
            self._is_indexed = True

        return len(node_ids)

    def _cosine_similarity(self, query_vec: np.ndarray, corpus_matrix: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and all corpus vectors.

        Uses efficient numpy operations.
        """
        if corpus_matrix.shape[0] == 0:
            return np.array([])

        # Normalize query
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)

        # Normalize corpus (row-wise)
        corpus_norms = np.linalg.norm(corpus_matrix, axis=1, keepdims=True) + 1e-8
        corpus_normalized = corpus_matrix / corpus_norms

        # Dot product = cosine similarity for normalized vectors
        similarities = corpus_normalized @ query_norm

        return similarities

    def search_semantic(
        self,
        query: str,
        top_k: int = 10
    ) -> list[SearchResult]:
        """
        Pure semantic search using embeddings.

        Args:
            query: Natural language query
            top_k: Number of results to return

        Returns:
            List of SearchResult ordered by similarity
        """
        if not self._is_indexed:
            self.build_index()

        if self._embedding_matrix is None or self._embedding_matrix.shape[0] == 0:
            return []

        # Embed query
        query_embedding = self.generator.embed_text(query)

        # Compute similarities
        similarities = self._cosine_similarity(query_embedding, self._embedding_matrix)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            node_id = self._node_ids[idx]
            node = self.graph.get_node(node_id)

            if node:
                results.append(SearchResult(
                    node_id=node_id,
                    name=node.name,
                    node_type=node.node_type.value if hasattr(node.node_type, 'value') else str(node.node_type),
                    file_path=node.file_path or "",
                    cosine_similarity=float(similarities[idx]),
                    combined_score=float(similarities[idx]),
                    matched_via="semantic"
                ))

        return results

    def set_context_file(self, file_path: str, depth: int = 2) -> int:
        """
        Set the active file context for proximity boosting.

        Args:
            file_path: Path to the currently active file
            depth: Neighborhood depth

        Returns:
            Number of neighbors in context
        """
        self._context_neighbors.clear()

        # Find the file node
        file_node_id = None
        for node_id, node in self.graph._node_index.items():
            if node.file_path == file_path or node.name == file_path:
                file_node_id = node_id
                break

        if not file_node_id:
            return 0

        # Get neighborhood
        subgraph = self.graph.get_related_subgraph(file_node_id, depth=depth)
        self._context_neighbors = set(subgraph._node_index.keys())

        logger.debug(f"Set context with {len(self._context_neighbors)} neighbors")
        return len(self._context_neighbors)

    def _compute_recency_score(self, node: Any) -> float:
        """
        Compute recency score for a node.

        Returns:
            Score between 0 and 1 (higher = more recent)
        """
        import time

        if not hasattr(node, 'last_accessed') or node.last_accessed is None:
            return 0.0

        age_seconds = time.time() - node.last_accessed
        age_days = age_seconds / 86400  # Convert to days

        # Decay function: 1 / (1 + age_days)
        # Recent = 1.0, 1 day old = 0.5, 7 days = 0.125
        return 1.0 / (1.0 + age_days)

    def _compute_context_proximity(self, node_id: str) -> float:
        """
        Compute context proximity score.

        Returns:
            1.0 if in context, 0.0 otherwise
        """
        return 1.0 if node_id in self._context_neighbors else 0.0

    def _get_node_content(self, node: Any) -> str:
        """Extract text content from a graph node for guardrail validation."""
        if not node:
            return ""

        # Try to get source snippet or docstring
        content_parts = []

        if hasattr(node, 'source_snippet') and node.source_snippet:
            content_parts.append(node.source_snippet)
        elif hasattr(node, 'docstring') and node.docstring:
            content_parts.append(node.docstring)

        if hasattr(node, 'name'):
            content_parts.append(f"name: {node.name}")

        if hasattr(node, 'metadata') and isinstance(node.metadata, dict):
            # Add relevant metadata
            if 'signature' in node.metadata:
                content_parts.append(f"signature: {node.metadata['signature']}")
            if 'body' in node.metadata:
                content_parts.append(node.metadata['body'][:500])  # First 500 chars

        return "\n".join(content_parts) if content_parts else ""


    def search_hybrid(
        self,
        query: str,
        top_k: int = 10,
        expand_depth: int = 2,
        context_file: str | None = None
    ) -> list[SearchResult]:
        """
        Hybrid semantic + structural search with recency and context.

        Algorithm:
        1. Get top-k/2 semantic matches
        2. Expand each via graph traversal (up to expand_depth)
        3. Fuse scores:
           - semantic_weight * cosine
           - structural_weight * (1 / distance)
           - recency_weight * recency_score
           - context_weight * context_proximity

        Args:
            query: Natural language query
            top_k: Total results to return
            expand_depth: Graph expansion depth
            context_file: Optional active file for proximity boosting

        Returns:
            List of SearchResult ordered by combined score
        """
        # Get semantic matches
        semantic_results = self.search_semantic(query, top_k=max(top_k // 2, 3))

        if not semantic_results:
            return []

        # Collect all candidates with scores
        candidates: dict[str, SearchResult] = {}

        # Add semantic matches
        for result in semantic_results:
            candidates[result.node_id] = result

        # Expand via graph
        import networkx as nx
        undirected = self.graph.graph.to_undirected()

        for result in semantic_results:
            try:
                # Get ego graph (local neighborhood)
                ego = nx.ego_graph(undirected, result.node_id, radius=expand_depth)

                for neighbor_id in ego.nodes():
                    if neighbor_id == result.node_id:
                        continue

                    # Calculate graph distance
                    try:
                        distance = nx.shortest_path_length(undirected, result.node_id, neighbor_id)
                    except nx.NetworkXNoPath:
                        distance = expand_depth + 1

                    if neighbor_id not in candidates:
                        node = self.graph.get_node(neighbor_id)
                        if node:
                            # Compute semantic similarity for expanded node
                            if hasattr(node, 'embedding_vector') and node.embedding_vector is not None:
                                if neighbor_id in self._node_ids:
                                    idx = self._node_ids.index(neighbor_id)
                                    query_emb = self.generator.embed_text(query)
                                    cos_sim = float(self._cosine_similarity(
                                        query_emb,
                                        self._embedding_matrix[idx:idx+1]
                                    )[0])
                                else:
                                    cos_sim = 0.0
                            else:
                                cos_sim = 0.0

                            candidates[neighbor_id] = SearchResult(
                                node_id=neighbor_id,
                                name=node.name,
                                node_type=node.node_type.value if hasattr(node.node_type, 'value') else str(node.node_type),
                                file_path=node.file_path or "",
                                cosine_similarity=cos_sim,
                                graph_distance=distance,
                                matched_via="structural"
                            )
                    else:
                        # Update distance if shorter
                        if distance < candidates[neighbor_id].graph_distance or candidates[neighbor_id].graph_distance < 0:
                            candidates[neighbor_id].graph_distance = distance
            except Exception as e:
                logger.debug(f"Graph expansion failed for {result.node_id}: {e}")

        # Set context if provided
        if context_file:
            self.set_context_file(context_file)

        # Compute combined scores with all factors
        for result in candidates.values():
            node = self.graph.get_node(result.node_id)

            # Semantic score
            semantic_score = result.cosine_similarity * self.semantic_weight

            # Structural score (graph distance)
            if result.graph_distance > 0:
                structural_score = (1.0 / result.graph_distance) * self.structural_weight
            elif result.graph_distance == 0:
                structural_score = self.structural_weight
            else:
                structural_score = 0.0

            # Recency score
            if node:
                result.recency_score = self._compute_recency_score(node)
            recency_score = result.recency_score * self.recency_weight

            # Context proximity score
            result.context_proximity = self._compute_context_proximity(result.node_id)
            context_score = result.context_proximity * self.context_weight

            # Combined score
            result.combined_score = semantic_score + structural_score + recency_score + context_score

            if result.matched_via == "semantic" and result.graph_distance >= 0:
                result.matched_via = "hybrid"

        # NEW: Apply Mimiry guardrails after scoring, before final ranking
        if self.guardrail:
            candidates_list = list(candidates.values())

            # Prepare candidates for validation
            validation_input = []
            for candidate in candidates_list:
                node = self.graph.get_node(candidate.node_id)
                content = self._get_node_content(node) if node else ""
                validation_input.append({
                    "node_id": candidate.node_id,
                    "content": content,
                    "score": candidate.combined_score
                })

            # Batch validate
            guardrail_results = self.guardrail.validate_batch(
                candidates=validation_input,
                query_context={"query": query, "context_file": context_file}
            )

            # Apply penalties and exclusions
            for candidate, guard_result in zip(candidates_list, guardrail_results, strict=True):
                if guard_result.action == "exclude":
                    # Hard exclude from results
                    candidates.pop(candidate.node_id, None)

                    # Trigger Warden veto if critical
                    if guard_result.warden_veto_triggered:
                        logger.warning(
                            f"Critical violation - node excluded: {candidate.node_id} - "
                            f"{guard_result.veto_reason}"
                        )

                elif guard_result.action == "downrank":
                    # Apply penalty to score
                    candidate.combined_score *= guard_result.penalty_multiplier
                    candidate.guardrail_violations = guard_result.violations
                    candidate.guardrail_penalty_applied = guard_result.penalty_multiplier
                # "allow" = no change

        # Sort by combined score
        sorted_results = sorted(candidates.values(), key=lambda r: r.combined_score, reverse=True)

        return sorted_results[:top_k]


# ═══════════════════════════════════════════════════════════════
# PERSISTENCE
# ═══════════════════════════════════════════════════════════════

class EmbeddingStore:
    """
    SQLite-based persistence for node embeddings.

    Features:
    - Base64 encoded numpy arrays for efficient storage
    - Dirty flag for incremental updates
    - Content hash for change detection
    """

    def __init__(self, db_path: Path | str):
        """Initialize the embedding store."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    node_id TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    text_hash TEXT,
                    updated_at TEXT,
                    dirty INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_dirty ON embeddings(dirty)
            """)
            conn.commit()

    def save_embedding(
        self,
        node_id: str,
        embedding: np.ndarray,
        text_hash: str = ""
    ) -> None:
        """Save a single embedding."""
        blob = self._encode_embedding(embedding)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO embeddings (node_id, embedding, text_hash, updated_at, dirty)
                VALUES (?, ?, ?, datetime('now'), 0)
            """, (node_id, blob, text_hash))
            conn.commit()

    def save_batch(self, results: list[EmbeddingResult]) -> int:
        """Save a batch of embeddings."""
        saved = 0
        with sqlite3.connect(self.db_path) as conn:
            for result in results:
                if result.success:
                    blob = self._encode_embedding(result.embedding)
                    conn.execute("""
                        INSERT OR REPLACE INTO embeddings (node_id, embedding, text_hash, updated_at, dirty)
                        VALUES (?, ?, ?, datetime('now'), 0)
                    """, (result.node_id, blob, result.text_hash))
                    saved += 1
            conn.commit()
        return saved

    def load_embedding(self, node_id: str) -> np.ndarray | None:
        """Load a single embedding."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT embedding FROM embeddings WHERE node_id = ?",
                (node_id,)
            )
            row = cursor.fetchone()
            if row:
                return self._decode_embedding(row[0])
        return None

    def load_all(self) -> dict[str, np.ndarray]:
        """Load all embeddings as a dict."""
        embeddings = {}
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT node_id, embedding FROM embeddings")
            for node_id, blob in cursor.fetchall():
                embeddings[node_id] = self._decode_embedding(blob)
        return embeddings

    def get_text_hash(self, node_id: str) -> str | None:
        """Get stored text hash for dirty-checking."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT text_hash FROM embeddings WHERE node_id = ?",
                (node_id,)
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def mark_dirty(self, node_id: str) -> None:
        """Mark a node as needing re-embedding."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE embeddings SET dirty = 1 WHERE node_id = ?",
                (node_id,)
            )
            conn.commit()

    def get_dirty_nodes(self) -> list[str]:
        """Get list of nodes needing re-embedding."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT node_id FROM embeddings WHERE dirty = 1")
            return [row[0] for row in cursor.fetchall()]

    def count(self) -> int:
        """Get count of stored embeddings."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
            return cursor.fetchone()[0]

    def _encode_embedding(self, embedding: np.ndarray) -> bytes:
        """Encode numpy array as base64 bytes."""
        return base64.b64encode(embedding.tobytes())

    def _decode_embedding(self, blob: bytes) -> np.ndarray:
        """Decode base64 bytes to numpy array."""
        raw = base64.b64decode(blob)
        return np.frombuffer(raw, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
# GRAPH INTEGRATION
# ═══════════════════════════════════════════════════════════════

class GraphEmbeddingManager:
    """
    Manages embedding generation and population for RepoGraph.

    Orchestrates:
    - Batch embedding generation
    - Graph node population
    - Persistence via EmbeddingStore
    - Incremental updates via dirty-checking
    """

    def __init__(
        self,
        graph: Any,  # RepoGraph
        store: EmbeddingStore | None = None,
        batch_size: int = 32,
        mimiry_guardrail: Any = None,  # MimiryGuardrail
        guardrails_enabled: bool = True  # NEW: Default ON
    ):
        """Initialize the manager."""
        self.graph = graph
        self.store = store
        self.generator = EmbeddingGenerator(batch_size=batch_size)
        self.search_engine = SemanticSearchEngine(graph, guardrail=mimiry_guardrail)
        self.guardrail = mimiry_guardrail  # Store reference
        self.guardrails_enabled = guardrails_enabled  # Feature flag

    def embed_all_nodes(self, force: bool = False) -> dict[str, Any]:
        """
        Generate embeddings for all graph nodes.

        Args:
            force: Re-embed even if content unchanged

        Returns:
            Statistics dict
        """
        nodes_to_embed = []

        for node_id, node in self.graph._node_index.items():
            # Check if needs embedding
            needs_embedding = force

            if not needs_embedding and self.store:
                stored_hash = self.store.get_text_hash(node_id)
                # Generate current hash
                metadata = node.metadata if hasattr(node, 'metadata') else {}
                result = self.generator.embed_node(
                    node_id=node_id,
                    name=node.name,
                    node_type=node.node_type.value if hasattr(node.node_type, 'value') else str(node.node_type),
                    metadata=metadata,
                    file_path=node.file_path or ""
                )
                if stored_hash != result.text_hash:
                    needs_embedding = True
            elif not self.store:
                needs_embedding = True

            if needs_embedding or not hasattr(node, 'embedding_vector') or node.embedding_vector is None:
                nodes_to_embed.append(node)

        if not nodes_to_embed:
            return {"embedded": 0, "skipped": len(self.graph._node_index), "errors": 0}

        # Batch generate
        results: list[EmbeddingResult] = []
        for node in nodes_to_embed:
            metadata = node.metadata if hasattr(node, 'metadata') else {}
            result = self.generator.embed_node(
                node_id=node.id,
                name=node.name,
                node_type=node.node_type.value if hasattr(node.node_type, 'value') else str(node.node_type),
                metadata=metadata,
                file_path=node.file_path or ""
            )
            results.append(result)

            # Populate graph node
            if result.success:
                node.embedding_vector = result.embedding

        # Persist
        if self.store:
            self.store.save_batch(results)

        # Rebuild search index
        self.search_engine.build_index()

        errors = sum(1 for r in results if not r.success)
        return {
            "embedded": len(results) - errors,
            "skipped": len(self.graph._node_index) - len(nodes_to_embed),
            "errors": errors
        }

    def load_embeddings_from_store(self) -> int:
        """Load persisted embeddings into graph nodes."""
        if not self.store:
            return 0

        loaded = 0
        embeddings = self.store.load_all()

        for node_id, embedding in embeddings.items():
            node = self.graph.get_node(node_id)
            if node:
                node.embedding_vector = embedding
                loaded += 1

        # Build search index
        self.search_engine.build_index()

        return loaded

    def search(
        self,
        query: str,
        top_k: int = 10,
        hybrid: bool = True,
        apply_guardrails: bool | None = None  # NEW: None = use instance default
    ) -> list[SearchResult]:
        """
        Search the graph.

        Args:
            query: Natural language query
            top_k: Number of results
            hybrid: Use hybrid semantic+structural search
            apply_guardrails: Override guardrail setting (None = use default)

        Returns:
            List of SearchResult
        """
        # Use instance default if not overridden
        if apply_guardrails is None:
            apply_guardrails = self.guardrails_enabled

        # Temporarily disable guardrail if requested
        original_guardrail = self.search_engine.guardrail
        if not apply_guardrails:
            self.search_engine.guardrail = None

        try:
            if hybrid:
                return self.search_engine.search_hybrid(query, top_k)
            else:
                return self.search_engine.search_semantic(query, top_k)
        finally:
            # Restore original guardrail
            self.search_engine.guardrail = original_guardrail

    def search_with_context(
        self,
        query: str,
        context_file: str | None = None,
        top_k: int = 10
    ) -> list[SearchResult]:
        """
        Search with active file context for proximity boosting.

        Args:
            query: Natural language query
            context_file: Currently active file path
            top_k: Number of results

        Returns:
            List of SearchResult with context-aware ranking
        """
        return self.search_engine.search_hybrid(
            query,
            top_k=top_k,
            context_file=context_file
        )

    def touch_result_nodes(self, results: list[SearchResult]) -> int:
        """
        Update last_accessed timestamp for result nodes.

        Call this when user interacts with search results.

        Returns:
            Number of nodes touched
        """
        touched = 0
        for result in results:
            node = self.graph.get_node(result.node_id)
            if node and hasattr(node, 'touch'):
                node.touch()
                touched += 1
        return touched

    def record_search_feedback(
        self,
        query: str,
        results: list[SearchResult],
        success: bool,
        optimizer: Any = None  # SovereignOptimizer
    ) -> bool:
        """
        Record search feedback to train the optimizer.

        This enables the optimizer to learn optimal fusion weights
        for different query types.

        Args:
            query: The search query
            results: The returned results
            success: Whether the search led to successful task completion
            optimizer: Optional SovereignOptimizer instance

        Returns:
            True if feedback was recorded
        """
        if optimizer is None:
            return False

        if not results:
            return False

        # Generate context vector from query embedding
        query_embedding = self.generator.embed_text(query)
        context_vector = query_embedding[:64].tolist()  # Truncate for optimizer

        # Create retrieval path from top results
        retrieval_path = [r.node_id for r in results[:5]]

        # Average confidence from top results
        confidence = sum(r.combined_score for r in results[:3]) / 3 if results else 0.0

        # Record feedback
        from uuid import uuid4
        task_id = f"search_{uuid4().hex[:8]}"

        optimizer.record_feedback(
            task_id=task_id,
            context_vector=context_vector,
            retrieval_path=retrieval_path,
            confidence=confidence,
            success=success
        )

        # Touch result nodes to update recency
        if success:
            self.touch_result_nodes(results[:3])

        return True

    def get_stats(self) -> dict[str, Any]:
        """Get embedding statistics."""
        embedded_count = sum(
            1 for node in self.graph._node_index.values()
            if hasattr(node, 'embedding_vector') and node.embedding_vector is not None
        )

        return {
            "total_nodes": len(self.graph._node_index),
            "embedded_nodes": embedded_count,
            "coverage": embedded_count / max(len(self.graph._node_index), 1),
            "stored": self.store.count() if self.store else 0,
            "indexed": len(self.search_engine._node_ids) if self.search_engine._is_indexed else 0,
            "weights": {
                "semantic": self.search_engine.semantic_weight,
                "structural": self.search_engine.structural_weight,
                "recency": self.search_engine.recency_weight,
                "context": self.search_engine.context_weight
            }
        }
