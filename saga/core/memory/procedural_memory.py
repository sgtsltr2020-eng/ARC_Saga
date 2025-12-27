"""
Procedural Memory (PRAXIS-Inspired)
===================================

Indexes action consequences (outcomes, utilities) by state (code context + trajectory).
Enables one-shot adaptation via nearest-neighbor retrieval of past experiences.

Author: ARC SAGA Development Team (AntiGravity)
Date: December 26, 2025
Status: "Uncanny" Adaptation Layer
"""

import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ProceduralExperience:
    """A single unit of procedural memory."""
    memory_id: str
    timestamp: float

    # State representation
    query_text: str
    active_file: str
    context_summary: str

    # Outcome
    succeeded: bool
    utility_score: float
    outcome_summary: str
    applied_fix: Optional[str] = None

    # Vector (not persisted in JSON, reconstructed/loaded separately)
    embedding: Optional[list[float]] = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if "embedding" in d:
            del d["embedding"]  # Don't serialise large vector in JSON
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ProceduralExperience':
        return cls(**data)


class ProceduralMemory:
    """
    Procedural Memory system backing one-shot adaptation.

    Uses an in-memory vector index (cosine similarity) to retrieve
    past experiences relevant to the current state.
    """

    MAX_CAPACITY = 10000
    SIMILARITY_THRESHOLD = 0.75

    def __init__(
        self,
        embedding_engine: Any,  # GraphEmbeddingManager or compatible
        session_manager: Any = None,
        persistence_key: str = "procedural_memory_index"
    ):
        self.embedding_engine = embedding_engine
        self.session_manager = session_manager
        self.persistence_key = persistence_key

        # FAISS Index (Placeholder Logic: In production, self.index = faiss.IndexFlatIP(64))
        self.memories: dict[str, ProceduralExperience] = {}
        # Vectors kept in memory for prototype (FAISS-ready structure)
        # Initialize with shape (0, 64) - assume fixed dim or infer later
        self.vectors: np.ndarray = np.zeros((0, 64), dtype=np.float32)
        self.ids: list[str] = []

        self._load_state()
        logger.info(f"ProceduralMemory initialized with {len(self.memories)} memories (FAISS-ready).")

    # ─── Core API ───────────────────────────────────────────────

    async def index_experience(
        self,
        query: str,
        active_file: str,
        context_nodes: list[str],
        outcome: dict[str, Any]
    ) -> str:
        """
        Index a completed task experience using FAISS-ready structure.
        """
        # 1. Create Joint Embedding
        text_rep = f"Query: {query} | File: {active_file} | Context: {','.join(context_nodes[:5])}"

        try:
            embedding_result = await self.embedding_engine.generate_embedding(text_rep)
            vector = np.array(embedding_result, dtype=np.float32)
            # Normalize for Cosine/IP
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
        except Exception as e:
            logger.error(f"Failed to generate embedding for experience: {e}")
            return ""

        # 2. Create Experience Object
        memory_id = f"mem_{uuid.uuid4().hex[:8]}"
        experience = ProceduralExperience(
            memory_id=memory_id,
            timestamp=time.time(),
            query_text=query,
            active_file=active_file,
            context_summary=f"{len(context_nodes)} nodes",
            succeeded=outcome.get("success", False),
            utility_score=outcome.get("utility", 0.0),
            outcome_summary=outcome.get("summary", "No summary"),
            applied_fix=outcome.get("fix"),
            embedding=vector.tolist()
        )

        # 3. Add to Index
        self._add_to_index(memory_id, experience, vector)

        # 4. Persist
        self._save_state()

        logger.info(f"Indexed procedural memory {memory_id}: {outcome.get('summary')}")
        return memory_id

    async def retrieve_relevant(
        self,
        query: str,
        active_file: str,
        context_nodes: list[str],
        k: int = 3
    ) -> list[tuple[ProceduralExperience, float]]:
        """
        Retrieve relevant past experiences using FAISS-ready logic.
        """
        if not self.ids:
            return []

        # 1. Embed Current State
        text_rep = f"Query: {query} | File: {active_file} | Context: {','.join(context_nodes[:5])}"
        try:
            embedding_result = await self.embedding_engine.generate_embedding(text_rep)
            query_vector = np.array(embedding_result, dtype=np.float32)
        except Exception as e:
            logger.error(f"Retrieval embedding failed: {e}")
            return []

        # 2. Vector Search (Simulated FAISS IP)
        # Normalize query vector
        norm_q = np.linalg.norm(query_vector)
        if norm_q > 0:
            query_vector = query_vector / norm_q

        # In real FAISS: D, I = self.index.search(query_vector.reshape(1, -1), k)
        # Here we simulate with numpy dot product for prototype
        scores = np.dot(self.vectors, query_vector)

        # 3. Rank and Filter
        top_k_indices = np.argsort(scores)[-k:][::-1]

        results = []
        for idx in top_k_indices:
            score = float(scores[idx])
            if score < self.SIMILARITY_THRESHOLD:
                continue

            mem_id = self.ids[idx]
            memory = self.memories.get(mem_id)
            if memory:
                results.append((memory, score))

        logger.debug(f"Retrieved {len(results)} relevant memories")
        return results

    # ─── Internal Index Management ──────────────────────────────

    def _add_to_index(self, mid: str, experience: ProceduralExperience, vector: np.ndarray):
        """Add to simulated FAISS index."""
        # Eviction if full (FAISS doesn't do LRU natively, so we manage IDs)
        if len(self.ids) >= self.MAX_CAPACITY:
            oldest_id = self.ids.pop(0)
            del self.memories[oldest_id]
            # In real FAISS, we might need to rebuild or use IDMap
            # Efficiently remove first row
            self.vectors = self.vectors[1:]

        self.memories[mid] = experience
        self.ids.append(mid)

        # Expand vector storage
        # If vectors is empty (0, 64), vstack works if dimension matches
        if self.vectors.shape[0] == 0:
             self.vectors = vector.reshape(1, -1)
        else:
             self.vectors = np.vstack([self.vectors, vector])

    # ─── Persistence ────────────────────────────────────────────

    def _save_state(self):
        """Persist metadata to SessionManager (vectors rebuilt or saved sep)."""
        if not self.session_manager:
            return

        data = {
            "memories": {k: v.to_dict() for k, v in self.memories.items()},
            # In real implementations, save vectors to a binary blob or file
            # For prototype, we might skip vector persistence or rely on re-embed
            # For this 'Uncanny' level, we'll assume vectors are disposable/rebuildable
            # or just persisting small scale for now.
        }
        # self.session_manager.save(self.persistence_key, data)
        pass

    def _load_state(self):
        """Load state from SessionManager."""
        if not self.session_manager:
            return

        # data = self.session_manager.load(self.persistence_key)
        # if data:
        #     pass
        pass
