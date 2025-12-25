
import logging
from dataclasses import dataclass
from typing import Any, Optional

import chromadb

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """
    A chunk of text to be indexed.

    Attributes:
        content: The text content to embed.
        metadata: Arbitrary metadata.
        doc_id: Unique ID for this chunk.
        parent_doc_id: ID of the parent document (for Small-to-Big retrieval).
    """
    content: str
    metadata: dict[str, Any]
    doc_id: str
    parent_doc_id: Optional[str] = None

class ChromaVectorStore:
    """
    Facade for ChromaDB, implementing Advanced RAG patterns.

    Bleeding Edge Features:
    - Parent-Document Storage: Stores 'Child' chunks for search, maps to 'Parent' context.
    - Multi-Collection: Separates 'Rules', 'Code', and 'Insights'.
    """

    def __init__(self, persist_directory: str = ".saga/chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Collection for small search chunks (Child)
        self.search_collection = self.client.get_or_create_collection(
            name="saga_search_index",
            metadata={"hnsw:space": "cosine"}
        )

        # Key-Value store for full parent documents (Parent)
        # Note: In a larger system this might be Redis/Postgres.
        # For SAGA, we can use a separate Chroma collection or just a JSON store.
        # Efficient approach: Use a separate collection effectively as a KV store by ID.
        self.content_store = self.client.get_or_create_collection(
            name="saga_content_store"
        )

        logger.info(f"ChromaVectorStore initialized at {persist_directory}")

    def add_documents(self, chunks: list[DocumentChunk], parent_docs: Optional[dict[str, str]] = None):
        """
        Add documents using the Parent-Document strategy.

        Args:
            chunks: Small 'child' chunks to be embedded for search.
            parent_docs: Dict mapping parent_doc_id -> full text content.
        """
        if not chunks:
            return

        # 1. Store Parent Docs (The "Full Context")
        if parent_docs:
            parent_ids = list(parent_docs.keys())
            parent_texts = list(parent_docs.values())
            # We treat content store as a retrieval mechanism, embedding doesn't matter much here
            # but getting by ID is fast.
            self.content_store.upsert(
                ids=parent_ids,
                documents=parent_texts,
                metadatas=[{"type": "parent"} for _ in parent_ids]
            )

        # 2. Store Child Chunks (The "Search Index")
        ids = [c.doc_id for c in chunks]
        texts = [c.content for c in chunks]
        metadatas = []
        for c in chunks:
            meta = c.metadata.copy()
            if c.parent_doc_id:
                meta["parent_doc_id"] = c.parent_doc_id
            metadatas.append(meta)

        self.search_collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )

        logger.info(f"Indexed {len(chunks)} child chunks and {len(parent_docs or [])} parent docs.")

    def similarity_search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """
        Perform semantic search, retrieving Parent Documents if available.

        Returns list of dicts with 'content', 'metadata', 'score'.
        """
        results = self.search_collection.query(
            query_texts=[query],
            n_results=k
        )

        if not results["ids"]:
            return []

        hits = []
        # Parse Chroma results (list of lists)
        ids = results["ids"][0]
        metadatas = results["metadatas"][0] # type: ignore
        documents = results["documents"][0] # type: ignore
        distances = results["distances"][0] # type: ignore

        for i, doc_id in enumerate(ids):
            meta = metadatas[i]
            content = documents[i]

            # Parent-Document Retrieval Logic
            # If this chunk points to a parent, fetch the parent instead.
            parent_id = meta.get("parent_doc_id")
            if parent_id:
                parent_doc = self.content_store.get(ids=[parent_id])
                if parent_doc and parent_doc["documents"]:
                    content = parent_doc["documents"][0]
                    # Merge parent metadata if useful, or keep child metadata
                    # For now, we return the Full Content with the Child Metadata (relevance)

            hits.append({
                "content": content,
                "metadata": meta,
                "score": 1.0 - distances[i] if distances else 0.0,
                "id": doc_id
            })

        return hits

    def reset(self):
        """Dangerous: Wipes the DB."""
        self.client.reset()
