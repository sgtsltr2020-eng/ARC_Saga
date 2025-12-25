
import logging
from typing import Any

from rank_bm25 import BM25Okapi

from saga.knowledge.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Bleeding Edge Retriever: Lexical-Semantic Fusion.

    Combines:
    1. Semantic Search (ChromaDB) - Finds concepts ("Async Transport").
    2. Lexical Search (BM25) - Finds precise terms ("aiohttp==3.9.5").
    """

    def __init__(self, vector_store: ChromaVectorStore):
        self.vector_store = vector_store
        self.bm25 = None
        self.corpus_map = {} # Map ID -> text for BM25

    def fit_bm25(self, documents: list[str], doc_ids: list[str]):
        """
        Initialize BM25 with a corpus.
        In production this would be an inverted index, but for Phase 3 prototype
        we build it in-memory from the ingested rules.
        """
        tokenized_corpus = [doc.split(" ") for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.corpus_map = {i: {"id": doc_ids[i], "content": documents[i]} for i in range(len(documents))}
        logger.info(f"BM25 index built with {len(documents)} documents")

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """
        Perform Hybrid Search.
        Merges results from Vector Store and BM25 using Reciprocal Rank Fusion (RRF).
        """
        # 1. Vector Search (Semantic)
        vector_results = self.vector_store.similarity_search(query, k=k)

        # 2. Keyword Search (Lexical)
        bm25_results = []
        if self.bm25:
            tokenized_query = query.split(" ")
            # Get top N scores
            doc_scores = self.bm25.get_scores(tokenized_query)
            top_n = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:k]

            for idx in top_n:
                if doc_scores[idx] > 0: # Only relevant hits
                    mapped = self.corpus_map[idx]
                    bm25_results.append({
                        "content": mapped["content"],
                        "id": mapped["id"],
                        "score": doc_scores[idx],
                        "metadata": {"source": "bm25"}
                    })

        # 3. Fusion (Simple deduplication for now, RRF in Phase 4)
        # We prioritize Vector results but inject high-score BM25 matches if missing
        combined_ids = {r["id"] for r in vector_results}
        final_results = vector_results

        for br in bm25_results:
            if br["id"] not in combined_ids:
                final_results.append(br)

        logger.info(f"Hybrid Search: {len(vector_results)} vector + {len(bm25_results)} keyword hits")
        return final_results[:k] # Return top k
