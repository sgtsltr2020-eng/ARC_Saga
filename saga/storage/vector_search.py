"""
Vector Search - Similarity Matching for LoreBook
=================================================

Uses TF-IDF + cosine similarity to find similar past decisions.
Pure Python implementation (no external vector DBs).

Author: ARC SAGA Development Team
Date: December 17, 2025
"""

import logging
from typing import Any, Optional

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None

from saga.core.lorebook import Decision

logger = logging.getLogger(__name__)


class VectorSearch:
    """
    Vector similarity search for LoreBook decisions.

    Uses TF-IDF embeddings + cosine similarity to find past decisions
    similar to current question + context.

    Algorithm:
        1. Vectorize question + context tags using TF-IDF
        2. Compute cosine similarity with all stored decisions
        3. Return top_k matches above threshold

    Performance:
        - O(1) to add decision
        - O(n) to search (n = total decisions)
        - Acceptable for <10,000 decisions per project
    """

    def __init__(self) -> None:
        """Initialize vector search engine."""
        if TfidfVectorizer is None:
            raise ImportError("scikit-learn is required for VectorSearch: pip install scikit-learn")

        self.decisions: list[Decision] = []
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        self.decision_vectors: Optional[np.ndarray[Any, Any]] = None
        self.fitted = False

    def add_decision(self, decision: Decision) -> None:
        """
        Add decision to search index.

        Args:
            decision: Decision to index

        Side effects:
            Rebuilds TF-IDF vectors (call sparingly in batch if possible)
        """
        self.decisions.append(decision)
        self._reindex()

        logger.debug(
            "Decision added to vector index",
            extra={
                "decision_id": decision.decision_id,
                "total_decisions": len(self.decisions)
            }
        )

    def find_similar(
        self,
        question: str,
        context: dict[str, Any],
        top_k: int = 5,
        threshold: float = 0.7
    ) -> list[tuple[Decision, float]]:
        """
        Find similar past decisions.

        Args:
            question: Current question being asked
            context: Current context (framework, domain, etc.)
            top_k: Maximum number of results to return
            threshold: Minimum similarity score (0-1)

        Returns:
            List of (Decision, similarity_score) tuples, sorted by score desc

        Algorithm:
            1. Create query vector from question + context
            2. Compute cosine similarity with all decision vectors
            3. Filter by threshold
            4. Return top_k
        """
        if not self.fitted or len(self.decisions) == 0:
            return []

        if self.decision_vectors is None:
             return []

        # Prepare query text (question + context as searchable text)
        query_text = self._prepare_text(question, context)

        try:
            # Vectorize query
            query_vector = self.vectorizer.transform([query_text])

            # Compute similarities
            # cosine_similarity returns [[score1, score2...]]
            similarities = cosine_similarity(query_vector, self.decision_vectors)[0]

            # Get top_k above threshold
            results: list[tuple[Decision, float]] = []

            for idx, score in enumerate(similarities):
                if score >= threshold:
                    results.append((self.decisions[idx], float(score)))

            # Sort by score descending
            results.sort(key=lambda x: x[1], reverse=True)

            logger.debug(
                "Vector search completed",
                extra={
                    "query": question,
                    "results_count": len(results),
                    "top_score": results[0][1] if results else 0.0
                }
            )

            return results[:top_k]

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def _reindex(self) -> None:
        """
        Rebuild TF-IDF vectors for all decisions.

        Called after adding decisions. For production, consider
        batching adds and calling reindex once.
        """
        if len(self.decisions) == 0:
            return

        # Prepare texts for all decisions
        texts = [
            self._prepare_text(d.question, d.context, d.tags)
            for d in self.decisions
        ]

        # Fit and transform
        self.decision_vectors = self.vectorizer.fit_transform(texts).toarray()
        self.fitted = True

        logger.debug(
            "Vector index rebuilt",
            extra={
                "decisions_count": len(self.decisions),
                "vocab_size": len(self.vectorizer.vocabulary_)
            }
        )

    def _prepare_text(
        self,
        question: str,
        context: dict[str, Any],
        tags: list[str] = []
    ) -> str:
        """
        Convert question + context into searchable text.

        Args:
            question: Main question text
            context: Context dict (framework, domain, etc.)
            tags: Optional tags for additional signal

        Returns:
            Combined text string for vectorization

        Weighting:
            - Question: 3x weight (repeat 3 times)
            - Context values: 1x weight
            - Tags: 2x weight (repeat 2 times)
        """
        parts = []

        # Question gets highest weight
        parts.extend([question] * 3)

        # Context
        for key, value in context.items():
            parts.append(f"{key}:{value}")

        # Tags
        if tags:
            parts.extend(tags * 2)

        return " ".join(str(p) for p in parts)

    def get_stats(self) -> dict[str, Any]:
        """Get search index statistics."""
        return {
            "decisions_indexed": len(self.decisions),
            "vocabulary_size": len(self.vectorizer.vocabulary_) if self.fitted else 0,
            "fitted": self.fitted
        }
