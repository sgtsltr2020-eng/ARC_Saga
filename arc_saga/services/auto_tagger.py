"""
Auto-tagging service using TF-IDF
Extracts relevant keywords from conversation content
"""

import re
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer


class AutoTagger:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10, stop_words="english", ngram_range=(1, 2)
        )

    def extract_tags(self, content: str, max_tags: int = 5) -> List[str]:
        """
        Extract top keywords from content using TF-IDF
        """
        if not content or len(content) < 10:
            return []

        # Clean content
        content = re.sub(r"[^\\w\\s]", " ", content.lower())

        try:
            # Fit and transform
            tfidf_matrix = self.vectorizer.fit_transform([content])
            feature_names = self.vectorizer.get_feature_names_out()

            # Get scores
            scores = tfidf_matrix.toarray()[0]

            # Sort and get top tags
            tag_scores = list(zip(feature_names, scores))
            tag_scores.sort(key=lambda x: x[1], reverse=True)

            tags = [tag for tag, score in tag_scores[:max_tags] if score > 0.1]

            return tags

        except Exception as e:
            print(f"Auto-tagging error: {e}")
            return []
