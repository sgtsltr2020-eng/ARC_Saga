"""
Lore Consolidator - LoreBook → Mythos Distillation Pipeline
============================================================

Automatically consolidates LoreEntries into enduring MythosChapters.
Runs in background, triggered by entry count or time threshold.

Author: ARC SAGA Development Team (AntiGravity)
Date: December 26, 2025
Status: USMA P1 Fix - Wisdom Accrual Pipeline
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

@dataclass
class ConsolidationConfig:
    """Configuration for the consolidation pipeline."""

    # Trigger thresholds
    entry_threshold: int = 30  # Consolidate after this many entries
    time_threshold_hours: int = 24  # Or after this many hours

    # Clustering settings
    min_cluster_size: int = 3  # Minimum entries to form a chapter
    max_cluster_size: int = 50  # Maximum entries per chapter
    similarity_threshold: float = 0.7  # Cosine similarity for clustering

    # Processing limits
    max_entries_per_run: int = 100  # Cap per consolidation run

    # LLM settings
    max_summary_tokens: int = 512  # Limit for summarization


# ═══════════════════════════════════════════════════════════════
# ENTRY CLUSTER
# ═══════════════════════════════════════════════════════════════

@dataclass
class EntryCluster:
    """A cluster of similar LoreEntries."""
    cluster_id: str = field(default_factory=lambda: str(uuid4()))
    entry_ids: list[str] = field(default_factory=list)
    entries: list[Any] = field(default_factory=list)  # LoreEntry objects
    centroid: np.ndarray | None = None

    # Aggregated metrics
    avg_utility: float = 0.0
    success_rate: float = 0.0
    dominant_tags: list[str] = field(default_factory=list)

    def compute_stats(self) -> None:
        """Compute aggregate statistics from entries."""
        if not self.entries:
            return

        # Compute success rate
        successes = sum(
            1 for e in self.entries
            if hasattr(e, 'codex_status') and e.codex_status == "COMPLIANT"
        )
        self.success_rate = successes / len(self.entries)

        # Aggregate tags
        tag_counts: dict[str, int] = {}
        for entry in self.entries:
            if hasattr(entry, 'tags'):
                for tag in entry.tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # Get top 5 tags
        self.dominant_tags = sorted(
            tag_counts.keys(),
            key=lambda t: tag_counts[t],
            reverse=True
        )[:5]


# ═══════════════════════════════════════════════════════════════
# LIGHTWEIGHT CLUSTERING
# ═══════════════════════════════════════════════════════════════

class LoreClusterer:
    """
    Lightweight agglomerative clustering for LoreEntries.

    Uses existing embeddings when available, falls back to
    text-based similarity heuristics.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        min_cluster_size: int = 3,
        max_cluster_size: int = 50
    ):
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size

    def cluster_entries(
        self,
        entries: list[Any],
        embeddings: dict[str, np.ndarray] | None = None
    ) -> list[EntryCluster]:
        """
        Cluster entries by semantic similarity.

        Args:
            entries: List of LoreEntry objects
            embeddings: Optional pre-computed embeddings {entry_id: vector}

        Returns:
            List of EntryCluster objects
        """
        if not entries:
            return []

        # Use embeddings if available
        if embeddings and len(embeddings) >= len(entries) * 0.5:
            return self._cluster_by_embedding(entries, embeddings)

        # Fallback to tag-based clustering
        return self._cluster_by_tags(entries)

    def _cluster_by_embedding(
        self,
        entries: list[Any],
        embeddings: dict[str, np.ndarray]
    ) -> list[EntryCluster]:
        """Cluster using cosine similarity on embeddings."""
        clusters: list[EntryCluster] = []
        assigned: set[str] = set()

        for entry in entries:
            entry_id = getattr(entry, 'entry_id', str(id(entry)))

            if entry_id in assigned:
                continue

            if entry_id not in embeddings:
                continue

            # Start new cluster
            cluster = EntryCluster(entry_ids=[entry_id], entries=[entry])
            cluster.centroid = embeddings[entry_id]
            assigned.add(entry_id)

            # Find similar entries
            for other in entries:
                other_id = getattr(other, 'entry_id', str(id(other)))

                if other_id in assigned or other_id not in embeddings:
                    continue

                if len(cluster.entries) >= self.max_cluster_size:
                    break

                similarity = self._cosine_similarity(
                    cluster.centroid,
                    embeddings[other_id]
                )

                if similarity >= self.similarity_threshold:
                    cluster.entry_ids.append(other_id)
                    cluster.entries.append(other)
                    assigned.add(other_id)

                    # Update centroid
                    cluster.centroid = self._update_centroid(cluster, embeddings)

            if len(cluster.entries) >= self.min_cluster_size:
                cluster.compute_stats()
                clusters.append(cluster)

        return clusters

    def _cluster_by_tags(self, entries: list[Any]) -> list[EntryCluster]:
        """Fallback clustering using tag overlap."""
        tag_groups: dict[str, list[Any]] = {}

        for entry in entries:
            tags = getattr(entry, 'tags', [])
            if tags:
                # Use first tag as group key
                key = tags[0] if tags else "unknown"
                if key not in tag_groups:
                    tag_groups[key] = []
                tag_groups[key].append(entry)

        # Convert to clusters
        clusters = []
        for key, group in tag_groups.items():
            if len(group) >= self.min_cluster_size:
                cluster = EntryCluster(
                    entry_ids=[getattr(e, 'entry_id', str(id(e))) for e in group],
                    entries=group
                )
                cluster.compute_stats()
                clusters.append(cluster)

        return clusters

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def _update_centroid(
        cluster: EntryCluster,
        embeddings: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Update cluster centroid as mean of member embeddings."""
        vectors = [
            embeddings[eid] for eid in cluster.entry_ids
            if eid in embeddings
        ]
        if vectors:
            return np.mean(vectors, axis=0)
        return cluster.centroid or np.zeros(384)


# ═══════════════════════════════════════════════════════════════
# CHAPTER GENERATOR
# ═══════════════════════════════════════════════════════════════

class ChapterGenerator:
    """
    Generates MythosChapters from EntryClusters.

    Uses LLM summarization when available, falls back to
    heuristic extraction.
    """

    SUMMARIZATION_PROMPT = """You are distilling lessons for a sovereign coding agent.
Given these decision outcomes from a coding project, produce a MythosChapter:

DECISIONS:
{decisions}

Generate:
1. Title: A memorable, concise pattern name (3-6 words)
2. Summary: The key lesson learned, triggers, and outcomes (2-3 sentences)
3. Tags: 3-5 relevant keywords
4. Principles: 1-3 universal principles that apply beyond this project

Be concise and timeless. Focus on patterns that would help future development.

Respond in JSON format:
{{
    "title": "...",
    "summary": "...",
    "tags": ["...", "..."],
    "principles": ["...", "..."]
}}"""

    def __init__(
        self,
        llm_summarizer: Callable[[str], str] | None = None,
        embedding_generator: Callable[[str], np.ndarray] | None = None
    ):
        """
        Initialize the chapter generator.

        Args:
            llm_summarizer: Optional LLM function for summarization
            embedding_generator: Optional function to generate embeddings
        """
        self.llm_summarizer = llm_summarizer
        self.embedding_generator = embedding_generator

    def generate_chapter(self, cluster: EntryCluster) -> dict[str, Any]:
        """
        Generate a MythosChapter from a cluster.

        Args:
            cluster: The entry cluster to summarize

        Returns:
            Dict with chapter data (title, summary, tags, etc.)
        """
        if self.llm_summarizer:
            return self._generate_with_llm(cluster)

        return self._generate_heuristic(cluster)

    def _generate_with_llm(self, cluster: EntryCluster) -> dict[str, Any]:
        """Generate chapter using LLM summarization."""
        try:
            # Build decision summaries
            decisions_text = self._format_entries_for_prompt(cluster.entries)

            prompt = self.SUMMARIZATION_PROMPT.format(decisions=decisions_text)
            response = self.llm_summarizer(prompt)

            # Parse JSON response
            import json
            result = json.loads(response)

            return {
                "title": result.get("title", "Untitled Chapter"),
                "summary": result.get("summary", "")[:512],  # Enforce limit
                "tags": result.get("tags", cluster.dominant_tags),
                "universal_principles": result.get("principles", []),
                "entry_ids": cluster.entry_ids,
                "entry_count": len(cluster.entries),
                "success_rate": cluster.success_rate
            }
        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}, using heuristic")
            return self._generate_heuristic(cluster)

    def _generate_heuristic(self, cluster: EntryCluster) -> dict[str, Any]:
        """Generate chapter using heuristic extraction."""
        # Generate title from dominant tags
        if cluster.dominant_tags:
            title = f"The {cluster.dominant_tags[0].title()} Pattern"
        else:
            title = f"Chapter {cluster.cluster_id[:8]}"

        # Build summary from entry summaries
        summaries = []
        for entry in cluster.entries[:5]:
            if hasattr(entry, 'summary') and entry.summary:
                summaries.append(entry.summary[:100])

        if summaries:
            summary = " | ".join(summaries)[:512]
        else:
            summary = f"Cluster of {len(cluster.entries)} related decisions."

        # Extract principles from successful patterns
        principles = []
        if cluster.success_rate > 0.7:
            principles.append(f"Pattern '{cluster.dominant_tags[0]}' has high success rate")

        return {
            "title": title,
            "summary": summary,
            "tags": cluster.dominant_tags,
            "universal_principles": principles,
            "entry_ids": cluster.entry_ids,
            "entry_count": len(cluster.entries),
            "success_rate": cluster.success_rate
        }

    def _format_entries_for_prompt(self, entries: list[Any]) -> str:
        """Format entries for LLM prompt."""
        lines = []
        for i, entry in enumerate(entries[:10]):  # Limit to 10 for token efficiency
            summary = getattr(entry, 'summary', '')
            status = getattr(entry, 'codex_status', 'UNKNOWN')
            tags = getattr(entry, 'tags', [])

            lines.append(
                f"{i+1}. [{status}] {summary[:150]} (tags: {', '.join(tags[:3])})"
            )

        return "\n".join(lines)

    def generate_chapter_embedding(self, chapter_data: dict[str, Any]) -> np.ndarray | None:
        """Generate embedding for chapter content."""
        if not self.embedding_generator:
            return None

        # Combine title and summary for embedding
        text = f"{chapter_data['title']}. {chapter_data['summary']}"

        try:
            return self.embedding_generator(text)
        except Exception as e:
            logger.warning(f"Failed to generate chapter embedding: {e}")
            return None


# ═══════════════════════════════════════════════════════════════
# CONSOLIDATION MANAGER
# ═══════════════════════════════════════════════════════════════

@dataclass
class ConsolidationStats:
    """Statistics from a consolidation run."""
    entries_processed: int = 0
    clusters_formed: int = 0
    chapters_created: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time


class LoreConsolidator:
    """
    Main consolidation pipeline manager.

    Handles:
    - Background trigger logic (entry count / time threshold)
    - Clustering and chapter generation
    - Integration with LoreBook and MythosLibrary
    - Persistence coordination
    """

    def __init__(
        self,
        config: ConsolidationConfig | None = None,
        lorebook: Any = None,
        mythos_library: Any = None,
        embedding_generator: Callable[[str], np.ndarray] | None = None,
        llm_summarizer: Callable[[str], str] | None = None
    ):
        """
        Initialize the consolidation pipeline.

        Args:
            config: Consolidation settings
            lorebook: LoreBook instance
            mythos_library: MythosLibrary instance
            embedding_generator: Function to generate text embeddings
            llm_summarizer: Function for LLM summarization
        """
        self.config = config or ConsolidationConfig()
        self.lorebook = lorebook
        self.mythos_library = mythos_library

        self.clusterer = LoreClusterer(
            similarity_threshold=self.config.similarity_threshold,
            min_cluster_size=self.config.min_cluster_size,
            max_cluster_size=self.config.max_cluster_size
        )

        self.generator = ChapterGenerator(
            llm_summarizer=llm_summarizer,
            embedding_generator=embedding_generator
        )

        # State
        self._unconsolidated_count = 0
        self._last_consolidation = time.time()
        self._lock = threading.Lock()

        # Background worker
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Stats
        self.total_consolidations = 0
        self.total_chapters_created = 0

        # Chapter embeddings storage
        self.chapter_embeddings: dict[str, np.ndarray] = {}

    # ─── Trigger Logic ─────────────────────────────────────────

    def on_entry_added(self) -> bool:
        """
        Called when a new LoreEntry is added.

        Returns:
            True if consolidation was triggered
        """
        with self._lock:
            self._unconsolidated_count += 1

        return self._check_triggers()

    def _check_triggers(self) -> bool:
        """Check if consolidation should be triggered."""
        # Entry count trigger
        if self._unconsolidated_count >= self.config.entry_threshold:
            self._schedule_consolidation()
            return True

        # Time trigger
        hours_since = (time.time() - self._last_consolidation) / 3600
        if hours_since >= self.config.time_threshold_hours:
            self._schedule_consolidation()
            return True

        return False

    def _schedule_consolidation(self) -> None:
        """Schedule consolidation in background thread."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return  # Already running

        self._worker_thread = threading.Thread(
            target=self._run_consolidation,
            daemon=True,
            name="LoreConsolidator-Worker"
        )
        self._worker_thread.start()
        logger.info("Consolidation scheduled in background")

    # ─── Main Consolidation ────────────────────────────────────

    def consolidate_now(self) -> ConsolidationStats:
        """
        Run consolidation immediately (synchronous).

        Returns:
            Statistics from this run
        """
        return self._run_consolidation()

    def _run_consolidation(self) -> ConsolidationStats:
        """Execute the consolidation pipeline."""
        stats = ConsolidationStats(start_time=time.time())

        try:
            with self._lock:
                self._unconsolidated_count = 0
                self._last_consolidation = time.time()

            # Get unconsolidated entries
            entries = self._get_unconsolidated_entries()
            if not entries:
                logger.info("No entries to consolidate")
                stats.end_time = time.time()
                return stats

            # Limit entries per run
            entries = entries[:self.config.max_entries_per_run]
            stats.entries_processed = len(entries)

            # Get embeddings for entries
            embeddings = self._get_entry_embeddings(entries)

            # Cluster entries
            clusters = self.clusterer.cluster_entries(entries, embeddings)
            stats.clusters_formed = len(clusters)

            logger.info(f"Formed {len(clusters)} clusters from {len(entries)} entries")

            # Generate chapters
            for cluster in clusters:
                chapter = self._create_chapter(cluster)
                if chapter:
                    stats.chapters_created += 1
                    self._mark_entries_consolidated(cluster.entry_ids, chapter.chapter_id)

            self.total_consolidations += 1
            self.total_chapters_created += stats.chapters_created

            logger.info(
                f"Consolidation complete: {stats.chapters_created} chapters "
                f"from {stats.entries_processed} entries"
            )

        except Exception as e:
            logger.error(f"Consolidation failed: {e}")

        stats.end_time = time.time()
        return stats

    def _get_unconsolidated_entries(self) -> list[Any]:
        """Get entries that haven't been consolidated yet."""
        if self.lorebook is None:
            return []

        # Get all lore entries
        all_entries = []

        # Try to get from lore_entries list
        if hasattr(self.lorebook, 'lore_entries'):
            entries = self.lorebook.lore_entries
            for entry in entries:
                # Check if not yet consolidated
                if not getattr(entry, 'chapter_id', None):
                    all_entries.append(entry)

        return all_entries

    def _get_entry_embeddings(self, entries: list[Any]) -> dict[str, np.ndarray]:
        """Get or generate embeddings for entries."""
        embeddings = {}

        for entry in entries:
            entry_id = getattr(entry, 'entry_id', str(id(entry)))

            # Check if entry has embedding
            if hasattr(entry, 'embedding') and entry.embedding is not None:
                embeddings[entry_id] = entry.embedding
            elif self.generator.embedding_generator:
                # Generate from summary
                text = getattr(entry, 'summary', '') or str(entry)
                try:
                    embeddings[entry_id] = self.generator.embedding_generator(text)
                except Exception:
                    pass

        return embeddings

    def _create_chapter(self, cluster: EntryCluster) -> Any | None:
        """Create a MythosChapter from a cluster."""
        try:
            # Generate chapter data
            chapter_data = self.generator.generate_chapter(cluster)

            # Import MythosChapter
            from saga.core.memory.mythos import MythosChapter, SolvedPattern

            # Create chapter
            chapter = MythosChapter(
                title=chapter_data["title"],
                summary=chapter_data["summary"],
                tags=chapter_data.get("tags", []),
                universal_principles=chapter_data.get("universal_principles", []),
                lore_entry_ids=chapter_data["entry_ids"],
                entry_count=chapter_data["entry_count"],
                phase=f"Consolidation-{self.total_consolidations + 1}"
            )

            # Add solved pattern if high success rate
            if chapter_data.get("success_rate", 0) > 0.7:
                pattern = SolvedPattern(
                    name=chapter_data["title"],
                    description=chapter_data["summary"][:200],
                    confidence=chapter_data["success_rate"],
                    tags=chapter_data.get("tags", [])
                )
                chapter.solved_patterns.append(pattern)

            # Generate and store chapter embedding
            embedding = self.generator.generate_chapter_embedding(chapter_data)
            if embedding is not None:
                self.chapter_embeddings[chapter.chapter_id] = embedding

            # Add to library
            if self.mythos_library:
                self.mythos_library.add_chapter(chapter)

            logger.info(f"Created chapter: {chapter.title}")
            return chapter

        except Exception as e:
            logger.error(f"Failed to create chapter: {e}")
            return None

    def _mark_entries_consolidated(self, entry_ids: list[str], chapter_id: str) -> None:
        """Mark entries as consolidated with chapter reference."""
        if self.lorebook is None:
            return

        # Update entries to reference chapter
        if hasattr(self.lorebook, 'lore_entries'):
            for entry in self.lorebook.lore_entries:
                if getattr(entry, 'entry_id', None) in entry_ids:
                    if hasattr(entry, 'chapter_id'):
                        entry.chapter_id = chapter_id

    # ─── Chapter Search ────────────────────────────────────────

    def search_chapters(
        self,
        query: str,
        top_k: int = 5,
        query_embedding: np.ndarray | None = None
    ) -> list[tuple[str, float]]:
        """
        Search chapters by semantic similarity.

        Args:
            query: Search query
            top_k: Number of results
            query_embedding: Pre-computed query embedding

        Returns:
            List of (chapter_id, similarity) tuples
        """
        if not self.chapter_embeddings:
            return []

        # Generate query embedding if not provided
        if query_embedding is None and self.generator.embedding_generator:
            try:
                query_embedding = self.generator.embedding_generator(query)
            except Exception:
                return []

        if query_embedding is None:
            return []

        # Compute similarities
        results = []
        for chapter_id, chapter_emb in self.chapter_embeddings.items():
            similarity = LoreClusterer._cosine_similarity(query_embedding, chapter_emb)
            results.append((chapter_id, similarity))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # ─── Background Worker ─────────────────────────────────────

    def start_background_worker(self, check_interval_minutes: int = 30) -> None:
        """Start background consolidation checker."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        self._stop_event.clear()

        def worker_loop():
            while not self._stop_event.is_set():
                self._stop_event.wait(timeout=check_interval_minutes * 60)
                if not self._stop_event.is_set():
                    self._check_triggers()

        self._worker_thread = threading.Thread(
            target=worker_loop,
            daemon=True,
            name="LoreConsolidator-Background"
        )
        self._worker_thread.start()
        logger.info(f"Background worker started ({check_interval_minutes}min interval)")

    def stop_background_worker(self) -> None:
        """Stop background worker."""
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)

    # ─── Persistence ───────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize consolidator state."""
        # Convert embeddings to base64
        import base64
        embeddings_serialized = {}
        for chapter_id, emb in self.chapter_embeddings.items():
            embeddings_serialized[chapter_id] = base64.b64encode(
                emb.astype(np.float32).tobytes()
            ).decode('ascii')

        return {
            "total_consolidations": self.total_consolidations,
            "total_chapters_created": self.total_chapters_created,
            "last_consolidation": self._last_consolidation,
            "unconsolidated_count": self._unconsolidated_count,
            "chapter_embeddings": embeddings_serialized
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        **kwargs
    ) -> "LoreConsolidator":
        """Deserialize consolidator state."""
        consolidator = cls(**kwargs)

        consolidator.total_consolidations = data.get("total_consolidations", 0)
        consolidator.total_chapters_created = data.get("total_chapters_created", 0)
        consolidator._last_consolidation = data.get("last_consolidation", time.time())
        consolidator._unconsolidated_count = data.get("unconsolidated_count", 0)

        # Restore embeddings
        import base64
        for chapter_id, encoded in data.get("chapter_embeddings", {}).items():
            raw = base64.b64decode(encoded.encode('ascii'))
            consolidator.chapter_embeddings[chapter_id] = np.frombuffer(raw, dtype=np.float32)

        return consolidator

    # ─── Stats ─────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Get consolidator statistics."""
        return {
            "total_consolidations": self.total_consolidations,
            "total_chapters_created": self.total_chapters_created,
            "unconsolidated_count": self._unconsolidated_count,
            "hours_since_last": (time.time() - self._last_consolidation) / 3600,
            "chapter_embeddings_count": len(self.chapter_embeddings),
            "worker_running": self._worker_thread is not None and self._worker_thread.is_alive()
        }
