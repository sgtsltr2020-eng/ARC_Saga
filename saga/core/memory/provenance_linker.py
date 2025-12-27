"""
Provenance Linker - Bidirectional Graph ↔ LoreBook Linking
============================================================

Provides full bidirectional provenance linking between RepoGraph nodes
and LoreBook entries, enabling traceability queries like:
- "What decisions affected this function?"
- "What code embodies this lesson?"

Author: ARC SAGA Development Team (AntiGravity)
Date: December 26, 2025
Status: USMA P1 Fix - Provenance Silo Resolution
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

@dataclass
class LinkerConfig:
    """Configuration for provenance linking."""

    # Linking limits (prevent bloat)
    max_links_per_node: int = 30
    max_links_per_entry: int = 30

    # Auto-linking thresholds
    similarity_threshold: float = 0.6  # Cosine similarity for auto-linking
    recency_window_hours: int = 24     # Consider entries within this window
    max_recent_entries: int = 50       # Max entries to scan for auto-linking

    # Granularity preference
    prefer_function_level: bool = True  # Prefer function/class over file


# ═══════════════════════════════════════════════════════════════
# LINK RECORD
# ═══════════════════════════════════════════════════════════════

@dataclass
class ProvenanceLink:
    """A single provenance link between node and entry."""
    node_id: str
    entry_id: str
    link_type: str = "auto"  # 'auto', 'manual', 'contextual'
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "entry_id": self.entry_id,
            "link_type": self.link_type,
            "confidence": self.confidence,
            "created_at": self.created_at
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProvenanceLink":
        return cls(**data)


# ═══════════════════════════════════════════════════════════════
# PROVENANCE LINKER
# ═══════════════════════════════════════════════════════════════

class ProvenanceLinker:
    """
    Bidirectional provenance linking between RepoGraph and LoreBook.

    Provides:
    - Auto-linking on entry creation and graph updates
    - Query helpers for traceability
    - Persistence integration
    """

    def __init__(
        self,
        graph: Any = None,  # RepoGraph
        lorebook: Any = None,  # LoreBook
        config: LinkerConfig | None = None,
        embedding_getter: Any = None  # Function to get embeddings
    ):
        """
        Initialize the provenance linker.

        Args:
            graph: RepoGraph instance
            lorebook: LoreBook instance
            config: Linker configuration
            embedding_getter: Optional function to get text embeddings
        """
        self.graph = graph
        self.lorebook = lorebook
        self.config = config or LinkerConfig()
        self.embedding_getter = embedding_getter

        # In-memory indexes for O(1) lookup
        self._node_to_entries: dict[str, set[str]] = {}  # node_id -> set of entry_ids
        self._entry_to_nodes: dict[str, set[str]] = {}  # entry_id -> set of node_ids

        # Link metadata
        self._links: dict[str, ProvenanceLink] = {}  # "node_id:entry_id" -> Link

        # Context tracking
        self._active_context_nodes: set[str] = set()
        self._recently_touched_nodes: list[tuple[str, float]] = []  # (node_id, timestamp)

        # Stats
        self.links_created = 0
        self.auto_links_created = 0

    # ─── Core Linking ──────────────────────────────────────────

    def link(
        self,
        node_id: str,
        entry_id: str,
        link_type: str = "manual",
        confidence: float = 1.0
    ) -> bool:
        """
        Create a bidirectional link between a node and an entry.

        Args:
            node_id: The graph node ID
            entry_id: The LoreEntry ID
            link_type: Type of link (manual, auto, contextual)
            confidence: Confidence score (0-1)

        Returns:
            True if link was created
        """
        link_key = f"{node_id}:{entry_id}"

        # Check if already linked
        if link_key in self._links:
            return False

        # Check limits
        if len(self._node_to_entries.get(node_id, set())) >= self.config.max_links_per_node:
            logger.debug(f"Node {node_id} at link limit")
            return False

        if len(self._entry_to_nodes.get(entry_id, set())) >= self.config.max_links_per_entry:
            logger.debug(f"Entry {entry_id} at link limit")
            return False

        # Create link
        link = ProvenanceLink(
            node_id=node_id,
            entry_id=entry_id,
            link_type=link_type,
            confidence=confidence
        )
        self._links[link_key] = link

        # Update indexes
        if node_id not in self._node_to_entries:
            self._node_to_entries[node_id] = set()
        self._node_to_entries[node_id].add(entry_id)

        if entry_id not in self._entry_to_nodes:
            self._entry_to_nodes[entry_id] = set()
        self._entry_to_nodes[entry_id].add(node_id)

        # Update graph node
        if self.graph:
            node = self.graph.get_node(node_id)
            if node and entry_id not in node.lore_entry_ids:
                node.lore_entry_ids.append(entry_id)

        # Update LoreEntry
        if self.lorebook and hasattr(self.lorebook, 'lore_entries'):
            for entry in self.lorebook.lore_entries:
                if getattr(entry, 'entry_id', None) == entry_id:
                    if node_id not in entry.related_entities:
                        entry.related_entities.append(node_id)
                    break

        self.links_created += 1
        if link_type == "auto":
            self.auto_links_created += 1

        return True

    def unlink(self, node_id: str, entry_id: str) -> bool:
        """Remove a link between a node and an entry."""
        link_key = f"{node_id}:{entry_id}"

        if link_key not in self._links:
            return False

        del self._links[link_key]

        # Update indexes
        if node_id in self._node_to_entries:
            self._node_to_entries[node_id].discard(entry_id)

        if entry_id in self._entry_to_nodes:
            self._entry_to_nodes[entry_id].discard(node_id)

        # Update graph node
        if self.graph:
            node = self.graph.get_node(node_id)
            if node and entry_id in node.lore_entry_ids:
                node.lore_entry_ids.remove(entry_id)

        return True

    # ─── Auto-Linking ──────────────────────────────────────────

    def on_entry_created(
        self,
        entry_id: str,
        context_nodes: list[str] | None = None,
        entry_text: str | None = None
    ) -> int:
        """
        Auto-link a new LoreEntry to relevant graph nodes.

        Args:
            entry_id: The new entry's ID
            context_nodes: Nodes active in current context
            entry_text: Entry text for similarity matching

        Returns:
            Number of links created
        """
        links_created = 0

        # 1. Link to explicitly provided context nodes
        if context_nodes:
            for node_id in context_nodes[:self.config.max_links_per_entry]:
                if self.link(node_id, entry_id, link_type="contextual"):
                    links_created += 1

        # 2. Link to active context nodes (from session tracking)
        for node_id in list(self._active_context_nodes)[:10]:
            if self.link(node_id, entry_id, link_type="contextual"):
                links_created += 1

        # 3. Link to recently touched nodes
        cutoff = time.time() - (self.config.recency_window_hours * 3600)
        for node_id, ts in self._recently_touched_nodes:
            if ts > cutoff:
                if self.link(node_id, entry_id, link_type="auto", confidence=0.8):
                    links_created += 1

        # 4. Similarity-based linking (if embedding available)
        if entry_text and self.embedding_getter and self.graph:
            try:
                entry_embedding = self.embedding_getter(entry_text)
                similar_nodes = self._find_similar_nodes(entry_embedding)

                for node_id, similarity in similar_nodes[:5]:
                    if similarity >= self.config.similarity_threshold:
                        if self.link(node_id, entry_id, link_type="auto", confidence=similarity):
                            links_created += 1
            except Exception as e:
                logger.debug(f"Similarity linking failed: {e}")

        logger.info(f"Entry {entry_id[:8]}: created {links_created} links")
        return links_created

    def on_graph_updated(
        self,
        changed_node_ids: list[str],
        changed_content: dict[str, str] | None = None
    ) -> int:
        """
        Auto-link updated graph nodes to relevant LoreEntries.

        Args:
            changed_node_ids: IDs of nodes that changed
            changed_content: Optional {node_id: new_content} for similarity

        Returns:
            Number of links created
        """
        links_created = 0

        if not self.lorebook or not hasattr(self.lorebook, 'lore_entries'):
            return 0

        # Get recent entries
        recent_entries = self._get_recent_entries()

        for node_id in changed_node_ids:
            # Track as recently touched
            self._recently_touched_nodes.append((node_id, time.time()))

            # Trim to limit
            if len(self._recently_touched_nodes) > 100:
                self._recently_touched_nodes = self._recently_touched_nodes[-100:]

            # Find relevant entries
            node_content = changed_content.get(node_id, "") if changed_content else None

            for entry in recent_entries:
                entry_id = getattr(entry, 'entry_id', None)
                if not entry_id:
                    continue

                # Check for overlap
                if self._has_overlap(node_id, entry, node_content):
                    if self.link(node_id, entry_id, link_type="auto", confidence=0.7):
                        links_created += 1

        return links_created

    def _find_similar_nodes(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Find nodes with similar embeddings."""
        results = []

        if not self.graph:
            return results

        for node_id, node in self.graph._node_index.items():
            if node.embedding_vector is None:
                continue

            try:
                similarity = self._cosine_similarity(query_embedding, node.embedding_vector)
                results.append((node_id, similarity))
            except Exception:
                pass

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _get_recent_entries(self) -> list[Any]:
        """Get recent LoreEntries for auto-linking."""
        if not self.lorebook or not hasattr(self.lorebook, 'lore_entries'):
            return []

        cutoff = time.time() - (self.config.recency_window_hours * 3600)
        recent = []

        for entry in self.lorebook.lore_entries:
            ts = getattr(entry, 'timestamp', None)
            if ts:
                # Convert datetime to timestamp if needed
                if hasattr(ts, 'timestamp'):
                    entry_ts = ts.timestamp()
                else:
                    entry_ts = float(ts)

                if entry_ts > cutoff:
                    recent.append(entry)

        return recent[:self.config.max_recent_entries]

    def _has_overlap(
        self,
        node_id: str,
        entry: Any,
        node_content: str | None = None
    ) -> bool:
        """Check if node and entry have meaningful overlap."""
        if not self.graph:
            return False

        node = self.graph.get_node(node_id)
        if not node:
            return False

        # Check file path overlap
        if node.file_path:
            entities = getattr(entry, 'related_entities', [])
            for entity in entities:
                if node.file_path in entity or entity in node.file_path:
                    return True

        # Check tag/content overlap
        node_tags = set()
        if node.name:
            node_tags.add(node.name.lower())
        if node_content:
            words = node_content.lower().split()[:20]
            node_tags.update(words)

        entry_tags = set(t.lower() for t in getattr(entry, 'semantic_tags', []))
        summary = getattr(entry, 'summary', '').lower()
        entry_tags.update(summary.split()[:20])

        overlap = node_tags & entry_tags
        return len(overlap) >= 2

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        a_flat = np.array(a).flatten()
        b_flat = np.array(b).flatten()

        norm_a = np.linalg.norm(a_flat)
        norm_b = np.linalg.norm(b_flat)

        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0

        return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))

    # ─── Context Tracking ──────────────────────────────────────

    def set_active_context(self, node_ids: list[str]) -> None:
        """Set nodes active in current context/session."""
        self._active_context_nodes = set(node_ids)

    def add_to_context(self, node_id: str) -> None:
        """Add a node to active context."""
        self._active_context_nodes.add(node_id)
        self._recently_touched_nodes.append((node_id, time.time()))

    def clear_context(self) -> None:
        """Clear active context."""
        self._active_context_nodes.clear()

    # ─── Query Helpers ─────────────────────────────────────────

    def get_lore_for_node(self, node_id: str) -> list[str]:
        """
        Get all LoreEntry IDs linked to a node.

        Args:
            node_id: The graph node ID

        Returns:
            List of entry_ids
        """
        # Check index first
        if node_id in self._node_to_entries:
            return list(self._node_to_entries[node_id])

        # Fallback to graph node
        if self.graph:
            node = self.graph.get_node(node_id)
            if node:
                return list(node.lore_entry_ids)

        return []

    def get_nodes_for_entry(self, entry_id: str) -> list[str]:
        """
        Get all graph node IDs linked to an entry.

        Args:
            entry_id: The LoreEntry ID

        Returns:
            List of node_ids
        """
        # Check index first
        if entry_id in self._entry_to_nodes:
            return list(self._entry_to_nodes[entry_id])

        # Fallback to LoreEntry
        if self.lorebook and hasattr(self.lorebook, 'lore_entries'):
            for entry in self.lorebook.lore_entries:
                if getattr(entry, 'entry_id', None) == entry_id:
                    return list(getattr(entry, 'related_entities', []))

        return []

    def get_provenance_path(
        self,
        node_id: str,
        max_depth: int = 3
    ) -> list[dict[str, Any]]:
        """
        Get provenance chain: lore → code impacts.

        Traces: Entry → Node → Affected Nodes → Entries

        Args:
            node_id: Starting node ID
            max_depth: Maximum chain depth

        Returns:
            List of {type, id, links} dictionaries
        """
        path = []
        visited_nodes = set()
        visited_entries = set()

        def trace(nid: str, depth: int):
            if depth > max_depth or nid in visited_nodes:
                return

            visited_nodes.add(nid)

            # Get entries for this node
            entries = self.get_lore_for_node(nid)

            node_info = {
                "type": "node",
                "id": nid,
                "entries": [],
                "affected_nodes": []
            }

            for eid in entries:
                if eid not in visited_entries:
                    visited_entries.add(eid)
                    node_info["entries"].append(eid)

            # Get affected nodes (via graph edges)
            if self.graph:
                edges = self.graph.get_edges_from(nid)
                for source, target, data in edges:
                    if data.get("edge_type") in ["CALLS", "AFFECTS", "IMPORTS"]:
                        if target not in visited_nodes:
                            node_info["affected_nodes"].append(target)

            path.append(node_info)

            # Recurse to affected nodes
            for affected in node_info["affected_nodes"]:
                trace(affected, depth + 1)

        trace(node_id, 0)
        return path

    def get_link_metadata(self, node_id: str, entry_id: str) -> ProvenanceLink | None:
        """Get metadata for a specific link."""
        link_key = f"{node_id}:{entry_id}"
        return self._links.get(link_key)

    # ─── Statistics ────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Get linker statistics."""
        return {
            "total_links": len(self._links),
            "links_created": self.links_created,
            "auto_links_created": self.auto_links_created,
            "nodes_with_links": len(self._node_to_entries),
            "entries_with_links": len(self._entry_to_nodes),
            "active_context_size": len(self._active_context_nodes),
            "recently_touched": len(self._recently_touched_nodes)
        }

    # ─── Persistence ───────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize linker state."""
        return {
            "links": [link.to_dict() for link in self._links.values()],
            "node_to_entries": {k: list(v) for k, v in self._node_to_entries.items()},
            "entry_to_nodes": {k: list(v) for k, v in self._entry_to_nodes.items()},
            "stats": {
                "links_created": self.links_created,
                "auto_links_created": self.auto_links_created
            }
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        graph: Any = None,
        lorebook: Any = None,
        config: LinkerConfig | None = None
    ) -> "ProvenanceLinker":
        """Deserialize linker state."""
        linker = cls(graph=graph, lorebook=lorebook, config=config)

        # Restore links
        for link_data in data.get("links", []):
            link = ProvenanceLink.from_dict(link_data)
            link_key = f"{link.node_id}:{link.entry_id}"
            linker._links[link_key] = link

        # Restore indexes
        linker._node_to_entries = {
            k: set(v) for k, v in data.get("node_to_entries", {}).items()
        }
        linker._entry_to_nodes = {
            k: set(v) for k, v in data.get("entry_to_nodes", {}).items()
        }

        # Restore stats
        stats = data.get("stats", {})
        linker.links_created = stats.get("links_created", 0)
        linker.auto_links_created = stats.get("auto_links_created", 0)

        return linker

    def sync_from_graph(self) -> int:
        """Sync links from graph node fields (rebuild index)."""
        synced = 0

        if not self.graph:
            return 0

        for node_id, node in self.graph._node_index.items():
            for entry_id in node.lore_entry_ids:
                link_key = f"{node_id}:{entry_id}"

                if link_key not in self._links:
                    self._links[link_key] = ProvenanceLink(
                        node_id=node_id,
                        entry_id=entry_id,
                        link_type="synced"
                    )

                    if node_id not in self._node_to_entries:
                        self._node_to_entries[node_id] = set()
                    self._node_to_entries[node_id].add(entry_id)

                    if entry_id not in self._entry_to_nodes:
                        self._entry_to_nodes[entry_id] = set()
                    self._entry_to_nodes[entry_id].add(node_id)

                    synced += 1

        logger.info(f"Synced {synced} links from graph")
        return synced
