"""
RepoGraph - Knowledge Graph Engine for USMA
============================================

Implements the Structural Intelligence Layer using NetworkX.
Provides multi-hop reasoning and impact analysis capabilities.

Author: ARC SAGA Development Team
Date: December 25, 2025
Status: USMA Phase 1 - Structural Vision
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


class NodeType(str, Enum):
    """Node types in the repository knowledge graph."""
    FILE = "FILE"
    CLASS = "CLASS"
    FUNCTION = "FUNCTION"
    LORE_ENTRY = "LORE_ENTRY"
    MODULE = "MODULE"


class EdgeType(str, Enum):
    """Edge types representing relationships between nodes."""
    # Core structural relationships
    IMPORTS = "IMPORTS"           # File/module imports another
    CALLS = "CALLS"               # Function calls another function
    INHERITS = "INHERITS"         # Class inherits from another class
    REFERENCES = "REFERENCES"     # Soft reference (comments, strings)
    DESCRIBES = "DESCRIBES"       # LoreEntry describes code entity
    CONTAINS = "CONTAINS"         # File contains class/function
    AFFECTS = "AFFECTS"           # Change impact relationship

    # Enhanced semantic relationships
    DATA_FLOW = "DATA_FLOW"       # Variable assignment/usage across functions
    INTER_REPO = "INTER_REPO"     # Cross-repository dependency
    PROVENANCE = "PROVENANCE"     # Link to LoreBook/Mythos source


@dataclass
class GraphNode:
    """
    Represents a node in the knowledge graph.

    Enriched with semantic content for deep codebase understanding.
    """
    # Core identification
    node_id: str
    node_type: NodeType
    name: str
    file_path: str | None = None
    line_number: int | None = None
    end_line: int | None = None  # End line for range

    # Rich semantic content
    docstring: str | None = None            # Full docstring from source
    summary: str | None = None              # Auto-generated summary (LLM/heuristic)
    source_snippet: str | None = None       # First N lines of source (for context)

    # Execution and learning signals
    execution_traces: list[dict[str, Any]] = field(default_factory=list)  # Shadow trial outcomes
    lore_entry_ids: list[str] = field(default_factory=list)  # Linked LoreBook entries
    mythos_chapter_ids: list[str] = field(default_factory=list)  # Linked Mythos chapters

    # Embeddings and metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding_vector: Any = None  # np.ndarray or list[float], 384-dim
    last_accessed: float | None = None  # Unix timestamp of last access
    last_modified: float | None = None  # Unix timestamp of last file modification
    content_hash: str | None = None  # Hash of content for dirty detection

    @property
    def id(self) -> str:
        """Alias for node_id for compatibility."""
        return self.node_id

    def touch(self) -> None:
        """Update last_accessed timestamp to now."""
        import time
        self.last_accessed = time.time()


@dataclass
class GraphEdge:
    """Represents an edge in the knowledge graph."""
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0  # Higher = stronger relationship
    metadata: dict[str, Any] = field(default_factory=dict)


class RepoGraph:
    """
    Repository Knowledge Graph using NetworkX.

    Provides structural vision for the USMA by mapping:
    - Files, Classes, Functions as nodes
    - Import/Call/Inheritance relationships as edges
    - LoreEntries linked to code via DESCRIBES edges

    Supports:
    - Multi-hop reasoning
    - Impact analysis for change propagation
    - Lazy loading for memory efficiency
    """

    def __init__(self, project_root: str | Path | None = None):
        """Initialize the graph engine."""
        self.project_root = Path(project_root) if project_root else None
        self._graph: nx.DiGraph = nx.DiGraph()
        self._loaded_paths: set[str] = set()
        self._node_index: dict[str, GraphNode] = {}
        logger.info("RepoGraph initialized")

    @property
    def graph(self) -> nx.DiGraph:
        """Access the underlying NetworkX graph."""
        return self._graph

    @property
    def node_count(self) -> int:
        """Return the number of nodes in the graph."""
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        """Return the number of edges in the graph."""
        return self._graph.number_of_edges()

    # --- Node Operations ---

    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph."""
        self._graph.add_node(
            node.node_id,
            node_type=node.node_type.value,
            name=node.name,
            file_path=node.file_path,
            line_number=node.line_number,
            **node.metadata
        )
        self._node_index[node.node_id] = node
        logger.debug(f"Added node: {node.node_id} ({node.node_type.value})")

    def get_node(self, node_id: str) -> GraphNode | None:
        """Retrieve a node by ID."""
        return self._node_index.get(node_id)

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists."""
        return node_id in self._node_index

    def get_nodes_by_type(self, node_type: NodeType) -> list[GraphNode]:
        """Get all nodes of a specific type."""
        return [
            node for node in self._node_index.values()
            if node.node_type == node_type
        ]

    # --- Edge Operations ---

    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the graph."""
        if not self.has_node(edge.source_id):
            logger.warning(f"Source node not found: {edge.source_id}")
            return
        if not self.has_node(edge.target_id):
            logger.warning(f"Target node not found: {edge.target_id}")
            return

        self._graph.add_edge(
            edge.source_id,
            edge.target_id,
            edge_type=edge.edge_type.value,
            weight=edge.weight,
            **edge.metadata
        )
        logger.debug(f"Added edge: {edge.source_id} --{edge.edge_type.value}--> {edge.target_id}")

    def get_edges_from(self, node_id: str) -> list[tuple[str, str, dict]]:
        """Get all outgoing edges from a node."""
        if not self.has_node(node_id):
            return []
        return list(self._graph.out_edges(node_id, data=True))

    def get_edges_to(self, node_id: str) -> list[tuple[str, str, dict]]:
        """Get all incoming edges to a node."""
        if not self.has_node(node_id):
            return []
        return list(self._graph.in_edges(node_id, data=True))

    # --- Impact Analysis ---

    def analyze_impact(
        self,
        target_node_id: str,
        max_depth: int = 3,
        edge_types: list[EdgeType] | None = None
    ) -> dict[str, Any]:
        """
        Analyze downstream impact of changing a node.

        Args:
            target_node_id: The node being changed
            max_depth: Maximum traversal depth
            edge_types: Filter to specific edge types (default: all)

        Returns:
            Dict with:
            - affected_nodes: List of nodes at risk
            - impact_score: Weighted risk score
            - paths: Dependency chains
        """
        if not self.has_node(target_node_id):
            logger.warning(f"Target node not found: {target_node_id}")
            return {"affected_nodes": [], "impact_score": 0.0, "paths": []}

        allowed_types = {e.value for e in (edge_types or list(EdgeType))}

        affected: dict[str, float] = {}  # node_id -> cumulative weight
        paths: list[list[str]] = []

        def traverse(node_id: str, depth: int, path: list[str], cumulative_weight: float):
            if depth > max_depth:
                return

            for _, target, data in self._graph.out_edges(node_id, data=True):
                if data.get("edge_type") not in allowed_types:
                    continue

                edge_weight = data.get("weight", 1.0)
                new_weight = cumulative_weight * edge_weight

                if target not in affected or affected[target] < new_weight:
                    affected[target] = new_weight

                new_path = path + [target]
                paths.append(new_path)
                traverse(target, depth + 1, new_path, new_weight)

        traverse(target_node_id, 1, [target_node_id], 1.0)

        # Sort by impact weight
        affected_sorted = sorted(
            affected.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            "affected_nodes": [
                {
                    "node_id": node_id,
                    "weight": weight,
                    "node": self.get_node(node_id)
                }
                for node_id, weight in affected_sorted
            ],
            "impact_score": sum(affected.values()),
            "paths": paths[:10]  # Limit for readability
        }

    # --- Subgraph Operations (Lazy Loading) ---

    def get_subgraph(self, node_ids: list[str]) -> "RepoGraph":
        """
        Get a subgraph containing only the specified nodes.
        Useful for lazy loading focused analysis.
        """
        subgraph = RepoGraph(self.project_root)
        subgraph._graph = self._graph.subgraph(node_ids).copy()
        subgraph._node_index = {
            nid: node for nid, node in self._node_index.items()
            if nid in node_ids
        }
        return subgraph

    def get_related_subgraph(self, node_id: str, depth: int = 2) -> "RepoGraph":
        """
        Get subgraph of nodes within N hops of target.
        Implements lazy loading pattern for memory efficiency.
        """
        if not self.has_node(node_id):
            return RepoGraph(self.project_root)

        # BFS to find all nodes within depth
        related_ids: set[str] = {node_id}
        frontier: set[str] = {node_id}

        for _ in range(depth):
            new_frontier: set[str] = set()
            for nid in frontier:
                # Outgoing edges
                for _, target, _ in self._graph.out_edges(nid, data=True):
                    if target not in related_ids:
                        new_frontier.add(target)
                        related_ids.add(target)
                # Incoming edges
                for source, _, _ in self._graph.in_edges(nid, data=True):
                    if source not in related_ids:
                        new_frontier.add(source)
                        related_ids.add(source)
            frontier = new_frontier

        return self.get_subgraph(list(related_ids))

    # --- Lore Bridging ---

    def link_lore_entry(
        self,
        lore_entry_id: str,
        entity_ids: list[str],
        summary: str = ""
    ) -> None:
        """
        Create DESCRIBES edges between a LoreEntry and code entities.

        Args:
            lore_entry_id: The LoreEntry's entry_id
            entity_ids: List of code entity node IDs (files/functions/classes)
            summary: Brief description for edge metadata
        """
        # Add LoreEntry node if not exists
        if not self.has_node(lore_entry_id):
            self.add_node(GraphNode(
                node_id=lore_entry_id,
                node_type=NodeType.LORE_ENTRY,
                name=f"LoreEntry:{lore_entry_id[:8]}",
                metadata={"summary": summary}
            ))

        # Create DESCRIBES edges
        for entity_id in entity_ids:
            if self.has_node(entity_id):
                self.add_edge(GraphEdge(
                    source_id=lore_entry_id,
                    target_id=entity_id,
                    edge_type=EdgeType.DESCRIBES,
                    weight=0.8,  # Soft reference weight
                    metadata={"summary": summary}
                ))

    # --- Serialization ---

    def to_dict(self) -> dict[str, Any]:
        """Serialize graph to dictionary for persistence."""
        return {
            "nodes": [
                {
                    "node_id": node.node_id,
                    "node_type": node.node_type.value,
                    "name": node.name,
                    "file_path": node.file_path,
                    "line_number": node.line_number,
                    "metadata": node.metadata
                }
                for node in self._node_index.values()
            ],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    **data
                }
                for u, v, data in self._graph.edges(data=True)
            ]
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], project_root: str | Path | None = None) -> "RepoGraph":
        """Deserialize graph from dictionary."""
        graph = cls(project_root)

        for node_data in data.get("nodes", []):
            graph.add_node(GraphNode(
                node_id=node_data["node_id"],
                node_type=NodeType(node_data["node_type"]),
                name=node_data["name"],
                file_path=node_data.get("file_path"),
                line_number=node_data.get("line_number"),
                metadata=node_data.get("metadata", {})
            ))

        for edge_data in data.get("edges", []):
            graph.add_edge(GraphEdge(
                source_id=edge_data["source"],
                target_id=edge_data["target"],
                edge_type=EdgeType(edge_data["edge_type"]),
                weight=edge_data.get("weight", 1.0),
                metadata={k: v for k, v in edge_data.items()
                         if k not in ("source", "target", "edge_type", "weight")}
            ))

        return graph

    def __repr__(self) -> str:
        return f"RepoGraph(nodes={self.node_count}, edges={self.edge_count})"
