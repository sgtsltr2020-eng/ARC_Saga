"""
Synthesis Engine - Creative Cross-Pollination
==============================================

Implements "Synthesis Sparks" - creative connections between distant
parts of the codebase that could yield novel optimizations.

Phase 8 Update: FQL integration with dual-citation requirement.

Author: ARC SAGA Development Team
Date: December 25, 2025
Status: Phase 8 - MAE Integration
"""

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

import numpy as np

from saga.core.mae.fql_schema import (
    FQLAction,
    FQLPacket,
    PrincipleCitation,
    StrictnessLevel,
    create_fql_packet,
)
from saga.core.memory.graph_engine import RepoGraph
from saga.core.memory.mythos import ArchitecturalDebt, MythosLibrary, SolvedPattern

logger = logging.getLogger(__name__)


@dataclass
class SynthesisSpark:
    """
    A creative insight connecting distant concepts.

    Generated when high-utility nodes with low co-occurrence
    could benefit from cross-pollination.
    """
    spark_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Source and target patterns
    source_pattern: str = ""
    source_domain: str = ""  # e.g., "Error Handling", "Streaming"
    target_problem: str = ""
    target_domain: str = ""

    # The creative bridge
    synthesis_prompt: str = ""
    confidence: float = 0.5

    # Graph metadata
    source_nodes: list[str] = field(default_factory=list)
    target_nodes: list[str] = field(default_factory=list)
    bridge_path: list[str] = field(default_factory=list)

    # Validation
    validated: bool = False
    validation_result: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "spark_id": self.spark_id,
            "created_at": self.created_at.isoformat(),
            "source_pattern": self.source_pattern,
            "source_domain": self.source_domain,
            "target_problem": self.target_problem,
            "target_domain": self.target_domain,
            "synthesis_prompt": self.synthesis_prompt,
            "confidence": self.confidence,
            "source_nodes": self.source_nodes,
            "target_nodes": self.target_nodes,
            "bridge_path": self.bridge_path,
            "validated": self.validated,
            "validation_result": self.validation_result
        }

    def to_fql_packet(
        self,
        citations: list[PrincipleCitation],
        strictness: StrictnessLevel = StrictnessLevel.FAANG_GOLDEN_PATH
    ) -> FQLPacket:
        """
        Convert spark to FQL packet with mandatory citations.

        Enforces the Dual-Citation Mandate: Every synthesis spark
        must cite at least 2 principles to justify the cross-pollination.

        Args:
            citations: List of PrincipleCitation objects (minimum 2 required)
            strictness: Strictness level for validation

        Returns:
            FQLPacket ready for Warden validation

        Raises:
            ValueError: If fewer than 2 citations provided
        """
        # Dual-Citation Mandate: Require 2+ citations for synthesis
        if len(citations) < 2:
            raise ValueError(
                f"Synthesis requires minimum 2 principle citations, got {len(citations)}. "
                "Cross-pollination must be justified by multiple codex references."
            )

        # Build subject from domains
        subject = f"{self.source_domain}→{self.target_domain}"
        if self.source_pattern:
            subject = f"{self.source_pattern}:{subject}"

        # Primary principle from first citation
        primary_principle = citations[0].rule_name

        # Build context with provenance (Spark ID for auditability)
        context = {
            "spark": self.to_dict(),
            "provenance": {
                "spark_id": self.spark_id,
                "created_at": self.created_at.isoformat(),
                "source_nodes": self.source_nodes,
                "target_nodes": self.target_nodes,
            },
            "citations": [
                {
                    "rule_id": c.rule_id,
                    "rule_name": c.rule_name,
                    "relevance": c.relevance,
                    "excerpt": c.excerpt,
                }
                for c in citations
            ],
            "cross_domain": self.source_domain != self.target_domain,
        }

        return create_fql_packet(
            sender="SynthesisAgent",
            action=FQLAction.VALIDATE_PATTERN,
            subject=subject,
            principle_id=primary_principle,
            context=context,
            strictness=strictness,
        )


class SynthesisAgent:
    """
    Agent for generating creative cross-pollination insights.

    Performs random-walk traversals on the RepoGraph to find:
    1. High-utility nodes (from Optimizer scores)
    2. Low co-occurrence pairs (rarely linked together)
    3. Pattern-debt bridges (solved patterns that could fix known debt)

    Generates SynthesisSparks - prompts to the Planner suggesting
    novel optimizations based on distant connections.
    """

    def __init__(
        self,
        graph: RepoGraph,
        mythos: MythosLibrary | None = None,
        optimizer: Any = None,  # SovereignOptimizer
        walk_length: int = 5,
        num_walks: int = 10,
        min_distance: int = 3,
        co_occurrence_threshold: float = 0.1
    ):
        """Initialize the SynthesisAgent."""
        self.graph = graph
        self.mythos = mythos or MythosLibrary()
        self.optimizer = optimizer
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.min_distance = min_distance
        self.co_occurrence_threshold = co_occurrence_threshold

        # Track co-occurrence for synthesis detection
        self._co_occurrence: dict[tuple[str, str], int] = {}
        self._node_utilities: dict[str, float] = {}

        logger.info("SynthesisAgent initialized")

    def update_utilities(self, utilities: dict[str, float]) -> None:
        """Update node utility scores from optimizer."""
        self._node_utilities.update(utilities)

    def _find_applicable_principles(
        self,
        source_domain: str,
        target_domain: str,
        source_pattern: str | None = None,
        target_problem: str | None = None
    ) -> list[PrincipleCitation]:
        """
        Find applicable codex principles for synthesis validation.

        Uses keyword matching against domain names and pattern descriptions
        to identify relevant principles. Falls back to generic principles
        if no specific matches found.

        Args:
            source_domain: Source code domain (e.g., "error_handling")
            target_domain: Target code domain (e.g., "api")
            source_pattern: Optional solved pattern name
            target_problem: Optional debt/problem name

        Returns:
            List of PrincipleCitation objects (minimum 2 for dual-citation)
        """
        citations: list[PrincipleCitation] = []

        # Domain-to-principle mapping (simplified heuristic)
        domain_principles = {
            "error": (1, "ERROR-HANDLING-01", "Error Handling Best Practices"),
            "api": (2, "API-DESIGN-01", "RESTful API Design Standards"),
            "auth": (3, "SECURITY-AUTH-01", "Authentication Security"),
            "security": (3, "SECURITY-AUTH-01", "Security Best Practices"),
            "async": (4, "ASYNC-PATTERNS-01", "Async/Await Patterns"),
            "database": (5, "DATA-PERSISTENCE-01", "Database Access Patterns"),
            "logging": (6, "OBSERVABILITY-01", "Structured Logging"),
            "test": (7, "TESTING-01", "Testing Best Practices"),
            "resilience": (8, "RESILIENCE-01", "Fault Tolerance Patterns"),
            "config": (9, "CONFIG-MGMT-01", "Configuration Management"),
            "memory": (10, "MEMORY-MGMT-01", "Memory Management"),
        }

        # Find principles for source domain
        for keyword, (rule_id, rule_name, excerpt) in domain_principles.items():
            if keyword in source_domain.lower():
                citations.append(PrincipleCitation(
                    rule_id=rule_id,
                    rule_name=rule_name,
                    relevance="HIGH",
                    excerpt=f"Source domain matches: {excerpt}"
                ))
                break

        # Find principles for target domain
        for keyword, (rule_id, rule_name, excerpt) in domain_principles.items():
            if keyword in target_domain.lower():
                # Avoid duplicate
                if not any(c.rule_id == rule_id for c in citations):
                    citations.append(PrincipleCitation(
                        rule_id=rule_id,
                        rule_name=rule_name,
                        relevance="HIGH",
                        excerpt=f"Target domain matches: {excerpt}"
                    ))
                break

        # Add pattern-specific principle if available
        if source_pattern:
            for keyword, (rule_id, rule_name, excerpt) in domain_principles.items():
                if keyword in source_pattern.lower():
                    if not any(c.rule_id == rule_id for c in citations):
                        citations.append(PrincipleCitation(
                            rule_id=rule_id,
                            rule_name=rule_name,
                            relevance="MEDIUM",
                            excerpt=f"Pattern matches: {excerpt}"
                        ))
                    break

        # Ensure minimum of 2 citations with fallback
        if len(citations) < 2:
            # Add generic cross-domain principle
            citations.append(PrincipleCitation(
                rule_id=99,
                rule_name="SYNTHESIS-CROSS-DOMAIN-01",
                relevance="MEDIUM",
                excerpt="Cross-domain synthesis requires careful integration testing"
            ))
        if len(citations) < 2:
            # Add integration principle
            citations.append(PrincipleCitation(
                rule_id=100,
                rule_name="INTEGRATION-PATTERNS-01",
                relevance="MEDIUM",
                excerpt="Component integration must maintain interface contracts"
            ))

        return citations[:3]  # Max 3 citations

    def record_co_occurrence(self, path: list[str]) -> None:
        """
        Record co-occurrence from a retrieval path.
        Used to identify rarely-linked high-utility nodes.
        """
        for i in range(len(path)):
            for j in range(i + 1, len(path)):
                key = tuple(sorted([path[i], path[j]]))
                self._co_occurrence[key] = self._co_occurrence.get(key, 0) + 1

    # --- Random Walk Traversal ---

    def random_walk(self, start_node: str | None = None) -> list[str]:
        """
        Perform a random walk on the graph.

        Args:
            start_node: Starting node (random if None)

        Returns:
            List of node IDs in the walk
        """
        nodes = list(self.graph._node_index.keys())
        if not nodes:
            return []

        if start_node is None or start_node not in nodes:
            start_node = random.choice(nodes)

        path = [start_node]
        current = start_node

        for _ in range(self.walk_length - 1):
            neighbors = list(self.graph.graph.successors(current))
            neighbors += list(self.graph.graph.predecessors(current))

            if not neighbors:
                break

            # Weight by edge weight if available
            weights = []
            for n in neighbors:
                if self.graph.graph.has_edge(current, n):
                    w = self.graph.graph[current][n].get("weight", 1.0)
                elif self.graph.graph.has_edge(n, current):
                    w = self.graph.graph[n][current].get("weight", 1.0)
                else:
                    w = 1.0
                weights.append(w)

            # Normalize
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
                current = random.choices(neighbors, weights=weights)[0]
            else:
                current = random.choice(neighbors)

            path.append(current)

        return path

    # --- Synthesis Detection ---

    def find_synthesis_candidates(self) -> list[tuple[str, str, float]]:
        """
        Find pairs of nodes that are:
        1. High utility (above average)
        2. Low co-occurrence (below threshold)
        3. Distant in graph (min hops)

        Returns:
            List of (node1, node2, score) tuples
        """
        candidates: list[tuple[str, str, float]] = []

        if not self._node_utilities:
            # Default: use all nodes with equal utility
            for node_id in self.graph._node_index:
                self._node_utilities[node_id] = 0.5

        # Find high-utility nodes
        avg_utility = np.mean(list(self._node_utilities.values()))
        high_utility_nodes = [
            n for n, u in self._node_utilities.items()
            if u >= avg_utility
        ]

        # Check pairs
        for i, node1 in enumerate(high_utility_nodes):
            for node2 in high_utility_nodes[i + 1:]:
                # Check co-occurrence
                key = tuple(sorted([node1, node2]))
                co_occur = self._co_occurrence.get(key, 0)

                # Low co-occurrence = synthesis potential
                if co_occur <= self.co_occurrence_threshold * max(len(self._co_occurrence), 1):
                    # Check distance
                    try:
                        import networkx as nx
                        distance = nx.shortest_path_length(
                            self.graph.graph.to_undirected(),
                            node1, node2
                        )
                        if distance >= self.min_distance:
                            # Score: utility * (1 / co-occurrence) * distance
                            u1 = self._node_utilities.get(node1, 0.5)
                            u2 = self._node_utilities.get(node2, 0.5)
                            score = (u1 + u2) / 2 * (1 / (co_occur + 1)) * (distance / 10)
                            candidates.append((node1, node2, score))
                    except Exception:
                        pass  # Nodes not connected

        # Sort by score
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:10]  # Top 10

    # --- Pattern-Debt Bridging ---

    def find_pattern_debt_bridges(self) -> list[tuple[SolvedPattern, ArchitecturalDebt, float]]:
        """
        Find solved patterns that could address known architectural debt.

        Returns:
            List of (pattern, debt, confidence) tuples
        """
        bridges: list[tuple[SolvedPattern, ArchitecturalDebt, float]] = []

        patterns = self.mythos.get_all_patterns()

        for chapter in self.mythos.chapters:
            for debt in chapter.architectural_debt:
                for pattern in patterns:
                    # Simple keyword matching
                    debt_words = set(debt.description.lower().split())
                    pattern_words = set(pattern.description.lower().split())

                    overlap = debt_words & pattern_words
                    if len(overlap) >= 2:  # At least 2 common words
                        confidence = len(overlap) / max(len(debt_words), 1)
                        bridges.append((pattern, debt, confidence))

        bridges.sort(key=lambda x: x[2], reverse=True)
        return bridges[:5]

    # --- Spark Generation ---

    def generate_spark(
        self,
        node1: str,
        node2: str,
        pattern: SolvedPattern | None = None,
        debt: ArchitecturalDebt | None = None
    ) -> SynthesisSpark:
        """
        Generate a SynthesisSpark from a candidate pair.

        Args:
            node1: First node ID
            node2: Second node ID
            pattern: Optional solved pattern to apply
            debt: Optional debt to address

        Returns:
            SynthesisSpark with creative prompt
        """
        n1 = self.graph.get_node(node1)
        n2 = self.graph.get_node(node2)

        source_domain = n1.file_path.split("/")[0] if n1 and n1.file_path else "unknown"
        target_domain = n2.file_path.split("/")[0] if n2 and n2.file_path else "unknown"

        # Build synthesis prompt
        if pattern and debt:
            prompt = (
                f"I found a potential synergy:\n"
                f"- SOLVED PATTERN ({pattern.name}): {pattern.description}\n"
                f"- ARCHITECTURAL DEBT ({debt.name}): {debt.description}\n"
                f"- BRIDGE: The pattern from '{source_domain}' may resolve debt in '{target_domain}'.\n"
                f"Should we apply this cross-pollination?"
            )
            confidence = 0.7
        else:
            n1_name = n1.name if n1 else node1
            n2_name = n2.name if n2 else node2
            prompt = (
                f"I see high-utility components that rarely interact:\n"
                f"- '{n1_name}' in {source_domain}\n"
                f"- '{n2_name}' in {target_domain}\n"
                f"Could there be a beneficial connection between them?"
            )
            confidence = 0.5

        return SynthesisSpark(
            source_pattern=pattern.name if pattern else "",
            source_domain=source_domain,
            target_problem=debt.name if debt else "",
            target_domain=target_domain,
            synthesis_prompt=prompt,
            confidence=confidence,
            source_nodes=[node1],
            target_nodes=[node2]
        )

    def discover_sparks(self, max_sparks: int = 3) -> list[SynthesisSpark]:
        """
        Run discovery pipeline to find potential synthesis sparks.

        Returns:
            List of SynthesisSpark objects
        """
        sparks: list[SynthesisSpark] = []

        # 1. Run random walks to update co-occurrence
        for _ in range(self.num_walks):
            path = self.random_walk()
            self.record_co_occurrence(path)

        # 2. Find synthesis candidates
        candidates = self.find_synthesis_candidates()

        # 3. Find pattern-debt bridges
        bridges = self.find_pattern_debt_bridges()

        # 4. Generate sparks from best candidates
        for node1, node2, score in candidates[:max_sparks]:
            # Check if there's a matching bridge
            matching_bridge = None
            for pattern, debt, conf in bridges:
                # Simple heuristic: check if debt keywords appear in node names
                n1 = self.graph.get_node(node1)
                n2 = self.graph.get_node(node2)
                if n1 and debt.name.lower() in n1.name.lower():
                    matching_bridge = (pattern, debt)
                    break
                if n2 and debt.name.lower() in n2.name.lower():
                    matching_bridge = (pattern, debt)
                    break

            if matching_bridge:
                spark = self.generate_spark(node1, node2, *matching_bridge)
            else:
                spark = self.generate_spark(node1, node2)

            sparks.append(spark)

        logger.info(f"Discovered {len(sparks)} synthesis sparks")
        return sparks

    def get_stats(self) -> dict[str, Any]:
        """Get synthesis agent statistics."""
        return {
            "nodes_tracked": len(self._node_utilities),
            "co_occurrences": len(self._co_occurrence),
            "mythos_patterns": len(self.mythos.get_all_patterns()),
            "mythos_chapters": len(self.mythos.chapters)
        }
