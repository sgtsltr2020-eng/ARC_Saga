"""
ContextWarden - Cache-Augmented Generation Engine
==================================================

Implements "Context Warming" by injecting immutable truths (Codex + Mythos)
into system prompts for persistent project wisdom across agent invocations.

Author: ARC SAGA Development Team
Date: December 25, 2025
Status: USMA Phase 2 - Instant Intuition
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from saga.core.memory.graph_engine import NodeType, RepoGraph
from saga.core.memory.mythos import MythosChapter, MythosLibrary

logger = logging.getLogger(__name__)


@dataclass
class CodexRule:
    """An immutable rule from the project Codex."""
    rule_id: str
    title: str
    description: str
    severity: str = "MUST"  # MUST, SHOULD, MAY
    examples: list[str] = field(default_factory=list)


@dataclass
class ContextPayload:
    """
    The assembled context payload for agent injection.
    Contains all "Immutable Truths" for the current session.
    """
    codex_rules: list[CodexRule]
    mythos_chapters: list[MythosChapter]
    active_subgraph_summary: str
    custom_instructions: list[str] = field(default_factory=list)

    def to_system_prompt(self) -> str:
        """
        Generate the system prompt injection block.
        This is prepended to agent system prompts for CAG.
        """
        sections: list[str] = []

        # Header
        sections.append("# SAGA Project Intelligence (Cached)")
        sections.append("")

        # Codex Rules (Immutable)
        if self.codex_rules:
            sections.append("## The Codex (Immutable Rules)")
            sections.append("These rules MUST be followed without exception:")
            sections.append("")
            for rule in self.codex_rules:
                sections.append(f"### [{rule.severity}] {rule.title}")
                sections.append(rule.description)
                if rule.examples:
                    sections.append("**Examples:**")
                    for ex in rule.examples[:2]:
                        sections.append(f"- {ex}")
                sections.append("")

        # Mythos Chapters (Recent Wisdom)
        if self.mythos_chapters:
            sections.append("## Project Mythos (Learned Wisdom)")
            sections.append("")
            for chapter in self.mythos_chapters:
                sections.append(chapter.to_context_block())
                sections.append("")

        # Active Subgraph (Structural Context)
        if self.active_subgraph_summary:
            sections.append("## Current Code Context")
            sections.append(self.active_subgraph_summary)
            sections.append("")

        # Custom Instructions
        if self.custom_instructions:
            sections.append("## Additional Instructions")
            for instruction in self.custom_instructions:
                sections.append(f"- {instruction}")

        return "\n".join(sections)


class ContextWarden:
    """
    Cache-Augmented Generation (CAG) engine for SAGA.

    Responsibilities:
    1. Load and cache Codex rules as immutable truths
    2. Cache the latest N Mythos chapters for session wisdom
    3. Query active subgraph from RepoGraph for structural context
    4. Generate system prompt injection for agent warming

    Usage:
        warden = ContextWarden(project_root, codex_path)
        await warden.initialize()
        payload = warden.warm_context(task_files=["saga/core/warden.py"])
        system_prompt = payload.to_system_prompt()
    """

    def __init__(
        self,
        project_root: str | Path,
        codex_path: str | Path | None = None,
        mythos_path: str | Path | None = None,
        repo_graph: RepoGraph | None = None,
        max_mythos_chapters: int = 3,
        max_codex_rules: int = 15
    ):
        """Initialize the ContextWarden."""
        self.project_root = Path(project_root)
        self.codex_path = Path(codex_path) if codex_path else None
        self.mythos_path = Path(mythos_path) if mythos_path else None
        self.repo_graph = repo_graph
        self.max_mythos_chapters = max_mythos_chapters
        self.max_codex_rules = max_codex_rules

        # Cached data
        self._codex_rules: list[CodexRule] = []
        self._mythos_library: MythosLibrary | None = None
        self._initialized: bool = False

        logger.info(f"ContextWarden created for: {self.project_root}")

    async def initialize(self) -> None:
        """Load cached data from persistence."""
        await self._load_codex()
        await self._load_mythos()
        self._initialized = True
        logger.info(
            f"ContextWarden initialized: {len(self._codex_rules)} rules, "
            f"{len(self._mythos_library.chapters) if self._mythos_library else 0} chapters"
        )

    async def _load_codex(self) -> None:
        """Load Codex rules from index file."""
        if not self.codex_path or not self.codex_path.exists():
            # Try default path
            default_path = self.project_root / ".saga" / "sagacodex_index.json"
            if default_path.exists():
                self.codex_path = default_path
            else:
                logger.warning("No Codex index found, running without immutable rules")
                return

        try:
            data = json.loads(self.codex_path.read_text(encoding="utf-8"))
            rules = data.get("rules", [])

            for rule_data in rules[:self.max_codex_rules]:
                self._codex_rules.append(CodexRule(
                    rule_id=rule_data.get("id", ""),
                    title=rule_data.get("title", ""),
                    description=rule_data.get("description", ""),
                    severity=rule_data.get("severity", "MUST"),
                    examples=rule_data.get("examples", [])
                ))

            logger.info(f"Loaded {len(self._codex_rules)} Codex rules")
        except Exception as e:
            logger.error(f"Failed to load Codex: {e}")

    async def _load_mythos(self) -> None:
        """Load Mythos library from persistence."""
        if not self.mythos_path:
            self.mythos_path = self.project_root / ".saga" / "mythos.json"

        if self.mythos_path.exists():
            try:
                data = json.loads(self.mythos_path.read_text(encoding="utf-8"))
                self._mythos_library = MythosLibrary.from_dict(data)
                logger.info(f"Loaded {len(self._mythos_library.chapters)} Mythos chapters")
            except Exception as e:
                logger.error(f"Failed to load Mythos: {e}")
                self._mythos_library = MythosLibrary()
        else:
            self._mythos_library = MythosLibrary()

    async def save_mythos(self) -> None:
        """Persist the Mythos library."""
        if self._mythos_library and self.mythos_path:
            self.mythos_path.parent.mkdir(parents=True, exist_ok=True)
            self.mythos_path.write_text(
                json.dumps(self._mythos_library.to_dict(), indent=2, default=str),
                encoding="utf-8"
            )
            logger.info("Mythos library saved")

    def add_mythos_chapter(self, chapter: MythosChapter) -> None:
        """Add a new Mythos chapter to the library."""
        if self._mythos_library:
            self._mythos_library.add_chapter(chapter)

    def warm_context(
        self,
        task_files: list[str] | None = None,
        custom_instructions: list[str] | None = None,
        include_codex: bool = True,
        include_mythos: bool = True,
        include_graph: bool = True
    ) -> ContextPayload:
        """
        Generate a warmed context payload for agent invocation.

        Args:
            task_files: Files relevant to the current task (for subgraph)
            custom_instructions: Additional instructions to inject
            include_codex: Whether to include Codex rules
            include_mythos: Whether to include Mythos chapters
            include_graph: Whether to include graph context

        Returns:
            ContextPayload ready for system prompt injection
        """
        # Codex rules
        codex_rules = self._codex_rules if include_codex else []

        # Mythos chapters
        mythos_chapters: list[MythosChapter] = []
        if include_mythos and self._mythos_library:
            mythos_chapters = self._mythos_library.get_recent_chapters(
                self.max_mythos_chapters
            )

        # Active subgraph summary
        subgraph_summary = ""
        if include_graph and self.repo_graph and task_files:
            subgraph_summary = self._generate_subgraph_summary(task_files)

        return ContextPayload(
            codex_rules=codex_rules,
            mythos_chapters=mythos_chapters,
            active_subgraph_summary=subgraph_summary,
            custom_instructions=custom_instructions or []
        )

    def _generate_subgraph_summary(self, task_files: list[str]) -> str:
        """Generate a summary of the code neighborhood for task files."""
        if not self.repo_graph:
            return ""

        lines: list[str] = ["**Active Code Neighborhood:**"]

        for file_path in task_files[:5]:  # Limit for token efficiency
            # Find file node
            file_id = f"file:{file_path}"
            if not self.repo_graph.has_node(file_id):
                continue

            node = self.repo_graph.get_node(file_id)
            if not node:
                continue

            lines.append(f"\n### {node.name}")

            # Get contained entities
            classes: list[str] = []
            functions: list[str] = []

            for _, target, data in self.repo_graph.get_edges_from(file_id):
                edge_type = data.get("edge_type")
                target_node = self.repo_graph.get_node(target)

                if edge_type == "CONTAINS" and target_node:
                    if target_node.node_type == NodeType.CLASS:
                        classes.append(target_node.name)
                    elif target_node.node_type == NodeType.FUNCTION:
                        functions.append(target_node.name)

            if classes:
                lines.append(f"- Classes: {', '.join(classes[:5])}")
            if functions:
                lines.append(f"- Functions: {', '.join(functions[:10])}")

            # Get dependencies (impact preview)
            impact = self.repo_graph.analyze_impact(file_id, max_depth=1)
            if impact["affected_nodes"]:
                affected = [n["node_id"].split(":")[-1] for n in impact["affected_nodes"][:5]]
                lines.append(f"- Affects: {', '.join(affected)}")

        return "\n".join(lines) if len(lines) > 1 else ""

    def get_cache_stats(self) -> dict[str, Any]:
        """Get statistics about cached context."""
        return {
            "codex_rules": len(self._codex_rules),
            "mythos_chapters": len(self._mythos_library.chapters) if self._mythos_library else 0,
            "total_lore_processed": self._mythos_library.total_lore_processed if self._mythos_library else 0,
            "initialized": self._initialized
        }
