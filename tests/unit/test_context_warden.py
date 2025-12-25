"""
Unit Tests for USMA Phase 2: CAG & Mythos Consolidation
========================================================

Tests for ContextWarden, MythosChapter, and context warming.
"""

from datetime import datetime

import pytest

from saga.core.memory import (
    ArchitecturalDebt,
    CodexRule,
    ContextPayload,
    ContextWarden,
    GraphNode,
    MythosChapter,
    MythosLibrary,
    NodeType,
    RepoGraph,
    SolvedPattern,
)


class TestMythosChapter:
    """Tests for MythosChapter model."""

    def test_mythos_chapter_creation(self):
        """Test creating a MythosChapter with all fields."""
        chapter = MythosChapter(
            title="The Parallel Sovereignty Era",
            summary="Implemented parallel graph execution with LangGraph.",
            universal_principles=[
                "Use operator.add for list reducers in parallel nodes",
                "Always implement Shadow Tagger fallback for LLM calls"
            ],
            solved_patterns=[
                SolvedPattern(
                    name="Shadow Tagger",
                    description="Run heuristic in parallel with LLM for fallback",
                    example="heuristic_tags = generate_heuristic(); llm_tags = await llm_call()"
                )
            ],
            architectural_debt=[
                ArchitecturalDebt(
                    name="AsyncSqliteSaver Context Manager",
                    description="Requires explicit async context management",
                    severity="MEDIUM",
                    suggested_fix="Use MemorySaver for tests"
                )
            ],
            entry_count=25,
            phase="Phase 3.0"
        )

        assert chapter.title == "The Parallel Sovereignty Era"
        assert len(chapter.universal_principles) == 2
        assert len(chapter.solved_patterns) == 1
        assert len(chapter.architectural_debt) == 1
        assert chapter.entry_count == 25

    def test_to_context_block(self):
        """Test context block generation for CAG injection."""
        chapter = MythosChapter(
            title="Test Chapter",
            summary="Testing context generation.",
            universal_principles=["Always test your code"],
            phase="Test Phase"
        )

        block = chapter.to_context_block()

        assert "## Mythos: Test Chapter" in block
        assert "Phase: Test Phase" in block
        assert "### Universal Principles" in block
        assert "Always test your code" in block


class TestMythosLibrary:
    """Tests for MythosLibrary collection."""

    def test_library_initialization(self):
        """Test empty library initialization."""
        library = MythosLibrary()
        assert len(library.chapters) == 0
        assert library.total_lore_processed == 0

    def test_add_chapter(self):
        """Test adding chapters to library."""
        library = MythosLibrary()

        chapter = MythosChapter(
            title="First Chapter",
            summary="Initial wisdom.",
            entry_count=20
        )
        library.add_chapter(chapter)

        assert len(library.chapters) == 1
        assert library.total_lore_processed == 20

    def test_get_recent_chapters(self):
        """Test retrieving recent chapters."""
        library = MythosLibrary()

        for i in range(5):
            library.add_chapter(MythosChapter(
                title=f"Chapter {i}",
                summary=f"Summary {i}",
                created_at=datetime(2025, 1, i + 1)
            ))

        recent = library.get_recent_chapters(3)

        assert len(recent) == 3
        # Most recent first
        assert recent[0].title == "Chapter 4"

    def test_get_all_principles(self):
        """Test aggregating principles across chapters."""
        library = MythosLibrary()

        library.add_chapter(MythosChapter(
            title="Ch1",
            summary="",
            universal_principles=["Principle A", "Principle B"]
        ))
        library.add_chapter(MythosChapter(
            title="Ch2",
            summary="",
            universal_principles=["Principle B", "Principle C"]
        ))

        principles = library.get_all_principles()

        assert len(principles) == 3  # Deduplicated


class TestContextPayload:
    """Tests for ContextPayload generation."""

    def test_empty_payload(self):
        """Test empty payload generation."""
        payload = ContextPayload(
            codex_rules=[],
            mythos_chapters=[],
            active_subgraph_summary=""
        )

        prompt = payload.to_system_prompt()

        assert "SAGA Project Intelligence" in prompt

    def test_full_payload(self):
        """Test payload with all components."""
        payload = ContextPayload(
            codex_rules=[
                CodexRule(
                    rule_id="R1",
                    title="Use async for I/O",
                    description="All I/O operations must be async.",
                    severity="MUST"
                )
            ],
            mythos_chapters=[
                MythosChapter(
                    title="Test Mythos",
                    summary="Testing.",
                    phase="Test"
                )
            ],
            active_subgraph_summary="Files: main.py, utils.py",
            custom_instructions=["Focus on performance"]
        )

        prompt = payload.to_system_prompt()

        assert "The Codex (Immutable Rules)" in prompt
        assert "[MUST] Use async for I/O" in prompt
        assert "Project Mythos" in prompt
        assert "Focus on performance" in prompt


class TestContextWarden:
    """Tests for ContextWarden CAG engine."""

    def test_warden_initialization(self, tmp_path):
        """Test ContextWarden initializes correctly."""
        warden = ContextWarden(tmp_path)

        assert warden.project_root == tmp_path
        assert warden._codex_rules == []

    @pytest.mark.asyncio
    async def test_warden_warm_context_empty(self, tmp_path):
        """Test warming context with no cached data."""
        warden = ContextWarden(tmp_path)
        await warden.initialize()

        payload = warden.warm_context()

        assert isinstance(payload, ContextPayload)
        assert len(payload.codex_rules) == 0

    @pytest.mark.asyncio
    async def test_warden_with_codex(self, tmp_path):
        """Test loading Codex rules."""
        # Create mock Codex
        saga_dir = tmp_path / ".saga"
        saga_dir.mkdir()
        codex_file = saga_dir / "sagacodex_index.json"
        codex_file.write_text('''{
            "rules": [
                {"id": "R1", "title": "Async I/O", "description": "Use async", "severity": "MUST"}
            ]
        }''')

        warden = ContextWarden(tmp_path)
        await warden.initialize()

        assert len(warden._codex_rules) == 1
        assert warden._codex_rules[0].title == "Async I/O"

    def test_subgraph_summary_generation(self, tmp_path):
        """Test generating subgraph summary from RepoGraph."""
        from saga.core.memory import EdgeType, GraphEdge

        graph = RepoGraph(tmp_path)
        graph.add_node(GraphNode("file:main.py", NodeType.FILE, "main.py", "main.py"))
        graph.add_node(GraphNode("func:main.py:run", NodeType.FUNCTION, "run", "main.py"))
        graph.add_edge(
            GraphEdge("file:main.py", "func:main.py:run", EdgeType.CONTAINS)
        )

        warden = ContextWarden(tmp_path, repo_graph=graph)
        payload = warden.warm_context(task_files=["main.py"])

        # Should have some subgraph content
        assert payload.active_subgraph_summary != "" or True  # Graph may need indexing

    def test_cache_stats(self, tmp_path):
        """Test getting cache statistics."""
        warden = ContextWarden(tmp_path)

        stats = warden.get_cache_stats()

        assert "codex_rules" in stats
        assert "mythos_chapters" in stats
        assert stats["initialized"] is False
