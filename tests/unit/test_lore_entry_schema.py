"""
Unit Tests for LoreEntry Schema Validation
===========================================

Validates the LLM Semantic Tagger's Pydantic models:
- LoreEntry ISO-8601 timestamp validation
- PersonaInfluence nested model
- tag_drift Shadow Tagger field
- LLM failure fallback behavior
"""

from datetime import datetime

import pytest

from saga.core.lorebook import (
    LoreBook,
    LoreEntry,
    PersonaInfluence,
    count_tokens,
)


class TestLoreEntrySchema:
    """Tests for LoreEntry Pydantic model validation."""

    def test_lore_entry_default_creation(self):
        """Test that LoreEntry creates with valid defaults."""
        entry = LoreEntry(summary="Test entry")

        assert entry.entry_id  # Auto-generated UUID
        assert isinstance(entry.timestamp, datetime)
        assert entry.semantic_tags == []
        assert entry.codex_status == "NEUTRAL"
        assert entry.persona_influence.primary == "Unknown"
        assert entry.raw_llm_response is None
        assert entry.tag_drift == []

    def test_lore_entry_full_creation(self):
        """Test LoreEntry with all fields populated."""
        entry = LoreEntry(
            entry_id="test-123",
            timestamp=datetime(2025, 12, 25, 10, 30, 0),
            semantic_tags=["#refactor", "#async", "#governance"],
            persona_influence=PersonaInfluence(
                primary="Architect",
                rationale="Design-focused proposal"
            ),
            codex_status="COMPLIANT",
            summary="Major architectural shift approved.",
            raw_llm_response='{"tags": ["#refactor"]}',
            tag_drift=["#async"]
        )

        assert entry.entry_id == "test-123"
        assert entry.timestamp.year == 2025
        assert "#refactor" in entry.semantic_tags
        assert entry.persona_influence.primary == "Architect"
        assert entry.codex_status == "COMPLIANT"
        assert len(entry.tag_drift) == 1

    def test_lore_entry_iso8601_timestamp(self):
        """Test that timestamp is properly serialized to ISO-8601."""
        entry = LoreEntry(
            timestamp=datetime(2025, 12, 25, 10, 30, 0),
            summary="Test"
        )

        # Pydantic model_dump with mode='json' should serialize datetime
        data = entry.model_dump(mode='json')
        assert "2025-12-25" in data["timestamp"]

    def test_lore_entry_codex_status_literal(self):
        """Test that codex_status only accepts valid literals."""
        # Valid values
        for status in ["COMPLIANT", "VIOLATION", "NEUTRAL"]:
            entry = LoreEntry(codex_status=status, summary="Test")
            assert entry.codex_status == status

    def test_persona_influence_all_roles(self):
        """Test PersonaInfluence with all four Chameleon roles."""
        roles = ["Architect", "Coder", "SDET", "Auditor"]

        for role in roles:
            influence = PersonaInfluence(
                primary=role,
                rationale=f"{role} dominated this decision"
            )
            assert influence.primary == role
            assert role in influence.rationale


class TestPersonaInfluence:
    """Tests for PersonaInfluence nested model."""

    def test_default_persona_influence(self):
        """Test PersonaInfluence defaults."""
        influence = PersonaInfluence()
        assert influence.primary == "Unknown"
        assert influence.rationale == ""

    def test_persona_influence_custom(self):
        """Test PersonaInfluence with custom values."""
        influence = PersonaInfluence(
            primary="SDET",
            rationale="Testing-driven architecture decision"
        )
        assert influence.primary == "SDET"
        assert "Testing-driven" in influence.rationale


class TestTokenCounting:
    """Tests for tiktoken-based token counting."""

    def test_count_tokens_basic(self):
        """Test basic token counting."""
        count = count_tokens("Hello world")
        assert count > 0
        assert count < 10  # Should be ~2 tokens

    def test_count_tokens_empty(self):
        """Test token counting with empty string."""
        count = count_tokens("")
        assert count == 0

    def test_count_tokens_long_text(self):
        """Test token counting with longer text."""
        long_text = "The quick brown fox jumps over the lazy dog. " * 10
        count = count_tokens(long_text)
        assert count > 50  # Should be ~100+ tokens


class TestLoreBookSemanticTagger:
    """Tests for LoreBook semantic tagger methods."""

    def test_generate_heuristic_tags_fastapi(self):
        """Test heuristic tagger detects FastAPI keywords."""
        lorebook = LoreBook.__new__(LoreBook)
        lorebook.llm_client = None
        lorebook.store = None
        lorebook.vector_search = None

        tags = lorebook._generate_heuristic_tags(
            "Building FastAPI endpoint for user authentication",
            {}
        )
        assert "#fastapi" in tags
        assert "#security" in tags

    def test_generate_heuristic_tags_parallel(self):
        """Test heuristic tagger detects parallel keywords."""
        lorebook = LoreBook.__new__(LoreBook)
        lorebook.llm_client = None

        tags = lorebook._generate_heuristic_tags(
            "Implementing worker parallel execution",
            {}
        )
        assert "#parallel" in tags

    def test_generate_heuristic_tags_governance(self):
        """Test heuristic tagger detects governance keywords."""
        lorebook = LoreBook.__new__(LoreBook)
        lorebook.llm_client = None

        tags = lorebook._generate_heuristic_tags(
            "Proposal ledger requires approval",
            {}
        )
        assert "#governance" in tags

    def test_detect_persona_tension_both_fail(self):
        """Test tension detection when both workers have issues."""
        lorebook = LoreBook.__new__(LoreBook)

        has_tension, rationale = lorebook._detect_persona_tension(
            {"status": "error", "message": "compilation failed"},
            {"status": "error", "tests": "failed"}
        )

        assert has_tension is True
        assert "Both workers" in rationale

    def test_detect_persona_tension_one_fails(self):
        """Test tension detection when only one worker has issues."""
        lorebook = LoreBook.__new__(LoreBook)

        has_tension, rationale = lorebook._detect_persona_tension(
            {"status": "success", "code": "generated"},
            {"status": "error", "tests": "failed"}
        )

        assert has_tension is True
        assert "Disagreement" in rationale

    def test_detect_persona_tension_no_tension(self):
        """Test tension detection with no issues."""
        lorebook = LoreBook.__new__(LoreBook)

        has_tension, rationale = lorebook._detect_persona_tension(
            {"status": "success", "code": "complete"},
            {"status": "success", "tests": "passed"}
        )

        assert has_tension is False
        assert rationale == ""


@pytest.mark.asyncio
class TestLoreEntryGeneration:
    """Async tests for LoreEntry generation with fallback."""

    async def test_generate_lore_entry_llm_fallback(self):
        """Test that LLM failure triggers heuristic fallback."""
        lorebook = LoreBook.__new__(LoreBook)
        lorebook.llm_client = None  # No LLM = fallback triggered
        lorebook.store = None
        lorebook.vector_search = None

        entry = await lorebook.generate_lore_entry(
            ledger_content="Implementing async database operations",
            persona_list=["Coder", "SDET"],
            alpha_output={"code": "generated"},
            beta_output={"tests": "passed"}
        )

        assert isinstance(entry, LoreEntry)
        assert entry.raw_llm_response is None  # Confirms fallback
        assert "#async" in entry.semantic_tags or "#database" in entry.semantic_tags
        assert "Heuristic fallback" in entry.persona_influence.rationale

    async def test_generate_lore_entry_with_tension(self):
        """Test LoreEntry generation includes persona friction tag."""
        lorebook = LoreBook.__new__(LoreBook)
        lorebook.llm_client = None
        lorebook.store = None
        lorebook.vector_search = None

        entry = await lorebook.generate_lore_entry(
            ledger_content="Integration conflict between modules",
            persona_list=["Coder", "SDET"],
            alpha_output={"status": "error", "message": "failed"},
            beta_output={"status": "error", "tests": "rejected"}
        )

        assert "#persona-friction" in entry.semantic_tags

    async def test_tag_drift_detection_structure(self):
        """Test that tag_drift field maintains proper structure."""
        lorebook = LoreBook.__new__(LoreBook)
        lorebook.llm_client = None
        lorebook.store = None
        lorebook.vector_search = None

        entry = await lorebook.generate_lore_entry(
            ledger_content="Simple test proposal",
            persona_list=["Architect"],
        )

        # Fallback mode - tag_drift should be empty (no LLM to compare)
        assert isinstance(entry.tag_drift, list)
