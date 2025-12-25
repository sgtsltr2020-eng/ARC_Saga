"""
Unit Tests for USMA Phase 5: Chronicler
========================================

Tests for Chronicler, NarrativeMapper, and storybook rendering.
"""


from saga.core.memory import (
    ArchitecturalDebt,
    Chronicle,
    Chronicler,
    MythosChapter,
    MythosLibrary,
    NarrativeMapper,
    NarrativeVoice,
    SolvedPattern,
    StoryBeat,
    SynthesisSpark,
)
from saga.core.memory.chronicler import NarrativeBeat


class TestNarrativeVoice:
    """Tests for narrative voice selection."""

    def test_voice_enum_values(self):
        """Test all voice enum values exist."""
        assert NarrativeVoice.EPIC_HISTORIAN.value == "epic_historian"
        assert NarrativeVoice.CYBERPUNK_CHRONICLER.value == "cyberpunk"
        assert NarrativeVoice.MINIMALIST_SCRIBE.value == "minimalist"
        assert NarrativeVoice.WHIMSICAL_BARD.value == "whimsical"


class TestNarrativeMapper:
    """Tests for NarrativeMapper."""

    def test_mapper_initialization(self):
        """Test mapper initializes with voice."""
        mapper = NarrativeMapper(NarrativeVoice.EPIC_HISTORIAN)
        assert mapper.voice == NarrativeVoice.EPIC_HISTORIAN

    def test_map_merge_conflict(self):
        """Test merge conflict mapping."""
        mapper = NarrativeMapper(NarrativeVoice.EPIC_HISTORIAN)
        result = mapper.map_event(NarrativeBeat.MERGE_CONFLICT, {})

        assert "CLASH" in result.upper() or "WILLS" in result.upper()

    def test_map_synthesis_spark(self):
        """Test synthesis spark mapping."""
        mapper = NarrativeMapper(NarrativeVoice.EPIC_HISTORIAN)
        spark = SynthesisSpark(
            synthesis_prompt="Error handling can fix streaming issues"
        )

        beat = mapper.map_synthesis_spark(spark)

        assert beat.beat_type == NarrativeBeat.SYNTHESIS_SPARK
        assert "INSIGHT" in beat.rendered_text.upper() or "ANCIENT" in beat.rendered_text.upper()

    def test_map_pattern(self):
        """Test pattern mapping."""
        mapper = NarrativeMapper(NarrativeVoice.CYBERPUNK_CHRONICLER)
        pattern = SolvedPattern(name="Retry Logic", description="Retries with backoff")

        beat = mapper.map_pattern(pattern)

        assert "Retry Logic" in beat.rendered_text

    def test_map_debt(self):
        """Test debt mapping."""
        mapper = NarrativeMapper(NarrativeVoice.MINIMALIST_SCRIBE)
        debt = ArchitecturalDebt(name="No Caching", description="Missing cache layer")

        beat = mapper.map_debt(debt)

        assert "No Caching" in beat.rendered_text

    def test_voice_change(self):
        """Test changing voice."""
        mapper = NarrativeMapper(NarrativeVoice.EPIC_HISTORIAN)
        mapper.set_voice(NarrativeVoice.CYBERPUNK_CHRONICLER)

        assert mapper.voice == NarrativeVoice.CYBERPUNK_CHRONICLER


class TestStoryBeat:
    """Tests for StoryBeat dataclass."""

    def test_beat_creation(self):
        """Test creating a story beat."""
        beat = StoryBeat(
            beat_type=NarrativeBeat.REFACTOR,
            rendered_text="The code was reshaped"
        )

        assert beat.beat_type == NarrativeBeat.REFACTOR
        assert "reshaped" in beat.rendered_text

    def test_beat_serialization(self):
        """Test beat to_dict."""
        beat = StoryBeat(
            beat_type=NarrativeBeat.BUG_FIX,
            rendered_text="Bug fixed"
        )

        data = beat.to_dict()

        assert data["beat_type"] == "bug_fix"


class TestChronicle:
    """Tests for Chronicle dataclass."""

    def test_chronicle_creation(self):
        """Test creating a chronicle."""
        chronicle = Chronicle(
            title="The Great Migration",
            phase="Phase 2.0",
            voice=NarrativeVoice.EPIC_HISTORIAN
        )

        assert chronicle.title == "The Great Migration"
        assert chronicle.voice == NarrativeVoice.EPIC_HISTORIAN

    def test_chronicle_render(self):
        """Test rendering a chronicle."""
        chronicle = Chronicle(
            title="Test Chronicle",
            phase="Testing",
            voice=NarrativeVoice.EPIC_HISTORIAN,
            beats=[
                StoryBeat(
                    beat_type=NarrativeBeat.PATTERN_SOLVED,
                    rendered_text="A pattern was learned"
                )
            ]
        )

        output = chronicle.render()

        assert "Test Chronicle" in output
        assert "pattern" in output.lower()


class TestChronicler:
    """Tests for Chronicler engine."""

    def test_chronicler_initialization(self):
        """Test chronicler initializes."""
        chronicler = Chronicler()

        assert chronicler.mapper is not None
        assert chronicler.mapper.voice == NarrativeVoice.EPIC_HISTORIAN

    def test_chronicler_with_mythos(self):
        """Test chronicler with mythos library."""
        mythos = MythosLibrary()
        mythos.add_chapter(MythosChapter(
            title="The First Chapter",
            summary="Beginning of the journey",
            phase="Phase 1.0",
            solved_patterns=[
                SolvedPattern(name="Singleton", description="One instance")
            ]
        ))

        chronicler = Chronicler(mythos=mythos)
        chronicles = chronicler.chronicle_mythos()

        assert len(chronicles) >= 1
        assert chronicles[0].title == "The First Chapter"

    def test_chronicler_voice_change(self):
        """Test changing chronicler voice."""
        chronicler = Chronicler(voice=NarrativeVoice.EPIC_HISTORIAN)
        chronicler.set_voice(NarrativeVoice.WHIMSICAL_BARD)

        assert chronicler.mapper.voice == NarrativeVoice.WHIMSICAL_BARD

    def test_create_tale(self):
        """Test creating a full tale."""
        mythos = MythosLibrary()
        mythos.add_chapter(MythosChapter(
            title="Genesis",
            summary="It began",
            phase="Phase 0"
        ))

        chronicler = Chronicler(mythos=mythos)
        tale = chronicler.create_tale(title="My Project Tale")

        assert "My Project Tale" in tale
        assert "Genesis" in tale

    def test_describe_unknown_change(self):
        """Test fallback for unknown modules."""
        chronicler = Chronicler()
        beat = chronicler.describe_unknown_change("mystery_module")

        assert beat.beat_type == NarrativeBeat.FALLBACK
        assert "mystery_module" in beat.rendered_text or "dark woods" in beat.rendered_text.lower()

    def test_describe_corrupted_source(self):
        """Test corrupted source graceful handling."""
        chronicler = Chronicler()
        beat = chronicler.describe_corrupted_source()

        assert beat.beat_type == NarrativeBeat.CORRUPTED

    def test_available_voices(self):
        """Test getting available voices."""
        chronicler = Chronicler()
        voices = chronicler.get_available_voices()

        assert "epic_historian" in voices
        assert "cyberpunk" in voices
        assert len(voices) == 4


class TestVoiceTemplates:
    """Tests for voice-specific templates."""

    def test_epic_historian_style(self):
        """Test Epic Historian has appropriate styling."""
        mapper = NarrativeMapper(NarrativeVoice.EPIC_HISTORIAN)
        result = mapper.map_event(NarrativeBeat.REFACTOR, {})

        # Should have epic language
        assert any(word in result.upper() for word in ["GREAT", "RESHAPING", "FOUNDATIONS"])

    def test_cyberpunk_style(self):
        """Test Cyberpunk has appropriate styling."""
        mapper = NarrativeMapper(NarrativeVoice.CYBERPUNK_CHRONICLER)
        result = mapper.map_event(NarrativeBeat.NEW_FEATURE, {"feature": "Auth Module"})

        # Should have tech language
        assert any(word in result.upper() for word in ["MODULE", "DEPLOYED", "PAYLOAD"])

    def test_whimsical_style(self):
        """Test Whimsical Bard has appropriate styling."""
        mapper = NarrativeMapper(NarrativeVoice.WHIMSICAL_BARD)
        result = mapper.map_event(NarrativeBeat.BUG_FIX, {"bug": "NullRef"})

        # Should have playful language
        assert any(char in result for char in ["🎯", "🎪", "♪", "!"])
