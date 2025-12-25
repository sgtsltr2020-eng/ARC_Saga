"""
Chronicler - The Tale of Your Code
===================================

Transforms technical metadata and memory logs into stylized,
immersive narratives. The "Illuminated Chronicle" of software evolution.

Author: ARC SAGA Development Team
Date: December 25, 2025
Status: USMA Phase 5 - The Storybook
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from saga.core.memory.mythos import ArchitecturalDebt, MythosChapter, MythosLibrary, SolvedPattern
from saga.core.memory.synthesis_engine import SynthesisSpark

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# NARRATIVE PERSONAS
# ═══════════════════════════════════════════════════════════════

class NarrativeVoice(str, Enum):
    """Available narrative personas for the chronicle."""
    EPIC_HISTORIAN = "epic_historian"
    CYBERPUNK_CHRONICLER = "cyberpunk"
    MINIMALIST_SCRIBE = "minimalist"
    WHIMSICAL_BARD = "whimsical"


# Voice-specific templates
VOICE_TEMPLATES = {
    NarrativeVoice.EPIC_HISTORIAN: {
        "chapter_open": "╔══════════════════════════════════════════════════════════════╗\n║  📜 {title}\n║  {phase}\n╚══════════════════════════════════════════════════════════════╝",
        "chapter_close": "───────────────────────────────────────────────────────────────\n  Thus concludes this chapter of the grand chronicle.\n───────────────────────────────────────────────────────────────",
        "merge_conflict": "⚔️  A CLASH OF WILLS at the gates of the Main Branch.\n    Two noble houses disputed the path forward, until wisdom prevailed.",
        "synthesis_spark": "✨ A FLASH OF ANCIENT INSIGHT from the Mythos!\n    The sages of old whispered: \"{insight}\"",
        "refactor": "🏛️  THE GREAT RESHAPING of the Foundations.\n    What was once scattered became unified; what was brittle became strong.",
        "bug_fix": "🛡️  A DARK CREATURE was vanquished from the realm.\n    Its name was {bug}, and it shall trouble us no more.",
        "new_feature": "🌟 A NEW TOWER rises on the horizon.\n    The builders have crafted: {feature}",
        "persona_friction": "⚡ TENSION IN THE COUNCIL CHAMBERS.\n    {alpha} and {beta} debated fiercely, their visions at odds.",
        "pattern_solved": "📚 A PATTERN was inscribed in the Great Codex.\n    \"{pattern}\" - let all who follow learn from this wisdom.",
        "debt_identified": "⚠️  A SHADOW lurks in the architecture.\n    The debt of {debt} must be addressed, lest it consume us.",
        "fallback": "🌲 In the dark woods of {module}, a change occurred...\n    The full tale remains shrouded in mystery.",
        "corrupted": "💀 THE RECORDS WERE LOST TO TIME.\n    (Rollback initiated - the archives must be restored)"
    },
    NarrativeVoice.CYBERPUNK_CHRONICLER: {
        "chapter_open": "┌──────────────────────────────────────────────────────────────┐\n│ ▓▓ {title} ▓▓\n│ :: {phase} ::\n└──────────────────────────────────────────────────────────────┘",
        "chapter_close": "══════════════════════════════════════════════════════════════\n  [END_TRANSMISSION] :: Chronicle node synced.\n══════════════════════════════════════════════════════════════",
        "merge_conflict": "⚡ [CONFLICT_DETECTED] :: Branch collision in MAIN_BRANCH\n    Two forks disputed control. Resolution: MERGE_COMPLETE.",
        "synthesis_spark": "💡 [PATTERN_UPLINK] :: Mythos sync detected\n    Neural bridge established: \"{insight}\"",
        "refactor": "🔧 [CORE_RESTRUCTURE] :: Foundation layers reshuffled.\n    Legacy cruft purged. New architecture online.",
        "bug_fix": "🐛 [BUG_TERMINATED] :: Hostile process eliminated.\n    Target: {bug} | Status: NEUTRALIZED",
        "new_feature": "📡 [NEW_MODULE_DEPLOYED] :: Feature injection complete.\n    Payload: {feature}",
        "persona_friction": "⚠️ [AGENT_CONFLICT] :: AI nodes in disagreement.\n    {alpha} vs {beta} - resolution pending human override.",
        "pattern_solved": "💾 [PATTERN_CACHED] :: New algorithm indexed.\n    Label: \"{pattern}\"",
        "debt_identified": "🔴 [TECH_DEBT_FLAGGED] :: System degradation detected.\n    Source: {debt} | Priority: HIGH",
        "fallback": "🌐 [PARTIAL_DATA] :: Node {module} - incomplete telemetry.\n    Proceeding with available fragments...",
        "corrupted": "💀 [CHECKSUM_FAILED] :: Memory corruption detected.\n    Initiating rollback protocol..."
    },
    NarrativeVoice.MINIMALIST_SCRIBE: {
        "chapter_open": "--- {title} ({phase}) ---",
        "chapter_close": "---",
        "merge_conflict": "• Merge conflict resolved",
        "synthesis_spark": "• Insight: {insight}",
        "refactor": "• Code refactored",
        "bug_fix": "• Bug fixed: {bug}",
        "new_feature": "• Feature added: {feature}",
        "persona_friction": "• Disagreement: {alpha} vs {beta}",
        "pattern_solved": "• Pattern: {pattern}",
        "debt_identified": "• Debt: {debt}",
        "fallback": "• Change in {module}",
        "corrupted": "• [Error: Data corrupted]"
    },
    NarrativeVoice.WHIMSICAL_BARD: {
        "chapter_open": "🎭 ═══ {title} ═══ 🎭\n   ♪ {phase} ♪",
        "chapter_close": "🎵 And so the code danced on... 🎵",
        "merge_conflict": "🎪 Oh what a tangle! Two jesters fought,\n    But harmony was finally sought!",
        "synthesis_spark": "🔮 The crystal ball revealed: \"{insight}\"\n    Magic flows through ancient code!",
        "refactor": "🎨 The artists painted fresh and new,\n    Old strokes erased, the canvas true!",
        "bug_fix": "🎯 A gremlin caught! Named {bug},\n    Now squashed beneath our debugging rug!",
        "new_feature": "🎁 A gift unwrapped: {feature}!\n    The crowd cheers with delight!",
        "persona_friction": "🎭 {alpha} and {beta} took the stage,\n    Their drama worthy of any age!",
        "pattern_solved": "📖 A spell was learned: \"{pattern}\"\n    Written in the wizard's pattern!",
        "debt_identified": "👻 A ghost haunts the halls: {debt}\n    Beware its creeping debt!",
        "fallback": "🌙 In the misty realm of {module}...\n    Something changed beneath the moon.",
        "corrupted": "💔 Alas! The scroll has crumbled!\n    (The bard weeps as bits are fumbled)"
    }
}


# ═══════════════════════════════════════════════════════════════
# NARRATIVE BEATS (Event Types)
# ═══════════════════════════════════════════════════════════════

class NarrativeBeat(str, Enum):
    """Types of events that become story beats."""
    MERGE_CONFLICT = "merge_conflict"
    SYNTHESIS_SPARK = "synthesis_spark"
    REFACTOR = "refactor"
    BUG_FIX = "bug_fix"
    NEW_FEATURE = "new_feature"
    PERSONA_FRICTION = "persona_friction"
    PATTERN_SOLVED = "pattern_solved"
    DEBT_IDENTIFIED = "debt_identified"
    FALLBACK = "fallback"
    CORRUPTED = "corrupted"


@dataclass
class StoryBeat:
    """A single narrative beat in the chronicle."""
    beat_id: str = field(default_factory=lambda: str(uuid4()))
    beat_type: NarrativeBeat = NarrativeBeat.FALLBACK
    timestamp: datetime = field(default_factory=datetime.utcnow)
    raw_data: dict[str, Any] = field(default_factory=dict)
    rendered_text: str = ""
    source_id: str = ""  # Mythos chapter, Lore entry, etc.

    def to_dict(self) -> dict[str, Any]:
        return {
            "beat_id": self.beat_id,
            "beat_type": self.beat_type.value,
            "timestamp": self.timestamp.isoformat(),
            "rendered_text": self.rendered_text,
            "source_id": self.source_id
        }


@dataclass
class Chronicle:
    """A complete chronicle/chapter of the story."""
    chronicle_id: str = field(default_factory=lambda: str(uuid4()))
    title: str = ""
    phase: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    beats: list[StoryBeat] = field(default_factory=list)
    voice: NarrativeVoice = NarrativeVoice.EPIC_HISTORIAN

    def render(self) -> str:
        """Render the full chronicle as styled text."""
        templates = VOICE_TEMPLATES[self.voice]

        lines = [
            templates["chapter_open"].format(title=self.title, phase=self.phase),
            ""
        ]

        for beat in self.beats:
            lines.append(beat.rendered_text)
            lines.append("")

        lines.append(templates["chapter_close"])

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# NARRATIVE MAPPER
# ═══════════════════════════════════════════════════════════════

class NarrativeMapper:
    """
    Maps technical events to stylized prose.

    Translates raw data from Mythos, LoreBook, and Synthesis
    into narrative beats with persona-appropriate language.
    """

    def __init__(self, voice: NarrativeVoice = NarrativeVoice.EPIC_HISTORIAN):
        """Initialize with chosen narrative voice."""
        self.voice = voice
        self.templates = VOICE_TEMPLATES[voice]

    def set_voice(self, voice: NarrativeVoice) -> None:
        """Change the narrative voice."""
        self.voice = voice
        self.templates = VOICE_TEMPLATES[voice]

    def map_event(self, beat_type: NarrativeBeat, data: dict[str, Any]) -> str:
        """
        Map a narrative beat type to prose.

        Args:
            beat_type: Type of event
            data: Event-specific data for template substitution

        Returns:
            Rendered prose string
        """
        template = self.templates.get(beat_type.value, self.templates["fallback"])

        try:
            return template.format(**data)
        except KeyError:
            # Missing template variables - use fallback
            return self.templates["fallback"].format(module=data.get("module", "unknown"))

    def map_synthesis_spark(self, spark: SynthesisSpark) -> StoryBeat:
        """Map a SynthesisSpark to a story beat."""
        rendered = self.map_event(
            NarrativeBeat.SYNTHESIS_SPARK,
            {"insight": spark.synthesis_prompt[:100]}
        )

        return StoryBeat(
            beat_type=NarrativeBeat.SYNTHESIS_SPARK,
            rendered_text=rendered,
            source_id=spark.spark_id,
            raw_data=spark.to_dict()
        )

    def map_pattern(self, pattern: SolvedPattern) -> StoryBeat:
        """Map a SolvedPattern to a story beat."""
        rendered = self.map_event(
            NarrativeBeat.PATTERN_SOLVED,
            {"pattern": pattern.name}
        )

        return StoryBeat(
            beat_type=NarrativeBeat.PATTERN_SOLVED,
            rendered_text=rendered,
            source_id=pattern.pattern_id
        )

    def map_debt(self, debt: ArchitecturalDebt) -> StoryBeat:
        """Map ArchitecturalDebt to a story beat."""
        rendered = self.map_event(
            NarrativeBeat.DEBT_IDENTIFIED,
            {"debt": debt.name}
        )

        return StoryBeat(
            beat_type=NarrativeBeat.DEBT_IDENTIFIED,
            rendered_text=rendered,
            source_id=debt.debt_id
        )

    def map_persona_friction(self, alpha: str, beta: str) -> StoryBeat:
        """Map persona friction to a story beat."""
        rendered = self.map_event(
            NarrativeBeat.PERSONA_FRICTION,
            {"alpha": alpha, "beta": beta}
        )

        return StoryBeat(
            beat_type=NarrativeBeat.PERSONA_FRICTION,
            rendered_text=rendered
        )

    def map_mythos_chapter(self, chapter: MythosChapter) -> Chronicle:
        """
        Map a MythosChapter to a full Chronicle.

        Extracts all narrative beats from the chapter's
        patterns, debt, and principles.
        """
        chronicle = Chronicle(
            title=chapter.title,
            phase=chapter.phase,
            created_at=chapter.created_at,
            voice=self.voice
        )

        # Add pattern beats
        for pattern in chapter.solved_patterns:
            chronicle.beats.append(self.map_pattern(pattern))

        # Add debt beats
        for debt in chapter.architectural_debt:
            chronicle.beats.append(self.map_debt(debt))

        # Add principle as a custom beat
        for principle in chapter.universal_principles[:3]:
            beat = StoryBeat(
                beat_type=NarrativeBeat.PATTERN_SOLVED,
                rendered_text=self.map_event(
                    NarrativeBeat.PATTERN_SOLVED,
                    {"pattern": principle}
                )
            )
            chronicle.beats.append(beat)

        return chronicle


# ═══════════════════════════════════════════════════════════════
# CHRONICLER ENGINE
# ═══════════════════════════════════════════════════════════════

class Chronicler:
    """
    The Chronicle Engine - transforms USMA into narrative.

    Pulls from:
    - Mythos (Tier 3 - Epic Tales)
    - LoreBook (Tier 2 - Daily Tales)
    - Synthesis Sparks (Heroic Acts)

    Outputs stylized chronicles with persona voices.
    """

    def __init__(
        self,
        mythos: MythosLibrary | None = None,
        voice: NarrativeVoice = NarrativeVoice.EPIC_HISTORIAN,
        optimizer: Any = None  # SovereignOptimizer
    ):
        """Initialize the Chronicler."""
        self.mythos = mythos or MythosLibrary()
        self.mapper = NarrativeMapper(voice)
        self.optimizer = optimizer
        self._chronicles: list[Chronicle] = []

        logger.info(f"Chronicler initialized with voice: {voice.value}")

    def set_voice(self, voice: NarrativeVoice) -> None:
        """Change the narrative voice."""
        self.mapper.set_voice(voice)

    def chronicle_mythos(self, max_chapters: int = 5) -> list[Chronicle]:
        """
        Generate chronicles from the Mythos library.

        Args:
            max_chapters: Maximum chapters to chronicle

        Returns:
            List of Chronicle objects
        """
        chronicles: list[Chronicle] = []

        recent_chapters = self.mythos.get_recent_chapters(max_chapters)

        for chapter in recent_chapters:
            chronicle = self.mapper.map_mythos_chapter(chapter)
            chronicles.append(chronicle)

        self._chronicles.extend(chronicles)
        return chronicles

    def chronicle_sparks(self, sparks: list[SynthesisSpark]) -> Chronicle:
        """
        Chronicle synthesis sparks as heroic acts.

        Args:
            sparks: List of synthesis sparks

        Returns:
            Chronicle of the sparks
        """
        chronicle = Chronicle(
            title="The Flashes of Brilliance",
            phase="Synthesis Arc",
            voice=self.mapper.voice
        )

        for spark in sparks:
            beat = self.mapper.map_synthesis_spark(spark)
            chronicle.beats.append(beat)

        self._chronicles.append(chronicle)
        return chronicle

    def create_tale(
        self,
        title: str = "The Tale of This Code",
        include_mythos: bool = True,
        include_sparks: bool = True,
        sparks: list[SynthesisSpark] | None = None
    ) -> str:
        """
        Create a complete tale from all available sources.

        Returns:
            Fully rendered narrative string
        """
        sections: list[str] = []

        # Header
        header = self._render_header(title)
        sections.append(header)

        # Mythos chronicles
        if include_mythos:
            chronicles = self.chronicle_mythos()
            for chronicle in chronicles:
                sections.append(chronicle.render())

        # Synthesis sparks
        if include_sparks and sparks:
            spark_chronicle = self.chronicle_sparks(sparks)
            sections.append(spark_chronicle.render())

        # Footer
        footer = self._render_footer()
        sections.append(footer)

        return "\n\n".join(sections)

    def _render_header(self, title: str) -> str:
        """Render the tale header."""
        if self.mapper.voice == NarrativeVoice.EPIC_HISTORIAN:
            return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     📖  {title.center(60)}  📖     ║
║                                                                              ║
║                    An Illuminated Chronicle of Code                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        elif self.mapper.voice == NarrativeVoice.CYBERPUNK_CHRONICLER:
            return f"""
┌──────────────────────────────────────────────────────────────────────────────┐
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
│                                                                              │
│  :: {title.upper().center(60)} ::  │
│                                                                              │
│  [NEURAL_CHRONICLE_INITIALIZED] :: Accessing memory banks...                 │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
"""
        else:
            return f"\n=== {title} ===\n"

    def _render_footer(self) -> str:
        """Render the tale footer."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        if self.mapper.voice == NarrativeVoice.EPIC_HISTORIAN:
            return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                         ~ The Chronicle Continues ~                          ║
║                                                                              ║
║                      Recorded on {timestamp.center(34)}                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        elif self.mapper.voice == NarrativeVoice.CYBERPUNK_CHRONICLER:
            return f"""
══════════════════════════════════════════════════════════════════════════════
  [END_TRANSMISSION] :: {timestamp}
  [STATUS] :: Chronicle archived. Memory banks synchronized.
══════════════════════════════════════════════════════════════════════════════
"""
        else:
            return f"\n--- Generated: {timestamp} ---\n"

    # --- Modularity Fallback ---

    def describe_unknown_change(self, module: str) -> StoryBeat:
        """
        Fallback for describing changes in unmapped modules.
        Uses community cluster description.
        """
        return StoryBeat(
            beat_type=NarrativeBeat.FALLBACK,
            rendered_text=self.mapper.map_event(
                NarrativeBeat.FALLBACK,
                {"module": module}
            )
        )

    def describe_corrupted_source(self) -> StoryBeat:
        """
        Describe a corrupted/invalid source gracefully.
        Uses ECC validation messaging.
        """
        return StoryBeat(
            beat_type=NarrativeBeat.CORRUPTED,
            rendered_text=self.mapper.templates["corrupted"]
        )

    # --- Feedback Integration ---

    def record_chapter_feedback(
        self,
        chronicle: Chronicle,
        positive: bool
    ) -> None:
        """
        Record user feedback on a chronicle for optimizer.

        If positive, boosts underlying Mythos entry weights.
        """
        if not self.optimizer:
            return

        reward = 1.0 if positive else -0.2

        for beat in chronicle.beats:
            if beat.source_id:
                # Create feedback for optimizer
                context = [float(ord(c) % 10) / 10 for c in beat.source_id[:64]]
                context.extend([0.0] * (64 - len(context)))

                self.optimizer.record_feedback(
                    task_id=f"chronicle_{chronicle.chronicle_id}",
                    context_vector=context,
                    retrieval_path=[beat.source_id],
                    confidence=0.8,
                    success=positive
                )

        logger.info(f"Chronicle feedback recorded: positive={positive}")

    def get_available_voices(self) -> list[str]:
        """Get list of available narrative voices."""
        return [v.value for v in NarrativeVoice]
