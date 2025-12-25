"""
SAGA Chronicle CLI Command
===========================

The storybook view for Saga: The Tale of Your Code.
Renders development history as stylized narrative.
"""

import argparse
import json
import sys
from pathlib import Path

from saga.core.memory import (
    Chronicler,
    MythosLibrary,
    NarrativeVoice,
    RepoGraph,
    SynthesisAgent,
)


def run_chronicle(args: argparse.Namespace) -> int:
    """Execute the chronicle command."""
    root = Path.cwd()

    # Load Mythos if available
    mythos_path = root / ".saga" / "mythos.json"
    mythos = MythosLibrary()

    if mythos_path.exists():
        try:
            data = json.loads(mythos_path.read_text(encoding="utf-8"))
            mythos = MythosLibrary.from_dict(data)
        except Exception as e:
            print(f"⚠️  Could not load Mythos: {e}", file=sys.stderr)

    # Select voice
    try:
        voice = NarrativeVoice(args.voice)
    except ValueError:
        voice = NarrativeVoice.EPIC_HISTORIAN

    # Initialize Chronicler
    chronicler = Chronicler(mythos=mythos, voice=voice)

    # Generate synthesis sparks if requested
    sparks = None
    if args.include_sparks:
        try:
            graph = RepoGraph(root)
            agent = SynthesisAgent(graph, mythos=mythos)
            sparks = agent.discover_sparks(max_sparks=3)
        except Exception:
            pass

    # Generate the tale
    if mythos.chapters:
        tale = chronicler.create_tale(
            title=args.title or "The Tale of This Code",
            include_sparks=args.include_sparks,
            sparks=sparks
        )
    else:
        # No Mythos - show demo
        tale = _generate_demo_tale(voice)

    print(tale)
    return 0


def _generate_demo_tale(voice: NarrativeVoice) -> str:
    """Generate a demo tale when no Mythos exists."""
    chronicler = Chronicler(voice=voice)

    header = chronicler._render_header("Saga: The Tale of Your Code")

    if voice == NarrativeVoice.EPIC_HISTORIAN:
        body = """
╔══════════════════════════════════════════════════════════════╗
║  📜 The Beginning
║  Genesis
╚══════════════════════════════════════════════════════════════╝

  In the beginning, there was only the void of an empty repository.

  But the builders came, and with their hands they crafted structure.

  Run `saga serve` and begin your journey.
  As you work, the Mythos will grow, and your tale shall be written.

───────────────────────────────────────────────────────────────
  Thus concludes this chapter of the grand chronicle.
───────────────────────────────────────────────────────────────
"""
    elif voice == NarrativeVoice.CYBERPUNK_CHRONICLER:
        body = """
┌──────────────────────────────────────────────────────────────┐
│ ▓▓ SYSTEM_BOOT ▓▓
│ :: Genesis Protocol ::
└──────────────────────────────────────────────────────────────┘

  [INITIALIZING] :: Memory banks empty.
  [STATUS] :: No neural patterns detected.

  Run `saga serve` to begin data collection.
  As you code, the AI will learn, and your legend shall be forged.

══════════════════════════════════════════════════════════════
  [END_TRANSMISSION] :: Chronicle node synced.
══════════════════════════════════════════════════════════════
"""
    else:
        body = """
--- The Beginning (Genesis) ---

• No history recorded yet.
• Run `saga serve` to start.

---
"""

    footer = chronicler._render_footer()
    return header + body + footer


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Chronicle your code's history as a stylized narrative."
    )

    parser.add_argument(
        "--voice",
        choices=["epic_historian", "cyberpunk", "minimalist", "whimsical"],
        default="epic_historian",
        help="Narrative voice/persona to use"
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom title for the tale"
    )
    parser.add_argument(
        "--include-sparks",
        action="store_true",
        help="Include synthesis sparks as heroic acts"
    )
    parser.add_argument(
        "--chapters",
        type=int,
        default=5,
        help="Maximum number of chapters to include"
    )

    args = parser.parse_args()
    sys.exit(run_chronicle(args))


if __name__ == "__main__":
    main()
