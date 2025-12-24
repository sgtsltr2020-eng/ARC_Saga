"""
SagaCodex Index Generator
=========================

Parses format-compliant Codex Markdown files and generates a machine-readable JSON index.
Follows the schema defined in docs/SagaCodex_Index_v1.md.
"""

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Heuristic Regex Patterns
HEADER_CATEGORY = re.compile(r"^##\s+(.+)$")
HEADER_RULE = re.compile(r"^###\s+(\d+)[\.:]\s+(.+)$")
KV_PAIR = re.compile(r"^\*\*(.+?)\*\*:\s*(.*)$")


@dataclass
class CodexRule:
    """Represents a single rule in the SagaCodex."""
    id: str
    title: str
    severity: str
    category: str
    tags: list[str]
    affected_artifacts: list[str]
    enforcement_phase: str
    description: str
    checklist_item: str
    detection_hint: str | None = None
    examples: list[dict[str, str]] = field(default_factory=list)
    antipatterns: list[dict[str, str]] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    related_rules: list[str] = field(default_factory=list)


class CodexIndexGenerator:
    """Generates sagacodex_index.json from Markdown source."""

    def __init__(self, codex_md_path: Path, output_path: Path) -> None:
        self.codex_md_path = codex_md_path
        self.output_path = output_path
        self.rules: list[CodexRule] = []

    def parse_markdown(self) -> list[CodexRule]:
        """Parse the Markdown file and extract rules using heuristics."""
        self.rules = [] # Clear rules to prevent duplicates
        if not self.codex_md_path.exists():
            logger.warning(f"Codex source not found at {self.codex_md_path}")
            return []

        text = self.codex_md_path.read_text(encoding="utf-8")
        lines = text.splitlines()

        current_category = "General"
        current_rule: CodexRule | None = None

        # Temporary buffers for description parts
        rule_desc_buffer = []

        def finalize_rule(rule: CodexRule | None):
            if rule:
                # Post-process description
                if rule_desc_buffer:
                    rule.description = " ".join(rule_desc_buffer).strip()

                # Default Logic for missing fields (MVP)
                if not rule.checklist_item:
                    rule.checklist_item = f"Ensure compliance with {rule.title}."

                # Rule 45 Specific Overrides/Fixes if parsing was partial
                if rule.id == "45":
                    self._apply_rule_45_overrides(rule)

                self.rules.append(rule)

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for Category Header (## )
            if category_match := HEADER_CATEGORY.match(line):
                finalize_rule(current_rule)
                current_rule = None
                rule_desc_buffer = []

                # Clean up category name (remove 'STANDARDS' etc if desired, or keep raw)
                raw_cat = category_match.group(1).strip()
                # Example: "CORE PYTHON STANDARDS" -> "Core Python"
                current_category = raw_cat.title().replace(" Standards", "")

            # Check for Rule Header (### )
            elif rule_match := HEADER_RULE.match(line):
                finalize_rule(current_rule)
                rule_desc_buffer = []

                rule_id = rule_match.group(1)
                rule_title = rule_match.group(2).strip()

                # Initialize new rule with defaults
                current_rule = CodexRule(
                    id=rule_id,
                    title=rule_title,
                    severity="CRITICAL",  # Default assumption from intro
                    category=current_category,
                    tags=[],
                    affected_artifacts=["all"], # Default
                    enforcement_phase="pre-merge", # Default
                    description="",
                    checklist_item="",
                )

            # Check for Key-Value pairs inside a rule
            elif current_rule and (kv_match := KV_PAIR.match(line)):
                key, val = kv_match.groups()
                key = key.lower()

                if key == "severity":
                    current_rule.severity = val.upper()
                elif key == "tags":
                    # Split by comma
                    current_rule.tags = [t.strip() for t in val.split(",") if t.strip()]
                elif key == "checklist item":
                    # Remove quotes if present
                    current_rule.checklist_item = val.strip('"').strip("'")
                elif key == "detection hint":
                    current_rule.detection_hint = val.strip()
                elif key == "rule":
                    rule_desc_buffer.append(val)
                elif key == "why":
                    rule_desc_buffer.append(f"Why: {val}")
                elif key == "enforcement":
                    # Sometimes enforcement implies phase
                    if "runtime" in val.lower():
                        current_rule.enforcement_phase = "runtime"

        # Finalize last rule
        finalize_rule(current_rule)

        return self.rules

    def _apply_rule_45_overrides(self, rule: CodexRule) -> None:
        """Force specific values for Rule 45 as requested."""
        if rule.severity == "CRITICAL": # If default wasn't overwritten by parsing
            rule.severity = "WARNING"

        # Ensure tags are present if parsing failed (it shouldn't if format is good)
        if not rule.tags:
            rule.tags = ["refactoring", "tests", "mypy", "diff-size"]

        if not rule.checklist_item or rule.checklist_item.startswith("Ensure compliance"):
             rule.checklist_item = "When fixing lint/mypy-only issues, prefer adding annotations or small edits instead of rewriting entire files—especially tests."

        if not rule.category or rule.category == "General":
            rule.category = "Process"

    def build_index(self, rules: list[CodexRule]) -> dict[str, Any]:
        """Construct the final JSON dictionary."""
        return {
            "version": "1.0.0",
            "language": "Python",
            "framework": "FastAPI",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "rules": [asdict(r) for r in rules]
        }

    def write_index(self) -> Path:
        """Parse, build, and write the index file."""
        self.parse_markdown()
        index_data = self.build_index(self.rules)

        # Ensure parent dir exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)

        logger.info(f"Generated Codex Index at {self.output_path} with {len(self.rules)} rules.")
        return self.output_path
