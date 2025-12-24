from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from saga.core.codex_index_client import CodexIndexClient


@dataclass
class ChangeContext:
    """Metadata about the proposed change."""
    user_request: str  # What the user asked for
    change_summary: str  # Brief description of the diff (e.g., "Rewrites 90% of tests/test_health.py")
    diff_size: Optional[int] = None  # Optional: number of lines changed


@dataclass
class FormattedExplanation:
    """Structured output from the formatter."""
    header: str  # e.g., "⚠️ Conflict: Codex Rule 45 (Process, WARNING)"
    context_block: str  # "You asked for X, this change does Y"
    concern: str  # The specific rule description + why it matters
    alternatives: list[str]  # Numbered actionable options
    learn_more: Optional[str]  # Optional link to docs/external resource


class DebateExplanationFormatter:
    """
    Formats AdminApproval explanations by pulling from Codex rules and LoreBook.

    Design principles:
    - Zero fluff: every sentence must be actionable or informative.
    - Friendly but firm: like a senior dev code review.
    - Token-efficient: reuse rule metadata instead of regenerating explanations.
    """

    def __init__(self, codex_client: CodexIndexClient, knowledge_base_path: Optional[Path] = None):
        self.codex_client = codex_client
        self.knowledge_base_path = knowledge_base_path or Path("saga/knowledge")

    def format_approval_request(
        self,
        violated_rules: list[str],
        change_context: ChangeContext,
        alternatives: list[str],
    ) -> FormattedExplanation:
        """
        Generate a perfect explanation for an AdminApprovalRequest.

        Args:
            violated_rules: List of rule IDs from Codex index
            change_context: What the user asked for and what the change does
            alternatives: Suggested compliant options

        Returns:
            FormattedExplanation with all sections populated
        """
        # For MVP, focus on the primary violated rule (first in list)
        primary_rule_id = violated_rules[0] if violated_rules else None

        if not primary_rule_id:
            # Fallback if no rule specified
            return self._fallback_explanation(change_context, alternatives)

        rule = self.codex_client.get_rule(primary_rule_id)
        if not rule:
            return self._fallback_explanation(change_context, alternatives)

        # Build sections
        header = self._build_header(rule)
        context_block = self._build_context_block(change_context)
        concern = self._build_concern(rule)
        formatted_alternatives = self._build_alternatives(alternatives)
        learn_more = self._build_learn_more(rule)

        return FormattedExplanation(
            header=header,
            context_block=context_block,
            concern=concern,
            alternatives=formatted_alternatives,
            learn_more=learn_more,
        )

    def _build_header(self, rule: dict[str, Any]) -> str:
        """Generate header: '⚠️ Conflict: Codex Rule X (Category, Severity)'"""
        severity = rule.get("severity", "WARNING")
        category = rule.get("category", "Unknown")
        title = rule.get("title", "Unnamed Rule")
        rule_id = rule.get("id", "?")

        icon = "⚠️" if severity == "WARNING" else "🚨" if severity == "CRITICAL" else "ℹ️"
        return f"{icon} Conflict: Codex Rule {rule_id} ({category}, {severity})\n{title}"

    def _build_context_block(self, change_context: ChangeContext) -> str:
        """'You asked for X\nThis change: Y'"""
        lines = [f'You asked for: "{change_context.user_request}"']
        if change_context.change_summary:
            lines.append(f"This change: {change_context.change_summary}")
        return "\n".join(lines)

    def _build_concern(self, rule: dict[str, Any]) -> str:
        """Pull rule description + relevant antipattern if available."""
        desc = rule.get("description", "This conflicts with project standards.")

        # If rule has antipatterns, pull the first one's description as added context
        antipatterns = rule.get("antipatterns", [])
        if antipatterns:
            antipattern_desc = antipatterns[0].get("description", "")
            if antipattern_desc:
                desc += f" {antipattern_desc}"

        return f"Concern:\n{desc}"

    def _build_alternatives(self, alternatives: list[str]) -> list[str]:
        """Format alternatives as numbered list."""
        if not alternatives:
            return ["Proceed as proposed (override)", "Cancel this change"]
        return alternatives

    def _build_learn_more(self, rule: dict[str, Any]) -> Optional[str]:
        """If rule has references, return first one; otherwise check knowledge base."""
        references = rule.get("references", [])
        if references:
            return f"Learn more: {references[0]}"

        # Future: check knowledge_base_path for relevant guide based on rule tags
        # For MVP, return None
        return None

    def _fallback_explanation(self, change_context: ChangeContext, alternatives: list[str]) -> FormattedExplanation:
        """Fallback when no rule is found."""
        return FormattedExplanation(
            header="⚠️ Review Required",
            context_block=self._build_context_block(change_context),
            concern="This change requires your review before proceeding.",
            alternatives=self._build_alternatives(alternatives),
            learn_more=None,
        )

    def to_text(self, explanation: FormattedExplanation) -> str:
        """Convert FormattedExplanation to a single readable text block."""
        sections = [
            explanation.header,
            "",
            explanation.context_block,
            "",
            explanation.concern,
            "",
            "Options:",
        ]

        for i, alt in enumerate(explanation.alternatives, 1):
            sections.append(f"{i}. {alt}")

        if explanation.learn_more:
            sections.append("")
            sections.append(explanation.learn_more)

        return "\n".join(sections)
