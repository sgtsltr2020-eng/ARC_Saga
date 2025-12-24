"""
SagaCodex Index Client
======================

Provides programmatic access to the `sagacodex_index.json` file.
Used by Warden and Mimiry to query rules, checklists, and patterns.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CodexIndexClient:
    """Read-only client for querying the SagaCodex index."""

    def __init__(self, index_path: Optional[Path] = None) -> None:
        # Default to .saga/sagacodex_index.json relative to CWD if not provided
        self.index_path = index_path if index_path else Path(".saga/sagacodex_index.json")
        self._index_data: dict[str, Any] = {}
        self._rules_map: dict[str, dict[str, Any]] = {}
        self._loaded = False

    def load(self) -> None:
        """Lazily load and parse the index file."""
        if self._loaded:
            return

        if not self.index_path.exists():
            logger.warning(f"Codex index not found at {self.index_path}. Using empty defaults.")
            self._loaded = True
            return

        try:
            with open(self.index_path, encoding="utf-8") as f:
                self._index_data = json.load(f)

            # Build quick lookup map
            for rule in self._index_data.get("rules", []):
                self._rules_map[rule["id"]] = rule

            self._loaded = True
            logger.debug(f"Loaded Codex Index with {len(self._rules_map)} rules.")

        except Exception as e:
            logger.error(f"Failed to load Codex Index: {e}")
            self._index_data = {}
            self._rules_map = {}
            self._loaded = True # Prevent retry loop

    def get_rule(self, rule_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a specific rule by ID."""
        self.load()
        return self._rules_map.get(str(rule_id))

    def find_rules(
        self,
        *,
        tags: Optional[list[str]] = None,
        affected_artifacts: Optional[list[str]] = None,
        category: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """
        Find rules matching criteria.

        Args:
            tags: List of tags (any match)
            affected_artifacts: List of artifact types (any match)
            category: Specific category (exact match)
        """
        self.load()
        results = []

        for rule in self._index_data.get("rules", []):
            match = True

            # Category filter
            if category and rule.get("category") != category:
                match = False

            # Tags filter (OR logic - if any tag matches)
            if match and tags is not None:
                rule_tags = rule.get("tags", [])
                if not any(tag in rule_tags for tag in tags):
                    match = False

            # Artifacts filter (OR logic)
            if match and affected_artifacts:
                rule_artifacts = rule.get("affected_artifacts", [])
                # "all" is a special wildcard in the schema
                if "all" not in rule_artifacts and not any(a in rule_artifacts for a in affected_artifacts):
                    match = False

            if match:
                results.append(rule)

        return results

    def get_checklist_for_task(self, task_tags: list[str]) -> list[str]:
        """
        Generate a checklist based on task tags.
        """
        self.load()
        checklist = []

        # 1. Find relevant rules
        # We assume strict matching for tags to avoid noise
        relevant_rules = self.find_rules(tags=task_tags)

        # 2. Extract unique checklist items
        seen_items = set()
        for rule in relevant_rules:
            item = rule.get("checklist_item")
            if item and item not in seen_items:
                checklist.append(item)
                seen_items.add(item)

        return checklist
