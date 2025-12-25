"""
Memory Janitor - Maintenance & Multi-Project Sync
===================================================

Handles Mythos compression, global wisdom sync, and cold storage
archival for long-term memory health.

Author: ARC SAGA Development Team
Date: December 25, 2025
Status: USMA Phase 7 - Maintenance Layer
"""

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from saga.core.memory.mythos import (
    ArchitecturalDebt,
    MythosLibrary,
    SolvedPattern,
)
from saga.core.memory.researcher import QuerySanitizer

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# PROJECT EPOCHS (Temporal Anchoring)
# ═══════════════════════════════════════════════════════════════

@dataclass
class ProjectEpoch:
    """
    A named time period in the project's history.

    Examples:
    - "The Genesis Era" (project creation)
    - "The Great Refactor of Dec 2025"
    - "The Pre-Database Era"
    """
    epoch_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    start_date: datetime = field(default_factory=datetime.utcnow)
    end_date: datetime | None = None

    # Linked entries
    lore_entry_ids: list[str] = field(default_factory=list)
    mythos_chapter_ids: list[str] = field(default_factory=list)

    # Metadata
    milestone_events: list[str] = field(default_factory=list)

    def is_active(self) -> bool:
        """Check if epoch is currently active."""
        return self.end_date is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch_id": self.epoch_id,
            "name": self.name,
            "description": self.description,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "lore_entry_ids": self.lore_entry_ids,
            "mythos_chapter_ids": self.mythos_chapter_ids,
            "milestone_events": self.milestone_events
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectEpoch":
        data["start_date"] = datetime.fromisoformat(data["start_date"])
        if data.get("end_date"):
            data["end_date"] = datetime.fromisoformat(data["end_date"])
        return cls(**data)


class EpochManager:
    """
    Manages project epochs for temporal anchoring.

    Groups events into logical time periods for
    historical storytelling and flashbacks.
    """

    def __init__(self, storage_path: Path | str | None = None):
        """Initialize the epoch manager."""
        self.storage_path = Path(storage_path) if storage_path else None
        self.epochs: list[ProjectEpoch] = []
        self._current_epoch: ProjectEpoch | None = None

        if self.storage_path and self.storage_path.exists():
            self._load()

    def create_epoch(
        self,
        name: str,
        description: str = "",
        start_date: datetime | None = None
    ) -> ProjectEpoch:
        """
        Create a new project epoch.

        Closes the current active epoch if one exists.
        """
        # Close current epoch
        if self._current_epoch and self._current_epoch.is_active():
            self._current_epoch.end_date = start_date or datetime.utcnow()

        epoch = ProjectEpoch(
            name=name,
            description=description,
            start_date=start_date or datetime.utcnow()
        )

        self.epochs.append(epoch)
        self._current_epoch = epoch
        self._save()

        logger.info(f"Created epoch: {name}")
        return epoch

    def get_current_epoch(self) -> ProjectEpoch | None:
        """Get the currently active epoch."""
        return self._current_epoch

    def get_epoch_for_date(self, date: datetime) -> ProjectEpoch | None:
        """Find the epoch that contains a specific date."""
        for epoch in self.epochs:
            if epoch.start_date <= date:
                if epoch.end_date is None or date <= epoch.end_date:
                    return epoch
        return None

    def link_lore_entry(self, epoch_id: str, lore_id: str) -> bool:
        """Link a lore entry to an epoch."""
        for epoch in self.epochs:
            if epoch.epoch_id == epoch_id:
                if lore_id not in epoch.lore_entry_ids:
                    epoch.lore_entry_ids.append(lore_id)
                    self._save()
                return True
        return False

    def get_timeline(self) -> list[dict[str, Any]]:
        """Get a timeline of all epochs."""
        return [
            {
                "name": e.name,
                "start": e.start_date.strftime("%Y-%m-%d"),
                "end": e.end_date.strftime("%Y-%m-%d") if e.end_date else "Present",
                "entries": len(e.lore_entry_ids),
                "active": e.is_active()
            }
            for e in sorted(self.epochs, key=lambda x: x.start_date)
        ]

    def _save(self) -> None:
        """Persist epochs to storage."""
        if not self.storage_path:
            return

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "epochs": [e.to_dict() for e in self.epochs],
            "current_epoch_id": self._current_epoch.epoch_id if self._current_epoch else None
        }
        self.storage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self) -> None:
        """Load epochs from storage."""
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            self.epochs = [ProjectEpoch.from_dict(e) for e in data.get("epochs", [])]

            current_id = data.get("current_epoch_id")
            if current_id:
                for epoch in self.epochs:
                    if epoch.epoch_id == current_id:
                        self._current_epoch = epoch
                        break
        except Exception as e:
            logger.warning(f"Failed to load epochs: {e}")


# ═══════════════════════════════════════════════════════════════
# COLD STORAGE (SQLite Archive)
# ═══════════════════════════════════════════════════════════════

class ColdStorage:
    """
    SQLite-based cold storage for archived patterns and debt.

    Stores compressed/deduplicated wisdom that is no longer
    in active use but should be preserved for traceability.
    """

    def __init__(self, db_path: Path | str):
        """Initialize cold storage."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS archived_patterns (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    example TEXT,
                    original_source_id TEXT,
                    archived_at TEXT,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS archived_debt (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    severity TEXT,
                    resolved_at TEXT,
                    resolution_notes TEXT,
                    original_source_id TEXT,
                    archived_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS compression_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    compressed_at TEXT,
                    patterns_before INTEGER,
                    patterns_after INTEGER,
                    debt_archived INTEGER,
                    notes TEXT
                )
            """)
            conn.commit()

    def archive_pattern(
        self,
        pattern: SolvedPattern,
        original_source_id: str = ""
    ) -> None:
        """Archive a pattern to cold storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO archived_patterns
                (id, name, description, example, original_source_id, archived_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern.pattern_id,
                pattern.name,
                pattern.description,
                pattern.example,
                original_source_id,
                datetime.utcnow().isoformat(),
                json.dumps({"tags": pattern.tags})
            ))
            conn.commit()

    def archive_resolved_debt(
        self,
        debt: ArchitecturalDebt,
        resolution_notes: str = ""
    ) -> None:
        """Archive resolved architectural debt."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO archived_debt
                (id, name, description, severity, resolved_at, resolution_notes, original_source_id, archived_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                debt.debt_id,
                debt.name,
                debt.description,
                debt.severity,
                datetime.utcnow().isoformat(),
                resolution_notes,
                "",
                datetime.utcnow().isoformat()
            ))
            conn.commit()

    def log_compression(
        self,
        patterns_before: int,
        patterns_after: int,
        debt_archived: int,
        notes: str = ""
    ) -> None:
        """Log a compression event."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO compression_log
                (compressed_at, patterns_before, patterns_after, debt_archived, notes)
                VALUES (?, ?, ?, ?, ?)
            """, (
                datetime.utcnow().isoformat(),
                patterns_before,
                patterns_after,
                debt_archived,
                notes
            ))
            conn.commit()

    def get_archived_pattern_count(self) -> int:
        """Get count of archived patterns."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM archived_patterns")
            return cursor.fetchone()[0]

    def search_archived_patterns(self, query: str) -> list[dict[str, Any]]:
        """Search archived patterns by name or description."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, name, description, example, archived_at
                FROM archived_patterns
                WHERE name LIKE ? OR description LIKE ?
                LIMIT 20
            """, (f"%{query}%", f"%{query}%"))

            return [
                {"id": row[0], "name": row[1], "description": row[2],
                 "example": row[3], "archived_at": row[4]}
                for row in cursor.fetchall()
            ]


# ═══════════════════════════════════════════════════════════════
# MEMORY JANITOR
# ═══════════════════════════════════════════════════════════════

class MemoryJanitor:
    """
    Maintains Mythos health through compression and cleanup.

    Features:
    - Pattern deduplication
    - Mythos compression when token threshold exceeded
    - Cold storage archival for paid-off debt
    - Preserves original source IDs for traceability
    """

    def __init__(
        self,
        mythos: MythosLibrary,
        cold_storage: ColdStorage | None = None,
        token_threshold: int = 50000,
        similarity_threshold: float = 0.8
    ):
        """Initialize the janitor."""
        self.mythos = mythos
        self.cold_storage = cold_storage
        self.token_threshold = token_threshold
        self.similarity_threshold = similarity_threshold

    def estimate_token_count(self) -> int:
        """Estimate total tokens in the Mythos."""
        total_chars = 0

        for chapter in self.mythos.chapters:
            total_chars += len(chapter.title) + len(chapter.summary)
            for pattern in chapter.solved_patterns:
                total_chars += len(pattern.name) + len(pattern.description)
            for debt in chapter.architectural_debt:
                total_chars += len(debt.name) + len(debt.description)
            for principle in chapter.universal_principles:
                total_chars += len(principle)

        # Rough estimate: 4 chars per token
        return total_chars // 4

    def needs_compression(self) -> bool:
        """Check if Mythos needs compression."""
        return self.estimate_token_count() > self.token_threshold

    def find_duplicate_patterns(self) -> list[tuple[SolvedPattern, SolvedPattern]]:
        """Find patterns that are semantically similar."""
        all_patterns: list[SolvedPattern] = []

        for chapter in self.mythos.chapters:
            all_patterns.extend(chapter.solved_patterns)

        duplicates: list[tuple[SolvedPattern, SolvedPattern]] = []

        for i, p1 in enumerate(all_patterns):
            for p2 in all_patterns[i + 1:]:
                similarity = self._compute_similarity(p1, p2)
                if similarity >= self.similarity_threshold:
                    duplicates.append((p1, p2))

        return duplicates

    def _compute_similarity(self, p1: SolvedPattern, p2: SolvedPattern) -> float:
        """Compute Jaccard similarity between two patterns."""
        words1 = set(p1.name.lower().split() + p1.description.lower().split())
        words2 = set(p2.name.lower().split() + p2.description.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def compress_mythos(self) -> dict[str, Any]:
        """
        Compress the Mythos by deduplicating patterns.

        Returns:
            Compression statistics
        """
        patterns_before = sum(len(c.solved_patterns) for c in self.mythos.chapters)
        debt_before = sum(len(c.architectural_debt) for c in self.mythos.chapters)

        # Find duplicates
        duplicates = self.find_duplicate_patterns()
        patterns_to_remove: set[str] = set()

        for p1, p2 in duplicates:
            # Keep the longer/more detailed one
            if len(p1.description) >= len(p2.description):
                patterns_to_remove.add(p2.pattern_id)
                if self.cold_storage:
                    self.cold_storage.archive_pattern(p2, original_source_id=p1.pattern_id)
            else:
                patterns_to_remove.add(p1.pattern_id)
                if self.cold_storage:
                    self.cold_storage.archive_pattern(p1, original_source_id=p2.pattern_id)

        # Remove duplicates from chapters
        for chapter in self.mythos.chapters:
            chapter.solved_patterns = [
                p for p in chapter.solved_patterns
                if p.pattern_id not in patterns_to_remove
            ]

        patterns_after = sum(len(c.solved_patterns) for c in self.mythos.chapters)

        # Log compression
        if self.cold_storage:
            self.cold_storage.log_compression(
                patterns_before=patterns_before,
                patterns_after=patterns_after,
                debt_archived=0,
                notes=f"Removed {len(patterns_to_remove)} duplicate patterns"
            )

        return {
            "patterns_before": patterns_before,
            "patterns_after": patterns_after,
            "duplicates_removed": len(patterns_to_remove),
            "token_estimate": self.estimate_token_count()
        }

    def archive_resolved_debt(self, debt_ids: list[str], resolution_notes: str = "") -> int:
        """Archive debt that has been resolved."""
        archived = 0

        for chapter in self.mythos.chapters:
            remaining_debt: list[ArchitecturalDebt] = []

            for debt in chapter.architectural_debt:
                if debt.debt_id in debt_ids:
                    if self.cold_storage:
                        self.cold_storage.archive_resolved_debt(debt, resolution_notes)
                    archived += 1
                else:
                    remaining_debt.append(debt)

            chapter.architectural_debt = remaining_debt

        return archived


# ═══════════════════════════════════════════════════════════════
# GLOBAL WISDOM BRIDGE
# ═══════════════════════════════════════════════════════════════

class GlobalWisdomBridge:
    """
    Enables cross-project wisdom sharing.

    Stores high-utility patterns in a global directory
    (~/.saga/global_wisdom/) that can be imported into
    any new project's ContextWarden.
    """

    DEFAULT_PATH = Path.home() / ".saga" / "global_wisdom"

    def __init__(
        self,
        global_path: Path | str | None = None,
        sanitizer: QuerySanitizer | None = None
    ):
        """Initialize the global bridge."""
        self.global_path = Path(global_path) if global_path else self.DEFAULT_PATH
        self.global_path.mkdir(parents=True, exist_ok=True)

        self.sanitizer = sanitizer or QuerySanitizer()
        self._wisdom_file = self.global_path / "patterns.json"
        self._metadata_file = self.global_path / "metadata.json"

        self._patterns: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """Load global patterns."""
        if self._wisdom_file.exists():
            try:
                self._patterns = json.loads(
                    self._wisdom_file.read_text(encoding="utf-8")
                )
            except Exception as e:
                logger.warning(f"Failed to load global wisdom: {e}")

    def _save(self) -> None:
        """Save global patterns."""
        self._wisdom_file.write_text(
            json.dumps(self._patterns, indent=2),
            encoding="utf-8"
        )

        # Update metadata
        metadata = {
            "last_updated": datetime.utcnow().isoformat(),
            "pattern_count": len(self._patterns),
            "version": "1.0"
        }
        self._metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def export_pattern(
        self,
        pattern: SolvedPattern,
        source_project: str,
        utility_score: float
    ) -> bool:
        """
        Export a high-utility pattern to global wisdom.

        Sanitizes the pattern to remove project-specific data.
        """
        # Sanitize description and example
        sanitized_desc = self.sanitizer.sanitize(pattern.description)
        sanitized_example = self.sanitizer.sanitize(pattern.example)

        # Check for duplicates
        for existing in self._patterns:
            if existing["name"] == pattern.name:
                # Update if higher utility
                if utility_score > existing.get("utility_score", 0):
                    existing["description"] = sanitized_desc
                    existing["utility_score"] = utility_score
                    self._save()
                return True

        # Add new pattern
        global_pattern = {
            "id": f"global_{pattern.pattern_id}",
            "name": pattern.name,
            "description": sanitized_desc,
            "example": sanitized_example,
            "source_project": hashlib.sha256(source_project.encode()).hexdigest()[:8],
            "utility_score": utility_score,
            "exported_at": datetime.utcnow().isoformat(),
            "tags": pattern.tags
        }

        self._patterns.append(global_pattern)
        self._save()

        logger.info(f"Exported pattern to global wisdom: {pattern.name}")
        return True

    def import_patterns(
        self,
        min_utility: float = 0.7,
        max_count: int = 10
    ) -> list[SolvedPattern]:
        """
        Import high-utility patterns from global wisdom.

        Returns:
            List of SolvedPattern objects
        """
        eligible = [
            p for p in self._patterns
            if p.get("utility_score", 0) >= min_utility
        ]

        # Sort by utility
        eligible.sort(key=lambda x: x.get("utility_score", 0), reverse=True)

        patterns: list[SolvedPattern] = []
        for p in eligible[:max_count]:
            patterns.append(SolvedPattern(
                pattern_id=p["id"],
                name=f"[Global] {p['name']}",
                description=p["description"],
                example=p.get("example", ""),
                tags=p.get("tags", [])
            ))

        return patterns

    def get_pattern_count(self) -> int:
        """Get count of global patterns."""
        return len(self._patterns)

    def sync_from_optimizer(
        self,
        optimizer: Any,  # SovereignOptimizer
        mythos: MythosLibrary,
        project_name: str,
        utility_threshold: float = 0.8
    ) -> int:
        """
        Sync high-utility patterns from optimizer to global.

        Returns:
            Number of patterns exported
        """
        exported = 0

        for chapter in mythos.chapters:
            for pattern in chapter.solved_patterns:
                # Check if pattern has high utility (simplistic check)
                # In production, query optimizer's internal scores
                if len(pattern.description) > 50:  # Has substantial content
                    self.export_pattern(
                        pattern=pattern,
                        source_project=project_name,
                        utility_score=0.85  # Would come from optimizer
                    )
                    exported += 1

        return exported

    def clear_global_wisdom(self) -> None:
        """Clear all global patterns (for testing)."""
        self._patterns = []
        self._save()
