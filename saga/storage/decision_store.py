"""
Decision Store - SQLite Backend for LoreBook
============================================

Persists SAGA's decisions and outcomes using async SQLite.
Handles JSON serialization for complex fields.

Author: ARC SAGA Development Team
Date: December 17, 2025
Status: Phase 3A - Storage Layer
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import aiosqlite

from saga.core.lorebook import Decision, Outcome

logger = logging.getLogger(__name__)


class DecisionStore:
    """
    Async SQLite persistence for LoreBook.

    Schema:
        - decisions: Main decision record
        - outcomes: Results and feedback
    """

    def __init__(self, project_root: str = ".") -> None:
        """Initialize store with path to database."""
        self.db_path = Path(project_root) / ".saga" / "lorebook.db"
        self.db: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        """Create tables if they don't exist."""
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.db = await aiosqlite.connect(self.db_path)
        self.db.row_factory = aiosqlite.Row

        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                decision_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                trace_id TEXT,
                question TEXT NOT NULL,
                context JSON,
                mimiry_ruling TEXT,
                lorebook_ruling TEXT,
                divergence_rationale TEXT,
                confidence REAL,
                cited_rules JSON,
                tags JSON,
                outcome_recorded INTEGER DEFAULT 0,
                success INTEGER
            )
        """)

        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS outcomes (
                outcome_id TEXT PRIMARY KEY,
                decision_id TEXT NOT NULL,
                learned_at TEXT NOT NULL,
                success INTEGER,
                metrics JSON,
                feedback TEXT,
                FOREIGN KEY(decision_id) REFERENCES decisions(decision_id)
            )
        """)

        await self.db.commit()
        logger.info(f"DecisionStore initialized at {self.db_path}")

    async def save_decision(self, decision: Decision) -> None:
        """Save a new decision."""
        if not self.db:
            raise RuntimeError("Database not initialized")

        await self.db.execute(
            """
            INSERT INTO decisions (
                decision_id, timestamp, trace_id, question, context,
                mimiry_ruling, lorebook_ruling, divergence_rationale,
                confidence, cited_rules, tags, outcome_recorded, success
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision.decision_id,
                decision.timestamp.isoformat(),
                decision.trace_id,
                decision.question,
                json.dumps(decision.context),
                decision.mimiry_ruling,
                decision.lorebook_ruling,
                decision.divergence_rationale,
                decision.confidence,
                json.dumps(decision.cited_rules),
                json.dumps(decision.tags),
                1 if decision.outcome_recorded else 0,
                1 if decision.success else (0 if decision.success is False else None)
            )
        )
        await self.db.commit()

    async def update_decision(self, decision: Decision) -> None:
        """Update an existing decision (e.g. after outcome)."""
        if not self.db:
            raise RuntimeError("Database not initialized")

        await self.db.execute(
            """
            UPDATE decisions SET
                confidence = ?,
                outcome_recorded = ?,
                success = ?
            WHERE decision_id = ?
            """,
            (
                decision.confidence,
                1 if decision.outcome_recorded else 0,
                1 if decision.success else (0 if decision.success is False else None),
                decision.decision_id
            )
        )
        await self.db.commit()

    async def get_decision(self, decision_id: str) -> Optional[Decision]:
        """Retrieve a decision by ID."""
        if not self.db:
            raise RuntimeError("Database not initialized")

        async with self.db.execute(
            "SELECT * FROM decisions WHERE decision_id = ?",
            (decision_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return self._row_to_decision(row)

    async def get_all_decisions(self) -> list[Decision]:
        """Retrieve all decisions."""
        if not self.db:
            raise RuntimeError("Database not initialized")

        async with self.db.execute("SELECT * FROM decisions") as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_decision(row) for row in rows]

    async def get_recent_decisions(self, limit: int = 10, tag: Optional[str] = None) -> list[Decision]:
        """Retrieve recent decisions, optionally filtered by tag."""
        if not self.db:
            raise RuntimeError("Database not initialized")

        query = "SELECT * FROM decisions"
        params: list[Any] = []

        if tag:
            # Simple JSON array check using LIKE for SQLite (limitations apply)
            # Ideal: SQLite JSON1 extension, but keep simple for now
            query += " WHERE tags LIKE ?"
            params.append(f'%"{tag}"%')

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        async with self.db.execute(query, tuple(params)) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_decision(row) for row in rows]

    async def save_outcome(self, outcome: Outcome) -> None:
        """Save an outcome."""
        if not self.db:
            raise RuntimeError("Database not initialized")

        await self.db.execute(
            """
            INSERT INTO outcomes (
                outcome_id, decision_id, learned_at, success, metrics, feedback
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                outcome.outcome_id,
                outcome.decision_id,
                outcome.learned_at.isoformat(),
                1 if outcome.success else 0,
                json.dumps(outcome.metrics),
                outcome.feedback
            )
        )
        await self.db.commit()

    async def get_all_outcomes(self) -> list[Outcome]:
        """Retrieve all outcomes."""
        if not self.db:
            raise RuntimeError("Database not initialized")

        async with self.db.execute("SELECT * FROM outcomes") as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_outcome(row) for row in rows]

    async def close(self) -> None:
        """Close database connection."""
        if self.db:
            await self.db.close()
            self.db = None

    def _row_to_decision(self, row: aiosqlite.Row) -> Decision:
        """Convert DB row to Decision object."""
        return Decision(
            decision_id=row["decision_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            trace_id=row["trace_id"] or "",
            question=row["question"],
            context=json.loads(row["context"]),
            mimiry_ruling=row["mimiry_ruling"] or "",
            lorebook_ruling=row["lorebook_ruling"] or "",
            divergence_rationale=row["divergence_rationale"] or "",
            confidence=row["confidence"],
            cited_rules=json.loads(row["cited_rules"]),
            tags=json.loads(row["tags"]),
            outcome_recorded=bool(row["outcome_recorded"]),
            success=bool(row["success"]) if row["success"] is not None else None
        )

    def _row_to_outcome(self, row: aiosqlite.Row) -> Outcome:
        """Convert DB row to Outcome object."""
        return Outcome(
            outcome_id=row["outcome_id"],
            decision_id=row["decision_id"],
            learned_at=datetime.fromisoformat(row["learned_at"]),
            success=bool(row["success"]),
            metrics=json.loads(row["metrics"]),
            feedback=row["feedback"] or ""
        )
