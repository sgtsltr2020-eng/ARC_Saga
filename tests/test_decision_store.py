"""
Tests for DecisionStore (SQLite Backend)
=========================================

Author: ARC SAGA Development Team
Date: December 17, 2025
"""

import asyncio
from pathlib import Path

import pytest

from saga.core.lorebook import Decision, Outcome
from saga.storage.decision_store import DecisionStore


# Helper to run async code synchronously
def run_async(coro):
    return asyncio.run(coro)

def test_creates_database_file(tmp_path: Path):
    """Test database file is created."""
    async def _test():
        store = DecisionStore(str(tmp_path))
        await store.initialize()

        db_path = tmp_path / ".saga" / "lorebook.db"
        assert db_path.exists()

        await store.close()

    run_async(_test())

def test_creates_tables(tmp_path: Path):
    """Test tables are created."""
    async def _test():
        store = DecisionStore(str(tmp_path))
        await store.initialize()

        if not store.db:
            pytest.fail("Connection not open")

        cursor = await store.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [row[0] for row in await cursor.fetchall()]

        assert "decisions" in tables
        assert "outcomes" in tables
        await store.close()

    run_async(_test())

def test_save_decision(tmp_path: Path):
    """Test saving a decision."""
    async def _test():
        store = DecisionStore(str(tmp_path))
        await store.initialize()

        decision = Decision(
            question="Test question",
            mimiry_ruling="Test ruling",
            lorebook_ruling="Test ruling"
        )

        await store.save_decision(decision)

        # Verify saved
        retrieved = await store.get_decision(decision.decision_id)
        assert retrieved is not None
        assert retrieved.question == decision.question
        await store.close()

    run_async(_test())

def test_get_all_decisions(tmp_path: Path):
    """Test retrieving all decisions."""
    async def _test():
        store = DecisionStore(str(tmp_path))
        await store.initialize()

        # Save 3 decisions
        for i in range(3):
            decision = Decision(
                question=f"Question {i}",
                mimiry_ruling=f"Ruling {i}",
                lorebook_ruling=f"Ruling {i}"
            )
            await store.save_decision(decision)

        all_decisions = await store.get_all_decisions()
        assert len(all_decisions) == 3
        await store.close()

    run_async(_test())

def test_update_decision(tmp_path: Path):
    """Test updating a decision."""
    async def _test():
        store = DecisionStore(str(tmp_path))
        await store.initialize()

        decision = Decision(
            question="Original",
            mimiry_ruling="Original",
            lorebook_ruling="Original",
            confidence=0.8
        )
        await store.save_decision(decision)

        # Update confidence
        decision.confidence = 0.9
        decision.outcome_recorded = True
        decision.success = True
        await store.update_decision(decision)

        # Verify updated
        updated = await store.get_decision(decision.decision_id)
        assert updated is not None
        assert updated.confidence == 0.9
        assert updated.outcome_recorded is True
        await store.close()

    run_async(_test())

def test_save_outcome(tmp_path: Path):
    """Test saving an outcome."""
    async def _test():
        store = DecisionStore(str(tmp_path))
        await store.initialize()

        # First save decision
        decision = Decision(
            question="Test",
            mimiry_ruling="Test",
            lorebook_ruling="Test"
        )
        await store.save_decision(decision)

        # Save outcome
        outcome = Outcome(
            decision_id=decision.decision_id,
            success=True,
            metrics={"coverage": 99},
            feedback="Great"
        )
        await store.save_outcome(outcome)

        # Verify
        outcomes = await store.get_all_outcomes()
        assert len(outcomes) == 1
        assert outcomes[0].decision_id == decision.decision_id
        await store.close()

    run_async(_test())
