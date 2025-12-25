"""
Tests for LoreBook Strict Schema (Phase 3.0)
=============================================
Verifies JSON serialization, timestamp enforcement, and new field handling.
"""

from datetime import datetime

from saga.core.lorebook import Decision


def test_decision_serialization_strict_schema():
    """Verify Decision serializes to the mandated JSON schema."""
    decision = Decision(
        trigger="Deviation",
        lesson="Don't use print()",
        context_str="warden.py",
        tags=["python", "logging"],
        user_comments="Use logger instead"
    )

    data = decision.to_dict()

    # Strict Schema Check
    assert "timestamp" in data
    assert "trigger" in data
    assert "lesson" in data
    assert "tags" in data

    # Verify timestamp format
    ts = data["timestamp"]
    assert "T" in ts # Basic ISO check
    assert isinstance(datetime.fromisoformat(ts), datetime)

def test_decision_deserialization():
    """Verify we can reconstruct Decision from JSON dict."""
    raw_data = {
        "decision_id": "test-123",
        "timestamp": "2025-12-24T22:00:00.000000",
        "trigger": "Success",
        "lesson": "Good job",
        "context_str": "test.py",
        "tags": ["test"],
        "user_comments": ""
    }

    decision = Decision.from_dict(raw_data)

    assert decision.decision_id == "test-123"
    assert isinstance(decision.timestamp, datetime)
    assert decision.trigger == "Success"

def test_mixed_legacy_fields():
    """Verify that legacy fields are still handled or allowed."""
    decision = Decision(
        question="What is X?",
        lesson="X is Y"
    )

    data = decision.to_dict()
    assert data["question"] == "What is X?"
    assert data["lesson"] == "X is Y"
