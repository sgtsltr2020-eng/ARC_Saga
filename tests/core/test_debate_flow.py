"""
Integration Tests for AdminApproval Debate Flow
================================================

Tests the complete debate flow from trigger to resolution.
Validates the implementation against Constitution_v1.md.

Author: ARC SAGA Development Team
Date: December 2025
"""

from unittest.mock import MagicMock

import pytest

from saga.core.admin_approval import (
    AdminApprovalResponse,
    ApprovalDecision,
    DebateRecord,
    TriggerSeverity,
    TriggerType,
)
from saga.core.debate_manager import DebateManager


class TestDebateManagerBasics:
    """Basic DebateManager functionality tests."""

    def test_create_manager(self):
        """Test that DebateManager can be instantiated."""
        manager = DebateManager()
        assert manager is not None
        assert not manager.has_pending_requests()

    def test_create_request(self):
        """Test creating an approval request."""
        manager = DebateManager()

        request = manager.create_request(
            trigger_type=TriggerType.CONSTITUTION,
            trigger_description="Secrets detected in output",
            original_request="Generate API key",
            violated_rules=["Rule 12: Cannot Execute Without Approval"],
            violation_explanation="API keys should not be in code",
            alternatives=["Use environment variables", "Use secret manager"],
            trigger_severity=TriggerSeverity.CRITICAL,
        )

        assert request is not None
        assert request.request_id is not None
        assert request.trigger_type == TriggerType.CONSTITUTION
        assert request.trigger_severity == TriggerSeverity.CRITICAL
        assert "Rule 12" in request.violated_rules[0]
        assert len(request.alternatives) == 2

    def test_pending_requests(self):
        """Test that created requests appear in pending list."""
        manager = DebateManager()

        # Initially no pending
        assert not manager.has_pending_requests()
        assert len(manager.get_pending_requests()) == 0

        # Create a request
        request = manager.create_request(
            trigger_type=TriggerType.CODEX,
            trigger_description="Sync database call",
            original_request="Use db.query()",
            violated_rules=["Rule 2: Async for I/O"],
        )

        # Now should have pending
        assert manager.has_pending_requests()
        pending = manager.get_pending_requests()
        assert len(pending) == 1
        assert pending[0].request_id == request.request_id

    def test_get_request_by_id(self):
        """Test retrieving a specific request by ID."""
        manager = DebateManager()

        request = manager.create_request(
            trigger_type=TriggerType.CONFIDENCE,
            trigger_description="Low confidence",
            original_request="Complex refactoring",
            violated_rules=["Confidence threshold"],
        )

        # Can retrieve by ID
        retrieved = manager.get_request(request.request_id)
        assert retrieved is not None
        assert retrieved.request_id == request.request_id

        # Unknown ID returns None
        unknown = manager.get_request("nonexistent-id")
        assert unknown is None


class TestDebateResolution:
    """Tests for debate resolution flow."""

    def test_approve_response(self):
        """Test approving a request."""
        manager = DebateManager()

        request = manager.create_request(
            trigger_type=TriggerType.CODEX,
            trigger_description="Print statement in code",
            original_request="Add print for debugging",
            violated_rules=["Rule 4: Structured Logging"],
            trigger_severity=TriggerSeverity.INFO,
        )

        response = AdminApprovalResponse(
            request_id=request.request_id,
            decision=ApprovalDecision.APPROVE,
            user_rationale="Temporary debugging, will remove",
        )

        record = manager.handle_response(response)

        assert record is not None
        assert record.user_choice == ApprovalDecision.APPROVE
        assert "Executed original request" in record.final_action

        # Request should be removed from pending
        assert not manager.has_pending_requests()

    def test_modify_response(self):
        """Test modifying a request."""
        manager = DebateManager()

        request = manager.create_request(
            trigger_type=TriggerType.CODEX,
            trigger_description="Bare except clause",
            original_request="except: pass",
            violated_rules=["Rule 3: Custom Exceptions"],
        )

        response = AdminApprovalResponse(
            request_id=request.request_id,
            decision=ApprovalDecision.MODIFY,
            modification_text="except Exception as e: logger.error(e)",
        )

        record = manager.handle_response(response)

        assert record.user_choice == ApprovalDecision.MODIFY
        assert "except Exception as e" in record.final_action

    def test_reject_response(self):
        """Test rejecting a request."""
        manager = DebateManager()

        request = manager.create_request(
            trigger_type=TriggerType.CONSTITUTION,
            trigger_description="Delete all files",
            original_request="rm -rf /",
            violated_rules=["Rule 12: Cannot Execute Without Approval"],
            trigger_severity=TriggerSeverity.CRITICAL,
        )

        response = AdminApprovalResponse(
            request_id=request.request_id,
            decision=ApprovalDecision.REJECT,
        )

        record = manager.handle_response(response)

        assert record.user_choice == ApprovalDecision.REJECT
        assert "no action taken" in record.final_action.lower()

    def test_unknown_request_raises(self):
        """Test that responding to unknown request raises ValueError."""
        manager = DebateManager()

        response = AdminApprovalResponse(
            request_id="nonexistent-id",
            decision=ApprovalDecision.REJECT,
        )

        with pytest.raises(ValueError, match="Request not found"):
            manager.handle_response(response)


class TestLoreBookIntegration:
    """Tests for LoreBook recording integration."""

    def test_constitution_override_recorded(self):
        """Test that Constitution overrides are always recorded.

        Per Constitution_v1.md Section 5.1:
        "User approves Constitution override" → MUST record.
        """
        mock_recorder = MagicMock()
        manager = DebateManager(lorebook_recorder=mock_recorder)

        request = manager.create_request(
            trigger_type=TriggerType.CONSTITUTION,
            trigger_description="Secrets detected",
            original_request="Include API key in config.py",
            violated_rules=["Rule 12: Cannot Execute Without Approval"],
            trigger_severity=TriggerSeverity.CRITICAL,
        )

        response = AdminApprovalResponse(
            request_id=request.request_id,
            decision=ApprovalDecision.APPROVE,
            user_rationale="This is a development environment",
        )

        manager.handle_response(response)

        # Recorder should have been called
        mock_recorder.assert_called_once()

        # Check the recorded decision
        recorded = mock_recorder.call_args[0][0]
        assert isinstance(recorded, DebateRecord)
        assert recorded.trigger_type == TriggerType.CONSTITUTION
        assert recorded.user_choice == ApprovalDecision.APPROVE
        assert recorded.should_generalize is True

    def test_codex_override_with_rationale_recorded(self):
        """Test that Codex overrides with reasoning are recorded.

        Per Constitution_v1.md Section 5.1:
        "User approves Codex override with reasoning" → MUST record.
        """
        mock_recorder = MagicMock()
        manager = DebateManager(lorebook_recorder=mock_recorder)

        request = manager.create_request(
            trigger_type=TriggerType.CODEX,
            trigger_description="Sync database call",
            original_request="Use db.query() for this script",
            violated_rules=["Rule 2: Async for I/O"],
            trigger_severity=TriggerSeverity.WARNING,
        )

        response = AdminApprovalResponse(
            request_id=request.request_id,
            decision=ApprovalDecision.APPROVE,
            user_rationale="This is a standalone CLI script, not FastAPI",
        )

        manager.handle_response(response)

        # Recorder should have been called
        mock_recorder.assert_called_once()

    def test_reject_without_rationale_not_recorded(self):
        """Test that rejections without rationale are not recorded.

        Per Constitution_v1.md Section 5.3:
        "User rejects without explanation" → MUST NOT record.
        """
        mock_recorder = MagicMock()
        manager = DebateManager(lorebook_recorder=mock_recorder)

        request = manager.create_request(
            trigger_type=TriggerType.CODEX,
            trigger_description="Some violation",
            original_request="Do something",
            violated_rules=["Some rule"],
            trigger_severity=TriggerSeverity.INFO,  # Low severity
        )

        response = AdminApprovalResponse(
            request_id=request.request_id,
            decision=ApprovalDecision.REJECT,
            # No rationale provided
        )

        manager.handle_response(response)

        # Recorder should NOT have been called
        mock_recorder.assert_not_called()


class TestEndToEndDebateFlow:
    """End-to-end integration test for the complete debate flow."""

    def test_complete_debate_flow(self):
        """Test the complete debate flow from trigger to resolution.

        Scenario per Constitution_v1.md Example 1:
        1. User requests: "Delete all Python files"
        2. SAGA triggers Constitution violation
        3. User approves with modification
        4. Decision is recorded to LoreBook
        """
        recorded_decisions: list[DebateRecord] = []

        def capture_recorder(record: DebateRecord) -> None:
            recorded_decisions.append(record)

        manager = DebateManager(lorebook_recorder=capture_recorder)

        # Step 1: Trigger is detected, request created
        request = manager.create_request(
            trigger_type=TriggerType.CONSTITUTION,
            trigger_description="Destructive operation detected",
            original_request="Delete all Python files in the project",
            violated_rules=[
                "Rule 12: Cannot Execute Without Approval",
            ],
            violation_explanation=(
                "This is a destructive operation that requires explicit "
                "approval with preview. 47 Python files would be deleted."
            ),
            alternatives=[
                "Move files to .archive/ instead of deleting",
                "Delete only generated/temporary files (12 files)",
                "Show the file list first, then confirm",
            ],
            risk_statement="47 Python files will be permanently deleted",
            trigger_severity=TriggerSeverity.CRITICAL,
            trace_id="test-trace-123",
        )

        # Step 2: Verify pending request exists
        assert manager.has_pending_requests()
        pending = manager.get_pending_requests()
        assert len(pending) == 1
        assert pending[0].trigger_severity == TriggerSeverity.CRITICAL

        # Step 3: User chooses to modify
        response = AdminApprovalResponse(
            request_id=request.request_id,
            decision=ApprovalDecision.MODIFY,
            modification_text="Move files to .archive/ instead",
            user_rationale="I want to keep backups just in case",
        )

        record = manager.handle_response(response)

        # Step 4: Verify resolution
        assert not manager.has_pending_requests()
        assert record.user_choice == ApprovalDecision.MODIFY
        assert "Move files to .archive/" in record.final_action
        assert record.trace_id == "test-trace-123"

        # Step 5: Verify LoreBook recording
        assert len(recorded_decisions) == 1
        recorded = recorded_decisions[0]
        assert recorded.trigger_type == TriggerType.CONSTITUTION
        assert "Rule 12" in recorded.violated_rules[0]
        assert recorded.should_generalize is True


class TestTagExtraction:
    """Tests for tag extraction from requests."""

    def test_tags_include_trigger_type(self):
        """Test that tags include the trigger type."""
        manager = DebateManager()

        request = manager.create_request(
            trigger_type=TriggerType.CODEX,
            trigger_description="Test",
            original_request="Test",
            violated_rules=["Rule 1: Type Safety"],
        )

        response = AdminApprovalResponse(
            request_id=request.request_id,
            decision=ApprovalDecision.REJECT,
            user_rationale="Testing tags",
        )

        record = manager.handle_response(response)

        assert "codex" in record.tags
        assert "rule_1" in record.tags


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
