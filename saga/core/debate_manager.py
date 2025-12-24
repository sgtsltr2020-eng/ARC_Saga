"""
DebateManager - AdminApproval Conflict Resolution
==================================================

Manages the debate protocol for SAGA's conflict resolution system.
Implements the flow defined in docs/Constitution_v1.md Section 3-4.

The DebateManager is responsible for:
1. Creating approval requests from triggers
2. Storing pending requests (in-memory for MVP)
3. Handling user responses
4. Recording decisions to LoreBook

Author: ARC SAGA Development Team
Date: December 2025
Status: MVP Implementation
"""

import logging
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any, Optional

from saga.core.admin_approval import (
    AdminApprovalRequest,
    AdminApprovalResponse,
    ApprovalDecision,
    DebateRecord,
    TriggerSeverity,
    TriggerType,
)
from saga.core.codex_index_client import CodexIndexClient
from saga.core.debate_formatter import ChangeContext, DebateExplanationFormatter

logger = logging.getLogger(__name__)


class DebateManager:
    """Manages the AdminApproval debate protocol.

    The DebateManager is the central coordinator for conflict resolution
    in SAGA. When a Constitution or Codex violation is detected, the
    DebateManager creates an AdminApprovalRequest and waits for user input.

    MVP Constraints (per Constitution_v1.md Section 6.3):
    - No auto-resolution: All conflicts go to AdminApproval
    - No weighted voting: Single user decision is final
    - No delegation: User cannot delegate approval to agents
    - No time-based expiry: AdminApproval blocks until resolved

    Usage:
        manager = DebateManager()

        # Create a request from a trigger
        request = manager.create_request(
            trigger_type=TriggerType.CONSTITUTION,
            trigger_description="Destructive operation detected",
            original_request="Delete all files",
            violated_rules=["Rule 12: Cannot Execute Without Approval"],
            alternatives=["Move to archive", "Delete only temp files"]
        )

        # Get pending requests
        pending = manager.get_pending_requests()

        # Handle user response
        result = manager.handle_response(AdminApprovalResponse(
            request_id=request.request_id,
            decision=ApprovalDecision.REJECT
        ))

    Attributes:
        _pending_requests: In-memory store of pending approval requests
        _resolved_records: Resolved debates (for audit trail)
        _lorebook_recorder: Callback to record decisions in LoreBook
    """

    def __init__(
        self,
        lorebook_recorder: Optional[Callable[[DebateRecord], None]] = None
    ):
        """Initialize the DebateManager.

        Args:
            lorebook_recorder: Optional callback to record decisions.
                               Signature: (DebateRecord) -> None
                               If not provided, uses a stub that logs.
        """
        self._pending_requests: dict[str, AdminApprovalRequest] = {}
        self._resolved_records: list[DebateRecord] = []
        self._lorebook_recorder = lorebook_recorder or self._stub_lorebook_recorder

        # Initialize formatter with Codex client
        self.formatter = DebateExplanationFormatter(codex_client=CodexIndexClient())

        logger.info("DebateManager initialized")

    def _stub_lorebook_recorder(self, record: DebateRecord) -> None:
        """Stub LoreBook recorder for MVP.

        In production, this would call LoreBook.record_decision().
        For now, it just logs the decision.
        """
        logger.info(
            "LoreBook recording (stub)",
            extra={
                "decision_id": record.decision_id,
                "trigger_type": record.trigger_type.value,
                "user_choice": record.user_choice.value,
                "should_generalize": record.should_generalize,
            }
        )

    def create_request(
        self,
        trigger_type: TriggerType,
        trigger_description: str,
        original_request: str,
        violated_rules: list[str],
        violation_explanation: str = "",
        alternatives: Optional[list[str]] = None,
        risk_statement: Optional[str] = None,
        trigger_severity: TriggerSeverity = TriggerSeverity.WARNING,
        trace_id: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> AdminApprovalRequest:
        """Create an AdminApprovalRequest from a trigger.

        This is called when SAGA detects a conflict that requires
        user approval. The request is stored and can be retrieved
        via get_pending_requests().

        Args:
            trigger_type: What triggered the debate
            trigger_description: Human-readable description
            original_request: What the user originally asked
            violated_rules: List of rule names/numbers violated
            violation_explanation: Why this conflicts with rules
            alternatives: Compliant alternatives to offer
            risk_statement: What happens if user approves
            trigger_severity: CRITICAL, WARNING, or INFO
            trace_id: Distributed tracing ID (generated if not provided)
            context: Additional context for the request

        Returns:
            The created AdminApprovalRequest

        Example:
            >>> request = manager.create_request(
            ...     trigger_type=TriggerType.CODEX,
            ...     trigger_description="Sync database call in async context",
            ...     original_request="Use db.query() for simplicity",
            ...     violated_rules=["Rule 2: Async for I/O Operations"],
            ...     alternatives=["Use await db.execute()"]
            ... )
        """
        request_id = str(uuid.uuid4())
        trace_id = trace_id or str(uuid.uuid4())
        safe_context = context or {}

        # Use formatter if appropriate (Codex trigger with rules)
        if trigger_type == TriggerType.CODEX and violated_rules:
            # Try to get change summary from context, fallback to trigger_description
            change_summary = safe_context.get("change_summary", trigger_description)

            change_ctx = ChangeContext(
                user_request=original_request,
                change_summary=change_summary,
                diff_size=safe_context.get("diff_size"),
            )

            formatted = self.formatter.format_approval_request(
                violated_rules=violated_rules,
                change_context=change_ctx,
                alternatives=alternatives or [],
            )

            # Override/Populate fields with formatted content
            trigger_description = formatted.header
            violation_explanation = self.formatter.to_text(formatted)

            # If formatted has alternatives and we didn't have them, use them
            if not alternatives and formatted.alternatives:
                 alternatives = formatted.alternatives

        request = AdminApprovalRequest(
            request_id=request_id,
            trace_id=trace_id,
            timestamp=datetime.utcnow(),
            trigger_type=trigger_type,
            trigger_severity=trigger_severity,
            trigger_description=trigger_description,
            original_request=original_request,
            violated_rules=violated_rules,
            violation_explanation=violation_explanation,
            alternatives=alternatives or [],
            risk_statement=risk_statement,
            context=safe_context,
        )

        self._pending_requests[request_id] = request

        logger.warning(
            "AdminApproval request created",
            extra={
                "request_id": request_id,
                "trigger_type": trigger_type.value,
                "trigger_severity": trigger_severity.value,
                "violated_rules": violated_rules,
            }
        )

        return request

    def get_pending_requests(self) -> list[AdminApprovalRequest]:
        """Get all pending approval requests.

        Returns:
            List of AdminApprovalRequest objects awaiting user decision
        """
        return list(self._pending_requests.values())

    def get_request(self, request_id: str) -> Optional[AdminApprovalRequest]:
        """Get a specific approval request by ID.

        Args:
            request_id: The request ID to look up

        Returns:
            The AdminApprovalRequest if found, None otherwise
        """
        return self._pending_requests.get(request_id)

    def has_pending_requests(self) -> bool:
        """Check if there are any pending approval requests.

        Returns:
            True if there are pending requests, False otherwise
        """
        return len(self._pending_requests) > 0

    def handle_response(
        self,
        response: AdminApprovalResponse
    ) -> DebateRecord:
        """Handle a user response to an approval request.

        This processes the user's decision, creates a DebateRecord,
        and records it to LoreBook if applicable.

        Args:
            response: The user's response

        Returns:
            The DebateRecord created from this resolution

        Raises:
            ValueError: If request_id not found in pending requests

        See Constitution_v1.md Section 4.2 for response handling rules.
        """
        request = self._pending_requests.get(response.request_id)
        if request is None:
            raise ValueError(f"Request not found: {response.request_id}")

        # Determine final action based on decision
        final_action = self._determine_final_action(request, response)

        # Create the debate record
        record = DebateRecord(
            decision_id=str(uuid.uuid4()),
            trace_id=request.trace_id,
            timestamp=datetime.utcnow(),
            trigger_type=request.trigger_type,
            violated_rules=request.violated_rules,
            original_request=request.original_request,
            user_choice=response.decision,
            user_reasoning=response.user_rationale,
            final_action=final_action,
            should_generalize=self._should_generalize(request, response),
            tags=self._extract_tags(request),
        )

        # Remove from pending
        del self._pending_requests[response.request_id]

        # Store in resolved records
        self._resolved_records.append(record)

        # Record to LoreBook if applicable
        if self._should_record_to_lorebook(request, response):
            self._lorebook_recorder(record)

        logger.info(
            "AdminApproval resolved",
            extra={
                "request_id": response.request_id,
                "decision": response.decision.value,
                "final_action": final_action,
                "recorded_to_lorebook": self._should_record_to_lorebook(request, response),
            }
        )

        return record

    def _determine_final_action(
        self,
        request: AdminApprovalRequest,
        response: AdminApprovalResponse
    ) -> str:
        """Determine the final action based on user decision.

        Args:
            request: The original request
            response: The user's response

        Returns:
            Description of the final action taken
        """
        match response.decision:
            case ApprovalDecision.APPROVE:
                return f"Executed original request (user override): {request.original_request}"
            case ApprovalDecision.MODIFY:
                return f"Executed modified request: {response.modification_text or 'No modification provided'}"
            case ApprovalDecision.REJECT:
                return "Request rejected, no action taken"

    def _should_generalize(
        self,
        request: AdminApprovalRequest,
        response: AdminApprovalResponse
    ) -> bool:
        """Determine if this decision should be generalized as a pattern.

        Per Constitution_v1.md Section 5.1, we should generalize when:
        - User provides explicit justification
        - Same debate triggered 3+ times (checked elsewhere)

        Args:
            request: The original request
            response: The user's response

        Returns:
            True if should be treated as a reusable pattern
        """
        # If user provided rationale, they want it remembered
        if response.user_rationale and len(response.user_rationale) > 10:
            return True

        # Constitution overrides should always be remembered
        if request.trigger_type == TriggerType.CONSTITUTION and response.decision == ApprovalDecision.APPROVE:
            return True

        return False

    def _should_record_to_lorebook(
        self,
        request: AdminApprovalRequest,
        response: AdminApprovalResponse
    ) -> bool:
        """Determine if this decision should be recorded to LoreBook.

        Per Constitution_v1.md Section 5:

        MUST record:
        - Constitution overrides
        - Codex overrides with reasoning

        MUST NOT record:
        - User rejects without explanation
        - Debate aborted (not applicable here)

        Args:
            request: The original request
            response: The user's response

        Returns:
            True if should record to LoreBook
        """
        # MUST record: Constitution override
        if request.trigger_type == TriggerType.CONSTITUTION and response.decision == ApprovalDecision.APPROVE:
            return True

        # MUST record: Codex override with reasoning
        if request.trigger_type == TriggerType.CODEX and response.decision == ApprovalDecision.APPROVE and response.user_rationale:
            return True

        # MUST NOT record: Reject without explanation
        if response.decision == ApprovalDecision.REJECT and not response.user_rationale:
            return False

        # Default: Record if severity is CRITICAL or WARNING
        if request.trigger_severity in (TriggerSeverity.CRITICAL, TriggerSeverity.WARNING):
            return True

        return False

    def _extract_tags(self, request: AdminApprovalRequest) -> list[str]:
        """Extract categorization tags from a request.

        Tags are used for pattern matching in LoreBook.

        Args:
            request: The approval request

        Returns:
            List of tags for categorization
        """
        tags = [request.trigger_type.value]

        # Add severity as tag
        tags.append(request.trigger_severity.value.lower())

        # Extract keywords from violated rules
        for rule in request.violated_rules:
            # Extract rule number if present
            if "Rule " in rule:
                tags.append(rule.split(":")[0].strip().lower().replace(" ", "_"))

        return tags

    def get_resolved_records(self) -> list[DebateRecord]:
        """Get all resolved debate records.

        Returns:
            List of DebateRecord objects from resolved debates
        """
        return list(self._resolved_records)

    def clear_pending(self) -> int:
        """Clear all pending requests (for testing/reset).

        Returns:
            Number of requests cleared
        """
        count = len(self._pending_requests)
        self._pending_requests.clear()
        logger.warning(f"Cleared {count} pending approval requests")
        return count



# Singleton instance
_debate_manager = None


def get_debate_manager() -> DebateManager:
    """Get the singleton DebateManager instance.

    Returns:
        The global DebateManager instance
    """
    global _debate_manager
    if _debate_manager is None:
        _debate_manager = DebateManager()
    return _debate_manager


# Export main class and accessor
__all__ = ["DebateManager", "get_debate_manager"]
