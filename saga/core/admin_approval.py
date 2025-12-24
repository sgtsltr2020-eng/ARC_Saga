"""
AdminApproval Data Models
=========================

Data models for the AdminApproval and Debate Protocol system.
Implements the structures defined in docs/Constitution_v1.md.

The AdminApproval system is the MVP conflict resolution strategy.
When SAGA detects a Constitution or Codex violation, it creates
an AdminApprovalRequest and waits for user decision.

Author: ARC SAGA Development Team
Date: December 2025
Status: MVP Implementation
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class TriggerType(str, Enum):
    """Type of event that triggered the debate.

    See Constitution_v1.md Section 2 for trigger definitions.
    """
    CONSTITUTION = "constitution"  # Immutable meta-rule violation
    CODEX = "codex"  # Code quality standard violation
    CONFIDENCE = "confidence"  # SAGA confidence below threshold
    DISAGREEMENT = "disagreement"  # LLM/agent disagreement


class TriggerSeverity(str, Enum):
    """Severity level of the trigger.

    - CRITICAL: Must resolve before proceeding
    - WARNING: Should resolve, but can override
    - INFO: Informational, low-friction resolution
    """
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


class ApprovalDecision(str, Enum):
    """User's decision on an AdminApprovalRequest.

    See Constitution_v1.md Section 4.2 for decision handling.
    """
    APPROVE = "APPROVE"  # Execute original request as user override
    MODIFY = "MODIFY"  # Execute user's modified version
    REJECT = "REJECT"  # Abort request, no changes made


class AdminApprovalRequest(BaseModel):
    """Request for user decision on a conflict.

    Created when SAGA detects a Constitution or Codex violation,
    or when confidence/agreement thresholds are not met.

    See Constitution_v1.md Section 4.1 for the full specification.

    Attributes:
        request_id: Unique identifier for this approval request
        trace_id: Distributed tracing ID for correlation
        timestamp: When the request was created
        trigger_type: What triggered the debate
        trigger_severity: CRITICAL, WARNING, or INFO
        trigger_description: Human-readable description of the trigger
        original_request: What the user originally asked
        violated_rules: List of rule numbers/names that were violated
        violation_explanation: Why this conflicts with the rules
        alternatives: Compliant alternatives SAGA can offer
        risk_statement: What happens if user approves the violation
    """
    request_id: str = Field(description="Unique identifier for this request")
    trace_id: str = Field(description="Distributed tracing ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Trigger information
    trigger_type: TriggerType
    trigger_severity: TriggerSeverity = TriggerSeverity.WARNING
    trigger_description: str

    # The conflict
    original_request: str
    violated_rules: list[str] = Field(default_factory=list)
    violation_explanation: str

    # Options for resolution
    alternatives: list[str] = Field(default_factory=list)
    risk_statement: Optional[str] = None

    # Metadata
    context: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False}


class AdminApprovalResponse(BaseModel):
    """User's response to an AdminApprovalRequest.

    See Constitution_v1.md Section 4.2 for response handling.

    Attributes:
        request_id: The request being responded to
        decision: APPROVE, MODIFY, or REJECT
        modification_text: If MODIFY, the user's modified request
        user_rationale: Optional explanation for the decision
        timestamp: When the response was submitted
    """
    request_id: str
    decision: ApprovalDecision
    modification_text: Optional[str] = None
    user_rationale: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"frozen": False}


class DebateRecord(BaseModel):
    """A complete debate record for LoreBook storage.

    Combines the request and response with learning metadata.
    See Constitution_v1.md Section 5.4 for the specification.

    Attributes:
        decision_id: Unique identifier for this decision
        trace_id: Distributed tracing ID
        timestamp: When the debate was resolved
        trigger_type: What triggered the debate
        violated_rules: Rules that were violated
        original_request: The original user request
        user_choice: The user's decision
        user_reasoning: User's explanation (if provided)
        final_action: What SAGA actually did
        should_generalize: Whether to treat as a reusable pattern
        tags: Categorization tags for pattern matching
    """
    decision_id: str
    trace_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Debate context
    trigger_type: TriggerType
    violated_rules: list[str]
    original_request: str

    # Resolution
    user_choice: ApprovalDecision
    user_reasoning: Optional[str] = None
    final_action: str

    # Learning metadata
    should_generalize: bool = False
    tags: list[str] = Field(default_factory=list)

    model_config = {"frozen": False}


# Export all models
__all__ = [
    "TriggerType",
    "TriggerSeverity",
    "ApprovalDecision",
    "AdminApprovalRequest",
    "AdminApprovalResponse",
    "DebateRecord",
]
