"""
AdminApproval FastAPI Router
============================

REST API endpoints for the AdminApproval debate protocol.
Implements the API layer defined in docs/Constitution_v1.md.

Endpoints:
- GET /approval/pending: List all pending approval requests
- GET /approval/{request_id}: Get details of a specific request
- POST /approval/{request_id}/respond: Submit a user decision

Author: ARC SAGA Development Team
Date: December 2025
Status: MVP Implementation
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from saga.core.admin_approval import (
    AdminApprovalRequest,
    AdminApprovalResponse,
    ApprovalDecision,
)
from saga.core.debate_manager import DebateManager

logger = logging.getLogger(__name__)

# Create router with prefix and tags
router = APIRouter(
    prefix="/approval",
    tags=["AdminApproval"],
    responses={404: {"description": "Request not found"}},
)

# Singleton DebateManager instance
# In production, this would be injected via dependency injection
_debate_manager: Optional[DebateManager] = None


def get_debate_manager() -> DebateManager:
    """Get or create the singleton DebateManager instance.

    In production, this would be replaced with proper DI.
    For MVP, we use a module-level singleton.
    """
    global _debate_manager
    if _debate_manager is None:
        _debate_manager = DebateManager()
    return _debate_manager


def set_debate_manager(manager: DebateManager) -> None:
    """Set the DebateManager instance (for testing).

    Args:
        manager: The DebateManager to use
    """
    global _debate_manager
    _debate_manager = manager


# Response models for API documentation
class PendingRequestsResponse(BaseModel):
    """Response model for listing pending requests."""
    count: int = Field(description="Number of pending requests")
    requests: list[AdminApprovalRequest] = Field(description="List of pending requests")


class RespondRequest(BaseModel):
    """Request body for submitting a decision."""
    decision: ApprovalDecision = Field(description="User's decision: APPROVE, MODIFY, or REJECT")
    modification_text: Optional[str] = Field(
        default=None,
        description="Modified request text (required if decision is MODIFY)"
    )
    user_rationale: Optional[str] = Field(
        default=None,
        description="User's explanation for the decision"
    )


class RespondResponse(BaseModel):
    """Response model for a submitted decision."""
    success: bool = Field(description="Whether the response was processed successfully")
    decision_id: str = Field(description="ID of the created DebateRecord")
    final_action: str = Field(description="Description of what action was taken")
    recorded_to_lorebook: bool = Field(description="Whether the decision was recorded to LoreBook")


@router.get(
    "/pending",
    response_model=PendingRequestsResponse,
    summary="List pending approval requests",
    description="Get all pending AdminApproval requests awaiting user decision.",
)
async def get_pending_requests() -> PendingRequestsResponse:
    """List all pending approval requests.

    Returns all AdminApprovalRequest objects that are awaiting
    user decision. These are conflicts that SAGA cannot resolve
    autonomously per Constitution_v1.md.

    Returns:
        PendingRequestsResponse with count and list of requests
    """
    manager = get_debate_manager()
    pending = manager.get_pending_requests()

    logger.info(f"Returning {len(pending)} pending approval requests")

    return PendingRequestsResponse(
        count=len(pending),
        requests=pending
    )


@router.get(
    "/{request_id}",
    response_model=AdminApprovalRequest,
    summary="Get approval request details",
    description="Get the details of a specific approval request by ID.",
)
async def get_request(request_id: str) -> AdminApprovalRequest:
    """Get details of a specific approval request.

    Args:
        request_id: The unique ID of the approval request

    Returns:
        The AdminApprovalRequest object

    Raises:
        HTTPException 404: If request_id not found
    """
    manager = get_debate_manager()
    request = manager.get_request(request_id)

    if request is None:
        logger.warning(f"Approval request not found: {request_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Approval request not found: {request_id}"
        )

    return request


@router.post(
    "/{request_id}/respond",
    response_model=RespondResponse,
    summary="Submit decision for approval request",
    description="Submit a user decision (APPROVE/MODIFY/REJECT) for an approval request.",
)
async def respond_to_request(
    request_id: str,
    body: RespondRequest
) -> RespondResponse:
    """Submit a user decision for an approval request.

    This resolves a pending AdminApprovalRequest with the user's
    decision. The decision is processed according to Constitution_v1.md
    Section 4.2:

    - APPROVE: Execute original request as user override
    - MODIFY: Execute user's modified version
    - REJECT: Abort request, no changes made

    Args:
        request_id: The unique ID of the approval request
        body: The user's decision and optional rationale

    Returns:
        RespondResponse with result details

    Raises:
        HTTPException 404: If request_id not found
        HTTPException 400: If MODIFY without modification_text
    """
    manager = get_debate_manager()

    # Validate MODIFY has modification text
    if body.decision == ApprovalDecision.MODIFY and not body.modification_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="modification_text is required when decision is MODIFY"
        )

    # Create response object
    response = AdminApprovalResponse(
        request_id=request_id,
        decision=body.decision,
        modification_text=body.modification_text,
        user_rationale=body.user_rationale,
    )

    try:
        record = manager.handle_response(response)
    except ValueError as e:
        logger.warning(f"Approval request not found: {request_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )

    # Determine if it was recorded to LoreBook
    # (We check the manager's internal logic by seeing if should_generalize is True)
    recorded = record.should_generalize or record.trigger_type.value == "constitution"

    logger.info(
        f"Approval request {request_id} resolved",
        extra={
            "decision": body.decision.value,
            "recorded_to_lorebook": recorded,
        }
    )

    return RespondResponse(
        success=True,
        decision_id=record.decision_id,
        final_action=record.final_action,
        recorded_to_lorebook=recorded,
    )


# Health check endpoint for the approval subsystem
@router.get(
    "/health",
    summary="Check approval system health",
    description="Check if the approval system is operational.",
)
async def health_check() -> dict:
    """Check if the approval system is operational.

    Returns:
        Health status including pending request count
    """
    manager = get_debate_manager()
    pending_count = len(manager.get_pending_requests())

    return {
        "status": "healthy",
        "pending_requests": pending_count,
        "subsystem": "AdminApproval",
    }
