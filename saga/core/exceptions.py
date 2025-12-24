"""
Core Exceptions for SAGA
========================

Exceptions used by core components (Warden, LoreBook, Constitution).
"""

class SagaEscalationException(Exception):
    """Raised when SAGA must escalate a decision to the user via AdminApproval.

    This exception interrupts the normal control flow to trigger the
    Debate Protocol. Caught by the orchestrator to suspend execution
    and wait for user input.

    Attributes:
        request_id: The ID of the AdminApprovalRequest created
    """
    def __init__(self, request_id: str):
        self.request_id = request_id
        super().__init__(f"SAGA must escalate to user. Request ID: {request_id}")


class SagaCoreException(Exception):
    """Base class for SAGA core exceptions."""
    pass
