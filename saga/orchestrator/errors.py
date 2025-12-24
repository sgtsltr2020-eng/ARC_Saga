"""Provider routing and engine execution errors.

Base exception hierarchy for ProviderRouter fallback logic.

Transient errors trigger retries with exponential backoff.
Permanent errors trigger immediate fallback to next provider.
"""

from __future__ import annotations


class ProviderError(Exception):
    """Base exception for provider routing failures.

    Raised when routing cannot complete successfully or when all fallback
    providers have been exhausted.
    """

    pass


class TransientError(ProviderError):
    """Errors that may resolve upon retry.

    Typical causes:
    - RateLimitError (rate limit hit, will recover after delay)
    - TimeoutError (request timeout, may succeed on retry)
    - TransientError from engine (temporary service issue)
    - Network hiccups (connection reset, DNS lookup failure)

    Router treatment: Retry with exponential backoff up to max_retries,
    then move to next provider.
    """

    pass


class PermanentError(ProviderError):
    """Errors that should not be retried.

    Typical causes:
    - AuthenticationError (invalid credentials, won't fix with retry)
    - InputValidationError (bad input, won't improve with retry)
    - Unsupported operation (engine can't handle task type)
    - Hard provider outage (service unavailable for hours)

    Router treatment: Move immediately to next provider without retrying.
    """

    pass


class BudgetExceededRoutingError(PermanentError):
    """Raised when routing cost estimation violates hard budget limits."""

    pass


class OrchestratorError(Exception):
    """Base exception for orchestrator errors."""

    def __init__(self, message: str, operation: str = "") -> None:
        """
        Initialize OrchestratorError.

        Args:
            message: Error description
            operation: Operation that failed
        """
        self.operation = operation
        super().__init__(f"Orchestrator {operation} failed: {message}")


class WorkflowError(OrchestratorError):
    """Exception raised when workflow execution fails."""

    def __init__(
        self,
        message: str,
        workflow_id: str,
        failed_tasks: list[str] | None = None,
    ) -> None:
        """
        Initialize WorkflowError.

        Args:
            message: Error description
            workflow_id: ID of the failed workflow
            failed_tasks: List of failed task IDs
        """
        self.workflow_id = workflow_id
        self.failed_tasks = failed_tasks or []
        super().__init__(message, "execute_workflow")


class BudgetExceededError(OrchestratorError):
    """Exception raised when workflow exceeds token budget."""

    def __init__(
        self,
        workflow_id: str,
        requested_tokens: int,
        remaining_tokens: int,
        total_budget: int,
    ) -> None:
        """
        Initialize BudgetExceededError.

        Args:
            workflow_id: ID of the workflow that exceeded budget
            requested_tokens: Tokens requested for the workflow
            remaining_tokens: Tokens remaining in budget
            total_budget: Total token budget
        """
        self.workflow_id = workflow_id
        self.requested_tokens = requested_tokens
        self.remaining_tokens = remaining_tokens
        self.total_budget = total_budget
        message = (
            f"Workflow {workflow_id} requires {requested_tokens} tokens, "
            "but only {remaining_tokens}/{total_budget} tokens remaining"
        )
        super().__init__(message, "execute_workflow")


class PolicyViolationError(OrchestratorError):
    """Exception raised when policy enforcement fails."""

    def __init__(self, policy_name: str, reason: str) -> None:
        """
        Initialize PolicyViolationError.

        Args:
            policy_name: Name of the violated policy
            reason: Reason for violation
        """
        self.policy_name = policy_name
        self.reason = reason
        super().__init__(
            f"Policy '{policy_name}' violated: {reason}",
            "enforce_policy",
        )
