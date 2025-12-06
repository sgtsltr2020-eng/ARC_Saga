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
