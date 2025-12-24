"""
Async Resilience Utilities
===========================

Timeouts, retries, circuit breakers for async operations.
"""

import asyncio
import logging
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


def with_timeout(seconds: float) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """
    Decorator to add timeout to async functions.
    
    Example:
        @with_timeout(30.0)
        async def call_llm():
            ...
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                logger.error(
                    f"{func.__name__} timed out after {seconds}s",
                    extra={"function": func.__name__, "timeout": seconds}
                )
                raise
        return wrapper
    return decorator


def with_retry(max_attempts: int = 3, backoff: float = 1.0) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """
    Decorator to add retry logic with exponential backoff.
    
    Example:
        @with_retry(max_attempts=3, backoff=2.0)
        async def flaky_api_call():
            ...
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts",
                            extra={"function": func.__name__, "error": str(e)}
                        )
                        raise last_exception
                    
                    wait_time = backoff * (2 ** (attempt - 1))
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt}/{max_attempts}), retrying in {wait_time}s",
                        extra={"function": func.__name__, "attempt": attempt, "wait_time": wait_time}
                    )
                    await asyncio.sleep(wait_time)
            # Should be unreachable because of re-raise in loop, but for type checkers:
            if last_exception:
                raise last_exception
            raise RuntimeError("Unreachable code in with_retry")
        return wrapper
    return decorator
