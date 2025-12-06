"""
Security-focused request validators (Pydantic-friendly).

Small helpers to keep API handlers lean while enforcing input constraints.
"""

from __future__ import annotations

from typing import Iterable

from fastapi import HTTPException, status


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=message,
        )


def validate_capture_request(
    source: str,
    role: str,
    content: str,
    metadata_keys: Iterable[str] | None = None,
) -> None:
    """Basic validation for capture payloads."""
    _require(bool(source.strip()), "source must be provided")
    _require(role.lower() in {"user", "assistant"}, "role must be user or assistant")
    _require(bool(content.strip()), "content cannot be empty")
    _require(len(content) <= 100_000, "content exceeds 100KB limit")
    if metadata_keys:
        for key in metadata_keys:
            _require(len(key) <= 128, "metadata keys too long")


def validate_search_request(query: str, limit: int) -> None:
    """Validate search inputs."""
    _require(limit > 0, "limit must be positive")
    _require(limit <= 500, "limit too large")
    _require(bool(query.strip()), "query cannot be empty")


def validate_perplexity_request(query: str) -> None:
    """Validate perplexity request payload."""
    _require(bool(query.strip()), "query cannot be empty")
