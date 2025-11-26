"""
arc_saga/arc_saga/__init__.py
ARC Saga Memory Layer package initialization.

Provides enterprise-grade persistent memory for AI conversations
with full-text search, auto-tagging, and multi-provider support.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from arc_saga.arc_saga.storage import StorageBackend
from arc_saga.arc_saga.models import Message, File, Provider, MessageRole

__all__ = [
    "StorageBackend",
    "Message",
    "File",
    "Provider",
    "MessageRole",
]
