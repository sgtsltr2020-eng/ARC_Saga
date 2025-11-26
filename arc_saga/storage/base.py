"""
arc_saga/arc_saga/storage/base.py
Abstract storage interface. Never changes.

This is the contract that all storage backends must implement.
Follows: Dependency Injection, Interface Segregation
"""

from abc import ABC, abstractmethod
from typing import Optional
from arc_saga.arc_saga.models import Message, File, SearchResult


class StorageBackend(ABC):
    """
    Abstract interface for data persistence.
    
    All storage implementations (SQLite, PostgreSQL, etc.) must implement this interface.
    Code that uses storage depends only on this abstract interface, not concrete implementations.
    """
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize storage (create tables, indexes, etc.).
        
        Must be called before any other operations.
        
        Raises:
            StorageError: If initialization fails
        """
        pass
    
    @abstractmethod
    async def save_message(self, message: Message) -> str:
        """
        Save a message and return its ID.
        
        Args:
            message: Message to save
        
        Returns:
            Message ID
        
        Raises:
            StorageError: If save fails
        """
        pass
    
    @abstractmethod
    async def save_file(self, file: File) -> str:
        """
        Save file metadata and return its ID.
        
        Args:
            file: File to save
        
        Returns:
            File ID
        
        Raises:
            StorageError: If save fails
        """
        pass
    
    @abstractmethod
    async def search_messages(
        self,
        query: str,
        tags: Optional[list[str]] = None,
        limit: int = 50
    ) -> list[SearchResult]:
        """
        Search messages by content and tags using full-text search.
        
        Args:
            query: Search query (supports FTS5 syntax)
            tags: Optional tag filter (AND operation)
            limit: Maximum results (default 50, max 500)
        
        Returns:
            List of search results sorted by relevance
        
        Raises:
            StorageError: If search fails
        """
        pass
    
    @abstractmethod
    async def get_message_by_id(self, message_id: str) -> Optional[Message]:
        """
        Get a specific message by ID.
        
        Args:
            message_id: UUID of the message
        
        Returns:
            Message object or None if not found
        
        Raises:
            StorageError: If operation fails
        """
        pass
    
    @abstractmethod
    async def get_file_by_id(self, file_id: str) -> Optional[File]:
        """
        Get a specific file by ID.
        
        Args:
            file_id: UUID of the file
        
        Returns:
            File object or None if not found
        
        Raises:
            StorageError: If operation fails
        """
        pass
    
    @abstractmethod
    async def get_by_session(self, session_id: str) -> list[Message]:
        """
        Get all messages in a session.
        
        Args:
            session_id: Session identifier
        
        Returns:
            List of messages in chronological order
        
        Raises:
            StorageError: If operation fails
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if storage backend is operational.
        
        Returns:
            True if healthy, False otherwise
        """
        pass
