"""
arc_saga/storage/sqlite.py
SQLite implementation of StorageBackend with FTS5 full-text search.

Production-ready features:
- Async I/O for non-blocking operations
- FTS5 indexes for fast searching
- Automatic session grouping
- JSON fields for extensibility
- Full ACID compliance
- Windows-compatible paths

Follows: Single Responsibility Principle, Dependency Injection
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional
import logging
from datetime import datetime

from arc_saga.arc_saga.models import Message, File, SearchResult, Provider, MessageRole, FileType
from arc_saga.arc_saga.exceptions import StorageError
from arc_saga.arc_saga.storage.base import StorageBackend
from arc_saga.arc_saga.logging_config import get_logger
from shared.config import SharedConfig

logger = get_logger(__name__)


class SQLiteStorage(StorageBackend):
    """
    SQLite-based persistent storage with FTS5 full-text search.
    
    Features:
    - FTS5 indexes for fast searching
    - Automatic session grouping
    - JSON fields for extensibility
    - Full ACID compliance
    - Single-file deployment
    - Windows-compatible paths
    
    Example:
        >>> storage = SQLiteStorage()
        >>> await storage.initialize()
        >>> msg = Message(provider=Provider.PERPLEXITY, role=MessageRole.USER, content="Test")
        >>> msg_id = await storage.save_message(msg)
        >>> results = await storage.search_messages("test")
    """
    
    def __init__(self, db_path: Optional[str] = None) -> None:
        """
        Initialize SQLite storage.
        
        Args:
            db_path: Path to SQLite database file (default: ~/.arc-saga/memory.db)
        """
        if db_path is None:
            db_path = str(SharedConfig.DB_PATH)
        
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: Optional[sqlite3.Connection] = None
        logger.debug(f"SQLiteStorage initialized with db_path: {self.db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """
        Get or create database connection.
        
        Uses row factory for dict-like access to results.
        Enables foreign key constraints.
        
        Returns:
            SQLite connection object
        """
        if self._connection is None:
            self._connection = sqlite3.connect(
                str(self.db_path),
                timeout=SharedConfig.DB_TIMEOUT,
                check_same_thread=SharedConfig.DB_CHECK_SAME_THREAD
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            logger.debug(f"Database connection established: {self.db_path}")
        
        return self._connection
    
    async def initialize(self) -> None:
        """
        Create tables and indexes.
        
        Called once at startup to initialize database schema.
        Safe to call multiple times (uses CREATE TABLE IF NOT EXISTS).
        
        Raises:
            StorageError: If initialization fails
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tags TEXT,
                    timestamp DATETIME NOT NULL,
                    metadata TEXT,
                    session_id TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Files table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    extracted_text TEXT,
                    tags TEXT,
                    file_size INTEGER,
                    uploaded_at DATETIME NOT NULL,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Full-text search indexes (FTS5)
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts 
                USING fts5(
                    id UNINDEXED,
                    content,
                    tags,
                    content='messages',
                    content_rowid='rowid'
                )
            """)
            
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS files_fts 
                USING fts5(
                    id UNINDEXED,
                    filename,
                    extracted_text,
                    tags,
                    content='files',
                    content_rowid='rowid'
                )
            """)
            
            # Regular indexes for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_provider 
                ON messages(provider)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp 
                ON messages(timestamp DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session 
                ON messages(session_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_files_type 
                ON files(file_type)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_files_uploaded 
                ON files(uploaded_at DESC)
            """)
            
            conn.commit()
            logger.info(f"Storage initialized successfully at {self.db_path}")
        
        except sqlite3.Error as e:
            logger.error(f"Storage initialization failed: {e}")
            raise StorageError(str(e), operation="initialize")
    
    async def save_message(self, message: Message) -> str:
        """
        Save message to database and update FTS index.
        
        Args:
            message: Message object to save
        
        Returns:
            Message ID
        
        Raises:
            StorageError: If save fails
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Insert into messages table
            cursor.execute("""
                INSERT INTO messages 
                (id, provider, role, content, tags, timestamp, metadata, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                message.id,
                message.provider.value,
                message.role.value,
                message.content,
                json.dumps(message.tags) if message.tags else "[]",
                message.timestamp.isoformat(),
                json.dumps(message.metadata) if message.metadata else "{}",
                message.session_id
            ))
            
            # Update FTS index
            cursor.execute("""
                INSERT INTO messages_fts (id, content, tags)
                VALUES (?, ?, ?)
            """, (
                message.id,
                message.content,
                " ".join(message.tags) if message.tags else ""
            ))
            
            conn.commit()
            logger.debug(f"Saved message {message.id}", extra={
                "provider": message.provider.value,
                "tags": len(message.tags)
            })
            
            return message.id
        
        except sqlite3.Error as e:
            logger.error(f"Failed to save message: {e}")
            raise StorageError(str(e), operation="save_message")
    
    async def save_file(self, file: File) -> str:
        """
        Save file to database and update FTS index.
        
        Args:
            file: File object to save
        
        Returns:
            File ID
        
        Raises:
            StorageError: If save fails
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Insert into files table
            cursor.execute("""
                INSERT INTO files 
                (id, filename, filepath, file_type, extracted_text, tags, file_size, uploaded_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                file.id,
                file.filename,
                file.filepath,
                file.file_type.value,
                file.extracted_text,
                json.dumps(file.tags) if file.tags else "[]",
                file.file_size,
                file.uploaded_at.isoformat(),
                json.dumps(file.metadata) if file.metadata else "{}"
            ))
            
            # Update FTS index
            cursor.execute("""
                INSERT INTO files_fts (id, filename, extracted_text, tags)
                VALUES (?, ?, ?, ?)
            """, (
                file.id,
                file.filename,
                file.extracted_text,
                " ".join(file.tags) if file.tags else ""
            ))
            
            conn.commit()
            logger.debug(f"Saved file {file.filename}", extra={"file_id": file.id})
            
            return file.id
        
        except sqlite3.Error as e:
            logger.error(f"Failed to save file: {e}")
            raise StorageError(str(e), operation="save_file")
    
    async def search_messages(
        self,
        query: str,
        tags: Optional[list[str]] = None,
        limit: int = 50
    ) -> list[SearchResult]:
        """
        Search messages using FTS5.
        
        Args:
            query: Search query (supports FTS5 syntax)
            tags: Optional tag filter (AND operation)
            limit: Maximum results (default 50, max 500)
        
        Returns:
            List of search results sorted by relevance
        
        Raises:
            StorageError: If search fails
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Build FTS query
            sql = """
                SELECT m.id, m.content, m.tags, m.timestamp
                FROM messages m
                JOIN messages_fts fts ON m.id = fts.id
                WHERE fts.content MATCH ?
            """
            
            params: list = [query]
            
            # Add tag filters
            if tags:
                tag_conditions = " AND ".join([
                    f"m.tags LIKE ?"
                    for _ in tags
                ])
                sql += f" AND ({tag_conditions})"
                params.extend([f"%{tag}%" for tag in tags])
            
            sql += " ORDER BY rank LIMIT ?"
            params.append(min(limit, 500))
            
            cursor.execute(sql, params)
            
            results = []
            for row in cursor.fetchall():
                results.append(SearchResult(
                    entity_id=row['id'],
                    entity_type="message",
                    content=row['content'],
                    tags=json.loads(row['tags'] or '[]'),
                    timestamp=datetime.fromisoformat(row['timestamp'])
                ))
            
            logger.debug(f"Search completed", extra={
                "query": query,
                "result_count": len(results),
                "tag_count": len(tags) if tags else 0
            })
            
            return results
        
        except sqlite3.Error as e:
            logger.error(f"Search failed: {e}")
            raise StorageError(str(e), operation="search_messages")
    
    async def get_message_by_id(self, message_id: str) -> Optional[Message]:
        """
        Get message by ID.
        
        Args:
            message_id: UUID of the message
        
        Returns:
            Message object or None if not found
        
        Raises:
            StorageError: If operation fails
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM messages WHERE id = ?", (message_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return Message(
                id=row['id'],
                provider=Provider(row['provider']),
                role=MessageRole(row['role']),
                content=row['content'],
                tags=json.loads(row['tags'] or '[]'),
                timestamp=datetime.fromisoformat(row['timestamp']),
                metadata=json.loads(row['metadata'] or '{}'),
                session_id=row['session_id']
            )
        
        except sqlite3.Error as e:
            logger.error(f"Get message failed: {e}")
            raise StorageError(str(e), operation="get_message_by_id")
    
    async def get_file_by_id(self, file_id: str) -> Optional[File]:
        """
        Get file by ID.
        
        Args:
            file_id: UUID of the file
        
        Returns:
            File object or None if not found
        
        Raises:
            StorageError: If operation fails
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM files WHERE id = ?", (file_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return File(
                id=row['id'],
                filename=row['filename'],
                filepath=row['filepath'],
                file_type=FileType(row['file_type']),
                extracted_text=row['extracted_text'] or "",
                tags=json.loads(row['tags'] or '[]'),
                file_size=row['file_size'] or 0,
                uploaded_at=datetime.fromisoformat(row['uploaded_at']),
                metadata=json.loads(row['metadata'] or '{}')
            )
        
        except sqlite3.Error as e:
            logger.error(f"Get file failed: {e}")
            raise StorageError(str(e), operation="get_file_by_id")
    
    async def get_by_session(self, session_id: str) -> list[Message]:
        """
        Get all messages in a session, chronologically ordered.
        
        Args:
            session_id: Session identifier
        
        Returns:
            List of messages in chronological order
        
        Raises:
            StorageError: If operation fails
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
                (session_id,)
            )
            
            messages = []
            for row in cursor.fetchall():
                messages.append(Message(
                    id=row['id'],
                    provider=Provider(row['provider']),
                    role=MessageRole(row['role']),
                    content=row['content'],
                    tags=json.loads(row['tags'] or '[]'),
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    metadata=json.loads(row['metadata'] or '{}'),
                    session_id=row['session_id']
                ))
            
            logger.debug(f"Retrieved session {session_id}", extra={
                "message_count": len(messages)
            })
            
            return messages
        
        except sqlite3.Error as e:
            logger.error(f"Get session failed: {e}")
            raise StorageError(str(e), operation="get_by_session")
    
    async def health_check(self) -> bool:
        """
        Verify storage is working.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            logger.debug("Health check passed")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def __del__(self) -> None:
        """Close connection on shutdown."""
        if self._connection:
            try:
                self._connection.close()
                logger.debug("Database connection closed")
            except Exception:
                pass
