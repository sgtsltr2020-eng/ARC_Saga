"""
Encrypted Token Store Implementation.

SQLite-backed encrypted token storage using Fernet (AES-256) encryption.
Provides persistent, secure token storage for OAuth2 tokens.

This module provides:
- SQLiteEncryptedTokenStore: IEncryptedStore implementation
- Key management: environment variable or file-based
- Automatic encryption/decryption with Fernet
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import aiosqlite
from cryptography.fernet import Fernet

from ..error_instrumentation import log_with_context
from ..exceptions.integration_exceptions import TokenStorageError


class SQLiteEncryptedTokenStore:
    """
    SQLite-backed encrypted token storage.

    Stores OAuth2 tokens encrypted with Fernet (AES-256) in SQLite database.
    Encryption key derived from environment variable or auto-generated and
    stored in user's home directory with 0600 permissions.

    Attributes:
        db_path: Path to SQLite database file
        encryption_key: Fernet encryption key (32 bytes, base64-encoded)

    Example:
        >>> store = SQLiteEncryptedTokenStore("~/.arc_saga/tokens.db")
        >>> await store.get_token("user123")
        >>> await store.save_token("user123", {"access_token": "...", ...})
    """

    def __init__(self, db_path: str, encryption_key: Optional[str] = None) -> None:
        """
        Initialize encrypted token store.

        Args:
            db_path: Path to SQLite database file
            encryption_key: Optional encryption key (from env or file if None)

        Raises:
            TokenStorageError: If encryption key is invalid
        """
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Get or create encryption key
        key_str = encryption_key or self._get_or_create_encryption_key()

        # Initialize Fernet cipher
        try:
            # _get_or_create_encryption_key() always returns str
            # encryption_key parameter is also str (from Optional[str])
            key_bytes = key_str.encode("utf-8")
            self._fernet = Fernet(key_bytes)
        except Exception as e:
            raise TokenStorageError(f"Invalid encryption key: {e}") from e

        log_with_context(
            "info",
            "encrypted_token_store_initialized",
            db_path=str(self.db_path),
            key_source="env" if encryption_key else "file",
        )

    async def initialize(self) -> None:
        """
        Initialize database schema.

        Creates encrypted_tokens table if it doesn't exist.
        Must be called before get_token() or save_token().

        Raises:
            TokenStorageError: If database initialization fails
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS encrypted_tokens (
                        user_id TEXT PRIMARY KEY,
                        encrypted_data BLOB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                await db.commit()

            log_with_context(
                "info",
                "token_store_initialized",
                db_path=str(self.db_path),
            )
        except Exception as e:
            raise TokenStorageError(f"Database initialization failed: {e}") from e

    async def get_token(self, user_id: str) -> Optional[dict[str, str | int]]:
        """
        Get stored token for a user.

        Args:
            user_id: Unique user identifier

        Returns:
            Token dictionary with access_token, refresh_token, expires_in, etc.
            Returns None if token not found.

        Raises:
            TokenStorageError: If decryption or database operation fails
        """
        if not user_id or not user_id.strip():
            raise ValueError("user_id cannot be empty")

        try:
            # Ensure database is initialized
            await self.initialize()

            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "SELECT encrypted_data FROM encrypted_tokens WHERE user_id = ?",
                    (user_id,),
                )
                row = await cursor.fetchone()

                if row is None:
                    log_with_context(
                        "info",
                        "token_store_get",
                        user_id=user_id,
                        found=False,
                    )
                    return None

                encrypted_data = row[0]

                # Decrypt token data
                try:
                    decrypted_bytes = self._fernet.decrypt(encrypted_data)
                    token_dict: dict[str, str | int] = json.loads(
                        decrypted_bytes.decode("utf-8")
                    )
                except Exception as e:
                    raise TokenStorageError(
                        f"Failed to decrypt token data for user {user_id}: {e}"
                    ) from e

                log_with_context(
                    "info",
                    "token_store_get",
                    user_id=user_id,
                    found=True,
                )

                return token_dict

        except TokenStorageError:
            raise
        except Exception as e:
            raise TokenStorageError(f"Database operation failed: {e}") from e

    async def save_token(self, user_id: str, token_dict: dict[str, str | int]) -> None:
        """
        Save encrypted token for a user.

        Args:
            user_id: Unique user identifier
            token_dict: Token dictionary to encrypt and store

        Raises:
            TokenStorageError: If encryption or database operation fails
        """
        if not user_id or not user_id.strip():
            raise ValueError("user_id cannot be empty")

        if not token_dict:
            raise ValueError("token_dict cannot be empty")

        try:
            # Ensure database is initialized
            await self.initialize()

            # Serialize and encrypt token data
            try:
                json_data = json.dumps(token_dict).encode("utf-8")
                encrypted_data = self._fernet.encrypt(json_data)
            except Exception as e:
                raise TokenStorageError(f"Failed to encrypt token data: {e}") from e

            # Upsert token in database
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT INTO encrypted_tokens (user_id, encrypted_data, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(user_id) DO UPDATE SET
                        encrypted_data = excluded.encrypted_data,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (user_id, encrypted_data),
                )
                await db.commit()

            log_with_context(
                "info",
                "token_store_save",
                user_id=user_id,
            )

        except TokenStorageError:
            raise
        except Exception as e:
            raise TokenStorageError(f"Database operation failed: {e}") from e

    def _get_or_create_encryption_key(self) -> str:
        """
        Get or create encryption key.

        Priority:
        1. Environment variable ARC_SAGA_TOKEN_ENCRYPTION_KEY
        2. File ~/.arc_saga/.token_key (load if exists)
        3. Generate new key and save to file

        Returns:
            Base64-encoded encryption key (32 bytes)

        Raises:
            TokenStorageError: If key file cannot be created or has invalid permissions
        """
        # Check environment variable first
        env_key = os.getenv("ARC_SAGA_TOKEN_ENCRYPTION_KEY")
        if env_key:
            try:
                # Validate key format
                Fernet(env_key.encode())
                return env_key
            except Exception as e:
                log_with_context(
                    "warning",
                    "token_store_invalid_env_key",
                    error=str(e),
                )
                # Fall through to file-based key

        # Check for existing key file
        key_file = Path.home() / ".arc_saga" / ".token_key"
        key_file.parent.mkdir(parents=True, exist_ok=True)

        if key_file.exists():
            try:
                key = key_file.read_text(encoding="utf-8").strip()
                # Validate key format
                Fernet(key.encode())
                # Ensure file permissions are 0600
                key_file.chmod(0o600)
                return key
            except Exception as e:
                log_with_context(
                    "warning",
                    "token_store_invalid_file_key",
                    key_file=str(key_file),
                    error=str(e),
                )
                # Fall through to generate new key

        # Generate new key
        try:
            key = Fernet.generate_key().decode("utf-8")
            key_file.write_text(key, encoding="utf-8")
            key_file.chmod(0o600)  # User-readable only

            log_with_context(
                "info",
                "token_store_key_generated",
                key_file=str(key_file),
            )

            return key

        except Exception as e:
            raise TokenStorageError(
                f"Failed to generate or save encryption key: {e}"
            ) from e
