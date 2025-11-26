"""
arc_saga/arc_saga/models/message.py
Core data models with Pydantic validation.

Follows: Single Responsibility Principle (data only, no logic)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any
from uuid import uuid4
from pydantic import BaseModel, Field, field_validator


# ENUMS

class Provider(str, Enum):
    """Supported AI providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    ANTIGRAVITY = "antigravity"
    PERPLEXITY = "perplexity"
    GROQ = "groq"


class MessageRole(str, Enum):
    """Message sender role."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class FileType(str, Enum):
    """Supported file types."""
    PDF = "pdf"
    DOCX = "docx"
    CODE = "code"
    IMAGE = "image"
    MARKDOWN = "markdown"
    TEXT = "text"


# DATACLASSES

@dataclass
class Message:
    """Represents a single message in a conversation."""
    provider: Provider
    role: MessageRole
    content: str
    tags: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate message after initialization."""
        if not self.content.strip():
            raise ValueError("Message content cannot be empty")
        if len(self.content) > 100_000:
            raise ValueError("Message content exceeds 100KB limit")
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "provider": self.provider.value,
            "role": self.role.value,
            "content": self.content,
            "tags": self.tags,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "session_id": self.session_id,
        }


@dataclass
class File:
    """Represents an uploaded or monitored file."""
    filename: str
    filepath: str
    file_type: FileType
    extracted_text: str = ""
    tags: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid4()))
    file_size: int = 0
    uploaded_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate file after initialization."""
        if not self.filename.strip():
            raise ValueError("Filename cannot be empty")
        if self.file_size > 100_000_000:
            raise ValueError("File exceeds 100MB limit")


@dataclass
class SearchResult:
    """Result from a search query."""
    entity_id: str
    entity_type: str
    content: str
    tags: list[str]
    relevance_score: float = 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    
    def raise_if_invalid(self) -> None:
        """Raise exception if validation failed."""
        if not self.is_valid:
            error_msg = "; ".join(self.errors)
            raise ValueError(f"Validation failed: {error_msg}")


# PYDANTIC MODELS

class MessageCreateRequest(BaseModel):
    """API request to create a message."""
    provider: str = Field(..., description="AI provider")
    role: str = Field(..., description="Message role")
    content: str = Field(..., min_length=1, max_length=100_000)
    session_id: Optional[str] = None
    
    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate provider is in allowed list."""
        if v.lower() not in [p.value for p in Provider]:
            raise ValueError(f"Unknown provider: {v}")
        return v.lower()
    
    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        """Validate role is in allowed list."""
        if v.lower() not in [r.value for r in MessageRole]:
            raise ValueError(f"Unknown role: {v}")
        return v.lower()


class SearchRequestModel(BaseModel):
    """API request to search messages."""
    query: str = Field(..., min_length=1, max_length=1000)
    tags: Optional[list[str]] = Field(None, description="Optional tag filters")
    limit: int = Field(50, ge=1, le=500)


class MessageResponseModel(BaseModel):
    """API response for a message."""
    id: str
    provider: str
    role: str
    content: str
    tags: list[str]
    timestamp: str
    session_id: Optional[str] = None
