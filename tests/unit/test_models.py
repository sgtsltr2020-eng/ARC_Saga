"""
arc_saga/tests/unit/test_models.py
Unit tests for data models.
"""

import pytest
from datetime import datetime
from arc_saga.arc_saga.models import (
    Message, File, Provider, MessageRole, FileType,
    SearchResult, ValidationResult
)


class TestMessage:
    """Test Message model."""
    
    def test_message_creation_valid(self) -> None:
        """Valid message creation."""
        msg = Message(
            provider=Provider.PERPLEXITY,
            role=MessageRole.USER,
            content="Test message",
            tags=["test"]
        )
        assert msg.id is not None
        assert msg.provider == Provider.PERPLEXITY
        assert msg.role == MessageRole.USER
        assert msg.content == "Test message"
        assert msg.tags == ["test"]
    
    def test_message_empty_content_raises(self) -> None:
        """Empty content should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Message(
                provider=Provider.PERPLEXITY,
                role=MessageRole.USER,
                content=""
            )
    
    def test_message_oversized_content_raises(self) -> None:
        """Content over 100KB should raise ValueError."""
        with pytest.raises(ValueError, match="exceeds 100KB"):
            Message(
                provider=Provider.PERPLEXITY,
                role=MessageRole.USER,
                content="x" * 101_000
            )
    
    def test_message_default_timestamp(self) -> None:
        """Default timestamp should be current time."""
        before = datetime.utcnow()
        msg = Message(
            provider=Provider.ANTIGRAVITY,
            role=MessageRole.ASSISTANT,
            content="Response"
        )
        after = datetime.utcnow()
        
        assert isinstance(msg.timestamp, datetime)
        assert before <= msg.timestamp <= after
    
    def test_message_to_dict(self) -> None:
        """Test Message.to_dict() conversion."""
        msg = Message(
            provider=Provider.OPENAI,
            role=MessageRole.USER,
            content="Hello",
            tags=["greeting"],
            session_id="session123"
        )
        
        msg_dict = msg.to_dict()
        assert msg_dict["id"] == msg.id
        assert msg_dict["provider"] == "openai"
        assert msg_dict["role"] == "user"
        assert msg_dict["content"] == "Hello"
        assert msg_dict["tags"] == ["greeting"]
        assert msg_dict["session_id"] == "session123"


class TestFile:
    """Test File model."""
    
    def test_file_creation_valid(self) -> None:
        """Valid file creation."""
        file = File(
            filename="test.py",
            filepath="files/test.py",
            file_type=FileType.CODE,
            extracted_text="print('hello')",
            tags=["python"]
        )
        assert file.id is not None
        assert file.filename == "test.py"
        assert file.file_type == FileType.CODE
    
    def test_file_empty_filename_raises(self) -> None:
        """Empty filename should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            File(
                filename="",
                filepath="files/",
                file_type=FileType.TEXT
            )
    
    def test_file_oversized_raises(self) -> None:
        """Files over 100MB should raise ValueError."""
        with pytest.raises(ValueError, match="exceeds 100MB"):
            File(
                filename="huge.bin",
                filepath="files/huge.bin",
                file_type=FileType.PDF,
                file_size=101_000_001
            )


class TestValidationResult:
    """Test ValidationResult model."""
    
    def test_valid_result_no_raise(self) -> None:
        """Valid result should not raise."""
        result = ValidationResult(is_valid=True)
        result.raise_if_invalid()  # Should not raise
    
    def test_invalid_result_raises(self) -> None:
        """Invalid result should raise ValueError."""
        result = ValidationResult(
            is_valid=False,
            errors=["Error 1", "Error 2"]
        )
        with pytest.raises(ValueError, match="Error 1"):
            result.raise_if_invalid()


class TestSharedConfig:
    """Test shared configuration."""
    
    def test_config_initialize_dirs(self) -> None:
        """Config.initialize_dirs() should create directories."""
        from shared.config import SharedConfig
        
        SharedConfig.initialize_dirs()
        
        assert SharedConfig.STORAGE_DIR.exists()
        assert SharedConfig.FILES_DIR.exists()
        assert SharedConfig.LOGS_DIR.exists()
    
    def test_config_validate(self) -> None:
        """Config.validate_config() should pass."""
        from shared.config import SharedConfig
        
        errors = SharedConfig.validate_config()
        assert isinstance(errors, list)
