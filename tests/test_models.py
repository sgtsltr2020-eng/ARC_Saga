"""Test Message and File models"""

from datetime import datetime
from saga.models import Message, MessageRole, Provider, File, FileType


class TestMessage:
    """Test Message model"""

    def test_message_creation(self):
        """Test creating a message"""
        msg = Message(
            provider=Provider.OPENAI, role=MessageRole.USER, content="Test message"
        )

        assert msg.provider == Provider.OPENAI
        assert msg.role == MessageRole.USER
        assert msg.content == "Test message"
        assert msg.id is not None
        assert isinstance(msg.timestamp, datetime)

    def test_message_to_dict(self):
        """Test message serialization"""
        msg = Message(provider=Provider.OPENAI, role=MessageRole.USER, content="Test")

        data = msg.to_dict()

        assert data["provider"] == "openai"
        assert data["role"] == "user"
        assert data["content"] == "Test"

    def test_message_with_session_id(self):
        """Test message with session ID"""
        msg = Message(
            provider=Provider.OPENAI,
            role=MessageRole.USER,
            content="Test",
            session_id="test-session-123",
        )

        assert msg.session_id == "test-session-123"


class TestFile:
    """Test File model"""

    def test_file_creation(self):
        """Test creating a file"""

        file = File(
            filename="test.pdf",
            filepath="files/test.pdf",  # Changed from file_path
            file_type=FileType.DOCUMENT,  # Use FileType enum
            file_size=1024,
        )

        assert file.filename == "test.pdf"
        assert file.file_size == 1024
        assert file.id is not None
