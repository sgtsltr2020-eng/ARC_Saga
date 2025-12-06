"""
Unit tests for exception classes.

Tests all exception constructors and message formatting.
"""

from __future__ import annotations


from arc_saga.exceptions.storage_exceptions import (
    ArcSagaException,
    StorageError,
    ValidationError,
    FileProcessingError,
    MonitoringError,
)


class TestArcSagaException:
    """Tests for ArcSagaException base class."""

    def test_init_with_default_error_code(self) -> None:
        """Test ArcSagaException with default error code."""
        exc = ArcSagaException("Test message")
        assert exc.message == "Test message"
        assert exc.error_code == "UNKNOWN"
        assert str(exc) == "[UNKNOWN] Test message"

    def test_init_with_custom_error_code(self) -> None:
        """Test ArcSagaException with custom error code."""
        exc = ArcSagaException("Test message", error_code="CUSTOM_ERROR")
        assert exc.message == "Test message"
        assert exc.error_code == "CUSTOM_ERROR"
        assert str(exc) == "[CUSTOM_ERROR] Test message"


class TestStorageError:
    """Tests for StorageError exception."""

    def test_init_with_default_operation(self) -> None:
        """Test StorageError with default operation."""
        exc = StorageError("Database connection failed")
        # Message includes formatted string
        assert "Database connection failed" in exc.message
        assert exc.operation == ""
        assert exc.error_code == "STORAGE_ERROR"
        assert "Storage operation" in str(exc)

    def test_init_with_operation(self) -> None:
        """Test StorageError with operation specified."""
        exc = StorageError("Database connection failed", operation="initialize")
        # Message includes formatted string
        assert "Database connection failed" in exc.message
        assert exc.operation == "initialize"
        assert exc.error_code == "STORAGE_ERROR"
        assert "initialize" in str(exc)
        assert "Storage operation" in str(exc)


class TestValidationError:
    """Tests for ValidationError exception."""

    def test_init_with_default_field_name(self) -> None:
        """Test ValidationError with default field name."""
        exc = ValidationError("Invalid value")
        # Message includes formatted string
        assert "Invalid value" in exc.message
        assert exc.field_name == ""
        assert exc.error_code == "VALIDATION_ERROR"
        assert "Validation failed" in str(exc)

    def test_init_with_field_name(self) -> None:
        """Test ValidationError with field name specified."""
        exc = ValidationError("Invalid value", field_name="email")
        # Message includes formatted string
        assert "Invalid value" in exc.message
        assert exc.field_name == "email"
        assert exc.error_code == "VALIDATION_ERROR"
        assert "email" in str(exc)
        assert "Validation failed" in str(exc)


class TestFileProcessingError:
    """Tests for FileProcessingError exception."""

    def test_init_with_filename_and_reason(self) -> None:
        """Test FileProcessingError with filename and reason."""
        exc = FileProcessingError("test.py", "File not found")
        assert exc.filename == "test.py"
        assert exc.error_code == "FILE_PROCESSING_ERROR"
        assert "test.py" in str(exc)
        assert "File not found" in str(exc)
        assert "Failed to process file" in str(exc)


class TestMonitoringError:
    """Tests for MonitoringError exception."""

    def test_init_with_provider_and_message(self) -> None:
        """Test MonitoringError with provider and message."""
        exc = MonitoringError("openai", "Rate limit exceeded")
        assert exc.provider == "openai"
        assert exc.error_code == "MONITORING_ERROR"
        assert "openai" in str(exc)
        assert "Rate limit exceeded" in str(exc)
        assert "Monitoring error" in str(exc)
