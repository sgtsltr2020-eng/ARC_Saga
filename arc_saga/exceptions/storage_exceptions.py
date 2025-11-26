"""
arc_saga/exceptions/storage_exceptions.py
Custom exceptions with actionable error messages.
"""


class ArcSagaException(Exception):
    """Base exception for all ARC Saga errors."""
    
    def __init__(self, message: str, error_code: str = "UNKNOWN") -> None:
        self.message = message
        self.error_code = error_code
        super().__init__(f"[{error_code}] {message}")


class StorageError(ArcSagaException):
    """Raised when storage operations fail."""
    
    def __init__(self, message: str, operation: str = "") -> None:
        super().__init__(
            f"Storage operation '{operation}' failed: {message}",
            error_code="STORAGE_ERROR"
        )
        self.operation = operation


class ValidationError(ArcSagaException):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, field_name: str = "") -> None:
        super().__init__(
            f"Validation failed on field '{field_name}': {message}",
            error_code="VALIDATION_ERROR"
        )
        self.field_name = field_name


class FileProcessingError(ArcSagaException):
    """Raised when file processing fails."""
    
    def __init__(self, filename: str, reason: str) -> None:
        super().__init__(
            f"Failed to process file '{filename}': {reason}",
            error_code="FILE_PROCESSING_ERROR"
        )
        self.filename = filename


class MonitoringError(ArcSagaException):
    """Raised when monitoring systems fail."""
    
    def __init__(self, provider: str, message: str) -> None:
        super().__init__(
            f"Monitoring error for provider '{provider}': {message}",
            error_code="MONITORING_ERROR"
        )
        self.provider = provider
