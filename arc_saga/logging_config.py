"""
arc_saga/arc_saga/logging_config.py
Enterprise-grade structured logging setup.

All logs are JSON formatted for aggregation into monitoring systems.
Follows: Single Responsibility Principle
"""

import json
import logging
from logging import LogRecord
from typing import Any
from pathlib import Path

from shared.config import SharedConfig


class StructuredFormatter(logging.Formatter):
    """Format logs as JSON for aggregation systems (ELK, Splunk, etc.)."""
    
    def format(self, record: LogRecord) -> str:
        """Convert log record to JSON."""
        log_data: dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add custom context fields if present
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        if hasattr(record, "session_id"):
            log_data["session_id"] = record.session_id
        
        return json.dumps(log_data)


def setup_logging(level: str = "INFO") -> None:
    """
    Initialize structured logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Console handler (JSON formatted, INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(console_handler)
    
    # File handler (all levels)
    SharedConfig.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(SharedConfig.LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(file_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    logging.getLogger("watchdog").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with consistent naming.
    
    Args:
        name: Module name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
