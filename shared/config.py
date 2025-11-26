"""
shared/config.py
Shared configuration for ARC Saga and validator integration.

Centralized settings that both modules can access.
Follows: Single Source of Truth principle
"""

import os
from pathlib import Path
from typing import Optional

class SharedConfig:
    """
    Shared configuration constants.
    
    Used by both arc_saga and validator modules.
    All paths are Windows-compatible.
    """
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    ARC_SAGA_ROOT = PROJECT_ROOT / "arc_saga"
    
    # Storage settings
    DB_PATH = Path.home() / ".arc-saga" / "memory.db"
    DB_TIMEOUT = 30  # seconds
    DB_CHECK_SAME_THREAD = False  # Allow multi-threading
    
    # Logging settings
    LOG_LEVEL = os.getenv("ARC_SAGA_LOG_LEVEL", "INFO")
    LOG_FORMAT = "json"  # or "text"
    LOG_PATH = PROJECT_ROOT / "logs"
    
    # Storage limits
    MAX_MESSAGE_LENGTH = 100_000  # characters
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
    MAX_TAGS_PER_MESSAGE = 5
    
    # Search settings
    MAX_SEARCH_RESULTS = 500
    DEFAULT_SEARCH_LIMIT = 50
    
    # Validator integration
    VALIDATOR_LOG_PATH = PROJECT_ROOT.parent / "ai-config-validator" / "logs"
    
    # Antigravity integration
    ANTIGRAVITY_LOG_PATH = Path(os.getenv(
        "ANTIGRAVITY_LOGS",
        str(Path.home() / "AppData" / "Roaming" / "Antigravity" / "logs")
    ))
    
    @classmethod
    def get_validator_log_path(cls) -> Optional[Path]:
        """Get validator log path if it exists."""
        if cls.VALIDATOR_LOG_PATH.exists():
            return cls.VALIDATOR_LOG_PATH
        return None
    
    @classmethod
    def get_antigravity_log_path(cls) -> Path:
        """Get Antigravity log path."""
        return cls.ANTIGRAVITY_LOG_PATH
    
    @classmethod
    def initialize_dirs(cls) -> None:
        """Create necessary directories."""
        cls.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        cls.LOG_PATH.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration."""
        try:
            cls.initialize_dirs()
            return True
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False
