"""
Session Manager - Central Persistence Layer for USMA
=====================================================

Provides unified auto-loading and auto-saving for all USMA components:
- SovereignOptimizer (RL policy weights)
- MythosLibrary (CAG chapters)
- RepoGraph (structure + embeddings)
- LoreBook (future integration)

Author: ARC SAGA Development Team (AntiGravity)
Date: December 26, 2025
Status: USMA P0 Fix - Persistence Wiring
"""

import atexit
import json
import logging
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

@dataclass
class PersistenceConfig:
    """Configuration for session persistence."""

    # Base directory
    base_path: Path = field(default_factory=lambda: Path.home() / ".saga" / "memory")

    # File paths (relative to base_path)
    graph_file: str = "graph.json"
    mythos_file: str = "mythos.json"
    optimizer_file: str = "optimizer.npz"
    embeddings_db: str = "embeddings.db"
    state_db: str = "state.db"
    snapshot_dir: str = ".snapshots"

    # Checkpoint settings
    checkpoint_interval_seconds: int = 900  # 15 minutes
    min_changes_for_checkpoint: int = 50  # Dirty threshold
    max_snapshots: int = 5  # Keep last N snapshots

    # Schema version for migrations
    schema_version: int = 1


# ═══════════════════════════════════════════════════════════════
# DIRTY TRACKING
# ═══════════════════════════════════════════════════════════════

@dataclass
class DirtyFlags:
    """Track which components need saving."""
    graph: bool = False
    mythos: bool = False
    optimizer: bool = False
    embeddings: bool = False

    # Change counters for threshold-based checkpoints
    graph_changes: int = 0
    mythos_changes: int = 0
    optimizer_updates: int = 0

    def mark_graph_dirty(self, changes: int = 1) -> None:
        """Mark graph as modified."""
        self.graph = True
        self.graph_changes += changes

    def mark_mythos_dirty(self, changes: int = 1) -> None:
        """Mark mythos as modified."""
        self.mythos = True
        self.mythos_changes += changes

    def mark_optimizer_dirty(self, updates: int = 1) -> None:
        """Mark optimizer as modified."""
        self.optimizer = True
        self.optimizer_updates += updates

    def should_checkpoint(self, threshold: int = 50) -> bool:
        """Check if enough changes accumulated for checkpoint."""
        total_changes = self.graph_changes + self.mythos_changes + self.optimizer_updates
        return total_changes >= threshold

    def reset(self) -> None:
        """Reset all flags and counters."""
        self.graph = False
        self.mythos = False
        self.optimizer = False
        self.embeddings = False
        self.graph_changes = 0
        self.mythos_changes = 0
        self.optimizer_updates = 0


# ═══════════════════════════════════════════════════════════════
# SESSION MANAGER (SINGLETON)
# ═══════════════════════════════════════════════════════════════

class SessionManagerMeta(type):
    """Metaclass for singleton pattern."""

    _instances: dict[type, Any] = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
            return cls._instances[cls]


class SessionManager(metaclass=SessionManagerMeta):
    """
    Central persistence manager for all USMA components.

    Features:
    - Singleton pattern ensures single point of persistence
    - Auto-load on startup, auto-save on shutdown
    - Dirty-flag tracking for efficient checkpoints
    - Atomic writes with snapshot backups
    - Schema versioning for migrations

    Usage:
        session = SessionManager()
        session.register_graph(graph)
        session.register_mythos(mythos)
        session.load_all()

        # ... work with components ...

        session.save_all()  # Called automatically on exit
    """

    def __init__(self, config: PersistenceConfig | None = None):
        """Initialize the session manager."""
        self.config = config or PersistenceConfig()
        self.dirty = DirtyFlags()

        # Registered components (set via register_*)
        self._graph: Any = None  # RepoGraph
        self._mythos: Any = None  # MythosLibrary
        self._optimizer: Any = None  # SovereignOptimizer

        # Thread safety
        self._lock = threading.Lock()
        self._initialized = False

        # Checkpoint timer
        self._checkpoint_thread: threading.Thread | None = None
        self._stop_checkpoint = threading.Event()

        # Last save time
        self._last_checkpoint: float = 0.0

        # Setup
        self._setup_directories()
        self._register_exit_handler()

        logger.info(f"SessionManager initialized at {self.config.base_path}")

    # ─── Directory Setup ───────────────────────────────────────

    def _setup_directories(self) -> None:
        """Create persistence directories if needed."""
        self.config.base_path.mkdir(parents=True, exist_ok=True)
        (self.config.base_path / self.config.snapshot_dir).mkdir(exist_ok=True)

    def _get_path(self, filename: str) -> Path:
        """Get full path for a persistence file."""
        return self.config.base_path / filename

    # ─── Component Registration ────────────────────────────────

    def register_graph(self, graph: Any) -> None:
        """Register a RepoGraph for persistence."""
        self._graph = graph
        logger.debug("Registered RepoGraph for persistence")

    def register_mythos(self, mythos: Any) -> None:
        """Register a MythosLibrary for persistence."""
        self._mythos = mythos
        logger.debug("Registered MythosLibrary for persistence")

    def register_optimizer(self, optimizer: Any) -> None:
        """Register a SovereignOptimizer for persistence."""
        self._optimizer = optimizer
        logger.debug("Registered SovereignOptimizer for persistence")

    # ─── Exit Handler ──────────────────────────────────────────

    def _register_exit_handler(self) -> None:
        """Register atexit handler for graceful shutdown."""
        atexit.register(self._on_exit)

    def _on_exit(self) -> None:
        """Called on process exit - save all components."""
        logger.info("SessionManager: Graceful shutdown initiated")
        self._stop_checkpoint.set()
        try:
            self.save_all()
        except Exception as e:
            logger.error(f"Error during shutdown save: {e}")

    # ─── Checkpoint Timer ──────────────────────────────────────

    def start_checkpoint_timer(self) -> None:
        """Start background checkpoint timer."""
        if self._checkpoint_thread is not None and self._checkpoint_thread.is_alive():
            return

        self._stop_checkpoint.clear()
        self._checkpoint_thread = threading.Thread(
            target=self._checkpoint_loop,
            daemon=True,
            name="SessionManager-Checkpoint"
        )
        self._checkpoint_thread.start()
        logger.info(f"Checkpoint timer started ({self.config.checkpoint_interval_seconds}s interval)")

    def _checkpoint_loop(self) -> None:
        """Background loop for periodic checkpoints."""
        while not self._stop_checkpoint.is_set():
            self._stop_checkpoint.wait(timeout=self.config.checkpoint_interval_seconds)

            if self._stop_checkpoint.is_set():
                break

            if self.dirty.should_checkpoint(self.config.min_changes_for_checkpoint):
                try:
                    self.checkpoint()
                except Exception as e:
                    logger.error(f"Checkpoint failed: {e}")

    # ─── Core Operations ───────────────────────────────────────

    def load_all(self) -> dict[str, bool]:
        """
        Load all registered components from disk.

        Returns:
            Dict mapping component name to load success
        """
        with self._lock:
            results = {}

            if self._graph is not None:
                results["graph"] = self._load_graph()

            if self._mythos is not None:
                results["mythos"] = self._load_mythos()

            if self._optimizer is not None:
                results["optimizer"] = self._load_optimizer()

            self.dirty.reset()
            self._initialized = True

            logger.info(f"Loaded components: {results}")
            return results

    def save_all(self) -> dict[str, bool]:
        """
        Save all registered components to disk.

        Returns:
            Dict mapping component name to save success
        """
        with self._lock:
            results = {}

            if self._graph is not None:
                results["graph"] = self._save_graph()

            if self._mythos is not None:
                results["mythos"] = self._save_mythos()

            if self._optimizer is not None:
                results["optimizer"] = self._save_optimizer()

            self.dirty.reset()
            self._last_checkpoint = time.time()

            logger.info(f"Saved components: {results}")
            return results

    def checkpoint(self) -> dict[str, bool]:
        """
        Save only dirty components.

        Creates a snapshot before saving for crash recovery.

        Returns:
            Dict mapping component name to save success
        """
        with self._lock:
            results = {}

            # Create snapshot first
            self._create_snapshot()

            if self.dirty.graph and self._graph is not None:
                results["graph"] = self._save_graph()

            if self.dirty.mythos and self._mythos is not None:
                results["mythos"] = self._save_mythos()

            if self.dirty.optimizer and self._optimizer is not None:
                results["optimizer"] = self._save_optimizer()

            self.dirty.reset()
            self._last_checkpoint = time.time()

            logger.info(f"Checkpoint completed: {results}")
            return results

    # ─── Snapshot Management ───────────────────────────────────

    def _create_snapshot(self) -> str | None:
        """
        Create a backup snapshot of current state.

        Returns:
            Snapshot directory name or None on failure
        """
        try:
            snapshot_dir = self.config.base_path / self.config.snapshot_dir
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_path = snapshot_dir / timestamp
            snapshot_path.mkdir(exist_ok=True)

            # Copy current files to snapshot
            for filename in [self.config.graph_file, self.config.mythos_file]:
                src = self._get_path(filename)
                if src.exists():
                    shutil.copy2(src, snapshot_path / filename)

            # Copy optimizer weights
            src = self._get_path(self.config.optimizer_file)
            if src.exists():
                shutil.copy2(src, snapshot_path / self.config.optimizer_file)

            # Cleanup old snapshots
            self._cleanup_snapshots()

            logger.debug(f"Created snapshot: {timestamp}")
            return timestamp
        except Exception as e:
            logger.warning(f"Failed to create snapshot: {e}")
            return None

    def _cleanup_snapshots(self) -> int:
        """Remove old snapshots beyond max limit."""
        snapshot_dir = self.config.base_path / self.config.snapshot_dir
        snapshots = sorted(snapshot_dir.iterdir(), key=lambda p: p.name, reverse=True)

        removed = 0
        for old_snapshot in snapshots[self.config.max_snapshots:]:
            try:
                shutil.rmtree(old_snapshot)
                removed += 1
            except Exception as e:
                logger.warning(f"Failed to remove snapshot {old_snapshot}: {e}")

        return removed

    def restore_from_snapshot(self, timestamp: str | None = None) -> bool:
        """
        Restore state from a snapshot.

        Args:
            timestamp: Specific snapshot to restore, or None for latest

        Returns:
            True if restored successfully
        """
        try:
            snapshot_dir = self.config.base_path / self.config.snapshot_dir

            if timestamp:
                snapshot_path = snapshot_dir / timestamp
            else:
                # Get latest snapshot
                snapshots = sorted(snapshot_dir.iterdir(), reverse=True)
                if not snapshots:
                    logger.warning("No snapshots available")
                    return False
                snapshot_path = snapshots[0]

            if not snapshot_path.exists():
                logger.error(f"Snapshot not found: {snapshot_path}")
                return False

            # Restore files
            for filename in [self.config.graph_file, self.config.mythos_file, self.config.optimizer_file]:
                src = snapshot_path / filename
                if src.exists():
                    dst = self._get_path(filename)
                    shutil.copy2(src, dst)

            logger.info(f"Restored from snapshot: {snapshot_path.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore snapshot: {e}")
            return False

    # ─── Graph Serialization ───────────────────────────────────

    def _save_graph(self) -> bool:
        """Save RepoGraph to JSON with atomic write."""
        try:
            path = self._get_path(self.config.graph_file)
            temp_path = path.with_suffix(".tmp")

            # Serialize graph
            data = {
                "schema_version": self.config.schema_version,
                "saved_at": datetime.now().isoformat(),
                "graph": self._graph.to_dict()
            }

            # Extract embeddings separately for base64 encoding
            embeddings = {}
            for node_id, node in self._graph._node_index.items():
                if hasattr(node, 'embedding_vector') and node.embedding_vector is not None:
                    if isinstance(node.embedding_vector, np.ndarray):
                        embeddings[node_id] = self._encode_array(node.embedding_vector)

            data["embeddings"] = embeddings

            # Write atomically
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            # Atomic rename
            temp_path.replace(path)

            logger.debug(f"Saved graph: {len(self._graph._node_index)} nodes")
            return True
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")
            return False

    def _load_graph(self) -> bool:
        """Load RepoGraph from JSON."""
        try:
            path = self._get_path(self.config.graph_file)

            if not path.exists():
                logger.info("No saved graph found, starting fresh")
                return True

            with open(path) as f:
                data = json.load(f)

            # Check version
            version = data.get("schema_version", 1)
            if version != self.config.schema_version:
                logger.warning(f"Graph schema version mismatch: {version} vs {self.config.schema_version}")
                # Future: run migrations here

            # Import graph from dict
            from saga.core.memory.graph_engine import RepoGraph
            loaded = RepoGraph.from_dict(data["graph"], self._graph.project_root)

            # Copy to registered graph
            self._graph._graph = loaded._graph
            self._graph._node_index = loaded._node_index

            # Restore embeddings
            embeddings = data.get("embeddings", {})
            for node_id, encoded in embeddings.items():
                node = self._graph.get_node(node_id)
                if node:
                    node.embedding_vector = self._decode_array(encoded)

            logger.debug(f"Loaded graph: {len(self._graph._node_index)} nodes")
            return True
        except Exception as e:
            logger.error(f"Failed to load graph: {e}")
            # Try restore from snapshot
            if self.restore_from_snapshot():
                return self._load_graph()
            return False

    # ─── Mythos Serialization ──────────────────────────────────

    def _save_mythos(self) -> bool:
        """Save MythosLibrary to JSON."""
        try:
            path = self._get_path(self.config.mythos_file)
            temp_path = path.with_suffix(".tmp")

            data = {
                "schema_version": self.config.schema_version,
                "saved_at": datetime.now().isoformat(),
                "mythos": self._mythos.to_dict()
            }

            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            temp_path.replace(path)

            logger.debug(f"Saved mythos: {len(self._mythos.chapters)} chapters")
            return True
        except Exception as e:
            logger.error(f"Failed to save mythos: {e}")
            return False

    def _load_mythos(self) -> bool:
        """Load MythosLibrary from JSON."""
        try:
            path = self._get_path(self.config.mythos_file)

            if not path.exists():
                logger.info("No saved mythos found, starting fresh")
                return True

            with open(path) as f:
                data = json.load(f)

            # Check version
            version = data.get("schema_version", 1)
            if version != self.config.schema_version:
                logger.warning(f"Mythos schema version mismatch: {version}")

            # Load from dict
            from saga.core.memory.mythos import MythosLibrary
            loaded = MythosLibrary.from_dict(data["mythos"])

            # Copy to registered mythos
            self._mythos.chapters = loaded.chapters
            self._mythos.last_consolidation = loaded.last_consolidation
            self._mythos.total_lore_processed = loaded.total_lore_processed

            logger.debug(f"Loaded mythos: {len(self._mythos.chapters)} chapters")
            return True
        except Exception as e:
            logger.error(f"Failed to load mythos: {e}")
            return False

    # ─── Optimizer Serialization ───────────────────────────────

    def _save_optimizer(self) -> bool:
        """Save SovereignOptimizer weights to NPZ."""
        try:
            path = self._get_path(self.config.optimizer_file)
            temp_path = path.with_suffix(".tmp.npz")

            # Get optimizer state
            state = self._optimizer._save_state() if hasattr(self._optimizer, '_save_state') else {}

            # Save weights
            save_dict = {
                "schema_version": np.array([self.config.schema_version]),
                "saved_at": np.array([datetime.now().isoformat()], dtype=object)
            }

            # Add MLP weights if available
            if hasattr(self._optimizer, 'w1'):
                save_dict["w1"] = self._optimizer.w1
            if hasattr(self._optimizer, 'w2'):
                save_dict["w2"] = self._optimizer.w2
            if hasattr(self._optimizer, 'b1'):
                save_dict["b1"] = self._optimizer.b1
            if hasattr(self._optimizer, 'b2'):
                save_dict["b2"] = self._optimizer.b2

            # Add replay buffer
            if hasattr(self._optimizer, 'replay_buffer'):
                buffer_data = [e.__dict__ for e in self._optimizer.replay_buffer]
                save_dict["replay_buffer"] = np.array(json.dumps(buffer_data), dtype=object)

            np.savez(temp_path, **save_dict)

            # Atomic rename (need to handle .npz suffix)
            final_temp = Path(str(temp_path).replace(".tmp.npz", ".tmp"))
            if temp_path.exists():
                shutil.move(temp_path, path)

            logger.debug("Saved optimizer weights")
            return True
        except Exception as e:
            logger.error(f"Failed to save optimizer: {e}")
            return False

    def _load_optimizer(self) -> bool:
        """Load SovereignOptimizer weights from NPZ."""
        try:
            path = self._get_path(self.config.optimizer_file)

            if not path.exists():
                logger.info("No saved optimizer found, starting fresh")
                return True

            data = np.load(path, allow_pickle=True)

            # Check version
            version = int(data.get("schema_version", [1])[0])
            if version != self.config.schema_version:
                logger.warning(f"Optimizer schema version mismatch: {version}")

            # Restore weights
            if "w1" in data and hasattr(self._optimizer, 'w1'):
                self._optimizer.w1 = data["w1"]
            if "w2" in data and hasattr(self._optimizer, 'w2'):
                self._optimizer.w2 = data["w2"]
            if "b1" in data and hasattr(self._optimizer, 'b1'):
                self._optimizer.b1 = data["b1"]
            if "b2" in data and hasattr(self._optimizer, 'b2'):
                self._optimizer.b2 = data["b2"]

            # Restore replay buffer
            if "replay_buffer" in data and hasattr(self._optimizer, 'replay_buffer'):
                buffer_json = str(data["replay_buffer"])
                # Replay buffer restoration would need FeedbackEvent reconstruction
                # Skipping for now - buffer rebuilds naturally

            logger.debug("Loaded optimizer weights")
            return True
        except Exception as e:
            logger.error(f"Failed to load optimizer: {e}")
            return False

    # ─── Encoding Utilities ────────────────────────────────────

    def _encode_array(self, arr: np.ndarray) -> str:
        """Encode numpy array as base64 string."""
        import base64
        return base64.b64encode(arr.astype(np.float32).tobytes()).decode('ascii')

    def _decode_array(self, encoded: str) -> np.ndarray:
        """Decode base64 string to numpy array."""
        import base64
        raw = base64.b64decode(encoded.encode('ascii'))
        return np.frombuffer(raw, dtype=np.float32)

    # ─── Status ────────────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        """Get current session status."""
        return {
            "initialized": self._initialized,
            "base_path": str(self.config.base_path),
            "dirty": {
                "graph": self.dirty.graph,
                "mythos": self.dirty.mythos,
                "optimizer": self.dirty.optimizer,
                "total_changes": self.dirty.graph_changes + self.dirty.mythos_changes + self.dirty.optimizer_updates
            },
            "registered": {
                "graph": self._graph is not None,
                "mythos": self._mythos is not None,
                "optimizer": self._optimizer is not None
            },
            "last_checkpoint": datetime.fromtimestamp(self._last_checkpoint).isoformat() if self._last_checkpoint else None,
            "checkpoint_timer_running": self._checkpoint_thread is not None and self._checkpoint_thread.is_alive()
        }

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset singleton instance (for testing)."""
        with SessionManagerMeta._lock:
            if cls in SessionManagerMeta._instances:
                instance = SessionManagerMeta._instances[cls]
                instance._stop_checkpoint.set()
                del SessionManagerMeta._instances[cls]


# ═══════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def get_session() -> SessionManager:
    """Get the singleton SessionManager instance."""
    return SessionManager()


def init_session(
    graph: Any = None,
    mythos: Any = None,
    optimizer: Any = None,
    auto_load: bool = True,
    start_timer: bool = True
) -> SessionManager:
    """
    Initialize session with components.

    Args:
        graph: RepoGraph to persist
        mythos: MythosLibrary to persist
        optimizer: SovereignOptimizer to persist
        auto_load: Load saved state on init
        start_timer: Start checkpoint timer

    Returns:
        Configured SessionManager
    """
    session = get_session()

    if graph:
        session.register_graph(graph)
    if mythos:
        session.register_mythos(mythos)
    if optimizer:
        session.register_optimizer(optimizer)

    if auto_load:
        session.load_all()

    if start_timer:
        session.start_checkpoint_timer()

    return session
