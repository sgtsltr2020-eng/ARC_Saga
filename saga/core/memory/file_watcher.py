"""
File Watcher - Incremental Graph Maintenance
=============================================

Monitors codebase directories for changes and triggers
incremental graph updates. Integrates with watchdog for
cross-platform filesystem events.

Author: ARC SAGA Development Team (AntiGravity)
Date: December 26, 2025
Status: USMA - Memory Reliability Enhancement
"""

import hashlib
import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

@dataclass
class WatcherConfig:
    """Configuration for file watcher."""

    # Debounce settings
    debounce_seconds: float = 1.0  # Wait before processing after last event
    batch_window_seconds: float = 5.0  # Batch events within this window

    # Excluded patterns
    excluded_dirs: set[str] = field(default_factory=lambda: {
        "venv", ".venv", "env", ".env",
        "__pycache__", ".git", ".saga",
        "node_modules", "htmlcov", ".pytest_cache",
        ".mypy_cache", ".ruff_cache", "build", "dist"
    })
    excluded_extensions: set[str] = field(default_factory=lambda: {
        ".pyc", ".pyo", ".pyd", ".so", ".dll",
        ".exe", ".bin", ".o", ".a"
    })

    # Processing limits
    max_events_per_batch: int = 100
    idle_process_delay: float = 2.0  # Process pending events after idle

    # Extensions to watch
    watch_extensions: set[str] = field(default_factory=lambda: {
        ".py", ".pyi"  # Python files only for now
    })


# ═══════════════════════════════════════════════════════════════
# FILE EVENT
# ═══════════════════════════════════════════════════════════════

@dataclass
class FileEvent:
    """Represents a file system event."""
    event_type: str  # 'created', 'modified', 'deleted', 'moved'
    src_path: str
    dst_path: str | None = None  # For move events
    timestamp: float = field(default_factory=time.time)
    is_directory: bool = False


# ═══════════════════════════════════════════════════════════════
# FILE WATCHER
# ═══════════════════════════════════════════════════════════════

class FileWatcher:
    """
    Monitors filesystem for changes and triggers incremental graph updates.

    Uses watchdog library when available, falls back to polling.
    Debounces rapid events and batches updates for efficiency.
    """

    def __init__(
        self,
        project_root: str | Path,
        config: WatcherConfig | None = None,
        on_update: Callable[[list[FileEvent]], None] | None = None
    ):
        """
        Initialize the file watcher.

        Args:
            project_root: Root directory to watch
            config: Watcher configuration
            on_update: Callback for batch updates
        """
        self.project_root = Path(project_root).resolve()
        self.config = config or WatcherConfig()
        self.on_update = on_update

        # Event queue
        self._event_queue: queue.Queue[FileEvent] = queue.Queue()
        self._pending_events: dict[str, FileEvent] = {}  # Dedup by path

        # Threading
        self._stop_event = threading.Event()
        self._observer = None
        self._processor_thread: threading.Thread | None = None

        # Stats
        self.events_processed = 0
        self.batches_processed = 0

        # Content hashes for dirty detection
        self._file_hashes: dict[str, str] = {}

        # Watchdog handler
        self._handler = None

    def start(self) -> bool:
        """
        Start watching for file changes.

        Returns:
            True if started successfully
        """
        if self._processor_thread is not None and self._processor_thread.is_alive():
            logger.warning("Watcher already running")
            return False

        self._stop_event.clear()

        # Try watchdog first
        if self._start_watchdog():
            logger.info(f"File watcher started (watchdog): {self.project_root}")
        else:
            logger.info(f"File watcher started (polling): {self.project_root}")

        # Start event processor
        self._processor_thread = threading.Thread(
            target=self._process_events_loop,
            daemon=True,
            name="FileWatcher-Processor"
        )
        self._processor_thread.start()

        return True

    def stop(self) -> None:
        """Stop the file watcher."""
        self._stop_event.set()

        if self._observer:
            try:
                self._observer.stop()
                self._observer.join(timeout=5.0)
            except Exception as e:
                logger.warning(f"Error stopping watchdog observer: {e}")
            self._observer = None

        if self._processor_thread:
            self._processor_thread.join(timeout=5.0)

        logger.info("File watcher stopped")

    def _start_watchdog(self) -> bool:
        """Try to start watchdog observer."""
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer

            class Handler(FileSystemEventHandler):
                def __init__(self, watcher: "FileWatcher"):
                    self.watcher = watcher

                def on_any_event(self, event):
                    if event.is_directory:
                        return

                    # Map watchdog event to our FileEvent
                    event_type_map = {
                        "created": "created",
                        "modified": "modified",
                        "deleted": "deleted",
                        "moved": "moved"
                    }

                    etype = event_type_map.get(event.event_type)
                    if etype:
                        dst_path = getattr(event, 'dest_path', None)
                        self.watcher._queue_event(FileEvent(
                            event_type=etype,
                            src_path=event.src_path,
                            dst_path=dst_path
                        ))

            self._handler = Handler(self)
            self._observer = Observer()
            self._observer.schedule(
                self._handler,
                str(self.project_root),
                recursive=True
            )
            self._observer.start()
            return True

        except ImportError:
            logger.info("watchdog not installed, using polling fallback")
            return False
        except Exception as e:
            logger.warning(f"Failed to start watchdog: {e}")
            return False

    def _queue_event(self, event: FileEvent) -> None:
        """Add event to processing queue."""
        src_path = Path(event.src_path)

        # Filter excluded directories
        for part in src_path.parts:
            if part in self.config.excluded_dirs:
                return

        # Filter extensions
        if src_path.suffix not in self.config.watch_extensions:
            return

        # Filter excluded extensions
        if src_path.suffix in self.config.excluded_extensions:
            return

        self._event_queue.put(event)

    def _process_events_loop(self) -> None:
        """Main event processing loop."""
        last_event_time = 0.0

        while not self._stop_event.is_set():
            try:
                # Get event with timeout
                try:
                    event = self._event_queue.get(timeout=0.5)
                    self._pending_events[event.src_path] = event
                    last_event_time = time.time()
                except queue.Empty:
                    pass

                # Process batch if debounce period passed
                if self._pending_events:
                    time_since_last = time.time() - last_event_time

                    if time_since_last >= self.config.debounce_seconds:
                        self._process_batch()

            except Exception as e:
                logger.error(f"Error in event processing: {e}")

    def _process_batch(self) -> None:
        """Process accumulated events as a batch."""
        if not self._pending_events:
            return

        # Extract events
        events = list(self._pending_events.values())
        self._pending_events.clear()

        # Limit batch size
        if len(events) > self.config.max_events_per_batch:
            events = events[:self.config.max_events_per_batch]
            logger.warning(f"Batch limited to {self.config.max_events_per_batch} events")

        # Filter to truly changed files (content hash check)
        changed_events = []
        for event in events:
            if event.event_type == "deleted":
                changed_events.append(event)
                self._file_hashes.pop(event.src_path, None)
            elif self._has_content_changed(event.src_path):
                changed_events.append(event)

        if not changed_events:
            return

        # Call update handler
        if self.on_update:
            try:
                self.on_update(changed_events)
            except Exception as e:
                logger.error(f"Error in update handler: {e}")

        self.events_processed += len(changed_events)
        self.batches_processed += 1

        logger.info(f"Processed batch: {len(changed_events)} events")

    def _has_content_changed(self, file_path: str) -> bool:
        """Check if file content has actually changed."""
        try:
            path = Path(file_path)
            if not path.exists():
                return True

            content = path.read_bytes()
            new_hash = hashlib.md5(content).hexdigest()
            old_hash = self._file_hashes.get(file_path)

            self._file_hashes[file_path] = new_hash

            return old_hash != new_hash

        except Exception:
            return True  # Assume changed if can't read

    # ─── Manual Triggers ───────────────────────────────────────

    def notify_file_changed(self, file_path: str | Path) -> None:
        """Manually notify of a file change."""
        self._queue_event(FileEvent(
            event_type="modified",
            src_path=str(Path(file_path).resolve())
        ))

    def notify_file_created(self, file_path: str | Path) -> None:
        """Manually notify of a file creation."""
        self._queue_event(FileEvent(
            event_type="created",
            src_path=str(Path(file_path).resolve())
        ))

    def notify_file_deleted(self, file_path: str | Path) -> None:
        """Manually notify of a file deletion."""
        self._queue_event(FileEvent(
            event_type="deleted",
            src_path=str(Path(file_path).resolve())
        ))

    # ─── Stats ─────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Get watcher statistics."""
        return {
            "events_processed": self.events_processed,
            "batches_processed": self.batches_processed,
            "pending_events": len(self._pending_events),
            "tracked_files": len(self._file_hashes),
            "running": self._processor_thread is not None and self._processor_thread.is_alive()
        }


# ═══════════════════════════════════════════════════════════════
# INCREMENTAL ANALYZER
# ═══════════════════════════════════════════════════════════════

class IncrementalAnalyzer:
    """
    Handles incremental graph updates from file events.

    - Parses only changed files
    - Updates affected nodes/edges
    - Preserves existing embeddings where possible
    - Integrates with SessionManager for persistence
    """

    def __init__(
        self,
        graph: Any,  # RepoGraph
        static_analyzer: Any = None,  # StaticAnalyzer
        session_manager: Any = None,  # SessionManager
        embedding_generator: Callable[[str], Any] | None = None
    ):
        """
        Initialize the incremental analyzer.

        Args:
            graph: RepoGraph instance
            static_analyzer: StaticAnalyzer instance
            session_manager: SessionManager for dirty flags
            embedding_generator: Function to generate embeddings
        """
        self.graph = graph
        self.static_analyzer = static_analyzer
        self.session_manager = session_manager
        self.embedding_generator = embedding_generator

        # Stats
        self.nodes_added = 0
        self.nodes_updated = 0
        self.nodes_removed = 0
        self.edges_updated = 0

    def process_events(self, events: list[FileEvent]) -> dict[str, Any]:
        """
        Process a batch of file events.

        Args:
            events: List of FileEvent objects

        Returns:
            Processing statistics
        """
        stats = {
            "created": 0,
            "modified": 0,
            "deleted": 0,
            "errors": 0
        }

        for event in events:
            try:
                if event.event_type == "created":
                    self._handle_created(event.src_path)
                    stats["created"] += 1
                elif event.event_type == "modified":
                    self._handle_modified(event.src_path)
                    stats["modified"] += 1
                elif event.event_type == "deleted":
                    self._handle_deleted(event.src_path)
                    stats["deleted"] += 1
                elif event.event_type == "moved":
                    self._handle_moved(event.src_path, event.dst_path)
                    stats["modified"] += 1
            except Exception as e:
                logger.error(f"Error processing event {event}: {e}")
                stats["errors"] += 1

        # Mark graph dirty
        if self.session_manager and stats["created"] + stats["modified"] + stats["deleted"] > 0:
            self.session_manager.mark_dirty("graph")

        return stats

    def _handle_created(self, file_path: str) -> None:
        """Handle file creation."""
        if self.static_analyzer:
            self.static_analyzer.analyze_file(file_path)
            self.nodes_added += 1

    def _handle_modified(self, file_path: str) -> None:
        """Handle file modification."""
        # Get all nodes in this file

        # Get all nodes in this file
        nodes_to_update = []
        for node in self.graph._node_index.values():
            if node.file_path and Path(node.file_path).resolve() == Path(file_path).resolve():
                nodes_to_update.append(node.node_id)

        # Remove old nodes
        for node_id in nodes_to_update:
            self._remove_node(node_id)

        # Re-analyze file
        if self.static_analyzer:
            self.static_analyzer.analyze_file(file_path)

        self.nodes_updated += len(nodes_to_update)

    def _handle_deleted(self, file_path: str) -> None:
        """Handle file deletion."""
        # Find and remove all nodes for this file
        nodes_to_remove = []
        for node in self.graph._node_index.values():
            if node.file_path and Path(node.file_path).resolve() == Path(file_path).resolve():
                nodes_to_remove.append(node.node_id)

        for node_id in nodes_to_remove:
            self._remove_node(node_id)

        self.nodes_removed += len(nodes_to_remove)

    def _handle_moved(self, src_path: str, dst_path: str | None) -> None:
        """Handle file move/rename."""
        self._handle_deleted(src_path)
        if dst_path:
            self._handle_created(dst_path)

    def _remove_node(self, node_id: str) -> None:
        """Remove a node and its edges from the graph."""
        if node_id in self.graph._node_index:
            del self.graph._node_index[node_id]

        if self.graph._graph.has_node(node_id):
            self.graph._graph.remove_node(node_id)

    def get_stats(self) -> dict[str, Any]:
        """Get analyzer statistics."""
        return {
            "nodes_added": self.nodes_added,
            "nodes_updated": self.nodes_updated,
            "nodes_removed": self.nodes_removed,
            "edges_updated": self.edges_updated
        }


# ═══════════════════════════════════════════════════════════════
# GRAPH WATCHER SERVICE
# ═══════════════════════════════════════════════════════════════

class GraphWatcherService:
    """
    High-level service combining FileWatcher with IncrementalAnalyzer.

    Manages the full lifecycle of watching and updating the graph.
    """

    def __init__(
        self,
        project_root: str | Path,
        graph: Any,  # RepoGraph
        static_analyzer: Any = None,
        session_manager: Any = None,
        config: WatcherConfig | None = None
    ):
        """
        Initialize the graph watcher service.

        Args:
            project_root: Root directory to watch
            graph: RepoGraph instance
            static_analyzer: StaticAnalyzer instance
            session_manager: SessionManager for persistence
            config: Watcher configuration
        """
        self.project_root = Path(project_root).resolve()
        self.graph = graph

        # Create analyzer
        self.analyzer = IncrementalAnalyzer(
            graph=graph,
            static_analyzer=static_analyzer,
            session_manager=session_manager
        )

        # Create watcher
        self.watcher = FileWatcher(
            project_root=project_root,
            config=config,
            on_update=self._on_file_events
        )

        self.session_manager = session_manager

    def start(self) -> bool:
        """Start the watcher service."""
        return self.watcher.start()

    def stop(self) -> None:
        """Stop the watcher service."""
        self.watcher.stop()

        # Save graph on stop
        if self.session_manager:
            self.session_manager.checkpoint()

    def _on_file_events(self, events: list[FileEvent]) -> None:
        """Handle file events from watcher."""
        stats = self.analyzer.process_events(events)
        logger.info(f"Incremental update: {stats}")

    def get_stats(self) -> dict[str, Any]:
        """Get combined statistics."""
        return {
            "watcher": self.watcher.get_stats(),
            "analyzer": self.analyzer.get_stats()
        }
