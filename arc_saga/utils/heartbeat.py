# arc_saga/utils/heartbeat.py
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_LOG_DIR = Path.home() / ".arc_saga" / "logs"
DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)

def _safe_task_filename(task_id: str) -> str:
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in task_id)
    return f"heartbeat_{safe}.log"

@dataclass
class HeartbeatConfig:
    task_id: str
    interval_seconds: int = 10
    log_dir: Path = DEFAULT_LOG_DIR
    per_task_file: bool = True
    logger_name: str = "arc_saga.heartbeat"

class HeartbeatManager:
    def __init__(
        self,
        task_id: str,
        interval_seconds: int = 10,
        log_dir: Optional[Path] = None,
        per_task_file: bool = True,
        logger_name: str = "arc_saga.heartbeat",
    ) -> None:
        if not task_id:
            raise ValueError("task_id must be a non-empty string")
        self.config = HeartbeatConfig(
            task_id=task_id,
            interval_seconds=interval_seconds,
            log_dir=(log_dir or DEFAULT_LOG_DIR),
            per_task_file=per_task_file,
            logger_name=logger_name,
        )
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self._task: Optional[asyncio.Task[None]] = None
        self._stopped = asyncio.Event()
        self._logger = self._configure_logger()

    def _configure_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.config.logger_name)
        logger.setLevel(logging.INFO)
        # Avoid duplicate handlers
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            filename = _safe_task_filename(self.config.task_id) if self.config.per_task_file else "heartbeat.log"
            file_path = self.config.log_dir / filename
            handler = logging.FileHandler(file_path, mode="a", encoding="utf-8")
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            handler.setLevel(logging.INFO)
            logger.addHandler(handler)
            # Stream handler for console
            if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
                stream_handler = logging.StreamHandler()
                stream_handler.setFormatter(formatter)
                logger.addHandler(stream_handler)
            # Ensure flush on each emit
            handler.flush = handler.stream.flush  # type: ignore[attr-defined]
        return logger

    async def _heartbeat_loop(self) -> None:
        interval = max(1, int(self.config.interval_seconds))
        tag = self.config.task_id
        try:
            while not self._stopped.is_set():
                try:
                    self._logger.info("[%s] Heartbeat still alive at %s", tag, time.strftime("%X"))
                except Exception:
                    with contextlib.suppress(Exception):
                        logging.getLogger("arc_saga.heartbeat").exception("Failed to write heartbeat for %s", tag)
                try:
                    await asyncio.wait_for(self._stopped.wait(), timeout=interval)
                except asyncio.TimeoutError:
                    continue
        finally:
            with contextlib.suppress(Exception):
                self._logger.info("[%s] Heartbeat stopped at %s", tag, time.strftime("%X"))

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stopped.clear()
        loop = asyncio.get_running_loop()
        self._task = loop.create_task(self._heartbeat_loop())

    async def stop(self, timeout: float = 5.0) -> None:
        if not self._task:
            return
        self._stopped.set()
        try:
            await asyncio.wait_for(self._task, timeout=timeout)
        except asyncio.TimeoutError:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        finally:
            self._task = None