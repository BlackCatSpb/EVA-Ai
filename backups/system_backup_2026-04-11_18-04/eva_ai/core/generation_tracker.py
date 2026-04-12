"""
Generation Tracker for EVA — async generation monitoring via DeferredCommandSystem.

Tracks pipeline generation lifecycle: command_id, status, timing, progress, current_step.
Publishes generation.progress events at each step.
"""

import time
import uuid
import threading
import logging
from typing import Dict, Any, Optional, Callable
from enum import Enum

logger = logging.getLogger("eva_ai.core.generation_tracker")


class GenerationStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class GenerationTracker:
    """Tracks active text-generation commands and exposes status via DeferredCommandSystem."""

    def __init__(self, deferred_system=None, event_bus=None):
        self.deferred_system = deferred_system
        self.event_bus = event_bus

        self._active: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public lifecycle API
    # ------------------------------------------------------------------

    def start_generation(self, query: str, source: str = "unknown") -> str:
        """Register a new generation command and return its command_id."""
        command_id = f"gen_{uuid.uuid4().hex[:12]}"
        entry = {
            "command_id": command_id,
            "status": GenerationStatus.PENDING.value,
            "query": query[:200],
            "source": source,
            "start_time": time.time(),
            "elapsed": 0.0,
            "progress": 0,
            "current_step": "registered",
            "response": None,
            "error": None,
        }
        with self._lock:
            self._active[command_id] = entry

        # Register with DeferredCommandSystem if available
        if self.deferred_system is not None:
            try:
                from eva_ai.core.deferred_command_system import CommandPriority
                self.deferred_system.add_command(
                    command=lambda: entry,
                    command_id=command_id,
                    priority=CommandPriority.HIGH,
                    max_retries=0,
                    timeout=360,
                )
            except Exception as e:
                logger.debug(f"DeferredCommandSystem registration failed: {e}")

        self._publish("generation.started", entry)
        logger.info(f"Generation started: {command_id} (source={source})")
        return command_id

    def update_progress(self, command_id: str, step: str, percent: float):
        """Update progress for an active generation."""
        with self._lock:
            entry = self._active.get(command_id)
            if entry is None:
                return
            entry["current_step"] = step
            entry["progress"] = max(0, min(100, percent))
            entry["elapsed"] = time.time() - entry["start_time"]
            entry["status"] = GenerationStatus.RUNNING.value

        self._publish("generation.progress", {
            "command_id": command_id,
            "step": step,
            "progress": entry["progress"],
            "elapsed": entry["elapsed"],
        })

    def complete(self, command_id: str, response: str):
        """Mark a generation as completed."""
        with self._lock:
            entry = self._active.get(command_id)
            if entry is None:
                return
            entry["status"] = GenerationStatus.COMPLETED.value
            entry["response"] = response
            entry["progress"] = 100
            entry["current_step"] = "done"
            entry["elapsed"] = time.time() - entry["start_time"]

        self._publish("generation.completed", {
            "command_id": command_id,
            "elapsed": entry["elapsed"],
            "response_length": len(response) if response else 0,
        })
        logger.info(f"Generation completed: {command_id} ({entry['elapsed']:.2f}s)")

    def fail(self, command_id: str, error: str):
        """Mark a generation as failed."""
        with self._lock:
            entry = self._active.get(command_id)
            if entry is None:
                return
            entry["status"] = GenerationStatus.FAILED.value
            entry["error"] = error
            entry["current_step"] = "error"
            entry["elapsed"] = time.time() - entry["start_time"]

        self._publish("generation.failed", {
            "command_id": command_id,
            "error": error,
            "elapsed": entry["elapsed"],
        })
        logger.warning(f"Generation failed: {command_id} — {error}")

    def timeout(self, command_id: str, timeout_seconds: float):
        """Mark a generation as timed out."""
        with self._lock:
            entry = self._active.get(command_id)
            if entry is None:
                return
            entry["status"] = GenerationStatus.TIMEOUT.value
            entry["current_step"] = "timeout"
            entry["elapsed"] = timeout_seconds

        self._publish("generation.timeout", {
            "command_id": command_id,
            "timeout_seconds": timeout_seconds,
        })
        logger.warning(f"Generation timeout: {command_id} ({timeout_seconds}s)")

    # ------------------------------------------------------------------
    # Status queries
    # ------------------------------------------------------------------

    def get_status(self, command_id: str) -> Optional[Dict[str, Any]]:
        """Return status dict for a single generation."""
        with self._lock:
            entry = self._active.get(command_id)
            if entry is None:
                return None
            snapshot = dict(entry)
            snapshot["elapsed"] = time.time() - entry["start_time"] if entry["status"] in (
                GenerationStatus.PENDING.value, GenerationStatus.RUNNING.value
            ) else entry["elapsed"]
            return snapshot

    def get_all_active(self) -> Dict[str, Dict[str, Any]]:
        """Return snapshots of all active (non-terminal) generations."""
        with self._lock:
            result = {}
            for cid, entry in self._active.items():
                if entry["status"] in (GenerationStatus.PENDING.value, GenerationStatus.RUNNING.value):
                    snapshot = dict(entry)
                    snapshot["elapsed"] = time.time() - entry["start_time"]
                    result[cid] = snapshot
            return result

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """Return snapshots of all tracked generations."""
        with self._lock:
            result = {}
            for cid, entry in self._active.items():
                snapshot = dict(entry)
                if entry["status"] in (GenerationStatus.PENDING.value, GenerationStatus.RUNNING.value):
                    snapshot["elapsed"] = time.time() - entry["start_time"]
                result[cid] = snapshot
            return result

    def cleanup_completed(self, max_age: float = 300.0):
        """Remove completed/failed/timeout entries older than max_age seconds."""
        now = time.time()
        with self._lock:
            to_remove = [
                cid for cid, e in self._active.items()
                if e["status"] in (
                    GenerationStatus.COMPLETED.value,
                    GenerationStatus.FAILED.value,
                    GenerationStatus.TIMEOUT.value,
                )
                and (now - e["start_time"] - e["elapsed"]) > max_age
            ]
            for cid in to_remove:
                del self._active[cid]
        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} old generation entries")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _publish(self, event_type: str, data: Dict[str, Any]):
        if self.event_bus is not None:
            try:
                self.event_bus.publish(event_type, data)
            except Exception as e:
                logger.debug(f"Event publish failed for {event_type}: {e}")
