"""
Parallel tokenization skeleton honoring a global in-memory window and IO-limited cache.
This module defines a dispatcher that will, in future iterations, run per-core workers
with individual tokenizer/model instances. For now, it provides a safe, testable stub.
"""
from __future__ import annotations
import os
import time
import math
import queue
import threading
from typing import Any, Dict, Iterable, List, Optional, Tuple, Callable
import logging

logger = logging.getLogger(__name__)

# Note: Memory/CPU/IO arbitration is delegated to CoreBrain.resource_queue (GlobalResourceQueue)

class ParallelTokenizer:
    """Dispatcher skeleton: accepts text batches, simulates tokenization, and records routing.

    This is a placeholder for the full per-core pipeline. It enforces the memory window via
    ByteSemaphore and integrates with CacheRouter for indexing segments.
    """
    def __init__(self, brain, max_data_window_bytes: int = 3 * 1024**3, worker_count: Optional[int] = None):
        self.brain = brain
        self.max_data_window_bytes = int(max_data_window_bytes)
        try:
            phys_cores = max(1, (os.cpu_count() or 2) // 2)
            default_workers = min(phys_cores, 4)
        except Exception:
            default_workers = 2
        self.worker_count = int(worker_count or default_workers)
        self._stop = threading.Event()
        self._in_q: "queue.Queue[Tuple[str, str]]" = queue.Queue(maxsize=64)  # (batch_id, text)
        self._threads: List[threading.Thread] = []

    def start(self) -> None:
        for i in range(self.worker_count):
            th = threading.Thread(target=self._worker, name=f"tok-worker-{i}", daemon=True)
            th.start()
            self._threads.append(th)
        logger.info(f"ParallelTokenizer started with {self.worker_count} workers")

    def stop(self) -> None:
        self._stop.set()
        for _ in self._threads:
            self._in_q.put(("__STOP__", ""))
        for th in self._threads:
            th.join(timeout=2.0)
        logger.info("ParallelTokenizer stopped")

    def submit(self, batch_id: str, text: str) -> bool:
        try:
            self._in_q.put_nowait((batch_id, text))
            return True
        except queue.Full:
            return False

    # --- Internal worker ---
    def _worker(self) -> None:
        # Acquire one CPU token for the lifetime of the worker
        rq = getattr(self.brain, 'resource_queue', None)
        got_cpu = False
        try:
            if rq is not None:
                got_cpu = rq.acquire_cpu(1, timeout=5.0)
                if not got_cpu:
                    logger.debug("Worker failed to acquire CPU token; exiting")
                    return
            while not self._stop.is_set():
                try:
                    batch_id, text = self._in_q.get(timeout=0.25)
                except queue.Empty:
                    continue
                if batch_id == "__STOP__":
                    break
                try:
                    self._process_item(batch_id, text)
                except Exception as e:
                    logger.warning(f"ParallelTokenizer worker error: {e}")
        finally:
            if rq is not None and got_cpu:
                try:
                    rq.release_cpu(1)
                except Exception:
                    pass

    def _process_item(self, batch_id: str, text: str) -> None:
        # Simulate tokenization: we encode bytes and split into segments of ~64KB
        # In the real implementation, we'd call the QWEN tokenizer/model.
        raw = text.encode("utf-8", errors="ignore")
        nbytes = len(raw)
        rq = getattr(self.brain, 'resource_queue', None)
        if rq is not None:
            ok = rq.acquire_memory(nbytes, timeout=5.0)
            if not ok:
                logger.debug("RAM window saturated; dropping item")
                return
        try:
            if getattr(self.brain, 'cache_router', None) is None:
                logger.debug("No cache_router; skipping index registration")
                return
            router = self.brain.cache_router
            router.register_batch(batch_id=batch_id, source="text", total_tokens=nbytes // 3, priority=0.0, status="processing")
            seg_size = 64 * 1024
            offset = 0
            while offset < nbytes:
                chunk = raw[offset: offset + seg_size]
                seg_id = router.register_segment(
                    batch_id=batch_id,
                    offset=offset,
                    length=len(chunk),
                    token_count=max(1, len(chunk) // 3),
                    disk_path=self._persist_stub(batch_id, offset, chunk),
                    checksum="",
                )
                # Minimal nodes: start/end boundaries to create addressable entries
                spans = [(0, max(1, len(chunk) // 3))]
                router.register_token_nodes(seg_id, spans)
                # Set a default weight placeholder
                router.set_weight(seg_id, item_type="segment", weight_type="task_relevance", value=0.0)
                offset += seg_size
            router.index.upsert_batch(batch_id, source="text", total_tokens=nbytes // 3, priority=0.0, status="done")
        finally:
            if rq is not None:
                try:
                    rq.release_memory(nbytes)
                except Exception:
                    pass

    def _persist_stub(self, batch_id: str, offset: int, data: bytes) -> str:
        # Persist to hybrid cache disk path in a deterministic subfolder for stubs
        base_dir = os.path.join(self.brain.cache_dir, "hybrid_cache", "disk_storage", "segments")
        os.makedirs(base_dir, exist_ok=True)
        fname = f"{batch_id}_{offset:012d}.bin"
        fpath = os.path.join(base_dir, fname)
        rq = getattr(self.brain, 'resource_queue', None)
        try:
            if rq is not None:
                rq.acquire_io(len(data))  # throttle by global IO rate
            # Underlying DiskCache also throttles; GRQ provides system-wide arbitration
            with open(fpath, "wb") as f:
                f.write(data)
        except Exception as e:
            logger.warning(f"persist_stub failed: {e}")
        return fpath
