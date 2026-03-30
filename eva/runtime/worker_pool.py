from __future__ import annotations

import os
import sys
import queue as pyqueue
import multiprocessing as mp
from contextlib import contextmanager
from importlib import import_module
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

import torch

from eva.adapters.torch_adapter import Batch
from eva.core.device_resolver import (
    DeviceConfig,
    resolve_device,
    select_precision,
    autocast_context,
)
from eva.core.batch_wrapper import assert_clean_batch, emit_wrapper_event, WrapperMetadata
import logging

logger = logging.getLogger(__name__)


def _resolve_callable(path: str):
    mod_name, func_name = path.rsplit(".", 1)
    mod = import_module(mod_name)
    return getattr(mod, func_name)


def _sanitize_batch_inplace(batch: Batch) -> list[str]:
    """Drop any forbidden wrapper artefact keys from batch.tensors if present.
    Returns list of actually dropped keys.
    """
    dropped: list[str] = []
    try:
        drop = {"_wrapper", "envelope_meta", "wrapper_meta"}
        if isinstance(batch, Batch) and isinstance(batch.tensors, dict):
            for k in list(batch.tensors.keys()):
                if k in drop:
                    batch.tensors.pop(k, None)
                    dropped.append(k)
    except Exception:
        return dropped
    return dropped


def _worker_entry(
    in_q: mp.Queue,
    out_q: mp.Queue,
    model_fn_path: str,
    torch_threads: int,
    interop_threads: int,
):
    # Configure threading to avoid oversubscription
    os.environ.setdefault("OMP_NUM_THREADS", str(torch_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(torch_threads))
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    torch.set_num_threads(max(1, int(torch_threads)))
    torch.set_num_interop_threads(max(1, int(interop_threads)))
    torch.set_grad_enabled(False)

    model_fn = _resolve_callable(model_fn_path)

    # Resolve device/precision locally in worker for AMP
    try:
        dev_cfg = DeviceConfig()
        device = resolve_device(dev_cfg)
        precision = select_precision(device, dev_cfg)
    except Exception:
        device = torch.device("cpu")
        precision = "fp32"  # type: ignore[assignment]

    while True:
        item = in_q.get()
        if item is None:
            break
        batch_id, batch = item  # type: ignore
        try:
            # Autocast on GPU with selected precision; inference_mode inside context
            # Core barrier: enforce clean batch before model call
            try:
                assert_clean_batch(batch)
            except AssertionError:
                logger.warning("[worker_pool] Detected wrapper artefacts in batch inside worker; sanitizing")
                _sanitize_batch_inplace(batch)
            with autocast_context(device, precision):
                result = model_fn(batch)
        except Exception as e:
            out_q.put((batch_id, e))
        else:
            out_q.put((batch_id, result))


class InferenceWorkerPool:
    def __init__(
        self,
        model_fn_path: str,
        num_workers: Optional[int] = None,
        torch_threads: int = 2,
        interop_threads: int = 1,
        queue_maxsize: int = 64,
        start_method: Optional[str] = None,
        events: Optional[Any] = None,
    ) -> None:
        self.model_fn_path = model_fn_path
        self.num_workers = num_workers or max(1, mp.cpu_count() // 2)
        self.torch_threads = torch_threads
        self.interop_threads = interop_threads
        self.queue_maxsize = queue_maxsize
        self._ctx = mp.get_context(start_method or "spawn")
        self._in_q: mp.Queue = self._ctx.Queue(maxsize=queue_maxsize)
        self._out_q: mp.Queue = self._ctx.Queue(maxsize=queue_maxsize)
        self._procs: list[mp.Process] = []
        self._next_id: int = 0
        # Optional event bus for telemetry
        self._events = events

    def start(self) -> None:
        if self._procs:
            return
        for _ in range(self.num_workers):
            p = self._ctx.Process(
                target=_worker_entry,
                args=(self._in_q, self._out_q, self.model_fn_path, self.torch_threads, self.interop_threads),
                daemon=True,
            )
            p.start()
            self._procs.append(p)

    def stop(self) -> None:
        if not self._procs:
            return
        for _ in self._procs:
            self._in_q.put(None)
        for p in self._procs:
            p.join(timeout=5)
        self._procs.clear()

    @contextmanager
    def running(self):
        self.start()
        try:
            yield self
        finally:
            self.stop()

    def submit(self, batch: Batch) -> int:
        # Core barrier in main process: ensure batch cleanliness before enqueue
        try:
            assert_clean_batch(batch)
        except AssertionError as e:
            # Sanitize and emit anomaly telemetry
            dropped = _sanitize_batch_inplace(batch)
            logger.warning("[worker_pool] Detected wrapper artefacts in batch on submit; sanitized keys=%s", dropped)
            try:
                emit_wrapper_event(
                    "wrapper_anomaly",
                    WrapperMetadata(source="worker_pool.submit"),
                    events=self._events,
                    extra={"reason": "forbidden_keys_dropped", "keys": dropped},
                )
            except Exception:
                pass
        bid = self._next_id
        self._next_id += 1
        self._in_q.put((bid, batch))
        return bid

    def recv(self, block: bool = True, timeout: Optional[float] = None) -> Tuple[int, Any]:
        return self._out_q.get(block=block, timeout=timeout)

    def infer_batches(self, batches: Iterable[Batch]) -> Iterator[Any]:
        pending: pyqueue.Queue[int] = pyqueue.Queue()
        # Submit pipeline
        for b in batches:
            bid = self.submit(b)
            pending.put(bid)
        # Collect
        received = 0
        total = pending.qsize()
        while received < total:
            bid, res = self.recv()
            received += 1
            if isinstance(res, Exception):
                raise res
            yield res
