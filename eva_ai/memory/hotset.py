
from __future__ import annotations

import time
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


@dataclass
class _Entry:
    key: Any
    bytes: int
    tensor: "torch.Tensor"  # device tensor
    cpu_mirror: Optional["torch.Tensor"] = None  # optional pinned CPU mirror
    last: float = 0.0
    priority: int = 0


class HotSetManager:
    """
    Simple LRU HotSet for GPU tensors with optional CPU pinned mirrors.

    Policy:
    - Keep usage under target fraction of total VRAM.
    - Evict LRU entries when free memory low or on demand.
    - Promote from CPU pinned or host-view to device non_blocking on a private stream.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        target_vram_frac: float = 0.65,
        max_bytes: Optional[int] = None,
        keep_cpu_mirror: bool = True,
    ) -> None:
        if torch is None or not torch.cuda.is_available():  # pragma: no cover
            raise RuntimeError("CUDA is not available; HotSetManager requires a CUDA device")
        self.device = device
        self.keep_cpu_mirror = keep_cpu_mirror
        self.target_vram_frac = max(0.1, min(0.95, float(target_vram_frac)))
        self.max_bytes = max_bytes  # Optional hard cap
        self._lru: "OrderedDict[Any, _Entry]" = OrderedDict()
        self._stream = torch.cuda.Stream(device=self.device)
        # Telemetry
        self._hits = 0
        self._promotions = 0
        self._evictions = 0
        self._bytes_promoted = 0
        self._bytes_evict = 0
        self._time_promo_ms = 0.0

    # -------------------- Memory helpers --------------------
    def _vram_budget(self) -> Tuple[int, int, int]:
        free, total = torch.cuda.mem_get_info()
        target = int(total * self.target_vram_frac)
        return int(free), int(total), target

    def _can_fit(self, bytes_needed: int) -> bool:
        free, _total, target = self._vram_budget()
        # We allow usage up to target; if free < bytes_needed and we're above target, we need to evict
        return free >= bytes_needed

    def _evict_until_can_fit(self, bytes_needed: int) -> None:
        tries = 0
        while not self._can_fit(bytes_needed) and self._lru:
            _k, e = self._lru.popitem(last=False)
            try:
                del e.tensor
                if e.cpu_mirror is not None:
                    del e.cpu_mirror
            except Exception as e:
                logger.debug(f"Failed to delete tensor during eviction: {e}")
            self._evictions += 1
            self._bytes_evict += e.bytes
            tries += 1
            if tries % 4 == 0:
                torch.cuda.empty_cache()
        # Hard cap enforcement
        if self.max_bytes is not None:
            while self._total_bytes_locked() > self.max_bytes and self._lru:
                _k, e = self._lru.popitem(last=False)
                try:
                    del e.tensor
                    if e.cpu_mirror is not None:
                        del e.cpu_mirror
                except Exception as e:
                    logger.debug(f"Failed to delete tensor during hard cap eviction: {e}")
                if len(self._lru) % 4 == 0:
                    torch.cuda.empty_cache()

    def _total_bytes_locked(self) -> int:
        # Heuristic sum of entry sizes; caller maintains sizes
        return sum(e.bytes for e in self._lru.values())

    # -------------------- Public API --------------------
    def touch(self, key: Any) -> None:
        if key in self._lru:
            e = self._lru.pop(key)
            e.last = time.perf_counter()
            self._lru[key] = e

    def get(self, key: Any) -> Optional["torch.Tensor"]:
        e = self._lru.get(key)
        if e is None:
            return None
        self.touch(key)
        self._hits += 1
        return e.tensor

    def put(self, key: Any, tensor: "torch.Tensor", size_bytes: int, cpu_mirror: Optional["torch.Tensor"] = None, priority: int = 0) -> "torch.Tensor":
        # Insert or update entry as MRU
        e = _Entry(key=key, bytes=int(size_bytes), tensor=tensor, cpu_mirror=cpu_mirror, last=time.perf_counter(), priority=int(priority))
        if key in self._lru:
            self._lru.pop(key)
        self._lru[key] = e
        return tensor

    def promote_from_host_view(self, key: Any, host_view: memoryview | bytes | bytearray, size_bytes: int, dtype: "torch.dtype" = torch.uint8, non_blocking: bool = True, priority: int = 0) -> "torch.Tensor":
        """
        Build CPU pinned tensor from host_view and promote to device.
        """
        # Prepare CPU pinned tensor
        try:
            cpu = torch.frombuffer(host_view, dtype=dtype)
        except Exception:
            cpu = torch.tensor(bytearray(host_view), dtype=dtype)
        if getattr(cpu, "is_pinned", False) is False:
            try:
                cpu = cpu.pin_memory()
            except Exception as e:
                logger.debug(f"Failed to pin memory for cpu tensor: {e}")

        # Ensure capacity
        self._evict_until_can_fit(size_bytes)

        # Async H2D on private stream
        t0 = time.perf_counter()
        with torch.cuda.stream(self._stream):
            dev = cpu.to(device=self.device, non_blocking=non_blocking)
        self._stream.synchronize()
        t1 = time.perf_counter()
        self._promotions += 1
        self._bytes_promoted += int(size_bytes)
        self._time_promo_ms += (t1 - t0) * 1000.0

        cpu_mirror = cpu if self.keep_cpu_mirror else None
        return self.put(key, dev, size_bytes=size_bytes, cpu_mirror=cpu_mirror, priority=priority)

    def promote_from_tensor(self, key: Any, cpu_tensor: "torch.Tensor", size_bytes: int, non_blocking: bool = True, priority: int = 0) -> "torch.Tensor":
        if getattr(cpu_tensor, "is_pinned", False) is False:
            try:
                cpu_tensor = cpu_tensor.pin_memory()
            except Exception as e:
                logger.debug(f"Failed to pin memory for cpu_tensor: {e}")
        self._evict_until_can_fit(size_bytes)
        t0 = time.perf_counter()
        with torch.cuda.stream(self._stream):
            dev = cpu_tensor.to(device=self.device, non_blocking=non_blocking)
        self._stream.synchronize()
        t1 = time.perf_counter()
        self._promotions += 1
        self._bytes_promoted += int(size_bytes)
        self._time_promo_ms += (t1 - t0) * 1000.0
        cpu_mirror = cpu_tensor if self.keep_cpu_mirror else None
        return self.put(key, dev, size_bytes=size_bytes, cpu_mirror=cpu_mirror, priority=priority)

    def evict(self, key: Any) -> bool:
        e = self._lru.pop(key, None)
        if e is None:
            return False
        try:
            del e.tensor
            if e.cpu_mirror is not None:
                del e.cpu_mirror
        except Exception as e:
            logger.debug(f"Failed to delete tensor during evict: {e}")
        torch.cuda.empty_cache()
        return True

    def stats(self) -> Dict[str, Any]:
        free, total, target = self._vram_budget()
        return {
            "entries": len(self._lru),
            "bytes": self._total_bytes_locked(),
            "free": free,
            "total": total,
            "target": target,
            "hits": self._hits,
            "promotions": self._promotions,
            "evictions": self._evictions,
            "bytes_promoted": self._bytes_promoted,
            "bytes_evict": self._bytes_evict,
            "time_promo_ms": round(self._time_promo_ms, 3),
        }

