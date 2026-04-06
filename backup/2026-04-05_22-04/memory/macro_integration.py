from __future__ import annotations

import os
import threading
from dataclasses import dataclass
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Callable, Any
import logging

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from .macro_archive import MacroArchive
from .hotset import HotSetManager
from .paged_store import (
    BatchAdapter,
    MacroBlockSpec,
    HierarchicalIndex,
)

logger = logging.getLogger(__name__)


@dataclass
class MacroPrefetchConfig:
    """Конфигурация для макро-префетчера."""
    root: Path
    window_tokens: int = 512
    max_layers_per_batch: int = 2
    tensor_kind: str = "kv"
    device: Optional[str] = None  # e.g., "cuda:0" to move immediately; None keeps on CPU
    use_pinned: bool = True
    # HotSet settings
    hotset_enabled: bool = False
    hotset_target_vram_frac: float = 0.65
    # Lazy emission settings
    lazy_enabled: bool = False
    io_budget_bytes_per_sec: int = 128 * 1024 * 1024  # 128MB/s budget
    max_pending: int = 8
    ram_cache_bytes: int = 256 * 1024 * 1024  # 256MB CPU cache for ready tensors


class MacroblockPrefetcher:
    """
    Собирает макроблоки для недавних окон токенов на слой, используя иерархический индекс.

    Produces a dict of tensors suitable to merge into TorchBatchAdapter's batch tensors.
    Keys follow: "kv_macroblock_l{layer}" mapping to torch.uint8 tensors (CPU or device).
    """

    def __init__(self, cfg: MacroPrefetchConfig) -> None:
        self.cfg = cfg
        self.archive = MacroArchive(cfg.root)
        logger.debug(
            "[MacroPrefetcher] init: root=%s, lazy=%s, hotset=%s, device=%s",
            cfg.root, cfg.lazy_enabled, cfg.hotset_enabled, cfg.device
        )
        # Use MacroArchive's minimal absolute-read store with BatchAdapter
        # BatchAdapter will fall back to absolute reads (block_id = -1) and skip block prefetch.
        store = self.archive.open_store()
        # type: ignore[arg-type]
        self.adapter = BatchAdapter(store, prefetch_depth=0, use_pinned=cfg.use_pinned)  # type: ignore
        self.hidx: HierarchicalIndex = self.archive.hindex
        self._lock = threading.RLock()
        self._hotset: Optional[HotSetManager] = None
        
        if self.cfg.hotset_enabled and torch is not None and torch.cuda.is_available():
            # If device is None, default to cuda:0
            dev = self.cfg.device or "cuda:0"
            try:
                self._hotset = HotSetManager(device=dev, target_vram_frac=self.cfg.hotset_target_vram_frac)
            except Exception as e:
                logger.warning(f"Failed to initialize HotSetManager: {e}")
                self._hotset = None
        
        # Lazy emission structures
        self._lazy_enabled = bool(self.cfg.lazy_enabled)
        self._ready: Dict[Tuple, Any] = {}
        self._ready_sizes: Dict[Tuple, int] = {}
        self._lru_keys: List[Tuple] = []
        self._ready_bytes: int = 0
        self._pending: set[Tuple] = set()
        self._queue: List[Tuple] = []
        self._last_bucket_ts = time.perf_counter()
        self._bucket_bytes = 0
        self._worker: Optional[threading.Thread] = None
        
        if self._lazy_enabled:
            self._start_worker()
            logger.debug("[MacroPrefetcher] lazy worker started")

    def _start_worker(self) -> None:
        """Запускает фоновый рабочий поток."""
        if self._worker is not None:
            return
        self._worker = threading.Thread(
            target=self._worker_loop, 
            name="MacroPrefetcherWorker", 
            daemon=True
        )
        self._worker.start()

    def _admit(self, key: Tuple) -> bool:
        """Проверяет, можно ли добавить ключ в очередь."""
        return len(self._pending) < int(self.cfg.max_pending)

    def _enqueue(self, key: Tuple) -> None:
        """Добавляет ключ в очередь."""
        if key in self._pending or key in self._ready:
            return
        if not self._admit(key):
            return
        self._pending.add(key)
        self._queue.append(key)

    def _evict_ready_until(self, need_bytes: int) -> None:
        """Вытесняет готовые элементы до освобождения нужного количества байт."""
        while self._ready_bytes + need_bytes > int(self.cfg.ram_cache_bytes) and self._lru_keys:
            k = self._lru_keys.pop(0)
            if k in self._ready:
                self._ready.pop(k, None)
                sz = self._ready_sizes.pop(k, 0)
                self._ready_bytes -= sz

    def _touch_ready(self, key: Tuple) -> None:
        """Обновляет позицию ключа в LRU списке."""
        if key in self._lru_keys:
            self._lru_keys.remove(key)
        self._lru_keys.append(key)

    def _io_token_bucket(self, bytes_amount: int) -> None:
        """Simple token bucket to limit IO throughput."""
        now = time.perf_counter()
        elapsed = max(0.0, now - self._last_bucket_ts)
        refill = elapsed * int(self.cfg.io_budget_bytes_per_sec)
        self._bucket_bytes = max(0, self._bucket_bytes - int(refill))
        self._last_bucket_ts = now
        self._bucket_bytes += bytes_amount
        over = self._bucket_bytes - int(self.cfg.io_budget_bytes_per_sec)
        if over > 0:
            # Sleep proportional to overflow
            time.sleep(over / max(1, int(self.cfg.io_budget_bytes_per_sec)))

    def _worker_loop(self) -> None:
        """Основной цикл фонового рабочего потока."""
        while True:
            key = None
            with self._lock:
                if self._queue:
                    key = self._queue.pop(0)
            
            if key is None:
                time.sleep(0.001)
                continue
            
            try:
                # Decode key
                tensor_kind, lid, head_range, token_range, dtype, shape = key
                spec = MacroBlockSpec(
                    tensor_kind=tensor_kind,
                    layer_id=lid,
                    head_range=head_range,
                    token_range=token_range,
                    dtype=dtype,
                    shape=shape,
                )
                # IO budget applies to reads; approximate by planned host bytes (window size)
                # Assemble on CPU (device=None), then optionally promote via hotset
                mb = self.adapter.assemble_macroblock(self.hidx, spec, device=None)
                host_view = mb.host_view
                
                if host_view is None:
                    continue
                
                size_bytes = len(host_view)
                self._io_token_bucket(size_bytes)
                
                tensor = None
                if self._hotset is not None:
                    try:
                        tensor = self._hotset.promote_from_host_view(
                            key=key,
                            host_view=host_view,
                            size_bytes=size_bytes,
                            dtype=torch.uint8,
                            non_blocking=True,
                            priority=0,
                        )
                    except Exception as e:
                        logger.debug(f"Failed to promote tensor via hotset: {e}")
                        tensor = None
                
                if tensor is None:
                    # Keep CPU tensor
                    try:
                        tensor = torch.frombuffer(host_view, dtype=torch.uint8)  # type: ignore[arg-type]
                    except Exception as e:
                        logger.debug(f"Failed to create tensor from buffer: {e}")
                        tensor = torch.tensor(bytearray(host_view), dtype=torch.uint8)
                
                with self._lock:
                    self._evict_ready_until(size_bytes)
                    self._ready[key] = tensor
                    self._ready_sizes[key] = size_bytes
                    self._ready_bytes += size_bytes
                    self._touch_ready(key)
            finally:
                with self._lock:
                    self._pending.discard(key)

    def _select_layers(self) -> List[int]:
        """Выбирает слои для обработки."""
        # Collect layers available in the index
        layers: List[int] = sorted({
            sb.layer_id for sb in self.hidx.subblocks 
            if sb.tensor_kind == self.cfg.tensor_kind
        })
        # Limit per batch to avoid long stalls
        return layers[: max(1, int(self.cfg.max_layers_per_batch))]

    def _layer_ranges(self, layer_id: int) -> Tuple[Tuple[int, int], Tuple[int, int], str, Tuple[int, ...]]:
        """Определяет диапазоны для слоя."""
        # Aggregate head and token ranges, dtype and a representative shape for the layer
        heads_min, heads_max = 1 << 30, -1
        tok_min, tok_max = 1 << 30, -1
        dtype = None
        shape: Tuple[int, ...] = ()
        
        for sb in self.hidx.subblocks:
            if sb.tensor_kind != self.cfg.tensor_kind or sb.layer_id != layer_id:
                continue
            h0, h1 = sb.head_range
            t0, t1 = sb.token_range
            heads_min = min(heads_min, h0)
            heads_max = max(heads_max, h1)
            tok_min = min(tok_min, t0)
            tok_max = max(tok_max, t1)
            if dtype is None:
                # Ensure hashable, consistent dtype representation
                try:
                    dtype = str(sb.dtype)
                except Exception:
                    dtype = f"{sb.dtype}"
            if not shape:
                # Ensure shape is a tuple for hashing
                try:
                    shape = tuple(sb.shape)
                except Exception:
                    shape = tuple(list(sb.shape))
        
        if heads_max <= heads_min or tok_max <= tok_min:
            raise RuntimeError(f"No valid ranges for layer {layer_id}")
        
        # Choose last window of tokens
        W = max(1, int(self.cfg.window_tokens))
        t1 = tok_max
        t0 = max(tok_min, t1 - W)
        return (heads_min, heads_max), (t0, t1), (dtype or "fp16"), shape

    def build(self, items: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        """Собирает макроблоки для передачи в батч."""
        # items currently unused; could be used to adapt window or choose layers
        if torch is None:
            return {}
        
        out: Dict[str, torch.Tensor] = {}
        with self._lock:
            layers = self._select_layers()
            logger.debug(
                "[MacroPrefetcher] build: layers=%s lazy=%s items=%d",
                layers, self._lazy_enabled, len(items)
            )
            
            for lid in layers:
                head_range, token_range, dtype, shape = self._layer_ranges(lid)
                spec = MacroBlockSpec(
                    tensor_kind=self.cfg.tensor_kind,
                    layer_id=lid,
                    head_range=head_range,
                    token_range=token_range,
                    dtype=dtype,
                    shape=shape,
                )
                cache_key = (
                    self.cfg.tensor_kind, 
                    lid, 
                    spec.head_range, 
                    spec.token_range, 
                    spec.dtype, 
                    spec.shape
                )
                tensor_key = f"{self.cfg.tensor_kind}_macroblock_l{lid}"

                # Lazy path: try ready cache, otherwise enqueue and skip
                if self._lazy_enabled:
                    if cache_key in self._ready:
                        t = self._ready[cache_key]
                        self._touch_ready(cache_key)
                        out[tensor_key] = t
                        continue
                    self._enqueue(cache_key)
                    # If not ready, we skip emitting this layer for this batch
                    continue

                # Non-lazy path: assemble immediately
                mb = self.adapter.assemble_macroblock(
                    self.hidx, 
                    spec, 
                    device=None if self._hotset is not None else self.cfg.device
                )
                
                if self._hotset is not None and mb.host_view is not None:
                    size_bytes = len(mb.host_view)
                    try:
                        dev_t = self._hotset.promote_from_host_view(
                            key=cache_key,
                            host_view=mb.host_view,
                            size_bytes=size_bytes,
                            dtype=torch.uint8,
                            non_blocking=True,
                            priority=0,
                        )
                        out[tensor_key] = dev_t
                        continue
                    except Exception as e:
                        logger.debug(f"Failed to create tensor via hotset: {e}")
                
                # Fallbacks: use device tensor produced by assemble_macroblock or build CPU tensor
                if mb.device_tensor is not None:
                    out[tensor_key] = mb.device_tensor
                else:
                    try:
                        t = torch.frombuffer(mb.host_view, dtype=torch.uint8)  # type: ignore[arg-type]
                    except Exception as e:
                        logger.debug(f"Failed to create tensor from buffer: {e}")
                        t = torch.tensor(bytearray(mb.host_view), dtype=torch.uint8)
                    out[tensor_key] = t
        
        return out

    def stats(self) -> Dict[str, Any]:
        """Возвращает статистику префетчера."""
        with self._lock:
            hs = self._hotset.stats() if self._hotset is not None else None
            s = {
                "lazy_enabled": self._lazy_enabled,
                "ready_entries": len(self._ready),
                "ready_bytes": self._ready_bytes,
                "pending": len(self._pending),
                "queue": len(self._queue),
                "io_budget_bps": int(self.cfg.io_budget_bytes_per_sec),
                "hotset": hs,
            }
            logger.debug("[MacroPrefetcher] stats: %s", s)
            return s


def make_prefetch_fn(
    root: str | Path,
    window_tokens: int = 512,
    max_layers_per_batch: int = 2,
    tensor_kind: str = "kv",
    device: Optional[str] = None,
    use_pinned: bool = True,
    hotset_enabled: bool = False,
    hotset_target_vram_frac: float = 0.65,
    lazy_enabled: bool = True,
    io_budget_bytes_per_sec: int = 96 * 1024 * 1024,
    max_pending: int = 6,
    ram_cache_bytes: int = 256 * 1024 * 1024,
) -> Callable[[Sequence[Dict[str, Any]]], Dict[str, "torch.Tensor"]]:
    """Создаёт функцию префетча с заданной конфигурацией."""
    cfg = MacroPrefetchConfig(
        root=Path(root),
        window_tokens=window_tokens,
        max_layers_per_batch=max_layers_per_batch,
        tensor_kind=tensor_kind,
        device=device,
        use_pinned=use_pinned,
        hotset_enabled=hotset_enabled,
        hotset_target_vram_frac=hotset_target_vram_frac,
        lazy_enabled=lazy_enabled,
        io_budget_bytes_per_sec=io_budget_bytes_per_sec,
        max_pending=max_pending,
        ram_cache_bytes=ram_cache_bytes,
    )
    prefetcher = MacroblockPrefetcher(cfg)

    # Return a wrapper object to ensure stats() is always accessible
    # (bound methods don't support setattr reliably)
    class _PrefetchFn:
        def __init__(self, p: MacroblockPrefetcher) -> None:
            self._p = p

        def __call__(self, items: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
            return self._p.build(items)

        def stats(self) -> Dict[str, Any]:
            return self._p.stats()

    fn = _PrefetchFn(prefetcher)
    logger.debug("[MacroPrefetcher] wrapper with stats() created")
    return fn