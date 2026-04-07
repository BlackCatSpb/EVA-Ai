
from __future__ import annotations

import time
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from eva_ai.core.batch_wrapper import (
    BatchEnvelope,
    unwrap_for_adapter,
    emit_wrapper_event,
    assert_clean_batch,
)
from eva_ai.core.device_resolver import resolve_device, should_pin_memory
import logging

logger = logging.getLogger(__name__)


@dataclass
class Meta:
    idx: int
    length: int | None = None
    extra: Dict[str, Any] | None = None


@dataclass
class Batch:
    tensors: Dict[str, torch.Tensor]
    metas: List[Meta]


def pad_1d(sequences: List[torch.Tensor], pad_value: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([s.numel() for s in sequences], dtype=torch.long)
    max_len = int(lengths.max() if lengths.numel() else 0)
    if max_len == 0:
        return torch.empty((len(sequences), 0), dtype=torch.long), lengths
    out = sequences[0].new_full((len(sequences), max_len), pad_value)
    for i, s in enumerate(sequences):
        n = s.numel()
        if n:
            out[i, :n] = s
    return out, lengths


def default_collate(items: Sequence[Dict[str, Any]], pad_value: int = 0) -> Tuple[Dict[str, torch.Tensor], List[Meta]]:
    metas: List[Meta] = [it.get("_meta") for it in items]  # type: ignore
    assert all(isinstance(m, Meta) for m in metas), "Each item must include _meta of type Meta"

    batch: Dict[str, torch.Tensor] = {}
    keys = [k for k in items[0].keys() if k != "_meta"]
    for k in keys:
        vals = [it[k] for it in items]
        if isinstance(vals[0], torch.Tensor):
            t0 = vals[0]
            if t0.dim() == 1 and t0.dtype in (torch.long, torch.int64, torch.int32):
                padded, lengths = pad_1d([v.to(dtype=torch.long) for v in vals], pad_value)
                batch[k] = padded
                batch[f"{k}_lengths"] = lengths
            else:
                batch[k] = torch.stack(vals)
        else:
            raise TypeError(f"Unsupported value type for key {k}: {type(vals[0])}")
    return batch, metas  # type: ignore


class TorchBatchAdapter:
    def __init__(
        self,
        max_items: int = 32,
        max_tokens: Optional[int] = None,
        timeout_ms: int = 5,
        key_for_length: str = "input_ids",
        pad_value: int = 0,
        collate_fn: Callable[[Sequence[Dict[str, Any]], int], Tuple[Dict[str, torch.Tensor], List[Meta]]] | None = None,
        pin_memory: bool = False,
        non_blocking: bool = True,
        prefetch_fn: Optional[Callable[[Sequence[Dict[str, Any]]], Dict[str, torch.Tensor]]] = None,
        events: Optional[Any] = None,
        allowed_tensor_keys: Optional[Sequence[str]] = None,
    ) -> None:
        assert max_items > 0
        self.max_items = int(max_items)
        self.max_tokens = int(max_tokens) if max_tokens is not None else None
        self.timeout_ms = int(timeout_ms)
        self.key_for_length = key_for_length
        self.pad_value = pad_value
        self._buf: List[Dict[str, Any]] = []
        self._t0: float | None = None
        self._collate = (lambda items, pad: default_collate(items, pad)) if collate_fn is None else collate_fn
        self._pin_memory = bool(pin_memory)
        self._non_blocking = bool(non_blocking)
        self._prefetch_fn = prefetch_fn
        self._events = events
        self._allowed_keys = set(allowed_tensor_keys) if allowed_tensor_keys is not None else {
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "position_ids",
            "labels",
        }

    def _now(self) -> float:
        return time.perf_counter()

    def push(self, item: Dict[str, Any] | BatchEnvelope) -> None:
        if isinstance(item, BatchEnvelope):
            payload, meta = unwrap_for_adapter(item)
            try:
                emit_wrapper_event("wrapper_removed", meta, events=self._events)
            except Exception as e:
                logger.debug("Failed to emit wrapper_removed event: %s", e)
            item = payload  # type: ignore[assignment]

        if "_meta" not in item:  # type: ignore[operator]
            length = None
            if self.key_for_length in item and isinstance(item[self.key_for_length], torch.Tensor):  # type: ignore[index]
                length = int(item[self.key_for_length].numel())
            item["_meta"] = Meta(idx=len(self._buf), length=length)  # type: ignore[index]
        sanitized: Dict[str, Any] = {k: v for k, v in item.items() if k in self._allowed_keys or k == "_meta"}  # type: ignore[union-attr]
        self._buf.append(sanitized)
        if self._t0 is None:
            self._t0 = self._now()

    def _should_emit(self) -> bool:
        if not self._buf:
            return False
        if len(self._buf) >= self.max_items:
            return True
        if self.max_tokens is not None:
            total = 0
            for it in self._buf:
                t = it.get(self.key_for_length)
                if isinstance(t, torch.Tensor):
                    total += int(t.numel())
            if total >= self.max_tokens:
                return True
        if self._t0 is not None and (self._now() - self._t0) * 1000.0 >= self.timeout_ms:
            return True
        return False

    def try_pop_batch(self) -> Optional[Batch]:
        if not self._should_emit():
            return None
        n_emit = min(len(self._buf), self.max_items)
        if self.max_tokens is not None:
            total = 0
            n_emit = 0
            for it in self._buf:
                t = it.get(self.key_for_length)
                if isinstance(t, torch.Tensor):
                    cand = total + int(t.numel())
                    if cand > self.max_tokens and n_emit > 0:
                        break
                    total = cand
                n_emit += 1
            n_emit = min(n_emit, self.max_items)
            if n_emit == 0:
                n_emit = 1
        items = self._buf[:n_emit]
        self._buf = self._buf[n_emit:]
        self._t0 = self._now() if self._buf else None
        tensors, metas = self._collate(items, self.pad_value)
        if self._prefetch_fn is not None:
            try:
                logger.debug("[TorchBatchAdapter] prefetch_fn call: n_items=%d", len(items))
                extra = self._prefetch_fn(items)
                if isinstance(extra, dict):
                    logger.debug("[TorchBatchAdapter] prefetch_fn returned keys=%s", list(extra.keys()))
                    for k, v in extra.items():
                        if k in tensors:
                            tensors[f"{k}+prefetched"] = v
                        else:
                            tensors[k] = v
            except Exception as e:
                logger.debug("Prefetch function failed: %s", e)
        try:
            assert_clean_batch(tensors)
        except AssertionError:
            drop = {"_wrapper", "envelope_meta", "wrapper_meta"}
            for k in list(tensors.keys()):
                if k in drop:
                    tensors.pop(k, None)
        if self._pin_memory and should_pin_memory(resolve_device()):
            pinned: Dict[str, torch.Tensor] = {}
            for k, v in tensors.items():
                if isinstance(v, torch.Tensor) and v.device.type == "cpu":
                    try:
                        pinned[k] = v.pin_memory()
                    except RuntimeError:
                        pinned[k] = v
                else:
                    pinned[k] = v
            tensors = pinned
        return Batch(tensors=tensors, metas=metas)

    def flush(self) -> Optional[Batch]:
        if not self._buf:
            return None
        items = self._buf
        self._buf = []
        self._t0 = None
        tensors, metas = self._collate(items, self.pad_value)
        if self._prefetch_fn is not None:
            try:
                logger.debug("[TorchBatchAdapter] prefetch_fn call (flush): n_items=%d", len(items))
                extra = self._prefetch_fn(items)
                if isinstance(extra, dict):
                    logger.debug("[TorchBatchAdapter] prefetch_fn returned keys (flush)=%s", list(extra.keys()))
                    for k, v in extra.items():
                        if k in tensors:
                            tensors[f"{k}+prefetched"] = v
                        else:
                            tensors[k] = v
            except Exception as e:
                logger.debug("Prefetch function failed in flush: %s", e)
        if self._pin_memory and should_pin_memory(resolve_device()):
            pinned: Dict[str, torch.Tensor] = {}
            for k, v in tensors.items():
                if isinstance(v, torch.Tensor) and v.device.type == "cpu":
                    try:
                        pinned[k] = v.pin_memory()
                    except RuntimeError:
                        pinned[k] = v
                else:
                    pinned[k] = v
            tensors = pinned
        return Batch(tensors=tensors, metas=metas)
