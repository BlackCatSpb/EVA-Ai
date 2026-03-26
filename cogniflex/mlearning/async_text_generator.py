"""
Async text generation adapter for CogniFlex.
Provides manual decode loop with async streaming and lightweight caching.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
import os
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

logger = logging.getLogger("cogniflex.async_text_generator")

import torch
from cogniflex.memory.disk_cache import DiskCache
try:
    # Available in newer Transformers (>=4.52)
    from transformers.cache_utils import DynamicCache as _TF_DynamicCache  # type: ignore
except Exception:  # pragma: no cover
    _TF_DynamicCache = None


@dataclass
class SamplingConfig:
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 0
    repetition_penalty: float = 1.0
    stream_interval: int = 1
    stop_sequences: List[str] = field(default_factory=list)


class SimpleKVCache:
    """In-memory KV cache with naive size control."""
    def __init__(self, max_items: int = 8):
        self.max_items = max_items
        self.store: Dict[str, Tuple[Any, float]] = {}

    def get(self, key: str):
        v = self.store.get(key)
        if not v:
            return None
        val, _ = v
        # update access time
        self.store[key] = (val, time.time())
        return val

    def put(self, key: str, value: Any):
        if len(self.store) >= self.max_items:
            # evict oldest
            oldest_key = min(self.store.items(), key=lambda x: x[1][1])[0]
            self.store.pop(oldest_key, None)
        self.store[key] = (value, time.time())


def _hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _sample_token(logits: torch.Tensor, cfg: SamplingConfig) -> int:
    # logits: [vocab]
    if cfg.temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / max(1e-6, cfg.temperature)
    probs = torch.softmax(logits, dim=-1)

    if cfg.top_k and cfg.top_k > 0:
        topk = torch.topk(probs, cfg.top_k)
        indices = topk.indices
        p = topk.values / torch.sum(topk.values)
        choice = torch.multinomial(p, 1)
        return int(indices[choice].item())

    if 0.0 < cfg.top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        cutoff = torch.searchsorted(cumulative, torch.tensor(cfg.top_p, device=probs.device))
        sorted_probs = sorted_probs[: cutoff + 1]
        sorted_probs = sorted_probs / torch.sum(sorted_probs)
        choice = torch.multinomial(sorted_probs, 1)
        return int(sorted_indices[choice].item())

    choice = torch.multinomial(probs, 1)
    return int(choice.item())


class AsyncTextGenerator:
    def __init__(self, device: str = "cpu"):
        self.device = device
        # Более узкое горячее окно KV по умолчанию
        self.kv_cache = SimpleKVCache(max_items=8)
        # prompt token cache (ids) separate from kv cache
        self.prompt_cache: Dict[str, Tuple[Any, Any]] = {}
        # Optional disk caches (lazy-init per options)
        self._disk_prompt_cache: Optional[DiskCache] = None
        self._disk_kv_meta_cache: Optional[DiskCache] = None

    async def generate_stream(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        text_processor: Optional[Any] = None,
        sampling: Optional[Dict[str, Any]] = None,
        cache_opts: Optional[Dict[str, Any]] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[str]:
        """
        Asynchronously generate tokens as text chunks.
        """
        cfg = SamplingConfig(**(sampling or {}))
        cache_opts = cache_opts or {}
        enable_prompt_cache = bool(cache_opts.get("enable_prompt_cache", True))
        enable_kv_cache = bool(cache_opts.get("enable_kv_cache", True))
        enable_disk_prompt_cache = bool(cache_opts.get("enable_disk_prompt_cache", True))
        disk_cache_dir = cache_opts.get("disk_cache_dir", os.path.join("token_cache", "disk_storage"))
        # Позволяем управлять размером горячего KV-окна на лету
        try:
            kv_max_items = int(cache_opts.get("kv_max_items", self.kv_cache.max_items))
            self.kv_cache.max_items = max(2, kv_max_items)
        except Exception as e:
            logger.warning(f"Failed to set KV cache max_items: {e}")

        # Lazy init disk caches if enabled
        if enable_prompt_cache and enable_disk_prompt_cache and self._disk_prompt_cache is None:
            try:
                self._disk_prompt_cache = DiskCache(
                    disk_cache_dir,
                    max_size_gb=float(cache_opts.get("disk_max_gb", 20.0)),
                )
            except Exception as e:
                logger.debug(f"Disk prompt cache init failed: {e}")
                self._disk_prompt_cache = None
        if self._disk_kv_meta_cache is None:
            try:
                self._disk_kv_meta_cache = DiskCache(os.path.join(disk_cache_dir, "kv_meta"), max_size_gb=1.0)
            except Exception as e:
                logger.debug(f"Disk KV meta cache init failed: {e}")
                self._disk_kv_meta_cache = None

        # Build cache keys
        prompt_key = _hash_text(prompt)
        model_key = getattr(model, "name_or_path", None) or "custom_model"
        tokenizer_key = getattr(tokenizer, "name_or_path", None) or "custom_tokenizer"
        kv_key = f"kv::{model_key}::{tokenizer_key}::{prompt_key}::{cfg.max_new_tokens}::{cfg.temperature}::{cfg.top_p}::{cfg.top_k}::{cfg.repetition_penalty}"
        pt_key = f"pt::{model_key}::{tokenizer_key}::{prompt_key}"
        # Filesystem-safe variants for disk caches
        def _safe_key(k: str) -> str:
            try:
                return "k_" + hashlib.sha1(k.encode("utf-8")).hexdigest()
            except Exception:
                return "k_" + _hash_text(k)[:40]
        pt_disk_key = _safe_key(pt_key)
        kv_disk_key = _safe_key(kv_key)

        # Tokenize (with cache)
        input_ids = None
        attn_mask = None
        # 1) in-memory prompt cache
        if enable_prompt_cache and pt_key in self.prompt_cache:
            input_ids, attn_mask = self.prompt_cache[pt_key]
        # 2) disk prompt cache
        elif enable_prompt_cache and enable_disk_prompt_cache and self._disk_prompt_cache is not None:
            disk_item = self._disk_prompt_cache.get(pt_disk_key)
            if isinstance(disk_item, dict) and "input_ids" in disk_item:
                try:
                    input_ids = disk_item["input_ids"].to(self.device)
                    attn_mask = disk_item.get("attention_mask")
                    if attn_mask is not None:
                        attn_mask = attn_mask.to(self.device)
                    # Warm up in-memory cache too
                    self.prompt_cache[pt_key] = (input_ids, attn_mask)
                except Exception:
                    input_ids = None
                    attn_mask = None
        # 3) tokenize fresh
        if input_ids is None:
            if text_processor and hasattr(text_processor, "encode"):
                toks = text_processor.encode(prompt)
                input_ids = torch.tensor([toks.ids], device=self.device)
                attn_mask = torch.tensor([toks.attention_mask], device=self.device) if hasattr(toks, "attention_mask") else None
            else:
                enc = tokenizer(prompt, return_tensors="pt")
                input_ids = enc["input_ids"].to(self.device)
                attn_mask = enc.get("attention_mask")
                if attn_mask is not None:
                    attn_mask = attn_mask.to(self.device)
            # store to caches
            if enable_prompt_cache:
                self.prompt_cache[pt_key] = (input_ids, attn_mask)
                if enable_disk_prompt_cache and self._disk_prompt_cache is not None:
                    try:
                        self._disk_prompt_cache.put(pt_disk_key, {
                            "input_ids": input_ids.detach().cpu(),
                            "attention_mask": attn_mask.detach().cpu() if attn_mask is not None else None,
                            "created_at": time.time(),
                        })
                    except Exception as e:
                        logger.warning(f"Disk prompt cache put failed: {e}")

        eos_token_id = tokenizer.eos_token_id if getattr(tokenizer, "eos_token_id", None) is not None else None

        # Prefill
        model.eval()
        past_key_values = None
        async with _maybe_async():
            with torch.inference_mode():
                outputs = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=True)
                past_key_values = getattr(outputs, "past_key_values", None)
                # Convert legacy tuple to DynamicCache if available
                if _TF_DynamicCache is not None and past_key_values is not None and isinstance(past_key_values, (list, tuple)):
                    try:
                        past_key_values = _TF_DynamicCache.from_legacy_cache(past_key_values)
                    except Exception as e:
                        logger.warning(f"Failed to convert legacy cache to DynamicCache: {e}")

        # Try reuse KV cache (after prefill we store it under kv_key)
        if enable_kv_cache:
            cached = self.kv_cache.get(kv_key)
            if cached is not None:
                past_key_values = cached
            else:
                if past_key_values is not None:
                    self.kv_cache.put(kv_key, past_key_values)
                    # Persist lightweight metadata to disk for observability
                    if self._disk_kv_meta_cache is not None:
                        try:
                            self._disk_kv_meta_cache.put(kv_disk_key, {
                                "prefill_cached": True,
                                "ts": time.time(),
                                "model": model_key,
                                "tok": tokenizer_key,
                                "prompt_hash": prompt_key,
                            })
                        except Exception as e:
                            logger.warning(f"Disk KV meta cache put failed: {e}")

        # Decode loop
        generated_ids: List[int] = []
        last_token = input_ids[:, -1:]
        step = 0
        text_buf = []

        while step < cfg.max_new_tokens:
            if cancel_event and cancel_event.is_set():
                break

            async with _maybe_async():
                with torch.inference_mode():
                    # past_key_values may be a DynamicCache or legacy tuple
                    outputs = model(input_ids=last_token, use_cache=True, past_key_values=past_key_values)
                    logits = outputs.logits[:, -1, :].squeeze(0)
                    pkv_new = getattr(outputs, "past_key_values", None)
                    if _TF_DynamicCache is not None and pkv_new is not None and isinstance(pkv_new, (list, tuple)):
                        try:
                            pkv_new = _TF_DynamicCache.from_legacy_cache(pkv_new)
                        except Exception as e:
                            logger.warning(f"Failed to convert legacy cache to DynamicCache in decode: {e}")
                    past_key_values = pkv_new

            next_id = _sample_token(logits, cfg)
            generated_ids.append(next_id)

            if eos_token_id is not None and next_id == eos_token_id:
                break

            # Detokenize chunk by chunk
            if (step + 1) % cfg.stream_interval == 0:
                chunk_text = self._decode_chunk(generated_ids, tokenizer, text_processor)
                if chunk_text:
                    text_buf.append(chunk_text)
                    yield chunk_text

            # Prepare next
            last_token = torch.tensor([[next_id]], device=self.device)
            step += 1

            # Stop sequences check (simple suffix check)
            if cfg.stop_sequences and text_buf:
                joined = "".join(text_buf)
                if any(joined.endswith(s) for s in cfg.stop_sequences):
                    break

        # Flush residual
        residual = self._decode_chunk(generated_ids, tokenizer, text_processor)
        if residual:
            yield residual

    @staticmethod
    def _decode_chunk(ids: List[int], tokenizer: Any, text_processor: Optional[Any]) -> str:
        if not ids:
            return ""
        try:
            if text_processor and hasattr(text_processor, "decode"):
                text = text_processor.decode(ids)
            else:
                text = tokenizer.decode(ids, skip_special_tokens=True)
            # clear ids after decoding to avoid re-decoding the same tokens
            del ids[:]
            return text
        except Exception:
            del ids[:]
            return ""


class _maybe_async:
    """Context manager placeholder for potential async offloading hooks."""
    async def __aenter__(self):
        return None

    async def __aexit__(self, exc_type, exc, tb):
        return False
