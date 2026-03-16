"""
CacheRouter: addressable routing over HybridTokenCache using CacheIndex.
- Stores and resolves batches/segments/nodes with weights.
- Provides rate-limited access via underlying DiskCache already configured in HybridTokenCache.
"""
from __future__ import annotations
import os
import hashlib
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import logging

from .cache_index import CacheIndex

logger = logging.getLogger(__name__)

class CacheRouter:
    def __init__(self, brain):
        self.brain = brain
        # Expect brain.token_cache to be initialized
        self.token_cache = getattr(brain, 'token_cache', None)
        if self.token_cache is None:
            logger.warning("CacheRouter: token_cache is None — routing limited")
        self.index = CacheIndex(brain)

    # ---- Addressable identifiers ----
    @staticmethod
    def make_segment_id(batch_id: str, offset: int, length: int) -> str:
        base = f"{batch_id}:{offset}:{length}"
        return hashlib.sha1(base.encode('utf-8')).hexdigest()

    @staticmethod
    def make_node_id(segment_id: str, token_start: int, token_end: int) -> str:
        base = f"{segment_id}:{token_start}:{token_end}"
        return hashlib.sha1(base.encode('utf-8')).hexdigest()

    # ---- Public API ----
    def register_batch(self, batch_id: str, source: str, total_tokens: int, priority: float = 0.0, status: str = "queued") -> None:
        self.index.upsert_batch(batch_id, source, total_tokens, priority, status)

    def register_segment(self, batch_id: str, offset: int, length: int, token_count: int, disk_path: str,
                         checksum: str = "") -> str:
        seg_id = self.make_segment_id(batch_id, offset, length)
        self.index.upsert_segment(seg_id, batch_id, offset, length, token_count, disk_path, checksum)
        return seg_id

    def register_token_nodes(self, segment_id: str, spans: Sequence[Tuple[int, int]]) -> List[str]:
        nodes = []
        for (ts, te) in spans:
            nid = self.make_node_id(segment_id, ts, te)
            nodes.append((nid, segment_id, int(ts), int(te), hashlib.sha1(f"{segment_id}:{ts}:{te}".encode()).hexdigest()))
        self.index.add_token_nodes(nodes)
        return [n[0] for n in nodes]

    def link_nodes_to_kg(self, node_to_kg: Sequence[Tuple[str, str]]):
        self.index.link_nodes_to_kg(node_to_kg)

    def set_weight(self, item_id: str, item_type: str, weight_type: str, value: float, context_id: Optional[str] = None):
        self.index.set_weight(item_id, item_type, weight_type, value, context_id)

    def rank_segments(self, top_k: int = 10, context_id: Optional[str] = None, weight_type: str = "task_relevance") -> List[str]:
        return self.index.rank_segments(top_k=top_k, context_id=context_id, weight_type=weight_type)

    def get_segment_bytes(self, segment_id: str) -> Optional[bytes]:
        # Fetch via token_cache's DiskCache path stored in index
        path = self.index.get_segment_path(segment_id)
        if not path or not os.path.exists(path):
            return None
        try:
            # Системный троттлинг IO через GlobalResourceQueue при прямом чтении
            chunk = 1024 * 1024  # 1MB
            out = bytearray()
            with open(path, 'rb') as f:
                while True:
                    # Запросить IO на следующий кусок
                    try:
                        if hasattr(self.brain, 'request_io'):
                            self.brain.request_io(chunk)
                        elif hasattr(self.brain, 'resource_queue') and self.brain.resource_queue is not None:
                            self.brain.resource_queue.acquire_io(chunk)
                    except Exception:
                        pass
                    buf = f.read(chunk)
                    if not buf:
                        break
                    out.extend(buf)
            return bytes(out)
        except Exception as e:
            logger.error(f"CacheRouter.get_segment_bytes: {e}")
            return None

    def address_of(self, segment_id: str, token_start: Optional[int] = None, token_end: Optional[int] = None) -> str:
        return self.index.address_of(segment_id, token_start, token_end)

    def close(self):
        try:
            self.index.close()
        except Exception:
            pass
