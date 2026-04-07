# -*- coding: utf-8 -*-
"""
KnowledgeHybridIndex: гибридный индекс для графа знаний (узлы/связи)
- In-memory LRU для "горячих" элементов
- DiskCache для долговременного хранения/оффлоада

Использование:
    idx = KnowledgeHybridIndex(base_cache_dir, namespace="kg_index", max_items_mem=50000)
    idx.put_node(node) / idx.get_node(id)
    idx.put_edge(edge) / idx.get_edge(id)

Зависимости: eva.memory.disk_cache.DiskCache
"""
from __future__ import annotations
import os
import sys
import json
import logging
from collections import OrderedDict
from typing import Optional, Any

_eva_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _eva_root not in sys.path:
    sys.path.insert(0, _eva_root)

logger = logging.getLogger(__name__)

try:
    from eva.memory.disk_cache import DiskCache  # type: ignore
except Exception:
    # Локальный минимальный фоллбэк, если путь импорта отличается
    from eva.memory.disk_cache import DiskCache  # type: ignore


class _LRU:
    def __init__(self, max_items: int):
        self.max = max(1, int(max_items))
        self.map = OrderedDict()

    def get(self, k: str) -> Optional[Any]:
        if k in self.map:
            self.map.move_to_end(k)
            return self.map[k]
        return None

    def put(self, k: str, v: Any):
        if k in self.map:
            self.map.move_to_end(k)
            self.map[k] = v
            return
        if len(self.map) >= self.max:
            self.map.popitem(last=False)
        self.map[k] = v

    def remove(self, k: str):
        if k in self.map:
            del self.map[k]


class KnowledgeHybridIndex:
    def __init__(self, base_cache_dir: str, namespace: str = "kg_index", max_items_mem: int = 50000,
                 write_mb_s: float = 40.0, read_mb_s: float = 200.0, burst_factor: float = 2.0):
        self.dir = os.path.join(base_cache_dir, namespace)
        os.makedirs(self.dir, exist_ok=True)
        self.disk = DiskCache(os.path.join(self.dir, "disk_storage"), max_size_gb=50.0,
                              write_mb_s=write_mb_s, read_mb_s=read_mb_s, burst_factor=burst_factor,
                              resource_queue=None)
        self.mem_nodes = _LRU(max_items_mem)
        self.mem_edges = _LRU(max_items_mem)

    # ---- Node API ----
    def put_node(self, node) -> None:
        try:
            key = f"node:{node.id}"
            payload = node.to_dict() if hasattr(node, 'to_dict') else node
            self.mem_nodes.put(node.id, payload)
            self.disk.put(key, payload)
        except Exception as e:
            logger.error(f"Failed to put node {node.id}: {e}")

    def get_node(self, node_id: str) -> Optional[dict]:
        obj = self.mem_nodes.get(node_id)
        if obj is not None:
            return obj
        key = f"node:{node_id}"
        return self.disk.get(key)

    def remove_node(self, node_id: str) -> None:
        self.mem_nodes.remove(node_id)
        try:
            self.disk.delete(f"node:{node_id}")
        except Exception as e:
            logger.error(f"Failed to remove node {node_id}: {e}")

    # ---- Edge API ----
    def put_edge(self, edge) -> None:
        try:
            key = f"edge:{edge.id}"
            payload = edge.to_dict() if hasattr(edge, 'to_dict') else edge
            self.mem_edges.put(edge.id, payload)
            self.disk.put(key, payload)
        except Exception as e:
            logger.error(f"Failed to put edge {edge.id}: {e}")

    def get_edge(self, edge_id: str) -> Optional[dict]:
        obj = self.mem_edges.get(edge_id)
        if obj is not None:
            return obj
        key = f"edge:{edge_id}"
        return self.disk.get(key)

    def remove_edge(self, edge_id: str) -> None:
        self.mem_edges.remove(edge_id)
        try:
            self.disk.delete(f"edge:{edge_id}")
        except Exception as e:
            logger.error(f"Failed to remove edge {edge_id}: {e}")
