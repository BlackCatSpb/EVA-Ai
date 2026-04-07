from __future__ import annotations
import time
import logging
import hashlib
from typing import Any, Dict, List, Optional
from collections import defaultdict
import numpy as np
import torch

logger = logging.getLogger("eva_ai.mlearning.fractal_store")


def get_statistics(self) -> Dict[str, Any]:
    """Возвращает статистику хранилища."""
    total_containers = sum(len(v) for v in self.fractal_tree.values())
    total_memory_mb = self.total_memory / (1024 * 1024) if self.total_memory else 0.0
    containers_by_level = {int(level): int(len(ids)) for level, ids in self.fractal_tree.items()}

    return {
        "model_id": self.model_id,
        "total_containers": total_containers,
        "containers_by_level": containers_by_level,
        "total_memory_bytes": int(self.total_memory),
        "total_memory_mb": float(total_memory_mb),
        "compression_ratio": 1.0,
        "levels": sorted(self.fractal_tree.keys()),
    }


def get_similar_tensors(self, vector: torch.Tensor, limit: int = 10) -> List[Dict[str, Any]]:
    """Find similar tensors based on vector similarity"""
    results = []
    if not self.node_vectors:
        return results
    query = vector.flatten()
    for node_id, node_vec in self.node_vectors.items():
        if node_vec.shape != query.shape:
            continue
        sim = torch.nn.functional.cosine_similarity(
            query.flatten(), node_vec.flatten(), dim=0
        ).item()
        results.append({"id": node_id, "similarity": sim})
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:limit]


def _analyze_container_usage(self) -> Dict[str, Any]:
    """Анализирует использование контейнеров."""
    stats: Dict[str, Any] = {
        "access_frequency": defaultdict(int),
        "spatial_locality": 0.0,
        "temporal_locality": 0.0,
        "access_pattern": [],
        "last_access": {},
    }
    now_ts = time.time()

    for container_id, container in self.containers.items():
        stats["access_frequency"][container_id] = int(getattr(container, "access_count", 0))
        stats["last_access"][container_id] = float(getattr(container, "last_accessed", 0.0))

        if float(container.last_accessed or 0.0) > now_ts - 3600.0:
            stats["access_pattern"].append(container_id)

    if len(stats["access_pattern"]) > 1:
        sequential_accesses = 0
        for i in range(1, len(stats["access_pattern"])):
            prev = stats["access_pattern"][i - 1]
            curr = stats["access_pattern"][i]
            if self._are_containers_adjacent(prev, curr):
                sequential_accesses += 1
        stats["spatial_locality"] = sequential_accesses / float(len(stats["access_pattern"]) - 1)

    recent = [cid for cid, ts in stats["last_access"].items() if (now_ts - float(ts)) < 300.0]
    if recent:
        unique_recent = len(set(recent))
        stats["temporal_locality"] = 1.0 - (unique_recent / float(len(recent)))

    return stats


def _are_containers_adjacent(self, cid_a: str, cid_b: str) -> bool:
    """Проверяет соседство контейнеров."""
    a = self.containers.get(cid_a)
    b = self.containers.get(cid_b)
    if a is None or b is None:
        return False

    try:
        if a.level == b.level:
            la = a.metadata.get("layer_name")
            lb = b.metadata.get("layer_name")
            if la == lb:
                pa = a.position[0] if a.position else -10**9
                pb = b.position[0] if b.position else -10**9
                if abs(int(pa) - int(pb)) == 1:
                    return True
            if a.parent and a.parent == b.parent and a.parent is not None:
                return True
    except Exception:
        return False
    return False


def _needs_reconfiguration(self, usage_stats: Dict[str, Any]) -> bool:
    """Определяет, требуется ли реконфигурация фрактальной структуры."""
    try:
        if usage_stats.get("spatial_locality", 0.0) < 0.4:
            logger.debug(f"Низкая пространственная локальность: {usage_stats.get('spatial_locality', 0.0):.2f}")
            return True
        if usage_stats.get("temporal_locality", 0.0) < 0.3:
            logger.debug(f"Низкая временная локальность: {usage_stats.get('temporal_locality', 0.0):.2f}")
            return True
        fragmentation = self._calculate_fragmentation()
        if fragmentation > 0.3:
            logger.debug(f"Высокая фрагментация: {fragmentation:.2f}")
            return True
        if self._usage_patterns_changed(usage_stats):
            logger.debug("Обнаружены изменения в паттернах использования")
            return True
    except Exception:
        return False
    return False


def _calculate_fragmentation(self) -> float:
    """Оценка фрагментации по разрывам в последовательностях позиций."""
    try:
        per_layer: List[float] = []
        layer_groups: Dict[str, List[int]] = defaultdict(list)

        for cid in self.fractal_tree.get(0, []):
            c = self.containers.get(cid)
            if c is None:
                continue
            layer = c.metadata.get("layer_name", "")
            pos0 = int(c.position[0]) if c.position else 0
            layer_groups[layer].append(pos0)

        for layer, positions in layer_groups.items():
            if len(positions) <= 1:
                per_layer.append(0.0)
                continue
            positions.sort()
            gaps = 0
            for i in range(1, len(positions)):
                if positions[i] - positions[i - 1] > 1:
                    gaps += 1
            denom = max(1, len(positions) - 1)
            per_layer.append(gaps / float(denom))

        if not per_layer:
            return 0.0
        return float(sum(per_layer) / float(len(per_layer)))
    except Exception:
        return 0.0


def _usage_patterns_changed(self, usage_stats: Dict[str, Any]) -> bool:
    """Сравнивает текущую сводку использования с предыдущей."""
    try:
        signature = (
            round(float(usage_stats.get("spatial_locality", 0.0)), 2),
            round(float(usage_stats.get("temporal_locality", 0.0)), 2),
            tuple(sorted((cid, int(cnt)) for cid, cnt in usage_stats.get("access_frequency", {}).items())[:50]),
        )
        last = getattr(self, "_last_usage_signature", None)
        self._last_usage_signature = signature

        if last is None:
            return False

        spatial_diff = abs(signature[0] - last[0])
        temporal_diff = abs(signature[1] - last[1])
        return (spatial_diff > 0.25) or (temporal_diff > 0.25)
    except Exception:
        return False


def compute_checksum(self) -> str:
    """Возвращает агрегированный SHA256 всех контейнеров."""
    h = hashlib.sha256()
    try:
        for level in sorted(self.fractal_tree.keys()):
            for cid in sorted(self.fractal_tree[level]):
                cont = self.containers.get(cid)
                if cont is None:
                    continue
                h.update(cid.encode("utf-8", errors="ignore"))
                try:
                    h.update(cont.data.tobytes())
                except Exception:
                    h.update(np.array(cont.data).tobytes())
        return h.hexdigest()
    except Exception:
        return ""
