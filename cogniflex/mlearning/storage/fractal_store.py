from __future__ import annotations
import time
import logging
import os
import json
from pathlib import Path
import hashlib
import gc
import math
import sys
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, Deque, Tuple, List, Optional, Iterable, Set
from collections import deque, OrderedDict, defaultdict
import numpy as np
import torch
import torch.nn as nn

try:
    import psutil
except Exception:
    psutil = None

try:
    from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
except Exception:
    AutoModelForCausalLM = None
    AutoModel = None
    AutoTokenizer = None

logger = logging.getLogger("cogniflex.mlearning.fractal_store")


# ----------------------- Прокси для графа знаний -----------------------
class NodeProxy:
    def __init__(self, node: Any) -> None:
        self.id = getattr(node, "id", None)
        if self.id is None and isinstance(node, dict):
            self.id = node.get("id")
        self.node_type = getattr(node, "node_type", None)
        if self.node_type is None and isinstance(node, dict):
            self.node_type = node.get("node_type", "unknown")
        self.content = getattr(node, "content", None)
        if self.content is None and isinstance(node, dict):
            self.content = node.get("content")


class EdgeProxy:
    def __init__(self, edge: Any) -> None:
        self.source = getattr(edge, "source", None)
        if self.source is None and isinstance(edge, dict):
            self.source = edge.get("source")
        self.target = getattr(edge, "target", None)
        if self.target is None and isinstance(edge, dict):
            self.target = edge.get("target")
        self.relation_type = getattr(edge, "relation_type", None)
        if self.relation_type is None and isinstance(edge, dict):
            self.relation_type = edge.get("relation_type", "rel")


class KnowledgeGraphProxy:
    """Лёгкий адаптер графа знаний."""
    
    def __init__(self, kg: Any) -> None:
        self._kg = kg

    def get_all_nodes(self) -> List[NodeProxy]:
        try:
            if hasattr(self._kg, "get_all_nodes"):
                nodes = self._kg.get_all_nodes()
            elif isinstance(self._kg, dict):
                nodes = self._kg.get("nodes", [])
            else:
                nodes = []
        except (AttributeError, TypeError, KeyError) as e:
            logger.warning(f"Failed to retrieve nodes from knowledge graph: {e}")
            nodes = []
        return [NodeProxy(n) for n in nodes]

    def get_all_edges(self) -> List[EdgeProxy]:
        try:
            if hasattr(self._kg, "get_all_edges"):
                edges = self._kg.get_all_edges()
            elif isinstance(self._kg, dict):
                edges = self._kg.get("edges", [])
            else:
                edges = []
        except (AttributeError, TypeError, KeyError) as e:
            logger.warning(f"Failed to retrieve edges from knowledge graph: {e}")
            edges = []
        return [EdgeProxy(e) for e in edges]


# ----------------------- Контейнер фрактальных данных -----------------------
@dataclass
class FractalContainer:
    id: str
    level: int
    position: Tuple[int, ...]
    data: np.ndarray
    shape: Tuple[int, ...]
    dtype: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    last_accessed: float = 0.0
    access_count: int = 0
    priority: float = 0.0

    def update_priority(self, value: float) -> None:
        try:
            self.priority = float(value)
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid priority value {value}: {e}")
            self.priority = 0.0

    def get_memory_size(self) -> int:
        try:
            itemsize = np.dtype(self.dtype).itemsize if isinstance(self.dtype, str) else self.data.dtype.itemsize
        except (TypeError, AttributeError) as e:
            logger.warning(f"Error getting itemsize for dtype {self.dtype}: {e}")
            itemsize = self.data.dtype.itemsize
        return int(self.data.size) * int(itemsize)


# ----------------------- Хранилище весов модели -----------------------
class FractalWeightStore:
    """Хранилище весов в фрактальной структуре для low-memory режима."""

    def __init__(
        self,
        block_size: int = 64,
        fractal_levels: int = 5,
        containers_per_group: int = 4,
        device: str = "cpu"
    ) -> None:
        self.containers: Dict[str, FractalContainer] = {}
        self.fractal_tree: Dict[int, List[str]] = {}
        self.hot_window: OrderedDict[str, float] = OrderedDict()
        self.total_memory: int = 0
        self.model_id: Optional[str] = None
        self.block_size: int = max(1, int(block_size))
        self.fractal_levels: int = max(1, int(fractal_levels))
        self.containers_per_group: int = max(1, int(containers_per_group))
        self.hot_window_size: int = 500 * 1024 * 1024
        self.SPECIAL_TOKENS: Dict[str, int] = {
            "NODE_START": 1,
            "NODE_END": 2,
            "EDGE_START": 3,
            "EDGE_END": 4,
        }
        self.NODE_OFFSET: int = 1_000_000
        self.graph_metadata: Dict[str, Any] = {}
        self.device: str = "cpu"
        self.gpu_hot_cache: Dict[str, torch.Tensor] = {}
        self.lazy_index: Dict[str, Any] = {}

        try:
            use_cuda = (device != "cpu")
            if use_cuda and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        except (RuntimeError, OSError) as e:
            logger.warning(f"Error setting device: {e}")
            self.device = "cpu"

        logger.info(f"FractalWeightStore initialized on {self.device}")

    # ----------------------- Публичные методы -----------------------
    def pack_model_weights(self, model: torch.nn.Module, model_id: str) -> bool:
        """Упаковывает веса модели в фрактальную структуру."""
        start_time = time.time()
        logger.info(f"Начата фрактальная упаковка весов модели {model_id}...")
        try:
            self.containers.clear()
            self.fractal_tree.clear()
            self.hot_window.clear()
            self.total_memory = 0
            self.model_id = model_id

            logger.debug("Создание контейнеров нулевого уровня...")
            for layer_name, layer in model.named_modules():
                if not hasattr(layer, 'weight') or layer.weight is None:
                    continue
                self._pack_layer_weights(layer_name, layer.weight.data.cpu().numpy(), model_id)

            logger.debug("Построение иерархии фрактала...")
            self._build_fractal_hierarchy()

            logger.debug("Инициализация горячего окна...")
            self._initialize_hot_window()

            logger.debug("Выполнение оптимизации структуры...")
            self._optimize_fractal_structure()

            stats = self.get_statistics()
            logger.info(
                f"Фрактальная упаковка весов завершена за {time.time() - start_time:.2f} сек. "
                f"Создано {stats['total_containers']} контейнеров. "
                f"Общий размер: {stats['total_memory_mb']:.2f} MB."
            )
            return True
        except Exception as e:
            logger.error(f"Критическая ошибка фрактальной упаковки весов: {e}", exc_info=True)
            return False

    def pack_state_dict(self, state_dict: Dict[str, torch.Tensor], model_id: str) -> bool:
        """Упаковывает полный state_dict в фрактальную структуру."""
        start_time = time.time()
        logger.info(f"Начата фрактальная упаковка state_dict модели {model_id}...")
        try:
            self.containers.clear()
            self.fractal_tree.clear()
            self.hot_window.clear()
            self.total_memory = 0
            self.model_id = model_id

            if 0 not in self.fractal_tree:
                self.fractal_tree[0] = []

            for tpath, tensor in (state_dict or {}).items():
                if not isinstance(tpath, str) or not tpath:
                    continue
                if tensor is None or not isinstance(tensor, torch.Tensor):
                    continue

                try:
                    arr = tensor.detach().cpu().numpy()
                except Exception:
                    continue

                try:
                    layer_name, param_name = tpath.rsplit('.', 1)
                except ValueError:
                    layer_name, param_name = tpath, "weight"

                layer_key = f"{layer_name}.{param_name}" if layer_name else tpath
                flat = arr.reshape(-1)
                total_elements = int(flat.size)
                original_shape = tuple(int(x) for x in arr.shape)

                is_critical = any(critical in tpath for critical in ["wte", "wpe", "ln_f", "lm_head"])
                storage_dtype = "float64" if is_critical else "float32"

                for i in range(0, total_elements, self.block_size):
                    block = flat[i:i + self.block_size]
                    if storage_dtype == "float64":
                        block = block.astype(np.float64, copy=False)
                    else:
                        block = block.astype(np.float32, copy=False)

                    position = (i // self.block_size,)
                    cid = self._generate_container_id(0, position, layer_key, model_id)

                    meta = {
                        "layer_name": layer_name,
                        "model_id": model_id,
                        "original_shape": original_shape,
                        "block_start": i,
                        "block_end": min(i + self.block_size, total_elements),
                        "is_critical": is_critical,
                        "storage_dtype": storage_dtype,
                        "param_name": param_name,
                        "tensor_path": tpath,
                    }

                    container = FractalContainer(
                        id=cid,
                        level=0,
                        position=position,
                        data=block,
                        shape=(int(block.size),),
                        dtype=storage_dtype,
                        metadata=meta,
                    )
                    self.containers[cid] = container
                    self.fractal_tree[0].append(cid)
                    self.total_memory += container.get_memory_size()

            self._build_fractal_hierarchy()
            self._initialize_hot_window()
            self._optimize_fractal_structure()

            stats = self.get_statistics()
            logger.info(
                f"Фрактальная упаковка state_dict завершена за {time.time() - start_time:.2f} сек. "
                f"Контейнеров: {stats['total_containers']}, размер: {stats['total_memory_mb']:.2f} MB"
            )
            return True
        except Exception as e:
            logger.error(f"Критическая ошибка фрактальной упаковки state_dict: {e}", exc_info=True)
            return False

    def get_container(self, container_id: str, load_from_disk: bool = True) -> Optional[FractalContainer]:
        """Получает контейнер по ID с динамической загрузкой в горячее окно."""
        if container_id not in self.containers:
            logger.warning(f"Контейнер {container_id} не найден")
            return None

        container = self.containers[container_id]
        container.last_accessed = time.time()
        container.access_count += 1

        if container_id in self.hot_window:
            return container

        base_priority = self._calculate_container_priority(container_id)
        container.update_priority(base_priority)

        required_space = container.get_memory_size()
        available_space = self._get_available_hot_window_space()

        if required_space <= available_space:
            self.hot_window[container_id] = container.priority
            logger.debug(f"Контейнер {container_id} добавлен в горячее окно (уровень {container.level})")
            return container

        logger.debug(f"Недостаточно места в горячем окне для контейнера {container_id}. Вытеснение...")
        self._evict_lowest_priority_containers(required_space - available_space)

        if required_space <= self._get_available_hot_window_space():
            self.hot_window[container_id] = container.priority
            logger.debug(f"Контейнер {container_id} добавлен в горячее окно после вытеснения")
            return container

        if load_from_disk and ("offload_path" in container.metadata):
            logger.debug(f"Загрузка контейнера {container_id} с диска...")
            if self._load_container_from_ssd(container_id):
                return self.get_container(container_id, load_from_disk=False)

        logger.warning(f"Не удалось добавить контейнер {container_id} в горячее окно")
        return None

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

    def clear(self) -> None:
        """Очищает хранилище."""
        self.containers.clear()
        self.fractal_tree.clear()
        self.hot_window.clear()
        self.total_memory = 0
        self.model_id = None
        try:
            if hasattr(self, "gpu_hot_cache"):
                self.gpu_hot_cache.clear()
            if getattr(self, "device", "cpu") == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def has_tensor(self, tensor_id: str) -> bool:
        """Check if a tensor exists in storage"""
        return tensor_id in self.containers

    def get_tensor(self, tensor_id: str) -> Optional[torch.Tensor]:
        """Get a tensor from storage"""
        container = self.get_container(tensor_id)
        if container is None:
            return None
        arr = container.data
        tensor = torch.from_numpy(np.asarray(arr))
        if self.device != "cpu":
            tensor = tensor.to(self.device)
        return tensor

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

    def save_weights(self, model_id: str, weights: Dict[str, torch.Tensor]) -> bool:
        """Save model weights to storage"""
        return self.pack_state_dict(weights, model_id)

    def load_weights(self, model_id: str) -> Dict[str, torch.Tensor]:
        """Load model weights from storage"""
        state_dict = self.reconstruct_state_dict(
            include_params=[k for k in self.index.keys() if k.startswith(model_id)]
        )
        return state_dict

    def unpack_model_weights(self, model: torch.nn.Module, model_id: str) -> bool:
        """Unpack model weights from storage into model"""
        try:
            state_dict = self.reconstruct_state_dict()
            model.load_state_dict(state_dict, strict=False)
            return True
        except Exception:
            return False

    def get_model_stats(self, model_id: str) -> Dict[str, Any]:
        """Get statistics for a stored model"""
        model_containers = [cid for cid in self.containers.keys() if model_id in cid]
        total_size = sum(self.containers[cid].get_memory_size() for cid in model_containers if cid in self.containers)
        return {
            "compressed_size": total_size,
            "num_containers": len(model_containers)
        }

    def remove_model(self, model_id: str) -> bool:
        """Remove a model from storage"""
        try:
            to_remove = [cid for cid in list(self.containers.keys()) if model_id in cid]
            for cid in to_remove:
                del self.containers[cid]
            for level in self.fractal_tree:
                self.fractal_tree[level] = [cid for cid in self.fractal_tree[level] if cid not in to_remove]
            return True
        except Exception:
            return False

    # ----------------------- Вспомогательные методы -----------------------
    def _pack_layer_weights(self, layer_name: str, weights: np.ndarray, model_id: str) -> None:
        """Упаковывает веса слоя в фрактальную структуру на уровне 0."""
        dtype = str(weights.dtype)
        shape = weights.shape
        flat_weights = weights.flatten()
        total_elements = len(flat_weights)

        is_critical_layer = any(critical in layer_name for critical in ["wte", "wpe", "ln_f", "lm_head"])
        storage_dtype = "float64" if is_critical_layer else "float32"

        logger.debug(f"Упаковка слоя {layer_name} ({shape}) в {storage_dtype}...")

        for i in range(0, total_elements, self.block_size):
            block_data = flat_weights[i:i + self.block_size]
            if storage_dtype == "float64":
                block_data = block_data.astype(np.float64)
            else:
                block_data = block_data.astype(np.float32)

            block_shape = (len(block_data),)
            position = (i // self.block_size,)
            container_id = self._generate_container_id(0, position, layer_name, model_id)

            metadata = {
                "layer_name": layer_name,
                "model_id": model_id,
                "original_shape": shape,
                "block_start": i,
                "block_end": min(i + self.block_size, total_elements),
                "is_critical": is_critical_layer,
                "storage_dtype": storage_dtype,
                "param_name": "weight",
                "tensor_path": f"{layer_name}.weight",
            }

            container = FractalContainer(
                id=container_id,
                level=0,
                position=position,
                data=block_data,
                shape=block_shape,
                dtype=storage_dtype,
                metadata=metadata
            )

            self.containers[container_id] = container
            if 0 not in self.fractal_tree:
                self.fractal_tree[0] = []
            self.fractal_tree[0].append(container_id)
            self.total_memory += container.get_memory_size()

            if len(self.fractal_tree[0]) % 1000 == 0:
                logger.debug(f"Создано {len(self.fractal_tree[0])} контейнеров для уровня 0")

    def _generate_container_id(
        self,
        level: int,
        position: Tuple[int, ...],
        layer_name: str,
        model_id: str
    ) -> str:
        return f"{model_id}::{layer_name}::L{level}::pos{'-'.join(map(str, position))}"

    def _build_fractal_hierarchy(self) -> None:
        """Строит иерархию фрактала на основе контейнеров нулевого уровня."""
        if 0 not in self.fractal_tree:
            self.fractal_tree[0] = []

        logger.info(f"Построение фрактальной иерархии ({self.fractal_levels} уровней)...")

        for level in range(1, self.fractal_levels):
            parent_containers = self.fractal_tree.get(level - 1, [])
            if not parent_containers:
                continue

            logger.debug(f"Построение уровня {level} из {len(parent_containers)} контейнеров...")
            k = self.containers_per_group
            total_groups = (len(parent_containers) + k - 1) // k
            progress_interval = max(1, total_groups // 10)

            for i in range(0, len(parent_containers), k):
                group = parent_containers[i: i + k]
                if not group:
                    continue

                grp_index = i // k
                if grp_index % progress_interval == 0:
                    progress = (grp_index + 1) / total_groups * 100.0
                    logger.debug(f"Уровень {level}: {progress:.1f}% завершено")

                position = (grp_index,)
                first_child = self.containers[group[0]]
                layer_name = first_child.metadata.get("layer_name", "")
                model_id = first_child.metadata.get("model_id", self.model_id or "")
                container_id = self._generate_container_id(level, position, layer_name, model_id)

                child_data_segments: List[np.ndarray] = []
                for child_id in group:
                    child = self.containers[child_id]
                    child_data_segments.append(child.data)
                    child.parent = container_id

                try:
                    combined_data = np.concatenate(child_data_segments, axis=0)
                except Exception:
                    combined_data = np.concatenate([seg.reshape(-1) for seg in child_data_segments], axis=0)

                if level == 1:
                    storage_dtype = "float32"
                elif level == 2:
                    storage_dtype = "float16"
                else:
                    storage_dtype = "int8"

                scale_val: Optional[float] = None
                zero_scale_flag: bool = False

                if storage_dtype == "float32":
                    f32_max = np.finfo(np.float32).max
                    f32_min = np.finfo(np.float32).min
                    arr = np.nan_to_num(combined_data, copy=False, nan=0.0, posinf=f32_max, neginf=f32_min)
                    combined_data = arr.astype(np.float32, copy=False)
                elif storage_dtype == "float16":
                    f16_max = 6.5e4
                    f16_min = -6.5e4
                    arr = np.nan_to_num(combined_data, copy=False, nan=0.0, posinf=f16_max, neginf=f16_min)
                    arr = np.clip(arr, f16_min, f16_max)
                    combined_data = arr.astype(np.float16, copy=False)
                elif storage_dtype == "int8":
                    quant_f, scale_val, zero_scale_flag = self._safe_quantize_to_int8(combined_data)
                    combined_data = quant_f

                metadata = {
                    "layer_name": layer_name,
                    "model_id": model_id,
                    "child_count": len(group),
                    "child_ids": group,
                    "storage_dtype": storage_dtype,
                }

                if isinstance(first_child, FractalContainer):
                    tp = first_child.metadata.get("tensor_path")
                    if tp is not None:
                        metadata["tensor_path"] = tp
                    pn = first_child.metadata.get("param_name")
                    if pn is not None:
                        metadata["param_name"] = pn

                if scale_val is not None:
                    metadata["quant_scale"] = scale_val
                    metadata["quantization_scale"] = scale_val
                if zero_scale_flag:
                    metadata["has_zero_scale"] = True

                container = FractalContainer(
                    id=container_id,
                    level=level,
                    position=position,
                    data=combined_data.copy(),
                    shape=(int(combined_data.size),),
                    dtype=storage_dtype,
                    metadata=metadata,
                    children=list(group),
                )

                self.containers[container_id] = container
                if level not in self.fractal_tree:
                    self.fractal_tree[level] = []
                self.fractal_tree[level].append(container_id)
                self.total_memory += container.get_memory_size()

    def _safe_quantize_to_int8(
        self,
        data: np.ndarray
    ) -> Tuple[np.ndarray, float, bool]:
        """Безопасное квантование данных в int8."""
        arr = np.nan_to_num(data, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
        if arr.size == 0:
            return arr.astype(np.int8, copy=False), 1.0, False

        max_abs = float(np.max(np.abs(arr)))
        if max_abs == 0.0:
            return np.zeros(arr.shape, dtype=np.int8), 1.0, True

        scale = max(max_abs / 127.0, 1e-12)
        normalized = arr / scale
        quantized = np.clip(np.round(normalized), -127, 127).astype(np.int8, copy=False)
        return quantized, scale, False

    def _initialize_hot_window(self) -> None:
        """Инициализирует горячее окно на основе фрактальной структуры."""
        logger.debug("Инициализация горячего окна фрактальной памяти...")
        candidates: List[Tuple[str, float]] = []

        top_level = self.fractal_levels - 1
        if top_level in self.fractal_tree:
            for container_id in self.fractal_tree[top_level]:
                priority = self._calculate_container_priority(container_id)
                candidates.append((container_id, priority))

        if self.fractal_levels > 2:
            mid_level = self.fractal_levels // 2
            if mid_level in self.fractal_tree and self.fractal_tree[mid_level]:
                mids = self.fractal_tree[mid_level]
                sample_ids = mids[:3] + (mids[-3:] if len(mids) > 3 else [])
                for container_id in sample_ids:
                    priority = self._calculate_container_priority(container_id)
                    candidates.append((container_id, priority))

        if 0 in self.fractal_tree:
            for container_id in self.fractal_tree[0]:
                container = self.containers[container_id]
                if container.metadata.get("is_critical", False):
                    priority = self._calculate_container_priority(container_id)
                    candidates.append((container_id, priority))

        candidates.sort(key=lambda x: x[1], reverse=True)

        self.hot_window.clear()
        current_size = 0
        now_ts = time.time()

        for container_id, priority in candidates:
            container = self.containers.get(container_id)
            if container is None:
                continue

            container_size = container.get_memory_size()
            if current_size + container_size <= self.hot_window_size:
                self.hot_window[container_id] = float(priority)
                current_size += container_size
                container.last_accessed = now_ts
                container.access_count += 1

                try:
                    if getattr(self, "device", "cpu") == "cuda" and torch.cuda.is_available():
                        if container_id not in self.gpu_hot_cache:
                            np_arr = container.data
                            if container.dtype == "float16":
                                tensor = torch.from_numpy(np_arr.astype(np.float16, copy=False))
                            elif container.dtype == "float32":
                                tensor = torch.from_numpy(np_arr.astype(np.float32, copy=False))
                            else:
                                tensor = torch.from_numpy(np_arr)
                            self.gpu_hot_cache[container_id] = tensor.to("cuda", non_blocking=True)
                except Exception:
                    pass
            else:
                break

        hot_window_size_mb = current_size / (1024 * 1024)
        logger.debug(
            f"Горячее окно фрактальной памяти инициализировано с {len(self.hot_window)} контейнерами. "
            f"Размер: {hot_window_size_mb:.2f} MB из {self.hot_window_size / (1024 * 1024):.2f} MB"
        )

    def _calculate_container_priority(self, container_id: str) -> float:
        """Вычисляет приоритет контейнера."""
        cont = self.containers.get(container_id)
        if cont is None:
            return 0.0

        level_factor = pow(0.9, max(0, int(cont.level)))
        t = max(0.0, time.time() - float(cont.last_accessed or 0.0))
        time_factor = 0.1 + 0.9 * (1.0 - min(t / 3600.0, 1.0))
        crit = 0.3 if cont.metadata.get("is_critical", False) else 1.0

        return float(level_factor * time_factor * crit)

    def _get_available_hot_window_space(self) -> int:
        used = 0
        for cid in self.hot_window.keys():
            cont = self.containers.get(cid)
            if cont is None:
                continue
            used += cont.get_memory_size()
        return max(0, int(self.hot_window_size - used))

    def _evict_lowest_priority_containers(self, bytes_needed: int) -> None:
        if bytes_needed <= 0 or not self.hot_window:
            return

        items = list(self.hot_window.items())
        items.sort(key=lambda x: x[1])
        freed = 0

        for cid, pr in items:
            cont = self.containers.get(cid)
            size = cont.get_memory_size() if cont is not None else 0
            self.hot_window.pop(cid, None)

            try:
                if hasattr(self, "gpu_hot_cache") and cid in self.gpu_hot_cache:
                    self.gpu_hot_cache.pop(cid, None)
            except Exception:
                pass

            freed += size
            if freed >= bytes_needed:
                break

    def _load_container_from_ssd(self, container_id: str) -> bool:
        """Загружает данные контейнера из offload-пути."""
        cont = self.containers.get(container_id)
        if cont is None:
            return False

        off_path = cont.metadata.get("offload_path")
        if not off_path:
            return False

        try:
            return False
        except Exception:
            logger.exception(f"Ошибка загрузки контейнера {container_id} с диска")
            return False

    def _optimize_fractal_structure(self) -> None:
        """Оптимизирует фрактальную структуру для повышения эффективности."""
        logger.info("Оптимизация фрактальной структуры...")
        usage_stats = self._analyze_container_usage()

        if self._needs_reconfiguration(usage_stats):
            logger.info("Требуется реконфигурация фрактальной структуры...")
            self._reconfigure_fractal_structure(usage_stats)

        logger.info("Оптимизация расположения контейнеров...")
        try:
            for cid in list(self.containers.keys()):
                pr = self._calculate_container_priority(cid)
                c = self.containers.get(cid)
                if c is not None:
                    c.update_priority(pr)
            self._initialize_hot_window()
        except Exception as e:
            logger.warning(f"Не удалось оптимизировать расположение контейнеров: {e}")

        logger.info("Обновление метаданных...")
        self._update_metadata()

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

    def _reconfigure_fractal_structure(self, usage_stats: Dict[str, Any]) -> None:
        """Перестраивает фрактальную структуру на основе статистики использования."""
        new_params = self._determine_new_parameters(usage_stats)
        logger.info(
            f"Реконфигурация фрактальной структуры: уровни={new_params['fractal_levels']}, блок={new_params['block_size']}"
        )

        all_data = self._extract_all_data()
        self.containers.clear()
        self.fractal_tree.clear()
        self.hot_window.clear()
        self.total_memory = 0

        self.fractal_levels = int(new_params["fractal_levels"])
        self.block_size = int(new_params["block_size"])

        model_id = self.model_id or "fractal"
        for layer_name, weights in all_data.items():
            try:
                self._pack_layer_weights(layer_name, weights, model_id)
            except Exception as e:
                logger.error(f"Не удалось переупаковать слой {layer_name}: {e}")

        self._build_fractal_hierarchy()
        self._initialize_hot_window()
        self._save_reconfiguration_history(new_params)
        logger.info("Фрактальная структура успешно реконфигурирована")

    def _determine_new_parameters(self, usage_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Определяет новые параметры фрактала на основе статистики использования."""
        new_params = {
            "fractal_levels": int(self.fractal_levels),
            "block_size": int(self.block_size),
        }

        spatial = float(usage_stats.get("spatial_locality", 0.0))
        if spatial < 0.4:
            new_params["fractal_levels"] = min(6, int(self.fractal_levels) + 1)
        elif spatial > 0.6:
            new_params["fractal_levels"] = max(3, int(self.fractal_levels) - 1)

        temporal = float(usage_stats.get("temporal_locality", 0.0))
        if temporal < 0.3:
            new_params["block_size"] = max(16, int(self.block_size) // 2)
        elif temporal > 0.5:
            new_params["block_size"] = min(128, int(self.block_size) * 2)

        return new_params

    def _extract_all_data(self) -> Dict[str, np.ndarray]:
        """Восстанавливает тензоры слоёв из контейнеров уровня 0."""
        layers: Dict[str, List[Tuple[int, np.ndarray, Tuple[int, ...]]]] = defaultdict(list)

        for cid in self.fractal_tree.get(0, []):
            cont = self.containers.get(cid)
            if cont is None:
                continue
            layer_name = cont.metadata.get("layer_name", "")
            orig_shape = tuple(cont.metadata.get("original_shape", cont.shape))
            start = int(cont.metadata.get("block_start", 0))
            layers[layer_name].append((start, cont.data.copy(), orig_shape))

        result: Dict[str, np.ndarray] = {}
        for layer, chunks in layers.items():
            if not chunks:
                continue
            chunks.sort(key=lambda x: x[0])
            arr = np.concatenate([c[1].reshape(-1) for c in chunks], axis=0)
            target_shape: Tuple[int, ...] = chunks[0][2]
            try:
                result[layer] = arr.reshape(target_shape)
            except Exception:
                result[layer] = arr
        return result

    def _update_metadata(self) -> None:
        """Лёгкое обновление агрегированных метаданных хранилища."""
        try:
            self.total_memory = 0
            for c in self.containers.values():
                self.total_memory += c.get_memory_size()
        except Exception:
            pass

    def _save_reconfiguration_history(self, params: Dict[str, Any]) -> None:
        """Сохраняет запись об изменении конфигурации в памяти."""
        try:
            rec = {
                "ts": time.time(),
                "params": dict(params),
                "stats": self.get_statistics() if hasattr(self, "get_statistics") else {},
            }
            hist = getattr(self, "reconfiguration_history", None)
            if not isinstance(hist, list):
                self.reconfiguration_history = []
            self.reconfiguration_history.append(rec)
        except Exception:
            pass

    def save_to_disk(self, output_path: str, knowledge_graph: Optional[Dict[str, Any]] = None) -> bool:
        """Сохраняет фрактальную структуру на диск."""
        try:
            out_dir = Path(output_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            data_dir = out_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            containers_jsonl = []
            for cid, cont in self.containers.items():
                sha1 = hashlib.sha1(cid.encode("utf-8")).hexdigest()
                file_name = f"{sha1}.npy"
                file_path = data_dir / file_name
                np.save(file_path, cont.data)

                containers_jsonl.append(
                    {
                        "id": cid,
                        "level": int(cont.level),
                        "position": list(cont.position),
                        "shape": list(cont.shape),
                        "dtype": str(cont.dtype),
                        "metadata": cont.metadata,
                        "parent": cont.parent,
                        "children": cont.children,
                        "data_path": str(file_path.relative_to(out_dir)),
                        "file_id_sha1": sha1,
                    }
                )

            with (out_dir / "containers.jsonl").open("w", encoding="utf-8") as f:
                for row in containers_jsonl:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            index = {
                "model_id": self.model_id,
                "created_ts": time.time(),
                "params": {
                    "block_size": self.block_size,
                    "fractal_levels": self.fractal_levels,
                    "containers_per_group": self.containers_per_group,
                    "hot_window_size": self.hot_window_size,
                },
                "stats": self.get_statistics(),
            }
            with (out_dir / "index.json").open("w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False, indent=2)

            return True
        except Exception:
            logger.exception("Ошибка сохранения фрактальной структуры")
            return False

    def load_from_disk(self, input_path: str, lazy: bool = False, progress_every: int = 500000) -> bool:
        """Загружает фрактальную структуру с диска."""
        try:
            in_dir = Path(input_path)
            index_path = in_dir / "index.json"
            containers_path = in_dir / "containers.jsonl"
            data_dir = in_dir / "data"
            shards_manifest = in_dir / "shards_manifest.jsonl"

            if index_path.exists() and data_dir.exists() and (not containers_path.exists()):
                report = self._load_from_disk_atomic_format(str(in_dir))
                if not report.get("ok", False):
                    logger.error(f"Не удалось загрузить (atomic): {report.get('error')}")
                    return bool(report.get("ok", False))

            if shards_manifest.exists():
                self.clear()
                if not hasattr(self, "lazy_index"):
                    self.lazy_index = {}

                if index_path.exists():
                    try:
                        with index_path.open("r", encoding="utf-8") as f:
                            idx = json.load(f)
                        self.model_id = idx.get("model_id")
                        params = idx.get("params", {})
                        self.block_size = int(params.get("block_size", self.block_size))
                        self.fractal_levels = int(params.get("fractal_levels", self.fractal_levels))
                        self.containers_per_group = int(params.get("containers_per_group", self.containers_per_group))
                        self.hot_window_size = int(params.get("hot_window_size", self.hot_window_size))
                    except Exception:
                        logger.debug("Не удалось разобрать index.json для шардированного формата", exc_info=True)

                loaded = 0
                with shards_manifest.open("r", encoding="utf-8") as mf:
                    for line_idx, line in enumerate(mf, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            cid = rec["id"]
                            level = int(rec["level"])
                            pos = tuple(int(x) for x in rec.get("position", []))
                            shape = tuple(int(x) for x in rec.get("shape", []))
                            dtype = str(rec.get("dtype", "float32"))
                            meta = rec.get("metadata", {}) or {}
                            parent = rec.get("parent")
                            children = rec.get("children", []) or []
                            shard_file = in_dir / rec["shard_file"]
                            key = rec["key"]

                            if lazy:
                                self.lazy_index[cid] = {
                                    "level": level,
                                    "position": pos,
                                    "shape": shape,
                                    "dtype": dtype,
                                    "metadata": meta,
                                    "parent": parent,
                                    "children": list(children),
                                    "shard_file": str(shard_file),
                                    "key": key,
                                }
                                self.fractal_tree.setdefault(level, []).append(cid)
                                loaded += 1
                            else:
                                with np.load(shard_file, allow_pickle=False) as zf:
                                    arr = zf[key]
                                    try:
                                        if shape and int(np.prod(shape)) == int(arr.size):
                                            arr = arr.reshape(shape)
                                    except Exception:
                                        pass
                                cont = FractalContainer(
                                    id=cid,
                                    level=level,
                                    position=pos,
                                    data=arr,
                                    shape=arr.shape,
                                    dtype=dtype,
                                    metadata=meta,
                                    parent=parent,
                                    children=list(children),
                                )
                                self.containers[cid] = cont
                                self.fractal_tree.setdefault(level, []).append(cid)
                                self.total_memory += cont.get_memory_size()
                                loaded += 1
                        except Exception:
                            logger.exception("Ошибка разбора записи из shards_manifest.jsonl")
                            continue

                        if lazy and progress_every and (line_idx % int(progress_every) == 0):
                            logger.info(f"Проиндексировано лениво: {line_idx} записей манифеста...")

                if loaded == 0:
                    logger.error("Не удалось загрузить ни одного контейнера из shards_manifest.jsonl")
                    return False

                try:
                    if not lazy:
                        self._initialize_hot_window()
                except Exception:
                    logger.warning("Не удалось инициализировать горячее окно после загрузки (sharded)", exc_info=True)

                logger.info(
                    f"Загружено (sharded){' (lazy index)' if lazy else ''} контейнеров: {loaded}. "
                    f"Уровни: {sorted(self.fractal_tree.keys())}."
                )
                return True

        except Exception as e:
            logger.error(f"Ошибка загрузки фрактальной структуры: {e}", exc_info=True)
            return False

        return self._load_legacy_format(index_path, containers_path, data_dir)

    def _load_from_disk_atomic_format(self, source_dir: str) -> Dict[str, Any]:
        """Загружает состояние фрактального хранилища (atomic-формат)."""
        try:
            base = Path(source_dir)
            with open(base / "index.json", "r", encoding="utf-8") as f:
                index = json.load(f)

            self.clear()
            self.model_id = index.get("model_id")
            self.fractal_levels = int(index.get("fractal_levels", self.fractal_levels))
            self.block_size = int(index.get("block_size", self.block_size))
            self.graph_metadata = index.get("graph_metadata", {})
            self.containers = {}
            self.fractal_tree = {int(k): list(v) for k, v in index.get("fractal_tree", {}).items()}

            for ci in index.get("containers", []):
                cid = ci["id"]
                level = int(ci["level"])
                pos = tuple(int(x) for x in ci.get("position", []))
                shape = tuple(int(x) for x in ci.get("shape", []))
                dtype = str(ci.get("dtype", "float32"))
                meta = ci.get("metadata", {})
                data_path = base / str(ci.get("file"))
                arr = np.load(data_path, allow_pickle=False)

                try:
                    arr = arr.reshape(shape)
                except Exception:
                    pass

                cont = FractalContainer(
                    id=cid,
                    level=level,
                    position=pos,
                    data=arr,
                    shape=arr.shape,
                    dtype=dtype,
                    metadata=meta,
                )
                self.containers[cid] = cont

            self._update_metadata()
            self._initialize_hot_window()

            disk_checksum = index.get("checksum", "")
            mem_checksum = self.compute_checksum()
            if disk_checksum and mem_checksum and disk_checksum != mem_checksum:
                return {"ok": False, "checksum": mem_checksum, "error": "checksum_mismatch"}

            return {"ok": True, "checksum": mem_checksum or disk_checksum}
        except Exception as e:
            return {"ok": False, "checksum": "", "error": str(e)}

    def _load_legacy_format(
        self,
        index_path: Path,
        containers_path: Path,
        data_dir: Path
    ) -> bool:
        """Загружает устаревший формат хранилища."""
        try:
            if not index_path.exists() or not containers_path.exists():
                return False

            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)

            self.clear()
            self.model_id = index.get("model_id")
            params = index.get("params", {})
            self.block_size = int(params.get("block_size", self.block_size))
            self.fractal_levels = int(params.get("fractal_levels", self.fractal_levels))

            with open(containers_path, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line.strip())
                    cid = rec["id"]
                    data_path = data_dir / rec["data_path"]
                    arr = np.load(data_path, allow_pickle=False)

                    cont = FractalContainer(
                        id=cid,
                        level=int(rec["level"]),
                        position=tuple(rec["position"]),
                        data=arr,
                        shape=tuple(rec["shape"]),
                        dtype=str(rec["dtype"]),
                        metadata=rec.get("metadata", {}),
                    )
                    self.containers[cid] = cont
                    self.fractal_tree.setdefault(cont.level, []).append(cid)
                    self.total_memory += cont.get_memory_size()

            self._initialize_hot_window()
            return True
        except Exception:
            logger.exception("Ошибка загрузки устаревшего формата")
            return False

    def compute_checksum(self) -> str:
        """Возвращает агрегированный SHA256 всех контейнеров."""
        import hashlib
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


# ----------------------- Утилиты переупаковки -----------------------
def repack_model_to_fractal(
    model_path: str,
    output_path: str,
    fractal_levels: int = 4,
    block_size: int = 64,
    device: str = "cpu",
) -> bool:
    """Переупаковывает модель в фрактальную структуру."""
    start_time = time.time()
    logger.info(f"Начата переупаковка модели из {model_path} в фрактальную структуру...")
    try:
        logger.info("Загрузка модели...")
        model: Optional[torch.nn.Module]
        if os.path.isdir(model_path):
            model = _load_hf_model_dir(model_path, device=device)
        else:
            model = _safe_load_model(model_path, device=device)

        if model is None:
            logger.error("Не удалось загрузить модель")
            return False

        store = FractalWeightStore(block_size=block_size, fractal_levels=fractal_levels)
        logger.info("Извлечение знаний из модели...")
        knowledge = store.extract_knowledge_from_model(model)

        logger.info("Построение графа знаний...")
        knowledge_graph = _build_knowledge_graph(knowledge)

        logger.info("Создание фрактальной структуры весов...")
        model_id = Path(model_path).stem
        if not store.pack_model_weights(model, model_id=model_id):
            logger.error("Не удалось упаковать веса модели")
            return False

        logger.info("Сохранение фрактальной структуры (шардировано)...")
        ok = store.save_to_disk(output_path, knowledge_graph=knowledge_graph)
        if not ok:
            return False

        stats = store.get_statistics()
        logger.info("Статистика фрактальной структуры:")
        logger.info(f"  Общее количество контейнеров: {stats['total_containers']}")
        logger.info(f"  Контейнеры по уровням: {stats['containers_by_level']}")
        logger.info(f"  Общий размер: {stats['total_memory_mb']:.2f} MB")
        logger.info(f"Переупаковка завершена за {time.time() - start_time:.2f} сек")
        return True
    except Exception:
        logger.exception("Критическая ошибка переупаковки модели")
        return False


def _safe_load_model(model_path: str, device: str = "cpu") -> Optional[torch.nn.Module]:
    """Простой загрузчик PyTorch-модели."""
    map_location = torch.device(device if device else "cpu")
    try:
        obj = torch.load(model_path, map_location=map_location, weights_only=False)
        if isinstance(obj, torch.nn.Module):
            obj.eval()
            return obj
        logger.error("Ожидалась сохранённая torch.nn.Module, получен другой объект")
        return None
    except Exception:
        logger.exception(f"Ошибка загрузки модели из {model_path}")
        return None


def _load_hf_model_dir(model_dir: str, device: str = "cpu") -> Optional[torch.nn.Module]:
    """Загружает модель из директории HuggingFace."""
    try:
        if AutoModelForCausalLM is None:
            raise ImportError("transformers не установлен: pip install transformers safetensors accelerate")

        base = Path(model_dir)
        real_dir = base

        if not (base / "config.json").exists():
            snaps = list((base / "snapshots").glob("*/config.json")) if (base / "snapshots").exists() else []
            if snaps:
                real_dir = snaps[0].parent
            else:
                found = list(base.rglob("config.json"))
                if found:
                    real_dir = found[0].parent

        model_path_str = str(real_dir)
        torch_dtype = torch.float16 if (device == "cuda" and torch.cuda.is_available()) else torch.float32

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path_str,
                local_files_only=True,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
                torch_dtype=torch_dtype,
            )
        except Exception:
            if AutoModel is None:
                raise
            model = AutoModel.from_pretrained(
                model_path_str,
                local_files_only=True,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
                torch_dtype=torch_dtype,
            )

        if device == "cuda" and torch.cuda.is_available():
            model.to("cuda")
        else:
            model.to("cpu")
        model.eval()
        return model
    except Exception:
        logger.exception(f"Ошибка загрузки HF-модели из директории {model_dir}")
        return None


def _build_knowledge_graph(knowledge: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Строит простой граф знаний из списка фактов/отношений/концептов."""
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []

    for item in knowledge:
        t = item.get("type", "unknown")
        if t == "attention_relation":
            subj = item.get("subject") or item.get("layer", "") + ":src"
            obj = item.get("object") or item.get("layer", "") + ":dst"
            pred = item.get("predicate", "attn")
            nodes.setdefault(subj, {"id": subj, "kind": "token"})
            nodes.setdefault(obj, {"id": obj, "kind": "token"})
            edges.append({"source": subj, "target": obj, "relation": pred, "weight": item.get("score", 0.0)})
        elif t.startswith("ffn_pattern") or t == "ffn_fact":
            nid = item.get("layer", "ffn")
            nodes.setdefault(nid, {"id": nid, "kind": "ffn"})
            edges.append({"source": nid, "target": nid, "relation": t, "weight": item.get("score", 0.0)})
        elif t == "embedding_concept":
            idx = item.get("details", {}).get("index")
            nid = f"concept_{idx}" if idx is not None else f"concept_{len(nodes)}"
            nodes.setdefault(nid, {"id": nid, "kind": "concept"})
        else:
            nid = f"node_{len(nodes)}"
            nodes.setdefault(nid, {"id": nid, "kind": t})

    return {
        "nodes": list(nodes.values()),
        "edges": edges,
        "meta": {"generated_ts": time.time(), "counts": {"nodes": len(nodes), "edges": len(edges)}},
    }


def export_hf_model_to_fractal(
    hf_model_dir_or_id: str,
    output_path: str,
    model_id: str,
    tokenizer_output_subdir: str = "tokenizer",
    device: str = "cpu",
    fractal_levels: int = 4,
    block_size: int = 64,
    local_files_only: bool = True,
) -> bool:
    """Экспортирует HF-модель в шардированное фрактальное хранилище."""
    if AutoModelForCausalLM is None:
        logger.error("transformers не установлен: невозможно экспортировать HF-модель")
        return False

    try:
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch_device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        torch_dtype = torch.float16 if torch_device == "cuda" else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            hf_model_dir_or_id,
            local_files_only=local_files_only,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype,
        )
        model.to(torch_device)
        model.eval()

        try:
            if hasattr(model, "config") and hasattr(model.config, "to_json_file"):
                model.config.to_json_file(str(out_dir / "config.json"))
        except Exception:
            pass

        store = FractalWeightStore(block_size=block_size, fractal_levels=fractal_levels, device=torch_device)
        if not store.pack_state_dict(model.state_dict(), model_id=model_id):
            return False

        if not store.save_to_disk_sharded(
            str(out_dir),
            knowledge_graph=None,
            shard_size=10000,
            by_level=True,
            compress=True,
        ):
            if not store.save_to_disk_with_recovery(str(out_dir)):
                return False

        try:
            if AutoTokenizer is not None:
                tok = AutoTokenizer.from_pretrained(
                    hf_model_dir_or_id,
                    local_files_only=local_files_only,
                    use_fast=True,
                )
                tok_dir = out_dir / tokenizer_output_subdir
                tok_dir.mkdir(parents=True, exist_ok=True)
                tok.save_pretrained(str(tok_dir))
        except Exception:
            pass

        return True
    except Exception:
        logger.exception("Ошибка экспорта HF-модели в фрактальное хранилище")
        return False