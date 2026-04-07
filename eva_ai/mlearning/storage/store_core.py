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

logger = logging.getLogger("eva_ai.mlearning.fractal_store")


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
            config_device = self.config.get('device', 'cpu') if hasattr(self, 'config') else device
            use_cuda = (config_device != "cpu") if isinstance(config_device, str) else False
            if use_cuda and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        except (RuntimeError, OSError) as e:
            logger.warning(f"Error setting device: {e}")
            self.device = "cpu"

        logger.info(f"FractalWeightStore initialized on {self.device}")

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


# ----------------------- Attach methods from submodules -----------------------
from .store_operations import (
    pack_model_weights, pack_state_dict, get_container, get_tensor,
    save_weights, load_weights, unpack_model_weights, reconstruct_state_dict,
    save_to_disk, load_from_disk, remove_model, has_tensor, get_model_stats,
    _pack_layer_weights, _generate_container_id, _build_fractal_hierarchy,
    _safe_quantize_to_int8, _load_from_disk_atomic_format, _load_legacy_format,
)
from .store_queries import (
    get_statistics, get_similar_tensors, _analyze_container_usage,
    _are_containers_adjacent, _needs_reconfiguration, _calculate_fragmentation,
    _usage_patterns_changed,
)
from .store_cache import (
    _initialize_hot_window, _calculate_container_priority,
    _get_available_hot_window_space, _evict_lowest_priority_containers,
    _load_container_from_ssd, _optimize_fractal_structure,
    _reconfigure_fractal_structure, _determine_new_parameters,
    _extract_all_data, _update_metadata, _save_reconfiguration_history,
)

FractalWeightStore.pack_model_weights = pack_model_weights
FractalWeightStore.pack_state_dict = pack_state_dict
FractalWeightStore.get_container = get_container
FractalWeightStore.get_tensor = get_tensor
FractalWeightStore.save_weights = save_weights
FractalWeightStore.load_weights = load_weights
FractalWeightStore.unpack_model_weights = unpack_model_weights
FractalWeightStore.reconstruct_state_dict = reconstruct_state_dict
FractalWeightStore.save_to_disk = save_to_disk
FractalWeightStore.load_from_disk = load_from_disk
FractalWeightStore.remove_model = remove_model
FractalWeightStore.has_tensor = has_tensor
FractalWeightStore.get_model_stats = get_model_stats
FractalWeightStore._pack_layer_weights = _pack_layer_weights
FractalWeightStore._generate_container_id = _generate_container_id
FractalWeightStore._build_fractal_hierarchy = _build_fractal_hierarchy
FractalWeightStore._safe_quantize_to_int8 = _safe_quantize_to_int8
FractalWeightStore._load_from_disk_atomic_format = _load_from_disk_atomic_format
FractalWeightStore._load_legacy_format = _load_legacy_format
FractalWeightStore.get_statistics = get_statistics
FractalWeightStore.get_similar_tensors = get_similar_tensors
FractalWeightStore._analyze_container_usage = _analyze_container_usage
FractalWeightStore._are_containers_adjacent = _are_containers_adjacent
FractalWeightStore._needs_reconfiguration = _needs_reconfiguration
FractalWeightStore._calculate_fragmentation = _calculate_fragmentation
FractalWeightStore._usage_patterns_changed = _usage_patterns_changed
FractalWeightStore._initialize_hot_window = _initialize_hot_window
FractalWeightStore._calculate_container_priority = _calculate_container_priority
FractalWeightStore._get_available_hot_window_space = _get_available_hot_window_space
FractalWeightStore._evict_lowest_priority_containers = _evict_lowest_priority_containers
FractalWeightStore._load_container_from_ssd = _load_container_from_ssd
FractalWeightStore._optimize_fractal_structure = _optimize_fractal_structure
FractalWeightStore._reconfigure_fractal_structure = _reconfigure_fractal_structure
FractalWeightStore._determine_new_parameters = _determine_new_parameters
FractalWeightStore._extract_all_data = _extract_all_data
FractalWeightStore._update_metadata = _update_metadata
FractalWeightStore._save_reconfiguration_history = _save_reconfiguration_history


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
    torch_dtype = torch.float32
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

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path_str,
                local_files_only=True,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
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

        model = AutoModelForCausalLM.from_pretrained(
            hf_model_dir_or_id,
            local_files_only=local_files_only,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
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
