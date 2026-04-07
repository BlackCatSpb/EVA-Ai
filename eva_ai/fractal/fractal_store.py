"""FractalStore - unified fractal storage interface for ЕВА."""
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

logger = logging.getLogger("eva_ai.fractal.fractal_store")


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
        except (ValueError, TypeError):
            self.priority = 0.0

    def get_memory_size(self) -> int:
        try:
            itemsize = np.dtype(self.dtype).itemsize if isinstance(self.dtype, str) else self.data.dtype.itemsize
        except (TypeError, AttributeError):
            itemsize = self.data.dtype.itemsize
        return int(self.data.size) * int(itemsize)


class FractalStore:
    """Unified fractal storage interface backed by FractalWeightStore."""

    def __init__(
        self,
        block_size: int = 64,
        fractal_levels: int = 5,
        containers_per_group: int = 4,
        device: str = "cpu",
        hot_window_size: int = 500 * 1024 * 1024,
    ) -> None:
        self.block_size = max(1, int(block_size))
        self.fractal_levels = max(1, int(fractal_levels))
        self.containers_per_group = max(1, int(containers_per_group))
        self.hot_window_size = int(hot_window_size)
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.containers: Dict[str, FractalContainer] = {}
        self.fractal_tree: Dict[int, List[str]] = {}
        self.hot_window: OrderedDict[str, float] = OrderedDict()
        self.total_memory: int = 0
        self.model_id: Optional[str] = None
        self.gpu_hot_cache: Dict[str, torch.Tensor] = {}
        self.lazy_index: Dict[str, Any] = {}
        self.graph_metadata: Dict[str, Any] = {}
        self.index: Dict[str, Any] = {}
        self.data_dir: Optional[str] = None
        logger.info(f"FractalStore initialized on {self.device}")

    def pack_model_weights(self, model: torch.nn.Module, model_id: str) -> bool:
        self.containers.clear()
        self.fractal_tree.clear()
        self.hot_window.clear()
        self.total_memory = 0
        self.model_id = model_id
        try:
            if 0 not in self.fractal_tree:
                self.fractal_tree[0] = []
            for layer_name, layer in model.named_modules():
                if not hasattr(layer, 'weight') or layer.weight is None:
                    continue
                self._pack_layer_weights(layer_name, layer.weight.data.cpu().numpy(), model_id)
            self._build_fractal_hierarchy()
            self._initialize_hot_window()
            self._optimize_fractal_structure()
            logger.info(f"FractalStore packed model {model_id}: {len(self.containers)} containers")
            return True
        except Exception:
            logger.exception("Error packing model weights")
            return False

    def pack_state_dict(self, state_dict: Dict[str, torch.Tensor], model_id: str) -> bool:
        self.containers.clear()
        self.fractal_tree.clear()
        self.hot_window.clear()
        self.total_memory = 0
        self.model_id = model_id
        if 0 not in self.fractal_tree:
            self.fractal_tree[0] = []
        try:
            for tpath, tensor in (state_dict or {}).items():
                if not isinstance(tpath, str) or not tpath or tensor is None or not isinstance(tensor, torch.Tensor):
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
                is_critical = any(c in tpath for c in ["wte", "wpe", "ln_f", "lm_head"])
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
            logger.info(f"FractalStore packed state_dict {model_id}: {len(self.containers)} containers")
            return True
        except Exception:
            logger.exception("Error packing state_dict")
            return False

    def get_container(self, container_id: str, load_from_disk: bool = True) -> Optional[FractalContainer]:
        if container_id not in self.containers:
            logger.warning(f"Container {container_id} not found")
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
            return container
        self._evict_lowest_priority_containers(required_space - available_space)
        if required_space <= self._get_available_hot_window_space():
            self.hot_window[container_id] = container.priority
            return container
        return None

    def get_container_data(self, cid: str) -> np.ndarray:
        cont = self.get_container(cid)
        if cont is not None:
            return cont.data
        entry = self.lazy_index.get(cid)
        if not entry:
            raise KeyError(f"Container {cid} not found")
        shard_file = entry["shard_file"]
        key = entry["key"]
        shape = tuple(int(x) for x in (entry.get("shape") or []))
        with np.load(shard_file, allow_pickle=False) as zf:
            arr = zf[key]
        if shape and int(np.prod(shape)) == int(arr.size):
            try:
                arr = arr.reshape(shape)
            except Exception:
                pass
        return arr

    def get_statistics(self) -> Dict[str, Any]:
        total_containers = sum(len(v) for v in self.fractal_tree.values())
        total_memory_mb = self.total_memory / (1024 * 1024) if self.total_memory else 0.0
        containers_by_level = {int(level): int(len(ids)) for level, ids in self.fractal_tree.items()}
        return {
            "model_id": self.model_id,
            "total_containers": total_containers,
            "containers_by_level": containers_by_level,
            "total_memory_bytes": int(self.total_memory),
            "total_memory_mb": float(total_memory_mb),
            "levels": sorted(self.fractal_tree.keys()),
        }

    def clear(self) -> None:
        self.containers.clear()
        self.fractal_tree.clear()
        self.hot_window.clear()
        self.total_memory = 0
        self.model_id = None
        try:
            self.gpu_hot_cache.clear()
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def save_to_disk(self, output_path: str, knowledge_graph: Optional[Dict[str, Any]] = None) -> bool:
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
                containers_jsonl.append({
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
                })
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
            logger.exception("Error saving fractal store")
            return False

    def load_from_disk(self, input_path: str, lazy: bool = False, progress_every: int = 500000) -> bool:
        try:
            in_dir = Path(input_path)
            index_path = in_dir / "index.json"
            containers_path = in_dir / "containers.jsonl"
            data_dir = in_dir / "data"
            shards_manifest = in_dir / "shards_manifest.jsonl"
            if index_path.exists() and data_dir.exists() and (not containers_path.exists()):
                return self._load_from_disk_atomic_format(str(in_dir))
            if shards_manifest.exists():
                self.clear()
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
                        logger.debug("Error parsing index.json for sharded format", exc_info=True)
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
                            logger.exception("Error parsing shards_manifest.jsonl")
                            continue
                        if lazy and progress_every and (line_idx % int(progress_every) == 0):
                            logger.info(f"Indexed lazy: {line_idx} manifest entries...")
                if loaded == 0:
                    logger.error("Failed to load any containers from shards_manifest.jsonl")
                    return False
                try:
                    if not lazy:
                        self._initialize_hot_window()
                except Exception:
                    logger.warning("Failed to initialize hot window after loading", exc_info=True)
                logger.info(f"Loaded (sharded){' (lazy index)' if lazy else ''} containers: {loaded}")
                return True
            return self._load_legacy_format(index_path, containers_path, data_dir)
        except Exception:
            logger.exception("Error loading fractal store")
            return False

    def _load_from_disk_atomic_format(self, source_dir: str) -> Dict[str, Any]:
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
            fractal_tree = index.get("fractal_tree", {})
            self.fractal_tree = {int(k): list(v) for k, v in fractal_tree.items()} if isinstance(fractal_tree, dict) else {}
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
            return {"ok": True}
        except Exception:
            logger.exception("Error loading atomic format")
            return {"ok": False, "error": "atomic_format_error"}

    def _load_legacy_format(self, index_path: Path, containers_path: Path, data_dir: Path) -> bool:
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
            logger.exception("Error loading legacy format")
            return False

    def _pack_layer_weights(self, layer_name: str, weights: np.ndarray, model_id: str) -> None:
        flat_weights = weights.flatten()
        total_elements = len(flat_weights)
        is_critical_layer = any(c in layer_name for c in ["wte", "wpe", "ln_f", "lm_head"])
        storage_dtype = "float64" if is_critical_layer else "float32"
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
                "original_shape": weights.shape,
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

    def _generate_container_id(self, level: int, position: Tuple[int, ...], layer_name: str, model_id: str) -> str:
        return f"{model_id}::{layer_name}::L{level}::pos{'-'.join(map(str, position))}"

    def _build_fractal_hierarchy(self) -> None:
        if 0 not in self.fractal_tree:
            self.fractal_tree[0] = []
        for level in range(1, self.fractal_levels):
            parent_containers = self.fractal_tree.get(level - 1, [])
            if not parent_containers:
                continue
            k = self.containers_per_group
            for i in range(0, len(parent_containers), k):
                group = parent_containers[i: i + k]
                if not group:
                    continue
                grp_index = i // k
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

    def _safe_quantize_to_int8(self, data: np.ndarray) -> Tuple[np.ndarray, float, bool]:
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
        candidates: List[Tuple[str, float]] = []
        top_level = self.fractal_levels - 1
        if top_level in self.fractal_tree:
            for container_id in self.fractal_tree[top_level]:
                priority = self._calculate_container_priority(container_id)
                candidates.append((container_id, priority))
        if 0 in self.fractal_tree:
            for container_id in self.fractal_tree[0]:
                container = self.containers.get(container_id)
                if container and container.metadata.get("is_critical", False):
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

    def _calculate_container_priority(self, container_id: str) -> float:
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
                if cid in self.gpu_hot_cache:
                    self.gpu_hot_cache.pop(cid, None)
            except Exception:
                pass
            freed += size
            if freed >= bytes_needed:
                break

    def _optimize_fractal_structure(self) -> None:
        try:
            for cid in list(self.containers.keys()):
                pr = self._calculate_container_priority(cid)
                c = self.containers.get(cid)
                if c is not None:
                    c.update_priority(pr)
            self._initialize_hot_window()
        except Exception as e:
            logger.warning(f"Error optimizing fractal structure: {e}")
        self._update_metadata()

    def _update_metadata(self) -> None:
        try:
            self.total_memory = 0
            for c in self.containers.values():
                self.total_memory += c.get_memory_size()
        except Exception:
            pass

    def store(self, key: str, data: Any) -> bool:
        try:
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
            container = FractalContainer(
                id=key,
                level=0,
                position=(0,),
                data=np.asarray(data),
                shape=np.asarray(data).shape,
                dtype=str(np.asarray(data).dtype),
            )
            self.containers[key] = container
            return True
        except Exception:
            return False

    def get(self, key: str) -> Optional[Any]:
        cont = self.containers.get(key)
        if cont is not None:
            cont.access_count += 1
            return cont.data
        return None
