from __future__ import annotations
import time
import logging
import json
from pathlib import Path
import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import OrderedDict, defaultdict
import numpy as np
import torch

logger = logging.getLogger("eva.mlearning.fractal_store")


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

                container = self.__class__.__bases__[0].__bases__[0].__module__  # just use FractalContainer
                from .store_core import FractalContainer
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


def get_container(self, container_id: str, load_from_disk: bool = True) -> Optional[Any]:
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


def reconstruct_state_dict(
    self,
    output_dtype: str = "float32",
    device: str = "cpu",
    limit_tensors: Optional[int] = None,
    include_params: Optional[List[str]] = None,
    resume_from: Optional[str] = None,
    processed_params: Optional[set] = None,
) -> Dict[str, torch.Tensor]:
    """Reconstruct state_dict from stored tensors."""
    if processed_params is None:
        processed_params = set()
    
    state_dict = {}
    tensor_count = 0
    
    try:
        available_tensors = list(self.containers.keys())
        
        for param_name in available_tensors:
            if include_params and param_name not in include_params:
                continue
            if param_name in processed_params:
                continue
            if limit_tensors and tensor_count >= limit_tensors:
                break
            
            try:
                tensor = self.get_tensor(param_name)
                if tensor is None:
                    continue
                
                if output_dtype == "float32":
                    tensor = tensor.float()
                elif output_dtype == "float16":
                    tensor = tensor.half()
                elif output_dtype == "int8":
                    tensor = tensor.char()
                
                if device != "cpu" and tensor.device == torch.device("cpu"):
                    tensor = tensor.to(device)
                
                state_dict[param_name] = tensor
                processed_params.add(param_name)
                tensor_count += 1
                
            except Exception as e:
                logger.warning(f"Error loading parameter {param_name}: {e}")
                continue
        
        logger.info(f"Reconstructed {tensor_count} tensors into state_dict")
        return state_dict
        
    except Exception as e:
        logger.error(f"Error reconstructing state_dict: {e}", exc_info=True)
        return {}


@property
def index(self) -> Dict:
    """Alias for containers to match expected interface."""
    return self.containers


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
                            from .store_core import FractalContainer
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

            from .store_core import FractalContainer
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

                from .store_core import FractalContainer
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


def has_tensor(self, tensor_id: str) -> bool:
    """Check if a tensor exists in storage"""
    return tensor_id in self.containers


def get_model_stats(self, model_id: str) -> Dict[str, Any]:
    """Get statistics for a stored model"""
    model_containers = [cid for cid in self.containers.keys() if model_id in cid]
    total_size = sum(self.containers[cid].get_memory_size() for cid in model_containers if cid in self.containers)
    return {
        "compressed_size": total_size,
        "num_containers": len(model_containers)
    }


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

        from .store_core import FractalContainer
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

            child_data_segments = []
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

            scale_val = None
            zero_scale_flag = False

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

            from .store_core import FractalContainer
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
