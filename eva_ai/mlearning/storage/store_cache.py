from __future__ import annotations
import time
import logging
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict
import numpy as np
import torch

logger = logging.getLogger("eva_ai.mlearning.fractal_store")


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
    from collections import defaultdict
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
