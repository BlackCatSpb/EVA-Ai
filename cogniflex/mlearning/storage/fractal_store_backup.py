from __future__ import annotations



import time

import logging

import os

import json

from pathlib import Path

import hashlib

import gc



import numpy as np

import torch

import torch.nn as nn

from typing import Any, Dict, List, Optional, Tuple, Union



logger = logging.getLogger(__name__)



class FractalContainer:

    """��������� ��� ����������� ������"""

    

    def __init__(self, data: np.ndarray, dtype: str = 'float32', priority: float = 1.0):

        self.data = data

        self.dtype = dtype

        self.priority = priority

        self.timestamp = time.time()

        self.access_count = 0

        

    def get_memory_size(self) -> int:

        """���������� ������ ������ � ������"""

        try:

            itemsize = np.dtype(self.dtype).itemsize if isinstance(self.dtype, str) else self.data.dtype.itemsize

        except (TypeError, AttributeError) as e:

            logger.warning(f"Error getting itemsize for dtype {self.dtype}: {e}")

            itemsize = self.data.dtype.itemsize

        return self.data.nbytes * itemsize



class FractalWeightStore:

    """��������� ����� ������ � ����������� ����������"""

    

    def __init__(self, device: str = 'cpu', max_memory_gb: float = 16.0):

        self.device = device

        self.max_memory_bytes = max_memory_gb * 1024**3

        self.containers: Dict[str, FractalContainer] = {}

        self.total_memory = 0

        

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

        

    def store(self, key: str, data: Union[np.ndarray, torch.Tensor], priority: float = 1.0) -> bool:

        """��������� ������ � ���������"""

        try:

            if isinstance(data, torch.Tensor):

                data = data.detach().cpu().numpy()

            

            container = FractalContainer(data, priority=priority)

            memory_size = container.get_memory_size()

            

            if self.total_memory + memory_size > self.max_memory_bytes:

                self._cleanup()

                

            if self.total_memory + memory_size > self.max_memory_bytes:

                logger.warning(f"Not enough memory for {key}")

                return False

                

            self.containers[key] = container

            self.total_memory += memory_size

            logger.debug(f"Stored {key}: {memory_size} bytes")

            return True

            

        except Exception as e:

            logger.error(f"Error storing {key}: {e}")

            return False

            

    def get(self, key: str) -> Optional[np.ndarray]:

        """�������� ������ �� ���������"""

        container = self.containers.get(key)

        if container:

            container.access_count += 1

            return container.data

        return None

        

    def _cleanup(self) -> None:

        """������� ������ ������"""

        if not self.containers:

            return

            

        # ��������� �� ���������� � ������� �������

        sorted_items = sorted(

            self.containers.items(),

            key=lambda x: (x[1].priority, x[1].timestamp, x[1].access_count)

        )

        

        # ������� 25% ����� ������ ������

        to_remove = len(sorted_items) // 4

        for key, container in sorted_items[:to_remove]:

            self.total_memory -= container.get_memory_size()

            del self.containers[key]

            

        logger.info(f"Cleaned up {to_remove} items, freed memory")

        

    def get_memory_usage(self) -> Dict[str, Any]:

        """���������� ���������� ������������� ������"""

        return {

            'total_memory_bytes': self.total_memory,

            'max_memory_bytes': self.max_memory_bytes,

            'items_count': len(self.containers),

            'utilization': self.total_memory / self.max_memory_bytes

        }

    

    def reconstruct_state_dict(self, output_dtype: str = 'float32', device: str = 'cpu') -> Dict[str, Any]:

        """�������������� state_dict �� ������������ ���������"""

        # ���������� ������ state_dict ��� ���� �����������

        state_dict = {}

        

        for key, container in self.containers.items():

            # ������������ ������ � ������ ������

            weights = container.data

            if output_dtype == 'float32':

                weights = weights.astype(np.float32)

            elif output_dtype == 'float16':

                weights = weights.astype(np.float16)

            

            # ������������ � torch tensor

            tensor = torch.from_numpy(weights).to(device)

            state_dict[key] = tensor

        

        return state_dict



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

from typing import Any, Dict, Deque, Tuple, List, Optional, Iterable

from collections import deque, OrderedDict, defaultdict



import numpy as np

import torch

try:

    import psutil  # опционально для мониторинга RAM

except Exception:  # pragma: no cover

    psutil = None  # type: ignore

try:

    # Опциональная зависимость: Transformers для загрузки моделей из директории HF

    from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer

except Exception:  # pragma: no cover

    AutoModelForCausalLM = None  # type: ignore

    AutoModel = None  # type: ignore

    AutoTokenizer = None  # type: ignore



logger = logging.getLogger("cogniflex.mlearning.fractal_store")





# ----------------------- Прокси для графа знаний -----------------------

class NodeProxy:

    def __init__(self, node: Any) -> None:

        # Допускаем как объектные поля, так и словари

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

    """Лёгкий адаптер графа: поддерживает объекты с методами get_all_nodes/get_all_edges

    или словарь вида {"nodes": [...], "edges": [...]}.

    """

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

        except Exception:

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

        except Exception:

            edges = []

        return [EdgeProxy(e) for e in edges]



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

        except Exception:

            self.priority = 0.0



    def get_memory_size(self) -> int:

        try:

            itemsize = np.dtype(self.dtype).itemsize if isinstance(self.dtype, str) else self.data.dtype.itemsize

        except Exception:

            itemsize = self.data.dtype.itemsize

        return int(self.data.size) * int(itemsize)





class FractalWeightStore:

    """Хранилище весов в фрактальной структуре для low-memory режима.



    Основные структуры:

    - containers: плоское хранилище контейнеров (id -> FractalContainer)

    - fractal_tree: иерархическая организация контейнеров (level -> [container_id])

    - hot_window: LRU/скользящее окно часто используемых блоков

    - total_memory: суммарный объём данных (в байтах)

    - model_id: текущая модель, для которой построено хранилище

    - block_size: базовый размер блока (по умолчанию 64 элементов)

    """



    def __init__(self, block_size: int = 64, fractal_levels: int = 5, containers_per_group: int = 4, device: str = "cpu") -> None:

        self.containers: Dict[str, FractalContainer] = {}

        self.fractal_tree: Dict[int, List[str]] = {}

        # храним контейнеры горячего окна как OrderedDict[id -> priority]

        self.hot_window: "OrderedDict[str, float]" = OrderedDict()

        self.total_memory: int = 0

        self.model_id: Optional[str] = None

        self.block_size: int = max(1, int(block_size))

        self.fractal_levels: int = max(1, int(fractal_levels))

        self.containers_per_group: int = max(1, int(containers_per_group))

        # размер горячего окна в байтах (по умолчанию 500 MB)

        self.hot_window_size: int = 500 * 1024 * 1024

        # Специальные токены и оффсет индексов узлов для сериализации графа знаний

        self.SPECIAL_TOKENS: Dict[str, int] = {

            "NODE_START": 1,

            "NODE_END": 2,

            "EDGE_START": 3,

            "EDGE_END": 4,

        }

        # Смещение для индекса узла, чтобы не пересекаться с контентными токенами

        self.NODE_OFFSET: int = 1_000_000

        # Метаданные последнего упакованного графа знаний

        self.graph_metadata: Dict[str, Any] = {}

        # Настройка устройства и GPU-кэша для горячего окна

        try:

            use_cuda = (device != "cpu")

            if use_cuda and torch.cuda.is_available():

                self.device = "cuda"

            else:

                self.device = "cpu"

        except Exception:

            self.device = "cpu"

        self.gpu_hot_cache: Dict[str, "torch.Tensor"] = {}



    # ----------------------- Публичные методы -----------------------

    def pack_model_weights(self, model: torch.nn.Module, model_id: str) -> bool:

        """

        Упаковывает веса модели в фрактальную структуру с учетом 64-битной оптимизации.

        

        Алгоритм:

        1. Очищает существующие данные

        2. Создает контейнеры нулевого уровня для каждого слоя

        3. Строит иерархию фрактала на основе контейнеров нулевого уровня

        4. Инициализирует горячее окно

        5. Выполняет оптимизацию структуры

        

        Args:

            model: Модель PyTorch для упаковки

            model_id: Уникальный идентификатор модели

            

        Returns:

            bool: Успех операции

        """

        start_time = time.time()

        logger.info(f"Начата фрактальная упаковка весов модели {model_id}...")



        try:

            # Очищаем существующие данные

            self.containers.clear()

            self.fractal_tree.clear()

            self.hot_window.clear()

            self.total_memory = 0

            self.model_id = model_id



            # Шаг 1: Создаем контейнеры нулевого уровня для каждого слоя

            logger.debug("Создание контейнеров нулевого уровня...")

            for layer_name, layer in model.named_modules():

                if not hasattr(layer, 'weight') or layer.weight is None:

                    continue

                

                # Упаковываем веса слоя

                self._pack_layer_weights(layer_name, layer.weight.data.cpu().numpy(), model_id)



            # Шаг 2: Создаем более высокие уровни фрактала

            logger.debug("Построение иерархии фрактала...")

            self._build_fractal_hierarchy()



            # Шаг 3: Инициализируем горячее окно

            logger.debug("Инициализация горячего окна...")

            self._initialize_hot_window()



            # Шаг 4: Выполняем оптимизацию структуры

            logger.debug("Выполнение оптимизации структуры...")

            self._optimize_fractal_structure()



            # Шаг 5: Сохраняем статистику

            stats = self.get_statistics()

            logger.info(f"Фрактальная упаковка весов завершена за {time.time() - start_time:.2f} сек. "

                       f"Создано {stats['total_containers']} контейнеров. "

                       f"Общий размер: {stats['total_memory_mb']:.2f} MB. "

                       f"Сжатие: {stats['compression_ratio']:.2f}x")

            return True

            

        except Exception as e:

            logger.error(f"Критическая ошибка фрактальной упаковки весов: {e}", exc_info=True)

            return False



    def pack_state_dict(self, state_dict: Dict[str, torch.Tensor], model_id: str) -> bool:

        """Упаковывает полный state_dict (включая bias и все параметры) в фрактальную структуру.



        Важно: используем tensor_path == ключу state_dict, чтобы reconstruct_state_dict мог

        восстановить оригинальный state_dict без эвристик.

        """

        start_time = time.time()

        logger.info(f"Начата фрактальная упаковка state_dict модели {model_id}...")



        try:

            # Очищаем существующие данные

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



                # Безопасно переносим на CPU numpy

                try:

                    arr = tensor.detach().cpu().numpy()

                except Exception:

                    continue



                # Метаданные, совместимые с reconstruct_state_dict

                try:

                    layer_name, param_name = tpath.rsplit('.', 1)

                except ValueError:

                    layer_name, param_name = tpath, "weight"



                # Уникальный ключ слоя для генерации container_id (иначе weight/bias конфликтуют)

                layer_key = f"{layer_name}.{param_name}" if layer_name else tpath



                flat = arr.reshape(-1)

                total_elements = int(flat.size)

                original_shape = tuple(int(x) for x in arr.shape)



                # dtype хранения

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



            # Строим уровни выше нулевого и инициализируем горячее окно

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



    def cleanup_incremental_artifacts(self, output_path: str, fresh: bool = False) -> Dict[str, Any]:

        """Очищает артефакты незавершённого инкрементального сохранения.



        - Удаляет временные файлы *.tmp в каталоге шардов

        - По умолчанию сохраняет существующие шарды/манифест и лишь удаляет state-файл

        - При fresh=True полностью удаляет shards/, shards_manifest.jsonl и state



        Возвращает отчет: { ok: bool, removed: List[str] }

        """

        base = Path(output_path)

        removed: List[str] = []

        try:

            state_path = base / "incremental_state.json"

            shards_dir = base / "shards"

            manifest_path = base / "shards_manifest.jsonl"



            if fresh:

                if shards_dir.exists():

                    try:

                        shutil.rmtree(shards_dir, ignore_errors=True)

                        removed.append(str(shards_dir))

                    except Exception:

                        pass

                if manifest_path.exists():

                    try:

                        manifest_path.unlink()

                        removed.append(str(manifest_path))

                    except Exception:

                        pass

                if state_path.exists():

                    try:

                        state_path.unlink()

                        removed.append(str(state_path))

                    except Exception:

                        pass

                return {"ok": True, "removed": removed}



            # Нежная очистка: удалить *.tmp и state-файл (по желанию оставляем state? здесь удаляем)

            if shards_dir.exists():

                try:

                    for p in shards_dir.rglob("*.tmp"):

                        try:

                            p.unlink(missing_ok=True)

                            removed.append(str(p))

                        except Exception:

                            continue

                except Exception:

                    pass

            if state_path.exists():

                try:

                    state_path.unlink()

                    removed.append(str(state_path))

                except Exception:

                    pass

            return {"ok": True, "removed": removed}

        except Exception as e:

            return {"ok": False, "error": str(e), "removed": removed}



    def pack_knowledge_graph(self, knowledge_graph: "KnowledgeGraph") -> None:

        """

        Упаковывает граф знаний в фрактальную структуру.



        Алгоритм:

        1) Сериализуем граф в последовательность токенов

        2) Упаковываем последовательность блоками в контейнеры уровня 0

        3) Строим иерархию и инициализируем горячее окно

        4) Сохраняем метаданные графа

        """

        logger.info("Упаковка графа знаний в фрактальную структуру...")

        start_time = time.time()

        try:

            logger.debug("Сериализация графа знаний...")

            serialized = self._serialize_knowledge_graph(KnowledgeGraphProxy(knowledge_graph))



            logger.debug("Упаковка последовательности...")

            self._pack_sequence(serialized)



            logger.debug("Сохранение метаданных графа...")

            self._store_graph_metadata(KnowledgeGraphProxy(knowledge_graph))



            logger.info(f"Граф знаний упакован за {time.time() - start_time:.2f} сек")

        except Exception as e:

            logger.error(f"Ошибка упаковки графа знаний: {e}", exc_info=True)

            raise



    def _serialize_knowledge_graph(self, knowledge_graph: "KnowledgeGraphProxy") -> np.ndarray:

        """

        Сериализует граф знаний в последовательность токенов.



        Формат:

        - Узел: [NODE_START, type_id, content..., NODE_END]

        - Ребро: [EDGE_START, source_idx+NODE_OFFSET, target_idx+NODE_OFFSET, relation_id, EDGE_END]

        """

        nodes = knowledge_graph.get_all_nodes()

        edges = knowledge_graph.get_all_edges()



        node_to_index = {node.id: i for i, node in enumerate(nodes)}



        seq_len = 0

        for node in nodes:

            seq_len += 3 + len(self._encode_content(node.content))

        for _ in edges:

            seq_len += 5



        serialized = np.zeros(seq_len, dtype=np.int32)

        pos = 0



        # Узлы

        for node in nodes:

            serialized[pos] = self.SPECIAL_TOKENS["NODE_START"]; pos += 1

            serialized[pos] = self._encode_node_type(node.node_type); pos += 1

            content_tokens = self._encode_content(node.content)

            if content_tokens.size:

                end = pos + int(content_tokens.size)

                serialized[pos:end] = content_tokens

                pos = end

            serialized[pos] = self.SPECIAL_TOKENS["NODE_END"]; pos += 1



        # Рёбра

        for edge in edges:

            serialized[pos] = self.SPECIAL_TOKENS["EDGE_START"]; pos += 1

            serialized[pos] = node_to_index.get(edge.source, -1) + self.NODE_OFFSET; pos += 1

            serialized[pos] = node_to_index.get(edge.target, -1) + self.NODE_OFFSET; pos += 1

            serialized[pos] = self._encode_relation(edge.relation_type); pos += 1

            serialized[pos] = self.SPECIAL_TOKENS["EDGE_END"]; pos += 1



        return serialized



    def _pack_sequence(self, sequence: np.ndarray) -> None:

        """Упаковывает последовательность в фрактальную структуру контейнеров."""

        # Очистка текущего состояния

        self.containers.clear()

        self.fractal_tree.clear()

        self.total_memory = 0



        # Нулевой уровень — поблочная упаковка

        for i in range(0, int(sequence.size), self.block_size):

            block = sequence[i:i + self.block_size]

            shape = (int(block.size),)

            position = (i // self.block_size,)

            # Используем текущий model_id, если он задан, иначе 'fractal'

            cid = self._generate_container_id(0, position, "knowledge_graph", self.model_id or "fractal")

            container = FractalContainer(

                id=cid,

                level=0,

                position=position,

                data=block.astype(np.int32, copy=False),

                shape=shape,

                dtype="int32",

                metadata={

                    "block_start": i,

                    "block_end": min(i + self.block_size, int(sequence.size)),

                    "source": "knowledge_graph",

                },

            )

            self.containers[cid] = container

            self.fractal_tree.setdefault(0, []).append(cid)

            self.total_memory += container.get_memory_size()



        # Строим уровни выше нулевого и инициализируем горячее окно

        self._build_fractal_hierarchy()

        self._initialize_hot_window()



    def _store_graph_metadata(self, knowledge_graph: "KnowledgeGraphProxy") -> None:

        """Сохраняет легковесные метаданные о структуре графа."""

        nodes = knowledge_graph.get_all_nodes()

        edges = knowledge_graph.get_all_edges()

        self.graph_metadata = {

            "node_count": len(nodes),

            "edge_count": len(edges),

            "types": list(sorted({getattr(n, "node_type", "unknown") for n in nodes})),

        }



    # ----------------------- Простейшие кодировщики -----------------------

    def _encode_node_type(self, node_type: str) -> int:

        table = {

            "token": 10,

            "ffn": 11,

            "concept": 12,

            "layer": 13,

            "unknown": 14,

        }

        if not isinstance(node_type, str):

            return table["unknown"]

        return table.get(node_type, 100 + (abs(hash(node_type)) % 1000))



    def _encode_relation(self, relation: str) -> int:

        table = {

            "attn": 20,

            "modulates": 21,

            "derives": 22,

            "correlates": 23,

        }

        if not isinstance(relation, str):

            return 24

        return table.get(relation, 200 + (abs(hash(relation)) % 1000))



    def _encode_content(self, content: Any) -> np.ndarray:

        """Лёгкая кодировка содержимого узла в последовательность int32 без внешних зависимостей."""

        tokens: List[int] = []

        try:

            if content is None:

                return np.zeros(0, dtype=np.int32)

            if isinstance(content, (int, np.integer)):

                tokens.append(int(content) % 1_000_000)

            elif isinstance(content, float):

                tokens.append(int(abs(content) * 1e6) % 1_000_000)

            elif isinstance(content, str):

                # Простая символьная кодировка с отсечением

                for ch in content[:64]:

                    tokens.append(ord(ch) % 1024)

            elif isinstance(content, dict):

                # Упорядоченная пара ключ-значение

                for k in sorted(content.keys()):

                    kv = f"{k}:{content[k]}"

                    for ch in str(kv)[:64]:

                        tokens.append(ord(ch) % 1024)

            elif isinstance(content, (list, tuple)):

                for x in content[:32]:

                    tokens.extend(list(self._encode_content(x)))

            else:

                # Фоллбек: хэш строки представления

                tokens.append(300 + (abs(hash(str(content))) % 10_000))

        except Exception:

            # Максимально устойчивое поведение в low-memory

            tokens.append(999_999)

        return np.array(tokens, dtype=np.int32)



    # ----------------------- Вспомогательные методы -----------------------

    def _pack_layer_weights(self, layer_name: str, weights: np.ndarray, model_id: str) -> None:

        """

        Упаковывает веса слоя в фрактальную структуру на уровне 0.

        

        Математическая основа:

        - Базовый размер блока B = 64 элемента

        - Для слоя с N элементами создается C₀ = ⌈N/B⌉ контейнеров

        - Каждый контейнер содержит S₀ = B элементов

        

        Оптимизация:

        - Для критически важных слоев (embedding, выходные слои) используем float64

        - Для остальных слоев используем float32

        

        Args:

            layer_name: Имя слоя

            weights: Веса слоя в виде numpy массива

            model_id: ID модели

        """

        dtype = str(weights.dtype)

        shape = weights.shape

        flat_weights = weights.flatten()

        total_elements = len(flat_weights)

        

        # Определяем, является ли слой критически важным

        is_critical_layer = any(critical in layer_name for critical in 

                               ["wte", "wpe", "ln_f", "lm_head"])

        

        # Выбираем тип данных на основе важности слоя

        storage_dtype = "float64" if is_critical_layer else "float32"

        

        # Создаем контейнеры нулевого уровня

        logger.debug(f"Упаковка слоя {layer_name} ({shape}) в {storage_dtype}...")

        

        for i in range(0, total_elements, self.block_size):

            block_data = flat_weights[i:i + self.block_size]

            

            # Конвертируем в нужный тип данных

            if storage_dtype == "float64":

                block_data = block_data.astype(np.float64)

            else:

                block_data = block_data.astype(np.float32)

                

            block_shape = (len(block_data),)

            

            # Создаем ID контейнера

            position = (i // self.block_size,)

            container_id = self._generate_container_id(0, position, layer_name, model_id)

            

            # Создаем метаданные

            metadata = {

                "layer_name": layer_name,

                "model_id": model_id,

                "original_shape": shape,

                "block_start": i,

                "block_end": min(i + self.block_size, total_elements),

                "is_critical": is_critical_layer,

                "storage_dtype": storage_dtype,

                # Добавляем точные указатели на параметр

                "param_name": "weight",

                "tensor_path": f"{layer_name}.weight",

            }

            

            # Создаем контейнер

            container = FractalContainer(

                id=container_id,

                level=0,

                position=position,

                data=block_data,

                shape=block_shape,

                dtype=storage_dtype,

                metadata=metadata

            )

            

            # Сохраняем контейнер

            self.containers[container_id] = container

            if 0 not in self.fractal_tree:

                self.fractal_tree[0] = []

            self.fractal_tree[0].append(container_id)

            self.total_memory += container.get_memory_size()

            

            # Логируем каждые 1000 контейнеров для отслеживания прогресса

            if len(self.fractal_tree[0]) % 1000 == 0:

                logger.debug(f"Создано {len(self.fractal_tree[0])} контейнеров для уровня 0")



    def _generate_container_id(

        self, level: int, position: Tuple[int, ...], layer_name: str, model_id: str

    ) -> str:

        return f"{model_id}::{layer_name}::L{level}::pos{'-'.join(map(str, position))}"



    def _safe_quantize_to_int8(self, data: np.ndarray) -> Tuple[np.ndarray, float, bool]:

        """

        Безопасное квантование данных в int8 с обработкой NaN/Inf и нулевого масштаба.



        Returns:

            quantized: np.ndarray np.int8 той же формы

            scale: float масштаб квантования (>= 1e-12, либо 1.0 при нулевом массиве)

            has_zero_scale: bool признак, что весь блок был нулевым

        """

        # Заменяем NaN/Inf на безопасные значения и ограничиваем экстремумы

        arr = np.nan_to_num(data, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)

        if arr.size == 0:

            return arr.astype(np.int8, copy=False), 1.0, False



        max_abs = float(np.max(np.abs(arr)))

        if max_abs == 0.0:

            # Весь блок нулевой

            return np.zeros(arr.shape, dtype=np.int8), 1.0, True



        scale = max(max_abs / 127.0, 1e-12)

        normalized = arr / scale

        # Обрезаем и округляем, затем кастим в int8

        quantized = np.clip(np.round(normalized), -127, 127).astype(np.int8, copy=False)

        return quantized, scale, False



    def _build_fractal_hierarchy(self) -> None:

        """

        Строит иерархию фрактала на основе контейнеров нулевого уровня.



        Математическая основа:

        - Количество контейнеров на уровне i: C_i = ceil(C_{i-1}/k)

        - Размер контейнера на уровне i: S_i = k^i * B

        - Где k = self.containers_per_group, B = self.block_size



        Типы данных:

        - уровень 1: float32

        - уровень 2: float16

        - уровни >=3: int8 с масштабом (quant_scale)

        """

        if 0 not in self.fractal_tree:

            self.fractal_tree[0] = []



        logger.info(f"Построение фрактальной иерархии ({self.fractal_levels} уровней)...")



        for level in range(1, self.fractal_levels):

            parent_containers = self.fractal_tree.get(level - 1, [])

            if not parent_containers:

                # Нечего агрегировать

                continue

            logger.debug(f"Построение уровня {level} из {len(parent_containers)} контейнеров...")



            k = self.containers_per_group

            total_groups = (len(parent_containers) + k - 1) // k

            progress_interval = max(1, total_groups // 10)



            for i in range(0, len(parent_containers), k):

                group = parent_containers[i : i + k]

                if not group:

                    continue



                # Прогресс

                grp_index = i // k

                if grp_index % progress_interval == 0:

                    progress = (grp_index + 1) / total_groups * 100.0

                    logger.debug(f"Уровень {level}: {progress:.1f}% завершено")



                position = (grp_index,)



                # Базовые метаданные берём из первого ребёнка

                first_child = self.containers[group[0]]

                layer_name = first_child.metadata.get("layer_name", "")

                model_id = first_child.metadata.get("model_id", self.model_id or "")

                container_id = self._generate_container_id(level, position, layer_name, model_id)



                # Собираем данные

                child_data_segments: List[np.ndarray] = []

                for child_id in group:

                    child = self.containers[child_id]

                    child_data_segments.append(child.data)

                    child.parent = container_id



                try:

                    combined_data = np.concatenate(child_data_segments, axis=0)

                except Exception:

                    # В редких случаях несовпадения формы — приводим к вектору

                    combined_data = np.concatenate([seg.reshape(-1) for seg in child_data_segments], axis=0)



                # Определяем dtype на текущем уровне

                if level == 1:

                    storage_dtype = "float32"

                elif level == 2:

                    storage_dtype = "float16"

                else:

                    storage_dtype = "int8"



                scale_val: Optional[float] = None

                zero_scale_flag: bool = False

                if storage_dtype == "float32":

                    # Защита от NaN/Inf и явное приведение к fp32

                    f32_max = np.finfo(np.float32).max

                    f32_min = np.finfo(np.float32).min

                    arr = np.nan_to_num(combined_data, copy=False, nan=0.0, posinf=f32_max, neginf=f32_min)

                    combined_data = arr.astype(np.float32, copy=False)

                elif storage_dtype == "float16":

                    # Устойчивое приведение к fp16: используем значения, совместимые с диапазоном float16

                    # float16 max ~ 65504, выберем немного меньше для запаса

                    f16_max = 6.5e4

                    f16_min = -6.5e4

                    arr = np.nan_to_num(combined_data, copy=False, nan=0.0, posinf=f16_max, neginf=f16_min)

                    arr = np.clip(arr, f16_min, f16_max)

                    combined_data = arr.astype(np.float16, copy=False)

                elif storage_dtype == "int8":

                    # Устойчивое квантование через хелпер

                    quant_f, scale_val, zero_scale_flag = self._safe_quantize_to_int8(combined_data)

                    combined_data = quant_f



                metadata = {

                    "layer_name": layer_name,

                    "model_id": model_id,

                    "child_count": len(group),

                    "child_ids": group,

                    "storage_dtype": storage_dtype,

                }

                # Протягиваем точные указатели на параметр из первого потомка, если доступны

                if isinstance(first_child, FractalContainer):

                    tp = first_child.metadata.get("tensor_path")

                    if tp is not None:

                        metadata["tensor_path"] = tp

                    pn = first_child.metadata.get("param_name")

                    if pn is not None:

                        metadata["param_name"] = pn

                if scale_val is not None:

                    # Дублируем ключ для совместимости с внешними сценариями

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



    def _initialize_hot_window(self) -> None:

        """

        Инициализирует горячее окно на основе фрактальной структуры.

        Стратегия: верхний уровень, часть средних, критические слои уровня 0.

        Приоритет: P = L_p * T_p * C_p

        """

        logger.debug("Инициализация горячего окна фрактальной памяти...")



        candidates: List[Tuple[str, float]] = []



        # 1) Самый высокий уровень

        top_level = self.fractal_levels - 1

        if top_level in self.fractal_tree:

            for container_id in self.fractal_tree[top_level]:

                priority = self._calculate_container_priority(container_id)

                candidates.append((container_id, priority))



        # 2) Средние уровни (берём крайние несколько)

        if self.fractal_levels > 2:

            mid_level = self.fractal_levels // 2

            if mid_level in self.fractal_tree and self.fractal_tree[mid_level]:

                mids = self.fractal_tree[mid_level]

                sample_ids = mids[:3] + (mids[-3:] if len(mids) > 3 else [])

                for container_id in sample_ids:

                    priority = self._calculate_container_priority(container_id)

                    candidates.append((container_id, priority))



        # 3) Критические блоки уровня 0

        if 0 in self.fractal_tree:

            for container_id in self.fractal_tree[0]:

                container = self.containers[container_id]

                if container.metadata.get("is_critical", False):

                    priority = self._calculate_container_priority(container_id)

                    candidates.append((container_id, priority))



        candidates.sort(key=lambda x: x[1], reverse=True)



        # Заполнение окна до лимита

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

                # Материализуем данные контейнера в GPU-кэш при необходимости

                try:

                    if getattr(self, "device", "cpu") == "cuda" and torch.cuda.is_available():

                        if container_id not in self.gpu_hot_cache:

                            np_arr = container.data  # np.ndarray

                            if container.dtype == "float8":

                                tensor = torch.from_numpy(np_arr.astype(np.int8, copy=False))

                            elif container.dtype == "float16":

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



        # Если менее 50% — пробуем расширить за счёт других уровней

        if hot_window_size_mb < (self.hot_window_size * 0.5) / (1024 * 1024):

            logger.warning("Горячее окно заполнено менее чем на 50%. Добавляем дополнительные контейнеры...")

            self._expand_hot_window(current_size)



    def _calculate_container_priority(self, container_id: str) -> float:

        """P = L_p * T_p * C_p, где:

        L_p = 0.9^level; T_p = 0.1 + 0.9*(1 - min(t/3600,1)); C_p = 0.3 для критических, иначе 1.0

        t — время с последнего доступа в секундах.

        """

        cont = self.containers.get(container_id)

        if cont is None:

            return 0.0

        level_factor = pow(0.9, max(0, int(cont.level)))

        t = max(0.0, time.time() - float(cont.last_accessed or 0.0))

        time_factor = 0.1 + 0.9 * (1.0 - min(t / 3600.0, 1.0))

        crit = 0.3 if cont.metadata.get("is_critical", False) else 1.0

        return float(level_factor * time_factor * crit)



    def _expand_hot_window(self, current_size: int) -> None:

        """Добираем контейнеры из остальных уровней по убыванию приоритета до 50% окна или исчерпания."""

        if current_size >= self.hot_window_size * 0.5:

            return

        # Собираем всех кандидатов, которых ещё нет в окне

        seen = set(self.hot_window.keys())

        candidates: List[Tuple[str, float]] = []

        for level, ids in sorted(self.fractal_tree.items(), key=lambda x: x[0], reverse=True):

            for cid in ids:

                if cid in seen:

                    continue

                pr = self._calculate_container_priority(cid)

                candidates.append((cid, pr))

        candidates.sort(key=lambda x: x[1], reverse=True)



        size = current_size

        now_ts = time.time()

        for cid, pr in candidates:

            cont = self.containers.get(cid)

            if cont is None:

                continue

            csz = cont.get_memory_size()

            if size + csz > self.hot_window_size:

                continue

            self.hot_window[cid] = float(pr)

            size += csz

            cont.last_accessed = now_ts

            cont.access_count += 1

            # Материализация на GPU

            try:

                if getattr(self, "device", "cpu") == "cuda" and torch.cuda.is_available():

                    if cid not in self.gpu_hot_cache:

                        np_arr = cont.data

                        if cont.dtype == "float8":

                            tensor = torch.from_numpy(np_arr.astype(np.int8, copy=False))

                        elif cont.dtype == "float16":

                            tensor = torch.from_numpy(np_arr.astype(np.float16, copy=False))

                        elif cont.dtype == "float32":

                            tensor = torch.from_numpy(np_arr.astype(np.float32, copy=False))

                        else:

                            tensor = torch.from_numpy(np_arr)

                        self.gpu_hot_cache[cid] = tensor.to("cuda", non_blocking=True)

            except Exception:

                pass

            if size >= self.hot_window_size * 0.5:

                break



    def _optimize_fractal_structure(self) -> None:

        """

        Оптимизирует фрактальную структуру для повышения эффективности.



        Алгоритм:

        1. Анализирует использование контейнеров

        2. Перестраивает фрактальную иерархию при необходимости

        3. Оптимизирует расположение контейнеров

        4. Обновляет метаданные



        Особенности:

        - Динамическая реконфигурация структуры

        - Оптимизация под текущие паттерны использования

        - Минимизация фрагментации

        - Самообучение системы



        Математическая основа:

        - Коэффициент пространственной локальности: L_spatial = число последовательных доступов / общее число доступов

        - Оптимальная структура максимизирует L_spatial

        """

        logger.info("Оптимизация фрактальной структуры...")



        # 1. Анализируем использование контейнеров

        usage_stats = self._analyze_container_usage()



        # 2. Проверяем, нужна ли реконфигурация

        if self._needs_reconfiguration(usage_stats):

            logger.info("Требуется реконфигурация фрактальной структуры...")

            self._reconfigure_fractal_structure(usage_stats)



        # 3. Оптимизируем расположение контейнеров (реалайн в рамках текущей иерархии)

        logger.info("Оптимизация расположения контейнеров...")

        try:

            # Простая эвристика: пересчитать приоритеты и перезаполнить горячее окно

            for cid in list(self.containers.keys()):

                pr = self._calculate_container_priority(cid)

                c = self.containers.get(cid)

                if c is not None:

                    c.update_priority(pr)

            self._initialize_hot_window()

        except Exception as e:

            logger.warning(f"Не удалось оптимизировать расположение контейнеров: {e}")



        # 4. Обновляем метаданные

        logger.info("Обновление метаданных...")

        self._update_metadata()



    def _analyze_container_usage(self) -> Dict[str, Any]:

        """

        Анализирует использование контейнеров.



        Собирает статистику:

        - Частота доступа

        - Пространственная локальность

        - Временная локальность

        - Паттерны использования



        Returns:

            Dict[str, Any]: Статистика использования

        """

        stats: Dict[str, Any] = {

            "access_frequency": defaultdict(int),

            "spatial_locality": 0.0,

            "temporal_locality": 0.0,

            "access_pattern": [],

            "last_access": {},

        }



        now_ts = time.time()

        # Собираем статистику по всем контейнерам

        for container_id, container in self.containers.items():

            # Частота доступа

            stats["access_frequency"][container_id] = int(getattr(container, "access_count", 0))



            # Последнее время доступа

            stats["last_access"][container_id] = float(getattr(container, "last_accessed", 0.0))



            # Добавляем в историю доступа (за последний час)

            if float(container.last_accessed or 0.0) > now_ts - 3600.0:

                stats["access_pattern"].append(container_id)



        # Вычисляем пространственную локальность

        if len(stats["access_pattern"]) > 1:

            sequential_accesses = 0

            for i in range(1, len(stats["access_pattern"])):

                prev = stats["access_pattern"][i - 1]

                curr = stats["access_pattern"][i]

                if self._are_containers_adjacent(prev, curr):

                    sequential_accesses += 1

            stats["spatial_locality"] = sequential_accesses / float(len(stats["access_pattern"]) - 1)



        # Вычисляем временную локальность: повторные доступы к тем же контейнерам за 5 минут

        recent = [cid for cid, ts in stats["last_access"].items() if (now_ts - float(ts)) < 300.0]

        if recent:

            unique_recent = len(set(recent))

            stats["temporal_locality"] = 1.0 - (unique_recent / float(len(recent)))



        return stats



    def _are_containers_adjacent(self, cid_a: str, cid_b: str) -> bool:

        """Соседство: один уровень, один слой, позиция отличается на 1 либо общий родитель.

        Падает в более широкую эвристику при отсутствии метаданных.

        """

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

            # Общий родитель в иерархии

            if a.parent and a.parent == b.parent and a.parent is not None:

                return True

        except Exception:

            return False

        return False



    def _needs_reconfiguration(self, usage_stats: Dict[str, Any]) -> bool:

        """

        Определяет, требуется ли реконфигурация фрактальной структуры.



        Критерии:

        - Низкая пространственная локальность (< 0.4)

        - Низкая временная локальность (< 0.3)

        - Высокая фрагментация

        - Изменение паттернов использования

        """

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

        """Оценка фрагментации по «дырам» в последовательностях позиций для каждого слоя и уровня 0.

        Возвращает среднее относительное число разрывов.

        """

        try:

            per_layer: List[float] = []

            # Анализируем уровень 0, где последовательность блоков слоя критична

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

        """Сравнивает текущую сводку использования с предыдущей.

        Если изменение метрик выше порога — считаем, что паттерны изменились.

        """

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

        """

        Перестраивает фрактальную структуру на основе статистики использования.



        Алгоритм:

        1. Определяет новые параметры фрактала

        2. Перепаковывает данные с новыми параметрами

        3. Сохраняет историю изменений

        """

        # Определяем новые параметры

        new_params = self._determine_new_parameters(usage_stats)



        logger.info(

            f"Реконфигурация фрактальной структуры: уровни={new_params['fractal_levels']}, блок={new_params['block_size']}"

        )



        # Сохраняем текущие данные (восстановление слоев из уровня 0)

        all_data = self._extract_all_data()



        # Очищаем текущую структуру

        self.containers.clear()

        self.fractal_tree.clear()

        self.hot_window.clear()

        self.total_memory = 0



        # Обновляем параметры

        self.fractal_levels = int(new_params["fractal_levels"])

        self.block_size = int(new_params["block_size"])



        # Перепаковываем данные

        model_id = self.model_id or "fractal"

        for layer_name, weights in all_data.items():

            try:

                self._pack_layer_weights(layer_name, weights, model_id)

            except Exception as e:

                logger.error(f"Не удалось перепаковать слой {layer_name}: {e}")



        # Строим новую иерархию и горячее окно

        self._build_fractal_hierarchy()

        self._initialize_hot_window()



        # Сохраняем историю изменений

        self._save_reconfiguration_history(new_params)



        logger.info("Фрактальная структура успешно реконфигурирована")



    def _determine_new_parameters(self, usage_stats: Dict[str, Any]) -> Dict[str, Any]:

        """Определяет новые параметры фрактала на основе статистики использования."""

        new_params = {

            "fractal_levels": int(self.fractal_levels),

            "block_size": int(self.block_size),

        }



        # Регулируем количество уровней (пространственная локальность)

        spatial = float(usage_stats.get("spatial_locality", 0.0))

        if spatial < 0.4:

            new_params["fractal_levels"] = min(6, int(self.fractal_levels) + 1)

        elif spatial > 0.6:

            new_params["fractal_levels"] = max(3, int(self.fractal_levels) - 1)



        # Регулируем размер блока (временная локальность)

        temporal = float(usage_stats.get("temporal_locality", 0.0))

        if temporal < 0.3:

            new_params["block_size"] = max(16, int(self.block_size) // 2)

        elif temporal > 0.5:

            new_params["block_size"] = min(128, int(self.block_size) * 2)



        return new_params



    def _extract_all_data(self) -> Dict[str, np.ndarray]:

        """Восстанавливает тензоры слоёв из контейнеров уровня 0 по layer_name и original_shape."""

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

                result[layer] = arr  # фоллбек

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

        """Сохраняет запись об изменении конфигурации в памяти (и при наличии — на диск)."""

        try:

            rec = {

                "ts": time.time(),

                "params": dict(params),

                "stats": self.get_statistics() if hasattr(self, "get_statistics") else {},

            }

            hist = getattr(self, "reconfiguration_history", None)

            if not isinstance(hist, list):

                self.reconfiguration_history = []  # type: ignore[attr-defined]

            self.reconfiguration_history.append(rec)  # type: ignore[attr-defined]

        except Exception:

            pass



    # ----------------------- Доступ к контейнерам -----------------------

    def get_container(self, container_id: str, load_from_disk: bool = True) -> Optional[FractalContainer]:

        """

        Получает контейнер по ID с динамической загрузкой в горячее окно.

        """

        if container_id not in self.containers:

            logger.warning(f"Контейнер {container_id} не найден")

            return None



        container = self.containers[container_id]



        # Обновляем статистику доступа

        container.last_accessed = time.time()

        container.access_count += 1



        # Если уже в горячем окне — возвращаем

        if container_id in self.hot_window:

            return container



        # Приоритет

        base_priority = self._calculate_container_priority(container_id)

        container.update_priority(base_priority)



        required_space = container.get_memory_size()

        available_space = self._get_available_hot_window_space()



        if required_space <= available_space:

            self.hot_window[container_id] = container.priority

            logger.debug(

                f"Контейнер {container_id} добавлен в горячее окно (уровень {container.level})"

            )

            return container



        logger.debug(

            f"Недостаточно места в горячем окне для контейнера {container_id}. Вытеснение..."

        )

        self._evict_lowest_priority_containers(required_space - available_space)



        if required_space <= self._get_available_hot_window_space():

            self.hot_window[container_id] = container.priority

            logger.debug(

                f"Контейнер {container_id} добавлен в горячее окно после вытеснения (уровень {container.level})"

            )

            return container



        if load_from_disk and ("offload_path" in container.metadata):

            logger.debug(f"Загрузка контейнера {container_id} с диска...")

            if self._load_container_from_ssd(container_id):

                # Повторная попытка

                return self.get_container(container_id, load_from_disk=False)



        logger.warning(f"Не удалось добавить контейнер {container_id} в горячее окно")

        return None



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

        # Сортируем текущие элементы окна по возрастанию приоритета

        items = list(self.hot_window.items())

        items.sort(key=lambda x: x[1])

        freed = 0

        for cid, pr in items:

            cont = self.containers.get(cid)

            size = cont.get_memory_size() if cont is not None else 0

            # удаляем из окна

            self.hot_window.pop(cid, None)

            # удаляем из GPU-кэша, если присутствует

            try:

                if hasattr(self, "gpu_hot_cache") and cid in self.gpu_hot_cache:

                    self.gpu_hot_cache.pop(cid, None)

            except Exception:

                pass

            freed += size

            if freed >= bytes_needed:

                break



    def _load_container_from_ssd(self, container_id: str) -> bool:

        """Загружает данные контейнера из offload-пути, если доступно. Заглушка."""

        cont = self.containers.get(container_id)

        if cont is None:

            return False

        off_path = cont.metadata.get("offload_path")

        if not off_path:

            return False

        try:

            # Здесь могла бы быть mmap/np.load. Пока заглушка: возвращаем False, чтобы не менять поведение.

            return False

        except Exception:

            logger.exception(f"Ошибка загрузки контейнера {container_id} с диска")

            return False



    def get_statistics(self) -> Dict[str, Any]:

        total_containers = sum(len(v) for v in self.fractal_tree.values())

        total_memory_mb = self.total_memory / (1024 * 1024) if self.total_memory else 0.0

        containers_by_level = {int(level): int(len(ids)) for level, ids in self.fractal_tree.items()}

        # Без факта реального сжатия пока считаем коэффициент 1.0

        stats = {

            "model_id": self.model_id,

            "total_containers": total_containers,

            "containers_by_level": containers_by_level,

            "total_memory_bytes": int(self.total_memory),

            "total_memory_mb": float(total_memory_mb),

            "compression_ratio": 1.0,

            "levels": sorted(self.fractal_tree.keys()),

        }

        return stats



    # ----------------------- Диск / транзакции / валидация -----------------------

    def compute_checksum(self) -> str:

        """Возвращает агрегированный SHA256 всех контейнеров (id + data) для контроля целостности."""

        import hashlib

        h = hashlib.sha256()

        try:

            # Фиксированный порядок: по уровням и id

            for level in sorted(self.fractal_tree.keys()):

                for cid in sorted(self.fractal_tree[level]):

                    cont = self.containers.get(cid)

                    if cont is None:

                        continue

                    h.update(cid.encode("utf-8", errors="ignore"))

                    try:

                        h.update(cont.data.tobytes())

                    except Exception:

                        # Фоллбек через копию в np.ndarray

                        h.update(np.array(cont.data).tobytes())

            return h.hexdigest()

        except Exception:

            return ""



    def validate_knowledge_graph_packing(self) -> Dict[str, Any]:

        """Валидирует упаковку графа знаний на уровне 0: проверяет непрерывность блоков и метки источника.



        Returns:

            { ok: bool, issues: List[str], total_blocks: int, total_length: int }

        """

        issues: List[str] = []

        try:

            # Выбираем блоки уровня 0, созданные pack_knowledge_graph (source == 'knowledge_graph')

            blocks: List[Tuple[int, int]] = []

            for cid in self.fractal_tree.get(0, []):

                c = self.containers.get(cid)

                if c is None:

                    continue

                if c.metadata.get("source") == "knowledge_graph":

                    bs = int(c.metadata.get("block_start", -1))

                    be = int(c.metadata.get("block_end", -1))

                    if bs < 0 or be <= bs:

                        issues.append(f"invalid_block_meta:{cid}:{bs}-{be}")

                    else:

                        blocks.append((bs, be))



            if not blocks:

                issues.append("no_kg_blocks_found")

                return {"ok": False, "issues": issues, "total_blocks": 0, "total_length": 0}



            # Проверяем непрерывность диапазонов [0..max_end)

            blocks.sort(key=lambda x: x[0])

            expected = 0

            for (bs, be) in blocks:

                if bs != expected:

                    issues.append(f"gap_or_overlap_at:{expected}->{bs}")

                    expected = be

                else:

                    expected = be

            total_len = blocks[-1][1]

            ok = len([i for i in issues if i.startswith("gap_or_overlap")]) == 0

            return {"ok": ok and len(issues) == 0, "issues": issues, "total_blocks": len(blocks), "total_length": total_len}

        except Exception as e:

            return {"ok": False, "issues": [f"exception:{e}"], "total_blocks": 0, "total_length": 0}



    def save_to_disk_atomic(self, target_dir: str) -> Dict[str, Any]:

        """Сохраняет состояние фрактального хранилища на диск атомарно.



        Структура директории:

          target_dir/

            index.json             # метаданные, дерево, графовые метаданные, контрольная сумма

            data/<cid>.npy         # данные контейнеров уровня 0..N



        Возвращает отчёт: { ok: bool, path: str, checksum: str, error?: str }

        """

        import tempfile

        import shutil



        try:

            target = Path(target_dir)

            tmp_dir = Path(tempfile.mkdtemp(prefix="fractal_store_"))

            data_dir = tmp_dir / "data"

            data_dir.mkdir(parents=True, exist_ok=True)



            # Пишем контейнеры в .npy

            for level in sorted(self.fractal_tree.keys()):

                for cid in self.fractal_tree[level]:

                    cont = self.containers.get(cid)

                    if cont is None:

                        continue

                    fn = data_dir / f"{self._safe_name(cid)}.npy"

                    try:

                        np.save(fn, cont.data, allow_pickle=False)

                    except Exception:

                        # Фоллбек через np.array

                        np.save(fn, np.array(cont.data), allow_pickle=False)



            # Индекс

            index = {

                "model_id": self.model_id,

                "fractal_levels": int(self.fractal_levels),

                "block_size": int(self.block_size),

                "containers": [],

                "fractal_tree": {str(k): list(v) for k, v in self.fractal_tree.items()},

                "graph_metadata": dict(self.graph_metadata),

            }

            for cid, c in self.containers.items():

                index["containers"].append(

                    {

                        "id": cid,

                        "level": int(c.level),

                        "position": list(c.position),

                        "shape": list(c.shape),

                        "dtype": str(c.dtype),

                        "metadata": c.metadata,

                        "file": f"data/{self._safe_name(cid)}.npy",

                    }

                )



            checksum = self.compute_checksum()

            index["checksum"] = checksum



            with open(tmp_dir / "index.json", "w", encoding="utf-8") as f:

                json.dump(index, f, ensure_ascii=False, indent=2)



            # Атомарная замена: переносим tmp_dir -> target_dir

            if target.exists():

                backup = target.with_suffix(".bak")

                try:

                    if backup.exists():

                        shutil.rmtree(backup, ignore_errors=True)

                    target.rename(backup)

                except Exception:

                    pass

                try:

                    shutil.rmtree(target, ignore_errors=True)

                except Exception:

                    pass

            shutil.move(str(tmp_dir), str(target))

            # Чистим возможный backup

            backup = target.with_suffix(".bak")

            if backup.exists():

                shutil.rmtree(backup, ignore_errors=True)



            return {"ok": True, "path": str(target), "checksum": checksum}

        except Exception as e:

            try:

                # На случай, если tmp_dir создан

                if 'tmp_dir' in locals():

                    shutil.rmtree(tmp_dir, ignore_errors=True)

            except Exception:

                pass

            return {"ok": False, "path": str(target_dir), "checksum": "", "error": str(e)}



    def _load_from_disk_atomic_format(self, source_dir: str) -> Dict[str, Any]:

        """Загружает состояние фрактального хранилища (atomic-формат: index.json + data/*.npy).



        Возвращает отчёт: { ok: bool, checksum: str, error?: str } и заполняет состояние объекта.

        """

        try:

            base = Path(source_dir)

            with open(base / "index.json", "r", encoding="utf-8") as f:

                index = json.load(f)



            # Сбрасываем текущее

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

                # Убеждаемся, что форма согласована

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



            # Пересчитываем total_memory и окно

            self._update_metadata()

            self._initialize_hot_window()



            disk_checksum = index.get("checksum", "")

            mem_checksum = self.compute_checksum()

            if disk_checksum and mem_checksum and disk_checksum != mem_checksum:

                return {"ok": False, "checksum": mem_checksum, "error": "checksum_mismatch"}

            return {"ok": True, "checksum": mem_checksum or disk_checksum}

        except Exception as e:

            return {"ok": False, "checksum": "", "error": str(e)}



    def _safe_name(self, s: str) -> str:

        """Безопасное имя для файла контейнера."""

        return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in s)[:200]



    def clear(self) -> None:

        self.containers.clear()

        self.fractal_tree.clear()

        self.hot_window.clear()

        self.total_memory = 0

        self.model_id = None

        # Очистка GPU-кэша горячего окна

        try:

            if hasattr(self, "gpu_hot_cache"):

                self.gpu_hot_cache.clear()

            if getattr(self, "device", "cpu") == "cuda" and torch.cuda.is_available():

                torch.cuda.empty_cache()

        except Exception:

            pass



    # ----------------------- Граф памяти: атомарный манифест -----------------------

    def save_graph_manifest_atomic(

        self,

        manifest_dir: str,

        records: "Iterable[Dict[str, Any]]",

        meta: Optional[Dict[str, Any]] = None,

        manifest_filename: str = "manifest.jsonl",

        meta_filename: str = "manifest_meta.json",

    ) -> Dict[str, Any]:

        """Атомарно сохраняет манифест графа памяти рядом с фрактальными данными.



        Структура:

          manifest_dir/

            manifest.jsonl       # построчные записи узлов/ребер графа памяти

            manifest_meta.json   # метаданные манифеста (версия, ссылки, модель)



        Args:

            manifest_dir: каталог для манифеста

            records: итератор записей (dict) формата node/edge

            meta: дополнительные метаданные манифеста

            manifest_filename: имя файла для jsonl

            meta_filename: имя файла для метаданных



        Returns:

            dict: { ok: bool, path: str, count: int, error?: str }

        """

        import os

        base = Path(manifest_dir)

        try:

            base.mkdir(parents=True, exist_ok=True)

            mf_path = base / manifest_filename

            tmp_path = mf_path.with_suffix(mf_path.suffix + ".tmp")



            count = 0

            # Пишем jsonl во временный файл с flush/fsync

            with open(tmp_path, "w", encoding="utf-8") as f:

                for rec in records:

                    try:

                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                        count += 1

                    except Exception:

                        # Пропускаем некорректные записи, но продолжаем

                        continue

                try:

                    f.flush()

                    os.fsync(f.fileno())

                except Exception:

                    pass

            os.replace(tmp_path, mf_path)



            # Метаданные манифеста

            meta_obj: Dict[str, Any] = {

                "version": 1,

                "created_ts": time.time(),

                "model_id": self.model_id,

            }

            if meta:

                try:

                    meta_obj.update(dict(meta))

                except Exception:

                    pass

            try:

                with open(base / meta_filename, "w", encoding="utf-8") as mf:

                    json.dump(meta_obj, mf, ensure_ascii=False, indent=2)

            except Exception:

                # Метаданные не критичны для консистентности jsonl

                pass



            return {"ok": True, "path": str(base), "count": count}

        except KeyboardInterrupt:

            try:

                if 'tmp_path' in locals() and Path(tmp_path).exists():

                    Path(tmp_path).unlink(missing_ok=True)

            except Exception:

                pass

            raise

        except Exception as e:

            try:

                if 'tmp_path' in locals() and Path(tmp_path).exists():

                    Path(tmp_path).unlink(missing_ok=True)

            except Exception:

                pass

            return {"ok": False, "path": str(manifest_dir), "count": 0, "error": str(e)}



    def load_graph_manifest(

        self,

        manifest_dir: str,

        manifest_filename: str = "manifest.jsonl",

        meta_filename: str = "manifest_meta.json",

        limit: Optional[int] = None,

    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

        """Загружает манифест графа памяти.



        Args:

            manifest_dir: каталог, где лежит манифест

            manifest_filename: имя jsonl

            meta_filename: имя meta json

            limit: максимум строк (для предварительного чтения/диагностики)



        Returns:

            Tuple[List[dict], Dict]: (records, meta)

        """

        base = Path(manifest_dir)

        mf_path = base / manifest_filename

        meta_path = base / meta_filename

        records: List[Dict[str, Any]] = []

        meta: Dict[str, Any] = {}

        try:

            if meta_path.exists():

                try:

                    with open(meta_path, "r", encoding="utf-8") as f:

                        meta = json.load(f) or {}

                except Exception:

                    meta = {}

            if not mf_path.exists():

                return (records, meta)

            with open(mf_path, "r", encoding="utf-8") as f:

                for i, line in enumerate(f):

                    if limit is not None and i >= int(limit):

                        break

                    line = line.strip()

                    if not line:

                        continue

                    try:

                        records.append(json.loads(line))

                    except Exception:

                        continue

            return (records, meta)

        except Exception:

            # Безопасный возврат даже при частичных сбоях

            return (records, meta)



    def build_shards_index(self, output_dir: str) -> Dict[str, Dict[str, str]]:

        """Строит индекс по `shards_manifest.jsonl`: id -> {shard_file, key}.



        Args:

            output_dir: каталог с результатами шардированного сохранения



        Returns:

            dict: mapping container_id -> {"shard_file": str, "key": str}

        """

        base = Path(output_dir)

        manifest = base / "shards_manifest.jsonl"

        index: Dict[str, Dict[str, str]] = {}

        if not manifest.exists():

            return index

        try:

            with manifest.open("r", encoding="utf-8") as mf:

                for line in mf:

                    line = line.strip()

                    if not line:

                        continue

                    try:

                        rec = json.loads(line)

                        cid = rec.get("id")

                        shard_file = rec.get("shard_file")

                        key = rec.get("key")

                        if cid and shard_file and key:

                            index[str(cid)] = {"shard_file": str(shard_file), "key": str(key)}

                    except Exception:

                        continue

        except Exception:

            pass

        return index



    # ----------------------- Извлечение знаний -----------------------

    def extract_knowledge_from_model(self, model: torch.nn.Module) -> List[Dict]:

        """

        Извлекает структурированные знания из весов модели (лёгкая эвристика).

        Методы: внимание (отношения), FFN (факты), эмбеддинги (концепты).

        """

        logger.info("Начато извлечение знаний из модели...")

        start_time = time.time()

        knowledge: List[Dict[str, Any]] = []



        # 1) Внимание

        logger.debug("Анализ слоев внимания...")

        for layer_name, layer in model.named_modules():

            if ("attn" in layer_name) or ("attention" in layer_name):

                try:

                    relations = self._analyze_attention_weights(layer, layer_name)

                    knowledge.extend(relations)

                except Exception:

                    logger.debug(f"Пропуск анализа внимания для {layer_name}", exc_info=True)



        # 2) FFN

        logger.debug("Анализ feed-forward сетей...")

        for layer_name, layer in model.named_modules():

            if ("mlp" in layer_name) or ("ffn" in layer_name) or ("feedforward" in layer_name):

                try:

                    facts = self._analyze_ffn_weights(layer, layer_name)

                    knowledge.extend(facts)

                except Exception:

                    logger.debug(f"Пропуск анализа FFN для {layer_name}", exc_info=True)



        # 3) Эмбеддинги

        logger.debug("Анализ эмбеддингов...")

        for layer_name, layer in model.named_modules():

            if ("embed" in layer_name) or ("wte" in layer_name) or ("wpe" in layer_name):

                try:

                    concepts = self._analyze_embeddings(layer, layer_name)

                    knowledge.extend(concepts)

                except Exception:

                    logger.debug(f"Пропуск анализа эмбеддингов для {layer_name}", exc_info=True)



        # 4) Оценка надёжности

        logger.debug("Оценка надежности знаний...")

        self._evaluate_knowledge_reliability(knowledge)



        # 5) Фильтрация

        initial_count = len(knowledge)

        filtered = [k for k in knowledge if float(k.get("reliability", 0.0)) >= 0.5]

        filtered_count = initial_count - len(filtered)

        logger.info(

            f"Извлечение знаний завершено за {time.time() - start_time:.2f} сек. "

            f"Извлечено {len(filtered)} знаний (отфильтровано {filtered_count})."

        )

        return filtered



    def _analyze_attention_weights(self, layer: torch.nn.Module, layer_name: str) -> List[Dict]:

        """

        Анализирует веса внимания: если доступны q_proj/k_proj и num_heads — строит отношения по головам,

        иначе fallback на норму весов как ранее.

        """

        results: List[Dict[str, Any]] = []



        has_qkv = all(hasattr(layer, attr) for attr in ("q_proj", "k_proj", "v_proj"))

        num_heads = getattr(layer, "num_heads", None)

        if has_qkv and isinstance(num_heads, int) and num_heads > 0:

            try:

                q_weights = layer.q_proj.weight.data.detach().cpu().numpy()

                k_weights = layer.k_proj.weight.data.detach().cpu().numpy()

                # v_weights = layer.v_proj.weight.data.detach().cpu().numpy()  # не используется напрямую здесь



                d_model = q_weights.shape[1]

                head_dim = max(1, d_model // num_heads)



                for head in range(num_heads):

                    start_idx = head * head_dim

                    end_idx = min((head + 1) * head_dim, d_model)

                    q_head = q_weights[:, start_idx:end_idx]

                    k_head = k_weights[:, start_idx:end_idx]



                    # Приближённая матрица внимания и softmax по строкам

                    attn_matrix = np.dot(q_head, k_head.T) / np.sqrt(float(head_dim))

                    attn_matrix = self._softmax(attn_matrix)



                    threshold = 0.7

                    strong_i, strong_j = np.where(attn_matrix > threshold)

                    for i, j in zip(strong_i, strong_j):

                        if i == j:

                            continue

                        strength = float(attn_matrix[i, j])

                        results.append(

                            {

                                "type": "attention_relation",

                                "subject": f"token_{i}",

                                "object": f"token_{j}",

                                "predicate": f"attention_head_{head}",

                                "strength": strength,

                                "score": strength,

                                "source_layer": layer.__class__.__name__,

                                "layer": layer_name,

                                "reliability": 0.0,

                            }

                        )

                logger.debug(f"Из слоя внимания {layer_name} извлечено {len(results)} отношений")

                return results

            except Exception as e:

                logger.error(f"Ошибка анализа весов внимания ({layer_name}): {e}")

                # падаем в фолбэк ниже



        # Fallback: оценка по нормам весов

        q = getattr(layer, "q_proj", None)

        k = getattr(layer, "k_proj", None)

        v = getattr(layer, "v_proj", None)

        matrices = [("Q", q), ("K", k), ("V", v)]

        norms: Dict[str, float] = {}

        for tag, mod in matrices:

            try:

                w = getattr(mod, "weight", None)

                if isinstance(w, torch.nn.Parameter):

                    norms[tag] = float(w.detach().abs().mean().cpu().item())

            except Exception:

                continue

        if not norms:

            w = getattr(layer, "weight", None)

            if isinstance(w, torch.nn.Parameter):

                norms["W"] = float(w.detach().abs().mean().cpu().item())



        score = float(sum(norms.values()) / max(1, len(norms))) if norms else 0.0

        if score > 0:

            results.append(

                {

                    "type": "attention_relation",

                    "layer": layer_name,

                    "details": {"norms": norms},

                    "score": score,

                    "reliability": 0.0,

                }

            )

        return results



    def _softmax(self, x: np.ndarray) -> np.ndarray:

        """Приближённый softmax по строкам матрицы."""

        x = x.astype(np.float64, copy=False)

        x = x - np.max(x, axis=1, keepdims=True)

        e_x = np.exp(x)

        denom = np.sum(e_x, axis=1, keepdims=True) + 1e-12

        return e_x / denom



    def _analyze_ffn_weights(self, layer: torch.nn.Module, layer_name: str) -> List[Dict]:

        """

        Анализирует FFN: если слой имеет компоненты c_fc/c_proj — извлекаем паттерны по W1/W2,

        иначе fallback: оцениваем линейные подслои по mean(|W|).

        """

        # Структурный путь (например GPT2Block.mlp: c_fc/c_proj)

        if hasattr(layer, "c_fc") and hasattr(layer, "c_proj"):

            try:

                w1 = layer.c_fc.weight.data.detach().cpu().numpy()

                w2 = layer.c_proj.weight.data.detach().cpu().numpy()

                patterns = self._extract_weight_patterns(w1, w2)

                facts: List[Dict[str, Any]] = []

                for pattern in patterns:

                    fact_type = self._determine_fact_type(pattern)

                    significance = float(pattern.get("significance", 0.0))

                    facts.append(

                        {

                            "type": fact_type,

                            "layer": layer_name,

                            "content": pattern.get("description", ""),

                            "pattern": pattern,

                            "score": significance,

                            "source_layer": layer.__class__.__name__,

                            "reliability": 0.0,

                        }

                    )

                logger.debug(f"Из FFN {layer_name} извлечено {len(facts)} фактов (паттерны)")

                return facts

            except Exception as e:

                logger.error(f"Ошибка анализа весов FFN ({layer_name}): {e}")

                # Падение в фолбэк ниже



        # Fallback: простой обзор всех линейных подслоёв

        results: List[Dict[str, Any]] = []

        for name, sub in layer.named_modules():

            if isinstance(sub, torch.nn.Linear):

                try:

                    w = sub.weight.detach().cpu()

                    val = float(w.abs().mean().item())

                    results.append(

                        {

                            "type": "ffn_fact",

                            "layer": f"{layer_name}.{name}" if name else layer_name,

                            "details": {"mean_abs_w": val, "out_features": int(sub.out_features)},

                            "score": val,

                            "reliability": 0.0,

                        }

                    )

                except Exception:

                    continue

        return results



    def _extract_weight_patterns(self, w1: np.ndarray, w2: np.ndarray) -> List[Dict]:

        """Извлекает значимые паттерны из W1/W2: нормализация -> легковесный k-means -> фильтрация."""

        patterns: List[Dict[str, Any]] = []

        w1_norm = self._normalize_weights(w1)

        # Согласуем размерности: работаем по строкам w1 и столбцам w2 (т.е. строкам w2.T)

        w2_norm = self._normalize_weights(w2.T)

        clusters = self._find_weight_clusters(w1_norm, w2_norm)

        for cid, cluster_idx in enumerate(clusters):

            if self._is_significant_cluster(cluster_idx):

                cluster_info = {

                    "id": f"pattern_{cid}",

                    "cluster": cluster_idx,

                }

                desc = self._describe_pattern(cluster_info)

                signif = self._calculate_pattern_significance(cluster_info)

                patterns.append(

                    {

                        "id": f"pattern_{cid}",

                        "cluster": cluster_idx,

                        "description": desc,

                        "significance": signif,

                    }

                )

        return patterns



    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:

        """L2-нормализация по строкам с защитой от деления на ноль."""

        weights = weights.astype(np.float32, copy=False)

        norms = np.linalg.norm(weights, axis=1, keepdims=True)

        norms = np.where(norms == 0.0, 1e-8, norms)

        return weights / norms



    def _find_weight_clusters(self, w1: np.ndarray, w2: np.ndarray) -> List[np.ndarray]:

        """Простой k-means без внешних зависимостей. Кластеризуем по feature=[w1|w2]."""

        combined = np.concatenate([w1, w2], axis=1)

        n = combined.shape[0]

        if n < 4:

            return [np.arange(n)]

        n_clusters = int(max(2, min(10, n // 10)))

        rng = np.random.default_rng(42)

        # Инициализация центроидов случайной подвыборкой

        init_idx = rng.choice(n, size=n_clusters, replace=False)

        centroids = combined[init_idx].copy()



        def assign(xx: np.ndarray, cc: np.ndarray) -> np.ndarray:

            # расстояния до центроидов

            d2 = ((xx[:, None, :] - cc[None, :, :]) ** 2).sum(axis=2)

            return d2.argmin(axis=1)



        max_iter = 20

        labels = np.zeros(n, dtype=np.int32)

        for _ in range(max_iter):

            new_labels = assign(combined, centroids)

            if np.array_equal(new_labels, labels):

                break

            labels = new_labels

            for k in range(n_clusters):

                mask = labels == k

                if np.any(mask):

                    centroids[k] = combined[mask].mean(axis=0)



        clusters: List[np.ndarray] = []

        for k in range(n_clusters):

            clusters.append(np.where(labels == k)[0])

        return clusters



    def _is_significant_cluster(self, cluster: np.ndarray) -> bool:

        """Кластер значим, если в нём больше 5 элементов."""

        return int(cluster.size) > 5



    def _describe_pattern(self, cluster: Dict) -> str:

        """Формирует краткое описание паттерна."""

        size = int(np.array(cluster.get("cluster", [])).size)

        return f"Значимый паттерн в весах с {size} элементами"



    def _calculate_pattern_significance(self, cluster: Dict) -> float:

        """Простая эвристика значимости: масштаб от размера кластера."""

        size = int(np.array(cluster.get("cluster", [])).size)

        return float(min(1.0, size / 100.0))



    def _determine_fact_type(self, pattern: Dict) -> str:

        """Определяет тип факта на основе свойств паттерна (эвристика)."""

        signif = float(pattern.get("significance", 0.0))

        if signif > 0.7:

            return "ffn_pattern_strong"

        if signif > 0.4:

            return "ffn_pattern_moderate"

        return "ffn_pattern_weak"



    def _analyze_embeddings(self, layer: torch.nn.Module, layer_name: str) -> List[Dict]:

        """Эмбеддинги: выбираем топ-N векторов по норме как кандидаты концептов (без тяжёлой кластеризации)."""

        results: List[Dict[str, Any]] = []

        weight = getattr(layer, "weight", None)

        if not isinstance(weight, torch.nn.Parameter):

            return results

        try:

            emb = weight.detach().cpu().float()

            if emb.ndim != 2:

                return results

            norms = torch.linalg.norm(emb, dim=1)

            topn = int(min(10, norms.numel()))

            if topn <= 0:

                return results

            top_vals, top_idx = torch.topk(norms, topn)

            for i in range(int(topn)):

                try:

                    idx = int(top_idx[i].item())

                    val = float(top_vals[i].item())

                    results.append(

                        {

                            "type": "concept",

                            "layer": layer_name,

                            "content": f"embedding_{idx}",

                            "score": val,

                            "reliability": 0.0,

                        }

                    )

                except Exception:

                    continue

        except Exception:

            pass

        return results



    # ----------------------- Перенос знаний -> графовый манифест -----------------------

    def knowledge_to_graph_records(

        self,

        knowledge: List[Dict[str, Any]],

        shard_index: Dict[str, Dict[str, Any]],

        max_refs_per_record: int = 4,

    ) -> List[Dict[str, Any]]:

        """

        Преобразует извлечённые знания в записи графа с привязкой к шардам.



        Находим контейнеры по совпадению `metadata.layer_name` с полем `layer` записи знания,

        формируем ссылки fractal_ref: [{shard_file, key, container_id, block_start, block_end, tensor_path}].

        """

        # Инвертированный индекс: layer_name -> [entry]

        by_layer: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for cid, entry in shard_index.items():

            try:

                md = entry.get("metadata", {}) or {}

                ln = md.get("layer_name")

                if isinstance(ln, str) and ln:

                    by_layer[ln].append({"container_id": cid, **entry})

            except Exception:

                continue



        records: List[Dict[str, Any]] = []

        for k in knowledge:

            try:

                layer_name = str(k.get("layer", k.get("source_layer", "")))

                refs: List[Dict[str, Any]] = []

                candidates = by_layer.get(layer_name, [])

                for ent in candidates[:max_refs_per_record]:

                    md = ent.get("metadata", {}) or {}

                    refs.append(

                        {

                            "shard_file": ent.get("shard_file"),

                            "key": ent.get("key"),

                            "container_id": ent.get("container_id"),

                            "block_start": md.get("block_start"),

                            "block_end": md.get("block_end"),

                            "tensor_path": md.get("tensor_path"),

                            "param_name": md.get("param_name"),

                            "level": ent.get("level"),

                        }

                    )



                rec = {

                    "record_type": "knowledge",  # общий тип записи

                    "type": k.get("type", "unknown"),

                    "score": float(k.get("score", 0.0)),

                    "reliability": float(k.get("reliability", 0.0)),

                    "attrs": {k_: v for k_, v in k.items() if k_ not in ("type", "score", "reliability")},

                    "fractal_ref": refs,

                }

                records.append(rec)

            except Exception:

                continue

        return records



    def transfer_model_to_graph(

        self,

        model: torch.nn.Module,

        output_dir: str,

        graph_manifest_dir: Optional[str] = None,

    ) -> Dict[str, Any]:

        """

        Полный цикл: извлечение знаний -> привязка к шардам -> атомарная запись манифеста.



        Предполагается, что фрактальная структура уже сохранена шардированно в `output_dir`.

        """

        try:

            base = Path(output_dir)

            if graph_manifest_dir is None:

                graph_manifest_dir = str(base)



            # 1) Извлечь знания из модели

            knowledge = self.extract_knowledge_from_model(model)



            # 2) Индекс шардов (из shards_manifest.jsonl)

            shard_index = self.build_shards_index(str(base))



            # 3) Записи графа с привязками к шардам

            records = self.knowledge_to_graph_records(knowledge, shard_index)



            # 4) Метаданные

            manifest_meta = {

                "format": {

                    "type": "cogniflex_knowledge_graph",

                    "version": 1,

                },

                "links": {

                    "shards_manifest": "shards_manifest.jsonl",

                },

                "model_id": self.model_id,

                "incremental": False,

            }



            # 5) Атомарная запись

            return self.save_graph_manifest_atomic(

                manifest_dir=str(graph_manifest_dir),

                records=records,

                meta=manifest_meta,

            )

        except Exception as e:

            return {"ok": False, "error": str(e), "path": str(graph_manifest_dir or output_dir), "count": 0}



    def _evaluate_knowledge_reliability(self, knowledge: List[Dict]) -> None:

        """Простая оценка надёжности по типу и нормализованному score."""

        if not knowledge:

            return

        # Нормируем score в [0,1] по всему набору

        scores = [float(k.get("score", 0.0)) for k in knowledge]

        smin, smax = min(scores), max(scores)

        rng = (smax - smin) if smax > smin else 1.0

        for k in knowledge:

            base = (float(k.get("score", 0.0)) - smin) / rng

            t = k.get("type", "")

            # Весовые коэффициенты по типам (эвристика)

            if t == "attention_relation":

                coef = 0.7

            elif t == "ffn_fact":

                coef = 0.6

            elif t == "embedding_concept":

                coef = 0.5

            else:

                coef = 0.4

            k["reliability"] = float(max(0.0, min(1.0, base * coef)))



    # ----------------------- Сохранение/загрузка -----------------------

    def save_to_disk(self, output_path: str, knowledge_graph: Optional[Dict[str, Any]] = None) -> bool:

        """

        Сохраняет фрактальную структуру на диск в простой файловой схеме:

        - <output_dir>/index.json: метаданные и статистика

        - <output_dir>/containers.jsonl: список контейнеров с метаданными и путём к данным

        - <output_dir>/data/<container_id>.npy: данные контейнера

        - <output_dir>/knowledge_graph.json: (опционально) граф знаний

        """

        try:

            out_dir = Path(output_path)

            out_dir.mkdir(parents=True, exist_ok=True)

            data_dir = out_dir / "data"

            data_dir.mkdir(parents=True, exist_ok=True)



            # Пишем контейнеры и собираем JSONL

            containers_jsonl = []

            for cid, cont in self.containers.items():

                # Безопасное короткое имя файла: SHA1(cid)

                # Это устраняет проблемы Windows с MAX_PATH и недопустимыми символами

                sha1 = hashlib.sha1(cid.encode("utf-8")).hexdigest()

                file_name = f"{sha1}.npy"

                file_path = data_dir / file_name

                # Сохраняем данные

                np.save(file_path, cont.data)

                # Запись метаданных

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



            # Пишем индекс/статистику

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



        # - <output_dir>/shards/level_<L>_<S>.npz — данные контейнеров (key = sha1(id))

        # - <output_dir>/shards_manifest.jsonl    — построчно метаданные каждого контейнера + {shard_file, key}

        # - <output_dir>/knowledge_graph.json     — опционально граф знаний



        # Заметки:

        # - По умолчанию группируем по уровням (by_level=True) и пишем примерно shard_size контейнеров на шард.

        # - Используем ключи внутри .npz как SHA1 от container_id для надёжности на Windows.

        # - Пишем прогресс и периодически чистим память (gc.collect, CUDA cache).

        

    def save_to_disk_sharded(

        self,

        output_path: str,

        knowledge_graph: Optional[Dict[str, Any]] = None,

        shard_size: int = 5000,

        by_level: bool = True,

        compress: bool = True,

    ) -> bool:

        """

        Шардированное сохранение фрактальной структуры на диск:

        - <output_dir>/shards/level_<L>_shard_<S>.npz — данные контейнеров (key = sha1(id))

        - <output_dir>/shards_manifest.jsonl         — построчно метаданные каждого контейнера + {shard_file, key}

        - <output_dir>/knowledge_graph.json          — опционально граф знаний

        """

        try:

            out_dir = Path(output_path)

            out_dir.mkdir(parents=True, exist_ok=True)

            shards_dir = out_dir / "shards"

            shards_dir.mkdir(parents=True, exist_ok=True)



            # Манифест контейнеров -> куда записан и под каким ключом

            manifest_path = out_dir / "shards_manifest.jsonl"

            if manifest_path.exists():

                try:

                    manifest_path.unlink()

                except Exception:

                    pass



            total = sum(len(v) for v in self.fractal_tree.values())

            written = 0



            def write_shard(npz_path: Path, items: List[Tuple[str, FractalContainer]]) -> None:

                # Собираем словарь ключ -> массив

                arrays: Dict[str, np.ndarray] = {}

                for cid, cont in items:

                    key = hashlib.sha1(cid.encode("utf-8")).hexdigest()

                    arrays[key] = cont.data

                # Запись .npz

                if compress:

                    np.savez_compressed(npz_path, **arrays)

                else:

                    np.savez(npz_path, **arrays)

                # Запись строк в манифест

                with manifest_path.open("a", encoding="utf-8") as mf:

                    for cid, cont in items:

                        key = hashlib.sha1(cid.encode("utf-8")).hexdigest()

                        rec = {

                            "id": cid,

                            "level": int(cont.level),

                            "position": list(cont.position),

                            "shape": list(cont.shape),

                            "dtype": str(cont.dtype),

                            "metadata": cont.metadata,

                            "parent": cont.parent,

                            "children": cont.children,

                            "shard_file": str(npz_path.relative_to(out_dir)),

                            "key": key,

                        }

                        

                        mf.write(json.dumps(rec, ensure_ascii=False) + "\n")



            start_ts = time.time()

            logger.info(

                f"Шардированное сохранение: всего контейнеров {total}, shard_size={shard_size}, by_level={by_level}"

            )



            if by_level:

                for level in sorted(self.fractal_tree.keys()):

                    ids = self.fractal_tree.get(level, [])

                    if not ids:

                        continue

                    logger.info(f"Уровень {level}: контейнеров {len(ids)}")

                    chunk: List[Tuple[str, FractalContainer]] = []

                    shard_idx = 0

                    progress_interval = max(1, len(ids) // 20)

                    for idx, cid in enumerate(ids):

                        cont = self.containers.get(cid)

                        if cont is None:

                            continue

                        chunk.append((cid, cont))

                        if len(chunk) >= shard_size:

                            shard_path = shards_dir / f"level_{level}_shard_{shard_idx}.npz"

                            write_shard(shard_path, chunk)

                            written += len(chunk)

                            shard_idx += 1

                            chunk = []

                            # Мониторим и чистим память

                            if (shard_idx % 2) == 0:

                                gc.collect()

                                try:

                                    if getattr(self, "device", "cpu") == "cuda" and torch.cuda.is_available():

                                        torch.cuda.empty_cache()

                                except Exception:

                                    pass

                        if idx % progress_interval == 0:

                            done_pct = (idx + 1) / max(1, len(ids)) * 100

                            logger.debug(f"Уровень {level}: {done_pct:.1f}%")

                    if chunk:

                        shard_path = shards_dir / f"level_{level}_shard_{shard_idx}.npz"

                        write_shard(shard_path, chunk)

                        written += len(chunk)

            else:

                # Плоское шардирование по всем контейнерам

                all_ids: List[str] = []

                for lvl in sorted(self.fractal_tree.keys()):

                    all_ids.extend(self.fractal_tree.get(lvl, []))

                logger.info(f"Всего контейнеров для записи: {len(all_ids)}")

                chunk = []

                shard_idx = 0

                progress_interval = max(1, len(all_ids) // 20)

                for idx, cid in enumerate(all_ids):

                    cont = self.containers.get(cid)

                    if cont is None:

                        continue

                    chunk.append((cid, cont))

                    if len(chunk) >= shard_size:

                        shard_path = shards_dir / f"shard_{shard_idx}.npz"

                        write_shard(shard_path, chunk)

                        written += len(chunk)

                        shard_idx += 1

                        chunk = []

                        if (shard_idx % 2) == 0:

                            gc.collect()

                            try:

                                if getattr(self, "device", "cpu") == "cuda" and torch.cuda.is_available():

                                    torch.cuda.empty_cache()

                            except Exception:

                                pass

                    if idx % progress_interval == 0:

                        done_pct = (idx + 1) / max(1, len(all_ids)) * 100

                        logger.debug(f"Общий прогресс: {done_pct:.1f}%")

                if chunk:

                    shard_path = shards_dir / f"shard_{shard_idx}.npz"

                    write_shard(shard_path, chunk)

                    written += len(chunk)



            # Индекс/статистика

            index = {

                "model_id": self.model_id,

                "created_ts": time.time(),

                "format": "sharded_npz",

                "params": {

                    "block_size": self.block_size,

                    "fractal_levels": self.fractal_levels,

                    "containers_per_group": self.containers_per_group,

                    "hot_window_size": self.hot_window_size,

                    "shard_size": int(shard_size),

                    "by_level": bool(by_level),

                    "compressed": bool(compress),

                },

                "stats": self.get_statistics(),

            }

            with (out_dir / "index.json").open("w", encoding="utf-8") as f:

                json.dump(index, f, ensure_ascii=False, indent=2)



            # Пишем граф знаний, если есть

            if knowledge_graph is not None:

                with (out_dir / "knowledge_graph.json").open("w", encoding="utf-8") as f:

                    json.dump(knowledge_graph, f, ensure_ascii=False, indent=2)



            took = time.time() - start_ts

            logger.info(

                f"Шардированное сохранение завершено: записано {written}/{total} контейнеров за {took:.2f} сек"

            )

            return True

        except Exception:

            logger.exception("Ошибка шардированного сохранения фрактальной структуры")

            return False



    def save_to_disk_incremental(

        self,

        output_path: str,

        knowledge_graph: Optional[Dict[str, Any]] = None,

        batch_size: int = 5000,

        resume: bool = False,

        by_level: bool = True,

        compress: bool = True,

        state_filename: str = "incremental_state.json",

        graph_manifest_records: Optional[Iterable[Dict[str, Any]]] = None,

        graph_manifest_dir: Optional[str] = None,

        max_items_per_session: Optional[int] = None,

        show_progress: bool = False,

        min_batch_size: int = 1025,

        offload_after_write: bool = True,

    ) -> bool:

        """

        Инкрементальное шардированное сохранение с возможностью возобновления.

        Пишет те же структуры, что и save_to_disk_sharded, но батчами и с состоянием.

        """

        try:

            out_dir = Path(output_path)

            shards_dir = out_dir / "shards"

            out_dir.mkdir(parents=True, exist_ok=True)

            shards_dir.mkdir(parents=True, exist_ok=True)



            manifest_path = out_dir / "shards_manifest.jsonl"

            state_path = out_dir / state_filename



            # Состояние

            state: Dict[str, Any] = {

                "version": 1,

                "by_level": bool(by_level),

                "compress": bool(compress),

                "batch_size": int(batch_size),

                "current_level": None,

                "level_index": 0,  # позиция внутри списка ID выбранного уровня

                "shard_idx": 0,

                "total_written": 0,

                "initialized": False,

            }

            if resume and state_path.exists():

                try:

                    with state_path.open("r", encoding="utf-8") as f:

                        loaded = json.load(f)

                    state.update(loaded)

                    logger.info(

                        f"Возобновление инкрементального сохранения: уровень={state.get('current_level')} "

                        f"index={state.get('level_index')} shard_idx={state.get('shard_idx')} "

                        f"total_written={state.get('total_written')}"

                    )

                except Exception:

                    logger.warning("Не удалось прочитать состояние, начинаем заново", exc_info=True)



            def write_state() -> None:

                try:

                    with state_path.open("w", encoding="utf-8") as f:

                        json.dump(state, f, ensure_ascii=False, indent=2)

                except Exception:

                    logger.debug("Не удалось сохранить состояние инкрементального сохранения", exc_info=True)



            # Инициализация индекса и графа (однократно)

            if not state.get("initialized"):

                index = {

                    "model_id": self.model_id,

                    "created_ts": time.time(),

                    "format": "sharded_npz",

                    "params": {

                        "block_size": self.block_size,

                        "fractal_levels": self.fractal_levels,

                        "containers_per_group": self.containers_per_group,

                        "hot_window_size": self.hot_window_size,

                        "shard_size": int(batch_size),

                        "by_level": bool(by_level),

                        "compressed": bool(compress),

                        "incremental": True,

                    },

                    "stats": self.get_statistics(),

                }

                with (out_dir / "index.json").open("w", encoding="utf-8") as f:

                    json.dump(index, f, ensure_ascii=False, indent=2)

                if knowledge_graph is not None and not (out_dir / "knowledge_graph.json").exists():

                    with (out_dir / "knowledge_graph.json").open("w", encoding="utf-8") as f:

                        json.dump(knowledge_graph, f, ensure_ascii=False, indent=2)

                # Если это не возобновление, удалим старый манифест

                if not resume and manifest_path.exists():

                    try:

                        manifest_path.unlink()

                    except Exception:

                        pass

                state["initialized"] = True

                write_state()



            def write_shard(npz_path: Path, items: List[Tuple[str, FractalContainer]]) -> None:

                # Собираем словарь ключ -> массив

                arrays: Dict[str, np.ndarray] = {}

                for cid, cont in items:

                    key = hashlib.sha1(cid.encode("utf-8")).hexdigest()

                    arrays[key] = cont.data

                # Атомарная запись .npz через временный файл и замену

                tmp_path = npz_path.with_suffix(npz_path.suffix + ".tmp")

                try:

                    with open(tmp_path, "wb") as f:

                        if compress:

                            np.savez_compressed(f, **arrays)

                        else:

                            np.savez(f, **arrays)

                        try:

                            f.flush()

                            os.fsync(f.fileno())

                        except Exception:

                            pass

                    os.replace(tmp_path, npz_path)

                except KeyboardInterrupt:

                    try:

                        if tmp_path.exists():

                            tmp_path.unlink(missing_ok=True)

                    except Exception:

                        pass

                    raise

                except Exception:

                    try:

                        if tmp_path.exists():

                            tmp_path.unlink(missing_ok=True)

                    except Exception:

                        pass

                    raise

                # Запись строк в манифест с явным flush/fsync

                try:

                    with manifest_path.open("a", encoding="utf-8") as mf:

                        for cid, cont in items:

                            key = hashlib.sha1(cid.encode("utf-8")).hexdigest()

                            rec = {

                                "id": cid,

                                "level": int(cont.level),

                                "position": list(cont.position),

                                "shape": list(cont.shape),

                                "dtype": str(cont.dtype),

                                "metadata": cont.metadata,

                                "parent": cont.parent,

                                "children": cont.children,

                                "shard_file": str(npz_path.relative_to(out_dir)),

                                "key": key,

                            }

                            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

                        try:

                            mf.flush()

                            os.fsync(mf.fileno())

                        except Exception:

                            pass

                except Exception:

                    # Манифест не критичен для целостности данных шарда; логика восстановления выше по стеку

                    logger.debug("Ошибка при записи в манифест шардов", exc_info=True)



            total = sum(len(v) for v in self.fractal_tree.values())

            logger.info(

                f"Инкрементальное сохранение: всего контейнеров {total}, batch_size={batch_size}, by_level={by_level}"

            )



            # Поддерживаем только by_level=True для простоты и предсказуемости возобновления

            if not by_level:

                logger.warning("by_level=False для инкрементального режима пока не поддержан. Включаем by_level=True")

                by_level = True



            # Гарантия минимального размера батча

            try:

                min_batch_size = int(max(1025, min_batch_size))

            except Exception:

                min_batch_size = 1025

            if batch_size < min_batch_size:

                batch_size = min_batch_size



            # Внутрисессионный счетчик: сколько записали за текущий вызов функции

            session_written = 0



            def render_progress(done: int, total_: int) -> None:

                if not show_progress:

                    return

                try:

                    pct = (done / max(1, total_))

                    bar_len = 24

                    filled = int(bar_len * pct)

                    bar = "#" * filled + "-" * (bar_len - filled)

                    sys.stdout.write(f"\r[ {bar} ] {pct*100:5.1f}%  ({done}/{total_})")

                    sys.stdout.flush()

                except Exception:

                    pass



            # первичный вывод прогресса

            render_progress(int(state.get("total_written", 0)), total)



            for level in sorted(self.fractal_tree.keys()):

                ids = self.fractal_tree.get(level, [])

                if not ids:

                    continue

                # Возобновление: пропуск уровней ниже текущего

                if state.get("current_level") is not None and int(level) < int(state["current_level"]):

                    continue

                # Установка уровня и стартового индекса

                if state.get("current_level") is None or int(level) > int(state.get("current_level", -1)):

                    state["current_level"] = int(level)

                    state["level_index"] = 0

                    state["shard_idx"] = 0

                    write_state()



                start_idx = int(state["level_index"])

                shard_idx = int(state["shard_idx"])

                while start_idx < len(ids):

                    end_idx = min(start_idx + int(batch_size), len(ids))

                    batch_ids = ids[start_idx:end_idx]

                    items: List[Tuple[str, FractalContainer]] = []

                    for cid in batch_ids:

                        cont = self.containers.get(cid)

                        if cont is None:

                            continue

                        items.append((cid, cont))

                    if not items:

                        start_idx = end_idx

                        state["level_index"] = start_idx

                        write_state()

                        continue

                    # Ограничение внутри сессии: если превышаем лимит, ужимаем items

                    if max_items_per_session is not None:

                        remaining = int(max(0, int(max_items_per_session) - session_written))

                        if remaining <= 0:

                            # Выходим из сессии, оставляя состояние для резюма

                            render_progress(int(state.get("total_written", 0)) + session_written, total)

                            logger.info(f"Сессионный лимит {max_items_per_session} достигнут. Прерываем до следующего запуска.")

                            write_state()

                            # Не удаляем state-файл, чтобы можно было продолжить

                            return True

                        if len(items) > remaining:

                            items = items[:remaining]

                            # скорректируем end_idx, чтобы прогресс и индексы были согласованы

                            end_idx = start_idx + len(items)

                    shard_path = shards_dir / f"level_{level}_shard_{shard_idx}.npz"

                    write_shard(shard_path, items)

                    # После записи шарда освобождаем память текущего батча, если требуется

                    if offload_after_write:

                        try:

                            for cid, cont in items:

                                # Сохраняем offload-метаданные для возможной последующей загрузки

                                key = hashlib.sha1(cid.encode("utf-8")).hexdigest()

                                cont.metadata["offload_shard"] = str(shard_path)

                                cont.metadata["offload_key"] = key

                                cont.metadata.setdefault("offload_format", "npz")

                                # Очищаем большие буферы данных, оставляя метаданные размеров/типа

                                try:

                                    dtype = cont.data.dtype

                                except Exception:

                                    dtype = np.float32

                                cont.data = np.empty((0,), dtype=dtype)

                        except Exception:

                            logger.debug("Не удалось выполнить offload_after_write для батча", exc_info=True)

                    state["total_written"] = int(state.get("total_written", 0)) + len(items)

                    session_written += len(items)

                    shard_idx += 1

                    start_idx = end_idx

                    state["level_index"] = start_idx

                    state["shard_idx"] = shard_idx

                    write_state()

                    # Очистка памяти

                    try:

                        # Сбрасываем ссылки на items ранним del, чтобы GC освободил память

                        del items

                    except Exception:

                        pass

                    gc.collect()

                    try:

                        if torch.cuda.is_available():

                            torch.cuda.empty_cache()

                    except Exception:

                        pass

                    # Небольшая пауза, чтобы ОС смогла вернуть страницы памяти

                    try:

                        time.sleep(0.01)

                    except Exception:

                        pass

                    # визуализация прогресса

                    render_progress(int(state.get("total_written", 0)), total)

                    # Адаптивная корректировка batch_size при высоком потреблении RAM

                    if psutil is not None:

                        try:

                            vm = psutil.virtual_memory()

                            if vm.percent > 85 and batch_size > min_batch_size:

                                # Уменьшаем, но не опускаемся ниже min_batch_size (строго >1024)

                                batch_size = max(min_batch_size, batch_size // 2)

                                logger.warning(f"Память >85%, уменьшаем batch_size до {batch_size}")

                        except Exception:

                            pass



                # уровень завершён

                state["current_level"] = int(level)

                state["level_index"] = len(ids)

                state["shard_idx"] = shard_idx

                write_state()



            # Завершение: удалим state-файл

            try:

                if state_path.exists():

                    state_path.unlink()

            except Exception:

                pass

            logger.info(

                f"Инкрементальное сохранение завершено: записано {state.get('total_written', 0)}/{total} контейнеров"

            )

            # Завершающий вывод прогресса

            render_progress(int(state.get("total_written", 0)), total)

            if show_progress:

                try:

                    sys.stdout.write("\n")

                except Exception:

                    pass

            # Сохраняем графовый манифест (если передан) после успешного завершения

            try:

                if graph_manifest_records is not None and graph_manifest_dir is not None:

                    manifest_meta = {

                        "fractal_format": "sharded_npz",

                        "links": {

                            "shards_manifest": "shards_manifest.jsonl",

                        },

                        "model_id": self.model_id,

                        "incremental": True,

                    }

                    self.save_graph_manifest_atomic(

                        graph_manifest_dir,

                        graph_manifest_records,

                        meta=manifest_meta,

                    )

            except Exception:

                logger.warning("Не удалось сохранить графовый манифест после инкрементального сохранения", exc_info=True)

            return True

        except torch.cuda.OutOfMemoryError:

            logger.warning("CUDA OOM при инкрементальном сохранении — уменьшите batch_size или используйте CPU")

            return False

        except Exception:

            logger.exception("Ошибка инкрементального сохранения")

            return False



    def auto_adjust_batch_size(self) -> int:

        """Оценка подходящего размера батча по доступной памяти CPU/GPU."""

        base_size = 5000

        try:

            # Избегаем вызовов CUDA при работе на CPU

            if getattr(self, "device", "cpu") == "cuda" and torch.cuda.is_available():

                props = torch.cuda.get_device_properties(0)

                total = getattr(props, "total_memory", 2 * 1024**3)

                free_like = max(256 * 1024**2, total - torch.cuda.memory_reserved())

                factor = float(free_like) / float(2 * 1024**3)

                base_size = int(base_size * max(0.25, min(2.0, factor)))

        except Exception:

            pass

        if psutil is not None:

            try:

                vm = psutil.virtual_memory()

                if vm.percent > 70:

                    base_size = int(base_size * 0.7)

            except Exception:

                pass

        return max(500, min(20000, base_size))



    def save_to_disk_with_recovery(self, output_path: str, max_attempts: int = 3) -> bool:

        """Инкрементальное сохранение с автоматическим восстановлением."""

        batch_size = self.auto_adjust_batch_size()

        for attempt in range(max_attempts):

            logger.info(f"Попытка сохранения {attempt + 1}/{max_attempts} с batch_size={batch_size}...")

            # Если есть state — резюмируем

            state_path = Path(output_path) / "incremental_state.json"

            resume = state_path.exists()

            if self.save_to_disk_incremental(output_path, batch_size=batch_size, resume=resume):

                logger.info("Сохранение успешно завершено")

                return True

            if attempt < max_attempts - 1:

                batch_size = max(500, batch_size // 2)

                logger.warning(f"Сохранение не завершено. Уменьшаем batch_size до {batch_size} и повторяем...")

                time.sleep(2)

        logger.error("Не удалось сохранить фрактальную структуру после нескольких попыток")

        return False



    def load_from_disk(self, input_path: str, lazy: bool = False, progress_every: int = 500000) -> bool:

        """

        Загружает фрактальную структуру с диска из каталога, созданного save_to_disk:

        - ожидает файлы: index.json, containers.jsonl, data/*.npy

        """

        try:

            in_dir = Path(input_path)

            index_path = in_dir / "index.json"

            containers_path = in_dir / "containers.jsonl"

            data_dir = in_dir / "data"

            shards_manifest = in_dir / "shards_manifest.jsonl"



            # Авто-детект atomic-формата: index.json + data/, но без containers.jsonl

            if index_path.exists() and data_dir.exists() and (not containers_path.exists()):

                report = self._load_from_disk_atomic_format(str(in_dir))

                if not report.get("ok", False):

                    logger.error(f"Не удалось загрузить (atomic): {report.get('error')}")

                return bool(report.get("ok", False))



            # Новый шардированный формат: наличие shards_manifest.jsonl

            if shards_manifest.exists():

                # Очищаем текущее состояние

                self.clear()

                # Подготовка ленивого индекса при необходимости

                if not hasattr(self, "lazy_index"):

                    self.lazy_index = {}

                # Загружаем index для метаданных, если есть

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

                                # Только индекс без загрузки массива

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

                                # Полная загрузка массива

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

                    (

                        f"Загружено (sharded){' (lazy index)' if lazy else ''} контейнеров: {loaded}. "

                        f"Уровни: {sorted(self.fractal_tree.keys())}. "

                        + (f"Общий размер: {self.total_memory / (1024*1024):.2f} MB" if not lazy else "")

                    )

                )

                return True



            # --- конец шардированной загрузки ---



        except Exception as e:

            logger.error(f"Ошибка загрузки фрактальной структуры: {e}", exc_info=True)

            return False



        # Ни atomic, ни sharded не сработали — пытаемся старый формат

        return self._load_legacy_format(index_path, containers_path, data_dir)



    # ----------------------- Ленивый доступ к контейнеру -----------------------

    def get_container_data(self, cid: str) -> np.ndarray:

        """Возвращает данные контейнера.



        - Если контейнер уже в self.containers — возвращает его data

        - Если активирован lazy-индекс (self.lazy_index) — подгружает массив из соответствующего npz и

          приводит форму к rec.shape при необходимости

        """

        if cid in self.containers:

            return self.containers[cid].data

        entry = getattr(self, "lazy_index", {}).get(cid)

        if not entry:

            raise KeyError(f"Контейнер {cid} не найден")

        shard_file = entry["shard_file"]

        key = entry["key"]

        shape = tuple(int(x) for x in (entry.get("shape") or []))

        with np.load(shard_file, allow_pickle=False) as zf:

            arr = zf[key]

        try:

            if shape and int(np.prod(shape)) == int(arr.size):

                arr = arr.reshape(shape)

        except Exception:

            pass

        return arr



    # ----------------------- Реконструкция state_dict -----------------------

    def reconstruct_state_dict(

        self,

        output_dtype: str = "float32",

        device: str = "cpu",

        limit_tensors: Optional[int] = None,

        include_params: Optional[List[str]] = None,

        resume_from: Optional[str] = None,

        processed_params: Optional["set[str]"] = None,

    ) -> Dict[str, torch.Tensor]:

        """

        Собирает исходные тензоры PyTorch по блокам из шардов, используя метаданные:

        - metadata.tensor_path: ключ параметра (например, 'transformer.wte.weight')

        - metadata.original_shape: исходная форма параметра

        - metadata.block_start/block_end: позиция блока в плоском представлении



        Args:

            output_dtype: целевой тип ('float32'|'float16'|'bfloat16'|'float64')

            device: 'cpu' или 'cuda'

            limit_tensors: если задано, собрать только N первых параметров (для быстрого теста)



        Returns:

            Dict[str, torch.Tensor]: state_dict для загрузки в модель

        """

        # Источник записей: если есть ленивый индекс — используем его приоритетно

        has_lazy = hasattr(self, "lazy_index") and len(self.lazy_index) > 0

        logger.info(f"[reconstruct] containers: {len(self.containers)}; lazy_index: {len(getattr(self, 'lazy_index', {}))}")

        use_lazy = has_lazy



        # Группируем записи по tensor_path

        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        if use_lazy:

            logger.info(f"[reconstruct] lazy_index size: {len(self.lazy_index)}")

            debug_samples = 0



            # 1) Предварительно собираем множество всех ключей (tpath), чтобы применить фильтрацию и лимит

            all_tpaths_set: "set[str]" = set()

            for _cid, _entry in self.lazy_index.items():

                # Веса state_dict восстанавливаем только из уровня 0 (сырые блоки).

                # Более высокие уровни — это агрегаты/структурные контейнеры и их нельзя смешивать в state_dict.

                try:

                    if int(_entry.get("level", -1)) != 0:

                        continue

                except Exception:

                    continue

                _meta = _entry.get("metadata", {}) or {}

                _tpath = _meta.get("tensor_path")

                if not _tpath:

                    _layer = _meta.get("layer_name") or _entry.get("layer_name")

                    _param = _meta.get("param_name") or "weight"

                    if _layer:

                        _tpath = f"{_layer}.{_param}"

                if _tpath:

                    all_tpaths_set.add(_tpath)



            # Применяем фильтры (include_params, processed_params, resume_from) и лимит по числу параметров

            selected_keys = sorted(all_tpaths_set)

            if include_params:

                flt = set()

                for k in selected_keys:

                    for pat in include_params:

                        if pat and pat in k:

                            flt.add(k)

                            break

                selected_keys = sorted(flt)

            if processed_params:

                selected_keys = [k for k in selected_keys if k not in processed_params]

            if resume_from:

                try:

                    idx = selected_keys.index(resume_from)

                    selected_keys = selected_keys[idx + 1 :]

                except ValueError:

                    pass

            if limit_tensors is not None:

                selected_keys = selected_keys[: int(limit_tensors)]



            selected_set = set(selected_keys)



            # 2) Второй проход: собираем записи только для выбранных ключей,

            #    ранний выход когда набрана полная покрывающая информация

            #    (определяется по original_shape и максимальному block_end)

            max_be: Dict[str, int] = {}

            target_total: Dict[str, int] = {}

            completed: "set[str]" = set()



            for cid, entry in self.lazy_index.items():

                # Только уровень 0 для корректной реконструкции state_dict

                try:

                    if int(entry.get("level", -1)) != 0:

                        continue

                except Exception:

                    continue

                meta = entry.get("metadata", {}) or {}

                tpath = meta.get("tensor_path")

                if not tpath:

                    layer_name = meta.get("layer_name") or entry.get("layer_name")

                    param_name = meta.get("param_name") or "weight"

                    if layer_name:

                        tpath = f"{layer_name}.{param_name}"

                # Если ключ не выбран — пропускаем сразу (ранняя фильтрация)

                if not tpath or (selected_set and tpath not in selected_set):

                    continue



                bs = meta.get("block_start")

                be = meta.get("block_end")

                if bs is None or be is None:

                    if debug_samples < 10:

                        logger.info(

                            f"[reconstruct][skip-no-blocks] cid={cid} bs={bs} be={be} meta_keys={list(meta.keys())[:10]}"

                        )

                        debug_samples += 1

                    continue



                rec = dict(entry)

                rec["block_start"] = int(bs)

                rec["block_end"] = int(be)

                rec["original_shape"] = tuple(int(x) for x in (meta.get("original_shape") or []))

                rec["tensor_path"] = tpath

                if debug_samples < 10:

                    logger.info(

                        f"[reconstruct][ok] cid={cid} tpath={tpath} bs={rec['block_start']} be={rec['block_end']} "

                        f"orig_shape={rec['original_shape']} shard_file={entry.get('shard_file')} key={entry.get('key')}"

                    )

                    debug_samples += 1

                groups[tpath].append(rec)



                # Обновляем агрегаты покрытия для раннего выхода

                mbe = max_be.get(tpath, 0)

                if rec["block_end"] > mbe:

                    max_be[tpath] = rec["block_end"]

                if tpath not in target_total:

                    try:

                        orig_shape = rec["original_shape"]

                        target_total[tpath] = int(np.prod(orig_shape)) if orig_shape else rec["block_end"]

                    except Exception:

                        target_total[tpath] = rec["block_end"]

                # Проверка завершения по tpath

                if max_be.get(tpath, 0) >= target_total.get(tpath, 1):

                    completed.add(tpath)



                # Если все выбранные ключи собраны — можно завершать сканирование lazy_index рано

                if selected_set and len(completed) >= len(selected_set):

                    break

        else:

            for cid, cont in self.containers.items():

                # Воссоздаём state_dict только по контейнерам уровня 0

                try:

                    if int(getattr(cont, "level", -1)) != 0:

                        continue

                except Exception:

                    continue

                meta = cont.metadata or {}

                tpath = meta.get("tensor_path")

                if not tpath:

                    layer_name = meta.get("layer_name") or getattr(cont, "layer_name", None)

                    param_name = meta.get("param_name") or "weight"

                    if layer_name:

                        tpath = f"{layer_name}.{param_name}"

                if not tpath:

                    continue

                bs = meta.get("block_start")

                be = meta.get("block_end")

                if bs is None or be is None:

                    continue

                rec = {

                    "block_start": int(bs),

                    "block_end": int(be),

                    "original_shape": tuple(int(x) for x in (meta.get("original_shape") or [])),

                    "tensor_path": tpath,

                    "shape": tuple(int(x) for x in (cont.shape or [])),

                    "dtype": cont.dtype,

                    "metadata": meta,

                    "container_id": cid,

                }

                groups[tpath].append(rec)



        # Функция для сортировки блоков по позиции

        def _sort_key(r: Dict[str, Any]) -> int:

            return int(r["block_start"])



        # Выбор выходного dtype torch

        torch_dtype = {

            "float16": torch.float16,

            "bfloat16": torch.bfloat16,

            "float32": torch.float32,

            "float64": torch.float64,

        }.get(str(output_dtype).lower(), torch.float32)



        # Подготовим список ключей параметров с учётом фильтрации/резюме

        all_keys = list(groups.keys())

        all_keys.sort()

        if include_params:

            # простая фильтрация по подстроке (можно расширить до regex)

            flt = set()

            for k in all_keys:

                for pat in include_params:

                    if pat and pat in k:

                        flt.add(k)

                        break

            keys = sorted(flt)

        else:

            keys = all_keys

        if processed_params:

            keys = [k for k in keys if k not in processed_params]

        if resume_from:

            # начать после указанного ключа

            try:

                idx = keys.index(resume_from)

                keys = keys[idx + 1 :]

            except ValueError:

                # если нет такого ключа — начинаем с начала

                pass



        # Диагностика: выведем первые несколько сгруппированных ключей и состав блоков

        if len(groups) > 0:

            sample_keys = list(groups.keys())[:5]

            logger.info(f"[reconstruct] grouped params: {len(groups)}; sample keys: {sample_keys}")

            for sk in sample_keys:

                recs = groups.get(sk, [])

                if recs:

                    first = recs[0]

                    logger.info(f"[reconstruct] key={sk} | records={len(recs)} | original_shape={first.get('original_shape')} | block_ranges=[{recs[0].get('block_start')}-{recs[-1].get('block_end')}]")

        else:

            logger.warning("[reconstruct] No groups found")



        state: Dict[str, torch.Tensor] = {}

        processed = 0

        for tpath in keys:

            recs = groups.get(tpath, [])

            if limit_tensors is not None and processed >= int(limit_tensors):

                break

            if not recs:

                continue

            recs.sort(key=_sort_key)



            # Определяем целевой размер

            original_shape = tuple(int(x) for x in (recs[0].get("original_shape") or []))

            if not original_shape:

                # Если нет формы — пытаемся инферить по последнему блоку

                total = int(recs[-1]["block_end"]) if recs else 0

                original_shape = (total,)



            total_elems = int(np.prod(original_shape))

            use_cuda = (str(device) == "cuda") and torch.cuda.is_available()

            if use_cuda:

                # Пишем результат напрямую в GPU-буфер

                tensor = torch.empty(original_shape, dtype=torch_dtype, device="cuda")

                # заполняем по блокам

                # Для универсальности работаем с плоским представлением на GPU

                gpu_flat = tensor.view(-1)

            else:

                flat = np.zeros(total_elems, dtype=np.float32)



            # Заполняем плоский буфер по блокам

            import time as _time 

            _t0 = _time.time()

            if use_lazy:

                # Группируем записи по shard_file, чтобы открывать .npz один раз

                by_shard: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

                for r in recs:

                    by_shard[r["shard_file"]].append(r)

                for shard_file, shard_recs in by_shard.items():

                    # Внутри шарда обрабатываем по порядку блоков

                    shard_recs.sort(key=_sort_key)

                    with np.load(shard_file, allow_pickle=False) as zf:

                        for r in shard_recs:

                            bs = int(r["block_start"])

                            be = int(r["block_end"])

                            arr = zf[r["key"]]

                            # Приводим к float32 (устойчиво)

                            if arr.dtype == np.float16:

                                arr = arr.astype(np.float32)

                            elif arr.dtype == np.int8:

                                scale = None

                                meta = r.get("metadata") or {}

                                if meta:

                                    scale = meta.get("quant_scale") or meta.get("quantization_scale")

                                if scale:

                                    arr = (arr.astype(np.float32)) * float(scale)

                                else:

                                    arr = arr.astype(np.float32)

                            elif arr.dtype != np.float32:

                                arr = arr.astype(np.float32)

                            # Вставка

                            if use_cuda:

                                chunk = torch.from_numpy(arr.reshape(-1)[: (be - bs)])

                                gpu_flat[bs:be].copy_(chunk.to("cuda"))

                            else:

                                flat[bs:be] = arr.reshape(-1)[: (be - bs)]

            else:

                for r in recs:

                    bs = int(r["block_start"])

                    be = int(r["block_end"])

                    # Берём данные из контейнера

                    cid = r.get("container_id")

                    arr = self.containers[cid].data  # type: ignore[index]

                    # Приводим к float32 (устойчиво)

                    if arr.dtype == np.float16:

                        arr = arr.astype(np.float32)

                    elif arr.dtype == np.int8:

                        scale = None

                        meta = r.get("metadata") or {}

                        if meta:

                            scale = meta.get("quant_scale") or meta.get("quantization_scale")

                        if scale:

                            arr = (arr.astype(np.float32)) * float(scale)

                        else:

                            arr = arr.astype(np.float32)

                    elif arr.dtype != np.float32:

                        arr = arr.astype(np.float32)

                    # Вставка

                    if use_cuda:

                        chunk = torch.from_numpy(arr.reshape(-1)[: (be - bs)])

                        gpu_flat[bs:be].copy_(chunk.to("cuda"))

                    else:

                        flat[bs:be] = arr.reshape(-1)[: (be - bs)]

            _dt = _time.time() - _t0

            if _dt > 2.0:

                logger.info(f"[reconstruct] assembled '{tpath}' in {_dt:.2f}s, shape={original_shape}, cuda={use_cuda}")



            if not use_cuda:

                tensor = torch.from_numpy(flat.reshape(original_shape)).to(dtype=torch_dtype, device=device)

            else:

                # Уже на GPU, при необходимости приведём dtype

                if tensor.dtype != torch_dtype:

                    tensor = tensor.to(dtype=torch_dtype)

            state[tpath] = tensor

            processed += 1



        logger.info(f"Собрано параметров: {len(state)} (limit={limit_tensors})")

        return state



# ----------------------- Утилита переупаковки -----------------------

def repack_model_to_fractal(

    model_path: str,

    output_path: str,

    fractal_levels: int = 4,

    block_size: int = 64,

    device: str = "cpu",

) -> bool:

    """

    Переупаковывает модель в фрактальную структуру с извлечением и сохранением графа знаний.

    """

    start_time = time.time()

    logger.info(f"Начата переупаковка модели из {model_path} в фрактальную структуру...")

    try:

        # 1) Загрузка модели

        logger.info("Загрузка модели...")

        # Если путь — директория, считаем что это локальная директория HF и используем Transformers

        model: Optional[torch.nn.Module]

        if os.path.isdir(model_path):

            model = _load_hf_model_dir(model_path, device=device)

        else:

            model = _safe_load_model(model_path, device=device)

        if model is None:

            logger.error("Не удалось загрузить модель")

            return False



        # 2) Извлечение знаний

        store = FractalWeightStore(block_size=block_size, fractal_levels=fractal_levels)

        logger.info("Извлечение знаний из модели...")

        knowledge = store.extract_knowledge_from_model(model)



        # 3) Построение графа знаний

        logger.info("Построение графа знаний...")

        knowledge_graph = _build_knowledge_graph(knowledge)



        # 4) Упаковка весов модели

        logger.info("Создание фрактальной структуры весов...")

        model_id = Path(model_path).stem

        if not store.pack_model_weights(model, model_id=model_id):

            logger.error("Не удалось упаковать веса модели")

            return False



        # 5) Сохранение на диск (шардированное)

        logger.info("Сохранение фрактальной структуры (шардировано)...")

        ok = store.save_to_disk_sharded(

            output_path,

            knowledge_graph=knowledge_graph,

            shard_size=10000,

            by_level=True,

            compress=True,

        )

        if not ok:

            logger.warning("Шардированное сохранение не удалось. Пытаемся инкрементально с восстановлением...")

            ok = store.save_to_disk_with_recovery(output_path)

            if not ok:

                return False



        stats = store.get_statistics()

        logger.info("Статистика фрактальной структуры:")

        logger.info(f"  Общее количество контейнеров: {stats['total_containers']}")

        logger.info(f"  Контейнеры по уровням: {stats['containers_by_level']}")

        logger.info(f"  Общий размер: {stats['total_memory_mb']:.2f} MB")

        logger.info(f"  Сжатие: {stats['compression_ratio']:.2f}x")



        logger.info(f"Переупаковка завершена за {time.time() - start_time:.2f} сек")

        return True

    except Exception:

        logger.exception("Критическая ошибка переупаковки модели")

        return False





def _safe_load_model(model_path: str, device: str = "cpu") -> Optional[torch.nn.Module]:

    """Простой загрузчик PyTorch-модели, сохранённой через torch.save(model)."""

    map_location = torch.device(device if device else "cpu")

    try:

        obj = torch.load(model_path, map_location=map_location)

        if isinstance(obj, torch.nn.Module):

            obj.eval()

            return obj

        # Если сохранён state_dict, пробуем найти простую оболочку — здесь возвращаем None

        logger.error("Ожидалась сохранённая torch.nn.Module, получен другой объект")

        return None

    except Exception:

        logger.exception(f"Ошибка загрузки модели из {model_path}")

        return None



def _load_hf_model_dir(model_dir: str, device: str = "cpu") -> Optional[torch.nn.Module]:

    """

    Загружает модель из директории HuggingFace (weights + config) через Transformers.

    Предпочитает AutoModelForCausalLM, иначе AutoModel. Работает только локально.

    """

    try:

        if AutoModelForCausalLM is None:

            raise ImportError("transformers не установлен: pip install transformers safetensors accelerate")

        # Разрешаем путь на случай, если указан корень репо HF-кэша (нужно найти snapshots/<rev>)

        base = Path(model_dir)

        real_dir = base

        if not (base / "config.json").exists():

            # Ищем в подкаталогах snapshots/*

            snaps = list((base / "snapshots").glob("*/config.json")) if (base / "snapshots").exists() else []

            if snaps:

                real_dir = snaps[0].parent

            else:

                # Ищем глубоко первый config.json

                found = list(base.rglob("config.json"))

                if found:

                    real_dir = found[0].parent

        model_path_str = str(real_dir)

        torch_dtype = torch.float16 if (device == "cuda" and torch.cuda.is_available()) else torch.float32

        # Сначала пробуем как CausalLM (rugpt3large), затем как базовую модель

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

    """Экспортирует HF-модель (например ruGPT-small) в шардированное фрактальное хранилище.



    Делает:

    - загружает модель локально (без сетевых запросов по умолчанию)

    - пакует полный state_dict в FractalWeightStore.pack_state_dict

    - сохраняет фрактальные шарды + index.json

    - сохраняет config.json (для AutoConfig.from_pretrained) и tokenizer



    Примечание: если модель не закэширована локально и local_files_only=True, экспорт завершится ошибкой.

    """

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

        logger.exception("������ �������� HF-������ �� ����������� ���������")

        return False



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

from typing import Any, Dict, Deque, Tuple, List, Optional, Iterable

from collections import deque, OrderedDict, defaultdict



import numpy as np

import torch

try:

    import psutil  # опционально для мониторинга RAM

except Exception:  # pragma: no cover

    psutil = None  # type: ignore

try:

    # Опциональная зависимость: Transformers для загрузки моделей из директории HF

    from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer

except Exception:  # pragma: no cover

    AutoModelForCausalLM = None  # type: ignore

    AutoModel = None  # type: ignore

    AutoTokenizer = None  # type: ignore



logger = logging.getLogger("cogniflex.mlearning.fractal_store")





# ----------------------- Прокси для графа знаний -----------------------

class NodeProxy:

    def __init__(self, node: Any) -> None:

        # Допускаем как объектные поля, так и словари

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

    """Лёгкий адаптер графа: поддерживает объекты с методами get_all_nodes/get_all_edges

    или словарь вида {"nodes": [...], "edges": [...]}.

    """

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





class FractalWeightStore:

    """Хранилище весов в фрактальной структуре для low-memory режима.



    Основные структуры:

    - containers: плоское хранилище контейнеров (id -> FractalContainer)

    - fractal_tree: иерархическая организация контейнеров (level -> [container_id])

    - hot_window: LRU/скользящее окно часто используемых блоков

    - total_memory: суммарный объём данных (в байтах)

    - model_id: текущая модель, для которой построено хранилище

    - block_size: базовый размер блока (по умолчанию 64 элементов)

    """



    def __init__(self, block_size: int = 64, fractal_levels: int = 5, containers_per_group: int = 4, device: str = "cpu") -> None:

        self.containers: Dict[str, FractalContainer] = {}

        self.fractal_tree: Dict[int, List[str]] = {}

        # храним контейнеры горячего окна как OrderedDict[id -> priority]

        self.hot_window: "OrderedDict[str, float]" = OrderedDict()

        self.total_memory: int = 0

        self.model_id: Optional[str] = None

        self.block_size: int = max(1, int(block_size))

        self.fractal_levels: int = max(1, int(fractal_levels))

        self.containers_per_group: int = max(1, int(containers_per_group))

        # размер горячего окна в байтах (по умолчанию 500 MB)

        self.hot_window_size: int = 500 * 1024 * 1024

        # Специальные токены и оффсет индексов узлов для сериализации графа знаний

        self.SPECIAL_TOKENS: Dict[str, int] = {

            "NODE_START": 1,

            "NODE_END": 2,

            "EDGE_START": 3,

            "EDGE_END": 4,

        }

        # Смещение для индекса узла, чтобы не пересекаться с контентными токенами

        self.NODE_OFFSET: int = 1_000_000

        # Метаданные последнего упакованного графа знаний

        self.graph_metadata: Dict[str, Any] = {}

        # Настройка устройства и GPU-кэша для горячего окна

        try:

            use_cuda = (device != "cpu")

            if use_cuda and torch.cuda.is_available():

                self.device = "cuda"

            else:

                self.device = "cpu"

        except (RuntimeError, OSError) as e:

            logger.warning(f"Error setting device: {e}")

            self.device = "cpu"

        self.gpu_hot_cache: Dict[str, "torch.Tensor"] = {}



    # ----------------------- Публичные методы -----------------------

    def pack_model_weights(self, model: torch.nn.Module, model_id: str) -> bool:

        """

        Упаковывает веса модели в фрактальную структуру с учетом 64-битной оптимизации.

        

        Алгоритм:

        1. Очищает существующие данные

        2. Создает контейнеры нулевого уровня для каждого слоя

        3. Строит иерархию фрактала на основе контейнеров нулевого уровня

        4. Инициализирует горячее окно

        5. Выполняет оптимизацию структуры

        

        Args:

            model: Модель PyTorch для упаковки

            model_id: Уникальный идентификатор модели

            

        Returns:

            bool: Успех операции

        """

        start_time = time.time()

        logger.info(f"Начата фрактальная упаковка весов модели {model_id}...")



        try:

            # Очищаем существующие данные

            self.containers.clear()

            self.fractal_tree.clear()

            self.hot_window.clear()

            self.total_memory = 0

            self.model_id = model_id



            # Шаг 1: Создаем контейнеры нулевого уровня для каждого слоя

            logger.debug("Создание контейнеров нулевого уровня...")

            for layer_name, layer in model.named_modules():

                if not hasattr(layer, 'weight') or layer.weight is None:

                    continue

                

                # Упаковываем веса слоя

                self._pack_layer_weights(layer_name, layer.weight.data.cpu().numpy(), model_id)



            # Шаг 2: Создаем более высокие уровни фрактала

            logger.debug("Построение иерархии фрактала...")

            self._build_fractal_hierarchy()



            # Шаг 3: Инициализируем горячее окно

            logger.debug("Инициализация горячего окна...")

            self._initialize_hot_window()



            # Шаг 4: Выполняем оптимизацию структуры

            logger.debug("Выполнение оптимизации структуры...")

            self._optimize_fractal_structure()



            # Шаг 5: Сохраняем статистику

            stats = self.get_statistics()

            logger.info(f"Фрактальная упаковка весов завершена за {time.time() - start_time:.2f} сек. "

                       f"Создано {stats['total_containers']} контейнеров. "

                       f"Общий размер: {stats['total_memory_mb']:.2f} MB. "

                       f"Сжатие: {stats['compression_ratio']:.2f}x")

            return True

            

        except Exception as e:

            logger.error(f"Критическая ошибка фрактальной упаковки весов: {e}", exc_info=True)

            return False



    def pack_state_dict(self, state_dict: Dict[str, torch.Tensor], model_id: str) -> bool:

        """Упаковывает полный state_dict (включая bias и все параметры) в фрактальную структуру.



        Важно: используем tensor_path == ключу state_dict, чтобы reconstruct_state_dict мог

        восстановить оригинальный state_dict без эвристик.

        """

        start_time = time.time()

        logger.info(f"Начата фрактальная упаковка state_dict модели {model_id}...")



        try:

            # Очищаем существующие данные

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



                # Безопасно переносим на CPU numpy

                try:

                    arr = tensor.detach().cpu().numpy()

                except Exception:

                    continue



                # Метаданные, совместимые с reconstruct_state_dict

                try:

                    layer_name, param_name = tpath.rsplit('.', 1)

                except ValueError:

                    layer_name, param_name = tpath, "weight"



                # Уникальный ключ слоя для генерации container_id (иначе weight/bias конфликтуют)

                layer_key = f"{layer_name}.{param_name}" if layer_name else tpath



                flat = arr.reshape(-1)

                total_elements = int(flat.size)

                original_shape = tuple(int(x) for x in arr.shape)



                # dtype хранения

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



            # Строим уровни выше нулевого и инициализируем горячее окно

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



    def cleanup_incremental_artifacts(self, output_path: str, fresh: bool = False) -> Dict[str, Any]:

        """Очищает артефакты незавершённого инкрементального сохранения.



        - Удаляет временные файлы *.tmp в каталоге шардов

        - По умолчанию сохраняет существующие шарды/манифест и лишь удаляет state-файл

        - При fresh=True полностью удаляет shards/, shards_manifest.jsonl и state



        Возвращает отчет: { ok: bool, removed: List[str] }

        """

        base = Path(output_path)

        removed: List[str] = []

        try:

            state_path = base / "incremental_state.json"

            shards_dir = base / "shards"

            manifest_path = base / "shards_manifest.jsonl"



            if fresh:

                if shards_dir.exists():

                    try:

                        shutil.rmtree(shards_dir, ignore_errors=True)

                        removed.append(str(shards_dir))

                    except Exception:

                        pass

                if manifest_path.exists():

                    try:

                        manifest_path.unlink()

                        removed.append(str(manifest_path))

                    except Exception:

                        pass

                if state_path.exists():

                    try:

                        state_path.unlink()

                        removed.append(str(state_path))

                    except Exception:

                        pass

                return {"ok": True, "removed": removed}



            # Нежная очистка: удалить *.tmp и state-файл (по желанию оставляем state? здесь удаляем)

            if shards_dir.exists():

                try:

                    for p in shards_dir.rglob("*.tmp"):

                        try:

                            p.unlink(missing_ok=True)

                            removed.append(str(p))

                        except Exception:

                            continue

                except Exception:

                    pass

            if state_path.exists():

                try:

                    state_path.unlink()

                    removed.append(str(state_path))

                except Exception:

                    pass

            return {"ok": True, "removed": removed}

        except Exception as e:

            return {"ok": False, "error": str(e), "removed": removed}



    def pack_knowledge_graph(self, knowledge_graph: "KnowledgeGraph") -> None:

        """

        Упаковывает граф знаний в фрактальную структуру.



        Алгоритм:

        1) Сериализуем граф в последовательность токенов

        2) Упаковываем последовательность блоками в контейнеры уровня 0

        3) Строим иерархию и инициализируем горячее окно

        4) Сохраняем метаданные графа

        """

        logger.info("Упаковка графа знаний в фрактальную структуру...")

        start_time = time.time()

        try:

            logger.debug("Сериализация графа знаний...")

            serialized = self._serialize_knowledge_graph(KnowledgeGraphProxy(knowledge_graph))



            logger.debug("Упаковка последовательности...")

            self._pack_sequence(serialized)



            logger.debug("Сохранение метаданных графа...")

            self._store_graph_metadata(KnowledgeGraphProxy(knowledge_graph))



            logger.info(f"Граф знаний упакован за {time.time() - start_time:.2f} сек")

        except Exception as e:

            logger.error(f"Ошибка упаковки графа знаний: {e}", exc_info=True)

            raise



    def _serialize_knowledge_graph(self, knowledge_graph: "KnowledgeGraphProxy") -> np.ndarray:

        """

        Сериализует граф знаний в последовательность токенов.



        Формат:

        - Узел: [NODE_START, type_id, content..., NODE_END]

        - Ребро: [EDGE_START, source_idx+NODE_OFFSET, target_idx+NODE_OFFSET, relation_id, EDGE_END]

        """

        nodes = knowledge_graph.get_all_nodes()

        edges = knowledge_graph.get_all_edges()



        node_to_index = {node.id: i for i, node in enumerate(nodes)}



        seq_len = 0

        for node in nodes:

            seq_len += 3 + len(self._encode_content(node.content))

        for _ in edges:

            seq_len += 5



        serialized = np.zeros(seq_len, dtype=np.int32)

        pos = 0



        # Узлы

        for node in nodes:

            serialized[pos] = self.SPECIAL_TOKENS["NODE_START"]; pos += 1

            serialized[pos] = self._encode_node_type(node.node_type); pos += 1

            content_tokens = self._encode_content(node.content)

            if content_tokens.size:

                end = pos + int(content_tokens.size)

                serialized[pos:end] = content_tokens

                pos = end

            serialized[pos] = self.SPECIAL_TOKENS["NODE_END"]; pos += 1



        # Рёбра

        for edge in edges:

            serialized[pos] = self.SPECIAL_TOKENS["EDGE_START"]; pos += 1

            serialized[pos] = node_to_index.get(edge.source, -1) + self.NODE_OFFSET; pos += 1

            serialized[pos] = node_to_index.get(edge.target, -1) + self.NODE_OFFSET; pos += 1

            serialized[pos] = self._encode_relation(edge.relation_type); pos += 1

            serialized[pos] = self.SPECIAL_TOKENS["EDGE_END"]; pos += 1



        return serialized



    def _pack_sequence(self, sequence: np.ndarray) -> None:

        """Упаковывает последовательность в фрактальную структуру контейнеров."""

        # Очистка текущего состояния

        self.containers.clear()

        self.fractal_tree.clear()

        self.total_memory = 0



        # Нулевой уровень — поблочная упаковка

        for i in range(0, int(sequence.size), self.block_size):

            block = sequence[i:i + self.block_size]

            shape = (int(block.size),)

            position = (i // self.block_size,)

            # Используем текущий model_id, если он задан, иначе 'fractal'

            cid = self._generate_container_id(0, position, "knowledge_graph", self.model_id or "fractal")

            container = FractalContainer(

                id=cid,

                level=0,

                position=position,

                data=block.astype(np.int32, copy=False),

                shape=shape,

                dtype="int32",

                metadata={

                    "block_start": i,

                    "block_end": min(i + self.block_size, int(sequence.size)),

                    "source": "knowledge_graph",

                },

            )

            self.containers[cid] = container

            self.fractal_tree.setdefault(0, []).append(cid)

            self.total_memory += container.get_memory_size()



        # Строим уровни выше нулевого и инициализируем горячее окно

        self._build_fractal_hierarchy()

        self._initialize_hot_window()



    def _store_graph_metadata(self, knowledge_graph: "KnowledgeGraphProxy") -> None:

        """Сохраняет легковесные метаданные о структуре графа."""

        nodes = knowledge_graph.get_all_nodes()

        edges = knowledge_graph.get_all_edges()

        self.graph_metadata = {

            "node_count": len(nodes),

            "edge_count": len(edges),

            "types": list(sorted({getattr(n, "node_type", "unknown") for n in nodes})),

        }



    # ----------------------- Простейшие кодировщики -----------------------

    def _encode_node_type(self, node_type: str) -> int:

        table = {

            "token": 10,

            "ffn": 11,

            "concept": 12,

            "layer": 13,

            "unknown": 14,

        }

        if not isinstance(node_type, str):

            return table["unknown"]

        return table.get(node_type, 100 + (abs(hash(node_type)) % 1000))



    def _encode_relation(self, relation: str) -> int:

        table = {

            "attn": 20,

            "modulates": 21,

            "derives": 22,

            "correlates": 23,

        }

        if not isinstance(relation, str):

            return 24

        return table.get(relation, 200 + (abs(hash(relation)) % 1000))



    def _encode_content(self, content: Any) -> np.ndarray:

        """Лёгкая кодировка содержимого узла в последовательность int32 без внешних зависимостей."""

        tokens: List[int] = []

        try:

            if content is None:

                return np.zeros(0, dtype=np.int32)

            if isinstance(content, (int, np.integer)):

                tokens.append(int(content) % 1_000_000)

            elif isinstance(content, float):

                tokens.append(int(abs(content) * 1e6) % 1_000_000)

            elif isinstance(content, str):

                # Простая символьная кодировка с отсечением

                for ch in content[:64]:

                    tokens.append(ord(ch) % 1024)

            elif isinstance(content, dict):

                # Упорядоченная пара ключ-значение

                for k in sorted(content.keys()):

                    kv = f"{k}:{content[k]}"

                    for ch in str(kv)[:64]:

                        tokens.append(ord(ch) % 1024)

            elif isinstance(content, (list, tuple)):

                for x in content[:32]:

                    tokens.extend(list(self._encode_content(x)))

            else:

                # Фоллбек: хэш строки представления

                tokens.append(300 + (abs(hash(str(content))) % 10_000))

        except (TypeError, ValueError, RecursionError) as e:

            logger.warning(f"Error encoding content {content}: {e}")

            tokens.append(999_999)

        return np.array(tokens, dtype=np.int32)



    # ----------------------- Вспомогательные методы -----------------------

    def _pack_layer_weights(self, layer_name: str, weights: np.ndarray, model_id: str) -> None:

        """

        Упаковывает веса слоя в фрактальную структуру на уровне 0.

        

        Математическая основа:

        - Базовый размер блока B = 64 элемента

        - Для слоя с N элементами создается C₀ = ⌈N/B⌉ контейнеров

        - Каждый контейнер содержит S₀ = B элементов

        

        Оптимизация:

        - Для критически важных слоев (embedding, выходные слои) используем float64

        - Для остальных слоев используем float32

        

        Args:

            layer_name: Имя слоя

            weights: Веса слоя в виде numpy массива

            model_id: ID модели

        """

        dtype = str(weights.dtype)

        shape = weights.shape

        flat_weights = weights.flatten()

        total_elements = len(flat_weights)

        

        # Определяем, является ли слой критически важным

        is_critical_layer = any(critical in layer_name for critical in 

                               ["wte", "wpe", "ln_f", "lm_head"])

        

        # Выбираем тип данных на основе важности слоя

        storage_dtype = "float64" if is_critical_layer else "float32"

        

        # Создаем контейнеры нулевого уровня

        logger.debug(f"Упаковка слоя {layer_name} ({shape}) в {storage_dtype}...")

        

        for i in range(0, total_elements, self.block_size):

            block_data = flat_weights[i:i + self.block_size]

            

            # Конвертируем в нужный тип данных

            if storage_dtype == "float64":

                block_data = block_data.astype(np.float64)

            else:

                block_data = block_data.astype(np.float32)

                

            block_shape = (len(block_data),)

            

            # Создаем ID контейнера

            position = (i // self.block_size,)

            container_id = self._generate_container_id(0, position, layer_name, model_id)

            

            # Создаем метаданные

            metadata = {

                "layer_name": layer_name,

                "model_id": model_id,

                "original_shape": shape,

                "block_start": i,

                "block_end": min(i + self.block_size, total_elements),

                "is_critical": is_critical_layer,

                "storage_dtype": storage_dtype,

                # Добавляем точные указатели на параметр

                "param_name": "weight",

                "tensor_path": f"{layer_name}.weight",

            }

            

            # Создаем контейнер

            container = FractalContainer(

                id=container_id,

                level=0,

                position=position,

                data=block_data,

                shape=block_shape,

                dtype=storage_dtype,

                metadata=metadata

            )

            

            # Сохраняем контейнер

            self.containers[container_id] = container

            if 0 not in self.fractal_tree:

                self.fractal_tree[0] = []

            self.fractal_tree[0].append(container_id)

            self.total_memory += container.get_memory_size()

            

            # Логируем каждые 1000 контейнеров для отслеживания прогресса

            if len(self.fractal_tree[0]) % 1000 == 0:

                logger.debug(f"Создано {len(self.fractal_tree[0])} контейнеров для уровня 0")



    def _generate_container_id(

        self, level: int, position: Tuple[int, ...], layer_name: str, model_id: str

    ) -> str:

        return f"{model_id}::{layer_name}::L{level}::pos{'-'.join(map(str, position))}"



    def _safe_quantize_to_int8(self, data: np.ndarray) -> Tuple[np.ndarray, float, bool]:

        """

        Безопасное квантование данных в int8 с обработкой NaN/Inf и нулевого масштаба.



        Returns:

            quantized: np.ndarray np.int8 той же формы

            scale: float масштаб квантования (>= 1e-12, либо 1.0 при нулевом массиве)

            has_zero_scale: bool признак, что весь блок был нулевым

        """

        # Заменяем NaN/Inf на безопасные значения и ограничиваем экстремумы

        arr = np.nan_to_num(data, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)

        if arr.size == 0:

            return arr.astype(np.int8, copy=False), 1.0, False



        max_abs = float(np.max(np.abs(arr)))

        if max_abs == 0.0:

            # Весь блок нулевой

            return np.zeros(arr.shape, dtype=np.int8), 1.0, True



        scale = max(max_abs / 127.0, 1e-12)

        normalized = arr / scale

        # Обрезаем и округляем, затем кастим в int8

        quantized = np.clip(np.round(normalized), -127, 127).astype(np.int8, copy=False)

        return quantized, scale, False



    def _build_fractal_hierarchy(self) -> None:

        """

        Строит иерархию фрактала на основе контейнеров нулевого уровня.



        Математическая основа:

        - Количество контейнеров на уровне i: C_i = ceil(C_{i-1}/k)

        - Размер контейнера на уровне i: S_i = k^i * B

        - Где k = self.containers_per_group, B = self.block_size



        Типы данных:

        - уровень 1: float32

        - уровень 2: float16

        - уровни >=3: int8 с масштабом (quant_scale)

        """

        if 0 not in self.fractal_tree:

            self.fractal_tree[0] = []



        logger.info(f"Построение фрактальной иерархии ({self.fractal_levels} уровней)...")



        for level in range(1, self.fractal_levels):

            parent_containers = self.fractal_tree.get(level - 1, [])

            if not parent_containers:

                # Нечего агрегировать

                continue

            logger.debug(f"Построение уровня {level} из {len(parent_containers)} контейнеров...")



            k = self.containers_per_group

            total_groups = (len(parent_containers) + k - 1) // k

            progress_interval = max(1, total_groups // 10)



            for i in range(0, len(parent_containers), k):

                group = parent_containers[i : i + k]

                if not group:

                    continue



                # Прогресс

                grp_index = i // k

                if grp_index % progress_interval == 0:

                    progress = (grp_index + 1) / total_groups * 100.0

                    logger.debug(f"Уровень {level}: {progress:.1f}% завершено")



                position = (grp_index,)



                # Базовые метаданные берём из первого ребёнка

                first_child = self.containers[group[0]]

                layer_name = first_child.metadata.get("layer_name", "")

                model_id = first_child.metadata.get("model_id", self.model_id or "")

                container_id = self._generate_container_id(level, position, layer_name, model_id)



                # Собираем данные

                child_data_segments: List[np.ndarray] = []

                for child_id in group:

                    child = self.containers[child_id]

                    child_data_segments.append(child.data)

                    child.parent = container_id



                try:

                    combined_data = np.concatenate(child_data_segments, axis=0)

                except Exception:

                    # В редких случаях несовпадения формы — приводим к вектору

                    combined_data = np.concatenate([seg.reshape(-1) for seg in child_data_segments], axis=0)



                # Определяем dtype на текущем уровне

                if level == 1:

                    storage_dtype = "float32"

                elif level == 2:

                    storage_dtype = "float16"

                else:

                    storage_dtype = "int8"



                scale_val: Optional[float] = None

                zero_scale_flag: bool = False

                if storage_dtype == "float32":

                    # Защита от NaN/Inf и явное приведение к fp32

                    f32_max = np.finfo(np.float32).max

                    f32_min = np.finfo(np.float32).min

                    arr = np.nan_to_num(combined_data, copy=False, nan=0.0, posinf=f32_max, neginf=f32_min)

                    combined_data = arr.astype(np.float32, copy=False)

                elif storage_dtype == "float16":

                    # Устойчивое приведение к fp16: используем значения, совместимые с диапазоном float16

                    # float16 max ~ 65504, выберем немного меньше для запаса

                    f16_max = 6.5e4

                    f16_min = -6.5e4

                    arr = np.nan_to_num(combined_data, copy=False, nan=0.0, posinf=f16_max, neginf=f16_min)

                    arr = np.clip(arr, f16_min, f16_max)

                    combined_data = arr.astype(np.float16, copy=False)

                elif storage_dtype == "int8":

                    # Устойчивое квантование через хелпер

                    quant_f, scale_val, zero_scale_flag = self._safe_quantize_to_int8(combined_data)

                    combined_data = quant_f



                metadata = {

                    "layer_name": layer_name,

                    "model_id": model_id,

                    "child_count": len(group),

                    "child_ids": group,

                    "storage_dtype": storage_dtype,

                }

                # Протягиваем точные указатели на параметр из первого потомка, если доступны

                if isinstance(first_child, FractalContainer):

                    tp = first_child.metadata.get("tensor_path")

                    if tp is not None:

                        metadata["tensor_path"] = tp

                    pn = first_child.metadata.get("param_name")

                    if pn is not None:

                        metadata["param_name"] = pn

                if scale_val is not None:

                    # Дублируем ключ для совместимости с внешними сценариями

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



    def _initialize_hot_window(self) -> None:

        """

        Инициализирует горячее окно на основе фрактальной структуры.

        Стратегия: верхний уровень, часть средних, критические слои уровня 0.

        Приоритет: P = L_p * T_p * C_p

        """

        logger.debug("Инициализация горячего окна фрактальной памяти...")



        candidates: List[Tuple[str, float]] = []



        # 1) Самый высокий уровень

        top_level = self.fractal_levels - 1

        if top_level in self.fractal_tree:

            for container_id in self.fractal_tree[top_level]:

                priority = self._calculate_container_priority(container_id)

                candidates.append((container_id, priority))



        # 2) Средние уровни (берём крайние несколько)

        if self.fractal_levels > 2:

            mid_level = self.fractal_levels // 2

            if mid_level in self.fractal_tree and self.fractal_tree[mid_level]:

                mids = self.fractal_tree[mid_level]

                sample_ids = mids[:3] + (mids[-3:] if len(mids) > 3 else [])

                for container_id in sample_ids:

                    priority = self._calculate_container_priority(container_id)

                    candidates.append((container_id, priority))



        # 3) Критические блоки уровня 0

        if 0 in self.fractal_tree:

            for container_id in self.fractal_tree[0]:

                container = self.containers[container_id]

                if container.metadata.get("is_critical", False):

                    priority = self._calculate_container_priority(container_id)

                    candidates.append((container_id, priority))



        candidates.sort(key=lambda x: x[1], reverse=True)



        # Заполнение окна до лимита

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

                # Материализуем данные контейнера в GPU-кэш при необходимости

                try:

                    if getattr(self, "device", "cpu") == "cuda" and torch.cuda.is_available():

                        if container_id not in self.gpu_hot_cache:

                            np_arr = container.data  # np.ndarray

                            if container.dtype == "float8":

                                tensor = torch.from_numpy(np_arr.astype(np.int8, copy=False))

                            elif container.dtype == "float16":

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



        # Если менее 50% — пробуем расширить за счёт других уровней

        if hot_window_size_mb < (self.hot_window_size * 0.5) / (1024 * 1024):

            logger.warning("Горячее окно заполнено менее чем на 50%. Добавляем дополнительные контейнеры...")

            self._expand_hot_window(current_size)



    def _calculate_container_priority(self, container_id: str) -> float:

        """P = L_p * T_p * C_p, где:

        L_p = 0.9^level; T_p = 0.1 + 0.9*(1 - min(t/3600,1)); C_p = 0.3 для критических, иначе 1.0

        t — время с последнего доступа в секундах.

        """

        cont = self.containers.get(container_id)

        if cont is None:

            return 0.0

        level_factor = pow(0.9, max(0, int(cont.level)))

        t = max(0.0, time.time() - float(cont.last_accessed or 0.0))

        time_factor = 0.1 + 0.9 * (1.0 - min(t / 3600.0, 1.0))

        crit = 0.3 if cont.metadata.get("is_critical", False) else 1.0

        return float(level_factor * time_factor * crit)



    def _expand_hot_window(self, current_size: int) -> None:

        """Добираем контейнеры из остальных уровней по убыванию приоритета до 50% окна или исчерпания."""

        if current_size >= self.hot_window_size * 0.5:

            return

        # Собираем всех кандидатов, которых ещё нет в окне

        seen = set(self.hot_window.keys())

        candidates: List[Tuple[str, float]] = []

        for level, ids in sorted(self.fractal_tree.items(), key=lambda x: x[0], reverse=True):

            for cid in ids:

                if cid in seen:

                    continue

                pr = self._calculate_container_priority(cid)

                candidates.append((cid, pr))

        candidates.sort(key=lambda x: x[1], reverse=True)



        size = current_size

        now_ts = time.time()

        for cid, pr in candidates:

            cont = self.containers.get(cid)

            if cont is None:

                continue

            csz = cont.get_memory_size()

            if size + csz > self.hot_window_size:

                continue

            self.hot_window[cid] = float(pr)

            size += csz

            cont.last_accessed = now_ts

            cont.access_count += 1

            # Материализация на GPU

            try:

                if getattr(self, "device", "cpu") == "cuda" and torch.cuda.is_available():

                    if cid not in self.gpu_hot_cache:

                        np_arr = cont.data

                        if cont.dtype == "float8":

                            tensor = torch.from_numpy(np_arr.astype(np.int8, copy=False))

                        elif cont.dtype == "float16":

                            tensor = torch.from_numpy(np_arr.astype(np.float16, copy=False))

                        elif cont.dtype == "float32":

                            tensor = torch.from_numpy(np_arr.astype(np.float32, copy=False))

                        else:

                            tensor = torch.from_numpy(np_arr)

                        self.gpu_hot_cache[cid] = tensor.to("cuda", non_blocking=True)

            except Exception:

                pass

            if size >= self.hot_window_size * 0.5:

                break



    def _optimize_fractal_structure(self) -> None:

        """

        Оптимизирует фрактальную структуру для повышения эффективности.



        Алгоритм:

        1. Анализирует использование контейнеров

        2. Перестраивает фрактальную иерархию при необходимости

        3. Оптимизирует расположение контейнеров

        4. Обновляет метаданные



        Особенности:

        - Динамическая реконфигурация структуры

        - Оптимизация под текущие паттерны использования

        - Минимизация фрагментации

        - Самообучение системы



        Математическая основа:

        - Коэффициент пространственной локальности: L_spatial = число последовательных доступов / общее число доступов

        - Оптимальная структура максимизирует L_spatial

        """

        logger.info("Оптимизация фрактальной структуры...")



        # 1. Анализируем использование контейнеров

        usage_stats = self._analyze_container_usage()



        # 2. Проверяем, нужна ли реконфигурация

        if self._needs_reconfiguration(usage_stats):

            logger.info("Требуется реконфигурация фрактальной структуры...")

            self._reconfigure_fractal_structure(usage_stats)



        # 3. Оптимизируем расположение контейнеров (реалайн в рамках текущей иерархии)

        logger.info("Оптимизация расположения контейнеров...")

        try:

            # Простая эвристика: пересчитать приоритеты и перезаполнить горячее окно

            for cid in list(self.containers.keys()):

                pr = self._calculate_container_priority(cid)

                c = self.containers.get(cid)

                if c is not None:

                    c.update_priority(pr)

            self._initialize_hot_window()

        except Exception as e:

            logger.warning(f"Не удалось оптимизировать расположение контейнеров: {e}")



        # 4. Обновляем метаданные

        logger.info("Обновление метаданных...")

        self._update_metadata()



    def _analyze_container_usage(self) -> Dict[str, Any]:

        """

        Анализирует использование контейнеров.



        Собирает статистику:

        - Частота доступа

        - Пространственная локальность

        - Временная локальность

        - Паттерны использования



        Returns:

            Dict[str, Any]: Статистика использования

        """

        stats: Dict[str, Any] = {

            "access_frequency": defaultdict(int),

            "spatial_locality": 0.0,

            "temporal_locality": 0.0,

            "access_pattern": [],

            "last_access": {},

        }



        now_ts = time.time()

        # Собираем статистику по всем контейнерам

        for container_id, container in self.containers.items():

            # Частота доступа

            stats["access_frequency"][container_id] = int(getattr(container, "access_count", 0))



            # Последнее время доступа

            stats["last_access"][container_id] = float(getattr(container, "last_accessed", 0.0))



            # Добавляем в историю доступа (за последний час)

            if float(container.last_accessed or 0.0) > now_ts - 3600.0:

                stats["access_pattern"].append(container_id)



        # Вычисляем пространственную локальность

        if len(stats["access_pattern"]) > 1:

            sequential_accesses = 0

            for i in range(1, len(stats["access_pattern"])):

                prev = stats["access_pattern"][i - 1]

                curr = stats["access_pattern"][i]

                if self._are_containers_adjacent(prev, curr):

                    sequential_accesses += 1

            stats["spatial_locality"] = sequential_accesses / float(len(stats["access_pattern"]) - 1)



        # Вычисляем временную локальность: повторные доступы к тем же контейнерам за 5 минут

        recent = [cid for cid, ts in stats["last_access"].items() if (now_ts - float(ts)) < 300.0]

        if recent:

            unique_recent = len(set(recent))

            stats["temporal_locality"] = 1.0 - (unique_recent / float(len(recent)))



        return stats



    def _are_containers_adjacent(self, cid_a: str, cid_b: str) -> bool:

        """Соседство: один уровень, один слой, позиция отличается на 1 либо общий родитель.

        Падает в более широкую эвристику при отсутствии метаданных.

        """

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

            # Общий родитель в иерархии

            if a.parent and a.parent == b.parent and a.parent is not None:

                return True

        except Exception:

            return False

        return False



    def _needs_reconfiguration(self, usage_stats: Dict[str, Any]) -> bool:

        """

        Определяет, требуется ли реконфигурация фрактальной структуры.



        Критерии:

        - Низкая пространственная локальность (< 0.4)

        - Низкая временная локальность (< 0.3)

        - Высокая фрагментация

        - Изменение паттернов использования

        """

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

        """Оценка фрагментации по «дырам» в последовательностях позиций для каждого слоя и уровня 0.

        Возвращает среднее относительное число разрывов.

        """

        try:

            per_layer: List[float] = []

            # Анализируем уровень 0, где последовательность блоков слоя критична

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

        """Сравнивает текущую сводку использования с предыдущей.

        Если изменение метрик выше порога — считаем, что паттерны изменились.

        """

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

        """

        Перестраивает фрактальную структуру на основе статистики использования.



        Алгоритм:

        1. Определяет новые параметры фрактала

        2. Перепаковывает данные с новыми параметрами

        3. Сохраняет историю изменений

        """

        # Определяем новые параметры

        new_params = self._determine_new_parameters(usage_stats)



        logger.info(

            f"Реконфигурация фрактальной структуры: уровни={new_params['fractal_levels']}, блок={new_params['block_size']}"

        )



        # Сохраняем текущие данные (восстановление слоев из уровня 0)

        all_data = self._extract_all_data()



        # Очищаем текущую структуру

        self.containers.clear()

        self.fractal_tree.clear()

        self.hot_window.clear()

        self.total_memory = 0



        # Обновляем параметры

        self.fractal_levels = int(new_params["fractal_levels"])

        self.block_size = int(new_params["block_size"])



        # Перепаковываем данные

        model_id = self.model_id or "fractal"

        for layer_name, weights in all_data.items():

            try:

                self._pack_layer_weights(layer_name, weights, model_id)

            except Exception as e:

                logger.error(f"Не удалось перепаковать слой {layer_name}: {e}")



        # Строим новую иерархию и горячее окно

        self._build_fractal_hierarchy()

        self._initialize_hot_window()



        # Сохраняем историю изменений

        self._save_reconfiguration_history(new_params)



        logger.info("Фрактальная структура успешно реконфигурирована")



    def _determine_new_parameters(self, usage_stats: Dict[str, Any]) -> Dict[str, Any]:

        """Определяет новые параметры фрактала на основе статистики использования."""

        new_params = {

            "fractal_levels": int(self.fractal_levels),

            "block_size": int(self.block_size),

        }



        # Регулируем количество уровней (пространственная локальность)

        spatial = float(usage_stats.get("spatial_locality", 0.0))

        if spatial < 0.4:

            new_params["fractal_levels"] = min(6, int(self.fractal_levels) + 1)

        elif spatial > 0.6:

            new_params["fractal_levels"] = max(3, int(self.fractal_levels) - 1)



        # Регулируем размер блока (временная локальность)

        temporal = float(usage_stats.get("temporal_locality", 0.0))

        if temporal < 0.3:

            new_params["block_size"] = max(16, int(self.block_size) // 2)

        elif temporal > 0.5:

            new_params["block_size"] = min(128, int(self.block_size) * 2)



        return new_params



    def _extract_all_data(self) -> Dict[str, np.ndarray]:

        """Восстанавливает тензоры слоёв из контейнеров уровня 0 по layer_name и original_shape."""

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

                result[layer] = arr  # фоллбек

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

        """Сохраняет запись об изменении конфигурации в памяти (и при наличии — на диск)."""

        try:

            rec = {

                "ts": time.time(),

                "params": dict(params),

                "stats": self.get_statistics() if hasattr(self, "get_statistics") else {},

            }

            hist = getattr(self, "reconfiguration_history", None)

            if not isinstance(hist, list):

                self.reconfiguration_history = []  # type: ignore[attr-defined]

            self.reconfiguration_history.append(rec)  # type: ignore[attr-defined]

        except Exception:

            pass



    # ----------------------- Доступ к контейнерам -----------------------

    def get_container(self, container_id: str, load_from_disk: bool = True) -> Optional[FractalContainer]:

        """

        Получает контейнер по ID с динамической загрузкой в горячее окно.

        """

        if container_id not in self.containers:

            logger.warning(f"Контейнер {container_id} не найден")

            return None



        container = self.containers[container_id]



        # Обновляем статистику доступа

        container.last_accessed = time.time()

        container.access_count += 1



        # Если уже в горячем окне — возвращаем

        if container_id in self.hot_window:

            return container



        # Приоритет

        base_priority = self._calculate_container_priority(container_id)

        container.update_priority(base_priority)



        required_space = container.get_memory_size()

        available_space = self._get_available_hot_window_space()



        if required_space <= available_space:

            self.hot_window[container_id] = container.priority

            logger.debug(

                f"Контейнер {container_id} добавлен в горячее окно (уровень {container.level})"

            )

            return container



        logger.debug(

            f"Недостаточно места в горячем окне для контейнера {container_id}. Вытеснение..."

        )

        self._evict_lowest_priority_containers(required_space - available_space)



        if required_space <= self._get_available_hot_window_space():

            self.hot_window[container_id] = container.priority

            logger.debug(

                f"Контейнер {container_id} добавлен в горячее окно после вытеснения (уровень {container.level})"

            )

            return container



        if load_from_disk and ("offload_path" in container.metadata):

            logger.debug(f"Загрузка контейнера {container_id} с диска...")

            if self._load_container_from_ssd(container_id):

                # Повторная попытка

                return self.get_container(container_id, load_from_disk=False)



        logger.warning(f"Не удалось добавить контейнер {container_id} в горячее окно")

        return None



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

        # Сортируем текущие элементы окна по возрастанию приоритета

        items = list(self.hot_window.items())

        items.sort(key=lambda x: x[1])

        freed = 0

        for cid, pr in items:

            cont = self.containers.get(cid)

            size = cont.get_memory_size() if cont is not None else 0

            # удаляем из окна

            self.hot_window.pop(cid, None)

            # удаляем из GPU-кэша, если присутствует

            try:

                if hasattr(self, "gpu_hot_cache") and cid in self.gpu_hot_cache:

                    self.gpu_hot_cache.pop(cid, None)

            except Exception:

                pass

            freed += size

            if freed >= bytes_needed:

                break



    def _load_container_from_ssd(self, container_id: str) -> bool:

        """Загружает данные контейнера из offload-пути, если доступно. Заглушка."""

        cont = self.containers.get(container_id)

        if cont is None:

            return False

        off_path = cont.metadata.get("offload_path")

        if not off_path:

            return False

        try:

            # Здесь могла бы быть mmap/np.load. Пока заглушка: возвращаем False, чтобы не менять поведение.

            return False

        except Exception:

            logger.exception(f"Ошибка загрузки контейнера {container_id} с диска")

            return False



    def get_statistics(self) -> Dict[str, Any]:

        total_containers = sum(len(v) for v in self.fractal_tree.values())

        total_memory_mb = self.total_memory / (1024 * 1024) if self.total_memory else 0.0

        containers_by_level = {int(level): int(len(ids)) for level, ids in self.fractal_tree.items()}

        # Без факта реального сжатия пока считаем коэффициент 1.0

        stats = {

            "model_id": self.model_id,

            "total_containers": total_containers,

            "containers_by_level": containers_by_level,

            "total_memory_bytes": int(self.total_memory),

            "total_memory_mb": float(total_memory_mb),

            "compression_ratio": 1.0,

            "levels": sorted(self.fractal_tree.keys()),

        }

        return stats



    # ----------------------- Диск / транзакции / валидация -----------------------

    def compute_checksum(self) -> str:

        """Возвращает агрегированный SHA256 всех контейнеров (id + data) для контроля целостности."""

        import hashlib

        h = hashlib.sha256()

        try:

            # Фиксированный порядок: по уровням и id

            for level in sorted(self.fractal_tree.keys()):

                for cid in sorted(self.fractal_tree[level]):

                    cont = self.containers.get(cid)

                    if cont is None:

                        continue

                    h.update(cid.encode("utf-8", errors="ignore"))

                    try:

                        h.update(cont.data.tobytes())

                    except Exception:

                        # Фоллбек через копию в np.ndarray

                        h.update(np.array(cont.data).tobytes())

            return h.hexdigest()

        except Exception:

            return ""



    def validate_knowledge_graph_packing(self) -> Dict[str, Any]:

        """Валидирует упаковку графа знаний на уровне 0: проверяет непрерывность блоков и метки источника.



        Returns:

            { ok: bool, issues: List[str], total_blocks: int, total_length: int }

        """

        issues: List[str] = []

        try:

            # Выбираем блоки уровня 0, созданные pack_knowledge_graph (source == 'knowledge_graph')

            blocks: List[Tuple[int, int]] = []

            for cid in self.fractal_tree.get(0, []):

                c = self.containers.get(cid)

                if c is None:

                    continue

                if c.metadata.get("source") == "knowledge_graph":

                    bs = int(c.metadata.get("block_start", -1))

                    be = int(c.metadata.get("block_end", -1))

                    if bs < 0 or be <= bs:

                        issues.append(f"invalid_block_meta:{cid}:{bs}-{be}")

                    else:

                        blocks.append((bs, be))



            if not blocks:

                issues.append("no_kg_blocks_found")

                return {"ok": False, "issues": issues, "total_blocks": 0, "total_length": 0}



            # Проверяем непрерывность диапазонов [0..max_end)

            blocks.sort(key=lambda x: x[0])

            expected = 0

            for (bs, be) in blocks:

                if bs != expected:

                    issues.append(f"gap_or_overlap_at:{expected}->{bs}")

                    expected = be

                else:

                    expected = be

            total_len = blocks[-1][1]

            ok = len([i for i in issues if i.startswith("gap_or_overlap")]) == 0

            return {"ok": ok and len(issues) == 0, "issues": issues, "total_blocks": len(blocks), "total_length": total_len}

        except Exception as e:

            return {"ok": False, "issues": [f"exception:{e}"], "total_blocks": 0, "total_length": 0}



    def save_to_disk_atomic(self, target_dir: str) -> Dict[str, Any]:

        """Сохраняет состояние фрактального хранилища на диск атомарно.



        Структура директории:

          target_dir/

            index.json             # метаданные, дерево, графовые метаданные, контрольная сумма

            data/<cid>.npy         # данные контейнеров уровня 0..N



        Возвращает отчёт: { ok: bool, path: str, checksum: str, error?: str }

        """

        import tempfile

        import shutil



        try:

            target = Path(target_dir)

            tmp_dir = Path(tempfile.mkdtemp(prefix="fractal_store_"))

            data_dir = tmp_dir / "data"

            data_dir.mkdir(parents=True, exist_ok=True)



            # Пишем контейнеры в .npy

            for level in sorted(self.fractal_tree.keys()):

                for cid in self.fractal_tree[level]:

                    cont = self.containers.get(cid)

                    if cont is None:

                        continue

                    fn = data_dir / f"{self._safe_name(cid)}.npy"

                    try:

                        np.save(fn, cont.data, allow_pickle=False)

                    except Exception:

                        # Фоллбек через np.array

                        np.save(fn, np.array(cont.data), allow_pickle=False)



            # Индекс

            index = {

                "model_id": self.model_id,

                "fractal_levels": int(self.fractal_levels),

                "block_size": int(self.block_size),

                "containers": [],

                "fractal_tree": {str(k): list(v) for k, v in self.fractal_tree.items()},

                "graph_metadata": dict(self.graph_metadata),

            }

            for cid, c in self.containers.items():

                index["containers"].append(

                    {

                        "id": cid,

                        "level": int(c.level),

                        "position": list(c.position),

                        "shape": list(c.shape),

                        "dtype": str(c.dtype),

                        "metadata": c.metadata,

                        "file": f"data/{self._safe_name(cid)}.npy",

                    }

                )



            checksum = self.compute_checksum()

            index["checksum"] = checksum



            with open(tmp_dir / "index.json", "w", encoding="utf-8") as f:

                json.dump(index, f, ensure_ascii=False, indent=2)



            # Атомарная замена: переносим tmp_dir -> target_dir

            if target.exists():

                backup = target.with_suffix(".bak")

                try:

                    if backup.exists():

                        shutil.rmtree(backup, ignore_errors=True)

                    target.rename(backup)

                except Exception:

                    pass

                try:

                    shutil.rmtree(target, ignore_errors=True)

                except Exception:

                    pass

            shutil.move(str(tmp_dir), str(target))

            # Чистим возможный backup

            backup = target.with_suffix(".bak")

            if backup.exists():

                shutil.rmtree(backup, ignore_errors=True)



            return {"ok": True, "path": str(target), "checksum": checksum}

        except Exception as e:

            try:

                # На случай, если tmp_dir создан

                if 'tmp_dir' in locals():

                    shutil.rmtree(tmp_dir, ignore_errors=True)

            except Exception:

                pass

            return {"ok": False, "path": str(target_dir), "checksum": "", "error": str(e)}



    def _load_from_disk_atomic_format(self, source_dir: str) -> Dict[str, Any]:

        """Загружает состояние фрактального хранилища (atomic-формат: index.json + data/*.npy).



        Возвращает отчёт: { ok: bool, checksum: str, error?: str } и заполняет состояние объекта.

        """

        try:

            base = Path(source_dir)

            with open(base / "index.json", "r", encoding="utf-8") as f:

                index = json.load(f)



            # Сбрасываем текущее

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

                # Убеждаемся, что форма согласована

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



            # Пересчитываем total_memory и окно

            self._update_metadata()

            self._initialize_hot_window()



            disk_checksum = index.get("checksum", "")

            mem_checksum = self.compute_checksum()

            if disk_checksum and mem_checksum and disk_checksum != mem_checksum:

                return {"ok": False, "checksum": mem_checksum, "error": "checksum_mismatch"}

            return {"ok": True, "checksum": mem_checksum or disk_checksum}

        except Exception as e:

            return {"ok": False, "checksum": "", "error": str(e)}



    def _safe_name(self, s: str) -> str:

        """Безопасное имя для файла контейнера."""

        return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in s)[:200]



    def clear(self) -> None:

        self.containers.clear()

        self.fractal_tree.clear()

        self.hot_window.clear()

        self.total_memory = 0

        self.model_id = None

        # Очистка GPU-кэша горячего окна

        try:

            if hasattr(self, "gpu_hot_cache"):

                self.gpu_hot_cache.clear()

            if getattr(self, "device", "cpu") == "cuda" and torch.cuda.is_available():

                torch.cuda.empty_cache()

        except Exception:

            pass



    # ----------------------- Граф памяти: атомарный манифест -----------------------

    def save_graph_manifest_atomic(

        self,

        manifest_dir: str,

        records: "Iterable[Dict[str, Any]]",

        meta: Optional[Dict[str, Any]] = None,

        manifest_filename: str = "manifest.jsonl",

        meta_filename: str = "manifest_meta.json",

    ) -> Dict[str, Any]:

        """Атомарно сохраняет манифест графа памяти рядом с фрактальными данными.



        Структура:

          manifest_dir/

            manifest.jsonl       # построчные записи узлов/ребер графа памяти

            manifest_meta.json   # метаданные манифеста (версия, ссылки, модель)



        Args:

            manifest_dir: каталог для манифеста

            records: итератор записей (dict) формата node/edge

            meta: дополнительные метаданные манифеста

            manifest_filename: имя файла для jsonl

            meta_filename: имя файла для метаданных



        Returns:

            dict: { ok: bool, path: str, count: int, error?: str }

        """

        import os

        base = Path(manifest_dir)

        try:

            base.mkdir(parents=True, exist_ok=True)

            mf_path = base / manifest_filename

            tmp_path = mf_path.with_suffix(mf_path.suffix + ".tmp")



            count = 0

            # Пишем jsonl во временный файл с flush/fsync

            with open(tmp_path, "w", encoding="utf-8") as f:

                for rec in records:

                    try:

                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                        count += 1

                    except Exception:

                        # Пропускаем некорректные записи, но продолжаем

                        continue

                try:

                    f.flush()

                    os.fsync(f.fileno())

                except Exception:

                    pass

            os.replace(tmp_path, mf_path)



            # Метаданные манифеста

            meta_obj: Dict[str, Any] = {

                "version": 1,

                "created_ts": time.time(),

                "model_id": self.model_id,

            }

            if meta:

                try:

                    meta_obj.update(dict(meta))

                except Exception:

                    pass

            try:

                with open(base / meta_filename, "w", encoding="utf-8") as mf:

                    json.dump(meta_obj, mf, ensure_ascii=False, indent=2)

            except Exception:

                # Метаданные не критичны для консистентности jsonl

                pass



            return {"ok": True, "path": str(base), "count": count}

        except KeyboardInterrupt:

            try:

                if 'tmp_path' in locals() and Path(tmp_path).exists():

                    Path(tmp_path).unlink(missing_ok=True)

            except Exception:

                pass

            raise

        except Exception as e:

            try:

                if 'tmp_path' in locals() and Path(tmp_path).exists():

                    Path(tmp_path).unlink(missing_ok=True)

            except Exception:

                pass

            return {"ok": False, "path": str(manifest_dir), "count": 0, "error": str(e)}



    def load_graph_manifest(

        self,

        manifest_dir: str,

        manifest_filename: str = "manifest.jsonl",

        meta_filename: str = "manifest_meta.json",

        limit: Optional[int] = None,

    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

        """Загружает манифест графа памяти.



        Args:

            manifest_dir: каталог, где лежит манифест

            manifest_filename: имя jsonl

            meta_filename: имя meta json

            limit: максимум строк (для предварительного чтения/диагностики)



        Returns:

            Tuple[List[dict], Dict]: (records, meta)

        """

        base = Path(manifest_dir)

        mf_path = base / manifest_filename

        meta_path = base / meta_filename

        records: List[Dict[str, Any]] = []

        meta: Dict[str, Any] = {}

        try:

            if meta_path.exists():

                try:

                    with open(meta_path, "r", encoding="utf-8") as f:

                        meta = json.load(f) or {}

                except Exception:

                    meta = {}

            if not mf_path.exists():

                return (records, meta)

            with open(mf_path, "r", encoding="utf-8") as f:

                for i, line in enumerate(f):

                    if limit is not None and i >= int(limit):

                        break

                    line = line.strip()

                    if not line:

                        continue

                    try:

                        records.append(json.loads(line))

                    except Exception:

                        continue

            return (records, meta)

        except Exception:

            # Безопасный возврат даже при частичных сбоях

            return (records, meta)



    def build_shards_index(self, output_dir: str) -> Dict[str, Dict[str, str]]:

        """Строит индекс по `shards_manifest.jsonl`: id -> {shard_file, key}.



        Args:

            output_dir: каталог с результатами шардированного сохранения



        Returns:

            dict: mapping container_id -> {"shard_file": str, "key": str}

        """

        base = Path(output_dir)

        manifest = base / "shards_manifest.jsonl"

        index: Dict[str, Dict[str, str]] = {}

        if not manifest.exists():

            return index

        try:

            with manifest.open("r", encoding="utf-8") as mf:

                for line in mf:

                    line = line.strip()

                    if not line:

                        continue

                    try:

                        rec = json.loads(line)

                        cid = rec.get("id")

                        shard_file = rec.get("shard_file")

                        key = rec.get("key")

                        if cid and shard_file and key:

                            index[str(cid)] = {"shard_file": str(shard_file), "key": str(key)}

                    except Exception:

                        continue

        except Exception:

            pass

        return index



    # ----------------------- Извлечение знаний -----------------------

    def extract_knowledge_from_model(self, model: torch.nn.Module) -> List[Dict]:

        """

        Извлекает структурированные знания из весов модели (лёгкая эвристика).

        Методы: внимание (отношения), FFN (факты), эмбеддинги (концепты).

        """

        logger.info("Начато извлечение знаний из модели...")

        start_time = time.time()

        knowledge: List[Dict[str, Any]] = []



        # 1) Внимание

        logger.debug("Анализ слоев внимания...")

        for layer_name, layer in model.named_modules():

            if ("attn" in layer_name) or ("attention" in layer_name):

                try:

                    relations = self._analyze_attention_weights(layer, layer_name)

                    knowledge.extend(relations)

                except Exception:

                    logger.debug(f"Пропуск анализа внимания для {layer_name}", exc_info=True)



        # 2) FFN

        logger.debug("Анализ feed-forward сетей...")

        for layer_name, layer in model.named_modules():

            if ("mlp" in layer_name) or ("ffn" in layer_name) or ("feedforward" in layer_name):

                try:

                    facts = self._analyze_ffn_weights(layer, layer_name)

                    knowledge.extend(facts)

                except Exception:

                    logger.debug(f"Пропуск анализа FFN для {layer_name}", exc_info=True)



        # 3) Эмбеддинги

        logger.debug("Анализ эмбеддингов...")

        for layer_name, layer in model.named_modules():

            if ("embed" in layer_name) or ("wte" in layer_name) or ("wpe" in layer_name):

                try:

                    concepts = self._analyze_embeddings(layer, layer_name)

                    knowledge.extend(concepts)

                except Exception:

                    logger.debug(f"Пропуск анализа эмбеддингов для {layer_name}", exc_info=True)



        # 4) Оценка надёжности

        logger.debug("Оценка надежности знаний...")

        self._evaluate_knowledge_reliability(knowledge)



        # 5) Фильтрация

        initial_count = len(knowledge)

        filtered = [k for k in knowledge if float(k.get("reliability", 0.0)) >= 0.5]

        filtered_count = initial_count - len(filtered)

        logger.info(

            f"Извлечение знаний завершено за {time.time() - start_time:.2f} сек. "

            f"Извлечено {len(filtered)} знаний (отфильтровано {filtered_count})."

        )

        return filtered



    def _analyze_attention_weights(self, layer: torch.nn.Module, layer_name: str) -> List[Dict]:

        """

        Анализирует веса внимания: если доступны q_proj/k_proj и num_heads — строит отношения по головам,

        иначе fallback на норму весов как ранее.

        """

        results: List[Dict[str, Any]] = []



        has_qkv = all(hasattr(layer, attr) for attr in ("q_proj", "k_proj", "v_proj"))

        num_heads = getattr(layer, "num_heads", None)

        if has_qkv and isinstance(num_heads, int) and num_heads > 0:

            try:

                q_weights = layer.q_proj.weight.data.detach().cpu().numpy()

                k_weights = layer.k_proj.weight.data.detach().cpu().numpy()

                # v_weights = layer.v_proj.weight.data.detach().cpu().numpy()  # не используется напрямую здесь



                d_model = q_weights.shape[1]

                head_dim = max(1, d_model // num_heads)



                for head in range(num_heads):

                    start_idx = head * head_dim

                    end_idx = min((head + 1) * head_dim, d_model)

                    q_head = q_weights[:, start_idx:end_idx]

                    k_head = k_weights[:, start_idx:end_idx]



                    # Приближённая матрица внимания и softmax по строкам

                    attn_matrix = np.dot(q_head, k_head.T) / np.sqrt(float(head_dim))

                    attn_matrix = self._softmax(attn_matrix)



                    threshold = 0.7

                    strong_i, strong_j = np.where(attn_matrix > threshold)

                    for i, j in zip(strong_i, strong_j):

                        if i == j:

                            continue

                        strength = float(attn_matrix[i, j])

                        results.append(

                            {

                                "type": "attention_relation",

                                "subject": f"token_{i}",

                                "object": f"token_{j}",

                                "predicate": f"attention_head_{head}",

                                "strength": strength,

                                "score": strength,

                                "source_layer": layer.__class__.__name__,

                                "layer": layer_name,

                                "reliability": 0.0,

                            }

                        )

                logger.debug(f"Из слоя внимания {layer_name} извлечено {len(results)} отношений")

                return results

            except Exception as e:

                logger.error(f"Ошибка анализа весов внимания ({layer_name}): {e}")

                # падаем в фолбэк ниже



        # Fallback: оценка по нормам весов

        q = getattr(layer, "q_proj", None)

        k = getattr(layer, "k_proj", None)

        v = getattr(layer, "v_proj", None)

        matrices = [("Q", q), ("K", k), ("V", v)]

        norms: Dict[str, float] = {}

        for tag, mod in matrices:

            try:

                w = getattr(mod, "weight", None)

                if isinstance(w, torch.nn.Parameter):

                    norms[tag] = float(w.detach().abs().mean().cpu().item())

            except Exception:

                continue

        if not norms:

            w = getattr(layer, "weight", None)

            if isinstance(w, torch.nn.Parameter):

                norms["W"] = float(w.detach().abs().mean().cpu().item())



        score = float(sum(norms.values()) / max(1, len(norms))) if norms else 0.0

        if score > 0:

            results.append(

                {

                    "type": "attention_relation",

                    "layer": layer_name,

                    "details": {"norms": norms},

                    "score": score,

                    "reliability": 0.0,

                }

            )

        return results



    def _softmax(self, x: np.ndarray) -> np.ndarray:

        """Приближённый softmax по строкам матрицы."""

        x = x.astype(np.float64, copy=False)

        x = x - np.max(x, axis=1, keepdims=True)

        e_x = np.exp(x)

        denom = np.sum(e_x, axis=1, keepdims=True) + 1e-12

        return e_x / denom



    def _analyze_ffn_weights(self, layer: torch.nn.Module, layer_name: str) -> List[Dict]:

        """

        Анализирует FFN: если слой имеет компоненты c_fc/c_proj — извлекаем паттерны по W1/W2,

        иначе fallback: оцениваем линейные подслои по mean(|W|).

        """

        # Структурный путь (например GPT2Block.mlp: c_fc/c_proj)

        if hasattr(layer, "c_fc") and hasattr(layer, "c_proj"):

            try:

                w1 = layer.c_fc.weight.data.detach().cpu().numpy()

                w2 = layer.c_proj.weight.data.detach().cpu().numpy()

                patterns = self._extract_weight_patterns(w1, w2)

                facts: List[Dict[str, Any]] = []

                for pattern in patterns:

                    fact_type = self._determine_fact_type(pattern)

                    significance = float(pattern.get("significance", 0.0))

                    facts.append(

                        {

                            "type": fact_type,

                            "layer": layer_name,

                            "content": pattern.get("description", ""),

                            "pattern": pattern,

                            "score": significance,

                            "source_layer": layer.__class__.__name__,

                            "reliability": 0.0,

                        }

                    )

                logger.debug(f"Из FFN {layer_name} извлечено {len(facts)} фактов (паттерны)")

                return facts

            except Exception as e:

                logger.error(f"Ошибка анализа весов FFN ({layer_name}): {e}")

                # Падение в фолбэк ниже



        # Fallback: простой обзор всех линейных подслоёв

        results: List[Dict[str, Any]] = []

        for name, sub in layer.named_modules():

            if isinstance(sub, torch.nn.Linear):

                try:

                    w = sub.weight.detach().cpu()

                    val = float(w.abs().mean().item())

                    results.append(

                        {

                            "type": "ffn_fact",

                            "layer": f"{layer_name}.{name}" if name else layer_name,

                            "details": {"mean_abs_w": val, "out_features": int(sub.out_features)},

                            "score": val,

                            "reliability": 0.0,

                        }

                    )

                except Exception:

                    continue

        return results



    def _extract_weight_patterns(self, w1: np.ndarray, w2: np.ndarray) -> List[Dict]:

        """Извлекает значимые паттерны из W1/W2: нормализация -> легковесный k-means -> фильтрация."""

        patterns: List[Dict[str, Any]] = []

        w1_norm = self._normalize_weights(w1)

        # Согласуем размерности: работаем по строкам w1 и столбцам w2 (т.е. строкам w2.T)

        w2_norm = self._normalize_weights(w2.T)

        clusters = self._find_weight_clusters(w1_norm, w2_norm)

        for cid, cluster_idx in enumerate(clusters):

            if self._is_significant_cluster(cluster_idx):

                cluster_info = {

                    "id": f"pattern_{cid}",

                    "cluster": cluster_idx,

                }

                desc = self._describe_pattern(cluster_info)

                signif = self._calculate_pattern_significance(cluster_info)

                patterns.append(

                    {

                        "id": f"pattern_{cid}",

                        "cluster": cluster_idx,

                        "description": desc,

                        "significance": signif,

                    }

                )

        return patterns



    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:

        """L2-нормализация по строкам с защитой от деления на ноль."""

        weights = weights.astype(np.float32, copy=False)

        norms = np.linalg.norm(weights, axis=1, keepdims=True)

        norms = np.where(norms == 0.0, 1e-8, norms)

        return weights / norms



    def _find_weight_clusters(self, w1: np.ndarray, w2: np.ndarray) -> List[np.ndarray]:

        """Простой k-means без внешних зависимостей. Кластеризуем по feature=[w1|w2]."""

        combined = np.concatenate([w1, w2], axis=1)

        n = combined.shape[0]

        if n < 4:

            return [np.arange(n)]

        n_clusters = int(max(2, min(10, n // 10)))

        rng = np.random.default_rng(42)

        # Инициализация центроидов случайной подвыборкой

        init_idx = rng.choice(n, size=n_clusters, replace=False)

        centroids = combined[init_idx].copy()



        def assign(xx: np.ndarray, cc: np.ndarray) -> np.ndarray:

            # расстояния до центроидов

            d2 = ((xx[:, None, :] - cc[None, :, :]) ** 2).sum(axis=2)

            return d2.argmin(axis=1)



        max_iter = 20

        labels = np.zeros(n, dtype=np.int32)

        for _ in range(max_iter):

            new_labels = assign(combined, centroids)

            if np.array_equal(new_labels, labels):

                break

            labels = new_labels

            for k in range(n_clusters):

                mask = labels == k

                if np.any(mask):

                    centroids[k] = combined[mask].mean(axis=0)



        clusters: List[np.ndarray] = []

        for k in range(n_clusters):

            clusters.append(np.where(labels == k)[0])

        return clusters



    def _is_significant_cluster(self, cluster: np.ndarray) -> bool:

        """Кластер значим, если в нём больше 5 элементов."""

        return int(cluster.size) > 5



    def _describe_pattern(self, cluster: Dict) -> str:

        """Формирует краткое описание паттерна."""

        size = int(np.array(cluster.get("cluster", [])).size)

        return f"Значимый паттерн в весах с {size} элементами"



    def _calculate_pattern_significance(self, cluster: Dict) -> float:

        """Простая эвристика значимости: масштаб от размера кластера."""

        size = int(np.array(cluster.get("cluster", [])).size)

        return float(min(1.0, size / 100.0))



    def _determine_fact_type(self, pattern: Dict) -> str:

        """Определяет тип факта на основе свойств паттерна (эвристика)."""

        signif = float(pattern.get("significance", 0.0))

        if signif > 0.7:

            return "ffn_pattern_strong"

        if signif > 0.4:

            return "ffn_pattern_moderate"

        return "ffn_pattern_weak"



    def _analyze_embeddings(self, layer: torch.nn.Module, layer_name: str) -> List[Dict]:

        """Эмбеддинги: выбираем топ-N векторов по норме как кандидаты концептов (без тяжёлой кластеризации)."""

        results: List[Dict[str, Any]] = []

        weight = getattr(layer, "weight", None)

        if not isinstance(weight, torch.nn.Parameter):

            return results

        try:

            emb = weight.detach().cpu().float()

            if emb.ndim != 2:

                return results

            norms = torch.linalg.norm(emb, dim=1)

            topn = int(min(10, norms.numel()))

            if topn <= 0:

                return results

            top_vals, top_idx = torch.topk(norms, topn)

            for i in range(int(topn)):

                try:

                    idx = int(top_idx[i].item())

                    val = float(top_vals[i].item())

                    results.append(

                        {

                            "type": "concept",

                            "layer": layer_name,

                            "content": f"embedding_{idx}",

                            "score": val,

                            "reliability": 0.0,

                        }

                    )

                except Exception:

                    continue

        except Exception:

            pass

        return results



    # ----------------------- Перенос знаний -> графовый манифест -----------------------

    def knowledge_to_graph_records(

        self,

        knowledge: List[Dict[str, Any]],

        shard_index: Dict[str, Dict[str, Any]],

        max_refs_per_record: int = 4,

    ) -> List[Dict[str, Any]]:

        """

        Преобразует извлечённые знания в записи графа с привязкой к шардам.



        Находим контейнеры по совпадению `metadata.layer_name` с полем `layer` записи знания,

        формируем ссылки fractal_ref: [{shard_file, key, container_id, block_start, block_end, tensor_path}].

        """

        # Инвертированный индекс: layer_name -> [entry]

        by_layer: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for cid, entry in shard_index.items():

            try:

                md = entry.get("metadata", {}) or {}

                ln = md.get("layer_name")

                if isinstance(ln, str) and ln:

                    by_layer[ln].append({"container_id": cid, **entry})

            except Exception:

                continue



        records: List[Dict[str, Any]] = []

        for k in knowledge:

            try:

                layer_name = str(k.get("layer", k.get("source_layer", "")))

                refs: List[Dict[str, Any]] = []

                candidates = by_layer.get(layer_name, [])

                for ent in candidates[:max_refs_per_record]:

                    md = ent.get("metadata", {}) or {}

                    refs.append(

                        {

                            "shard_file": ent.get("shard_file"),

                            "key": ent.get("key"),

                            "container_id": ent.get("container_id"),

                            "block_start": md.get("block_start"),

                            "block_end": md.get("block_end"),

                            "tensor_path": md.get("tensor_path"),

                            "param_name": md.get("param_name"),

                            "level": ent.get("level"),

                        }

                    )



                rec = {

                    "record_type": "knowledge",  # общий тип записи

                    "type": k.get("type", "unknown"),

                    "score": float(k.get("score", 0.0)),

                    "reliability": float(k.get("reliability", 0.0)),

                    "attrs": {k_: v for k_, v in k.items() if k_ not in ("type", "score", "reliability")},

                    "fractal_ref": refs,

                }

                records.append(rec)

            except Exception:

                continue

        return records



    def transfer_model_to_graph(

        self,

        model: torch.nn.Module,

        output_dir: str,

        graph_manifest_dir: Optional[str] = None,

    ) -> Dict[str, Any]:

        """

        Полный цикл: извлечение знаний -> привязка к шардам -> атомарная запись манифеста.



        Предполагается, что фрактальная структура уже сохранена шардированно в `output_dir`.

        """

        try:

            base = Path(output_dir)

            if graph_manifest_dir is None:

                graph_manifest_dir = str(base)



            # 1) Извлечь знания из модели

            knowledge = self.extract_knowledge_from_model(model)



            # 2) Индекс шардов (из shards_manifest.jsonl)

            shard_index = self.build_shards_index(str(base))



            # 3) Записи графа с привязками к шардам

            records = self.knowledge_to_graph_records(knowledge, shard_index)



            # 4) Метаданные

            manifest_meta = {

                "format": {

                    "type": "cogniflex_knowledge_graph",

                    "version": 1,

                },

                "links": {

                    "shards_manifest": "shards_manifest.jsonl",

                },

                "model_id": self.model_id,

                "incremental": False,

            }



            # 5) Атомарная запись

            return self.save_graph_manifest_atomic(

                manifest_dir=str(graph_manifest_dir),

                records=records,

                meta=manifest_meta,

            )

        except Exception as e:

            return {"ok": False, "error": str(e), "path": str(graph_manifest_dir or output_dir), "count": 0}



    def _evaluate_knowledge_reliability(self, knowledge: List[Dict]) -> None:

        """Простая оценка надёжности по типу и нормализованному score."""

        if not knowledge:

            return

        # Нормируем score в [0,1] по всему набору

        scores = [float(k.get("score", 0.0)) for k in knowledge]

        smin, smax = min(scores), max(scores)

        rng = (smax - smin) if smax > smin else 1.0

        for k in knowledge:

            base = (float(k.get("score", 0.0)) - smin) / rng

            t = k.get("type", "")

            # Весовые коэффициенты по типам (эвристика)

            if t == "attention_relation":

                coef = 0.7

            elif t == "ffn_fact":

                coef = 0.6

            elif t == "embedding_concept":

                coef = 0.5

            else:

                coef = 0.4

            k["reliability"] = float(max(0.0, min(1.0, base * coef)))



    # ----------------------- Сохранение/загрузка -----------------------

    def save_to_disk(self, output_path: str, knowledge_graph: Optional[Dict[str, Any]] = None) -> bool:

        """

        Сохраняет фрактальную структуру на диск в простой файловой схеме:

        - <output_dir>/index.json: метаданные и статистика

        - <output_dir>/containers.jsonl: список контейнеров с метаданными и путём к данным

        - <output_dir>/data/<container_id>.npy: данные контейнера

        - <output_dir>/knowledge_graph.json: (опционально) граф знаний

        """

        try:

            out_dir = Path(output_path)

            out_dir.mkdir(parents=True, exist_ok=True)

            data_dir = out_dir / "data"

            data_dir.mkdir(parents=True, exist_ok=True)



            # Пишем контейнеры и собираем JSONL

            containers_jsonl = []

            for cid, cont in self.containers.items():

                # Безопасное короткое имя файла: SHA1(cid)

                # Это устраняет проблемы Windows с MAX_PATH и недопустимыми символами

                sha1 = hashlib.sha1(cid.encode("utf-8")).hexdigest()

                file_name = f"{sha1}.npy"

                file_path = data_dir / file_name

                # Сохраняем данные

                np.save(file_path, cont.data)

                # Запись метаданных

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



            # Пишем индекс/статистику

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



        # - <output_dir>/shards/level_<L>_<S>.npz — данные контейнеров (key = sha1(id))

        # - <output_dir>/shards_manifest.jsonl    — построчно метаданные каждого контейнера + {shard_file, key}

        # - <output_dir>/knowledge_graph.json     — опционально граф знаний



        # Заметки:

        # - По умолчанию группируем по уровням (by_level=True) и пишем примерно shard_size контейнеров на шард.

        # - Используем ключи внутри .npz как SHA1 от container_id для надёжности на Windows.

        # - Пишем прогресс и периодически чистим память (gc.collect, CUDA cache).

        

    def save_to_disk_sharded(

        self,

        output_path: str,

        knowledge_graph: Optional[Dict[str, Any]] = None,

        shard_size: int = 5000,

        by_level: bool = True,

        compress: bool = True,

    ) -> bool:

        """

        Шардированное сохранение фрактальной структуры на диск:

        - <output_dir>/shards/level_<L>_shard_<S>.npz — данные контейнеров (key = sha1(id))

        - <output_dir>/shards_manifest.jsonl         — построчно метаданные каждого контейнера + {shard_file, key}

        - <output_dir>/knowledge_graph.json          — опционально граф знаний

        """

        try:

            out_dir = Path(output_path)

            out_dir.mkdir(parents=True, exist_ok=True)

            shards_dir = out_dir / "shards"

            shards_dir.mkdir(parents=True, exist_ok=True)



            # Манифест контейнеров -> куда записан и под каким ключом

            manifest_path = out_dir / "shards_manifest.jsonl"

            if manifest_path.exists():

                try:

                    manifest_path.unlink()

                except Exception:

                    pass



            total = sum(len(v) for v in self.fractal_tree.values())

            written = 0



            def write_shard(npz_path: Path, items: List[Tuple[str, FractalContainer]]) -> None:

                # Собираем словарь ключ -> массив

                arrays: Dict[str, np.ndarray] = {}

                for cid, cont in items:

                    key = hashlib.sha1(cid.encode("utf-8")).hexdigest()

                    arrays[key] = cont.data

                # Запись .npz

                if compress:

                    np.savez_compressed(npz_path, **arrays)

                else:

                    np.savez(npz_path, **arrays)

                # Запись строк в манифест

                with manifest_path.open("a", encoding="utf-8") as mf:

                    for cid, cont in items:

                        key = hashlib.sha1(cid.encode("utf-8")).hexdigest()

                        rec = {

                            "id": cid,

                            "level": int(cont.level),

                            "position": list(cont.position),

                            "shape": list(cont.shape),

                            "dtype": str(cont.dtype),

                            "metadata": cont.metadata,

                            "parent": cont.parent,

                            "children": cont.children,

                            "shard_file": str(npz_path.relative_to(out_dir)),

                            "key": key,

                        }

                        

                        mf.write(json.dumps(rec, ensure_ascii=False) + "\n")



            start_ts = time.time()

            logger.info(

                f"Шардированное сохранение: всего контейнеров {total}, shard_size={shard_size}, by_level={by_level}"

            )



            if by_level:

                for level in sorted(self.fractal_tree.keys()):

                    ids = self.fractal_tree.get(level, [])

                    if not ids:

                        continue

                    logger.info(f"Уровень {level}: контейнеров {len(ids)}")

                    chunk: List[Tuple[str, FractalContainer]] = []

                    shard_idx = 0

                    progress_interval = max(1, len(ids) // 20)

                    for idx, cid in enumerate(ids):

                        cont = self.containers.get(cid)

                        if cont is None:

                            continue

                        chunk.append((cid, cont))

                        if len(chunk) >= shard_size:

                            shard_path = shards_dir / f"level_{level}_shard_{shard_idx}.npz"

                            write_shard(shard_path, chunk)

                            written += len(chunk)

                            shard_idx += 1

                            chunk = []

                            # Мониторим и чистим память

                            if (shard_idx % 2) == 0:

                                gc.collect()

                                try:

                                    if getattr(self, "device", "cpu") == "cuda" and torch.cuda.is_available():

                                        torch.cuda.empty_cache()

                                except Exception:

                                    pass

                        if idx % progress_interval == 0:

                            done_pct = (idx + 1) / max(1, len(ids)) * 100

                            logger.debug(f"Уровень {level}: {done_pct:.1f}%")

                    if chunk:

                        shard_path = shards_dir / f"level_{level}_shard_{shard_idx}.npz"

                        write_shard(shard_path, chunk)

                        written += len(chunk)

            else:

                # Плоское шардирование по всем контейнерам

                all_ids: List[str] = []

                for lvl in sorted(self.fractal_tree.keys()):

                    all_ids.extend(self.fractal_tree.get(lvl, []))

                logger.info(f"Всего контейнеров для записи: {len(all_ids)}")

                chunk = []

                shard_idx = 0

                progress_interval = max(1, len(all_ids) // 20)

                for idx, cid in enumerate(all_ids):

                    cont = self.containers.get(cid)

                    if cont is None:

                        continue

                    chunk.append((cid, cont))

                    if len(chunk) >= shard_size:

                        shard_path = shards_dir / f"shard_{shard_idx}.npz"

                        write_shard(shard_path, chunk)

                        written += len(chunk)

                        shard_idx += 1

                        chunk = []

                        if (shard_idx % 2) == 0:

                            gc.collect()

                            try:

                                if getattr(self, "device", "cpu") == "cuda" and torch.cuda.is_available():

                                    torch.cuda.empty_cache()

                            except Exception:

                                pass

                    if idx % progress_interval == 0:

                        done_pct = (idx + 1) / max(1, len(all_ids)) * 100

                        logger.debug(f"Общий прогресс: {done_pct:.1f}%")

                if chunk:

                    shard_path = shards_dir / f"shard_{shard_idx}.npz"

                    write_shard(shard_path, chunk)

                    written += len(chunk)



            # Индекс/статистика

            index = {

                "model_id": self.model_id,

                "created_ts": time.time(),

                "format": "sharded_npz",

                "params": {

                    "block_size": self.block_size,

                    "fractal_levels": self.fractal_levels,

                    "containers_per_group": self.containers_per_group,

                    "hot_window_size": self.hot_window_size,

                    "shard_size": int(shard_size),

                    "by_level": bool(by_level),

                    "compressed": bool(compress),

                },

                "stats": self.get_statistics(),

            }

            with (out_dir / "index.json").open("w", encoding="utf-8") as f:

                json.dump(index, f, ensure_ascii=False, indent=2)



            # Пишем граф знаний, если есть

            if knowledge_graph is not None:

                with (out_dir / "knowledge_graph.json").open("w", encoding="utf-8") as f:

                    json.dump(knowledge_graph, f, ensure_ascii=False, indent=2)



            took = time.time() - start_ts

            logger.info(

                f"Шардированное сохранение завершено: записано {written}/{total} контейнеров за {took:.2f} сек"

            )

            return True

        except Exception:

            logger.exception("Ошибка шардированного сохранения фрактальной структуры")

            return False



    def save_to_disk_incremental(

        self,

        output_path: str,

        knowledge_graph: Optional[Dict[str, Any]] = None,

        batch_size: int = 5000,

        resume: bool = False,

        by_level: bool = True,

        compress: bool = True,

        state_filename: str = "incremental_state.json",

        graph_manifest_records: Optional[Iterable[Dict[str, Any]]] = None,

        graph_manifest_dir: Optional[str] = None,

        max_items_per_session: Optional[int] = None,

        show_progress: bool = False,

        min_batch_size: int = 1025,

        offload_after_write: bool = True,

    ) -> bool:

        """

        Инкрементальное шардированное сохранение с возможностью возобновления.

        Пишет те же структуры, что и save_to_disk_sharded, но батчами и с состоянием.

        """

        try:

            out_dir = Path(output_path)

            shards_dir = out_dir / "shards"

            out_dir.mkdir(parents=True, exist_ok=True)

            shards_dir.mkdir(parents=True, exist_ok=True)



            manifest_path = out_dir / "shards_manifest.jsonl"

            state_path = out_dir / state_filename



            # Состояние

            state: Dict[str, Any] = {

                "version": 1,

                "by_level": bool(by_level),

                "compress": bool(compress),

                "batch_size": int(batch_size),

                "current_level": None,

                "level_index": 0,  # позиция внутри списка ID выбранного уровня

                "shard_idx": 0,

                "total_written": 0,

                "initialized": False,

            }

            if resume and state_path.exists():

                try:

                    with state_path.open("r", encoding="utf-8") as f:

                        loaded = json.load(f)

                    state.update(loaded)

                    logger.info(

                        f"Возобновление инкрементального сохранения: уровень={state.get('current_level')} "

                        f"index={state.get('level_index')} shard_idx={state.get('shard_idx')} "

                        f"total_written={state.get('total_written')}"

                    )

                except Exception:

                    logger.warning("Не удалось прочитать состояние, начинаем заново", exc_info=True)



            def write_state() -> None:

                try:

                    with state_path.open("w", encoding="utf-8") as f:

                        json.dump(state, f, ensure_ascii=False, indent=2)

                except Exception:

                    logger.debug("Не удалось сохранить состояние инкрементального сохранения", exc_info=True)



            # Инициализация индекса и графа (однократно)

            if not state.get("initialized"):

                index = {

                    "model_id": self.model_id,

                    "created_ts": time.time(),

                    "format": "sharded_npz",

                    "params": {

                        "block_size": self.block_size,

                        "fractal_levels": self.fractal_levels,

                        "containers_per_group": self.containers_per_group,

                        "hot_window_size": self.hot_window_size,

                        "shard_size": int(batch_size),

                        "by_level": bool(by_level),

                        "compressed": bool(compress),

                        "incremental": True,

                    },

                    "stats": self.get_statistics(),

                }

                with (out_dir / "index.json").open("w", encoding="utf-8") as f:

                    json.dump(index, f, ensure_ascii=False, indent=2)

                if knowledge_graph is not None and not (out_dir / "knowledge_graph.json").exists():

                    with (out_dir / "knowledge_graph.json").open("w", encoding="utf-8") as f:

                        json.dump(knowledge_graph, f, ensure_ascii=False, indent=2)

                # Если это не возобновление, удалим старый манифест

                if not resume and manifest_path.exists():

                    try:

                        manifest_path.unlink()

                    except Exception:

                        pass

                state["initialized"] = True

                write_state()



            def write_shard(npz_path: Path, items: List[Tuple[str, FractalContainer]]) -> None:

                # Собираем словарь ключ -> массив

                arrays: Dict[str, np.ndarray] = {}

                for cid, cont in items:

                    key = hashlib.sha1(cid.encode("utf-8")).hexdigest()

                    arrays[key] = cont.data

                # Атомарная запись .npz через временный файл и замену

                tmp_path = npz_path.with_suffix(npz_path.suffix + ".tmp")

                try:

                    with open(tmp_path, "wb") as f:

                        if compress:

                            np.savez_compressed(f, **arrays)

                        else:

                            np.savez(f, **arrays)

                        try:

                            f.flush()

                            os.fsync(f.fileno())

                        except Exception:

                            pass

                    os.replace(tmp_path, npz_path)

                except KeyboardInterrupt:

                    try:

                        if tmp_path.exists():

                            tmp_path.unlink(missing_ok=True)

                    except Exception:

                        pass

                    raise

                except Exception:

                    try:

                        if tmp_path.exists():

                            tmp_path.unlink(missing_ok=True)

                    except Exception:

                        pass

                    raise

                # Запись строк в манифест с явным flush/fsync

                try:

                    with manifest_path.open("a", encoding="utf-8") as mf:

                        for cid, cont in items:

                            key = hashlib.sha1(cid.encode("utf-8")).hexdigest()

                            rec = {

                                "id": cid,

                                "level": int(cont.level),

                                "position": list(cont.position),

                                "shape": list(cont.shape),

                                "dtype": str(cont.dtype),

                                "metadata": cont.metadata,

                                "parent": cont.parent,

                                "children": cont.children,

                                "shard_file": str(npz_path.relative_to(out_dir)),

                                "key": key,

                            }

                            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

                        try:

                            mf.flush()

                            os.fsync(mf.fileno())

                        except Exception:

                            pass

                except Exception:

                    # Манифест не критичен для целостности данных шарда; логика восстановления выше по стеку

                    logger.debug("Ошибка при записи в манифест шардов", exc_info=True)



            total = sum(len(v) for v in self.fractal_tree.values())

            logger.info(

                f"Инкрементальное сохранение: всего контейнеров {total}, batch_size={batch_size}, by_level={by_level}"

            )



            # Поддерживаем только by_level=True для простоты и предсказуемости возобновления

            if not by_level:

                logger.warning("by_level=False для инкрементального режима пока не поддержан. Включаем by_level=True")

                by_level = True



            # Гарантия минимального размера батча

            try:

                min_batch_size = int(max(1025, min_batch_size))

            except Exception:

                min_batch_size = 1025

            if batch_size < min_batch_size:

                batch_size = min_batch_size



            # Внутрисессионный счетчик: сколько записали за текущий вызов функции

            session_written = 0



            def render_progress(done: int, total_: int) -> None:

                if not show_progress:

                    return

                try:

                    pct = (done / max(1, total_))

                    bar_len = 24

                    filled = int(bar_len * pct)

                    bar = "#" * filled + "-" * (bar_len - filled)

                    sys.stdout.write(f"\r[ {bar} ] {pct*100:5.1f}%  ({done}/{total_})")

                    sys.stdout.flush()

                except Exception:

                    pass



            # первичный вывод прогресса

            render_progress(int(state.get("total_written", 0)), total)



            for level in sorted(self.fractal_tree.keys()):

                ids = self.fractal_tree.get(level, [])

                if not ids:

                    continue

                # Возобновление: пропуск уровней ниже текущего

                if state.get("current_level") is not None and int(level) < int(state["current_level"]):

                    continue

                # Установка уровня и стартового индекса

                if state.get("current_level") is None or int(level) > int(state.get("current_level", -1)):

                    state["current_level"] = int(level)

                    state["level_index"] = 0

                    state["shard_idx"] = 0

                    write_state()



                start_idx = int(state["level_index"])

                shard_idx = int(state["shard_idx"])

                while start_idx < len(ids):

                    end_idx = min(start_idx + int(batch_size), len(ids))

                    batch_ids = ids[start_idx:end_idx]

                    items: List[Tuple[str, FractalContainer]] = []

                    for cid in batch_ids:

                        cont = self.containers.get(cid)

                        if cont is None:

                            continue

                        items.append((cid, cont))

                    if not items:

                        start_idx = end_idx

                        state["level_index"] = start_idx

                        write_state()

                        continue

                    # Ограничение внутри сессии: если превышаем лимит, ужимаем items

                    if max_items_per_session is not None:

                        remaining = int(max(0, int(max_items_per_session) - session_written))

                        if remaining <= 0:

                            # Выходим из сессии, оставляя состояние для резюма

                            render_progress(int(state.get("total_written", 0)) + session_written, total)

                            logger.info(f"Сессионный лимит {max_items_per_session} достигнут. Прерываем до следующего запуска.")

                            write_state()

                            # Не удаляем state-файл, чтобы можно было продолжить

                            return True

                        if len(items) > remaining:

                            items = items[:remaining]

                            # скорректируем end_idx, чтобы прогресс и индексы были согласованы

                            end_idx = start_idx + len(items)

                    shard_path = shards_dir / f"level_{level}_shard_{shard_idx}.npz"

                    write_shard(shard_path, items)

                    # После записи шарда освобождаем память текущего батча, если требуется

                    if offload_after_write:

                        try:

                            for cid, cont in items:

                                # Сохраняем offload-метаданные для возможной последующей загрузки

                                key = hashlib.sha1(cid.encode("utf-8")).hexdigest()

                                cont.metadata["offload_shard"] = str(shard_path)

                                cont.metadata["offload_key"] = key

                                cont.metadata.setdefault("offload_format", "npz")

                                # Очищаем большие буферы данных, оставляя метаданные размеров/типа

                                try:

                                    dtype = cont.data.dtype

                                except Exception:

                                    dtype = np.float32

                                cont.data = np.empty((0,), dtype=dtype)

                        except Exception:

                            logger.debug("Не удалось выполнить offload_after_write для батча", exc_info=True)

                    state["total_written"] = int(state.get("total_written", 0)) + len(items)

                    session_written += len(items)

                    shard_idx += 1

                    start_idx = end_idx

                    state["level_index"] = start_idx

                    state["shard_idx"] = shard_idx

                    write_state()

                    # Очистка памяти

                    try:

                        # Сбрасываем ссылки на items ранним del, чтобы GC освободил память

                        del items

                    except Exception:

                        pass

                    gc.collect()

                    try:

                        if torch.cuda.is_available():

                            torch.cuda.empty_cache()

                    except Exception:

                        pass

                    # Небольшая пауза, чтобы ОС смогла вернуть страницы памяти

                    try:

                        time.sleep(0.01)

                    except Exception:

                        pass

                    # визуализация прогресса

                    render_progress(int(state.get("total_written", 0)), total)

                    # Адаптивная корректировка batch_size при высоком потреблении RAM

                    if psutil is not None:

                        try:

                            vm = psutil.virtual_memory()

                            if vm.percent > 85 and batch_size > min_batch_size:

                                # Уменьшаем, но не опускаемся ниже min_batch_size (строго >1024)

                                batch_size = max(min_batch_size, batch_size // 2)

                                logger.warning(f"Память >85%, уменьшаем batch_size до {batch_size}")

                        except Exception:

                            pass



                # уровень завершён

                state["current_level"] = int(level)

                state["level_index"] = len(ids)

                state["shard_idx"] = shard_idx

                write_state()



            # Завершение: удалим state-файл

            try:

                if state_path.exists():

                    state_path.unlink()

            except Exception:

                pass

            logger.info(

                f"Инкрементальное сохранение завершено: записано {state.get('total_written', 0)}/{total} контейнеров"

            )

            # Завершающий вывод прогресса

            render_progress(int(state.get("total_written", 0)), total)

            if show_progress:

                try:

                    sys.stdout.write("\n")

                except Exception:

                    pass

            # Сохраняем графовый манифест (если передан) после успешного завершения

            try:

                if graph_manifest_records is not None and graph_manifest_dir is not None:

                    manifest_meta = {

                        "fractal_format": "sharded_npz",

                        "links": {

                            "shards_manifest": "shards_manifest.jsonl",

                        },

                        "model_id": self.model_id,

                        "incremental": True,

                    }

                    self.save_graph_manifest_atomic(

                        graph_manifest_dir,

                        graph_manifest_records,

                        meta=manifest_meta,

                    )

            except Exception:

                logger.warning("Не удалось сохранить графовый манифест после инкрементального сохранения", exc_info=True)

            return True

        except torch.cuda.OutOfMemoryError:

            logger.warning("CUDA OOM при инкрементальном сохранении — уменьшите batch_size или используйте CPU")

            return False

        except Exception:

            logger.exception("Ошибка инкрементального сохранения")

            return False



    def auto_adjust_batch_size(self) -> int:

        """Оценка подходящего размера батча по доступной памяти CPU/GPU."""

        base_size = 5000

        try:

            # Избегаем вызовов CUDA при работе на CPU

            if getattr(self, "device", "cpu") == "cuda" and torch.cuda.is_available():

                props = torch.cuda.get_device_properties(0)

                total = getattr(props, "total_memory", 2 * 1024**3)

                free_like = max(256 * 1024**2, total - torch.cuda.memory_reserved())

                factor = float(free_like) / float(2 * 1024**3)

                base_size = int(base_size * max(0.25, min(2.0, factor)))

        except Exception:

            pass

        if psutil is not None:

            try:

                vm = psutil.virtual_memory()

                if vm.percent > 70:

                    base_size = int(base_size * 0.7)

            except Exception:

                pass

        return max(500, min(20000, base_size))



    def save_to_disk_with_recovery(self, output_path: str, max_attempts: int = 3) -> bool:

        """Инкрементальное сохранение с автоматическим восстановлением."""

        batch_size = self.auto_adjust_batch_size()

        for attempt in range(max_attempts):

            logger.info(f"Попытка сохранения {attempt + 1}/{max_attempts} с batch_size={batch_size}...")

            # Если есть state — резюмируем

            state_path = Path(output_path) / "incremental_state.json"

            resume = state_path.exists()

            if self.save_to_disk_incremental(output_path, batch_size=batch_size, resume=resume):

                logger.info("Сохранение успешно завершено")

                return True

            if attempt < max_attempts - 1:

                batch_size = max(500, batch_size // 2)

                logger.warning(f"Сохранение не завершено. Уменьшаем batch_size до {batch_size} и повторяем...")

                time.sleep(2)

        logger.error("Не удалось сохранить фрактальную структуру после нескольких попыток")

        return False



    def load_from_disk(self, input_path: str, lazy: bool = False, progress_every: int = 500000) -> bool:

        """

        Загружает фрактальную структуру с диска из каталога, созданного save_to_disk:

        - ожидает файлы: index.json, containers.jsonl, data/*.npy

        """

        try:

            in_dir = Path(input_path)

            index_path = in_dir / "index.json"

            containers_path = in_dir / "containers.jsonl"

            data_dir = in_dir / "data"

            shards_manifest = in_dir / "shards_manifest.jsonl"



            # Авто-детект atomic-формата: index.json + data/, но без containers.jsonl

            if index_path.exists() and data_dir.exists() and (not containers_path.exists()):

                report = self._load_from_disk_atomic_format(str(in_dir))

                if not report.get("ok", False):

                    logger.error(f"Не удалось загрузить (atomic): {report.get('error')}")

                return bool(report.get("ok", False))



            # Новый шардированный формат: наличие shards_manifest.jsonl

            if shards_manifest.exists():

                # Очищаем текущее состояние

                self.clear()

                # Подготовка ленивого индекса при необходимости

                if not hasattr(self, "lazy_index"):

                    self.lazy_index = {}

                # Загружаем index для метаданных, если есть

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

                                # Только индекс без загрузки массива

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

                                # Полная загрузка массива

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

                    (

                        f"Загружено (sharded){' (lazy index)' if lazy else ''} контейнеров: {loaded}. "

                        f"Уровни: {sorted(self.fractal_tree.keys())}. "

                        + (f"Общий размер: {self.total_memory / (1024*1024):.2f} MB" if not lazy else "")

                    )

                )

                return True



            # --- конец шардированной загрузки ---



        except Exception as e:

            logger.error(f"Ошибка загрузки фрактальной структуры: {e}", exc_info=True)

            return False



        # Ни atomic, ни sharded не сработали — пытаемся старый формат

        return self._load_legacy_format(index_path, containers_path, data_dir)



    # ----------------------- Ленивый доступ к контейнеру -----------------------

    def get_container_data(self, cid: str) -> np.ndarray:

        """Возвращает данные контейнера.



        - Если контейнер уже в self.containers — возвращает его data

        - Если активирован lazy-индекс (self.lazy_index) — подгружает массив из соответствующего npz и

          приводит форму к rec.shape при необходимости

        """

        if cid in self.containers:

            return self.containers[cid].data

        entry = getattr(self, "lazy_index", {}).get(cid)

        if not entry:

            raise KeyError(f"Контейнер {cid} не найден")

        shard_file = entry["shard_file"]

        key = entry["key"]

        shape = tuple(int(x) for x in (entry.get("shape") or []))

        with np.load(shard_file, allow_pickle=False) as zf:

            arr = zf[key]

        try:

            if shape and int(np.prod(shape)) == int(arr.size):

                arr = arr.reshape(shape)

        except Exception:

            pass

        return arr



    # ----------------------- Реконструкция state_dict -----------------------

    def reconstruct_state_dict(

        self,

        output_dtype: str = "float32",

        device: str = "cpu",

        limit_tensors: Optional[int] = None,

        include_params: Optional[List[str]] = None,

        resume_from: Optional[str] = None,

        processed_params: Optional["set[str]"] = None,

    ) -> Dict[str, torch.Tensor]:

        """

        Собирает исходные тензоры PyTorch по блокам из шардов, используя метаданные:

        - metadata.tensor_path: ключ параметра (например, 'transformer.wte.weight')

        - metadata.original_shape: исходная форма параметра

        - metadata.block_start/block_end: позиция блока в плоском представлении



        Args:

            output_dtype: целевой тип ('float32'|'float16'|'bfloat16'|'float64')

            device: 'cpu' или 'cuda'

            limit_tensors: если задано, собрать только N первых параметров (для быстрого теста)



        Returns:

            Dict[str, torch.Tensor]: state_dict для загрузки в модель

        """

        # Источник записей: если есть ленивый индекс — используем его приоритетно

        has_lazy = hasattr(self, "lazy_index") and len(self.lazy_index) > 0

        logger.info(f"[reconstruct] containers: {len(self.containers)}; lazy_index: {len(getattr(self, 'lazy_index', {}))}")

        use_lazy = has_lazy



        # Группируем записи по tensor_path

        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        if use_lazy:

            logger.info(f"[reconstruct] lazy_index size: {len(self.lazy_index)}")

            debug_samples = 0



            # 1) Предварительно собираем множество всех ключей (tpath), чтобы применить фильтрацию и лимит

            all_tpaths_set: "set[str]" = set()

            for _cid, _entry in self.lazy_index.items():

                # Веса state_dict восстанавливаем только из уровня 0 (сырые блоки).

                # Более высокие уровни — это агрегаты/структурные контейнеры и их нельзя смешивать в state_dict.

                try:

                    if int(_entry.get("level", -1)) != 0:

                        continue

                except Exception:

                    continue

                _meta = _entry.get("metadata", {}) or {}

                _tpath = _meta.get("tensor_path")

                if not _tpath:

                    _layer = _meta.get("layer_name") or _entry.get("layer_name")

                    _param = _meta.get("param_name") or "weight"

                    if _layer:

                        _tpath = f"{_layer}.{_param}"

                if _tpath:

                    all_tpaths_set.add(_tpath)



            # Применяем фильтры (include_params, processed_params, resume_from) и лимит по числу параметров

            selected_keys = sorted(all_tpaths_set)

            if include_params:

                flt = set()

                for k in selected_keys:

                    for pat in include_params:

                        if pat and pat in k:

                            flt.add(k)

                            break

                selected_keys = sorted(flt)

            if processed_params:

                selected_keys = [k for k in selected_keys if k not in processed_params]

            if resume_from:

                try:

                    idx = selected_keys.index(resume_from)

                    selected_keys = selected_keys[idx + 1 :]

                except ValueError:

                    pass

            if limit_tensors is not None:

                selected_keys = selected_keys[: int(limit_tensors)]



            selected_set = set(selected_keys)



            # 2) Второй проход: собираем записи только для выбранных ключей,

            #    ранний выход когда набрана полная покрывающая информация

            #    (определяется по original_shape и максимальному block_end)

            max_be: Dict[str, int] = {}

            target_total: Dict[str, int] = {}

            completed: "set[str]" = set()



            for cid, entry in self.lazy_index.items():

                # Только уровень 0 для корректной реконструкции state_dict

                try:

                    if int(entry.get("level", -1)) != 0:

                        continue

                except Exception:

                    continue

                meta = entry.get("metadata", {}) or {}

                tpath = meta.get("tensor_path")

                if not tpath:

                    layer_name = meta.get("layer_name") or entry.get("layer_name")

                    param_name = meta.get("param_name") or "weight"

                    if layer_name:

                        tpath = f"{layer_name}.{param_name}"

                # Если ключ не выбран — пропускаем сразу (ранняя фильтрация)

                if not tpath or (selected_set and tpath not in selected_set):

                    continue



                bs = meta.get("block_start")

                be = meta.get("block_end")

                if bs is None or be is None:

                    if debug_samples < 10:

                        logger.info(

                            f"[reconstruct][skip-no-blocks] cid={cid} bs={bs} be={be} meta_keys={list(meta.keys())[:10]}"

                        )

                        debug_samples += 1

                    continue



                rec = dict(entry)

                rec["block_start"] = int(bs)

                rec["block_end"] = int(be)

                rec["original_shape"] = tuple(int(x) for x in (meta.get("original_shape") or []))

                rec["tensor_path"] = tpath

                if debug_samples < 10:

                    logger.info(

                        f"[reconstruct][ok] cid={cid} tpath={tpath} bs={rec['block_start']} be={rec['block_end']} "

                        f"orig_shape={rec['original_shape']} shard_file={entry.get('shard_file')} key={entry.get('key')}"

                    )

                    debug_samples += 1

                groups[tpath].append(rec)



                # Обновляем агрегаты покрытия для раннего выхода

                mbe = max_be.get(tpath, 0)

                if rec["block_end"] > mbe:

                    max_be[tpath] = rec["block_end"]

                if tpath not in target_total:

                    try:

                        orig_shape = rec["original_shape"]

                        target_total[tpath] = int(np.prod(orig_shape)) if orig_shape else rec["block_end"]

                    except Exception:

                        target_total[tpath] = rec["block_end"]

                # Проверка завершения по tpath

                if max_be.get(tpath, 0) >= target_total.get(tpath, 1):

                    completed.add(tpath)



                # Если все выбранные ключи собраны — можно завершать сканирование lazy_index рано

                if selected_set and len(completed) >= len(selected_set):

                    break

        else:

            for cid, cont in self.containers.items():

                # Воссоздаём state_dict только по контейнерам уровня 0

                try:

                    if int(getattr(cont, "level", -1)) != 0:

                        continue

                except Exception:

                    continue

                meta = cont.metadata or {}

                tpath = meta.get("tensor_path")

                if not tpath:

                    layer_name = meta.get("layer_name") or getattr(cont, "layer_name", None)

                    param_name = meta.get("param_name") or "weight"

                    if layer_name:

                        tpath = f"{layer_name}.{param_name}"

                if not tpath:

                    continue

                bs = meta.get("block_start")

                be = meta.get("block_end")

                if bs is None or be is None:

                    continue

                rec = {

                    "block_start": int(bs),

                    "block_end": int(be),

                    "original_shape": tuple(int(x) for x in (meta.get("original_shape") or [])),

                    "tensor_path": tpath,

                    "shape": tuple(int(x) for x in (cont.shape or [])),

                    "dtype": cont.dtype,

                    "metadata": meta,

                    "container_id": cid,

                }

                groups[tpath].append(rec)



        # Функция для сортировки блоков по позиции

        def _sort_key(r: Dict[str, Any]) -> int:

            return int(r["block_start"])



        # Выбор выходного dtype torch

        torch_dtype = {

            "float16": torch.float16,

            "bfloat16": torch.bfloat16,

            "float32": torch.float32,

            "float64": torch.float64,

        }.get(str(output_dtype).lower(), torch.float32)



        # Подготовим список ключей параметров с учётом фильтрации/резюме

        all_keys = list(groups.keys())

        all_keys.sort()

        if include_params:

            # простая фильтрация по подстроке (можно расширить до regex)

            flt = set()

            for k in all_keys:

                for pat in include_params:

                    if pat and pat in k:

                        flt.add(k)

                        break

            keys = sorted(flt)

        else:

            keys = all_keys

        if processed_params:

            keys = [k for k in keys if k not in processed_params]

        if resume_from:

            # начать после указанного ключа

            try:

                idx = keys.index(resume_from)

                keys = keys[idx + 1 :]

            except ValueError:

                # если нет такого ключа — начинаем с начала

                pass



        # Диагностика: выведем первые несколько сгруппированных ключей и состав блоков

        if len(groups) > 0:

            sample_keys = list(groups.keys())[:5]

            logger.info(f"[reconstruct] grouped params: {len(groups)}; sample keys: {sample_keys}")

            for sk in sample_keys:

                recs = groups.get(sk, [])

                if recs:

                    first = recs[0]

                    logger.info(f"[reconstruct] key={sk} | records={len(recs)} | original_shape={first.get('original_shape')} | block_ranges=[{recs[0].get('block_start')}-{recs[-1].get('block_end')}]")

        else:

            logger.warning("[reconstruct] No groups found")



        state: Dict[str, torch.Tensor] = {}

        processed = 0

        for tpath in keys:

            recs = groups.get(tpath, [])

            if limit_tensors is not None and processed >= int(limit_tensors):

                break

            if not recs:

                continue

            recs.sort(key=_sort_key)



            # Определяем целевой размер

            original_shape = tuple(int(x) for x in (recs[0].get("original_shape") or []))

            if not original_shape:

                # Если нет формы — пытаемся инферить по последнему блоку

                total = int(recs[-1]["block_end"]) if recs else 0

                original_shape = (total,)



            total_elems = int(np.prod(original_shape))

            use_cuda = (str(device) == "cuda") and torch.cuda.is_available()

            if use_cuda:

                # Пишем результат напрямую в GPU-буфер

                tensor = torch.empty(original_shape, dtype=torch_dtype, device="cuda")

                # заполняем по блокам

                # Для универсальности работаем с плоским представлением на GPU

                gpu_flat = tensor.view(-1)

            else:

                flat = np.zeros(total_elems, dtype=np.float32)



            # Заполняем плоский буфер по блокам

            import time as _time 

            _t0 = _time.time()

            if use_lazy:

                # Группируем записи по shard_file, чтобы открывать .npz один раз

                by_shard: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

                for r in recs:

                    by_shard[r["shard_file"]].append(r)

                for shard_file, shard_recs in by_shard.items():

                    # Внутри шарда обрабатываем по порядку блоков

                    shard_recs.sort(key=_sort_key)

                    with np.load(shard_file, allow_pickle=False) as zf:

                        for r in shard_recs:

                            bs = int(r["block_start"])

                            be = int(r["block_end"])

                            arr = zf[r["key"]]

                            # Приводим к float32 (устойчиво)

                            if arr.dtype == np.float16:

                                arr = arr.astype(np.float32)

                            elif arr.dtype == np.int8:

                                scale = None

                                meta = r.get("metadata") or {}

                                if meta:

                                    scale = meta.get("quant_scale") or meta.get("quantization_scale")

                                if scale:

                                    arr = (arr.astype(np.float32)) * float(scale)

                                else:

                                    arr = arr.astype(np.float32)

                            elif arr.dtype != np.float32:

                                arr = arr.astype(np.float32)

                            # Вставка

                            if use_cuda:

                                chunk = torch.from_numpy(arr.reshape(-1)[: (be - bs)])

                                gpu_flat[bs:be].copy_(chunk.to("cuda"))

                            else:

                                flat[bs:be] = arr.reshape(-1)[: (be - bs)]

            else:

                for r in recs:

                    bs = int(r["block_start"])

                    be = int(r["block_end"])

                    # Берём данные из контейнера

                    cid = r.get("container_id")

                    arr = self.containers[cid].data  # type: ignore[index]

                    # Приводим к float32 (устойчиво)

                    if arr.dtype == np.float16:

                        arr = arr.astype(np.float32)

                    elif arr.dtype == np.int8:

                        scale = None

                        meta = r.get("metadata") or {}

                        if meta:

                            scale = meta.get("quant_scale") or meta.get("quantization_scale")

                        if scale:

                            arr = (arr.astype(np.float32)) * float(scale)

                        else:

                            arr = arr.astype(np.float32)

                    elif arr.dtype != np.float32:

                        arr = arr.astype(np.float32)

                    # Вставка

                    if use_cuda:

                        chunk = torch.from_numpy(arr.reshape(-1)[: (be - bs)])

                        gpu_flat[bs:be].copy_(chunk.to("cuda"))

                    else:

                        flat[bs:be] = arr.reshape(-1)[: (be - bs)]

            _dt = _time.time() - _t0

            if _dt > 2.0:

                logger.info(f"[reconstruct] assembled '{tpath}' in {_dt:.2f}s, shape={original_shape}, cuda={use_cuda}")



            if not use_cuda:

                tensor = torch.from_numpy(flat.reshape(original_shape)).to(dtype=torch_dtype, device=device)

            else:

                # Уже на GPU, при необходимости приведём dtype

                if tensor.dtype != torch_dtype:

                    tensor = tensor.to(dtype=torch_dtype)

            state[tpath] = tensor

            processed += 1



        logger.info(f"Собрано параметров: {len(state)} (limit={limit_tensors})")

        return state



# ----------------------- Утилита переупаковки -----------------------

def repack_model_to_fractal(

    model_path: str,

    output_path: str,

    fractal_levels: int = 4,

    block_size: int = 64,

    device: str = "cpu",

) -> bool:

    """

    Переупаковывает модель в фрактальную структуру с извлечением и сохранением графа знаний.

    """

    start_time = time.time()

    logger.info(f"Начата переупаковка модели из {model_path} в фрактальную структуру...")

    try:

        # 1) Загрузка модели

        logger.info("Загрузка модели...")

        # Если путь — директория, считаем что это локальная директория HF и используем Transformers

        model: Optional[torch.nn.Module]

        if os.path.isdir(model_path):

            model = _load_hf_model_dir(model_path, device=device)

        else:

            model = _safe_load_model(model_path, device=device)

        if model is None:

            logger.error("Не удалось загрузить модель")

            return False



        # 2) Извлечение знаний

        store = FractalWeightStore(block_size=block_size, fractal_levels=fractal_levels)

        logger.info("Извлечение знаний из модели...")

        knowledge = store.extract_knowledge_from_model(model)



        # 3) Построение графа знаний

        logger.info("Построение графа знаний...")

        knowledge_graph = _build_knowledge_graph(knowledge)



        # 4) Упаковка весов модели

        logger.info("Создание фрактальной структуры весов...")

        model_id = Path(model_path).stem

        if not store.pack_model_weights(model, model_id=model_id):

            logger.error("Не удалось упаковать веса модели")

            return False



        # 5) Сохранение на диск (шардированное)

        logger.info("Сохранение фрактальной структуры (шардировано)...")

        ok = store.save_to_disk_sharded(

            output_path,

            knowledge_graph=knowledge_graph,

            shard_size=10000,

            by_level=True,

            compress=True,

        )

        if not ok:

            logger.warning("Шардированное сохранение не удалось. Пытаемся инкрементально с восстановлением...")

            ok = store.save_to_disk_with_recovery(output_path)

            if not ok:

                return False



        stats = store.get_statistics()

        logger.info("Статистика фрактальной структуры:")

        logger.info(f"  Общее количество контейнеров: {stats['total_containers']}")

        logger.info(f"  Контейнеры по уровням: {stats['containers_by_level']}")

        logger.info(f"  Общий размер: {stats['total_memory_mb']:.2f} MB")

        logger.info(f"  Сжатие: {stats['compression_ratio']:.2f}x")



        logger.info(f"Переупаковка завершена за {time.time() - start_time:.2f} сек")

        return True

    except Exception:

        logger.exception("Критическая ошибка переупаковки модели")

        return False





def _safe_load_model(model_path: str, device: str = "cpu") -> Optional[torch.nn.Module]:

    """Простой загрузчик PyTorch-модели, сохранённой через torch.save(model)."""

    map_location = torch.device(device if device else "cpu")

    try:

        obj = torch.load(model_path, map_location=map_location)

        if isinstance(obj, torch.nn.Module):

            obj.eval()

            return obj

        # Если сохранён state_dict, пробуем найти простую оболочку — здесь возвращаем None

        logger.error("Ожидалась сохранённая torch.nn.Module, получен другой объект")

        return None

    except Exception:

        logger.exception(f"Ошибка загрузки модели из {model_path}")

        return None



def _load_hf_model_dir(model_dir: str, device: str = "cpu") -> Optional[torch.nn.Module]:

    """

    Загружает модель из директории HuggingFace (weights + config) через Transformers.

    Предпочитает AutoModelForCausalLM, иначе AutoModel. Работает только локально.

    """

    try:

        if AutoModelForCausalLM is None:

            raise ImportError("transformers не установлен: pip install transformers safetensors accelerate")

        # Разрешаем путь на случай, если указан корень репо HF-кэша (нужно найти snapshots/<rev>)

        base = Path(model_dir)

        real_dir = base

        if not (base / "config.json").exists():

            # Ищем в подкаталогах snapshots/*

            snaps = list((base / "snapshots").glob("*/config.json")) if (base / "snapshots").exists() else []

            if snaps:

                real_dir = snaps[0].parent

            else:

                # Ищем глубоко первый config.json

                found = list(base.rglob("config.json"))

                if found:

                    real_dir = found[0].parent

        model_path_str = str(real_dir)

        torch_dtype = torch.float16 if (device == "cuda" and torch.cuda.is_available()) else torch.float32

        # Сначала пробуем как CausalLM (rugpt3large), затем как базовую модель

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

    """Экспортирует HF-модель (например ruGPT-small) в шардированное фрактальное хранилище.



    Делает:

    - загружает модель локально (без сетевых запросов по умолчанию)

    - пакует полный state_dict в FractalWeightStore.pack_state_dict

    - сохраняет фрактальные шарды + index.json

    - сохраняет config.json (для AutoConfig.from_pretrained) и tokenizer



    Примечание: если модель не закэширована локально и local_files_only=True, экспорт завершится ошибкой.

    """

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



