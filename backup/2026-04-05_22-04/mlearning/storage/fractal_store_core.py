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
    """Контейнер для фрактальных данных"""
    
    def __init__(self, data: np.ndarray, dtype: str = 'float32', priority: float = 1.0):
        self.data = data
        self.dtype = dtype
        self.priority = priority
        self.timestamp = time.time()
        self.access_count = 0
        
    def get_memory_size(self) -> int:
        """Возвращает размер памяти в байтах"""
        try:
            itemsize = np.dtype(self.dtype).itemsize if isinstance(self.dtype, str) else self.data.dtype.itemsize
        except (TypeError, AttributeError) as e:
            logger.warning(f"Error getting itemsize for dtype {self.dtype}: {e}")
            itemsize = self.data.dtype.itemsize
        return self.data.nbytes * itemsize

class FractalWeightStore:
    """Хранилище весов модели с фрактальной структурой"""
    
    def __init__(self, device: str = 'cpu', max_memory_gb: float = 16.0):
        self.device = device
        self.max_memory_bytes = max_memory_gb * 1024**3
        self.containers: Dict[str, FractalContainer] = {}
        self.total_memory = 0
        
        try:
            config_device = getattr(self, 'config', {}).get('device', 'cpu') if hasattr(self, 'config') else device
            use_cuda = (config_device != "cpu") if isinstance(config_device, str) else False
            if use_cuda and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        except Exception as e:
            logger.warning(f"Ошибка инициализации CUDA: {e}")
            self.device = "cpu"
        
        self.index = {}
        self.data_dir = None
        self.lazy_index = {}
        
        logger.info(f"FractalWeightStore инициализирован на {self.device}")
    
    def _load_legacy_format(self, index_path: str, containers_path: str, data_dir: str) -> bool:
        """
        Загружает фрактальную структуру в устаревшем формате.
        
        Args:
            index_path: Путь к файлу индекса
            containers_path: Путь к файлу контейнеров
            data_dir: Директория с данными
            
        Returns:
            bool: Успешность загрузки
        """
        try:
            logger.info("Попытка загрузки в устаревшем формате...")
            
            # Загружаем индекс
            if os.path.exists(index_path):
                with open(index_path, 'r', encoding='utf-8') as f:
                    self.index = json.load(f)
                logger.info(f"Индекс загружен: {len(self.index)} записей")
            else:
                logger.error(f"Файл индекса не найден: {index_path}")
                return False
            
            # Загружаем контейнеры
            if os.path.exists(containers_path):
                with open(containers_path, 'r', encoding='utf-8') as f:
                    containers_data = json.load(f)
                
                # Восстанавливаем контейнеры
                self.containers = {}
                for cid, container_info in containers_data.items():
                    container = FractalContainer(
                        id=cid,
                        data=np.array([]),  # Будет загружено позже по запросу
                        metadata=container_info.get('metadata', {})
                    )
                    self.containers[cid] = container
                
                logger.info(f"Контейнеры загружены: {len(self.containers)}")
            else:
                logger.warning(f"Файл контейнеров не найден: {containers_path}")
                self.containers = {}
            
            # Проверяем директорию данных
            if os.path.exists(data_dir):
                self.data_dir = data_dir
                logger.info(f"Директория данных установлена: {data_dir}")
            else:
                logger.warning(f"Директория данных не найдена: {data_dir}")
                self.data_dir = None
            
            logger.info("Загрузка в устаревшем формате завершена успешно")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки в устаревшем формате: {e}", exc_info=True)
            return False
    
    def get_container_data(self, cid: str) -> np.ndarray:
        """Возвращает данные контейнера"""
        if cid in self.containers:
            return self.containers[cid].data
        entry = getattr(self, "lazy_index", {}).get(cid)
        if not entry:
            raise KeyError(f"Контейнер {cid} не найден")
        # Загрузка данных по запросу
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
        Собирает исходные тензоры PyTorch по блокам из шардов
        """
        if processed_params is None:
            processed_params = set()
        
        state_dict = {}
        tensor_count = 0
        
        try:
            # Собираем тензоры из контейнеров
            for param_name, container_info in self.index.items():
                if include_params and param_name not in include_params:
                    continue
                
                if param_name in processed_params:
                    continue
                
                if limit_tensors and tensor_count >= limit_tensors:
                    break
                
                # Получаем данные контейнера
                try:
                    container_data = self.get_container_data(container_info["container_id"])
                    
                    # Конвертируем в тензор
                    tensor = torch.from_numpy(container_data.astype(output_dtype))
                    
                    if device != "cpu":
                        tensor = tensor.to(device)
                    
                    state_dict[param_name] = tensor
                    processed_params.add(param_name)
                    tensor_count += 1
                    
                except Exception as e:
                    logger.warning(f"Ошибка загрузки параметра {param_name}: {e}")
                    continue
            
            logger.info(f"Собрано {tensor_count} тензоров в state_dict")
            return state_dict
            
        except Exception as e:
            logger.error(f"Ошибка реконструкции state_dict: {e}", exc_info=True)
            return {}
    
    def load_from_disk(self, model_dir: str, lazy: bool = False) -> bool:
        """
        Загружает фрактальную структуру с диска
        
        Args:
            model_dir: Директория модели
            lazy: Использовать ленивую загрузку
            
        Returns:
            bool: Успешность загрузки
        """
        try:
            index_path = os.path.join(model_dir, "index.json")
            containers_path = os.path.join(model_dir, "containers.json")
            data_dir = os.path.join(model_dir, "data")
            
            if lazy:
                # Ленивая загрузка
                if os.path.exists(index_path):
                    with open(index_path, 'r', encoding='utf-8') as f:
                        self.index = json.load(f)
                    
                    # Загружаем индекс для ленивой загрузки
                    lazy_index_path = os.path.join(model_dir, "lazy_index.json")
                    if os.path.exists(lazy_index_path):
                        with open(lazy_index_path, 'r', encoding='utf-8') as f:
                            self.lazy_index = json.load(f)
                    
                    self.data_dir = data_dir
                    logger.info("Ленивая загрузка завершена")
                    return True
                else:
                    logger.error("Файл индекса не найден")
                    return False
            else:
                # Полная загрузка
                return self._load_legacy_format(index_path, containers_path, data_dir)
                
        except Exception as e:
            logger.error(f"Ошибка загрузки с диска: {e}", exc_info=True)
            return False
