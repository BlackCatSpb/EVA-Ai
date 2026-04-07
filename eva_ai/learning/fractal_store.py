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
        except (RuntimeError, OSError) as e:
            logger.warning(f"Error setting device: {e}")
            self.device = "cpu"
            
        logger.info(f"FractalWeightStore initialized on {self.device}")
        
    def store(self, key: str, data: Union[np.ndarray, torch.Tensor], priority: float = 1.0) -> bool:
        """Сохраняет данные в хранилище"""
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
            
        except KeyError as e:
            logger.warning(f"Key error storing {key}: {e}")
            return False
        except (TypeError, ValueError) as e:
            logger.warning(f"Type/value error storing {key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error storing {key}: {e}")
            return False
            
    def get(self, key: str) -> Optional[np.ndarray]:
        """Получает данные из хранилища"""
        container = self.containers.get(key)
        if container:
            container.access_count += 1
            return container.data
        return None
        
    def _cleanup(self) -> None:
        """Очищает старые данные"""
        if not self.containers:
            return
            
        # Сортируем по приоритету и времени доступа
        sorted_items = sorted(
            self.containers.items(),
            key=lambda x: (x[1].priority, x[1].timestamp, x[1].access_count)
        )
        
        # Удаляем 25% самых старых данных
        to_remove = len(sorted_items) // 4
        for key, container in sorted_items[:to_remove]:
            self.total_memory -= container.get_memory_size()
            del self.containers[key]
            
        logger.info(f"Cleaned up {to_remove} items, freed memory")
        
    def get_memory_usage(self) -> Dict[str, Any]:
        """Возвращает статистику использования памяти"""
        return {
            "total_memory_bytes": self.total_memory,
            "max_memory_bytes": self.max_memory_bytes,
            "items_count": len(self.containers),
            "utilization": self.total_memory / self.max_memory_bytes
        }