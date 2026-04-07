"""
Base Fractal Storage для PyTorch моделей.
Реализует torch.Storage API с фрактальной адресацией.
"""
import os
import logging
import threading
from typing import Dict, Any, Optional, List

logger = logging.getLogger("eva_ai.memory.fractal_torch_storage.base")


class FractalWeightStorage:
    """
    Фрактальное хранилище весов PyTorch моделей.
    
    Архитектура:
    - Индексация по слоям и компонентам
    - Горячие/холодные зоны для оптимизации доступа
    - Сжатие редко используемых весов
    
    Примечание: Это прототип. Для реального использования
    нужно наследовать torch.Storage и переопределить методы.
    """
    
    def __init__(
        self,
        storage_dir: Optional[str] = None,
        max_cache_size_gb: float = 28.0
    ):
        """
        Args:
            storage_dir: Директория для хранения
            max_cache_size_gb: Максимальный размер кэша (28GB для Qwen3-14B)
        """
        self.storage_dir = storage_dir or os.path.join(
            os.path.dirname(__file__), "fractal_torch_data"
        )
        os.makedirs(self.storage_dir, exist_ok=True)
        
        self.max_cache_bytes = int(max_cache_size_gb * 1024**3)
        
        # Индекс слоёв
        from .weight_index import WeightIndex
        self.index = WeightIndex()
        
        # Менеджер слоёв
        from .layer_manager import LayerManager
        self.layer_manager = LayerManager()
        
        # Компрессор
        from .compression import WeightCompressor
        self.compressor = WeightCompressor()
        
        # Данные в памяти
        self._data: Dict[str, bytes] = {}
        self._lock = threading.RLock()
        
        # Статистика
        self.stats = {
            "reads": 0,
            "writes": 0,
            "compressions": 0,
            "decompressions": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        logger.info(
            f"FractalWeightStorage инициализирован: "
            f"max_cache={max_cache_size_gb}GB, dir={self.storage_dir}"
        )
    
    def store_weight(
        self,
        layer_name: str,
        tensor_name: str,
        data: bytes,
        shape: tuple,
        dtype: str
    ) -> str:
        """
        Сохраняет вес тензора.
        
        Args:
            layer_name: Имя слоя (e.g., "model.layers.0.self_attn.q_proj")
            tensor_name: Имя тензора (e.g., "weight", "bias")
            data: Данные в байтах
            shape: Форма тензора
            dtype: Тип данных
            
        Returns:
            str: Ключ для доступа
        """
        with self._lock:
            key = f"{layer_name}.{tensor_name}"
            
            # Компрессия если нужно
            if self.compressor.should_compress(shape):
                compressed = self.compressor.compress(data, shape)
                storage_data = compressed
                self.stats["compressions"] += 1
            else:
                storage_data = data
            
            # Сохраняем
            self._data[key] = storage_data
            
            # Обновляем индекс
            self.index.register(
                key=key,
                layer_name=layer_name,
                tensor_name=tensor_name,
                shape=shape,
                dtype=dtype,
                size_bytes=len(storage_data)
            )
            
            # Обновляем менеджер слоёв
            self.layer_manager.register_layer(layer_name)
            
            self.stats["writes"] += 1
            
            return key
    
    def load_weight(self, key: str) -> Optional[bytes]:
        """
        Загружает вес тензора.
        
        Args:
            key: Ключ тензора
            
        Returns:
            Optional[bytes]: Данные или None
        """
        with self._lock:
            if key in self._data:
                data = self._data[key]
                meta = self.index.get(key)
                
                # Декомпрессия если нужно
                if meta and meta.get("compressed"):
                    data = self.compressor.decompress(data, meta.get("shape"))
                    self.stats["decompressions"] += 1
                
                self.stats["reads"] += 1
                self.stats["cache_hits"] += 1
                
                return data
            
            self.stats["cache_misses"] += 1
            return None
    
    def get_layer_weights(self, layer_name: str) -> Dict[str, bytes]:
        """
        Получает все веса слоя.
        
        Args:
            layer_name: Имя слоя
            
        Returns:
            Dict[str, bytes]: {tensor_name: data}
        """
        result = {}
        
        for key in self.index.get_layer_keys(layer_name):
            data = self.load_weight(key)
            if data is not None:
                tensor_name = key.split(".")[-1]
                result[tensor_name] = data
        
        return result
    
    def promote_to_hot(self, layer_name: str):
        """
        Переводит слой в горячую зону (приоритетный доступ).
        
        Args:
            layer_name: Имя слоя
        """
        self.layer_manager.promote_to_hot(layer_name)
        logger.debug(f"Слой {layer_name} переведён в горячую зону")
    
    def get_stats(self) -> Dict:
        """Возвращает статистику хранилища."""
        with self._lock:
            total_bytes = sum(len(v) for v in self._data.values())
            
            return {
                **self.stats,
                "total_weights": len(self._data),
                "total_bytes": total_bytes,
                "total_gb": total_bytes / (1024**3),
                "layers": len(self.layer_manager.layers),
                "hot_layers": len(self.layer_manager.hot_layers)
            }
    
    def save_to_disk(self, filename: str = "fractal_weights.bin"):
        """Сохраняет хранилище на диск."""
        filepath = os.path.join(self.storage_dir, filename)
        
        with self._lock:
            import pickle
            
            data = {
                "weights": self._data,
                "index": self.index.to_dict(),
                "layers": self.layer_manager.to_dict(),
                "stats": self.stats
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Хранилище сохранено: {filepath}")
    
    def load_from_disk(self, filename: str = "fractal_weights.bin"):
        """Загружает хранилище с диска."""
        filepath = os.path.join(self.storage_dir, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"Файл не найден: {filepath}")
            return
        
        with self._lock:
            import pickle
            
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self._data = data.get("weights", {})
            self.index.from_dict(data.get("index", {}))
            self.layer_manager.from_dict(data.get("layers", {}))
            self.stats = data.get("stats", self.stats)
            
            logger.info(f"Хранилище загружено: {filepath}")
