"""
Утилиты и дополнительные методы для FractalWeightStore
"""
import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple
from .fractal_store_core import FractalWeightStore, FractalContainer

logger = logging.getLogger(__name__)

class FractalStoreUtils:
    """Утилиты для работы с фрактальным хранилищем"""
    
    @staticmethod
    def optimize_memory_usage(store: FractalWeightStore) -> bool:
        """
        Оптимизирует использование памяти в хранилище
        
        Args:
            store: Экземпляр FractalWeightStore
            
        Returns:
            bool: Успешность оптимизации
        """
        try:
            # Удаляем неиспользуемые контейнеры
            unused_containers = []
            for cid, container in store.containers.items():
                if container.access_count == 0 and (time.time() - container.timestamp) > 3600:
                    unused_containers.append(cid)
            
            for cid in unused_containers:
                del store.containers[cid]
                logger.info(f"Удален неиспользуемый контейнер: {cid}")
            
            # Пересчитываем общую память
            store.total_memory = sum(c.get_memory_size() for c in store.containers.values())
            
            logger.info(f"Оптимизация памяти завершена. Удалено: {len(unused_containers)} контейнеров")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка оптимизации памяти: {e}")
            return False
    
    @staticmethod
    def get_memory_stats(store: FractalWeightStore) -> Dict[str, Any]:
        """
        Возвращает статистику использования памяти
        
        Args:
            store: Экземпляр FractalWeightStore
            
        Returns:
            Dict[str, Any]: Статистика памяти
        """
        try:
            total_containers = len(store.containers)
            total_memory = sum(c.get_memory_size() for c in store.containers.values())
            
            # Статистика по приоритетам
            priority_stats = {}
            for container in store.containers.values():
                priority = container.priority
                if priority not in priority_stats:
                    priority_stats[priority] = {"count": 0, "memory": 0}
                priority_stats[priority]["count"] += 1
                priority_stats[priority]["memory"] += container.get_memory_size()
            
            return {
                "total_containers": total_containers,
                "total_memory_bytes": total_memory,
                "total_memory_mb": total_memory / (1024 * 1024),
                "max_memory_bytes": store.max_memory_bytes,
                "memory_usage_percent": (total_memory / store.max_memory_bytes) * 100,
                "priority_stats": priority_stats,
                "device": store.device
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения статистики памяти: {e}")
            return {}
    
    @staticmethod
    def validate_store_integrity(store: FractalWeightStore) -> Tuple[bool, List[str]]:
        """
        Проверяет целостность хранилища
        
        Args:
            store: Экземпляр FractalWeightStore
            
        Returns:
            Tuple[bool, List[str]]: (Валидность, список ошибок)
        """
        errors = []
        
        try:
            # Проверяем индекс
            if not store.index:
                errors.append("Индекс хранилища пуст")
            
            # Проверяем контейнеры
            for cid, container in store.containers.items():
                if not hasattr(container, 'data'):
                    errors.append(f"Контейнер {cid} не имеет атрибута data")
                elif not isinstance(container.data, np.ndarray):
                    errors.append(f"Контейнер {cid} имеет неверный тип данных")
            
            # Проверяем соответствие индекса и контейнеров
            for param_name, param_info in store.index.items():
                container_id = param_info.get("container_id")
                if container_id and container_id not in store.containers:
                    errors.append(f"Для параметра {param_name} не найден контейнер {container_id}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            return False, [f"Ошибка проверки целостности: {e}"]
    
    @staticmethod
    def export_store(store: FractalWeightStore, export_path: str) -> bool:
        """
        Экспортирует хранилище в файл
        
        Args:
            store: Экземпляр FractalWeightStore
            export_path: Путь для экспорта
            
        Returns:
            bool: Успешность экспорта
        """
        try:
            import json
            
            export_data = {
                "index": store.index,
                "containers": {},
                "metadata": {
                    "device": store.device,
                    "total_memory": store.total_memory,
                    "max_memory_bytes": store.max_memory_bytes,
                    "export_timestamp": time.time()
                }
            }
            
            # Экспортируем контейнеры
            for cid, container in store.containers.items():
                export_data["containers"][cid] = {
                    "data": container.data.tolist(),
                    "dtype": str(container.data.dtype),
                    "priority": container.priority,
                    "timestamp": container.timestamp,
                    "access_count": container.access_count
                }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Хранилище экспортировано в {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка экспорта хранилища: {e}")
            return False
    
    @staticmethod
    def import_store(store: FractalWeightStore, import_path: str) -> bool:
        """
        Импортирует хранилище из файла
        
        Args:
            store: Экземпляр FractalWeightStore
            import_path: Путь для импорта
            
        Returns:
            bool: Успешность импорта
        """
        try:
            import json
            import time
            
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # Импортируем индекс
            store.index = import_data.get("index", {})
            
            # Импортируем контейнеры
            containers_data = import_data.get("containers", {})
            store.containers = {}
            
            for cid, container_info in containers_data.items():
                data = np.array(container_info["data"], dtype=container_info["dtype"])
                container = FractalContainer(
                    data=data,
                    dtype=container_info["dtype"],
                    priority=container_info["priority"]
                )
                container.timestamp = container_info["timestamp"]
                container.access_count = container_info["access_count"]
                store.containers[cid] = container
            
            # Восстанавливаем метаданные
            metadata = import_data.get("metadata", {})
            store.device = metadata.get("device", "cpu")
            store.total_memory = metadata.get("total_memory", 0)
            store.max_memory_bytes = metadata.get("max_memory_bytes", 16 * 1024**3)
            
            logger.info(f"Хранилище импортировано из {import_path}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка импорта хранилища: {e}")
            return False
