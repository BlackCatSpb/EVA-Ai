"""
Обновленный FractalWeightStore - разделенный на модули
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

from .fractal_store_core import FractalWeightStore, FractalContainer
from .fractal_store_utils import FractalStoreUtils

logger = logging.getLogger(__name__)

# Экспортируем основные классы для обратной совместимости
__all__ = ['FractalWeightStore', 'FractalContainer', 'FractalStoreUtils']

# Дополнительные методы для основного класса
class EnhancedFractalWeightStore(FractalWeightStore):
    """Расширенная версия FractalWeightStore с дополнительными методами"""
    
    def __init__(self, device: str = 'cpu', max_memory_gb: float = 16.0):
        super().__init__(device, max_memory_gb)
        self.utils = FractalStoreUtils()
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """Возвращает статистику использования памяти"""
        return self.utils.get_memory_stats(self)
    
    def optimize_memory(self) -> bool:
        """Оптимизирует использование памяти"""
        return self.utils.optimize_memory_usage(self)
    
    def validate_integrity(self) -> Tuple[bool, List[str]]:
        """Проверяет целостность хранилища"""
        return self.utils.validate_store_integrity(self)
    
    def export_to_file(self, path: str) -> bool:
        """Экспортирует хранилище в файл"""
        return self.utils.export_store(self, path)
    
    def import_from_file(self, path: str) -> bool:
        """Импортирует хранилище из файла"""
        return self.utils.import_store(self, path)

# Для обратной совместимости используем расширенную версию
FractalWeightStore = EnhancedFractalWeightStore
