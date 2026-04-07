"""
LayerManager - управление слоями моделей.
Горячие/холодные зоны для оптимизации доступа.
"""
import logging
import time
from typing import Dict, Set, Optional

logger = logging.getLogger("eva_ai.memory.fractal_torch_storage.layers")


class LayerManager:
    """
    Управляет слоями модели с приоритетным доступом.
    
    Зоны:
    - Hot: Часто используемые слои (в RAM)
    - Warm: Периодически используемые (в RAM, сжатые)
    - Cold: Редко используемые (на диске, сжатые)
    """
    
    def __init__(self, max_hot_layers: int = 8, max_warm_layers: int = 32):
        self.max_hot = max_hot_layers
        self.max_warm = max_warm_layers
        
        # Слои по зонам
        self.hot_layers: Set[str] = set()
        self.warm_layers: Set[str] = set()
        self.cold_layers: Set[str] = set()
        
        # Статистика доступа
        self._access_count: Dict[str, int] = {}
        self._last_access: Dict[str, float] = {}
        
        logger.info(
            f"LayerManager инициализирован: hot={max_hot_layers}, warm={max_warm_layers}"
        )
    
    def register_layer(self, layer_name: str):
        """
        Регистрирует слой (начально в cold зоне).
        
        Args:
            layer_name: Имя слоя
        """
        if layer_name not in self.hot_layers:
            if layer_name not in self.warm_layers:
                self.cold_layers.add(layer_name)
    
    def promote_to_hot(self, layer_name: str):
        """
        Переводит слой в горячую зону.
        
        Args:
            layer_name: Имя слоя
        """
        # Удаляем из текущей зоны
        self.cold_layers.discard(layer_name)
        self.warm_layers.discard(layer_name)
        
        # Добавляем в hot
        self.hot_layers.add(layer_name)
        
        # Проверяем лимит hot
        if len(self.hot_layers) > self.max_hot:
            self._demote_from_hot()
    
    def promote_to_warm(self, layer_name: str):
        """
        Переводит слой в тёплую зону.
        
        Args:
            layer_name: Имя слоя
        """
        self.cold_layers.discard(layer_name)
        self.hot_layers.discard(layer_name)
        
        self.warm_layers.add(layer_name)
        
        # Проверяем лимит warm
        if len(self.warm_layers) > self.max_warm:
            self._demote_from_warm()
    
    def record_access(self, layer_name: str):
        """
        Записывает факт обращения к слою.
        Автоматически продвигает часто используемые слои.
        
        Args:
            layer_name: Имя слоя
        """
        self._access_count[layer_name] = self._access_count.get(layer_name, 0) + 1
        self._last_access[layer_name] = time.time()
        
        # Автоматическое продвижение
        count = self._access_count[layer_name]
        
        if count >= 100 and layer_name in self.cold_layers:
            self.promote_to_warm(layer_name)
        
        if count >= 500 and layer_name in self.warm_layers:
            self.promote_to_hot(layer_name)
    
    def _demote_from_hot(self):
        """Понижает наименее используемый hot слой."""
        if not self.hot_layers:
            return
        
        # Находим самый редко используемый
        min_count = float('inf')
        min_layer = None
        
        for layer in self.hot_layers:
            count = self._access_count.get(layer, 0)
            if count < min_count:
                min_count = count
                min_layer = layer
        
        if min_layer:
            self.hot_layers.discard(min_layer)
            self.warm_layers.add(min_layer)
            logger.debug(f"Слой {min_layer} понижен hot → warm")
    
    def _demote_from_warm(self):
        """Понижает наименее используемый warm слой."""
        if not self.warm_layers:
            return
        
        min_count = float('inf')
        min_layer = None
        
        for layer in self.warm_layers:
            count = self._access_count.get(layer, 0)
            if count < min_count:
                min_count = count
                min_layer = layer
        
        if min_layer:
            self.warm_layers.discard(min_layer)
            self.cold_layers.add(min_layer)
            logger.debug(f"Слой {min_layer} понижен warm → cold")
    
    def get_zone(self, layer_name: str) -> str:
        """
        Возвращает зону слоя.
        
        Args:
            layer_name: Имя слоя
            
        Returns:
            str: "hot", "warm" или "cold"
        """
        if layer_name in self.hot_layers:
            return "hot"
        elif layer_name in self.warm_layers:
            return "warm"
        else:
            return "cold"
    
    def get_stats(self) -> Dict:
        """Возвращает статистику менеджера."""
        return {
            "total_layers": len(self.hot_layers) + len(self.warm_layers) + len(self.cold_layers),
            "hot_layers": len(self.hot_layers),
            "warm_layers": len(self.warm_layers),
            "cold_layers": len(self.cold_layers),
            "total_accesses": sum(self._access_count.values())
        }
    
    def to_dict(self) -> Dict:
        """Конвертирует в словарь."""
        return {
            "hot_layers": list(self.hot_layers),
            "warm_layers": list(self.warm_layers),
            "cold_layers": list(self.cold_layers),
            "access_count": self._access_count,
            "last_access": self._last_access
        }
    
    def from_dict(self, data: Dict):
        """Загружает из словаря."""
        self.hot_layers = set(data.get("hot_layers", []))
        self.warm_layers = set(data.get("warm_layers", []))
        self.cold_layers = set(data.get("cold_layers", []))
        self._access_count = data.get("access_count", {})
        self._last_access = data.get("last_access", {})
    
    @property
    def layers(self) -> Set[str]:
        """Возвращает все слои."""
        return self.hot_layers | self.warm_layers | self.cold_layers
