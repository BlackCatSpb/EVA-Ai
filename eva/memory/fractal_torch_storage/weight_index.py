"""
WeightIndex - индексация весов моделей.
Фрактальная структура для быстрого поиска по слоям и компонентам.
"""
import logging
from typing import Dict, List, Optional

logger = logging.getLogger("eva.memory.fractal_torch_storage.index")


class WeightIndex:
    """
    Индекс весов моделей с фрактальной структурой.
    
    Иерархия:
    root → model → layers → components → tensors
    """
    
    def __init__(self):
        # Главный индекс
        self._index: Dict[str, Dict] = {}
        
        # Индекс по слоям
        self._layer_index: Dict[str, List[str]] = {}
        
        # Индекс по компонентам
        self._component_index: Dict[str, List[str]] = {}
        
        # Индекс по форме тензоров
        self._shape_index: Dict[tuple, List[str]] = {}
        
        logger.info("WeightIndex инициализирован")
    
    def register(
        self,
        key: str,
        layer_name: str,
        tensor_name: str,
        shape: tuple,
        dtype: str,
        size_bytes: int
    ):
        """
        Регистрирует вес в индексе.
        
        Args:
            key: Уникальный ключ
            layer_name: Имя слоя
            tensor_name: Имя тензора
            shape: Форма тензора
            dtype: Тип данных
            size_bytes: Размер в байтах
        """
        # Основной индекс
        self._index[key] = {
            "layer": layer_name,
            "tensor": tensor_name,
            "shape": shape,
            "dtype": dtype,
            "size_bytes": size_bytes
        }
        
        # Индекс по слоям
        if layer_name not in self._layer_index:
            self._layer_index[layer_name] = []
        self._layer_index[layer_name].append(key)
        
        # Индекс по компонентам
        component = self._extract_component(layer_name)
        if component not in self._component_index:
            self._component_index[component] = []
        self._component_index[component].append(key)
        
        # Индекс по форме
        shape_key = tuple(shape)
        if shape_key not in self._shape_index:
            self._shape_index[shape_key] = []
        self._shape_index[shape_key].append(key)
    
    def get(self, key: str) -> Optional[Dict]:
        """
        Получает метаданные по ключу.
        
        Args:
            key: Ключ тензора
            
        Returns:
            Optional[Dict]: Метаданные или None
        """
        return self._index.get(key)
    
    def get_layer_keys(self, layer_name: str) -> List[str]:
        """
        Получает все ключи слоя.
        
        Args:
            layer_name: Имя слоя
            
        Returns:
            List[str]: Список ключей
        """
        return self._layer_index.get(layer_name, [])
    
    def get_component_keys(self, component: str) -> List[str]:
        """
        Получает все ключи компонента.
        
        Args:
            component: Имя компонента
            
        Returns:
            List[str]: Список ключей
        """
        return self._component_index.get(component, [])
    
    def get_by_shape(self, shape: tuple) -> List[str]:
        """
        Получает все ключи с данной формой.
        
        Args:
            shape: Форма тензора
            
        Returns:
            List[str]: Список ключей
        """
        return self._shape_index.get(shape, [])
    
    def _extract_component(self, layer_name: str) -> str:
        """Извлекает компонент из имени слоя."""
        parts = layer_name.split(".")
        
        # Ищем известные компоненты
        components = ["self_attn", "mlp", "input_layernorm", "post_attention_layernorm"]
        
        for part in parts:
            if part in components:
                return part
        
        # Fallback: второй с конца
        if len(parts) >= 2:
            return parts[-2]
        
        return parts[0] if parts else "unknown"
    
    @property
    def layer_names(self) -> List[str]:
        """Возвращает список имён слоёв."""
        return list(self._layer_index.keys())
    
    @property
    def component_names(self) -> List[str]:
        """Возвращает список имён компонентов."""
        return list(self._component_index.keys())
    
    @property
    def total_weights(self) -> int:
        """Возвращает общее количество весов."""
        return len(self._index)
    
    @property
    def total_bytes(self) -> int:
        """Возвращает общий размер в байтах."""
        return sum(m.get("size_bytes", 0) for m in self._index.values())
    
    def to_dict(self) -> Dict:
        """Конвертирует индекс в словарь."""
        return {
            "index": self._index,
            "layer_index": self._layer_index,
            "component_index": {k: v for k, v in self._component_index.items()},
            "shape_index": {str(k): v for k, v in self._shape_index.items()}
        }
    
    def from_dict(self, data: Dict):
        """Загружает индекс из словаря."""
        self._index = data.get("index", {})
        self._layer_index = data.get("layer_index", {})
        self._component_index = data.get("component_index", {})
        
        # Восстанавливаем shape_index
        shape_index = {}
        for k, v in data.get("shape_index", {}).items():
            try:
                shape = eval(k) if isinstance(k, str) else k
                shape_index[shape] = v
            except Exception:
                pass
        self._shape_index = shape_index
