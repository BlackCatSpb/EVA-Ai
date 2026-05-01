import re
import numpy as np
import openvino as ov
from typing import Dict, List, Callable
import logging

logger = logging.getLogger("FCP.Injector")

class LayerwiseStateInjector:
    """
    Низкоуровневый модуль для прямого доступа к KV-кешу Stateful LLM.
    Обеспечивает чтение, запись и трансформацию тензоров Key/Value.
    """
    def __init__(self, model_path: str, device: str = "CPU"):
        self.core = ov.Core()
        self.model = self.core.read_model(model_path)
        self.compiled = self.core.compile_model(self.model, device)
        self.request = self.compiled.create_infer_request()
        
        self._state_map: Dict[int, Dict[str, ov.State]] = {}
        self._layer_indices: List[int] = []
        self._kv_heads: int = 0
        self._head_dim: int = 0
        
        self._build_state_map()
        if self._layer_indices:
            self._infer_shapes()

    def _build_state_map(self):
        """Парсинг имен состояний OpenVINO (past_key_values.N.type)."""
        states = self.request.query_state()
        pattern = re.compile(r"past_key_values\.(\d+)\.(key|value)")
        
        # OpenVINO API: state имеет атрибут name вместо метода get_name()
        for state in states:
            try:
                state_name = state.name if hasattr(state, 'name') else str(state)
            except:
                state_name = f"state_{id(state)}"
            
            match = pattern.search(state_name)
            if match:
                layer_idx = int(match.group(1))
                stype = match.group(2)
                self._state_map.setdefault(layer_idx, {})[stype] = state
        
        self._layer_indices = sorted(self._state_map.keys())
        
        # Определяем API для работы с состояниями
        self._use_new_api = False
        if self._layer_indices:
            test_state = self._state_map[self._layer_indices[0]]["value"]
            if hasattr(test_state, 'data'):
                self._use_new_api = True
                logger.info("Using OpenVINO new API for states")
    
    def _get_state_data(self, state, write=False):
        """Получить данные из состояния (OpenVINO 2026.1 API)."""
        # VariableState имеет свойство .state которое возвращает Tensor
        # Tensor имеет .data который является numpy массивом
        if hasattr(state, 'state'):
            tensor = state.state
            if hasattr(tensor, 'data'):
                return tensor.data
        raise AttributeError(f"Cannot get data from state: {state}")
    
    def _infer_shapes(self):
        """Определение размерностей (поддержка GQA)."""
        val_state = self._state_map[self._layer_indices[0]]["value"]
        try:
            tensor = val_state.state
            shape = tensor.data.shape
            self._kv_heads = shape[1]
            self._head_dim = shape[3]
            logger.info(f"Inferred shapes: kv_heads={self._kv_heads}, head_dim={self._head_dim}")
        except Exception as e:
            logger.warning(f"Could not infer shapes: {e}")
            self._kv_heads = 8
            self._head_dim = 128

    def reset_all_states(self):
        """Сброс KV-кеша перед новой сессией."""
        for state in self.request.query_state():
            try:
                state.reset()
            except:
                pass

    def get_value(self, layer_idx: int) -> np.ndarray:
        state = self._state_map[layer_idx]["value"]
        return np.array(state.state.data).copy()

    def set_value(self, layer_idx: int, data: np.ndarray):
        state = self._state_map[layer_idx]["value"]
        data = np.asarray(data, dtype=np.float32)
        # Используем set_state для записи
        state.set_state(ov.Tensor(data))

    def get_key(self, layer_idx: int) -> np.ndarray:
        state = self._state_map[layer_idx]["key"]
        return np.array(state.state.data).copy()

    def set_key(self, layer_idx: int, data: np.ndarray):
        state = self._state_map[layer_idx]["key"]
        data = np.asarray(data, dtype=np.float32)
        state.set_state(ov.Tensor(data))

    def get_all_layer_indices(self) -> List[int]:
        return self._layer_indices

    def transform_values(self, layer_indices: List[int], func: Callable, **kwargs):
        """Применяет функцию модификации к Value тензорам указанных слоев."""
        for idx in layer_indices:
            if idx in self._state_map:
                val = self.get_value(idx)
                val = func(val, **kwargs)
                self.set_value(idx, val)

    def transform_keys(self, layer_indices: List[int], func: Callable, **kwargs):
        """Применяет функцию модификации к Key тензорам указанных слоев."""
        for idx in layer_indices:
            if idx in self._state_map:
                key = self.get_key(idx)
                key = func(key, **kwargs)
                self.set_key(idx, key)