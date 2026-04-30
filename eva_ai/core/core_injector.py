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
        for state in states:
            match = pattern.search(state.get_name())
            if match:
                layer_idx = int(match.group(1))
                stype = match.group(2)
                self._state_map.setdefault(layer_idx, {})[stype] = state
        self._layer_indices = sorted(self._state_map.keys())

    def _infer_shapes(self):
        """Определение размерностей (поддержка GQA)."""
        val_state = self._state_map[self._layer_indices[0]]["value"]
        shape = val_state.get_state().get_shape()
        self._kv_heads = shape[1]
        self._head_dim = shape[3]

    def reset_all_states(self):
        """Сброс KV-кеша перед новой сессией."""
        for state in self.request.query_state():
            state.reset()

    def get_value(self, layer_idx: int) -> np.ndarray:
        return self._state_map[layer_idx]["value"].get_state().data.copy()

    def set_value(self, layer_idx: int, data: np.ndarray):
        state = self._state_map[layer_idx]["value"]
        current_tensor = state.get_state()
        
        # 1. Проверка типа данных состояния модели (FP16, FP32, и т.д.)
        target_type = current_tensor.get_element_type()
        
        # 2. Приведение numpy-массива к нужному типу
        if str(target_type) == "f16":
            data = data.astype(np.float16)
        elif str(target_type) == "f32":
            data = data.astype(np.float32)
        # Если вдруг INT8 (редко для кеша, но возможно)
        elif str(target_type) == "u8":
            # Требуется квантование вектора, иначе данные будут искажены
            data = np.clip(data * 127, -128, 127).astype(np.int8) 
        
        # 3. Запись
        state.set_state(ov.Tensor(data))

    def get_key(self, layer_idx: int) -> np.ndarray:
        return self._state_map[layer_idx]["key"].get_state().data.copy()

    def set_key(self, layer_idx: int, data: np.ndarray):
        state = self._state_map[layer_idx]["key"]
        current_tensor = state.get_state()
        
        # 1. Проверка типа данных состояния модели (FP16, FP32, и т.д.)
        target_type = current_tensor.get_element_type()
        
        # 2. Приведение numpy-массива к нужному типу
        if str(target_type) == "f16":
            data = data.astype(np.float16)
        elif str(target_type) == "f32":
            data = data.astype(np.float32)
        # Если вдруг INT8 (редко для кеша, но возможно)
        elif str(target_type) == "u8":
            # Требуется квантование вектора, иначе данные будут искажены
            data = np.clip(data * 127, -128, 127).astype(np.int8) 
        
        # 3. Запись
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