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
        
        # Компилируем с динамическими формами - разрешаем любую длину последовательности
        # Сначала устанавливаем динамические формы для всех входов
        for inp in self.model.inputs:
            inp_name = list(inp.names)[0] if inp.names else str(inp)
            shape = list(inp.shape)
            # Делаем последнее измерение динамическим (-1)
            if len(shape) > 0:
                # Проверяем, уже ли форма динамическая
                is_already_dynamic = any(s == -1 or isinstance(s, str) for s in shape)
                if not is_already_dynamic:
                    shape[-1] = -1
                    try:
                        self.model.reshape({inp_name: shape})
                        logger.info(f"[StateInjector] Set dynamic shape for {inp_name}: {shape}")
                    except Exception as e:
                        logger.warning(f"[StateInjector] Could not reshape {inp_name}: {e}")
                else:
                    logger.info(f"[StateInjector] Shape already dynamic for {inp_name}: {shape}")
        
        # Компилируем модель - пробуем сначала с динамическими формами
        try:
            self.compiled = self.core.compile_model(self.model, device)
            logger.info("[StateInjector] Model compiled with dynamic shapes")
        except Exception as e:
            logger.warning(f"[StateInjector] Dynamic shape compile failed: {e}, trying fixed shape")
            # Fallback: компилируем с фиксированной формой (1 токен, 2048 макс)
            try:
                fixed_model = self.core.read_model(model_path)
                for inp in fixed_model.inputs:
                    inp_name = list(inp.names)[0] if inp.names else str(inp)
                    shape = list(inp.shape)
                    # Фиксируем форму: [batch, seq]
                    if len(shape) >= 2:
                        shape[-1] = 1  # Один токен
                    fixed_model.reshape({inp_name: shape})
                self.compiled = self.core.compile_model(fixed_model, device)
                logger.info("[StateInjector] Model compiled with fixed shape (fallback)")
            except Exception as e2:
                logger.error(f"[StateInjector] Both dynamic and fixed compile failed: {e2}")
                raise RuntimeError(f"StateInjector cannot compile model: {e2}")
        
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
            # Проверяем на динамические формы
            if any(s == -1 or isinstance(s, str) for s in shape):
                logger.warning(f"[StateInjector] Dynamic shape detected in _infer_shapes: {shape}")
                # Пробуем получить конкретные размерности из другого источника
                # Или используем значения по умолчанию
                self._kv_heads = 8  # Значения по умолчанию для Qwen
                self._head_dim = 128
            else:
                self._kv_heads = shape[1]
                self._head_dim = shape[3]
            logger.info(f"Inferred shapes: kv_heads={self._kv_heads}, head_dim={self._head_dim}")
        except Exception as e:
            logger.warning(f"Could not infer shapes: {e}, using defaults (8, 128)")
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
        if layer_idx not in self._state_map:
            return np.array([])
        try:
            state = self._state_map[layer_idx]["value"]
            return np.array(state.state.data).copy()
        except Exception as e:
            logger.warning(f"[StateInjector] get_value failed at layer {layer_idx}: {e}")
            return np.array([])

    def set_value(self, layer_idx: int, data: np.ndarray):
        if layer_idx not in self._state_map:
            return
        try:
            state = self._state_map[layer_idx]["value"]
            data = np.asarray(data, dtype=np.float32)
            # Проверяем форму
            current_shape = state.state.shape
            if any(s == -1 or isinstance(s, str) for s in current_shape):
                logger.warning(f"[StateInjector] set_value: dynamic shape {current_shape}, skipping")
                return
            state.set_state(ov.Tensor(data))
        except Exception as e:
            logger.warning(f"[StateInjector] set_value failed at layer {layer_idx}: {e}")

    def get_key(self, layer_idx: int) -> np.ndarray:
        if layer_idx not in self._state_map:
            return np.array([])
        try:
            state = self._state_map[layer_idx]["key"]
            return np.array(state.state.data).copy()
        except Exception as e:
            logger.warning(f"[StateInjector] get_key failed at layer {layer_idx}: {e}")
            return np.array([])

    def set_key(self, layer_idx: int, data: np.ndarray):
        if layer_idx not in self._state_map:
            return
        try:
            state = self._state_map[layer_idx]["key"]
            data = np.asarray(data, dtype=np.float32)
            # Проверяем форму
            current_shape = state.state.shape
            if any(s == -1 or isinstance(s, str) for s in current_shape):
                logger.warning(f"[StateInjector] set_key: dynamic shape {current_shape}, skipping")
                return
            state.set_state(ov.Tensor(data))
        except Exception as e:
            logger.warning(f"[StateInjector] set_key failed at layer {layer_idx}: {e}")

    def get_all_layer_indices(self) -> List[int]:
        """Получить список всех индексов слоёв."""
        return self._layer_indices
    
    def get_all_layer_states(self) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Получить состояния всех слоёв KV-кеша.
        
        Returns:
            {layer_idx: {"key": np.ndarray, "value": np.ndarray}, ...}
        """
        all_states = {}
        for layer_idx in self._layer_indices:
            if layer_idx in self._state_map:
                key_data = self.get_key(layer_idx)
                val_data = self.get_value(layer_idx)
                all_states[layer_idx] = {
                    "key": key_data,
                    "value": val_data
                }
        return all_states
    
    def set_all_layer_states(self, states_dict: Dict[int, Dict[str, np.ndarray]]):
        """
        Установить состояния всех слоёв KV-кеша.
        
        Args:
            states_dict: {layer_idx: {"key": np.ndarray, "value": np.ndarray}, ...}
        """
        for layer_idx, states in states_dict.items():
            if layer_idx not in self._state_map:
                continue
            if "key" in states and len(states["key"]) > 0:
                self.set_key(layer_idx, states["key"])
            if "value" in states and len(states["value"]) > 0:
                self.set_value(layer_idx, states["value"])
    
    def has_dynamic_shapes(self) -> bool:
        """Проверить, есть ли динамические формы в состояниях."""
        if not self._layer_indices:
            return True
        try:
            val_state = self._state_map[self._layer_indices[0]]["value"]
            state_data = self._get_state_data(val_state)
            if state_data is None:
                return True
            shape = state_data.shape
            return any(s == -1 or isinstance(s, str) for s in shape)
        except Exception:
            return True
    
    def get_layer_count(self) -> int:
        """Получить количество слоёв в модели (ожидается 36 для FCP)."""
        return len(self._layer_indices)
    
    def is_layer_supported(self, layer_idx: int) -> bool:
        """Проверить, поддерживается ли слой с данным индексом."""
        return layer_idx in self._state_map
    
    def transform_values(self, layer_indices: List[int], func: Callable, **kwargs):
        """Применяет функцию модификации к Value тензорам указанных слоев."""
        for idx in layer_indices:
            if idx in self._state_map:
                try:
                    val = self.get_value(idx)
                    val = func(val, **kwargs)
                    self.set_value(idx, val)
                except Exception as e:
                    logger.warning(f"[StateInjector] transform_values failed at layer {idx}: {e}")

    def transform_keys(self, layer_indices: List[int], func: Callable, **kwargs):
        """Применяет функцию модификации к Key тензорам указанных слоев."""
        for idx in layer_indices:
            if idx in self._state_map:
                try:
                    key = self.get_key(idx)
                    key = func(key, **kwargs)
                    self.set_key(idx, key)
                except Exception as e:
                    logger.warning(f"[StateInjector] transform_keys failed at layer {idx}: {e}")
