"""
FCPInferenceAPI - Низкоуровневый OpenVINO инференс API для EVA
Простой и надежный инференс без State API
"""
import os
import logging
import numpy as np
from typing import Optional, Tuple, List, Dict, Any

logger = logging.getLogger("eva_ai.fcp_inference_api")


class FCPInferenceAPI:
    """
    Низкоуровневый API для инференса модели.
    Работает напрямую с OpenVINO без State API.
    """
    
    def __init__(self, model_path: str, device: str = "CPU", max_seq_len: int = 2048):
        self.model_path = model_path
        self.device = device
        self.max_seq_len = max_seq_len
        
        self.core = None
        self.model = None
        self.compiled = None
        self.request = None
        self._initialized = False
        
        self._init_model()
    
    def _init_model(self):
        """Инициализация модели"""
        try:
            import openvino as ov
            
            self.core = ov.Core()
            
            model_xml = os.path.join(self.model_path, "openvino_model.xml")
            logger.info(f"[FCP API] Loading model from: {model_xml}")
            
            self.model = self.core.read_model(model_xml)
            
            # Пробуем установить partial shapes для входов
            for inp in self.model.inputs:
                inp_name = list(inp.names)[0] if inp.names else str(inp)
                
                # Partial shape с диапазоном: [batch=1, seq=1..max_seq_len]
                ps = ov.PartialShape([1, ov.Dimension(1, self.max_seq_len)])
                
                try:
                    self.model.reshape({inp_name: ps})
                    logger.info(f"[FCP API] Reshaped {inp_name} to partial")
                except Exception as e:
                    logger.debug(f"[FCP API] reshape for {inp_name} failed: {e}")
            
            # Компилируем модель
            self.compiled = self.core.compile_model(self.model, self.device)
            self.request = self.compiled.create_infer_request()
            self._initialized = True
            
            logger.info("[FCP API] Model initialized successfully")
            
        except Exception as e:
            logger.error(f"[FCP API] Failed to initialize: {e}")
            self._initialized = False
    
    def is_initialized(self) -> bool:
        """Проверка инициализации"""
        return self._initialized and self.request is not None
    
    def get_inputs_info(self) -> Dict[str, Any]:
        """Получить информацию о входах модели"""
        if not self.is_initialized():
            return {}
        
        info = {}
        for inp in self.compiled.inputs:
            name = list(inp.names)[0] if inp.names else str(inp)
            info[name] = {"dtype": inp.element_type}
        return info
    
    def get_outputs_info(self) -> Dict[str, Any]:
        """Получить информацию о выходах модели"""
        if not self.is_initialized():
            return {}
        
        info = {}
        for out in self.compiled.outputs:
            name = list(out.names)[0] if out.names else str(out)
            info[name] = {"dtype": out.element_type}
        return info
    
    def infer(
        self, 
        input_ids: np.ndarray, 
        attention_mask: Optional[np.ndarray] = None,
        position_ids: Optional[np.ndarray] = None,
        beam_idx: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """Инференс модели"""
        if not self.is_initialized():
            raise RuntimeError("[FCP API] Model not initialized")
        
        inputs = {}
        
        if input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)
        inputs["input_ids"] = input_ids.astype(np.int64)
        
        if attention_mask is None:
            attention_mask = np.ones_like(input_ids)
        inputs["attention_mask"] = attention_mask.astype(np.int64)
        
        if position_ids is None:
            seq_len = input_ids.shape[1]
            position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
        inputs["position_ids"] = position_ids.astype(np.int64)
        
        # beam_idx required for ScaledDotProductAttentionWithKVCache models
        batch_size = input_ids.shape[0]
        if beam_idx is None:
            beam_idx = np.zeros((batch_size,), dtype=np.int32)
        inputs["beam_idx"] = beam_idx.astype(np.int32)
        
        results = self.request.infer(inputs)
        
        outputs = {}
        for name, tensor in results.items():
            outputs[name] = np.array(tensor)
        
        return outputs
    
    def infer_step(self, input_id: int) -> Tuple[np.ndarray, int]:
        """Инференс одного токена"""
        if not self.is_initialized():
            raise RuntimeError("[FCP API] Model not initialized")
        
        input_ids = np.array([[input_id]], dtype=np.int64)
        attention_mask = np.ones_like(input_ids, dtype=np.int64)
        
        outputs = self.infer(input_ids, attention_mask)
        
        logits = None
        for key in outputs:
            if "logits" in key.lower() or "head" in key.lower():
                logits = outputs[key]
                break
        
        if logits is None:
            raise RuntimeError("[FCP API] Cannot find logits")
        
        logits_last = logits[0, -1, :] if logits.ndim == 3 else logits[0, -1]
        next_token = int(np.argmax(logits_last))
        
        return logits_last, next_token
    
    def generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9
    ) -> List[int]:
        """Генерация токенов"""
        if not self.is_initialized():
            raise RuntimeError("[FCP API] Model not initialized")
        
        generated = prompt_tokens.copy()
        
        for _ in range(max_new_tokens):
            logits, next_token = self.infer_step(generated[-1])
            
            if temperature != 1.0:
                logits = logits / temperature
            
            if top_k > 0:
                indices = np.argsort(logits)[-top_k:]
                mask = np.zeros_like(logits)
                mask[indices] = logits[indices]
                logits = mask
            
            if top_p < 1.0:
                sorted_idx = np.argsort(logits)[::-1]
                probs = np.exp(logits[sorted_idx])
                cumsum = np.cumsum(probs)
                mask = cumsum > top_p
                logits[sorted_idx[mask]] = -np.inf
            
            probs = np.exp(logits) / np.sum(np.exp(logits))
            next_token = int(np.argmax(probs))
            
            generated.append(next_token)
            
            if next_token == 2:
                break
        
        return generated
    
    def get_kv_cache(self) -> Dict[str, np.ndarray]:
        """Получить KV кэш"""
        if not self.is_initialized():
            return {}
        
        try:
            states = self.request.query_state()
            kv_cache = {}
            for state in states:
                name = state.name if hasattr(state, 'name') else str(state)
                kv_cache[name] = np.array(state.state)
            return kv_cache
        except Exception as e:
            logger.debug(f"[FCP API] Cannot get KV cache: {e}")
            return {}
    
    def reset_states(self):
        """Сбросить состояния"""
        if not self.is_initialized():
            return
        
        try:
            states = self.request.query_state()
            for state in states:
                shape = state.state.shape
                zeros = np.zeros(shape, dtype=np.float32)
                state.state = zeros
        except Exception as e:
            logger.debug(f"[FCP API] Cannot reset states: {e}")
    
    def close(self):
        """Закрыть модель"""
        self.request = None
        self.compiled = None
        self.model = None
        logger.info("[FCP API] Model closed")


class TokenizerWrapper:
    """Обертка для токенизатора"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def encode(self, text: str) -> np.ndarray:
        """Кодировать текст"""
        try:
            tokenized = self.tokenizer.encode(text)
            
            if hasattr(tokenized, 'input_ids'):
                ids = tokenized.input_ids
                if hasattr(ids, 'data'):
                    size = ids.get_size() if hasattr(ids, 'get_size') else len(ids.data)
                    data = np.frombuffer(ids.data, dtype=np.int64, count=size)
                    return data
            
            return np.asarray(tokenized, dtype=np.int64)
        except Exception as e:
            logger.error(f"[TokenizerWrapper] Encode failed: {e}")
            return np.array([], dtype=np.int64)
    
    def decode(self, token_ids: List[int]) -> str:
        """Декодировать токены"""
        try:
            if hasattr(self.tokenizer, 'decode'):
                return self.tokenizer.decode(token_ids)
            return ""
        except Exception as e:
            logger.error(f"[TokenizerWrapper] Decode failed: {e}")
            return ""


def create_inference_api(model_path: str, tokenizer, device: str = "CPU", max_seq_len: int = 2048):
    """Создать инференс API"""
    api = FCPInferenceAPI(model_path, device, max_seq_len)
    wrapper = TokenizerWrapper(tokenizer)
    return api, wrapper