"""
Adaptive Parameter Controller for Recursive Model Pipeline.
Handles semantic stuck detection and parameter adaptation.
"""

import logging
import math
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class AdaptiveParameterController:
    """
    Адаптивный контроллер параметров генерации.
    Анализирует причины провалов + семантическую схожесть ответов через эмбеддер.
    """
    
    DEFAULT_PARAMS = {
        'temperature': 0.3,
        'top_p': 0.9,
        'top_k': 40,
        'repeat_penalty': 1.5,
        'max_tokens': 1024,
    }
    
    PARAM_RANGES = {
        'temperature': (0.05, 1.5),
        'top_p': (0.1, 1.0),
        'top_k': (10, 100),
        'repeat_penalty': (0.5, 3.0),
        'max_tokens': (64, 4096),
    }
    
    SEMANTIC_STUCK_THRESHOLD = 0.85
    
    def __init__(self, base_params: Dict[str, float] = None):
        self.base_params = base_params or dict(self.DEFAULT_PARAMS)
        self.current_params = dict(self.base_params)
        self.failure_history: List[Dict] = []
        self.failed_response_texts: List[str] = []
        self.failed_response_embeddings: list = []
        self.success_count = 0
        self.failure_count = 0
        self._embedder = None
    
    def _get_embedder(self):
        """Ленивая загрузка эмбеддера для семантического анализа."""
        if self._embedder is None:
            try:
                from eva_ai.mlearning.sentence_transformers_cache import get_sentence_transformer
                self._embedder = get_sentence_transformer('intfloat/multilingual-e5-base', device='cpu')
                if self._embedder is not None:
                    logger.debug("AdaptiveController: эмбеддер загружен для семантического анализа")
            except Exception as e:
                logger.debug(f"AdaptiveController: эмбеддер недоступен: {e}")
        return self._embedder
    
    def _compute_embedding(self, text: str) -> Optional[list]:
        """Вычисляет эмбеддинг текста через Model A (или fallback)."""
        embedder = self._get_embedder()
        if embedder is None or not text.strip():
            return None
        try:
            if hasattr(embedder, 'create_embedding'):
                result = embedder.create_embedding([text.strip()])
                if result and len(result) > 0:
                    return result[0]
            else:
                embedding = embedder.encode([text.strip()])[0]
                return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        except Exception as e:
            logger.debug(f"AdaptiveController: ошибка вычисления эмбеддинга: {e}")
        return None
    
    def _cosine_similarity(self, a: list, b: list) -> float:
        """Вычисляет косинусную схожесть двух векторов."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    
    def _are_embeddings_stuck(self) -> Dict[str, Any]:
        """Проверяет, застряли ли мы в семантически одинаковых ответах."""
        embeddings = self.failed_response_embeddings
        if len(embeddings) < 2:
            return {'is_stuck': False, 'max_similarity': 0.0, 'stuck_count': 0}
        
        max_sim = 0.0
        stuck_count = 0
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[-1])
            max_sim = max(max_sim, sim)
            if sim > self.SEMANTIC_STUCK_THRESHOLD:
                stuck_count += 1
        
        return {
            'is_stuck': max_sim > self.SEMANTIC_STUCK_THRESHOLD,
            'max_similarity': max_sim,
            'stuck_count': stuck_count,
        }
    
    def get_params_for_attempt(self, attempt: int, failure_reasons: List[str] = None) -> Dict[str, float]:
        """Возвращает адаптированные параметры для попытки."""
        semantic_info = self._are_embeddings_stuck()
        if semantic_info['is_stuck']:
            logger.warning(f"Model ЗАСТРЯЛА: семантическая схожесть={semantic_info['max_similarity']:.2f} "
                         f"(порог={self.SEMANTIC_STUCK_THRESHOLD}), застряло попыток: {semantic_info['stuck_count']}")
        
        if not failure_reasons and not semantic_info['is_stuck']:
            return dict(self.base_params)
        
        params = dict(self.base_params)
        
        if failure_reasons:
            for reason in failure_reasons:
                reason_lower = reason.lower()
                
                if 'зацикл' in reason_lower or 'повтор' in reason_lower or 'loop' in reason_lower:
                    params['temperature'] = min(1.0, params.get('temperature', 0.3) + 0.25)
                    params['repeat_penalty'] = min(2.5, params.get('repeat_penalty', 1.5) + 0.4)
                    params['top_k'] = max(20, params.get('top_k', 40) - 10)
                    params['top_p'] = max(0.7, params.get('top_p', 0.9) - 0.1)
                
                elif 'китайск' in reason_lower or 'chinese' in reason_lower:
                    params['temperature'] = max(0.1, params.get('temperature', 0.3) - 0.15)
                    params['top_p'] = max(0.5, params.get('top_p', 0.9) - 0.2)
                    params['repeat_penalty'] = min(2.5, params.get('repeat_penalty', 1.5) + 0.3)
                
                elif 'английск' in reason_lower or 'english' in reason_lower or 'latin' in reason_lower:
                    params['temperature'] = max(0.1, params.get('temperature', 0.3) - 0.1)
                    params['top_p'] = max(0.5, params.get('top_p', 0.9) - 0.15)
                
                elif 'фраз' in reason_lower or 'паразит' in reason_lower or 'filler' in reason_lower:
                    params['temperature'] = min(1.0, params.get('temperature', 0.3) + 0.2)
                    params['top_k'] = min(80, params.get('top_k', 40) + 20)
                    params['repeat_penalty'] = min(2.5, params.get('repeat_penalty', 1.5) + 0.2)
                
                elif 'пуст' in reason_lower or 'коротк' in reason_lower or 'empty' in reason_lower:
                    params['max_tokens'] = min(2048, params.get('max_tokens', 1024) + 256)
                    params['temperature'] = min(1.0, params.get('temperature', 0.3) + 0.15)
                
                elif 'гласн' in reason_lower or 'vowel' in reason_lower:
                    params['temperature'] = min(1.2, params.get('temperature', 0.3) + 0.35)
                    params['repeat_penalty'] = min(2.5, params.get('repeat_penalty', 1.5) + 0.3)
        
        if semantic_info['is_stuck']:
            stuck_factor = min(1.0, semantic_info['stuck_count'] / 2.0)
            logger.info(f"Semantic adaptation: stuck_factor={stuck_factor:.2f}, "
                       f"drastically changing parameters")
            params['temperature'] = min(1.5, params.get('temperature', 0.3) + 0.3 + stuck_factor * 0.3)
            params['top_p'] = max(0.3, min(1.0, params.get('top_p', 0.9) + 0.1 * (1 - stuck_factor)))
            params['top_k'] = max(15, min(80, params.get('top_k', 40) + int(20 * stuck_factor)))
            params['repeat_penalty'] = min(3.0, params.get('repeat_penalty', 1.5) + 0.3 + stuck_factor * 0.3)
        
        if len(self.failure_history) >= 2:
            cumulative_factor = min(0.5, len(self.failure_history) * 0.1)
            params['temperature'] = min(1.5, params['temperature'] + cumulative_factor)
            params['repeat_penalty'] = min(3.0, params['repeat_penalty'] + cumulative_factor * 0.5)
        
        for param_name, (min_val, max_val) in self.PARAM_RANGES.items():
            if param_name in params:
                params[param_name] = max(min_val, min(max_val, params[param_name]))
            else:
                params[param_name] = self.base_params.get(param_name, self.DEFAULT_PARAMS.get(param_name, 1.0))
        
        return params
    
    def record_failure(self, attempt: int, reasons: List[str], params_used: Dict, response_text: str = None):
        """Записывает провал + эмбеддинг ответа."""
        self.failure_history.append({
            'attempt': attempt,
            'reasons': reasons,
            'params': params_used,
        })
        self.failure_count += 1
        
        if response_text:
            self.failed_response_texts.append(response_text)
            embedding = self._compute_embedding(response_text)
            if embedding:
                self.failed_response_embeddings.append(embedding)
        
        self.failed_response_embeddings = self.failed_response_embeddings[-100:]
        self.failed_response_texts = self.failed_response_texts[-100:]
        self.failure_history = self.failure_history[-100:]
    
    def record_success(self):
        """Записывает успех."""
        self.success_count += 1
        self.failed_response_embeddings = self.failed_response_embeddings[-100:]
        self.failed_response_texts = self.failed_response_texts[-100:]
        self.failure_history = self.failure_history[-100:]
    
    def reset(self):
        """Сбрасывает состояние для нового запроса."""
        self.current_params = dict(self.base_params)
        self.failed_response_texts = []
        self.failed_response_embeddings = []
        self.failure_history = []
    
    def get_stats(self) -> Dict:
        return {
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'total_attempts': self.success_count + self.failure_count,
            'recent_failures': self.failure_history[-5:],
            'semantic_analysis_enabled': self._get_embedder() is not None,
        }
    
    def cleanup(self):
        """Освобождает ресурсы эмбеддера."""
        self._embedder = None
        self.failed_response_embeddings = []
        self.failed_response_texts = []
        self.failure_history = []

    def adapt_to_resources(self, resource_usage: dict) -> dict:
        """Адаптация параметров на основе использования ресурсов."""
        params = self.current_params.copy()
        
        cpu = resource_usage.get('cpu', 0)
        ram = resource_usage.get('ram', 0)
        
        if cpu > 0.85:
            params['max_tokens'] = min(params.get('max_tokens', 512), 256)
            params['temperature'] = max(params.get('temperature', 0.5), 0.7)
        elif cpu > 0.70:
            params['max_tokens'] = min(params.get('max_tokens', 512), 512)
        
        if ram > 0.90:
            params['max_tokens'] = 256
        
        return params
    
    def should_skip_model_c(self, resource_usage: dict) -> bool:
        """Определяет, нужно ли пропустить Model C при высокой нагрузке."""
        cpu = resource_usage.get('cpu', 0)
        ram = resource_usage.get('ram', 0)
        
        if cpu > 0.85 or ram > 0.90:
            return True
        return False
    
    def get_deferred_params(self, resource_usage: dict) -> dict:
        """Параметры для отложенной генерации."""
        return {
            'max_tokens': 256,
            'temperature': 0.3,
            'top_p': 0.9,
            'repeat_penalty': 1.5,
        }
