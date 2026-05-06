"""
Прогнозная деградация (Proactive Fallback) и мониторинг качества генерации.
Мониторит метрики в реальном времени и превентивно переключает fallback до отказа.
"""
import logging
import re
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger("eva_ai.proactive_fallback")

@dataclass
class GenerationMetrics:
    """Метрики качества генерации для прогнозной деградации."""
    token_variance: float = 0.0
    repeat_rate: float = 0.0
    avg_token_length: float = 0.0
    latency: float = 0.0
    
    # Пороги для превентивного переключения
    THRESHOLD_TOKEN_VARIANCE = 0.3
    THRESHOLD_REPEAT_RATE = 0.4
    THRESHOLD_LATENCY = 10.0  # секунд
    
    def should_degrade(self) -> bool:
        """Определяет, нужно ли превентивно переключаться на fallback."""
        return (
            self.token_variance < self.THRESHOLD_TOKEN_VARIANCE or
            self.repeat_rate > self.THRESHOLD_REPEAT_RATE or
            self.latency > self.THRESHOLD_LATENCY
        )
    
    def get_degradation_reason(self) -> Optional[str]:
        """Возвращает причину деградации."""
        if self.token_variance < self.THRESHOLD_TOKEN_VARIANCE:
            return "low_token_variance"
        if self.repeat_rate > self.THRESHOLD_REPEAT_RATE:
            return "high_repeat_rate"
        if self.latency > self.THRESHOLD_LATENCY:
            return "high_latency"
        return None


class ProactiveDegradationMonitor:
    """Мониторинг метрик генерации для прогнозной деградации."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.THRESHOLD_TOKEN_VARIANCE = self.config.get('token_variance_threshold', 0.3)
        self.THRESHOLD_REPEAT_RATE = self.config.get('repeat_rate_threshold', 0.4)
        self.THRESHOLD_LATENCY = self.config.get('latency_threshold', 10.0)
        
    def analyze_response(self, response: str, latency: float = 0.0) -> GenerationMetrics:
        """Анализирует ответ и вычисляет метрики качества."""
        metrics = GenerationMetrics()
        metrics.latency = latency
        
        if not response:
            return metrics
        
        # Токенизация (простая)
        tokens = response.split()
        if not tokens:
            return metrics
        
        # 1. Token variance (уникальность токенов)
        unique_tokens = len(set(tokens))
        total_tokens = len(tokens)
        metrics.token_variance = unique_tokens / max(total_tokens, 1)
        
        # 2. Repeat rate (повторяемость)
        if len(tokens) > 1:
            repeats = 0
            for i in range(1, len(tokens)):
                if tokens[i] == tokens[i-1]:
                    repeats += 1
            metrics.repeat_rate = repeats / max(len(tokens) - 1, 1)
        
        # 3. Average token length
        metrics.avg_token_length = np.mean([len(t) for t in tokens]) if tokens else 0
        
        return metrics
    
    def should_trigger_fallback(self, metrics: GenerationMetrics) -> bool:
        """Определяет, нужно ли превентивно переключаться."""
        return (
            metrics.token_variance < self.THRESHOLD_TOKEN_VARIANCE or
            metrics.repeat_rate > self.THRESHOLD_REPEAT_RATE or
            metrics.latency > self.THRESHOLD_LATENCY
        )


class StatePreservingFallback:
    """
    Fallback с сохранением состояния между уровнями.
    Передаёт partial_context и intermediate_artifacts между уровнями fallback.
    """
    
    def __init__(self):
        self._state: Dict[str, Any] = {}
        self._artifacts: Dict[str, Any] = {}
        
    def set_state(self, key: str, value: Any):
        """Сохраняет состояние для передачи между уровнями."""
        self._state[key] = value
        
    def get_state(self, key: str, default: Any = None) -> Any:
        """Получает сохранённое состояние."""
        return self._state.get(key, default)
    
    def set_artifact(self, key: str, value: Any):
        """Сохраняет промежуточный артефакт (эмбеддинги, узлы, промпты)."""
        self._artifacts[key] = value
        
    def get_artifact(self, key: str, default: Any = None) -> Any:
        """Получает промежуточный артефакт."""
        return self._artifacts.get(key, default)
    
    def get_partial_context(self) -> Dict[str, Any]:
        """Возвращает partial_context для передачи между уровнями."""
        return {
            "embeddings": self._artifacts.get("embeddings"),
            "generated_nodes": self._artifacts.get("generated_nodes"),
            "prompts": self._artifacts.get("prompts"),
            "query": self._state.get("query"),
            "user_context": self._state.get("user_context"),
            "partial_response": self._state.get("partial_response")
        }
    
    def clear(self):
        """Очищает состояние после использования."""
        self._state.clear()
        self._artifacts.clear()
    
    def copy_to_next_level(self, next_fallback):
        """Копирует состояние на следующий уровень fallback."""
        if hasattr(next_fallback, 'set_state'):
            for key, value in self._state.items():
                next_fallback.set_state(key, value)
        if hasattr(next_fallback, 'set_artifact'):
            for key, value in self._artifacts.items():
                next_fallback.set_artifact(key, value)


class FallbackErrorMapper:
    """Сопоставляет тип ошибки с оптимальным fallback."""
    
    ERROR_FALLBACK_MAP = {
        "ethics_violation": "ethics_fallback",
        "model_crash": "memory_based_response",
        "timeout": "keyword_response_incomplete",
        "low_confidence": "enhanced_retry",
        "invalid_response": "basic_fallback",
        "connection_error": "offline_fallback"
    }
    
    @classmethod
    def get_fallback_type(cls, error_type: str) -> str:
        """Возвращает оптимальный тип fallback для ошибки."""
        return cls.ERROR_FALLBACK_MAP.get(error_type, "default_fallback")


def create_proactive_fallback() -> tuple:
    """Создаёт инстансы для прогнозной деградации."""
    monitor = ProactiveDegradationMonitor()
    state_preserving = StatePreservingFallback()
    return monitor, state_preserving
