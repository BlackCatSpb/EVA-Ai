"""
DualGenerator with Pie Architecture Support

Модифицированный DualGenerator с интеграцией Pie Architecture.
Может работать в режиме fallback когда GGUF pipeline недоступен.

Usage:
    from eva_ai.memory.fractal_graph_v2.dual_generator_pie import DualGeneratorPie
    
    generator = DualGeneratorPie(
        fractal_graph=graph,
        use_pie=True,  # Включить Pie Architecture
        enable_fallback=True  # Использовать fallback при ошибках
    )
"""

import time
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

# Импорты текущей реализации
from .dual_generator import DualGenerator, CondensedGenerator, ExtendedGenerator

# Импорты Pie Architecture
from ..pie_integration import (
    PieIntegration,
    create_pie_integration,
    GenerationMetadata
)
from ...core.pie_fallback import PieFallbackPipeline

logger = logging.getLogger("eva_ai.fractal_graph_v2.dual_generator_pie")


@dataclass
class GenerationResult:
    """Результат генерации."""
    text: str
    success: bool
    mode: str
    source: str  # 'model_a', 'model_b', 'fallback', 'pie'
    latency_ms: float
    tokens: int
    routing_params: Optional[Dict] = None
    metadata: Optional[Dict] = None


class DualGeneratorPie:
    """
    DualGenerator с поддержкой Pie Architecture.
    
    Features:
    - L2: Адаптивная маршрутизация через RoutingEngine
    - L1: Профилирование через ActivationProfiler
    - Fallback: Автоматический fallback при ошибках
    - Backward compatibility: Полная совместимость с DualGenerator
    """
    
    def __init__(
        self,
        fractal_graph,
        model_a=None,
        model_b=None,
        use_pie: bool = False,
        enable_fallback: bool = True,
        pie_config: Optional[Dict] = None,
        **kwargs
    ):
        """
        Инициализация DualGenerator с Pie поддержкой.
        
        Args:
            fractal_graph: FractalGraphV2 instance
            model_a: Модель A (llama.cpp) или None
            model_b: Модель B (llama.cpp) или None
            use_pie: Включить Pie Architecture
            enable_fallback: Включить fallback механизм
            pie_config: Конфигурация Pie
            **kwargs: Дополнительные параметры для DualGenerator
        """
        self.fractal_graph = fractal_graph
        self.use_pie = use_pie
        self.enable_fallback = enable_fallback
        
        # Оригинальный DualGenerator (если есть модели)
        self.dual_generator = None
        if model_a is not None:
            try:
                self.dual_generator = DualGenerator(
                    fractal_graph=fractal_graph,
                    model_a=model_a,
                    model_b=model_b,
                    **kwargs
                )
                logger.info("Original DualGenerator initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize DualGenerator: {e}")
        
        # Pie Architecture
        self.pie = None
        if use_pie:
            try:
                self.pie = create_pie_integration(
                    fractal_graph=fractal_graph,
                    config=pie_config,
                    enable_l1=True,
                    enable_l2=True
                )
                logger.info("Pie Architecture initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Pie: {e}")
        
        # Fallback pipeline
        self.fallback = None
        if enable_fallback:
            try:
                self.fallback = PieFallbackPipeline(
                    fractal_graph=fractal_graph,
                    config=pie_config
                )
                logger.info("Fallback pipeline initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize fallback: {e}")
        
        # Статистика
        self.stats = {
            'total_calls': 0,
            'pie_calls': 0,
            'fallback_calls': 0,
            'error_calls': 0,
            'avg_latency_ms': 0.0
        }
    
    def generate(
        self,
        query: str,
        mode: str = "auto",
        use_pie_routing: bool = True,
        **kwargs
    ) -> GenerationResult:
        """
        Генерация с поддержкой Pie Architecture.
        
        Args:
            query: Запрос пользователя
            mode: Режим ('auto', 'condensed', 'extended', 'pie')
            use_pie_routing: Использовать L2 маршрутизацию
            **kwargs: Дополнительные параметры
            
        Returns:
            GenerationResult с ответом и метаданными
        """
        start_time = time.time()
        self.stats['total_calls'] += 1
        
        try:
            # Режим 'pie' - принудительно использовать Pie
            if mode == 'pie':
                return self._generate_pie(query, start_time)
            
            # Получаем параметры от L2 (если включено)
            routing_params = None
            if self.use_pie and use_pie_routing and self.pie:
                routing_params = self.pie.get_generation_params(query, mode)
                # Обновляем kwargs параметрами от L2
                kwargs.update({
                    'temperature': routing_params.get('temperature', kwargs.get('temperature', 0.7)),
                    'repeat_penalty': routing_params.get('repeat_penalty', kwargs.get('repeat_penalty', 1.1)),
                    'max_tokens': routing_params.get('max_tokens', kwargs.get('max_tokens', 512))
                })
            
            # Пробуем оригинальный DualGenerator
            if self.dual_generator:
                try:
                    result = self._generate_with_dual(query, mode, **kwargs)
                    
                    # Записываем в L1
                    if self.pie:
                        self._record_to_l1(query, result, routing_params, start_time)
                    
                    return result
                    
                except Exception as e:
                    logger.warning(f"DualGenerator failed: {e}")
                    if not self.enable_fallback:
                        raise
            
            # Fallback на Pie
            if self.enable_fallback and self.fallback:
                return self._generate_fallback(query, start_time)
            
            # Если ничего не работает
            return GenerationResult(
                text="Система временно недоступна. Пожалуйста, попробуйте позже.",
                success=False,
                mode=mode,
                source='error',
                latency_ms=(time.time() - start_time) * 1000,
                tokens=0,
                routing_params=routing_params,
                metadata={'error': 'No generation method available'}
            )
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            self.stats['error_calls'] += 1
            
            return GenerationResult(
                text="Произошла ошибка при генерации. Попробуйте ещё раз.",
                success=False,
                mode=mode,
                source='error',
                latency_ms=(time.time() - start_time) * 1000,
                tokens=0,
                metadata={'error': str(e)}
            )
    
    def _generate_with_dual(
        self,
        query: str,
        mode: str,
        **kwargs
    ) -> GenerationResult:
        """Генерация через оригинальный DualGenerator."""
        start_time = time.time()
        
        # Вызываем оригинальный generate
        response = self.dual_generator.generate(query, mode=mode, **kwargs)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return GenerationResult(
            text=response,
            success=True,
            mode=mode,
            source='model_a' if mode == 'condensed' else 'model_b',
            latency_ms=latency_ms,
            tokens=len(response.split()),
            metadata={'generator': 'dual'}
        )
    
    def _generate_pie(self, query: str, start_time: float) -> GenerationResult:
        """Генерация через Pie Architecture (без моделей)."""
        self.stats['pie_calls'] += 1
        
        # Используем fallback pipeline
        if self.fallback:
            result = self.fallback.generate(query)
            
            return GenerationResult(
                text=result.text,
                success=result.success,
                mode='pie',
                source=result.source,
                latency_ms=result.latency_ms,
                tokens=len(result.text.split()) if result.text else 0,
                metadata=result.metadata
            )
        
        return GenerationResult(
            text="Pie mode not available",
            success=False,
            mode='pie',
            source='error',
            latency_ms=(time.time() - start_time) * 1000,
            tokens=0
        )
    
    def _generate_fallback(
        self,
        query: str,
        start_time: float
    ) -> GenerationResult:
        """Генерация через fallback."""
        self.stats['fallback_calls'] += 1
        
        result = self.fallback.generate(query)
        
        return GenerationResult(
            text=result.text,
            success=result.success,
            mode='fallback',
            source=result.source,
            latency_ms=result.latency_ms,
            tokens=len(result.text.split()) if result.text else 0,
            metadata=result.metadata
        )
    
    def _record_to_l1(
        self,
        query: str,
        result: GenerationResult,
        routing_params: Optional[Dict],
        start_time: float
    ):
        """Записать генерацию в L1."""
        if not self.pie or not self.pie.profiler:
            return
        
        try:
            metadata = GenerationMetadata(
                domain=routing_params.get('domain', 'general') if routing_params else 'general',
                model_id=result.source,
                entropy=0.5,
                latency_ms=result.latency_ms,
                quality=0.8 if result.success else 0.3
            )
            
            self.pie.record_generation(query, result.text, metadata)
            
        except Exception as e:
            logger.debug(f"Failed to record to L1: {e}")
    
    def feedback(self, positive: bool, rule_id: Optional[str] = None):
        """Обратная связь для L2."""
        if self.pie:
            return self.pie.record_feedback(positive, rule_id)
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику."""
        total = self.stats['total_calls']
        
        return {
            **self.stats,
            'pie_enabled': self.use_pie,
            'fallback_enabled': self.enable_fallback,
            'dual_generator_available': self.dual_generator is not None,
            'pie_integration_available': self.pie is not None,
            'fallback_available': self.fallback is not None,
            'pie_stats': self.pie.get_stats() if self.pie else None,
            'fallback_stats': self.fallback.get_stats() if self.fallback else None,
        }
    
    def get_routing_params(self, query: str, mode: str = "auto") -> Optional[Dict]:
        """Получить параметры маршрутизации от L2."""
        if self.pie:
            return self.pie.get_generation_params(query, mode)
        return None


# Для обратной совместимости
class PieEnabledDualGenerator(DualGeneratorPie):
    """Alias for DualGeneratorPie."""
    pass
