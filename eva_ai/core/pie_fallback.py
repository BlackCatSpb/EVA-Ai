"""
Pie Fallback Pipeline

Fallback механизм генерации когда GGUF pipeline недоступен.
Использует Pie Architecture для генерации через API или другие backend'ы.

Usage:
    from eva_ai.core.pie_fallback import PieFallbackPipeline
    
    fallback = PieFallbackPipeline(graph, config)
    response = fallback.generate("Hello!")
"""

import time
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

# Pie Architecture imports
from eva_ai.memory.pie_integration import (
    create_l1l2_graph,
    create_default_profiler,
    create_default_engine,
    RoutingParams
)

logger = logging.getLogger("eva_ai.core.pie_fallback")


@dataclass
class FallbackResult:
    """Результат fallback генерации."""
    text: str
    success: bool
    source: str  # 'pie', 'cache', 'keyword', 'error'
    confidence: float
    latency_ms: float
    metadata: Dict[str, Any]


class PieFallbackPipeline:
    """
    Fallback pipeline на базе Pie Architecture.
    
    Активируется когда:
    - GGUF модели не загружены
    - Произошла ошибка в основном pipeline
    - Запрошен режим 'fallback'
    
    Использует:
    - L1: ActivationProfiler для статистики
    - L2: RoutingEngine для выбора стратегии
    - L3: FractalGraphV2 для контекста
    - Keyword/Cache fallback если всё остальное недоступно
    """
    
    def __init__(
        self,
        fractal_graph,
        config: Optional[Dict] = None,
        enable_profiling: bool = True,
        enable_routing: bool = True
    ):
        """
        Инициализация fallback pipeline.
        
        Args:
            fractal_graph: FractalGraphV2 instance
            config: Конфигурация
            enable_profiling: Включить L1 профилирование
            enable_routing: Включить L2 маршрутизацию
        """
        self.fractal_graph = fractal_graph
        self.config = config or {}
        
        # Инициализация L1/L2
        self._init_pie_components(enable_profiling, enable_routing)
        
        # Fallback цепочка
        self.fallback_chain = self.config.get(
            'fallback_chain',
            ['keyword', 'cache', 'minimal']
        )
        
        # Статистика
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'avg_latency_ms': 0.0
        }
        
        logger.info("PieFallbackPipeline initialized")
    
    def _init_pie_components(self, enable_profiling: bool, enable_routing: bool):
        """Инициализация компонентов Pie Architecture."""
        # Получаем путь к БД от fractal_graph
        db_path = self._get_db_path()
        
        # Создаём L1/L2 граф
        self.l1l2_graph = create_l1l2_graph(db_path)
        
        # L1: Activation Profiler
        if enable_profiling:
            self.profiler = create_default_profiler(self.l1l2_graph)
            logger.info("L1 ActivationProfiler initialized")
        else:
            self.profiler = None
        
        # L2: Routing Engine
        if enable_routing:
            self.router = create_default_engine(
                self.l1l2_graph,
                self.profiler
            )
            # Создаём дефолтные правила
            self.router.create_default_rules()
            logger.info("L2 RoutingEngine initialized with default rules")
        else:
            self.router = None
    
    def _get_db_path(self) -> str:
        """Получить путь к SQLite БД."""
        # Пробуем получить от fractal_graph
        if hasattr(self.fractal_graph, 'db_path'):
            return self.fractal_graph.db_path
        
        if hasattr(self.fractal_graph, 'path'):
            return str(self.fractal_graph.path)
        
        # Fallback - используем временную директорию
        import tempfile
        return str(tempfile.gettempdir() / 'eva_pie_l1_l2.sqlite')
    
    def generate(
        self,
        query: str,
        context: Optional[str] = None,
        mode: str = "fallback",
        **kwargs
    ) -> FallbackResult:
        """
        Генерация ответа через fallback механизм.
        
        Args:
            query: Запрос пользователя
            context: Контекст (опционально)
            mode: Режим ('fallback', 'keyword', 'cache', 'minimal')
            **kwargs: Дополнительные параметры
            
        Returns:
            FallbackResult с ответом и метаданными
        """
        start_time = time.time()
        self.stats['total_calls'] += 1
        
        try:
            # Определяем стратегию через L2
            if self.router:
                routing_params = self.router.get_rule_for_generation(query, mode)
                strategy = self._determine_strategy(routing_params)
            else:
                strategy = self.fallback_chain[0]
            
            # Выполняем генерацию по цепочке fallback
            result = self._execute_fallback_chain(query, context, strategy)
            
            # Записываем статистику в L1
            if self.profiler and result.success:
                latency_ms = (time.time() - start_time) * 1000
                self.profiler.record_generation(
                    domain=kwargs.get('domain', 'fallback'),
                    model_id='fallback',
                    query=query,
                    response=result.text,
                    entropy=0.5,
                    latency_ms=latency_ms,
                    quality=result.confidence
                )
            
            # Обновляем статистику
            if result.success:
                self.stats['successful_calls'] += 1
            else:
                self.stats['failed_calls'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Fallback generation error: {e}")
            self.stats['failed_calls'] += 1
            
            return FallbackResult(
                text="Извините, я не могу сейчас ответить. Попробуйте позже.",
                success=False,
                source='error',
                confidence=0.0,
                latency_ms=(time.time() - start_time) * 1000,
                metadata={'error': str(e)}
            )
    
    def _determine_strategy(self, routing_params: Dict) -> str:
        """Определить стратегию fallback из параметров маршрутизации."""
        fallback_chain = routing_params.get('fallback_chain', self.fallback_chain)
        return fallback_chain[0] if fallback_chain else 'keyword'
    
    def _execute_fallback_chain(
        self,
        query: str,
        context: Optional[str],
        primary_strategy: str
    ) -> FallbackResult:
        """Выполнить цепочку fallback стратегий."""
        strategies = [primary_strategy] + [s for s in self.fallback_chain if s != primary_strategy]
        
        for strategy in strategies:
            try:
                if strategy == 'keyword':
                    result = self._keyword_fallback(query)
                elif strategy == 'cache':
                    result = self._cache_fallback(query)
                elif strategy == 'graph':
                    result = self._graph_fallback(query, context)
                elif strategy == 'minimal':
                    result = self._minimal_fallback(query)
                else:
                    continue
                
                if result.success:
                    return result
                    
            except Exception as e:
                logger.warning(f"Fallback strategy {strategy} failed: {e}")
                continue
        
        # Если все стратегии провалились
        return FallbackResult(
            text="Я не смог обработать ваш запрос. Пожалуйста, попробуйте переформулировать.",
            success=False,
            source='error',
            confidence=0.0,
            latency_ms=0.0,
            metadata={'attempted_strategies': strategies}
        )
    
    def _keyword_fallback(self, query: str) -> FallbackResult:
        """Простой fallback на основе ключевых слов."""
        start_time = time.time()
        
        # Базовые ответы на частые вопросы
        keyword_responses = {
            'привет': 'Привет! Я EVA, ваш AI ассистент. Чем могу помочь?',
            'hello': 'Hello! I am EVA, your AI assistant. How can I help you?',
            'как дела': 'У меня всё отлично, спасибо! Готов помочь вам.',
            'кто ты': 'Я EVA — интеллектуальный ассистент, созданный для помощи вам.',
            'что ты умеешь': 'Я могу отвечать на вопросы, помогать с анализом данных, обучением и многим другим.',
            'помощь': 'Я готов помочь! Задайте мне вопрос или опишите задачу.',
            'help': 'I\'m here to help! Ask me a question or describe your task.',
        }
        
        # Поиск по ключевым словам
        query_lower = query.lower().strip()
        
        for keyword, response in keyword_responses.items():
            if keyword in query_lower:
                return FallbackResult(
                    text=response,
                    success=True,
                    source='keyword',
                    confidence=0.7,
                    latency_ms=(time.time() - start_time) * 1000,
                    metadata={'matched_keyword': keyword}
                )
        
        return FallbackResult(
            text="",
            success=False,
            source='keyword',
            confidence=0.0,
            latency_ms=(time.time() - start_time) * 1000,
            metadata={'no_keyword_match': True}
        )
    
    def _cache_fallback(self, query: str) -> FallbackResult:
        """Fallback на основе кэшированных ответов."""
        start_time = time.time()
        
        # Пробуем найти похожий запрос в кэше графа
        if hasattr(self.fractal_graph, 'semantic_search'):
            try:
                # Простой поиск (без эмбеддинга для скорости)
                results = self.fractal_graph.search_nodes(content=query[:50])
                
                if results:
                    best_match = results[0]
                    return FallbackResult(
                        text=best_match.get('content', ''),
                        success=True,
                        source='cache',
                        confidence=0.6,
                        latency_ms=(time.time() - start_time) * 1000,
                        metadata={'cache_hit': True, 'node_id': best_match.get('id')}
                    )
            except Exception as e:
                logger.debug(f"Cache fallback error: {e}")
        
        return FallbackResult(
            text="",
            success=False,
            source='cache',
            confidence=0.0,
            latency_ms=(time.time() - start_time) * 1000,
            metadata={'cache_miss': True}
        )
    
    def _graph_fallback(self, query: str, context: Optional[str]) -> FallbackResult:
        """Fallback на основе графа знаний."""
        start_time = time.time()
        
        try:
            # Поиск релевантных концептов
            if hasattr(self.fractal_graph, 'search_nodes'):
                concepts = self.fractal_graph.search_nodes(
                    content=query,
                    node_type='concept',
                    limit=3
                )
                
                if concepts:
                    # Формируем ответ из концептов
                    concept_names = [c.get('content', '') for c in concepts]
                    response = f"Вот что я знаю по этой теме: {', '.join(concept_names)}."
                    
                    return FallbackResult(
                        text=response,
                        success=True,
                        source='graph',
                        confidence=0.5,
                        latency_ms=(time.time() - start_time) * 1000,
                        metadata={'concepts_found': len(concepts)}
                    )
        except Exception as e:
            logger.debug(f"Graph fallback error: {e}")
        
        return FallbackResult(
            text="",
            success=False,
            source='graph',
            confidence=0.0,
            latency_ms=(time.time() - start_time) * 1000,
            metadata={'no_concepts': True}
        )
    
    def _minimal_fallback(self, query: str) -> FallbackResult:
        """Минимальный fallback когда всё остальное недоступно."""
        return FallbackResult(
            text="Я получил ваш запрос. К сожалению, сейчас я работаю в ограниченном режиме и не могу дать полноценный ответ. Пожалуйста, попробуйте позже или обратитесь к администратору.",
            success=True,  # Возвращаем success=True чтобы не показывать ошибку
            source='minimal',
            confidence=0.3,
            latency_ms=0.0,
            metadata={'minimal_mode': True}
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику fallback pipeline."""
        total = self.stats['total_calls']
        success_rate = (
            self.stats['successful_calls'] / total * 100
            if total > 0 else 0
        )
        
        return {
            **self.stats,
            'success_rate_percent': success_rate,
            'l1_enabled': self.profiler is not None,
            'l2_enabled': self.router is not None,
            'fallback_chain': self.fallback_chain
        }
    
    def feedback(self, positive: bool, rule_id: Optional[str] = None):
        """Обратная связь для L2."""
        if self.router:
            return self.router.record_feedback(rule_id, positive)
        return False


# Глобальный экземпляр для быстрого доступа
_fallback_pipeline: Optional[PieFallbackPipeline] = None


def get_fallback_pipeline(
    fractal_graph=None,
    config=None,
    force_new=False
) -> PieFallbackPipeline:
    """
    Получить глобальный экземпляр fallback pipeline.
    
    Usage:
        fallback = get_fallback_pipeline(graph)
        result = fallback.generate("Hello!")
    """
    global _fallback_pipeline
    
    if _fallback_pipeline is None or force_new:
        if fractal_graph is None:
            raise ValueError("fractal_graph required for initialization")
        
        _fallback_pipeline = PieFallbackPipeline(
            fractal_graph=fractal_graph,
            config=config
        )
    
    return _fallback_pipeline


def clear_fallback_pipeline():
    """Очистить глобальный экземпляр."""
    global _fallback_pipeline
    _fallback_pipeline = None
