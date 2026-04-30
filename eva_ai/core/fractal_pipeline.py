"""
FractalPipeline - Интеграция EVAGenerator с RecursiveModelPipeline

Заменяет Model A + Model B на единый EVAGenerator с:
- HybridTokenizer (виртуальные токены из графа)
- SemanticContextCache (семантический поиск)
- GGUFShadowProfiler (маршрутизация)
- Prompt templates (типы запросов)

Использует ту же GGUF модель что и RecursiveModelPipeline
"""

import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger("eva_ai.core.fractal_pipeline")

from eva_ai.memory.fractal_graph_v2 import FractalGraphV2
from eva_ai.memory.fractal_graph_v2.eva_generator import (
    EVAGenerator, 
    GenerationRequest, 
    GenerationResult
)
from eva_ai.memory.fractal_graph_v2.gguf_shadow import GGUFShadowProfiler


@dataclass
class PipelineResult:
    """Результат FractalPipeline."""
    response: str
    confidence: float
    quality_score: float
    reasoning_steps: List[Dict]
    virtual_tokens_used: List[str]
    processing_time: float
    query_type: str
    model_used: str = "evagenerator"


class FractalPipeline:
    """
    Упрощённый пайплайн на базе EVAGenerator.
    
    Преимущества над RecursiveModelPipeline:
    1. Один вызов модели вместо A + B
    2. Виртуальные токены для сущностей из графа
    3. Семантический поиск по необработанному контексту
    4. Адаптивные промты под тип запроса
    5. Интегрированный quality check
    """
    
    def __init__(
        self,
        fractal_graph: FractalGraphV2,
        gguf_model=None,  # Загруженная GGUF модель
        model_path: str = None,
        n_ctx: int = 2048,
        n_threads: int = None,  # По умолчанию: все ядра CPU
        max_semantic_contexts: int = 500,
        semantic_cache_dir: str = None,
        llama_a=None,
        llama_b=None,
        llama_c=None
    ):
        import os
        self.graph = fractal_graph
        self.gguf_model = gguf_model
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads or os.cpu_count() or 12  # Все ядра CPU
        self.llama_a = llama_a
        self.llama_b = llama_b
        self.llama_c = llama_c
        
        # Инициализируем GGUFShadowProfiler
        self.gguf_shadow = GGUFShadowProfiler(
            fractal_graph=fractal_graph,
            model_path=model_path
        )
        
        # Инициализируем EVAGenerator с llama моделями
        self.generator = EVAGenerator(
            fractal_graph=fractal_graph,
            gguf_shadow=self.gguf_shadow,
            max_semantic_contexts=max_semantic_contexts,
            llama_a=llama_a,
            llama_b=llama_b,
            n_ctx=n_ctx,
            n_threads=n_threads
        )
        
        logger.info(f"FractalPipeline инициализирован: llama_a={llama_a is not None}, llama_b={llama_b is not None}")
        
        logger.info("FractalPipeline инициализирован")
    
    def process_query(
        self,
        query: str,
        conversation_history: List[Dict] = None,
        max_tokens: int = 512,
        temperature: float = 0.5,
        session_id: str = None,
        query_type: str = None
    ) -> Dict[str, Any]:
        """
        Обработать запрос через EVAGenerator.
        
        Args:
            query: Текст запроса
            conversation_history: История разговора
            max_tokens: Максимум токенов
            temperature: Температура генерации
            session_id: ID сессии
            query_type: Тип запроса (None = авто)
            
        Returns:
            Dict с response, quality, reasoning_steps и т.д.
        """
        start_time = time.time()
        
        # Определяем тип запроса если не задан
        if query_type is None:
            query_type = self._detect_query_type(query)
        
        # Формируем запрос
        request = GenerationRequest(
            text=query,
            query_type=query_type,
            conversation_history=conversation_history or [],
            max_tokens=max_tokens,
            temperature=temperature,
            session_id=session_id
        )
        
        # Генерируем
        result = self.generator.generate(request)
        
        # Форматируем результат
        return {
            'response': result.response,
            'natural_response': result.response,
            'final_response': result.response,
            'confidence': result.confidence,
            'quality': {
                'score': result.quality_score,
                'is_gibberish': result.quality_score < 0.3,
                'reasons': []
            },
            'query_type': result.query_type,
            'virtual_tokens_used': result.virtual_tokens_used,
            'reasoning_steps': result.reasoning_steps,
            'processing_time': result.processing_time,
            'model_a_result': None,
            'model_b_result': {
                'natural_response': result.response,
                'quality': {'score': result.quality_score}
            },
            'has_code': False,
            'fractal_context': None
        }
    
    def _detect_query_type(self, query: str) -> str:
        """Определить тип запроса."""
        query_lower = query.lower()
        
        if any(kw in query_lower for kw in ['кратко', 'вкратце', 'суть', 'что такое']):
            return 'кратко'
        elif any(kw in query_lower for kw in ['подробно', 'детально', 'расскажи', 'объясни']):
            return 'подробно'
        elif any(kw in query_lower for kw in ['факт', 'правда', 'верно', 'подтверди']):
            return 'факт'
        elif any(kw in query_lower for kw in ['анализ', 'сравни', 'оцени']):
            return 'анализ'
        
        return 'default'
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Получить статистику контекста."""
        return {
            'graph_nodes': len(self.graph.nodes),
            'semantic_cache_size': len(self.generator.semantic_cache.contexts) if self.generator.semantic_cache else 0,
            'virtual_tokens': len(self.generator.tokenizer.node_to_virtual),
            'model_meta': self.gguf_shadow.model_meta if self.gguf_shadow else {}
        }


class FractalPipelineAdapter:
    """
    Адаптер для прозрачной замены RecursiveModelPipeline на FractalPipeline.
    
    Использование:
    ```python
    # Вместо:
    pipeline = RecursiveModelPipeline(model_a_path, model_b_path, ...)
    result = pipeline.process_query(query)
    
    # Теперь можно:
    pipeline = FractalPipelineAdapter(
        fractal_graph=graph,
        model_a=model_a,
        model_b=model_b,
        ...
    )
    result = pipeline.process_query(query)  # Совместимый API
    ```
    """
    
    def __init__(
        self,
        fractal_graph: FractalGraphV2,
        model_a,  # GGUF модель для генерации
        model_b = None,  # Может быть None если не используется
        **kwargs
    ):
        self.fractal_pipeline = FractalPipeline(
            fractal_graph=fractal_graph,
            gguf_model=model_a,
            **kwargs
        )
        
        # Сохраняем оригинальные модели для fallback
        self.model_a = model_a
        self.model_b = model_b
        
        # Режим работы
        self.use_fractal = True
        self.use_hybrid = False  # Гибридный режим с оригинальными моделями
    
    def process_query(
        self,
        query: str,
        max_iterations: int = 1,
        gen_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Совместимый с RecursiveModelPipeline API.
        """
        if self.use_fractal:
            return self.fractal_pipeline.process_query(
                query=query,
                conversation_history=gen_params.get('conversation_history') if gen_params else None,
                max_tokens=gen_params.get('model_a', {}).get('max_tokens', 512) if gen_params else 512,
                temperature=gen_params.get('model_a', {}).get('temperature', 0.5) if gen_params else 0.5
            )
        
        # Fallback к оригинальному поведению
        return self._fallback_process(query, gen_params)
    
    def _fallback_process(self, query: str, gen_params: Dict = None) -> Dict[str, Any]:
        """Fallback через оригинальные модели."""
        # Здесь можно вызвать оригинальный RecursiveModelPipeline
        # Для обратной совместимости
        raise NotImplementedError("Fallback to original models not implemented")
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Получить статистику."""
        return self.fractal_pipeline.get_context_stats()
    
    @property
    def fractal_memory(self):
        """Совместимость с fractal_memory атрибутом."""
        return self.fractal_pipeline.graph
