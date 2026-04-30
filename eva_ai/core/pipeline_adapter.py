"""
PipelineAdapter - Адаптер для совместимости UnifiedGenerator с существующим кодом

Позволяет использовать UnifiedGenerator (Pie-based) вместо TwoModelPipeline
без изменения всего остального кода.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("eva_ai.pipeline_adapter")


class PipelineAdapter:
    """
    Адаптер который делает UnifiedGenerator совместимым с интерфейсом TwoModelPipeline.
    
    Проксирует вызовы process_query и другие методы к UnifiedGenerator.
    """
    
    def __init__(self, unified_generator):
        """
        Args:
            unified_generator: Экземпляр UnifiedGenerator
        """
        self._generator = unified_generator
        self.ready = unified_generator is not None
        
        # Атрибуты для совместимости со старым TwoModelPipeline
        self.model_a = None  # LOGIC model
        self.model_b = None  # CONTEXT model
        
        # Инициализируем ссылки на модели если UnifiedGenerator загружен
        if unified_generator and hasattr(unified_generator, '_model_paths'):
            paths = unified_generator._model_paths
            from .unified_generator import ModelType
            if ModelType.LOGIC in paths:
                self.model_a = str(paths[ModelType.LOGIC])
            if ModelType.CONTEXT in paths:
                self.model_b = str(paths[ModelType.CONTEXT])
        
        logger.info("PipelineAdapter initialized")
    
    @property
    def openvino_models(self) -> Dict[str, Any]:
        """Получить доступ к OpenVINO генераторам."""
        if not self._generator:
            return {'cpu': None, 'gpu': None}
        
        ug = self._generator
        return {
            'cpu': getattr(ug, '_openvino_cpu', None),
            'gpu': getattr(ug, '_openvino_gpu', None),
            'generator': ug
        }
    
    @property
    def is_ready(self) -> bool:
        """Проверить готовность."""
        return self.ready and self._generator is not None
    
    def process_query(self, query: str, context: Optional[Dict] = None, gen_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Обработать запрос (совместимый интерфейс с TwoModelPipeline).
        
        Использует двухэтапную генерацию: LOGIC -> CONTEXT
        
        Args:
            query: Запрос пользователя
            context: Контекст (опционально)
            gen_params: Параметры генерации (опционально, для совместимости)
            
        Returns:
            Dict с response, status, source и т.д.
            Включает поля model_a_result/model_b_result для совместимости с SelfReasoningEngine.
        """
        if not self._generator:
            return {
                "response": "Ошибка: генератор не инициализирован",
                "status": "error",
                "source": "pipeline_adapter"
            }
        
        try:
            # Определяем параметры генерации
            user_context = context or {}
            params = gen_params or {}
            max_tokens = params.get('max_new_tokens', 4096)  # Минимум 4096
            temperature = params.get('temperature', 0.7)
            
            # Используем итеративную генерацию с проверкой противоречий
            # task_type='query' для CRITICAL приоритета в ModelAccessManager
            result = self._generator.generate_iterative(
                query=query,
                context=None,
                max_tokens_logic=4096,
                max_tokens_context=4096,
                temperature=temperature,
                check_contradictions=True,
                check_concepts=True,
                task_type='query'
            )
            
            # Формируем ответ в формате TwoModelPipeline
            response_text = result.text if result else "Нет ответа"
            
            return {
                "response": response_text,
                "status": "ok",
                "source": "pipeline_adapter",
                "model_used": result.model_used if result else "none",
                "tokens": result.tokens_generated if result else 0,
                "time": result.generation_time if result else 0,
                "confidence": result.confidence if result else 0,
                # Для совместимости с SRE
                "model_a_result": {
                    "natural_response": "logic_completed",
                    "confidence": 0.8,
                    "tokens": result.tokens_generated // 2 if result else 0,
                    "model": "logic"
                },
                "model_b_result": {
                    "natural_response": response_text[:300],
                    "confidence": result.confidence if result else 0,
                    "tokens": result.tokens_generated // 2 if result else 0,
                    "model": "context"
                },
                "reasoning_steps": [
                    {
                        "step": 1,
                        "phase": "logic_primary",
                        "thought": "LOGIC: краткий ответ сгенерирован",
                        "confidence": 0.8,
                        "model": "logic"
                    },
                    {
                        "step": 2,
                        "phase": "context_expansion",
                        "thought": "CONTEXT: расширение с концептами и противоречиями",
                        "confidence": result.confidence if result else 0,
                        "model": "context",
                        "input": query
                    }
                ]
            }
            
        except Exception as e:
            logger.error(f"PipelineAdapter error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "response": f"Ошибка обработки: {e}",
                "status": "error",
                "source": "pipeline_adapter"
            }
    
    def generate(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.7) -> str:
        """Прямая генерация (для совместимости)."""
        if not self._generator:
            return "Ошибка: генератор не инициализирован"
        
        result = self._generator.generate(
            query=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return result.text
    
    def generate_streaming(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7, chunk_size: int = 30, task_type: str = "default"):
        """
        Генерация со стримингом чанков.
        
        Args:
            prompt: Запрос
            max_tokens: Максимум токенов
            temperature: Температура
            chunk_size: Размер чанка для выдачи (30 символов оптимально)
            task_type: Тип задачи для роутинга (context, logic, coder, self_dialog)
            
        Yields:
            Dict с 'type', 'text', 'is_final', 'tokens_count', 'elapsed_ms'
        """
        if not self._generator:
            yield {
                'type': 'error',
                'text': 'Генератор не инициализирован',
                'is_final': True,
                'tokens_count': 0,
                'elapsed_ms': 0
            }
            return
        
        yield from self._generator.generate_streaming(
            query=prompt,
            context=None,
            max_tokens=max_tokens,
            temperature=temperature,
            chunk_size=chunk_size,
            task_type=task_type
        )
    
    def generate_with_context(self, query: str, context: str = "", max_tokens: int = 4096, temperature: float = 0.7):
        """
        Генерация с контекстом и чанкованием.
        
        Args:
            query: Запрос
            context: Контекст из FractalGraph/HybridCache
            max_tokens: Максимум токенов
            temperature: Температура
            
        Yields:
            Dict с 'type', 'text', 'is_final', 'tokens_count', 'elapsed_ms'
        """
        if not self._generator:
            yield {
                'type': 'error',
                'text': 'Генератор не инициализирован',
                'is_final': True,
                'tokens_count': 0,
                'elapsed_ms': 0
            }
            return
        
        # Используем chunked контекст для больших запросов
        if len(context) > 1000:
            result = self._generator.generate_with_chunked_context(
                query=query,
                context=context,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            yield {
                'type': 'complete',
                'text': result.text,
                'is_final': True,
                'tokens_count': result.tokens_generated,
                'elapsed_ms': int(result.generation_time * 1000)
            }
        else:
            # Малый контекст - обычный стриминг
            yield from self._generator.generate_streaming(
                query=query,
                context=context,
                max_tokens=max_tokens,
                temperature=temperature
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику."""
        if not self._generator:
            return {"error": "not_initialized"}
        return self._generator.get_stats()
    
    def unload_models(self):
        """Выгрузить модели (для совместимости)."""
        if self._generator:
            self._generator.unload_all()
            self.ready = False
    
    def load_models(self):
        """Загрузить модели (для совместимости)."""
        # Модели загружаются лениво при первом использовании
        self.ready = self._generator is not None
    
    @property
    def is_ready(self) -> bool:
        """Проверить готовность."""
        return self.ready and self._generator is not None


def create_pipeline_adapter(
    logic_model_path=None,
    context_model_path=None,
    n_ctx=32768,
    n_threads=None,  # None = испольовать все ядра (12 для i5-12450H)
    fractal_graph=None,
    brain=None,
    use_openvino=False,
    cpu_device="CPU",
    gpu_device="GPU.0"
) -> Optional[PipelineAdapter]:
    """
    Создать PipelineAdapter с UnifiedGenerator.
    
    Использует две модели (обе на CPU):
    - LOGIC (RuadaptQwen3-4B): для логики, рассуждений
    - CONTEXT (RuadaptQwen3-4B): для длинных контекстов
    
    GPU для эмбеддингов.
    """
    try:
        from pathlib import Path
        from .unified_generator import UnifiedGenerator
        
        logic_path = Path(logic_model_path) if logic_model_path else None
        context_path = Path(context_model_path) if context_model_path else None
        
        event_bus = getattr(brain, 'event_bus', None) or getattr(brain, '_new_event_bus', None)
        
        generator = UnifiedGenerator(
            logic_model_path=logic_path,
            context_model_path=context_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            fractal_graph=fractal_graph,
            brain=brain,
            use_openvino=use_openvino,
            cpu_device=cpu_device,
            gpu_device=gpu_device,
            event_bus=event_bus
        )
        
        return PipelineAdapter(generator)
        
    except Exception as e:
        logger.error(f"Failed to create PipelineAdapter: {e}")
        import traceback
        traceback.print_exc()
        return None
