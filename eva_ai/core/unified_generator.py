"""
UnifiedGenerator - Единая система генерации на базе Pie Architecture

Использует:
- RuadaptQwen3-4B для общих задач
- Qwen Coder 1.5B для кода  
- L2 Роутинг для выбора модели
- Интеграция с FractalGraph V2
- ChunkedContextProcessor для обработки больших контекстов
- ModelAccessManager для координации доступа к модели
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Generator
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

from .context_chunking import ChunkedContextProcessor, StreamingGenerator, ContextChunk
from .model_access_manager import ModelAccessManager, AccessPriority

logger = logging.getLogger("eva_ai.unified_generator")


class ModelType(Enum):
    """Типы моделей для генерации.
    
    Три модели:
    - LOGIC (RuadaptQwen3-4B condensed): для логики, рассуждений
    - CONTEXT (RuadaptQwen3-4B extended): для длинных контекстов
    - CODER (Qwen Coder 1.5B): для кода
    """
    LOGIC = "logic"      # RuadaptQwen3-4B condensed (4096 ctx)
    CONTEXT = "context"  # RuadaptQwen3-4B extended (32768 ctx)
    CODER = "coder"      # Qwen Coder 1.5B


@dataclass
class GenerationResult:
    """Результат генерации."""
    text: str
    model_used: str
    generation_time: float
    tokens_generated: int
    confidence: float = 0.8
    metadata: Optional[Dict] = None


class SimpleRouter:
    """L2 Роутер для выбора между LOGIC, CONTEXT и CODER моделями.
    
    CODER - для кода и программирования
    CONTEXT - для длинных контекстов и развернутых ответов
    LOGIC (default) - для логики и рассуждений
    """
    
    # Ключевые слова для CODER модели
    CODER_KEYWORDS = [
        'код', 'code', 'python', 'javascript', 'js', 'typescript', 'ts',
        'function', 'функция', 'class', 'класс', 'api', 'rest',
        'html', 'css', 'sql', 'query', 'debug', 'debugging',
        'error', 'ошибка', 'exception', 'syntax', 'синтаксис',
        'алгоритм', 'algorithm', 'регулярное выражение', 'regex',
        'библиотека', 'library', 'модуль', 'module', 'pip', 'npm',
        'программа', 'program', 'скрипт', 'script', 'git', 'github'
    ]
    
    # Ключевые слова для CONTEXT модели
    CONTEXT_KEYWORDS = [
        'подробно', 'детально', 'развернуто', 'подробный ответ',
        'объясни подробно', 'расскажи детально', 'в деталях',
        'пошагово', 'пошаговое', 'пошаговая инструкция',
        'длинный', 'развернутый', 'полный ответ',
        'контекст', 'context', 'история', 'history',
        'предыдущее', 'previous', 'выше', 'above',
        'суммаризируй', 'summarize', 'итог', 'summary',
        'анализируй', 'analyze', 'сравни', 'compare',
        'документация', 'documentation', 'спецификация',
        # Общие вопросительные слова - для использования CONTEXT модели
        'что такое', 'кто такой', 'как работает', 'почему происходит',
        'объясни', 'расскажи', 'опиши', 'характеристики',
        'преимущества', 'недостатки', 'разница между', 'отличие',
        'что это', 'кто это', 'для чего', 'зачем нужно'
    ]

    # Минимальная длина для CONTEXT модели
    MIN_LENGTH_FOR_CONTEXT = 25
    
    def route(self, query: str) -> ModelType:
        """Определить тип модели для запроса."""
        query_lower = query.lower()
        
        # Проверяем код (приоритет)
        coder_score = sum(1 for kw in self.CODER_KEYWORDS if kw in query_lower)
        if coder_score >= 2:
            return ModelType.CODER
        
        # Проверяем длинный контекст или вопросительные слова
        context_score = sum(1 for kw in self.CONTEXT_KEYWORDS if kw in query_lower)
        
        # Если есть ключевые слова контекста ИЛИ запрос достаточно длинный
        if context_score >= 1 or len(query) >= self.MIN_LENGTH_FOR_CONTEXT:
            return ModelType.CONTEXT
        
        # По умолчанию - LOGIC
        return ModelType.LOGIC


class UnifiedGenerator:
    """
    Единый генератор на базе GGUF моделей.
    
    Архитектура:
    - CPU: Logic/Context модель (основная генерация)
    - GPU.0: Coder модель (код) + Self-dialog
    
    Использует OpenVINO GenAI для эффективной генерации.
    """
    
    def __init__(
        self,
        logic_model_path: Optional[Path] = None,
        context_model_path: Optional[Path] = None,
        coder_model_path: Optional[Path] = None,
        n_ctx: int = 16384,
        n_threads: int = 4,
        fractal_graph=None,
        brain=None,
        use_openvino: bool = False,
        cpu_device: str = "CPU",
        gpu_device: str = "GPU.0",
        event_bus=None
    ):
        """
        Инициализация Unified Generator.
        
        Args:
            logic_model_path: Путь к LOGIC модели (CPU)
            context_model_path: Путь к CONTEXT модели (CPU)
            coder_model_path: Путь к CODER модели (GPU)
            n_ctx: Размер контекста
            n_threads: Количество потоков
            fractal_graph: FractalGraph V2 для контекста
            brain: Ссылка на CoreBrain
            use_openvino: Использовать OpenVINO вместо llama.cpp
            cpu_device: Устройство для Logic/Context
            gpu_device: Устройство для Coder/Self-dialog
            event_bus: EventBus для интеграции с ModelAccessManager
        """
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.fractal_graph = fractal_graph
        self.brain = brain
        self.router = SimpleRouter()
        self.use_openvino = use_openvino
        self.cpu_device = cpu_device
        self.gpu_device = gpu_device
        self.event_bus = event_bus
        
        self.models: Dict[ModelType, Any] = {}
        self._model_paths: Dict[ModelType, Path] = {}
        self._openvino_cpu: Optional['OpenVINOGenerator'] = None
        self._openvino_gpu: Optional['OpenVINOGenerator'] = None
        self._openvino_coder: Optional['OpenVINOGenerator'] = None
        
        self._model_access: Optional[ModelAccessManager] = None
        self._init_model_access_manager()
        
        self._load_model_paths(logic_model_path, context_model_path, coder_model_path)
        
        if use_openvino:
            self._init_openvino_devices()
        
        logger.info(f"UnifiedGenerator initialized")
        logger.info(f"  use_openvino={use_openvino}")
        logger.info(f"  CPU device: {cpu_device} (Logic/Context)")
        logger.info(f"  GPU device: {gpu_device} (Coder/Self-dialog)")
        logger.info(f"  LOGIC model: {self._model_paths.get(ModelType.LOGIC)}")
        logger.info(f"  CONTEXT model: {self._model_paths.get(ModelType.CONTEXT)}")
        logger.info(f"  CODER model: {self._model_paths.get(ModelType.CODER)}")
        logger.info(f"  ModelAccessManager: {'enabled' if self._model_access else 'disabled'}")
    
    def _init_model_access_manager(self):
        """Инициализировать ModelAccessManager для координации доступа."""
        if self._model_access is not None:
            return
        
        try:
            self._model_access = ModelAccessManager(
                event_bus=self.event_bus,
                max_workers=4
            )
            self._model_access.start()
            logger.info("ModelAccessManager initialized")
        except Exception as e:
            logger.warning(f"Failed to init ModelAccessManager: {e}")
            self._model_access = None
    
    def _get_priority_for_task(self, task_type: str) -> AccessPriority:
        """Определить приоритет доступа по типу задачи."""
        priority_map = {
            'query': AccessPriority.CRITICAL,
            'self_dialog': AccessPriority.HIGH,
            'concept_mining': AccessPriority.HIGH,
            'contradiction_mining': AccessPriority.HIGH,
            'coder': AccessPriority.HIGH,
            'default': AccessPriority.NORMAL
        }
        return priority_map.get(task_type, AccessPriority.NORMAL)
    
    def _load_model_paths(self, logic_path: Optional[Path], context_path: Optional[Path], coder_path: Optional[Path] = None):
        """Загрузить пути к трем моделям: LOGIC, CONTEXT и CODER."""
        try:
            from eva_ai.core.pie_model_paths import get_pie_model_path
            
            # LOGIC и CONTEXT - одна модель ruadapt_qwen3_4b
            if logic_path is None:
                logic_path = get_pie_model_path('ruadapt_qwen3_4b', 'condensed')
            if context_path is None:
                # CONTEXT использует тот же файл, но с другим n_ctx
                context_path = get_pie_model_path('ruadapt_qwen3_4b', 'condensed')
            
            # CODER - отдельная модель
            if coder_path is None:
                coder_path = get_pie_model_path('qwen_coder_1_5b', 'condensed')
                
        except Exception as e:
            logger.warning(f"Could not load model paths from config: {e}")
            # Default paths
            base = Path(r"C:\Users\black\OneDrive\Desktop\CogniFlex\eva_pie_architecture\models\gguf_models")
            if logic_path is None:
                logic_path = base / "ruadapt_qwen3_4b_q4_k_m.gguf"
            if context_path is None:
                context_path = base / "ruadapt_qwen3_4b_q4_k_m.gguf"
            if coder_path is None:
                coder_path = base / "qwen2.5-coder-1.5b-instruct" / "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
        
        if logic_path and logic_path.exists():
            self._model_paths[ModelType.LOGIC] = logic_path
        
        if context_path and context_path.exists():
            self._model_paths[ModelType.CONTEXT] = context_path
            
        if coder_path and coder_path.exists():
            self._model_paths[ModelType.CODER] = coder_path
    
    def _init_openvino(self, device: str = "CPU") -> bool:
        """Инициализировать OpenVINO Generator на указанном устройстве."""
        try:
            from eva_ai.core.openvino_generator import OpenVINOGenerator
            
            model_path = self._model_paths.get(ModelType.CONTEXT) or self._model_paths.get(ModelType.LOGIC)
            if not model_path:
                logger.error("No model path for OpenVINO")
                return False
            
            generator = OpenVINOGenerator(
                model_path=model_path,
                device=device
            )
            
            logger.info(f"OpenVINO initialized on {device}")
            return generator
            
        except Exception as e:
            logger.error(f"Failed to init OpenVINO: {e}")
            return None
    
    def _init_openvino_devices(self) -> bool:
        """
        Инициализировать OpenVINO генераторы на CPU и GPU.
        
        CPU: Logic/Context модель (основная генерация)
        GPU: Coder модель (код + self-dialog)
        """
        try:
            import os
            from eva_ai.core.openvino_generator import OpenVINOGenerator
            
            cpu_count = os.cpu_count() or 4
            logger.info(f"CPU cores detected: {cpu_count}")
            
            logic_model = self._model_paths.get(ModelType.LOGIC)
            coder_model = self._model_paths.get(ModelType.CODER)
            
            cpu_scheduler = {
                'cache_size': min(4, cpu_count // 2),
                'max_num_seqs': min(8, cpu_count),
                'max_num_batched_tokens': cpu_count * 512,
                'enable_prefix_caching': True
            }
            
            gpu_scheduler = {
                'cache_size': 2,
                'max_num_seqs': 4,
                'max_num_batched_tokens': 2048,
                'enable_prefix_caching': True
            }
            
            if logic_model:
                self._openvino_cpu = OpenVINOGenerator(
                    model_path=logic_model,
                    device=self.cpu_device,
                    performance_hint="LATENCY",
                    scheduler_config=cpu_scheduler,
                    num_streams="AUTO"
                )
                logger.info(f"CPU OpenVINO ready: {self.cpu_device} (Logic/Context)")
            
            if coder_model:
                self._openvino_coder = OpenVINOGenerator(
                    model_path=coder_model,
                    device=self.gpu_device,
                    performance_hint="THROUGHPUT",
                    scheduler_config=gpu_scheduler
                )
                logger.info(f"GPU OpenVINO ready: {self.gpu_device} (Coder/Self-dialog)")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to init OpenVINO devices: {e}")
            return False
    
    def _route_device(self, query: str, task_type: str = "default") -> str:
        """
        Определить устройство и тип генератора для задачи.
        
        GPU → Coder: task_type=coder, task_type=self_dialog
        CPU → Logic/Context: всё остальное
        """
        gpu_tasks = {'coder', 'self_dialog'}
        
        if task_type in gpu_tasks:
            return ('gpu', task_type)
        
        return ('cpu', 'logic')
    
    def _get_generator_for_task(self, task_type: str):
        """
        Получить генератор для типа задачи.
        
        Args:
            task_type: coder, self_dialog, или default
            
        Returns:
            Tuple (generator, model_type)
        """
        if task_type in ('coder', 'self_dialog') and self._openvino_coder:
            return (self._openvino_coder, ModelType.CODER)
        
        if self._openvino_cpu:
            return (self._openvino_cpu, ModelType.LOGIC)
        
        if self._openvino_coder:
            return (self._openvino_coder, ModelType.CODER)
        
        return (None, None)
    
    def _detect_optimal_threads(self) -> int:
        """Автоопределение оптимального количества потоков для CPU."""
        import os
        cpu_count = os.cpu_count() or 4
        # Оставляем ядра для системы и embedder
        return max(4, min(8, cpu_count - 4))
    
    def _load_model(self, model_type: ModelType) -> bool:
        """Загрузить модель если нужно."""
        if model_type in self.models and self.models[model_type] is not None:
            return True
        
        if model_type not in self._model_paths:
            logger.error(f"No path configured for {model_type.value}")
            return False
        
        try:
            from llama_cpp import Llama
            
            path = self._model_paths[model_type]
            logger.info(f"Loading {model_type.value} from {path}")
            
            # Автоопределение оптимальных параметров CPU
            n_threads = self._detect_optimal_threads()
            
            # Оптимальный контекст с учётом памяти для embedder
            if model_type == ModelType.CONTEXT:
                n_ctx = 16384  # Extended context
            elif model_type == ModelType.LOGIC:
                n_ctx = 16384  # Оптимальный для RuadaptQwen3-4B
            else:
                n_ctx = 16384  # CODER - оставляем память для embedder
            
            start = time.time()
            model = Llama(
                model_path=str(path),
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=0,  # CPU-only - GPU для embedder
                use_mlock=True,   # Заблокировать модель в RAM
                use_mmap=True,    # Memory-mapped файлы
                n_batch=256,      # Batch для промпта
                nUMA=False,       # Отключено для простоты
                verbose=False
            )
            load_time = time.time() - start
            
            self.models[model_type] = model
            logger.info(f"{model_type.value} loaded in {load_time:.1f}s (n_ctx={n_ctx}, n_threads={n_threads})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load {model_type.value}: {e}")
            return False
    
    def generate(
        self,
        query: str,
        context: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        task_type: str = "default"
    ) -> GenerationResult:
        """
        Генерация ответа.
        
        Args:
            query: Запрос пользователя
            context: Дополнительный контекст из FractalGraph
            max_tokens: Максимум токенов
            temperature: Температура
            system_prompt: Системный промпт
            task_type: Тип задачи для приоритизации
            
        Returns:
            GenerationResult
        """
        start_time = time.time()
        logger.info(f"[GENERATE] Начало генерации для: {query[:30]}...")
        
        # Use ModelAccessManager if available
        if self._model_access is not None:
            priority = self._get_priority_for_task(task_type)
            request_id = self._model_access.request_access(
                priority=priority,
                task_type=task_type,
                callback=self._do_generate,
                query=query,
                context=context,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
                timeout=60.0
            )
            
            try:
                return self._model_access.get_result(request_id, timeout=60.0)
            except Exception as e:
                logger.error(f"ModelAccess error: {e}")
        
        # Direct generation
        return self._do_generate(query, context, max_tokens, temperature, system_prompt)
    
    def _do_generate(
        self,
        query: str,
        context: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> GenerationResult:
        """Выполняет генерацию (вызывается через ModelAccessManager)."""
        start_time = time.time()
        logger.info(f"[GENERATE] Начало генерации для: {query[:30]}...")
        
        # Определяем модель
        model_type = self.router.route(query)
        logger.info(f"[GENERATE] Выбрана модель: {model_type}, длина запроса: {len(query)}")
        
        # Загружаем модель
        if not self._load_model(model_type):
            # Fallback на другую модель
            fallback_type = ModelType.CONTEXT if model_type == ModelType.LOGIC else ModelType.LOGIC
            if not self._load_model(fallback_type):
                return GenerationResult(
                    text="Ошибка: модели недоступны",
                    model_used="none",
                    generation_time=0.0,
                    tokens_generated=0,
                    confidence=0.0
                )
            model_type = fallback_type
        
        model = self.models[model_type]
        
        # Асинхронно предзагружаем контекст
        self._prefetch_context_async(query)
        
        # Формируем промпт с контекстом из FractalGraph и HybridCache
        full_context = self._build_context(query, context)
        prompt = self._format_prompt(query, full_context, system_prompt, model_type)
        
        # Генерируем
        try:
            output = model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["<|im_end|>", "<|im_start|>", "<|endoftext|>"],
                echo=False,
                repeat_penalty=1.1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                mirostat_mode=0,
                logprobs=None
            )
            
            text = output['choices'][0]['text']
            tokens = output['usage']['completion_tokens']
            
            # Очищаем ответ от токенов-маркеров
            text = text.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
            text = text.replace("<|endoftext|>", "").strip()
            
            # Сохраняем в FractalGraph
            self._save_to_graph(query, text)
            
            return GenerationResult(
                text=text,
                model_used=model_type.value,
                generation_time=time.time() - start_time,
                tokens_generated=tokens,
                confidence=0.9 if model_type == ModelType.CODER else (0.85 if model_type == ModelType.CONTEXT else 0.8)
            )
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return GenerationResult(
                text=f"Ошибка генерации: {e}",
                model_used=model_type.value,
                generation_time=time.time() - start_time,
                tokens_generated=0,
                confidence=0.0
            )
    
    def generate_dual(
        self,
        query: str,
        context: Optional[str] = None,
        max_tokens_logic: int = 1024,
        max_tokens_context: int = 4096,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> GenerationResult:
        """
        Двухэтапная генерация:
        1. LOGIC - даёт короткий ответ/план
        2. CONTEXT - расширяет ответ используя вывод LOGIC как промт
        
        Args:
            query: Запрос пользователя
            context: Дополнительный контекст из FractalGraph
            max_tokens_logic: Максимум токенов для LOGIC модели (1024)
            max_tokens_context: Максимум токенов для CONTEXT модели (4096)
            temperature: Температура
            system_prompt: Системный промпт
            
        Returns:
            GenerationResult от CONTEXT модели
        """
        start_time = time.time()
        logger.info(f"[DUAL_GENERATE] Начало двухэтапной генерации для: {query[:30]}...")
        
        # Этап 1: LOGIC - короткий ответ
        logger.info("[DUAL_GENERATE] Этап 1: LOGIC модель...")
        
        if not self._load_model(ModelType.LOGIC):
            return GenerationResult(
                text="Ошибка: LOGIC модель недоступна",
                model_used="none",
                generation_time=0.0,
                tokens_generated=0,
                confidence=0.0
            )
        
        logic_model = self.models[ModelType.LOGIC]
        full_context = self._build_context(query, context)
        logic_prompt = self._format_prompt(query, full_context, system_prompt, ModelType.LOGIC)
        
        logic_output = logic_model(
            logic_prompt,
            max_tokens=max_tokens_logic,
            temperature=temperature,
            stop=["<|im_end|>", "<|im_start|>", "<|endoftext|>"],
            echo=False,
            repeat_penalty=1.1,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        logic_text = logic_output['choices'][0]['text']
        logic_text = logic_text.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
        logic_tokens = logic_output['usage']['completion_tokens']
        
        logger.info(f"[DUAL_GENERATE] LOGIC вывод: {logic_text[:100]}... ({logic_tokens} токенов)")
        
        # Этап 2: CONTEXT - расширение с выводом LOGIC
        logger.info("[DUAL_GENERATE] Этап 2: CONTEXT модель...")
        
        if not self._load_model(ModelType.CONTEXT):
            return GenerationResult(
                text=logic_text,  # Return LOGIC output if CONTEXT fails
                model_used="logic_only",
                generation_time=time.time() - start_time,
                tokens_generated=logic_tokens,
                confidence=0.7
            )
        
        context_model = self.models[ModelType.CONTEXT]
        
        # Формируем промт для CONTEXT: оригинальный запрос + ответ от LOGIC
        combined_prompt = self._format_prompt(
            query=f"Запрос: {query}\n\nКраткий ответ: {logic_text}",
            context=full_context,
            system_prompt=system_prompt or "Дай развёрнутый, подробный ответ на основе краткого ответа.",
            model_type=ModelType.CONTEXT
        )
        
        context_output = context_model(
            combined_prompt,
            max_tokens=max_tokens_context,
            temperature=temperature,
            stop=["<|im_end|>", "<|im_start|>", "<|endoftext|>"],
            echo=False,
            repeat_penalty=1.1,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        context_text = context_output['choices'][0]['text']
        context_text = context_text.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
        context_tokens = context_output['usage']['completion_tokens']
        
        total_time = time.time() - start_time
        logger.info(f"[DUAL_GENERATE] CONTEXT вывод: {context_text[:100]}... ({context_tokens} токенов)")
        
        return GenerationResult(
            text=context_text,
            model_used="logic+context",
            generation_time=total_time,
            tokens_generated=logic_tokens + context_tokens,
            confidence=0.85
        )
    
    def generate_unified(
        self,
        query: str,
        context: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> GenerationResult:
        """
        Одноэтапная генерация с единым промтом и XML-выводом (metod.txt).
        
        Объединяет LOGIC + CONTEXT в один LLM вызов:
        - Концепты и противоречия добавляются в контекст ДО генерации
        - Модель генерирует ответ в XML-формате
        - Парсинг XML для получения short_answer, conclusion, full_answer
        
        Args:
            query: Запрос пользователя
            context: Дополнительный контекст
            max_tokens: Максимум токенов для генерации
            temperature: Температура
            system_prompt: Системный промпт
            
        Returns:
            GenerationResult с объединённым выводом
        """
        import re
        start_time = time.time()
        logger.info(f"[UNIFIED] Начало одноэтапной генерации для: {query[:30]}...")
        
        # Проверка модели
        if not self._load_model(ModelType.CONTEXT):
            if not self._load_model(ModelType.LOGIC):
                return GenerationResult(
                    text="Ошибка: модель недоступна",
                    model_used="none",
                    generation_time=0.0,
                    tokens_generated=0,
                    confidence=0.0
                )
            model_type = ModelType.LOGIC
        else:
            model_type = ModelType.CONTEXT
        
        model = self.models[model_type]
        
        # Формируем промт с концептами и противоречиями
        full_context = self._build_context(query, context)
        
        # Unified system prompt с инструкциями для XML-формата
        unified_system = system_prompt or (
            "Ты — когнитивная нейросетевая система ЕВА. "
            "Ответь строго в указанном XML-формате. "
            "Используй концепты и противоречия из контекста для формирования ответа. "
            "Убедись, что развёрнутый ответ логически согласуется с кратким выводом."
        )
        
        # Универсальный промт с XML-инструкциями (metod.txt)
        user_message = f"""Запрос: {query}
Контекст: {full_context if full_context else 'Нет дополнительного контекста'}

Сформируй ответ строго в формате:
<short_answer>1-2 предложения краткого ответа</short_answer>
<conclusion>На вопрос о {query[:30]}... можно сказать, что ...</conclusion>
<full_answer>Подробный развёрнутый ответ с учётом концептов и противоречий из контекста...</full_answer>"""
        
        prompt = self._format_prompt(query=user_message, context="", system_prompt=unified_system, model_type=model_type)
        
        # Генерация
        output = model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|im_end|>", "<|im_start|>", "<|endoftext|>"],
            echo=False,
            repeat_penalty=1.1,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        raw_text = output['choices'][0]['text']
        raw_text = raw_text.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
        tokens = output['usage']['completion_tokens']
        
        # Парсинг XML с fallback
        parsed = self._parse_structured_output(raw_text)
        
        total_time = time.time() - start_time
        combined_text = f"{parsed['short_answer']}\n\n{parsed['conclusion']}\n\n{parsed['full_answer']}"
        
        logger.info(f"[UNIFIED] Результат: {parsed['short_answer'][:50]}... ({tokens} токенов, {total_time:.1f}s)")
        
        return GenerationResult(
            text=combined_text,
            model_used=f"unified_{model_type.value}",
            generation_time=total_time,
            tokens_generated=tokens,
            confidence=0.85,
            metadata={
                'short_answer': parsed['short_answer'],
                'conclusion': parsed['conclusion'],
                'full_answer': parsed['full_answer']
            }
        )
    
    def _parse_structured_output(self, text: str) -> Dict[str, str]:
        """
        Надёжный парсинг XML-тегов с fallback-логикой (metod.txt).
        
        Args:
            text: Текст для парсинга
            
        Returns:
            Dict с short_answer, conclusion, full_answer
        """
        import re
        
        def extract(tag: str) -> str:
            match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
            return match.group(1).strip() if match else ""
        
        short = extract("short_answer")
        conclusion = extract("conclusion")
        full = extract("full_answer")
        
        if not (short and full):
            # Fallback: разделение по первому абзацу, если модель нарушила формат
            parts = text.strip().split("\n\n", 1)
            short = short or parts[0].strip()
            full = full or (parts[1] if len(parts) > 1 else text)
        
        return {
            "short_answer": short or "Ответ недоступен",
            "conclusion": conclusion or f"На вопрос можно сказать, что {short[:50]}" if short else "Вывод недоступен",
            "full_answer": full or text
        }
    
    def generate_iterative(
        self,
        query: str,
        context: Optional[str] = None,
        max_tokens_logic: int = 1024,
        max_tokens_context: int = 4096,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        check_contradictions: bool = True,
        check_concepts: bool = True,
        task_type: str = "default"
    ) -> GenerationResult:
        """
        Двухэтапная генерация с обогащением контекста концептами и противоречиями:
        
        1. LOGIC - даёт краткий ответ
        2. CONTEXT - расширяет ответ + добавляет связанные концепты и противоречия из FractalGraph
        
        Args:
            query: Запрос пользователя
            context: Дополнительный контекст из FractalGraph
            max_tokens_logic: Максимум токенов для LOGIC модели (1024)
            max_tokens_context: Максимум токенов для CONTEXT модели (4096)
            temperature: Температура
            system_prompt: Системный промпт
            check_contradictions: Включить противоречия в контекст
            check_concepts: Включить концепты в контекст
            task_type: Тип задачи для приоритизации
            
        Returns:
            GenerationResult от CONTEXT модели
        """
        start_time = time.time()
        logger.info(f"[ITERATIVE] Начало генерации для: {query[:30]}...")
        
        # Use ModelAccessManager if available
        if self._model_access is not None:
            priority = self._get_priority_for_task(task_type)
            request_id = self._model_access.request_access(
                priority=priority,
                task_type=task_type,
                callback=self._do_generate_iterative,
                query=query,
                context=context,
                max_tokens_logic=max_tokens_logic,
                max_tokens_context=max_tokens_context,
                temperature=temperature,
                system_prompt=system_prompt,
                check_contradictions=check_contradictions,
                check_concepts=check_concepts,
                timeout=120.0
            )
            
            try:
                return self._model_access.get_result(request_id, timeout=120.0)
            except Exception as e:
                logger.error(f"ModelAccess error: {e}")
        
        # Direct generation
        return self._do_generate_iterative(
            query, context, max_tokens_logic, max_tokens_context,
            temperature, system_prompt, check_contradictions, check_concepts
        )
    
    def _do_generate_iterative(
        self,
        query: str,
        context: Optional[str] = None,
        max_tokens_logic: int = 256,
        max_tokens_context: int = 512,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        check_contradictions: bool = True,
        check_concepts: bool = True
    ) -> GenerationResult:
        """Выполняет iterative генерацию (вызывается через ModelAccessManager)."""
        start_time = time.time()
        logger.info(f"[ITERATIVE] Начало генерации для: {query[:30]}...")
        
        # Получаем дополнительный контекст из FractalGraph
        full_context = self._build_context(query, context)
        
        # Этап 1: LOGIC - первичный краткий ответ
        logger.info("[ITERATIVE] Этап 1: LOGIC...")
        
        if not self._load_model(ModelType.LOGIC):
            return GenerationResult(
                text="Ошибка: LOGIC модель недоступна",
                model_used="none",
                generation_time=0.0,
                tokens_generated=0,
                confidence=0.0
            )
        
        logic_model = self.models[ModelType.LOGIC]
        logic_prompt = self._format_prompt(query, full_context, system_prompt, ModelType.LOGIC)
        
        logic_output = logic_model(
            logic_prompt,
            max_tokens=max_tokens_logic,
            temperature=temperature,
            stop=["<|im_end|>", "<|im_start|>", "<|endoftext|>"],
            echo=False,
            repeat_penalty=1.1
        )
        
        logic_text = logic_output['choices'][0]['text']
        logic_text = logic_text.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
        logic_tokens = logic_output['usage']['completion_tokens']
        
        logger.info(f"[ITERATIVE] LOGIC: {logic_text[:80]}...")
        
        # Этап 2: CONTEXT - расширение с добавлением концептов и противоречий
        logger.info("[ITERATIVE] Этап 2: CONTEXT (с концептами и противоречиями)...")
        
        if not self._load_model(ModelType.CONTEXT):
            return GenerationResult(text=logic_text, model_used="logic_only", generation_time=time.time()-start_time, tokens_generated=logic_tokens, confidence=0.7)
        
        context_model = self.models[ModelType.CONTEXT]
        
        # Формируем расширенный контекст с концептами, противоречиями и веб-поиском
        enriched_context = full_context
        
        if check_concepts:
            concepts_context = self._get_concepts_context(query)
            if concepts_context:
                enriched_context += f"\n\nСвязанные концепты: {concepts_context}"
        
        if check_contradictions:
            contradictions_context = self._get_contradictions_context(query)
            if contradictions_context:
                enriched_context += f"\n\nИзвестные противоречия: {contradictions_context}"
        
        # Веб-поиск для актуальной информации
        web_context = self._get_web_search_context(query)
        if web_context:
            enriched_context += f"\n\nАктуальная информация из интернета: {web_context}"
        
        combined_prompt = self._format_prompt(
            query=f"Запрос: {query}\n\nКраткий ответ: {logic_text}",
            context=enriched_context,
            system_prompt=system_prompt or "Дай развёрнутый, подробный ответ на основе краткого ответа и связанных концептов.",
            model_type=ModelType.CONTEXT
        )
        
        context_output = context_model(
            combined_prompt,
            max_tokens=max_tokens_context,
            temperature=temperature,
            stop=["<|im_end|>", "<|im_start|>", "<|endoftext|>"],
            echo=False,
            repeat_penalty=1.1
        )
        
        final_text = context_output['choices'][0]['text']
        final_text = final_text.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
        final_tokens = context_output['usage']['completion_tokens']
        
        total_time = time.time() - start_time
        total_tokens = logic_tokens + final_tokens
        
        logger.info(f"[ITERATIVE] Финальный ответ: {final_text[:80]}... ({final_tokens} токенов)")
        
        return GenerationResult(
            text=final_text,
            model_used="logic+context",
            generation_time=total_time,
            tokens_generated=total_tokens,
            confidence=0.85
        )
    
    def _get_concepts_context(self, query: str) -> str:
        """Получить связанные концепты из FractalGraphV2."""
        try:
            from eva_ai.memory.fractal_graph_v2 import get_fractal_graph
            fg = get_fractal_graph()
            if not fg:
                return ""
            
            # Поиск похожих концептов
            concepts = fg.search_nodes(
                query=query,
                node_type='concept',
                limit=5
            )
            
            if not concepts:
                return ""
            
            concepts_text = []
            for c in concepts:
                name = c.get('name', '')
                description = c.get('description', '')[:100]
                if name and description:
                    concepts_text.append(f"- {name}: {description}")
            
            return "\n".join(concepts_text) if concepts_text else ""
            
        except Exception as e:
            logger.debug(f"Could not get concepts: {e}")
            return ""
    
    def _get_contradictions_context(self, query: str) -> str:
        """Получить противоречия из FractalGraphV2."""
        try:
            from eva_ai.memory.fractal_graph_v2 import get_fractal_graph
            fg = get_fractal_graph()
            if not fg:
                return ""
            
            # Поиск противоречий
            contradictions = fg.search_nodes(
                query=query,
                node_type='contradiction',
                limit=3
            )
            
            if not contradictions:
                return ""
            
            contr_text = []
            for c in contradictions:
                title = c.get('title', '')
                description = c.get('description', '')[:100]
                if title and description:
                    contr_text.append(f"- {title}: {description}")
            
            return "\n".join(contr_text) if contr_text else ""
            
        except Exception as e:
            logger.debug(f"Could not get contradictions: {e}")
            return ""
    
    def _get_web_search_context(self, query: str) -> str:
        """Получить актуальную информацию из веб-поиска."""
        try:
            # Проверяем нужен ли веб-поиск
            from eva_ai.core.brain_query import needs_web_search
            need_search, _ = needs_web_search(query)
            
            if not need_search:
                return ""
            
            from eva_ai.websearch.web_search_integrated import get_web_search_engine
            web_search = get_web_search_engine()
            if not web_search or not hasattr(web_search, 'search'):
                return ""
            
            # Выполняем поиск
            results = web_search.search(query, max_results=3)
            if not results:
                return ""
            
            search_text = []
            for r in results:
                title = r.get('title', '')[:80]
                content = r.get('content', '')[:200]
                if title and content:
                    search_text.append(f"- {title}: {content}")
            
            return "\n".join(search_text) if search_text else ""
            
        except Exception as e:
            logger.debug(f"Could not get web search: {e}")
            return ""
    
    def generate_code(
        self,
        query: str,
        language: str = "python",
        max_tokens: int = 1024
    ) -> GenerationResult:
        """Генерация кода через Coder модель."""
        system_prompt = f"You are a {language} coding assistant. Generate clean, well-commented code."
        return self.generate(
            query=query,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=0.3
        )
    
    def _build_concept_contradiction_context(self, query: str) -> str:
        """
        Извлекает концепты и противоречия ДО генерации (metod.txt).
        
        Использует concept_extractor и contradiction_generator для получения
        релевантных концептов и противоречий по запросу, чтобы добавить их
        в контекст до генерации ответа.
        
        Returns:
            Строка с форматированными концептами и противоречиями
        """
        parts = []
        
        # 1. Концепты через ConceptExtractor
        if self.brain and hasattr(self.brain, 'concept_extractor'):
            try:
                concept_context = self.brain.concept_extractor.get_concepts_for_prompt(query, max_concepts=3)
                if concept_context:
                    parts.append(concept_context)
            except Exception as e:
                logger.debug(f"Concept extraction error: {e}")
        
        # 2. Противоречия через ContradictionGenerator
        if self.brain and hasattr(self.brain, 'contradiction_generator'):
            try:
                # Extract key term for contradiction lookup
                words = query.lower().split()
                stop_words = {'что', 'как', 'где', 'когда', 'почему', 'это', 'какой', 'какая', 'какое', 'какие'}
                key_terms = [w for w in words if w not in stop_words and len(w) > 3]
                
                for term in key_terms[:2]:  # Проверяем первые 2 значимых термина
                    contr_context = self.brain.contradiction_generator.get_contradictions_for_prompt(term)
                    if contr_context:
                        parts.append(contr_context)
                        break  # Достаточно одного найденного
            except Exception as e:
                logger.debug(f"Contradiction lookup error: {e}")
        
        if parts:
            logger.debug(f"Added concept/contradiction context: {len(parts)} blocks")
            return "\n".join(parts)
        
        return ""
    
    def _build_context(self, query: str, provided_context: Optional[str]) -> str:
        """Построить контекст из FractalGraph и HybridTokenCache с асинхронной загрузкой."""
        if provided_context:
            return provided_context
        
        contexts = []
        
        # === 0. Концепты и противоречия ДО генерации (metod.txt) ===
        concept_contr_context = self._build_concept_contradiction_context(query)
        if concept_contr_context:
            contexts.append(concept_contr_context)
        
        # 1. FractalGraph через semantic search
        if self.fractal_graph:
            try:
                if hasattr(self.fractal_graph, 'semantic_search'):
                    results = self.fractal_graph.semantic_search(query, top_k=20)
                    if results:
                        for r in results[:10]:
                            content = r.get('content', '')
                            if content:
                                contexts.append(content[:1000])  # Больше контента
            except Exception as e:
                logger.debug(f"Semantic search error: {e}")
        
        # 2. HybridTokenCache - история разговоров и горячие данные
        if self.brain:
            cache = getattr(self.brain, 'hybrid_cache', None)
            if cache:
                try:
                    # Semantic поиск в кэше
                    cache_results = []
                    if hasattr(cache, 'search'):
                        cache_results = cache.search(query, top_k=10)
                    elif hasattr(cache, 'get_similar'):
                        cache_results = cache.get_similar(query, top_k=10)
                    
                    for item in cache_results:
                        if isinstance(item, dict):
                            text = item.get('text', '') or item.get('content', '') or item.get('value', '')
                        else:
                            text = str(item)
                        if text and len(text) > 20:
                            contexts.append(text[:2000])  # Больше из кэша
                except Exception as e:
                    logger.debug(f"HybridCache search error: {e}")
                
                # Загружаем недавние разговоры из RAM кэша
                try:
                    if hasattr(cache, 'ram_cache'):
                        for key, value in list(cache.ram_cache.items())[:5]:
                            if isinstance(value, dict):
                                text = value.get('text', '') or value.get('content', '') or ''
                            else:
                                text = str(value)
                            if text and len(text) > 50:
                                contexts.append(text[:1500])
                except Exception as e:
                    logger.debug(f"RAM cache access error: {e}")
        
        # 3. Контекст из conversation history (если есть)
        try:
            if self.brain and hasattr(self.brain, 'session_history'):
                history = getattr(self.brain, 'session_history', [])
                for conv in history[-5:]:  # Последние 5 сообщений
                    if isinstance(conv, dict):
                        q = conv.get('query', '') or conv.get('question', '')
                        a = conv.get('response', '') or conv.get('answer', '')
                        if q:
                            contexts.append(f"Q: {q[:500]}")
                        if a:
                            contexts.append(f"A: {a[:1000]}")
        except Exception as e:
            logger.debug(f"History access error: {e}")
        
        # 4. Fallback на простой поиск в графе
        if len(contexts) < 3 and self.fractal_graph:
            try:
                if hasattr(self.fractal_graph, 'nodes'):
                    query_words = query.lower().split()[:5]
                    for node_id, node in list(self.fractal_graph.nodes.items())[:50]:
                        content = getattr(node, 'content', '')
                        if content and any(w in content.lower() for w in query_words):
                            contexts.append(content[:500])
            except Exception as e:
                logger.debug(f"Simple search error: {e}")
        
        # Объединяем контексты с адаптивным лимитом
        if contexts:
            # Адаптивный лимит: больше для сложных запросов
            max_context_chars = 15000  # ~10k токенов через гибридный кэш
            combined = '\n\n'.join(contexts[:8])  # До 8 блоков
            
            if len(combined) > max_context_chars:
                combined = combined[:max_context_chars] + '\n\n[контекст обрезан]'
            
            logger.debug(f"Контекст собран: {len(contexts)} блоков, {len(combined)} символов")
            return combined
        
        return ""
    
    def _prefetch_context_async(self, query: str) -> None:
        """Асинхронно предзагрузить релевантный контекст в кэш."""
        if not self.brain:
            return
        
        try:
            import threading
            
            def _async_prefetch():
                cache = getattr(self.brain, 'hybrid_cache', None)
                if not cache:
                    return
                
                try:
                    # Предзагружаем похожие запросы
                    if hasattr(cache, 'prefetch'):
                        cache.prefetch(query)
                    elif hasattr(cache, 'warm_up'):
                        cache.warm_up(query)
                    
                    # Пытаемся загрузить из дискового кэша горячие данные
                    if hasattr(cache, 'disk_cache') and hasattr(cache.disk_cache, 'get_recent'):
                        recent = cache.disk_cache.get_recent(limit=10)
                        for item in recent:
                            if item and isinstance(item, dict):
                                text = item.get('text', '') or item.get('content', '')
                                if text and self._is_relevant(text, query):
                                    if hasattr(cache, 'add_token'):
                                        cache.add_token(f"prefetch_{id(item)}", item)
                except Exception as e:
                    logger.debug(f"Async prefetch error: {e}")
            
            thread = threading.Thread(target=_async_prefetch, daemon=True)
            thread.start()
        except Exception as e:
            logger.debug(f"Thread creation error: {e}")
    
    def _is_relevant(self, text: str, query: str, threshold: float = 0.3) -> bool:
        """Проверка релевантности текста к запросу."""
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        if not query_words:
            return True
        
        intersection = query_words & text_words
        if not intersection:
            return False
        
        # Проверяем density пересечения
        relevance = len(intersection) / len(query_words)
        return relevance >= threshold
    
    def _format_prompt(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str],
        model_type: ModelType
    ) -> str:
        """Форматировать промпт для модели Qwen с правильным chat template."""
        text = ""
        
        # System prompt - всегда в system role
        if system_prompt:
            text += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        else:
            default_systems = {
                ModelType.CODER: "Ты - Eva AI, продвинутый ИИ-ассистент. Пиши чистый, хорошо комментируемый код.",
                ModelType.CONTEXT: "Ты - Eva AI, продвинутый ИИ-ассистент. Давай развёрнутые, подробные ответы.",
                ModelType.LOGIC: "Ты - Eva AI, продвинутый ИИ-ассистент. Давай точные, логичные ответы."
            }
            text += f"<|im_start|>system\n{default_systems.get(model_type, 'Ты - Eva AI, продвинутый ИИ-ассистент.')}<|im_end|>\n"
        
        # Контекст + запрос В ОДНОМ user сообщении
        user_message = query
        if context:
            user_message = f"Контекст: {context}\n\nВопрос: {query}"
        
        text += f"<|im_start|>user\n{user_message}<|im_end|>\n"
        
        # Маркер для начала ответа
        text += "<|im_start|>assistant\n"
        
        return text
    
    def _save_to_graph(self, query: str, response: str):
        """Сохранить в FractalGraph."""
        if not self.fractal_graph:
            return
        
        try:
            if hasattr(self.fractal_graph, 'add_conversation'):
                self.fractal_graph.add_conversation(query, response)
            elif hasattr(self.fractal_graph, 'add_node'):
                import hashlib
                node_id = hashlib.md5(f"conv:{query}".encode()).hexdigest()[:12]
                self.fractal_graph.add_node(
                    node_id=node_id,
                    node_type='conversation',
                    content=f"Q: {query}\nA: {response}",
                    metadata={'query': query, 'response': response}
                )
        except Exception as e:
            logger.debug(f"Could not save to graph: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику."""
        return {
            "models_loaded": {
                mt.value: mt in self.models and self.models[mt] is not None
                for mt in ModelType
            },
            "model_paths": {
                mt.value: str(path) if path else None
                for mt, path in self._model_paths.items()
            },
            "router_decisions": self.router.CONTEXT_KEYWORDS[:5]
        }
    
    def unload_all(self):
        """Выгрузить все модели."""
        for model_type in list(self.models.keys()):
            if self.models[model_type] is not None:
                del self.models[model_type]
                self.models[model_type] = None
                logger.info(f"Unloaded {model_type.value}")
    
    def generate_with_chunked_context(
        self,
        query: str,
        context: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        max_context_chunks: int = 3
    ) -> GenerationResult:
        """
        Генерация с чанкованным контекстом.
        
        Разбивает контекст на чанки и обрабатывает их поэтапно,
        что эффективнее для больших контекстов.
        """
        # 1. Получаем контекст
        full_context = context or self._build_context(query, None)
        
        if not full_context or len(full_context) < 200:
            # Малый контекст - обычная генерация
            return self.generate(query, context, max_tokens, temperature)
        
        # 2. Чанкуем контекст
        processor = ChunkedContextProcessor(max_chunk_size=800)
        chunks = processor.process(full_context, query)
        
        if not chunks:
            return self.generate(query, context, max_tokens, temperature)
        
        # 3. Обрабатываем чанки
        all_texts = []
        
        for i, chunk in enumerate(chunks[:max_context_chunks]):
            # Добавляем контекст от предыдущих чанков
            previous = '\n\n'.join(all_texts)
            
            # Расширяем промпт с чанком
            extended_query = f"{query}\n\n[Дополнительный контекст {i+1}/{len(chunks[:max_context_chunks])}]:\n{chunk.content}"
            
            result = self.generate(
                query=extended_query,
                context=previous if previous else None,
                max_tokens=max_tokens // max_context_chunks,
                temperature=temperature
            )
            
            if result and result.text:
                all_texts.append(result.text)
        
        # 4. Синтезируем финальный ответ
        final_text = '\n\n'.join(all_texts)
        
        return GenerationResult(
            text=final_text,
            model_used="logic_chunked",
            generation_time=sum(r.generation_time for r in [self.generate(query, "", 1, 0.7)] if r) if all_texts else 0,
            tokens_generated=len(final_text.split()),
            confidence=0.85
        )
    
    def generate_streaming(
        self,
        query: str,
        context: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        chunk_size: int = 80,
        task_type: str = "default"
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Генерация со стримингом токенов в реальном времени.
        
        Yields чанки текста ПО МЕРЕ генерации для натурального UX.
        Использует OpenVINO если use_openvino=True, иначе llama.cpp.
        
        Args:
            query: Запрос
            context: Дополнительный контекст
            max_tokens: Максимум токенов
            temperature: Температура
            chunk_size: Размер чанка для выдачи (символы)
            task_type: Тип задачи для роутинга устройств
                       (self_dialog, concept_mining, contradiction_mining, coder, default)
            
        Yields:
            Dict с 'type', 'text', 'is_final', 'tokens_count', 'elapsed_ms'
        """
        import time
        
        start_time = time.time()
        
        # Check if ModelAccessManager is available and use it
        if self._model_access is not None:
            priority = self._get_priority_for_task(task_type)
            
            request_id = self._model_access.request_access(
                priority=priority,
                task_type=task_type,
                callback=self._do_generate_streaming,
                query=query,
                context=context,
                max_tokens=max_tokens,
                temperature=temperature,
                chunk_size=chunk_size,
                timeout=120.0
            )
            
            # Yield from the generator result
            try:
                for chunk in self._model_access.get_result(request_id, timeout=120.0):
                    yield chunk
                return
            except Exception as e:
                logger.error(f"ModelAccess error: {e}")
                # Fall through to direct generation
        
        # Direct generation without ModelAccessManager
        yield from self._do_generate_streaming(
            query, context, max_tokens, temperature, chunk_size, task_type
        )
    
    def _do_generate_streaming(
        self,
        query: str,
        context: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        chunk_size: int = 80,
        task_type: str = "default"
    ) -> Generator[Dict[str, Any], None, None]:
        """Выполняет генерацию со стримингом (вызывается через ModelAccessManager)."""
        import time
        
        start_time = time.time()
        
        # OpenVINO path
        if self.use_openvino:
            gen, model_type = self._get_generator_for_task(task_type)
            if gen:
                yield from self._generate_streaming_openvino(
                    query, context, max_tokens, temperature, gen, model_type
                )
                return
        
        # Fallback: определяем модель через роутер
        model_type = self.router.route(query)
        
        # Загружаем модель
        if not self._load_model(model_type):
            fallback_type = ModelType.CONTEXT if model_type == ModelType.LOGIC else ModelType.LOGIC
            if not self._load_model(fallback_type):
                yield {
                    'type': 'error',
                    'text': 'Модели недоступны',
                    'is_final': True,
                    'tokens_count': 0,
                    'elapsed_ms': 0
                }
                return
            model_type = fallback_type
        
        model = self.models[model_type]
        
        # Формируем промпт
        full_context = self._build_context(query, context)
        prompt = self._format_prompt(query, full_context, None, model_type)
        
        # Реальный streaming от GGUF
        full_text = ""
        buffer = ""
        delimiter = "\n"
        
        try:
            stream = model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["<|im_end|>", "<|im_start|>", "<|endoftext|>"],
                echo=False,
                stream=True
            )
            
            for chunk in stream:
                text = chunk.get('choices', [{}])[0].get('text', '')
                if not text:
                    continue
                
                buffer += text
                full_text += text
                
                # Отправляем чанк когда накопилось достаточно символов или есть разделитель
                if len(buffer) >= chunk_size or delimiter in buffer:
                    parts = buffer.split(delimiter)
                    while len(parts) > 1:
                        elapsed = int((time.time() - start_time) * 1000)
                        tokens_count = len(full_text.split())
                        
                        # Debug: логируем скорость генерации
                        if tokens_count % 50 == 0:
                            chars_per_sec = len(full_text) / (elapsed / 1000) if elapsed > 0 else 0
                            logger.debug(f"[STREAM] tokens={tokens_count}, elapsed={elapsed}ms, speed={chars_per_sec:.1f} chars/s")
                        
                        yield {
                            'type': 'chunk',
                            'text': parts[0] + (delimiter if parts[0] else ''),
                            'is_final': False,
                            'tokens_count': tokens_count,
                            'elapsed_ms': elapsed,
                            'chunk_index': 0,
                            'total_chunks': 0,
                            'progress': 0
                        }
                        parts = parts[1:]
                    buffer = parts[0] if parts else ""
            
            # Отправляем оставшийся текст
            if buffer:
                yield {
                    'type': 'complete',
                    'text': buffer,
                    'is_final': True,
                    'tokens_count': len(full_text.split()),
                    'elapsed_ms': int((time.time() - start_time) * 1000),
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'progress': 1.0
                }
            else:
                yield {
                    'type': 'complete',
                    'text': '',
                    'is_final': True,
                    'tokens_count': len(full_text.split()),
                    'elapsed_ms': int((time.time() - start_time) * 1000),
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'progress': 1.0
                }
                
        except Exception as e:
            logger.error(f"Streaming generation error: {e}")
            yield {
                'type': 'error',
                'text': f'Ошибка генерации: {str(e)}',
                'is_final': True,
                'tokens_count': len(full_text.split()),
                'elapsed_ms': int((time.time() - start_time) * 1000)
            }
    
    def _generate_streaming_openvino(
        self,
        query: str,
        context: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        generator=None,
        model_type=None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Генерация со стримингом через OpenVINO.
        
        Args:
            query: Запрос
            context: Дополнительный контекст
            max_tokens: Максимум токенов
            temperature: Температура
            generator: OpenVINOGenerator
            model_type: ModelType для форматирования промпта
            
        Yields:
            Dict с данными чанка
        """
        if model_type is None:
            model_type = self.router.route(query)
        
        full_context = self._build_context(query, context)
        prompt = self._format_prompt(query, full_context, None, model_type)
        
        try:
            for chunk in generator.generate_streaming(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop_tokens=["<|im_end|>", "<|im_start|>", "<|endoftext|>"],
                chunk_size=25
            ):
                if chunk['type'] == 'complete':
                    yield {
                        'type': 'complete',
                        'text': chunk.get('text', ''),
                        'is_final': True,
                        'tokens_count': chunk.get('tokens_count', 0),
                        'elapsed_ms': chunk.get('elapsed_ms', 0),
                        'chunk_index': 0,
                        'total_chunks': 1,
                        'progress': 1.0
                    }
                else:
                    yield chunk
                
        except Exception as e:
            logger.error(f"OpenVINO streaming error: {e}")
            yield {
                'type': 'error',
                'text': f'Ошибка: {e}',
                'is_final': True,
                'tokens_count': 0,
                'elapsed_ms': 0
            }
    
    def _split_text_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Разбить текст на чанки для стриминга."""
        if len(text) <= chunk_size:
            return [text] if text else []
        
        chunks = []
        remaining = text
        delimiters = '.!?\n'
        
        while remaining:
            chunk_end = min(chunk_size, len(remaining))
            
            if chunk_end < len(remaining):
                # Ищем разделитель предложений
                found = False
                for delim in delimiters:
                    pos = remaining[:chunk_end].rfind(delim)
                    if pos > chunk_size // 2:
                        chunk_end = pos + 1
                        found = True
                        break
                
                if not found:
                    # Ищем пробел
                    space_pos = remaining[:chunk_end].rfind(' ')
                    if space_pos > chunk_size // 3:
                        chunk_end = space_pos + 1
            
            chunk = remaining[:chunk_end].strip()
            if chunk:
                chunks.append(chunk)
            
            remaining = remaining[chunk_end:].lstrip()
        
        return chunks if chunks else [text]


# Фабрика для создания UnifiedGenerator
def create_unified_generator(
    config: Optional[Dict] = None,
    fractal_graph=None,
    brain=None
) -> Optional[UnifiedGenerator]:
    """
    Создать UnifiedGenerator из конфигурации.
    
    Args:
        config: Конфигурация
        fractal_graph: FractalGraph V2
        brain: CoreBrain
        event_bus: EventBus для интеграции с ModelAccessManager
        
    Returns:
        UnifiedGenerator или None
    """
    try:
        general_path = None
        code_path = None
        
        if config:
            model_config = config.get('model', {})
            general_path = model_config.get('general_model_path')
            code_path = model_config.get('code_model_path')
        
        event_bus = getattr(brain, 'event_bus', None) or getattr(brain, '_new_event_bus', None)
        
        return UnifiedGenerator(
            logic_model_path=Path(general_path) if general_path else None,
            context_model_path=Path(general_path) if general_path else None,
            coder_model_path=Path(code_path) if code_path else None,
            fractal_graph=fractal_graph,
            brain=brain,
            event_bus=event_bus
        )
    except Exception as e:
        logger.error(f"Failed to create UnifiedGenerator: {e}")
        return None
