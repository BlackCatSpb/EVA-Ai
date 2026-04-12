"""
UnifiedGenerator - Единая система генерации на базе Pie Architecture

Использует:
- RuadaptQwen3-4B для общих задач
- Qwen Coder 1.5B для кода  
- L2 Роутинг для выбора модели
- Интеграция с FractalGraph V2
- ChunkedContextProcessor для обработки больших контекстов
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Generator
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

from .context_chunking import ChunkedContextProcessor, StreamingGenerator, ContextChunk

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
    
    Заменяет Two-Model Pipeline и Pie Fallback.
    Использует только RuadaptQwen3-4B и Qwen Coder 1.5B.
    """
    
    def __init__(
        self,
        logic_model_path: Optional[Path] = None,
        context_model_path: Optional[Path] = None,
        coder_model_path: Optional[Path] = None,
        n_ctx: int = 16384,  # Максимальный контекст для Pie
        n_threads: int = 4,
        fractal_graph=None,
        brain=None
    ):
        """
        Инициализация Unified Generator.
        
        Args:
            logic_model_path: Путь к LOGIC модели (RuadaptQwen3-4B condensed)
            context_model_path: Путь к CONTEXT модели (RuadaptQwen3-4B extended)
            coder_model_path: Путь к CODER модели (Qwen Coder 1.5B)
            n_ctx: Размер контекста
            n_threads: Количество потоков
            fractal_graph: FractalGraph V2 для контекста
            brain: Ссылка на CoreBrain
        """
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.fractal_graph = fractal_graph
        self.brain = brain
        self.router = SimpleRouter()
        
        self.models: Dict[ModelType, Any] = {}
        self._model_paths: Dict[ModelType, Path] = {}
        
        # Загружаем пути моделей
        self._load_model_paths(logic_model_path, context_model_path, coder_model_path)
        
        logger.info(f"UnifiedGenerator initialized")
        logger.info(f"  LOGIC model: {self._model_paths.get(ModelType.LOGIC)}")
        logger.info(f"  CONTEXT model: {self._model_paths.get(ModelType.CONTEXT)}")
        logger.info(f"  CODER model: {self._model_paths.get(ModelType.CODER)}")
    
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
        system_prompt: Optional[str] = None
    ) -> GenerationResult:
        """
        Генерация ответа.
        
        Args:
            query: Запрос пользователя
            context: Дополнительный контекст из FractalGraph
            max_tokens: Максимум токенов
            temperature: Температура
            system_prompt: Системный промпт
            
        Returns:
            GenerationResult
        """
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
        max_tokens_logic: int = 256,
        max_tokens_context: int = 512,
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
            max_tokens_logic: Максимум токенов для LOGIC модели
            max_tokens_context: Максимум токенов для CONTEXT модели
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
    
    def generate_iterative(
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
        """
        Двухэтапная генерация с обогащением контекста концептами и противоречиями:
        
        1. LOGIC - даёт краткий ответ
        2. CONTEXT - расширяет ответ + добавляет связанные концепты и противоречия из FractalGraph
        
        Args:
            query: Запрос пользователя
            context: Дополнительный контекст из FractalGraph
            max_tokens_logic: Максимум токенов для LOGIC модели
            max_tokens_context: Максимум токенов для CONTEXT модели
            temperature: Температура
            system_prompt: Системный промпт
            check_contradictions: Включить противоречия в контекст
            check_concepts: Включить концепты в контекст
            
        Returns:
            GenerationResult от CONTEXT модели
        """
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
    
    def _build_context(self, query: str, provided_context: Optional[str]) -> str:
        """Построить контекст из FractalGraph и HybridTokenCache с асинхронной загрузкой."""
        if provided_context:
            return provided_context
        
        contexts = []
        
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
        # Используем chat template как в qwen_model_manager
        text = ""
        
        # System prompt
        if system_prompt:
            text += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        else:
            # Дефолтный system prompt для разных типов
            default_systems = {
                ModelType.CODER: "Ты - Eva AI, продвинутый ИИ-ассистент. Пиши чистый, хорошо комментируемый код.",
                ModelType.CONTEXT: "Ты - Eva AI, продвинутый ИИ-ассистент. Давай развёрнутые, подробные ответы.",
                ModelType.LOGIC: "Ты - Eva AI, продвинутый ИИ-ассистент. Давай точные, логичные ответы."
            }
            text += f"<|im_start|>system\n{default_systems.get(model_type, 'Ты - Eva AI, продвинутый ИИ-ассистент.')}<|im_end|>\n"
        
        # Контекст из FractalGraph
        if context:
            text += f"<|im_start|>user\nКонтекст: {context}<|im_end|>\n"
        
        # Запрос пользователя
        text += f"<|im_start|>user\n{query}<|im_end|>\n"
        
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
        chunk_size: int = 80
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Генерация со стримингом токенов в реальном времени.
        
        Yields чанки текста ПО МЕРЕ генерации для натурального UX.
        Использует реальный streaming от GGUF модели.
        
        Args:
            query: Запрос
            context: Дополнительный контекст
            max_tokens: Максимум токенов
            temperature: Температура
            chunk_size: Размер чанка для выдачи (символы)
            
        Yields:
            Dict с 'type', 'text', 'is_final', 'tokens_count', 'elapsed_ms'
        """
        import time
        
        start_time = time.time()
        
        # Определяем модель
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
                        yield {
                            'type': 'chunk',
                            'text': parts[0] + (delimiter if parts[0] else ''),
                            'is_final': False,
                            'tokens_count': len(full_text.split()),
                            'elapsed_ms': int((time.time() - start_time) * 1000),
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
        
        return UnifiedGenerator(
            general_model_path=Path(general_path) if general_path else None,
            code_model_path=Path(code_path) if code_path else None,
            fractal_graph=fractal_graph,
            brain=brain
        )
    except Exception as e:
        logger.error(f"Failed to create UnifiedGenerator: {e}")
        return None
