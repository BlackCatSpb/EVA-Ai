"""
HybridKnowledgeDialogManager - Гибридный менеджер диалога с интеграцией графа знаний

Интегрирует:
1. Chat template форматирование (Qwen3-4B)
2. Prefix caching для истории диалога
3. Виртуальные токены знаний из FractalGraphV2
4. Концепты и противоречия из ConceptExtractor/ContradictionMiner
5. Валидация через ContradictionResolver

Использование:
    manager = HybridKnowledgeDialogManager(
        brain=brain,
        fractal_graph=graph,
        concept_extractor=extractor,
        contradiction_manager=cm
    )
    
    response, validated = await manager.process_user_input("Как работает ИИ?")
"""

import os
import time
import hashlib
import logging
import asyncio
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from collections import OrderedDict
from functools import partial

logger = logging.getLogger("eva_ai.core.hybrid_dialog_manager")

try:
    import openvino_genai as ov_genai
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    logger.warning("OpenVINO GenAI не доступен")


@dataclass
class DialogMessage:
    """Сообщение в диалоге."""
    role: str  # system, user, assistant
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeContext:
    """Контекст знаний из графа."""
    concepts: List[Dict[str, Any]] = field(default_factory=list)
    contradictions: List[Dict[str, Any]] = field(default_factory=list)
    recent_facts: List[str] = field(default_factory=list)
    virtual_tokens: str = ""
    graph_hash: str = ""


@dataclass
class GenerationResult:
    """Результат генерации с метаданными."""
    response: str
    validated: bool
    validation_notes: str
    knowledge_used: List[str]
    contradictions_resolved: List[str]
    processing_time: float
    cache_hit: bool
    reasoning: Optional[str] = None
    reasoning_steps: Optional[List[Dict]] = None


class PrefixCache:
    """
    Кэш префиксов для оптимизации генерации.
    
    Хранит KV-кеш для неизменных частей промпта:
    - Системный промпт
    - История диалога (до текущего сообщения)
    - Виртуальные токены знаний
    """
    
    def __init__(self, max_entries: int = 100):
        self.max_entries = max_entries
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
    
    def _compute_hash(self, prompt: str, max_tokens: int = 100) -> str:
        """Вычисляет хеш префикса промпта."""
        prefix = prompt[:max_tokens]
        return hashlib.sha256(prefix.encode('utf-8')).hexdigest()[:16]
    
    def get(self, prompt: str) -> Optional[Any]:
        """Получить кешированный результат."""
        key = self._compute_hash(prompt)
        with self._lock:
            if key in self._cache:
                self._hits += 1
                self._cache.move_to_end(key)
                return self._cache[key]
            self._misses += 1
            return None
    
    def set(self, prompt: str, value: Any):
        """Сохранить в кеш."""
        key = self._compute_hash(prompt)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return
            self._cache[key] = value
            if len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)
    
    def clear(self):
        """Очистить кеш."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def get_stats(self) -> Dict[str, int]:
        """Статистика кеша."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0
            return {
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'size': len(self._cache)
            }


class HybridKnowledgeDialogManager:
    """
    Гибридный менеджер диалога EVA с интеграцией графа знаний.
    
    Ключевые возможности:
    1. Chat template форматирование для Qwen3-4B
    2. Prefix caching для истории диалога
    3. Виртуальные токены знаний из FractalGraphV2
    4. Автоматическое извлечение концептов
    5. Детекция и разрешение противоречий
    6. Валидация ответов
    7. Две физические модели для последовательной генерации
    
    Параметры:
        brain: CoreBrain - ядро системы
        fractal_graph: FractalMemoryGraph - граф памяти
        concept_extractor: ConceptExtractor - извлекатель концептов
        contradiction_manager: ContradictionManager - менеджер противоречий
        model_path: str - путь к модели GGUF (Model A)
        model_b_path: str - путь к модели B (копия)
        device: str - устройство (CPU/GPU)
        max_history: int - максимум сообщений в истории
        enable_validation: bool - включить валидацию ответов
    """
    
    VIRTUAL_TOKEN_PLACEHOLDER = "<|knowledge|>"
    
    def __init__(
        self,
        brain=None,
        fractal_graph=None,
        concept_extractor=None,
        contradiction_manager=None,
        model_path: str = None,
        model_b_path: str = None,
        device: str = "CPU",
        max_history: int = 50,
        max_tokens: int = 512,
        temperature: float = 0.7,
        enable_validation: bool = True,
        validation_temperature: float = 0.1,
        **kwargs
    ):
        self.brain = brain
        self.fractal_graph = fractal_graph
        self.concept_extractor = concept_extractor
        self.contradiction_manager = contradiction_manager
        
        # Параметры генерации
        self.model_path = model_path
        self.model_b_path = model_b_path  # Путь к модели B
        self.device = device
        self.max_history = max_history
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_validation = enable_validation
        self.validation_temperature = validation_temperature
        
        # History
        self.messages: List[DialogMessage] = []
        self._history_lock = threading.Lock()
        
        # Кеш
        self.prefix_cache = PrefixCache(max_entries=100)
        self.knowledge_cache: Dict[str, KnowledgeContext] = {}
        
        # Pipeline - две физические модели
        self._pipeline = None
        self._pipeline_b = None  # Model B
        self._tokenizer = None
        self._scheduler_config = None
        self._initialized = False
        
        # Callbacks
        self._on_concept_extracted: Optional[Callable] = None
        self._on_contradiction_detected: Optional[Callable] = None
        
        logger.info(f"HybridKnowledgeDialogManager инициализирован: "
                    f"device={device}, validation={enable_validation}")
    
    @property
    def initialized(self) -> bool:
        """Проверка инициализации."""
        return self._initialized
    
    def initialize(self, model_path: str = None, model_b_path: str = None, device: str = None):
        """
        Инициализация ДВУХ OpenVINO pipelines (Model A и Model B).
        
        Args:
            model_path: Путь к модели A
            model_b_path: Путь к модели B (копия)
            device: Устройство
        """
        if not OPENVINO_AVAILABLE:
            logger.error("OpenVINO GenAI не установлен")
            return False
        
        if model_path:
            self.model_path = model_path
        if model_b_path:
            self.model_b_path = model_b_path
        if device:
            self.device = device
        
        if not self.model_path:
            logger.error("model_path не указан")
            return False
        
        if not os.path.exists(self.model_path):
            logger.error(f"Model A не найдена: {self.model_path}")
            return False
        
        try:
            from eva_ai.core.openvino_generator import OpenVINOGenerator
            
            model_a_path_obj = Path(self.model_path)
            model_b_path_obj = Path(model_b_path) if model_b_path else None
            
            # === Инициализация двух физических моделей ===
            generator_a = None
            generator_b = None
            
            if self.brain and hasattr(self.brain, 'two_model_pipeline'):
                pipeline = self.brain.two_model_pipeline
                
                if hasattr(pipeline, '_openvino_cpu') and pipeline._openvino_cpu and pipeline._openvino_cpu._pipeline:
                    generator_a = pipeline._openvino_cpu
                    logger.info("HybridKnowledgeDialogManager: используем Model A")
                
                if hasattr(pipeline, '_openvino_gpu') and pipeline._openvino_gpu and pipeline._openvino_gpu._pipeline:
                    generator_b = pipeline._openvino_gpu
                    logger.info("HybridKnowledgeDialogManager: используем Model B")
            
            # Fallback: создаём независимые экземпляры
            if not generator_a or not generator_a._pipeline:
                logger.warning("HybridKnowledgeDialogManager: создаём Model A (fallback)")
                generator_a = OpenVINOGenerator(
                    model_path=model_a_path_obj,
                    device=self.device,
                    max_tokens=512,
                    temperature=self.temperature,
                    use_registry=False
                )
            
            # Model B
            if not generator_b or not generator_b._pipeline:
                if model_b_path_obj and model_b_path_obj.exists():
                    logger.warning("HybridKnowledgeDialogManager: создаём Model B (копия)")
                    generator_b = OpenVINOGenerator(
                        model_path=model_b_path_obj,
                        device=self.device,
                        max_tokens=2048,
                        temperature=self.temperature,
                        use_registry=False
                    )
                else:
                    generator_b = generator_a
                    logger.info("HybridKnowledgeDialogManager: Model B = Model A (fallback)")
            
            # Pipeline от Model A
            if generator_a and generator_a._pipeline:
                self._pipeline = generator_a._pipeline
                try:
                    self._tokenizer = self._pipeline.get_tokenizer()
                except:
                    self._tokenizer = None
                logger.info("HybridKnowledgeDialogManager: Model A pipeline готов")
            else:
                logger.error("Не удалось получить Model A pipeline")
                return False
            
            # Pipeline от Model B
            if generator_b and generator_b._pipeline:
                self._pipeline_b = generator_b._pipeline
                logger.info("HybridKnowledgeDialogManager: Model B pipeline готов")
            else:
                self._pipeline_b = self._pipeline
            
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Ошибка инициализации pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # =========================================================================
    # Chat Template форматирование
    # =========================================================================
    
    def _format_chat_prompt(
        self,
        messages: List[DialogMessage],
        add_generation_prompt: bool = True
    ) -> str:
        """
        Форматирует историю сообщений через chat template.
        
        Args:
            messages: Список сообщений
            add_generation_prompt: Добавить приглашение для генерации
            
        Returns:
            Отформатированный промпт
        """
        if not self._tokenizer:
            # Fallback - просто объединяем
            return self._format_simple_prompt(messages, add_generation_prompt)
        
        try:
            # Конвертируем в формат для OpenVINO
            chat_messages = [
                {"role": m.role, "content": m.content}
                for m in messages
            ]
            
            # OpenVINO tokenizer API: apply_chat_template(history, add_generation_prompt)
            prompt = self._tokenizer.apply_chat_template(
                chat_messages,
                add_generation_prompt
            )
            return prompt
            
        except Exception as e:
            logger.debug(f"Chat template error: {e}, используем fallback")
            return self._format_simple_prompt(messages, add_generation_prompt)
    
    def _format_simple_prompt(
        self,
        messages: List[DialogMessage],
        add_generation_prompt: bool = True
    ) -> str:
        """Простое форматирование без chat template."""
        parts = []
        
        for msg in messages:
            if msg.role == "system":
                parts.append(f"Система: {msg.content}")
            elif msg.role == "user":
                parts.append(f"Пользователь: {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"Ассистент: {msg.content}")
        
        prompt = "\n\n".join(parts)
        
        if add_generation_prompt:
            prompt += "\n\nАссистент: "
        
        return prompt
    
    # =========================================================================
    # Управление историей
    # =========================================================================
    
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """Добавить сообщение в историю."""
        with self._history_lock:
            msg = DialogMessage(
                role=role,
                content=content,
                metadata=metadata or {}
            )
            self.messages.append(msg)
            
            # Обрезаем историю если нужно
            self._trim_history()
    
    def _trim_history(self):
        """Обрезает историю до max_history."""
        if len(self.messages) <= self.max_history:
            return
        
        # Оставляем системное сообщение если есть
        system_msgs = [m for m in self.messages if m.role == "system"]
        other_msgs = [m for m in self.messages if m.role != "system"]
        
        # Берём последние max_history-1 сообщений
        keep = other_msgs[-(self.max_history - len(system_msgs)):]
        
        self.messages = system_msgs + keep
    
    def get_history(self) -> List[DialogMessage]:
        """Получить копию истории."""
        with self._history_lock:
            return list(self.messages)
    
    def clear_history(self):
        """Очистить историю."""
        with self._history_lock:
            # Оставляем системное сообщение если есть
            self.messages = [m for m in self.messages if m.role == "system"]
    
    # =========================================================================
    # Интеграция с графом знаний
    # =========================================================================
    
    def _compute_graph_state_hash(self) -> str:
        """
        Вычисляет хеш состояния графа для инвалидации кеша знаний.
        """
        if not self.fractal_graph:
            return ""
        
        try:
            # Берём хеш от состояния графа
            if hasattr(self.fractal_graph, 'get_state_hash'):
                return self.fractal_graph.get_state_hash()
            
            # Или от последних изменений
            if hasattr(self.fractal_graph, 'get_recent_nodes_hash'):
                return self.fractal_graph.get_recent_nodes_hash()
            
            # Fallback - используем время последнего изменения
            if hasattr(self.fractal_graph, '_last_update'):
                return str(self.fractal_graph._last_update)
            
            return str(id(self.fractal_graph))
            
        except Exception as e:
            logger.warning(f"Не удалось вычислить hash графа: {e}")
            return ""
    
    async def _get_knowledge_context(self) -> KnowledgeContext:
        """
        Получает контекст знаний из графа памяти.
        
        Returns:
            KnowledgeContext с концептами, противоречиями и виртуальными токенами
        """
        graph_hash = self._compute_graph_state_hash()
        
        # Проверяем кеш
        if graph_hash in self.knowledge_cache:
            logger.debug("Knowledge context cache hit")
            return self.knowledge_cache[graph_hash]
        
        context = KnowledgeContext(graph_hash=graph_hash)
        
        # 1. Извлекаем концепты из графа
        if self.fractal_graph:
            try:
                # Используем FractalGraphV2 API
                if hasattr(self.fractal_graph, 'get_context_for_query'):
                    # Для FGv2 используем get_context_for_query
                    knowledge_context_str = self.fractal_graph.get_context_for_query(
                        query="",  # Пустой запрос = все концепты
                        max_length=1000,
                        min_similarity=0.3
                    )
                    if knowledge_context_str:
                        context.virtual_tokens = knowledge_context_str
                
                # Получаем узлы для концептов
                if hasattr(self.fractal_graph, 'get_nodes_list'):
                    nodes = self.fractal_graph.get_nodes_list(limit=20)
                    # Фильтруем только concept nodes
                    concept_nodes = [n for n in nodes if getattr(n, 'node_type', '') == 'concept']
                    context.concepts = [
                        {"name": getattr(n, 'content', '')[:100], "description": getattr(n, 'description', '')}
                        for n in concept_nodes[:10]
                    ]
                    
            except Exception as e:
                logger.warning(f"Ошибка извлечения концептов из FractalGraphV2: {e}")
        
        # 2. Получаем противоречия
        if self.contradiction_manager:
            try:
                if hasattr(self.contradiction_manager, 'get_active_contradictions'):
                    contradictions = self.contradiction_manager.get_active_contradictions()
                    context.contradictions = [
                        {
                            "id": c.get('id', ''),
                            "title": c.get('title', ''),
                            "description": c.get('description', '')
                        }
                        for c in contradictions[:5]  # Максимум 5 противоречий
                    ]
            except Exception as e:
                logger.warning(f"Ошибка получения противоречий: {e}")
        
        # 3. Генерируем виртуальные токены
        context.virtual_tokens = self._generate_virtual_tokens(context)
        
        # Сохраняем в кеш
        self.knowledge_cache[graph_hash] = context
        
        # Ограничиваем размер кеша
        if len(self.knowledge_cache) > 20:
            # Удаляем старые записи (не сам hash, а по LRU)
            oldest = list(self.knowledge_cache.keys())[0]
            del self.knowledge_cache[oldest]
        
        return context
    
    def _generate_virtual_tokens(self, context: KnowledgeContext) -> str:
        """
        Генерирует строку виртуальных токенов из контекста знаний.
        
        Формат:
        <|knowledge|>
        Концепты: {список}
        Факты: {список}
        Противоречия: {список}
        </|knowledge|>
        """
        parts = [self.VIRTUAL_TOKEN_PLACEHOLDER + "\n"]
        
        if context.concepts:
            concept_names = [c['name'] for c in context.concepts[:5]]
            parts.append(f"Концепты: {', '.join(concept_names)}")
        
        if context.recent_facts:
            parts.append(f"Известные факты: {'; '.join(context.recent_facts[:3])}")
        
        if context.contradictions:
            contr_titles = [c['title'] for c in context.contradictions[:3]]
            parts.append(f"Противоречия для анализа: {', '.join(contr_titles)}")
        
        if len(parts) > 1:
            parts.append(f"</|knowledge|>")
            return "\n".join(parts)
        
        return ""
    
    # =========================================================================
    # Обработка ввода
    # =========================================================================
    
    async def process_user_input(
        self,
        user_input: str,
        system_prompt: str = None,
        extract_concepts: bool = True,
        **generation_kwargs
    ) -> GenerationResult:
        """
        Обрабатывает сообщение пользователя.
        
        Args:
            user_input: Текст сообщения
            system_prompt: Системный промпт (если не стандартный)
            extract_concepts: Извлекать концепты из ввода
            **generation_kwargs: Дополнительные параметры генерации
            
        Returns:
            GenerationResult с ответом и метаданными
        """
        start_time = time.time()
        cache_hit = False
        
        # 1. Добавляем сообщение пользователя
        self.add_message("user", user_input)
        
        # 2. Извлекаем концепты если нужно
        if extract_concepts and self.concept_extractor:
            try:
                # Создаём пустой контекст ответа
                temp_response = ""
                if hasattr(self.concept_extractor, 'extract_concepts'):
                    concepts = self.concept_extractor.extract_concepts(
                        user_input, temp_response
                    )
                    if concepts and self._on_concept_extracted:
                        self._on_concept_extracted(concepts)
            except Exception as e:
                logger.warning(f"Ошибка извлечения концептов: {e}")
        
        # 3. Получаем контекст знаний
        knowledge_context = await self._get_knowledge_context()
        
        # 4. Формируем промпт
        history = self.get_history()
        
        # Проверяем кеш для префикса
        prefix_text = self._format_chat_prompt(history, add_generation_prompt=True)
        cached = self.prefix_cache.get(prefix_text)
        
        if cached:
            cache_hit = True
            prompt = prefix_text
        else:
            prompt = prefix_text
            self.prefix_cache.set(prefix_text, True)
        
        # 5. Добавляем виртуальные токены знаний
        if knowledge_context.virtual_tokens:
            # Вставляем после system prompt но до истории
            prompt = self._insert_knowledge_tokens(prompt, knowledge_context.virtual_tokens)
        
        # 6. Генерируем ответ
        response_text = await self._generate_response(prompt, **generation_kwargs)
        
        # 7. Валидация если включена
        validated = True
        validation_notes = ""
        resolved_contradictions = []
        
        if self.enable_validation and self.contradiction_manager:
            validated, validation_notes, resolved_contradictions = \
                await self._validate_response(response_text, knowledge_context)
        
        # 8. Сохраняем ответ в историю
        if validated:
            self.add_message("assistant", response_text, {
                "knowledge_used": [c['name'] for c in knowledge_context.concepts],
                "validated": True
            })
        else:
            # Сохраняем с пометкой о невалидности
            self.add_message("assistant", response_text, {
                "validated": False,
                "validation_notes": validation_notes
            })
        
        # 9. Извлекаем концепты из ответа
        if extract_concepts and self.concept_extractor and validated:
            try:
                if hasattr(self.concept_extractor, 'extract_concepts'):
                    self.concept_extractor.extract_concepts(user_input, response_text)
            except Exception as e:
                logger.warning(f"Ошибка извлечения концептов из ответа: {e}")
        
        processing_time = time.time() - start_time
        
        # Извлекаем рассуждения из <think>...</think>
        reasoning, reasoning_steps = self._parse_think_tags(response_text)
        if reasoning:
            logger.info(f"[HYBRID] Extracted reasoning: {len(reasoning_steps) if reasoning_steps else 0} steps")
        
        return GenerationResult(
            response=response_text,
            validated=validated,
            validation_notes=validation_notes,
            knowledge_used=[c['name'] for c in knowledge_context.concepts],
            contradictions_resolved=resolved_contradictions,
            processing_time=processing_time,
            cache_hit=cache_hit,
            reasoning=reasoning,
            reasoning_steps=reasoning_steps
        )
    
    def _parse_think_tags(self, text: str) -> Tuple[Optional[str], Optional[List[Dict]]]:
        """Извлечь рассуждения из текста с <think>...</think> тегами."""
        import re
        
        reasoning = None
        reasoning_steps = None
        
        think_pattern = r'<think>\s*(.*?)\s*</think>'
        matches = re.findall(think_pattern, text, re.DOTALL)
        
        if matches:
            reasoning = '\n'.join(matches)
            lines = [l.strip() for l in reasoning.split('\n') if l.strip()]
            if len(lines) <= 1:
                sentences = re.split(r'(?<=[.!?])\s+', reasoning)
                lines = [s.strip() for s in sentences if s.strip()]
            
            reasoning_steps = []
            for i, step in enumerate(lines[:20]):
                reasoning_steps.append({
                    'step': i + 1,
                    'phase': 'reasoning',
                    'thought': step[:500],
                    'confidence': 0.8
                })
        
        return reasoning, reasoning_steps
    
    def _insert_knowledge_tokens(self, prompt: str, knowledge_tokens: str) -> str:
        """
        Вставляет виртуальные токены знаний в промпт.
        
        Вставляет после system prompt или в начало если system prompt не найден.
        """
        # Ищем позицию после system prompt
        system_marker = "Система:"
        knowledge_marker = self.VIRTUAL_TOKEN_PLACEHOLDER
        
        if knowledge_marker in prompt:
            # Уже есть, не добавляем
            return prompt
        
        if system_marker in prompt:
            # Вставляем после system prompt
            idx = prompt.find(system_marker)
            # Находим конец первого параграфа system
            end_idx = prompt.find("\n\n", idx)
            if end_idx > 0:
                return prompt[:end_idx] + "\n\n" + knowledge_tokens + "\n" + prompt[end_idx:]
        
        # Вставляем в начало
        return knowledge_tokens + "\n\n" + prompt
    
    async def _generate_response(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> str:
        """
        Генерирует ответ через OpenVINO pipeline.
        
        Args:
            prompt: Промпт для генерации
            max_tokens: Максимум токенов
            temperature: Температура
            
        Returns:
            Сгенерированный текст
        """
        if not self._initialized or not self._pipeline:
            logger.error("Pipeline не инициализирован")
            return "[Ошибка: система не готова]"
        
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        
        try:
            # Настройка генерации
            config = ov_genai.GenerationConfig()
            config.max_new_tokens = max_tokens
            config.temperature = temperature
            config.do_sample = temperature > 0
            
            # Генерация в отдельном потоке
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                partial(self._pipeline.generate, prompt, config)
            )
            
            return result.strip()
            
        except Exception as e:
            logger.error(f"Ошибка генерации: {e}")
            return f"[Ошибка генерации: {str(e)[:100]}]"
    
    async def _validate_response(
        self,
        response: str,
        knowledge_context: KnowledgeContext
    ) -> Tuple[bool, str, List[str]]:
        """
        Валидирует ответ на противоречия.
        
        Returns:
            (валидность, заметки, разрешённые противоречия)
        """
        if not self.contradiction_manager:
            return True, "", []
        
        try:
            # Проверяем на противоречия с известными фактами
            if knowledge_context.recent_facts:
                # Простой check - ищем противоречивые утверждения
                contradictions_found = []
                
                for fact in knowledge_context.recent_facts[:3]:
                    if self._check_contradiction(response, fact):
                        contradictions_found.append(fact)
                
                if contradictions_found:
                    return False, f"Противоречие с известными фактами: {contradictions_found[0]}", []
            
            return True, "", []
            
        except Exception as e:
            logger.warning(f"Ошибка валидации: {e}")
            return True, "", []
    
    def _check_contradiction(self, response: str, fact: str) -> bool:
        """
        Проверяет противоречие между ответом и фактом.
        
        Упрощённая проверка - в реальности нужна NLI модель.
        """
        # Ищем явные отрицания
        response_lower = response.lower()
        fact_lower = fact.lower()
        
        # Проверяем на прямое противоречие (упрощённо)
        negation_words = ['не', 'нет', 'никогда', 'невозможно', 'ложь']
        
        for neg in negation_words:
            if neg in response_lower and fact_lower[:50] in response_lower:
                return True
        
        return False
    
# =========================================================================
    # Streaming интерфейс
    # ============================================

    def _analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """Анализирует сложность запроса и определяет стратегию генерации."""
        query_lower = query.lower()
        
        # Признаки сложного запроса
        complex_indicators = [
            'почему', 'как работает', 'объясни', 'подробно', 'опиши',
            'сравни', 'проанализируй', 'рассмотри', 'исследуй',
            ' что ', ' как ', 'почему ', ' зачем ',
            '?', '??', '???'
        ]
        
        # Признаки простого запроса
        simple_indicators = [
            'да', 'нет', 'привет', 'hi', 'hello', 'спасибо',
            'сколько', 'когда', 'где', ' кто '
        ]
        
        # Код/факты
        code_indicators = ['код', 'программ', 'функци', 'def ', 'class ', 'import ']
        
        complex_score = sum(1 for ind in complex_indicators if ind in query_lower)
        simple_score = sum(1 for ind in simple_indicators if ind in query_lower)
        code_score = sum(1 for ind in code_indicators if ind in query_lower)
        
        # Определяем тип и стратегию
        if code_score > 0:
            return {'mode': 'code', 'use_dual': False, 'max_tokens': 1024, 'lora': 'eva_code'}
        elif complex_score > simple_score + 1:
            return {'mode': 'complex', 'use_dual': True, 'max_tokens': 2048, 'lora': 'eva_knowledge'}
        elif simple_score > complex_score:
            return {'mode': 'simple', 'use_dual': False, 'max_tokens': 512, 'lora': 'eva_logic'}
        else:
            # По умолчанию - средняя сложность
            return {'mode': 'auto', 'use_dual': len(query) > 100, 'max_tokens': 1024, 'lora': 'eva_knowledge'}
    
    def _get_generator_for_model(self, model_id: str):
        """Получает генератор для указанной модели."""
        if model_id == 'A':
            if self._pipeline:
                return self._pipeline
        elif model_id == 'B':
            if self._pipeline_b and self._pipeline_b != self._pipeline:
                return self._pipeline_b
        return self._pipeline
    
    def _apply_lora_to_generator(self, pipeline, lora_name: str):
        """Применяет LoRA к генератору."""
        if not pipeline or not lora_name:
            return
        
        # Получаем генератор из brain для применения LoRA
        if self.brain and hasattr(self.brain, 'two_model_pipeline'):
            pipeline_obj = self.brain.two_model_pipeline
            try:
                if hasattr(pipeline_obj, '_openvino_cpu') and pipeline_obj._openvino_cpu:
                    pipeline_obj._openvino_cpu.set_active_lora(lora_name)
                if hasattr(pipeline_obj, '_openvino_gpu') and pipeline_obj._openvino_gpu:
                    pipeline_obj._openvino_gpu.set_active_lora(lora_name)
            except Exception as e:
                logger.warning(f"LoRA application error: {e}")
    
    def generate_streaming(
        self,
        user_input: str,
        system_prompt: str = None,
        max_tokens: int = None,
        temperature: float = None,
        chunk_size: int = 25,
        force_mode: str = None
    ):
        """
        Генерация со стримингом.
        
        Автоматически определяет:
        - Простой запрос → Model A only
        - Сложный запрос → Model A + Model B последовательно
        - Код → Model A with eva_code LoRA
        
        Events:
        - model_start: начало генерации (A или B)
        - reasoning_start/text/end: рассуждения
        - chunk: основной текст
        - complete: завершение
        """
        if not self._initialized or not self._pipeline:
            yield {'type': 'error', 'text': 'Система не готова', 'is_final': True}
            return
        
        start_time = time.time()
        
        # Анализируем сложность запроса
        strategy = self._analyze_query_complexity(user_input)
        if force_mode:
            strategy['mode'] = force_mode
            strategy['use_dual'] = (force_mode == 'extended')
        
        use_dual = strategy['use_dual']
        lora_name = strategy.get('lora')
        max_tokens = max_tokens or strategy.get('max_tokens', self.max_tokens)
        temperature = temperature or self.temperature
        
        logger.info(f"[STREAM] mode={strategy['mode']}, use_dual={use_dual}, lora={lora_name}")
        
        try:
            self.add_message("user", user_input)
            
            history = self.get_history()
            prompt = self._format_chat_prompt(history, add_generation_prompt=True)
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            knowledge_context = loop.run_until_complete(self._get_knowledge_context())
            loop.close()
            
            if knowledge_context.virtual_tokens:
                prompt = self._insert_knowledge_tokens(prompt, knowledge_context.virtual_tokens)
            
            import queue
            token_queue = queue.Queue()
            full_text = []
            full_reasoning = ""
            
            # === MODEL A: Быстрый ответ ===
            yield {'type': 'model_start', 'model': 'A', 'lora': lora_name, 'is_final': False}
            
            # Применяем LoRA
            self._apply_lora_to_generator(self._pipeline, lora_name)
            
            config_a = ov_genai.GenerationConfig()
            config_a.max_new_tokens = max_tokens // 2 if use_dual else max_tokens
            config_a.temperature = temperature
            config_a.do_sample = temperature > 0
            
            def streamer_a(text: str) -> bool:
                token_queue.put(('a', text))
                return False
            
            def generate_model_a():
                try:
                    self._pipeline.generate(prompt, config_a, streamer=streamer_a)
                except Exception as e:
                    token_queue.put(('error', str(e)))
                finally:
                    token_queue.put(('done_a', None))
            
            thread_a = threading.Thread(target=generate_model_a, daemon=True)
            thread_a.start()
            
            in_thinking = False
            text_buffer = ""
            model_a_done = False
            
            # Читаем ответ от Model A
            while not model_a_done:
                try:
                    item = token_queue.get(timeout=60)
                except queue.Empty:
                    if not thread_a.is_alive():
                        break
                    continue
                
                if item is None or item[0] in ('done_a', 'done_b'):
                    model_a_done = True
                    break
                
                if item[0] == 'error':
                    yield {'type': 'error', 'text': item[1], 'is_final': True}
                    return
                
                combined = item[1]
                full_text.append(combined)
                
                # Обработка рассуждений
                if not in_thinking and '<think>' in combined:
                    in_thinking = True
                    before_think = combined.split('<think>')[0]
                    if before_think:
                        text_buffer += before_think
                        if len(text_buffer) >= chunk_size:
                            yield {'type': 'chunk', 'text': text_buffer, 'is_final': False}
                            text_buffer = ""
                    
                    yield {'type': 'reasoning_start', 'is_final': False}
                    after_think = combined.split('<think>')[1] if '<think>' in combined else ''
                    
                    if '</think>' in after_think:
                        reasoning_content = after_think.split('</think>')[0]
                        full_reasoning += reasoning_content
                        yield {'type': 'reasoning_text', 'text': reasoning_content, 'is_final': False}
                        yield {'type': 'reasoning_end', 'is_final': False, 'full_text': full_reasoning}
                        in_thinking = False
                    else:
                        full_reasoning += after_think
                        yield {'type': 'reasoning_text', 'text': after_think, 'is_final': False}
                
                elif in_thinking:
                    if '</think>' in combined:
                        reasoning_content = combined.split('</think>')[0]
                        full_reasoning += reasoning_content
                        yield {'type': 'reasoning_text', 'text': reasoning_content, 'is_final': False}
                        yield {'type': 'reasoning_end', 'is_final': False, 'full_text': full_reasoning}
                        in_thinking = False
                    else:
                        full_reasoning += combined
                        yield {'type': 'reasoning_text', 'text': combined, 'is_final': False}
                else:
                    text_buffer += combined
                    if len(text_buffer) >= chunk_size:
                        yield {'type': 'chunk', 'text': text_buffer, 'is_final': False}
                        text_buffer = ""
            
            if text_buffer:
                yield {'type': 'chunk', 'text': text_buffer, 'is_final': False}
            
            response_a = ''.join(full_text)
            yield {
                'type': 'model_complete',
                'model': 'A',
                'text': response_a,
                'is_final': False,
                'reasoning': full_reasoning
            }
            
            # === MODEL B: Расширенный ответ (если нужно) ===
            model_b_success = False
            response_a_backup = response_a  # Сохраняем ответ A как резерв
            
            # ОЧИЩАЕМ full_text для Model B - иначе будет дубликат!
            full_text.clear()
            
            if use_dual and self._pipeline_b and self._pipeline_b != self._pipeline:
                yield {'type': 'model_start', 'model': 'B', 'lora': lora_name, 'is_final': False}
                
                try:
                    # Продолжаем с ответом от A как контекстом
                    extended_prompt = f"{prompt}\n\nКраткий ответ: {response_a}\n\nДай развёрнутый и подробный ответ:"
                    
                    config_b = ov_genai.GenerationConfig()
                    config_b.max_new_tokens = max_tokens
                    config_b.temperature = temperature
                    config_b.do_sample = temperature > 0
                    
                    in_thinking = False
                    text_buffer = ""
                    full_reasoning = ""  # Очищаем reasoning для Model B
                    
                    def streamer_b(text: str) -> bool:
                        token_queue.put(('b', text))
                        return False
                    
                    def generate_model_b():
                        try:
                            self._pipeline_b.generate(extended_prompt, config_b, streamer=streamer_b)
                        except Exception as e:
                            token_queue.put(('error', str(e)))
                        finally:
                            token_queue.put(('done_b', None))
                    
                    thread_b = threading.Thread(target=generate_model_b, daemon=True)
                    thread_b.start()
                    
                    while True:
                        try:
                            item = token_queue.get(timeout=60)
                        except queue.Empty:
                            if not thread_b.is_alive():
                                break
                            continue
                        
                        if item is None or item[0] in ('done_a', 'done_b'):
                            break
                        
                        if item[0] == 'error':
                            logger.error(f"Model B generation error: {item[1]}")
                            break
                        
                        combined = item[1]
                        full_text.append(combined)
                        model_b_success = True
                        
                        if not in_thinking and '<think>' in combined:
                            in_thinking = True
                            yield {'type': 'reasoning_start', 'is_final': False}
                            after_think = combined.split('<think>')[1] if '<think>' in combined else ''
                            
                            if '</think>' in after_think:
                                reasoning_content = after_think.split('</think>')[0]
                                full_reasoning += reasoning_content
                                yield {'type': 'reasoning_text', 'text': reasoning_content, 'is_final': False}
                                yield {'type': 'reasoning_end', 'is_final': False, 'full_text': full_reasoning}
                                in_thinking = False
                            else:
                                full_reasoning += after_think
                                yield {'type': 'reasoning_text', 'text': after_think, 'is_final': False}
                        
                        elif in_thinking:
                            if '</think>' in combined:
                                reasoning_content = combined.split('</think>')[0]
                                full_reasoning += reasoning_content
                                yield {'type': 'reasoning_text', 'text': reasoning_content, 'is_final': False}
                                yield {'type': 'reasoning_end', 'is_final': False, 'full_text': full_reasoning}
                                in_thinking = False
                            else:
                                full_reasoning += combined
                                yield {'type': 'reasoning_text', 'text': combined, 'is_final': False}
                        
                        else:
                            text_buffer += combined
                            if len(text_buffer) >= chunk_size:
                                yield {'type': 'chunk', 'text': text_buffer, 'is_final': False}
                                text_buffer = ""
                    
                    if text_buffer:
                        yield {'type': 'chunk', 'text': text_buffer, 'is_final': False}
                        
                except Exception as e:
                    logger.error(f"Model B failed: {e}")
                    model_b_success = False
            
            # Если Model B не удалась - восстанавливаем ответ A
            if not model_b_success:
                logger.info("Using Model A response (Model B failed)")
                full_text.clear()
                full_text.append(response_a_backup)
                full_reasoning = ""  # Очищаем reasoning от B
            
            full_response = ''.join(full_text)
            yield {
                'type': 'complete',
                'text': full_response,
                'is_final': True,
                'tokens_count': len(full_response.split()),
                'elapsed_ms': int((time.time() - start_time) * 1000),
                'reasoning': full_reasoning,
                'mode': strategy['mode'],
                'lora': lora_name
}
            
            self.add_message("assistant", full_response)
        
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield {'type': 'error', 'text': str(e), 'is_final': True}
    
    def _split_text_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Разбивает текст на чанки для стриминга."""
        if len(text) <= chunk_size:
            return [text] if text else []
        
        chunks = []
        remaining = text
        delimiters = '.!?\n'
        
        while remaining:
            chunk_end = min(chunk_size, len(remaining))
            
            if chunk_end < len(remaining):
                found = False
                for delim in delimiters:
                    pos = remaining[:chunk_end].rfind(delim)
                    if pos > chunk_size // 2:
                        chunk_end = pos + 1
                        found = True
                        break
                
                if not found:
                    space_pos = remaining[:chunk_end].rfind(' ')
                    if space_pos > chunk_size // 3:
                        chunk_end = space_pos + 1
            
            chunk = remaining[:chunk_end].strip()
            if chunk:
                chunks.append(chunk)
            
            remaining = remaining[chunk_end:].lstrip()
        
        return chunks if chunks else [text]
    
    # =========================================================================
    # Синхронный интерфейс (для обратной совместимости)
    # =========================================================================
    
    def process(self, user_input: str, **kwargs) -> Dict:
        """
        Синхронная обёртка для process_user_input.
        
        Используется для обратной совместимости с существующим кодом.
        Возвращает Dict с response, reasoning и reasoning_steps.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(
                self.process_user_input(user_input, **kwargs)
            )
            
            return {
                'response': result.response,
                'reasoning': result.reasoning,
                'reasoning_steps': result.reasoning_steps,
                'validated': result.validated,
                'knowledge_used': result.knowledge_used,
                'processing_time': result.processing_time
            }
            
        except Exception as e:
            logger.error(f"Ошибка синхронной обработки: {e}")
            return {
                'response': f"[Ошибка: {str(e)[:100]}]",
                'reasoning': None,
                'reasoning_steps': None
            }
    
    # =========================================================================
    # Управление сессией
    # =========================================================================
    
    def reset_session(self):
        """Сбрасывает сессию но сохраняет системный промпт."""
        self.clear_history()
        self.prefix_cache.clear()
        self.knowledge_cache.clear()
        logger.info("Сессия сброшена")
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику менеджера."""
        return {
            "initialized": self._initialized,
            "messages_count": len(self.messages),
            "cache_stats": self.prefix_cache.get_stats(),
            "knowledge_cache_size": len(self.knowledge_cache),
            "device": self.device,
            "enable_validation": self.enable_validation
        }
    
    def shutdown(self):
        """Корректное завершение."""
        self._initialized = False
        self._pipeline = None
        self.prefix_cache.clear()
        self.knowledge_cache.clear()
        logger.info("HybridKnowledgeDialogManager завершён")


def create_hybrid_dialog_manager(
    brain=None,
    fractal_graph=None,
    concept_extractor=None,
    contradiction_manager=None,
    model_path: str = None,
    model_b_path: str = None,
    device: str = "CPU",
    **kwargs
) -> HybridKnowledgeDialogManager:
    """
    Фабричная функция для создания HybridKnowledgeDialogManager с двумя моделями.
    """
    manager = HybridKnowledgeDialogManager(
        brain=brain,
        fractal_graph=fractal_graph,
        concept_extractor=concept_extractor,
        contradiction_manager=contradiction_manager,
        model_path=model_path,
        model_b_path=model_b_path,
        device=device,
        **kwargs
    )
    
    if model_path:
        manager.initialize(model_path, model_b_path, device)
    
    return manager
