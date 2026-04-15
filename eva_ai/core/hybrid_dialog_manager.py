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
    
    Параметры:
        brain: CoreBrain - ядро системы
        fractal_graph: FractalMemoryGraph - граф памяти
        concept_extractor: ConceptExtractor - извлекатель концептов
        contradiction_manager: ContradictionManager - менеджер противоречий
        model_path: str - путь к модели GGUF
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
        
        # Pipeline
        self._pipeline = None
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
    
    def initialize(self, model_path: str = None, device: str = None):
        """
        Инициализация OpenVINO pipeline и токенизатора.
        
        Args:
            model_path: Путь к модели (если не указан при создании)
            device: Устройство (если не указано при создании)
        """
        if not OPENVINO_AVAILABLE:
            logger.error("OpenVINO GenAI не установлен")
            return False
        
        if model_path:
            self.model_path = model_path
        if device:
            self.device = device
        
        if not self.model_path:
            logger.error("model_path не указан")
            return False
        
        if not os.path.exists(self.model_path):
            logger.error(f"Модель не найдена: {self.model_path}")
            return False
        
        try:
            # Настройка планировщика с prefix caching
            self._scheduler_config = ov_genai.SchedulerConfig()
            self._scheduler_config.enable_prefix_caching = True
            self._scheduler_config.cache_size = 2  # GB
            self._scheduler_config.max_num_seqs = 4
            self._scheduler_config.max_num_batched_tokens = 2048
            
            # Создание pipeline
            config = {"scheduler_config": self._scheduler_config}
            self._pipeline = ov_genai.LLMPipeline(self.model_path, self.device, config=config)
            
            # Токенизатор (используем chat template из модели)
            self._tokenizer = self._pipeline.get_tokenizer()
            
            self._initialized = True
            logger.info(f"Pipeline инициализирован: {self.model_path} на {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка инициализации pipeline: {e}")
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
                # Получаем недавние концепты
                if hasattr(self.fractal_graph, 'get_recent_concepts'):
                    recent_concepts = self.fractal_graph.get_recent_concepts(limit=10)
                    context.concepts = [
                        {"name": c.get('name', ''), "description": c.get('description', '')}
                        for c in recent_concepts
                    ]
                
                # Получаем факты
                if hasattr(self.fractal_graph, 'get_recent_facts'):
                    context.recent_facts = self.fractal_graph.get_recent_facts(limit=5)
                    
            except Exception as e:
                logger.warning(f"Ошибка извлечения концептов: {e}")
        
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
        
        return GenerationResult(
            response=response_text,
            validated=validated,
            validation_notes=validation_notes,
            knowledge_used=[c['name'] for c in knowledge_context.concepts],
            contradictions_resolved=resolved_contradictions,
            processing_time=processing_time,
            cache_hit=cache_hit
        )
    
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
    # =========================================================================
    
    def generate_streaming(
        self,
        user_input: str,
        system_prompt: str = None,
        max_tokens: int = None,
        temperature: float = None,
        chunk_size: int = 25
    ):
        """
        Генерация со стримингом для SSE - отправляет чанки ПО МЕРЕ генерации.
        
        Args:
            user_input: Текст сообщения пользователя
            system_prompt: Системный промпт (если не стандартный)
            max_tokens: Максимум токенов
            temperature: Температура генерации
            chunk_size: Размер чанка для буферизации
            
        Yields:
            Dict с данными чанка для SSE
        """
        if not self._initialized or not self._pipeline:
            yield {'type': 'error', 'text': 'Система не готова', 'is_final': True}
            return
        
        start_time = time.time()
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        
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
            
            config = ov_genai.GenerationConfig()
            config.max_new_tokens = max_tokens
            config.temperature = temperature
            config.do_sample = temperature > 0
            
            buffer = ""
            chunk_count = 0
            full_text = []
            
            import queue
            token_queue = queue.Queue()
            is_complete = [False]
            error_msg = [None]
            
            def streamer(text: str) -> bool:
                """Callback вызывается для каждого токена - отправляем сразу."""
                full_text.append(text)
                token_queue.put(text)
                return False
            
            def generate():
                try:
                    self._pipeline.generate(prompt, config, streamer=streamer)
                except Exception as e:
                    error_msg[0] = str(e)
                    token_queue.put(('error', str(e)))
                finally:
                    is_complete[0] = True
                    token_queue.put(None)
            
            thread = threading.Thread(target=generate, daemon=True)
            thread.start()
            
            while True:
                try:
                    item = token_queue.get(timeout=60)
                except queue.Empty:
                    if not thread.is_alive():
                        break
                    continue
                
                if item is None:
                    break
                
                if isinstance(item, tuple) and item[0] == 'error':
                    yield {'type': 'error', 'text': item[1], 'is_final': True}
                    return
                
                buffer += item
                
                if len(buffer) >= chunk_size:
                    elapsed = int((time.time() - start_time) * 1000)
                    yield {
                        'type': 'chunk',
                        'text': buffer,
                        'is_final': False,
                        'tokens_count': len(''.join(full_text)),
                        'elapsed_ms': elapsed
                    }
                    buffer = ""
                    chunk_count += 1
            
            if buffer:
                yield {
                    'type': 'chunk',
                    'text': buffer,
                    'is_final': False,
                    'tokens_count': len(''.join(full_text)),
                    'elapsed_ms': int((time.time() - start_time) * 1000)
                }
            
            full_response = ''.join(full_text)
            yield {
                'type': 'complete',
                'text': full_response,
                'is_final': True,
                'tokens_count': len(full_response.split()),
                'elapsed_ms': int((time.time() - start_time) * 1000)
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
    
    def process(self, user_input: str, **kwargs) -> str:
        """
        Синхронная обёртка для process_user_input.
        
        Используется для обратной совместимости с существующим кодом.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Создаём новый loop если текущий занят
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(
                self.process_user_input(user_input, **kwargs)
            )
            return result.response
            
        except Exception as e:
            logger.error(f"Ошибка синхронной обработки: {e}")
            return f"[Ошибка: {str(e)[:100]}]"
    
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
    device: str = "CPU",
    **kwargs
) -> HybridKnowledgeDialogManager:
    """
    Фабричная функция для создания HybridKnowledgeDialogManager.
    
    Args:
        brain: CoreBrain системы
        fractal_graph: FractalMemoryGraph
        concept_extractor: ConceptExtractor
        contradiction_manager: ContradictionManager
        model_path: Путь к GGUF модели
        device: Устройство (CPU/GPU)
        
    Returns:
        Инициализированный HybridKnowledgeDialogManager
    """
    manager = HybridKnowledgeDialogManager(
        brain=brain,
        fractal_graph=fractal_graph,
        concept_extractor=concept_extractor,
        contradiction_manager=contradiction_manager,
        model_path=model_path,
        device=device,
        **kwargs
    )
    
    # Инициализируем если есть путь к модели
    if model_path:
        manager.initialize(model_path, device)
    
    return manager
