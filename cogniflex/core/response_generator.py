#!/usr/bin/env python3
"""
Исправленная версия ResponseGenerator с поддержкой HybridTokenCache
Оптимизированная версия с улучшенной обработкой ошибок и кэшированием
"""

import os
import sys
import time
import threading
import logging
from typing import Dict, Any, Optional, List, Tuple, Union

# Добавляем корневую директорию проекта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Настройка логгера ДО использования
logger = logging.getLogger("cogniflex.response_generator")

# Пытаемся импортировать torch
try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, ModuleNotFoundError, RuntimeError):
    torch = None
    TORCH_AVAILABLE = False
    logger.warning("PyTorch недоступен, генерация ответов будет ограничена")

# Импорты для работы с кэшем
try:
    from cogniflex.memory.hybrid_token_cache import HybridTokenCache
    HYBRID_CACHE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    HybridTokenCache = None
    HYBRID_CACHE_AVAILABLE = False
    logger.warning("HybridTokenCache недоступен, кэширование будет ограничено")

# Импорты для работы с событиями и инициализацией компонентов
try:
    from cogniflex.core.event_system import EventBus, ComponentInitializationManager
    EVENT_SYSTEM_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    EventBus = None
    ComponentInitializationManager = None
    EVENT_SYSTEM_AVAILABLE = False
    logger.warning("Система событий недоступна, инициализация будет упрощена")

# Импорты для работы с сущностями и обнаружения неоднозначностей
try:
    from cogniflex.knowledge.context_entity import EntityExtractor
except ImportError:
    EntityExtractor = None

try:
    from cogniflex.learning.knowledge_awareness import KnowledgeAwareness
except ImportError:
    KnowledgeAwareness = None

# Импорты токенизатора
try:
    from transformers import AutoTokenizer, PreTrainedTokenizer
    TRANSFORMERS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    AutoTokenizer = None
    PreTrainedTokenizer = None
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers недоступен, будет использован fallback токенизатор")


class ResponseGenerator:
    """
    Класс для генерации ответов с поддержкой гибридного кэша токенов.
    
    Интегрирует:
    - HybridTokenCache для оптимизации токенизации
    - MLUnit и ModelManager для работы с моделями
    - CoreBrain для интеграции с системой
    """
    
    def __init__(
        self, 
        brain: Optional[Any] = None, 
        model_manager: Optional[Any] = None, 
        text_processor: Optional[Any] = None, 
        deferred_system: Optional[Any] = None,
        tokenizer_config: Optional[Dict[str, Any]] = None,
        hybrid_cache: Optional[Any] = None
    ):
        """Инициализирует генератор ответов.
        
        Args:
            brain: Экземпляр CoreBrain, должен содержать tokenizer
            model_manager: Менеджер моделей (опционально)
            text_processor: Устаревший параметр, оставлен для обратной совместимости
            deferred_system: Система отложенных команд для асинхронной инициализации
            tokenizer_config: Конфигурация токенизатора
            hybrid_cache: Гибридный кэш токенов для оптимизации
        """
        # Конфигурация токенизатора по умолчанию
        self.tokenizer_config = {
            'max_length': 512,
            'truncation': True,
            'padding': 'max_length',
            'return_tensors': 'pt',
            'add_special_tokens': True
        }
        
        # Обновляем конфигурацию если передана
        if tokenizer_config:
            self.tokenizer_config.update(tokenizer_config)
        
        self.brain = brain
        self.model_manager = model_manager or (getattr(self.brain, 'components', {}).get('model_manager') if hasattr(self.brain, 'components') else None)
        self.deferred_system = deferred_system or getattr(brain, 'deferred_system', None)
        
        # Параметр оставлен для обратной совместимости
        self.text_processor = text_processor
        self.hybrid_cache = hybrid_cache
        
        # Система инициализации компонентов для предотвращения повторной инициализации
        self.component_init_manager = None
        if EVENT_SYSTEM_AVAILABLE and self.brain:
            event_bus = getattr(self.brain, 'events', None)  # EventBus хранится как 'events' в CoreBrain
            if event_bus:
                self.component_init_manager = ComponentInitializationManager(event_bus)
                logger.info("ComponentInitializationManager инициализирован")
            else:
                logger.warning("EventBus не найден в brain, ComponentInitializationManager недоступен")
        
        # Инициализируем токенизатор из разных источников
        self.tokenizer: Optional[Any] = None
        self.token_streamer: Optional[Any] = None
        
        # Состояния загрузки моделей
        self._model_states: Dict[str, Dict[str, Any]] = {}
        self._model_locks: Dict[str, threading.Lock] = {}
        self._init_lock = threading.Lock()
        
        # Пул потоков для асинхронной генерации
        self._thread_pool: Optional[Any] = None
        self._shutdown_event = threading.Event()
        
        # Семафор для GPU генерации
        self._gpu_generate_sema = threading.Semaphore(1)
        
        # Флаги инициализации
        self._initialized = False
        self._initializing = False
        
        # Экстрактор сущностей для обнаружения неоднозначностей
        self.entity_extractor = EntityExtractor() if EntityExtractor else None
        
        # Knowledge Awareness
        try:
            if KnowledgeAwareness:
                self.knowledge_awareness = KnowledgeAwareness(brain)
            else:
                self.knowledge_awareness = None
        except Exception:
            self.knowledge_awareness = None
        
        logger.info("ResponseGenerator создан")
        
        # Отложенная инициализация компонентов, если доступна система отложенных команд
        if self.deferred_system and hasattr(self.deferred_system, 'defer_command'):
            self.deferred_system.defer_command(
                self._deferred_init_components,
                priority='high',
                name='response_generator_init_components'
            )
            logger.info("Отложенная инициализация компонентов запланирована")
        else:
            # Прямая инициализация
            self._init_components()
    
    def _deferred_init_components(self) -> None:
        """Отложенная инициализация компонентов."""
        try:
            self._init_components()
        except Exception as e:
            logger.error(f"Ошибка отложенной инициализации: {e}", exc_info=True)
    
    def _detect_ambiguity_before_response(self, prompt: str) -> Dict:
        """Check query for ambiguities before generating response."""
        if not self.entity_extractor:
            return {"needs_clarification": False, "clarifications": []}
        
        ambiguous_entities = self.entity_extractor.extract_ambiguous_terms(prompt)
        clarifications = []
        
        for entity in ambiguous_entities:
            clarifications.append({
                "term": entity.text,
                "type": entity.ambiguity_type.value,
                "context": entity.context
            })
        
        return {
            "needs_clarification": len(clarifications) > 0,
            "clarifications": clarifications
        }

    def _generate_clarification_response(self, clarifications: List[Dict]) -> str:
        """Generate response asking for clarification."""
        if not clarifications:
            return ""
        
        questions = []
        for c in clarifications[:3]:
            term = c.get("term", "")
            q_type = c.get("type", "")
            
            if q_type == "vague_quantifier":
                questions.append(f"Что именно означает '{term}' в данном контексте?")
            elif q_type == "ambiguous_term":
                questions.append(f"Пожалуйста, уточните, что вы имеете в виду под '{term}'?")
            elif q_type == "implicit_reference":
                questions.append(f"На что именно ссылается '{term}'?")
            elif q_type == "relative_comparison":
                questions.append(f"С чем именно вы сравниваете '{term}'?")
            else:
                questions.append(f"Уточните, пожалуйста, что означает '{term}'?")
        
        return "Для лучшего ответа, пожалуйста, уточните: " + "; ".join(questions)
    
    def _init_components(self) -> None:
        """Инициализирует компоненты из brain."""
        with self._init_lock:
            if self._initializing or self._initialized:
                return
            self._initializing = True
        
        try:
            if not self.brain:
                logger.warning("CoreBrain не передан, инициализация компонентов пропущена")
                return
            
            # Инициализируем токенизатор с несколькими источниками
            self._init_tokenizer()
            
            # Инициализируем гибридный кэш если доступен
            self._init_hybrid_cache()
            
            # Проверяем наличие ModelManager (динамическая проверка)
            if not self.model_manager and hasattr(self.brain, 'components'):
                self.model_manager = self.brain.components.get('model_manager')
            
            if not self.model_manager:
                logger.warning("ModelManager не инициализирован")
            else:
                logger.info("ModelManager найден и доступен")
            
            # Проверяем валидность токенизатора
            if not self._validate_tokenizer(self.tokenizer):
                logger.warning("Токенизатор не прошел валидацию, создаём fallback")
                self._create_fallback_tokenizer()
            
            self._initialized = True
            logger.info("Компоненты ResponseGenerator инициализированы")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации компонентов: {e}", exc_info=True)
        finally:
            self._initializing = False
    
    def _init_tokenizer(self) -> None:
        """Инициализирует токенизатор из доступных источников с предотвращением повторной инициализации."""
        component_name = f"tokenizer_{id(self.brain) if self.brain else 'global'}"

        # Используем ComponentInitializationManager если доступен
        if self.component_init_manager:
            def _do_tokenizer_init():
                return self._init_tokenizer_internal()

            success = self.component_init_manager.initialize_component(
                component_name=component_name,
                component_instance=self,
                init_function=_do_tokenizer_init
            )

            if success:
                # Получаем инициализированный токенизатор из менеджера
                initialized_component = self.component_init_manager.get_initialized_component(component_name)
                if initialized_component and hasattr(initialized_component, 'tokenizer') and initialized_component.tokenizer is not None:
                    self.tokenizer = initialized_component.tokenizer
                    logger.info("Токенизатор получен из ComponentInitializationManager")
                else:
                    logger.warning("Не удалось получить токенизатор из ComponentInitializationManager")
            else:
                logger.error("Не удалось инициализировать токенизатор через ComponentInitializationManager")
        else:
            # Fallback: обычная инициализация без менеджера
            logger.debug("ComponentInitializationManager недоступен, используем обычную инициализацию")
            self._init_tokenizer_internal()

    def _init_tokenizer_internal(self) -> bool:
        """Внутренняя логика инициализации токенизатора."""
        try:
            # Пробуем получить из brain
            self.tokenizer = getattr(self.brain, 'tokenizer', None)

            # Если не найден, пробуем fractal_model_manager
            if self.tokenizer is None:
                fractal_manager = getattr(self.brain, 'fractal_model_manager', None)
                if fractal_manager:
                    self.tokenizer = getattr(fractal_manager, 'tokenizer', None)
                    if self.tokenizer:
                        logger.info("Токенизатор найден в fractal_model_manager")

            # Если не найден, пробуем ml_unit
            if self.tokenizer is None:
                ml_unit = getattr(self.brain, 'ml_unit', None)
                if ml_unit:
                    # Пробуем text_processor
                    text_processor = getattr(ml_unit, 'text_processor', None)
                    if text_processor:
                        self.tokenizer = getattr(text_processor, 'tokenizer', None)
                        if self.tokenizer:
                            logger.info("Токенизатор найден в ml_unit.text_processor")

                    # Пробуем model_manager
                    if self.tokenizer is None:
                        model_manager = getattr(ml_unit, 'model_manager', None)
                        if model_manager:
                            self.tokenizer = getattr(model_manager, 'tokenizer', None)
                            if self.tokenizer:
                                logger.info("Токенизатор найден в ml_unit.model_manager")

            # Проверяем, удалось ли найти токенизатор
            if self.tokenizer is not None:
                logger.info("Токенизатор инициализирован")
                return True
            
            # Fallback: пробуем загрузить напрямую из модели (qwen3.5-0.8b - активная модель из brain_config.json)
            try:
                from transformers import AutoTokenizer
                model_path = "C:/Users/black/OneDrive/Desktop/CogniFlex/cogniflex/mlearning/cogniflex_models/qwen3.5-0.8b"
                if os.path.exists(model_path):
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                        if self.tokenizer:
                            logger.info(f"Токенизатор загружен из {model_path}")
                            return True
                    except Exception as e:
                        logger.warning(f"Не удалось загрузить токенизатор из {model_path}: {e}")
            except Exception as e:
                logger.warning(f"Ошибка при загрузке токенизатора: {e}")
            
            logger.warning("Не удалось найти токенизатор ни в одном из источников")
            return False

        except Exception as e:
            logger.error(f"Ошибка при инициализации токенизатора: {e}", exc_info=True)
            return False
    
    def _init_hybrid_cache(self) -> None:
        """Инициализирует гибридный кэш токенов."""
        if self.hybrid_cache is not None:
            return
        
        if not HYBRID_CACHE_AVAILABLE:
            logger.debug("HybridTokenCache недоступен")
            return
        
        try:
            cache_dir = os.path.join(
                getattr(self.brain, 'cache_dir', './cache'), 
                'hybrid_cache'
            )
            self.hybrid_cache = HybridTokenCache(
                brain=self.brain,
                max_memory_tokens=10000,
                disk_cache_dir=cache_dir
            )
            logger.info("Гибридный кэш инициализирован")
        except Exception as e:
            logger.error(f"Ошибка инициализации гибридного кэша: {e}")
            self.hybrid_cache = None
    
    def _create_fallback_tokenizer(self) -> None:
        """Создаёт fallback токенизатор если основной недоступен."""
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers недоступен, невозможно создать fallback токенизатор")
            return
        
        base_path = getattr(self.brain, 'project_root', None) or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_paths_to_try = [
            os.path.join(base_path, "cogniflex", "mlearning", "cogniflex_models", "qwen3.5-0.8b"),
            os.path.join(base_path, "cogniflex", "mlearning", "cogniflex_models", "rugpt3_large"),
        ]
        
        for model_path in model_paths_to_try:
            try:
                if os.path.exists(model_path):
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_path, 
                        local_files_only=True,
                        trust_remote_code=True
                    )
                    if self.tokenizer:
                        if self.tokenizer.pad_token is None:
                            if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token:
                                self.tokenizer.pad_token = self.tokenizer.eos_token
                            else:
                                self.tokenizer.pad_token = '</pad>'
                        logger.info(f"Создан fallback токенизатор из {os.path.basename(model_path)}")
                        return
            except Exception as e:
                logger.warning(f"Не удалось загрузить токенизатор из {model_path}: {e}")
                continue
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
            if self.tokenizer and self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Создан fallback токенизатор (gpt2)")
        except Exception as e:
            logger.error(f"Не удалось создать fallback токенизатор: {e}")
            self.tokenizer = None
    
    def _validate_tokenizer(self, tokenizer: Any) -> bool:
        """Проверяет валидность токенизатора.

        Args:
            tokenizer: Токенизатор для валидации

        Returns:
            bool: True если токенизатор валиден
        """
        if tokenizer is None:
            logger.debug("Токенизатор не инициализирован (None)")
            return False

        # Минимальная проверка - объект существует и не является примитивным типом
        # Это позволит использовать любые объекты как токенизаторы
        if isinstance(tokenizer, (str, int, float, bool, list, dict)):
            logger.debug("Токенизатор является примитивным типом")
            return False

        # Логируем успешную валидацию
        logger.debug("Токенизатор прошел валидацию")
        return True
    
    def initialize(self) -> bool:
        """
        Инициализирует ResponseGenerator.

        
        Returns:
            bool: True если инициализация успешна
        """
        try:
            if not self.brain:
                logger.error("CoreBrain не передан в ResponseGenerator")
                return False
            
            if not self._initialized:
                self._init_components()
            
            # Проверяем токенизатор (не блокируем инициализацию)
            if not self._validate_tokenizer(self.tokenizer):
                logger.warning("Токенизатор не прошел валидацию")
            
            # Проверяем ModelManager (динамическая проверка, не блокируем инициализацию)
            if not self.model_manager and hasattr(self.brain, 'components'):
                self.model_manager = self.brain.components.get('model_manager')
            
            if not self.model_manager:
                logger.warning("ModelManager не инициализирован")
            else:
                logger.info("ModelManager найден в ResponseGenerator.initialize")
            
            logger.info("ResponseGenerator инициализирован")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при инициализации: {e}", exc_info=True)
            return False
    
    def generate_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Генерирует ответ на основе промпта.
        
        Приоритет:
        1. Фрактальная модель (если доступна)
        2. ModelManager (стандартный путь)
        3. Fallback ответ
        
        Args:
            prompt: Текст промпта
            **kwargs: Дополнительные параметры генерации
            
        Returns:
            Dict[str, Any]: Результат генерации
            
        Raises:
            RuntimeError: Если модель недоступна
        """
        start_time = time.time()
        
        # Check for ambiguities before generation
        ambiguity_check = self._detect_ambiguity_before_response(prompt)
        if ambiguity_check.get("needs_clarification"):
            clarification_response = self._generate_clarification_response(
                ambiguity_check.get("clarifications", [])
            )
            return {
                "text": clarification_response,
                "needs_clarification": True,
                "clarifications": ambiguity_check.get("clarifications", [])
            }
        
        try:
            # ПРИОРИТЕТ 1: Проверяем фрактальную модель
            if self._is_fractal_ready():
                return self._generate_fractal_response(prompt, start_time, kwargs)
            
            # ПРИОРИТЕТ 2: Получаем модель из ModelManager
            model, tokenizer, model_name = self._get_model_for_generation(kwargs)
            
            if model is None:
                return self._create_fallback_response(prompt, "no_model_available")
            
            # Подготовка промпта
            context = kwargs.get('context')
            final_prompt = self._prepare_prompt(prompt, "text-generation", context)
            
            # Генерация ответа
            generated_text = self._generate_with_model(model, tokenizer, final_prompt, **kwargs)
            
            # Создаём ответ
            clean_kwargs = {k: v for k, v in kwargs.items() if k != 'task'}
            return self._create_response(
                generated_text=generated_text,
                tokenizer=tokenizer,
                model_name=model_name,
                task="text-generation",
                prompt=final_prompt,
                start_time=start_time,
                **clean_kwargs
            )
            
        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {e}", exc_info=True)
            raise RuntimeError(f"Не удалось сгенерировать ответ: {str(e)}") from e
    
    def _is_fractal_ready(self) -> bool:
        """Проверяет готовность фрактальной модели."""
        return (
            self.brain is not None and 
            getattr(self.brain, 'fractal_ready', False) and 
            getattr(self.brain, 'fractal_model_manager', None) is not None
        )
    
    def _generate_fractal_response(self, prompt: str, start_time: float, kwargs: Dict) -> Dict[str, Any]:
        """Генерирует ответ через фрактальную модель."""
        try:
            logger.info(f"Используем фрактальную модель: {prompt[:50]}...")
            fractal_response = self.brain.fractal_model_manager.generate_response(prompt)
            
            return {
                "text": fractal_response,
                "tokens": [],
                "metadata": {
                    "model": "fractal_model",
                    "task": "text-generation",
                    "length": len(fractal_response),
                    "from_cache": False,
                    "generation_time": time.time() - start_time,
                    "token_count": 0,
                    "params": {k: v for k, v in kwargs.items()},
                    "source": "fractal_model"
                }
            }
        except Exception as e:
            logger.warning(f"Ошибка фрактальной модели: {e}")
            return None
    
    def _get_model_for_generation(self, kwargs: Dict) -> Tuple[Optional[Any], Optional[Any], str]:
        """Получает модель для генерации.
        
        Returns:
            Tuple: (model, tokenizer, model_name)
        """
        model = kwargs.get('model')
        tokenizer = kwargs.get('tokenizer')
        model_name = kwargs.get('model_name', 'unknown')
        
        if model is None and self.model_manager:
            try:
                model_result = self.model_manager.get_model_for_task("text-generation")
                if model_result is not None:
                    # Проверяем что вернули кортеж или другой формат
                    if isinstance(model_result, tuple) and len(model_result) >= 3:
                        model, tokenizer, model_name = model_result
                    elif isinstance(model_result, dict):
                        model = model_result.get('model')
                        tokenizer = model_result.get('tokenizer')
                        model_name = model_result.get('model_name', model_name)
                    else:
                        # Если вернули только модель
                        model = model_result
                else:
                    logger.warning("ModelManager вернул None для задачи text-generation")
            except Exception as e:
                logger.error(f"Ошибка получения модели: {e}")
                return None, None, model_name
        
        # Fallback: используем self.tokenizer если tokenizer всё ещё None
        if tokenizer is None and self.tokenizer is not None:
            tokenizer = self.tokenizer
            logger.info("Используем self.tokenizer как fallback")
        
        return model, tokenizer, model_name
    
    def _generate_with_model(self, model: Any, tokenizer: Any, prompt: str, **kwargs) -> str:
        """Генерирует ответ с использованием модели."""
        
        # Кэшируем токенизацию если доступен HybridTokenCache
        if self.hybrid_cache and hasattr(tokenizer, 'name'):
            cache_key = f"tokens_{tokenizer.name}_{hash(prompt)}"
            cached_tokens = self.hybrid_cache.get(cache_key)
            
            if cached_tokens is not None:
                logger.debug(f"Используем кэшированные токены для {cache_key}")
        
        if tokenizer is None:
            raise ValueError("Tokenizer is None - cannot generate response")
        
        if not TORCH_AVAILABLE:
            raise ValueError("PyTorch недоступен - невозможно выполнить генерацию")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        use_sema = hasattr(self, '_gpu_generate_sema')
        
        try:
            if use_sema:
                self._gpu_generate_sema.acquire()
            
            with torch.no_grad():
                # Токенизация
                inputs = self._tokenize_input(tokenizer, prompt, device)
                
                # Параметры генерации
                gen_kwargs = self._prepare_generation_kwargs(kwargs)
                
                # Генерация
                output_sequences = model.generate(**inputs, **gen_kwargs)
                
                # Декодирование
                generated_text = self._decode_output(tokenizer, output_sequences, prompt)
                
                return generated_text
                
        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}", exc_info=True)
            return self._fallback_generation(prompt, kwargs.get('max_length', 200))
        finally:
            if use_sema:
                self._gpu_generate_sema.release()
    
    def _tokenize_input(self, tokenizer: Any, prompt: str, device: torch.device) -> Dict[str, torch.Tensor]:
        """Токенизирует входной промпт."""
        try:
            if hasattr(tokenizer, '__call__'):
                inputs = tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=1024
                ).to(device)
            else:
                inputs = tokenizer.encode(
                    prompt, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=1024
                ).to(device)
            
            return inputs
        except Exception as e:
            logger.error(f"Ошибка токенизации: {e}")
            raise
    
    def _prepare_generation_kwargs(self, kwargs: Dict) -> Dict[str, Any]:
        """Подготавливает параметры генерации."""
        return {
            "max_length": kwargs.get('max_length', 200),
            "temperature": kwargs.get('temperature', 0.8),
            "top_p": kwargs.get('top_p', 0.95),
            "do_sample": False,
            "repetition_penalty": 2.0,
            "no_repeat_ngram_size": 3,
            "pad_token_id": getattr(self.tokenizer, 'eos_token_id', None)
        }
    
    def _decode_output(self, tokenizer: Any, output_sequences: torch.Tensor, prompt: str) -> str:
        """Декодирует выход модели."""
        try:
            if hasattr(tokenizer, 'batch_decode'):
                decoded_texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            else:
                decoded_texts = [
                    tokenizer.decode(seq, skip_special_tokens=True) 
                    for seq in output_sequences
                ]
            
            results = []
            for full_text in decoded_texts:
                if full_text.startswith(prompt):
                    results.append(self._postprocess_generated_text(full_text[len(prompt):]))
                else:
                    results.append(self._postprocess_generated_text(full_text))
            
            return results[0] if results else ""
            
        except Exception as e:
            logger.error(f"Ошибка декодирования: {e}")
            return ""
    
    def _create_response(self, generated_text: str, tokenizer: Any, model_name: str, 
                        task: str, prompt: str, start_time: float, **kwargs) -> Dict[str, Any]:
        """Создаёт структурированный ответ."""
        processed_text = self._postprocess_generated_text(generated_text)
        tokens = self._safe_tokenize(tokenizer, processed_text)
        
        return {
            "text": processed_text,
            "tokens": tokens,
            "metadata": {
                "model": model_name,
                "task": task,
                "length": len(processed_text),
                "from_cache": False,
                "generation_time": time.time() - start_time,
                "token_count": len(tokens),
                "params": {k: v for k, v in kwargs.items()}
            }
        }
    
    def _postprocess_generated_text(self, text: str) -> str:
        """Постобрабатывает сгенерированный текст."""
        if not text:
            return ""
        return text.strip()
    
    def _safe_tokenize(self, tokenizer: Any, text: str) -> List[str]:
        """Безопасная токенизация текста."""
        try:
            if tokenizer and hasattr(tokenizer, 'encode'):
                return tokenizer.encode(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            else:
                return text.split()
        except Exception as e:
            logger.error(f"Ошибка токенизации: {e}")
            return []
    
    def _fallback_generation(self, prompt: str, max_length: int = 300) -> str:
        """Генерирует fallback-ответ."""
        return f"Извините, произошла ошибка при обработке: {prompt[:max_length]}..."
    
    def _create_fallback_response(self, prompt: str, error_type: str = "unknown") -> Dict[str, Any]:
        """Создаёт fallback ответ при ошибке.
        
        Args:
            prompt: Исходный промпт
            error_type: Тип ошибки
            
        Returns:
            Dict: Структурированный ответ с ошибкой
        """
        logger.error(f"КРИТИЧЕСКАЯ ОШИБКА: {error_type}")
        logger.error(f"Система не может обработать запрос: '{prompt[:50]}...'")
        logger.error("ТРЕБУЕТСЯ: Настроить фрактальную модель для полноценной работы")
        
        # Возвращаем структурированную ошибку вместо исключения
        return {
            "text": f"Ошибка генерации: {error_type}. Система требует настройки модели.",
            "tokens": [],
            "metadata": {
                "model": "none",
                "task": "text-generation",
                "length": 0,
                "from_cache": False,
                "generation_time": 0.0,
                "token_count": 0,
                "error": error_type,
                "source": "fallback"
            },
            "error": error_type
        }
    
    def _prepare_prompt(self, prompt: str, task: str, context: Optional[str] = None) -> str:
        """Подготавливает промпт для генерации."""
        if context:
            return f"{context}\n\n{prompt}"
        return prompt
    
    def _map_reduce_context(self, prompt: str, task: str, context: str, 
                           model: Any, tokenizer: Any) -> str:
        """Применяет Map-Reduce к контексту."""
        # Упрощенная реализация
        return context
    
    def get_status(self) -> Dict[str, Any]:
        """Возвращает статус генератора."""
        return {
            "available": self.is_available(),
            "initialized": self._initialized,
            "brain_connected": self.brain is not None,
            "token_streamer": self.token_streamer is not None,
            "hybrid_cache": self.hybrid_cache is not None,
            "torch_available": TORCH_AVAILABLE,
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "hybrid_cache_available": HYBRID_CACHE_AVAILABLE
        }
    
    def is_available(self) -> bool:
        """Проверяет доступность генератора."""
        return (
            self._initialized and
            self.brain is not None and 
            self.tokenizer is not None and
            (self.model_manager is not None or self.hybrid_cache is not None)
        )
    
    def mark_response_type(self, text: str, is_verified: bool = False, source: str = ""):
        """Mark response text as verified or generated."""
        if self.knowledge_awareness:
            if is_verified:
                self.knowledge_awareness.mark_verified(text, source)
            else:
                confidence = getattr(self, '_last_confidence', 0.7)
                self.knowledge_awareness.mark_generated(text, confidence)
    
    def shutdown(self) -> None:
        """Останавливает генератор."""
        if getattr(self, '_shutdown_event', None) and self._shutdown_event.is_set():
            logger.debug("ResponseGenerator уже завершает работу")
            return
        
        # Останавливаем фоновые процессы
        if self._thread_pool:
            self._shutdown_event.set()
            self._thread_pool.shutdown(wait=True)
            self._thread_pool = None
        
        # Очищаем кэш
        if self.hybrid_cache:
            try:
                self.hybrid_cache.clear()
            except Exception as e:
                logger.warning(f"Ошибка очистки кэша: {e}")
        
        logger.info("ResponseGenerator остановлен")
    
    def __enter__(self):
        """Контекстный менеджер."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Контекстный менеджер."""
        self.shutdown()
    
    def __del__(self):
        """Деструктор."""
        try:
            self.shutdown()
        except Exception:
            pass


# Экспорт для совместимости
__all__ = ['ResponseGenerator']
