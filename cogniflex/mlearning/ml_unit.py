"""
Главный модуль машинного обучения для CogniFlex - объединяет все компоненты
с поддержкой асинхронной потоковой токенизации и гибридного кэширования
"""
import os
import logging
import time
import torch
import hashlib
import math
import json
from typing import Dict, Any, Optional, List, Tuple, Union



logger = logging.getLogger("cogniflex.ml_unit")

# Импортируем компоненты с обработкой возможных ошибок
try:
    from cogniflex.mlearning.ml_core import MLCore
    logger.info("MLCore успешно импортирован из cogniflex.mlearning.ml_core")
except ImportError as e:
    logger.warning(f"Ошибка импорта MLCore: {e}")
    MLCore = None

try:
    from cogniflex.mlearning.model_manager import ModelManager
    logger.info("ModelManager успешно импортирован из cogniflex.mlearning.model_manager")
except ImportError as e:
    logger.warning(f"Ошибка импорта ModelManager: {e}")
    ModelManager = None

try:
    from cogniflex.mlearning.unified_text_processor import UnifiedTextProcessor
    logger.info("UnifiedTextProcessor успешно импортирован из cogniflex.mlearning.unified_text_processor")
except ImportError as e:
    logger.warning(f"Ошибка импорта UnifiedTextProcessor: {e}")
    try:
        # Попытка импорта из альтернативного пути
        from unified_text_processor import UnifiedTextProcessor
        logger.info("UnifiedTextProcessor успешно импортирован из cogniflex.mlearning.text_processor")
    except ImportError as e2:
        logger.warning(f"Альтернативный импорт UnifiedTextProcessor также не удался: {e2}")
        UnifiedTextProcessor = None

# Пытаемся импортировать TrainingOrchestrator (отдельно от UnifiedTextProcessor)
try:
    from cogniflex.mlearning.training_orchestrator import TrainingOrchestrator
    logger.info("TrainingOrchestrator успешно импортирован из cogniflex.mlearning.training_orchestrator")
except ImportError as e:
    logger.warning(f"Ошибка импорта TrainingOrchestrator: {e}")
    TrainingOrchestrator = None

# Удален импорт ResponseGenerator для устранения циклической зависимости
# ResponseGenerator теперь доступен через brain.response_generator
ResponseGenerator = None

# Импорты для гибридного кэша и токенизации
try:
    from cogniflex.memory.hybrid_token_cache import HybridTokenCache
    logger.debug("HybridTokenCache импортирован успешно")
except ImportError as e:
    logger.warning(f"Ошибка импорта HybridTokenCache: {e}")
    HybridTokenCache = None

try:
    from cogniflex.mlearning.unified_text_processor import UnifiedTextProcessor
    logger.debug("UnifiedTextProcessor импортирован успешно")
    # Используем UnifiedTextProcessor как замену StreamTokenizer
    StreamTokenizer = UnifiedTextProcessor
except ImportError:
    logger.warning("UnifiedTextProcessor недоступен, асинхронная токенизация будет ограничена")
    StreamTokenizer = None

class MLUnit:
    """
    Модуль машинного обучения для CogniFlex - объединяет все компоненты.
    
    Основные улучшения:
    - Полная поддержка асинхронной потоковой токенизации
    - Интеграция с гибридным кэшем (память + диск)
    - Обеспечение корректного взаимодействия между всеми компонентами
    - Улучшенная обработка ошибок и fallback-сценарии
    """
    
    def __init__(self, brain=None, cache_dir: Optional[str] = None,
                 use_gpu: bool = True, max_models: int = 5,
                 max_workers: int = 4, hybrid_cache_size: int = 10000):
        """
        Инициализирует модуль машинного обучения.
        
        Args:
            brain: Ссылка на ядро системы
            cache_dir: Путь к директории кэша
            use_gpu: Использовать GPU если доступен
            max_models: Максимальное количество загружаемых моделей
            max_workers: Количество рабочих потоков для асинхронной обработки
            hybrid_cache_size: Размер гибридного кэша
        """
        self.brain = brain
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "ml_cache")
        self.use_gpu = use_gpu
        self.max_models = max_models
        self.max_workers = max_workers
        self.hybrid_cache_size = hybrid_cache_size
        self.initialized = False
        self.running = False
        
        # Создаем директорию кэша если её нет
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Инициализация компонентов
        self.ml_core = None
        self.model_manager = None
        self.text_processor = None
        self.response_generator = None
        self.token_streamer = None
        self.hybrid_cache = None
        self.active_model_id = None
        self.training_orchestrator = None
        
        # Статистика
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
            "last_request_time": 0.0
        }
        
        # Инициализация компонентов
        self._init_components()
        
        logger.info(f"MLUnit инициализирован с use_gpu={use_gpu}, max_models={max_models}")
    
    def _init_components(self):
        """Инициализирует все компоненты MLUnit в правильном порядке."""
        try:
            # 1. Сначала MemoryManager (гибридный кэш)
            self._init_hybrid_cache()
            
            # 2. Затем UnifiedTextProcessor
            if self.brain and hasattr(self.brain, 'text_processor') and self.brain.text_processor:
                self.text_processor = self.brain.text_processor
                logger.info("Используется текстовый процессор из brain")
            else:
                self._init_text_processor()
            
            # Вызываем обратные вызовы для text_processor
            if self.text_processor and hasattr(self.brain, 'on_text_processor_ready'):
                for callback in self.brain.on_text_processor_ready:
                    try:
                        callback(self.text_processor)
                    except Exception as e:
                        logger.error(f"Ошибка в обратном вызове: {e}")
            
            # 3. Потом ModelManager
            self._init_model_manager()
            
            # 4. И только потом ResponseGenerator
            self._init_response_generator()
            
            # 5. Остальные компоненты
            self._init_ml_core()
            
            # 6. Проверка целостности компонентов
            self._validate_components()
            
            self.initialized = True
            logger.info("MLUnit полностью инициализирован")
            
            # 7. Инициализация оркестратора обучения после связывания компонентов
            self._init_training_orchestrator()
            
        except Exception as e:
            logger.error(f"Критическая ошибка инициализации MLUnit: {e}", exc_info=True)
            self.initialized = False
    
    def _tokenize_text(self, text: str, model_name: str = None) -> Dict[str, Any]:
        """Токенизирует текст с использованием UnifiedTextProcessor."""
        try:
            if self.token_streamer:
                # Используем асинхронную токенизацию из UnifiedTextProcessor
                return self.token_streamer.tokenize_async([text])[0]
            else:
                # Резервная токенизация
                tokens = text.split()
                return {
                    "tokens": tokens,
                    "token_count": len(tokens),
                    "original_text": text[:100] + "..." if len(text) > 100 else text
                }
        except Exception as e:
            logger.error(f"Ошибка токенизации: {e}", exc_info=True)
            return {
                "tokens": [],
                "token_count": 0,
                "error": str(e)
            }

    def _init_hybrid_cache(self):
        """Инициализирует гибридный кэш."""
        try:
            # Если кэш уже есть в ядре, используем его
            if self.brain and hasattr(self.brain, 'memory_manager') and self.brain.memory_manager and self.brain.memory_manager.hybrid_cache:
                self.hybrid_cache = self.brain.memory_manager.hybrid_cache
                logger.debug("Используется гибридный кэш из MemoryManager")
            else:
                # Создаем новый гибридный кэш ТОЛЬКО ЕСЛИ НЕТ MemoryManager
                if HybridTokenCache:
                    # ИСПОЛЬЗУЕМ ПРАВИЛЬНЫЙ ПАРАМЕТР cache_path вместо cache_dir
                    cache_path = os.path.join(self.cache_dir, "hybrid_cache")
                    os.makedirs(cache_path, exist_ok=True)
                    
                    self.hybrid_cache = HybridTokenCache(
                        brain=self.brain,
                        max_memory_tokens=5000,
                        disk_cache_dir="hybrid_cache"
                    )
                    logger.info(f"Инициализирован гибридный кэш с размером {self.hybrid_cache_size}")
                    
                    # Сохраняем кэш в memory_manager, если он существует
                    if self.brain and hasattr(self.brain, 'memory_manager'):
                        self.brain.memory_manager.hybrid_cache = self.hybrid_cache
                        logger.debug("Гибридный кэш сохранен в MemoryManager")
                else:
                    logger.warning("HybridTokenCache недоступен, кэширование будет ограничено")
        
        except Exception as e:
            logger.error(f"Ошибка инициализации гибридного кэша: {e}", exc_info=True)
    
    def _init_token_streamer(self):
        """Инициализирует потоковый токенизатор."""
        try:
            # Используем существующий text_processor как token_streamer
            if hasattr(self, 'text_processor') and self.text_processor:
                self.token_streamer = self.text_processor
                logger.debug("Используем UnifiedTextProcessor для асинхронной токенизации")
            else:
                # НЕ ПЫТАЕМСЯ ИСПОЛЬЗОВАТЬ StreamTokenizer - его НЕТ В ПРОЕКТЕ
                logger.warning("UnifiedTextProcessor недоступен для токенизации")
                self.token_streamer = None
        except Exception as e:
            logger.error(f"Ошибка инициализации token_streamer: {e}", exc_info=True)
            self.token_streamer = None
    
    def _init_ml_core(self):
        """Инициализирует ядро ML."""
        try:
            if MLCore:
                self.ml_core = MLCore(
                    brain=self.brain,
                    cache_dir=self.cache_dir,
                    hybrid_cache=self.hybrid_cache,
                    token_streamer=self.token_streamer
                )
                logger.info("MLCore инициализирован")
            else:
                logger.warning("MLCore недоступен, функциональность ограничена")
        
        except Exception as e:
            logger.error(f"Ошибка инициализации MLCore: {e}", exc_info=True)
    
    def _init_model_manager(self):
        """Инициализирует менеджер моделей."""
        try:
            if ModelManager:
                self.model_manager = ModelManager(
                    brain=self.brain,
                    cache_dir=os.path.join(self.cache_dir, "models"),
                    use_gpu=self.use_gpu,
                    max_workers=self.max_workers,
                    hybrid_cache_size=self.hybrid_cache_size,
                    autoload=not self._is_training_mode()
                )
                
                # Передаем text_processor напрямую в ModelManager
                if self.text_processor:
                    self.model_manager.text_processor = self.text_processor
                    self.model_manager.token_streamer = self.text_processor
                    logger.debug("text_processor передан в ModelManager")
                    
                    # Вызываем обратные вызовы для text_processor
                    if hasattr(self.brain, 'on_text_processor_ready'):
                        for callback in self.brain.on_text_processor_ready:
                            try:
                                callback(self.text_processor)
                            except Exception as e:
                                logger.error(f"Ошибка в обратном вызове: {e}")
                
                logger.info("ModelManager инициализирован")
            else:
                logger.warning("ModelManager недоступен, загрузка моделей невозможна")
        
        except Exception as e:
            logger.error(f"Ошибка инициализации ModelManager: {e}", exc_info=True)
    
    def _init_text_processor(self):
        """Инициализирует текстовый процессор."""
        try:
            if UnifiedTextProcessor:
                # Проверяем, есть ли уже гибридный кэш
                hybrid_cache = self.hybrid_cache
                if not hybrid_cache and self.brain and hasattr(self.brain, 'memory_manager') and self.brain.memory_manager:
                    hybrid_cache = self.brain.memory_manager.hybrid_cache
                
                self.text_processor = UnifiedTextProcessor(
                    brain=self.brain,
                    cache_dir=os.path.join(self.cache_dir, "text_processing"),
                    use_gpu=self.use_gpu,
                    max_workers=self.max_workers,
                    hybrid_cache=hybrid_cache
                )
                logger.info("UnifiedTextProcessor инициализирован")
            else:
                logger.warning("UnifiedTextProcessor недоступен для токенизации")
                self.text_processor = None
        except Exception as e:
            logger.error(f"Ошибка инициализации текстового процессора: {e}", exc_info=True)
            self.text_processor = None
    
    def _init_response_generator(self):
        """Инициализирует генератор ответов, получая его из brain."""
        try:
            # Пытаемся получить response_generator из brain
            if self.brain and hasattr(self.brain, 'response_generator') and self.brain.response_generator:
                self.response_generator = self.brain.response_generator
                logger.info("ResponseGenerator успешно получен из brain.")
            else:
                logger.warning("ResponseGenerator недоступен в brain, генерация ответов невозможна.")
                self.response_generator = None
        
        except Exception as e:
            logger.error(f"Ошибка получения ResponseGenerator из brain: {e}", exc_info=True)
            self.response_generator = None

    def _progress_callback(self, payload: Dict[str, Any]) -> None:
        """Безопасно пробрасывает события прогресса обучения в brain или лог."""
        try:
            if self.brain is None:
                logger.debug(f"Training progress: {payload}")
                return
            # Предпочитаем явный эмиттер событий, если есть
            if hasattr(self.brain, 'emit_training_event') and callable(getattr(self.brain, 'emit_training_event')):
                self.brain.emit_training_event(payload)
                return
            # Иначе пробуем список колбэков
            callbacks = getattr(self.brain, 'on_training_progress', None)
            if callbacks:
                for cb in callbacks:
                    try:
                        cb(payload)
                    except Exception as cb_e:
                        logger.debug(f"Ошибка в колбэке прогресса обучения: {cb_e}")
                return
            # Фоллбек: лог
            logger.debug(f"Training progress: {payload}")
        except Exception as e:
            logger.debug(f"Прогресс колбэк завершился с ошибкой: {e}")

    def _init_training_orchestrator(self):
        """Инициализирует оркестратор обучения знаний."""
        try:
            if TrainingOrchestrator is None:
                logger.warning("TrainingOrchestrator недоступен, обучение знаний будет недоступно")
                self.training_orchestrator = None
                return
            self.training_orchestrator = TrainingOrchestrator(
                brain=self.brain,
                cache_dir=os.path.join(self.cache_dir, 'training'),
                progress_cb=self._progress_callback,
                auto_adapt=True,
            )
            logger.info("TrainingOrchestrator инициализирован")
        except Exception as e:
            logger.error(f"Ошибка инициализации TrainingOrchestrator: {e}", exc_info=True)
            self.training_orchestrator = None
    
    def _validate_components(self):
        """Регистрирует отложенную команду для связывания компонентов."""
        if self.brain and hasattr(self.brain, 'add_deferred_command'):
            logger.info("Регистрация отложенной команды для связывания компонентов MLUnit...")
            self.brain.add_deferred_command(self._link_components)
        else:
            logger.warning("Не удалось зарегистрировать отложенную команду: brain или add_deferred_command недоступны.")
            # Как fallback, можно попробовать вызвать связывание напрямую, но это рискованно
            self._link_components()

    def _link_components(self):
        """Устанавливает взаимные ссылки между компонентами после полной инициализации."""
        logger.info("Выполнение отложенной компоновки компонентов MLUnit...")
        try:
            # Устанавливаем взаимные ссылки между компонентами
            if self.ml_core:
                self.ml_core.model_manager = self.model_manager
                self.ml_core.text_processor = self.text_processor
                self.ml_core.response_generator = self.response_generator
            
            if self.model_manager:
                self.model_manager.ml_core = self.ml_core
                self.model_manager.text_processor = self.text_processor
                self.model_manager.response_generator = self.response_generator
            
            if self.text_processor:
                self.text_processor.ml_core = self.ml_core
                self.text_processor.model_manager = self.model_manager
                self.text_processor.response_generator = self.response_generator
            
            if self.response_generator:
                # Дополнительно передаем компоненты, если они не были установлены при инициализации
                if self.model_manager and not self.response_generator.model_manager:
                    self.response_generator.model_manager = self.model_manager
                if self.text_processor and not self.response_generator.text_processor:
                    self.response_generator.text_processor = self.text_processor
                self.response_generator.ml_core = self.ml_core

            logger.info("Компоненты MLUnit успешно связаны.")

            # Обновляем ссылки оркестратора обучения теперь, когда компоненты связаны
            try:
                if self.training_orchestrator is not None:
                    # Убедимся, что оркестратор видит актуальные компоненты
                    self.training_orchestrator.ml_unit = self
                    if hasattr(self, 'token_streamer'):
                        self.training_orchestrator.token_streamer = self.token_streamer
                    if hasattr(self, 'hybrid_cache'):
                        self.training_orchestrator.hybrid_cache = self.hybrid_cache
                    if self.brain is not None and hasattr(self.brain, 'knowledge_graph'):
                        self.training_orchestrator.knowledge_graph = self.brain.knowledge_graph
            except Exception as e:
                logger.debug(f"Не удалось обновить ссылки TrainingOrchestrator после связывания: {e}")

            # Проверяем основные функции после связывания (пропускаем в режиме обучения/тренировки)
            if not self._is_training_mode():
                self._verify_basic_functionality()
            
        except Exception as e:
            logger.error(f"Ошибка при отложенной компоновке компонентов: {e}", exc_info=True)
    
    def _verify_basic_functionality(self):
        """Проверяет базовую функциональность компонентов."""
        # В режиме тренировки пропускаем тяжелые самопроверки, чтобы не загружать модели/токены
        if self._is_training_mode():
            logger.debug("Пропуск _verify_basic_functionality в режиме тренировки")
            return
        try:
            # Проверяем доступность моделей
            if self.model_manager:
                models = self.model_manager.get_available_models()
                logger.info(f"Доступно моделей: {len(models)}")
            
            # Проверяем токенизацию
            if self.text_processor:
                test_text = "Это тестовый текст для проверки токенизации."
                analysis = self.text_processor.process_text(test_text)
                tokens_count = len(analysis.get('tokens', [])) if isinstance(analysis, dict) else 0
                logger.info(f"Тестовая токенизация успешна. Токенов: {tokens_count}")
            
            # Проверяем генерацию ответа
            if self.response_generator:
                test_prompt = "Кратко опиши, что такое искусственный интеллект."
                response = self.response_generator.generate_response(
                    prompt=test_prompt,
                    max_length=100,
                    temperature=0.7,
                    top_p=0.9,
                    task="text-generation"
                )
                logger.info(f"Тестовая генерация ответа успешна. Длина: {len(response.get('text', ''))}")
        
        except Exception as e:
            logger.error(f"Ошибка проверки базовой функциональности: {e}", exc_info=True)
    
    def initialize(self) -> bool:
        """
        Дополнительная инициализация после создания объекта.
        
        Returns:
            bool: Успешно ли инициализировано
        """
        try:
            # Загружаем модели
            if self.model_manager and not self._is_training_mode():
                self.model_manager.load_models()
            
            # Проверяем здоровье системы
            health = self.get_system_health()
            logger.info(f"Состояние ML системы: {health['status']}, score: {health['score']:.2f}")
            
            return health['score'] > 0.5
        
        except Exception as e:
            logger.error(f"Ошибка дополнительной инициализации MLUnit: {e}", exc_info=True)
            return False
    
    def start(self):
        """Запускает фоновые процессы MLUnit."""
        if self.running:
            return
            
        self.running = True
        
        # Запускаем фоновые процессы компонентов
        if self.text_processor and hasattr(self.text_processor, 'start'):
            self.text_processor.start()
        
        if self.response_generator and hasattr(self.response_generator, 'start'):
            self.response_generator.start()
        
        if self.model_manager and hasattr(self.model_manager, 'start'):
            self.model_manager.start()
        
        logger.info("MLUnit запущен")
    
    def stop(self):
        """Останавливает фоновые процессы MLUnit."""
        if not self.running:
            return
            
        self.running = False
        
        # Останавливаем фоновые процессы компонентов
        if self.text_processor and hasattr(self.text_processor, 'stop'):
            self.text_processor.stop()
        
        if self.response_generator and hasattr(self.response_generator, 'stop'):
            self.response_generator.stop()
        
        if self.model_manager and hasattr(self.model_manager, 'stop'):
            self.model_manager.stop()
        
        # Сохраняем кэш
        if self.hybrid_cache and hasattr(self.hybrid_cache, 'stop'):
            self.hybrid_cache.stop()
        
        logger.info("MLUnit остановлен")
    
    def generate_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Генерирует ответ на запрос.
        
        Args:
            prompt: Текст запроса
            **kwargs: Дополнительные параметры
            
        Returns:
            Dict[str, Any]: Структурированный ответ
        """
        start_time = time.time()
        self.stats["total_requests"] += 1
        
        try:
            if not self.response_generator:
                logger.error("ResponseGenerator недоступен")
                return self._create_fallback_response(prompt, "response_generator_unavailable")
            
            # Генерируем ответ
            response = self.response_generator.generate_response(prompt, **kwargs)
            
            # Обновляем статистику
            self._update_statistics(start_time, True)
            
            return response
            
        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}", exc_info=True)
            self._update_statistics(start_time, False)
            return self._create_fallback_response(prompt, str(e))
    
    def _update_statistics(self, start_time: float, success: bool):
        """Обновляет статистику запросов."""
        processing_time = time.time() - start_time
        self.stats["total_processing_time"] += processing_time
        self.stats["last_request_time"] = time.time()
        
        if success:
            self.stats["successful_requests"] += 1
        else:
            self.stats["failed_requests"] += 1
        
        # Emit normalized metrics via brain if available
        try:
            metrics = [
                {
                    "name": "ml_unit.requests_total",
                    "component": "ml_unit",
                    "type": "counter",
                    "value": 1.0,
                    "labels": {"result": "success" if success else "failure"},
                },
                {
                    "name": "ml_unit.response_time_seconds",
                    "component": "ml_unit",
                    "type": "summary",
                    "value": float(processing_time),
                },
            ]
            self._emit_metrics(metrics)
        except Exception:
            pass
    
    def _emit_metrics(self, metrics: List[Dict[str, Any]]) -> None:
        """Safely forwards normalized metrics via event bus ('metrics') and direct emit for compatibility."""
        try:
            brain = getattr(self, "brain", None)
            if not brain:
                return
            # Unified transport via event system
            try:
                if hasattr(brain, "events") and brain.events:
                    brain.events.trigger('metrics', metrics)
            except Exception:
                pass
            # Backward compatibility direct emit
            try:
                if hasattr(brain, "emit_metrics"):
                    brain.emit_metrics(metrics)
            except Exception:
                pass
        except Exception:
            # Never raise from metrics
            pass
    
    def _create_fallback_response(self, prompt: str, error: str) -> Dict[str, Any]:
        """
        Создает fallback-ответ при ошибке.
        
        Args:
            prompt: Исходный запрос
            error: Описание ошибки
            
        Returns:
            Dict[str, Any]: Структурированный fallback-ответ
        """
        # Попытаемся создать более информативный ответ
        try:
            # Простой анализ запроса для fallback
            analysis = self.analyze_query(prompt)
            keywords = analysis.get("keywords", [])
            
            fallback_text = f"Получен запрос: '{prompt}'. "
            if keywords:
                fallback_text += f"Обнаружены ключевые слова: {', '.join(keywords[:3])}. "
            fallback_text += "ResponseGenerator временно недоступен."
            
        except Exception:
            fallback_text = "Извините, произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте повторить запрос позже."
        
        return {
            "text": fallback_text,
            "tokens": [],
            "metadata": {
                "model": "fallback",
                "task": "text-generation",
                "length": 0,
                "error": error,
                "original_prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt
            },
            "reasoning": None,
            "contradiction_detected": False,
            "contradictions": [],
            "sentiment": None
        }
    
    def process_text(self, text: str, **kwargs) -> Any:
        """
        Обрабатывает текст с использованием NLP-моделей.
        
        Args:
            text: Текст для обработки
            **kwargs: Дополнительные параметры
            
        Returns:
            Any: Результаты обработки
        """
        if not self.text_processor:
            logger.error("TextProcessor недоступен")
            return None
        
        return self.text_processor.process_text(text, **kwargs)
    
    def get_available_hf_models(self) -> List[Dict[str, Any]]:
        """
        Возвращает список доступных моделей Hugging Face.
        
        Returns:
            List[Dict[str, Any]]: Список моделей
        """
        if not self.model_manager:
            logger.error("ModelManager недоступен")
            return []
        
        return self.model_manager.get_available_models()
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Возвращает состояние системы ML.
        
        Returns:
            Dict[str, Any]: Состояние системы
        """
        health = {
            "score": 0.0,
            "status": "critical",
            "components": {
                "core": {"status": "offline", "score": 0.0},
                "models": {"status": "offline", "score": 0.0},
                "text_processing": {"status": "offline", "score": 0.0},
                "response_generation": {"status": "offline", "score": 0.0}
            },
            "stats": self.stats.copy()
        }
        
        try:
            # Проверяем ядро
            core_score = 0.0
            if self.ml_core:
                core_score = 0.9
                health["components"]["core"] = {"status": "online", "score": core_score}
            else:
                health["components"]["core"] = {"status": "offline", "score": core_score}
            
            # Проверяем модели
            models_score = 0.0
            if self.model_manager:
                try:
                    models = self.model_manager.get_available_models()
                    models_score = min(0.9, 0.2 + 0.7 * (len(models) / 5))
                    health["components"]["models"] = {
                        "status": "online" if models_score > 0.3 else "warning",
                        "score": models_score,
                        "count": len(models)
                    }
                except Exception:
                    health["components"]["models"] = {"status": "offline", "score": models_score}
            else:
                health["components"]["models"] = {"status": "offline", "score": models_score}
            
            # Проверяем обработку текста
            text_score = 0.0
            if self.text_processor:
                try:
                    # Проверяем базовую токенизацию
                    test_text = "Это тест."
                    analysis = self.text_processor.process_text(test_text)
                    tokens = analysis.get('tokens', []) if isinstance(analysis, dict) else []
                    text_score = 0.8 if tokens else 0.3
                    health["components"]["text_processing"] = {
                        "status": "online" if text_score > 0.5 else "warning",
                        "score": text_score
                    }
                except Exception:
                    health["components"]["text_processing"] = {"status": "offline", "score": text_score}
            else:
                health["components"]["text_processing"] = {"status": "offline", "score": text_score}
            
            # Проверяем генерацию ответов
            response_score = 0.0
            if self.response_generator and not self._is_training_mode():
                try:
                    # Проверяем базовую генерацию
                    response = self.response_generator.generate_response("Привет", max_length=50)
                    response_score = 0.8 if response.get("text") else 0.3
                    health["components"]["response_generation"] = {
                        "status": "online" if response_score > 0.5 else "warning",
                        "score": response_score
                    }
                except Exception:
                    health["components"]["response_generation"] = {"status": "offline", "score": response_score}
            else:
                health["components"]["response_generation"] = {"status": "offline", "score": response_score}
            
            # Вычисляем общий балл
            total_score = (core_score * 0.2 + 
                          models_score * 0.3 + 
                          text_score * 0.25 + 
                          response_score * 0.25)
            
            health["score"] = total_score
            
            # Определяем статус
            if total_score > 0.7:
                health["status"] = "healthy"
            elif total_score > 0.4:
                health["status"] = "warning"
            else:
                health["status"] = "critical"
                
            return health
            
        except Exception as e:
            logger.error(f"Ошибка проверки состояния системы: {e}", exc_info=True)
            return {
                "score": 0.0,
                "status": "critical",
                "error": str(e),
                "components": {},
                "stats": self.stats.copy()
            }

    def _is_training_mode(self) -> bool:
        """Возвращает True, если система находится в режиме обучения/тренировки.
        Проверяет флаг у brain и переменную окружения COGNIFLEX_TRAINING.
        """
        try:
            import os as _os
            env_flag = str(_os.environ.get("COGNIFLEX_TRAINING", "0")).lower() in ("1", "true", "yes")
        except Exception:
            env_flag = False
        brain_flag = False
        try:
            if self.brain is not None:
                brain_flag = bool(getattr(self.brain, "_in_training", False)) or bool(getattr(self.brain, "in_training", False))
        except Exception:
            brain_flag = False
        return bool(env_flag or brain_flag)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Возвращает метрики производительности.
        
        Returns:
            Dict[str, Any]: Метрики производительности
        """
        metrics = {
            "request_rate": self.stats["successful_requests"] / max(1, time.time() - self.stats["last_request_time"]),
            "success_rate": self.stats["successful_requests"] / max(1, self.stats["total_requests"]),
            "avg_processing_time": self.stats["total_processing_time"] / max(1, self.stats["successful_requests"]),
            "system_health": self.get_system_health()
        }
        
        # Добавляем метрики компонентов
        if self.text_processor and hasattr(self.text_processor, 'get_performance_metrics'):
            try:
                metrics["text_processor"] = self.text_processor.get_performance_metrics()
            except Exception as e:
                logger.debug(f"Не удалось получить метрики текстового процессора: {e}")
        
        if self.response_generator and hasattr(self.response_generator, 'get_performance_metrics'):
            try:
                metrics["response_generator"] = self.response_generator.get_performance_metrics()
            except Exception as e:
                logger.debug(f"Не удалось получить метрики генератора ответов: {e}")
        
        if self.model_manager and hasattr(self.model_manager, 'get_performance_metrics'):
            try:
                metrics["model_manager"] = self.model_manager.get_performance_metrics()
            except Exception as e:
                logger.debug(f"Не удалось получить метрики менеджера моделей: {e}")
        
        # Добавляем метрики кэша
        if self.hybrid_cache and hasattr(self.hybrid_cache, 'get_cache_stats'):
            try:
                metrics["cache"] = self.hybrid_cache.get_cache_stats()
            except Exception as e:
                logger.debug(f"Не удалось получить метрики кэша: {e}")
        
        return metrics
    
    def clear_cache(self, memory_only=False):
        """Очищает кэш."""
        if self.hybrid_cache and hasattr(self.hybrid_cache, 'clear_cache'):
            self.hybrid_cache.clear_cache(memory_only=memory_only)
            logger.info(f"Кэш {'только в памяти' if memory_only else 'полностью'} очищен")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Возвращает статистику кэша."""
        if self.hybrid_cache and hasattr(self.hybrid_cache, 'get_cache_stats'):
            return self.hybrid_cache.get_cache_stats()
        return {"error": "hybrid_cache_unavailable"}
    
    def update_stats(self, success: bool, response_time: float, tokens_processed: int):
        """
        Обновляет статистику обработки запросов.
        
        Args:
            success: Успешно ли обработан запрос
            response_time: Время обработки
            tokens_processed: Количество обработанных токенов
        """
        self.stats["total_requests"] += 1
        self.stats["total_processing_time"] += response_time
        self.stats["last_request_time"] = time.time()
        
        if success:
            self.stats["successful_requests"] += 1
        else:
            self.stats["failed_requests"] += 1
    
    def get_model_health(self, model_name: str) -> Dict[str, Any]:
        """
        Возвращает состояние конкретной модели.
        
        Args:
            model_name: Имя модели
            
        Returns:
            Dict[str, Any]: Состояние модели
        """
        if self.model_manager and hasattr(self.model_manager, 'get_model_health'):
            return self.model_manager.get_model_health(model_name)
        return {"error": "model_manager_unavailable"}
    
    def get_all_model_health(self) -> Dict[str, Dict[str, Any]]:
        """
        Возвращает состояние всех моделей.
        
        Returns:
            Dict[str, Dict[str, Any]]: Состояние всех моделей
        """
        if self.model_manager and hasattr(self.model_manager, 'get_all_model_health'):
            return self.model_manager.get_all_model_health()
        return {"error": "model_manager_unavailable"}
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Анализирует запрос и возвращает метаданные.
        
        Args:
            query: Текст запроса
            
        Returns:
            Dict[str, Any]: Метаданные запроса
        """
        try:
            # Обрабатываем текст
            analysis = self.process_text(query)
            
            # Формируем метаданные с проверкой типа analysis
            if isinstance(analysis, dict):
                # Если analysis - словарь, используем безопасное извлечение
                metadata = {
                    "language": analysis.get("language", "unknown"),
                    "sentiment": analysis.get("sentiment", "neutral"),
                    "keywords": analysis.get("keywords", [])[:5],
                    "entities": analysis.get("entities", [])[:5],
                    "token_count": analysis.get("token_count", 0),
                    "processing_time": analysis.get("processing_time", 0.0),
                    "from_cache": analysis.get("from_cache", False)
                }
            else:
                # Если analysis - объект, используем атрибуты
                metadata = {
                    "language": getattr(analysis, 'language', 'unknown'),
                    "sentiment": getattr(analysis, 'sentiment', 'neutral'),
                    "keywords": [kw.get("word", "") for kw in getattr(analysis, 'keywords', [])[:5]],
                    "entities": [ent.get("text", "") for ent in getattr(analysis, 'named_entities', [])[:5]],
                    "token_count": len(getattr(analysis, 'tokens', [])),
                    "processing_time": getattr(analysis, 'processing_time', 0.0),
                    "from_cache": getattr(analysis, 'from_cache', False)
                }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Ошибка анализа запроса: {e}", exc_info=True)
            return {
                "error": str(e),
                "language": "unknown",
                "token_count": 0
            }
    
    def get_component(self, component_name: str) -> Any:
        """
        Возвращает компонент по имени.
        
        Args:
            component_name: Имя компонента
            
        Returns:
            Any: Компонент или None
        """
        components = {
            "ml_core": self.ml_core,
            "model_manager": self.model_manager,
            "text_processor": self.text_processor,
            "response_generator": self.response_generator,
            "hybrid_cache": self.hybrid_cache,
            "token_streamer": self.token_streamer,
            "training_orchestrator": self.training_orchestrator
        }
        return components.get(component_name)

    def train_from_document(self, imported_doc: Any, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Запускает обучение/обновление графа знаний из переданного документа через TrainingOrchestrator."""
        try:
            if not self.training_orchestrator:
                return {"status": "error", "reason": "training_orchestrator_unavailable"}
            return self.training_orchestrator.train_from_document(imported_doc, model_id=model_id)
        except Exception as e:
            logger.error(f"Ошибка обучения из документа: {e}", exc_info=True)
            return {"status": "error", "reason": str(e)}
    
    def is_initialized(self) -> bool:
        """Проверяет, инициализирован ли MLUnit."""
        return self.initialized
    
    def is_running(self) -> bool:
        """Проверяет, запущен ли MLUnit."""
        return self.running
    
    def get_available_tasks(self) -> List[str]:
        """
        Возвращает список доступных задач.
        
        Returns:
            List[str]: Список задач
        """
        if self.model_manager:
            return self.model_manager.get_available_tasks()
        return []
    
    def get_model_for_task(self, task: str) -> Tuple[Any, Any, str]:
        """
        Возвращает модель для конкретной задачи.
        
        Args:
            task: Название задачи
            
        Returns:
            Tuple[Any, Any, str]: Модель, токенизатор, имя модели
        """
        if self.model_manager:
            return self.model_manager.get_model_for_task(task)
        raise RuntimeError("ModelManager недоступен")
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Возвращает информацию о модели.
        
        Args:
            model_name: Имя модели
            
        Returns:
            Dict[str, Any]: Информация о модели
        """
        if self.model_manager:
            return self.model_manager.get_model_info(model_name)
        return {"error": "model_manager_unavailable"}
    
    def get_supported_tasks(self) -> List[str]:
        """
        Возвращает список поддерживаемых задач.
        
        Returns:
            List[str]: Список задач
        """
        return [
            "text-generation",
            "text-classification",
            "token-classification",
            "question-answering",
            "summarization",
            "translation",
            "conversational",
            "sentiment-analysis",
            "knowledge_graph"
        ]
    
    def get_model_capabilities(self, model_name: str) -> Dict[str, Any]:
        """
        Возвращает возможности модели.
        
        Args:
            model_name: Имя модели
            
        Returns:
            Dict[str, Any]: Возможности модели
        """
        if self.model_manager:
            return self.model_manager.get_model_capabilities(model_name)
        return {"error": "model_manager_unavailable"}
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """
        Возвращает статистику по моделям.
        
        Returns:
            Dict[str, Any]: Статистика по моделям
        """
        if self.model_manager:
            return self.model_manager.get_model_statistics()
        return {"error": "model_manager_unavailable"}
    
    def get_text_processor(self):
        """Возвращает текстовый процессор."""
        return self.text_processor
    
    def get_response_generator(self):
        """Возвращает генератор ответов."""
        return self.response_generator
    
    def get_model_manager(self):
        """Возвращает менеджер моделей."""
        return self.model_manager
    
    def get_ml_core(self):
        """Возвращает ядро ML."""
        return self.ml_core
    
    def get_token_streamer(self):
        """Возвращает потоковый токенизатор."""
        return self.token_streamer
    
    def get_hybrid_cache(self):
        """Возвращает гибридный кэш."""
        return self.hybrid_cache
    
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику MLUnit."""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Сбрасывает статистику."""
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
            "last_request_time": 0.0
        }
    
    def get_detailed_health_report(self) -> Dict[str, Any]:
        """
        Возвращает детальный отчет о состоянии системы.
        
        Returns:
            Dict[str, Any]: Детальный отчет
        """
        report = {
            "timestamp": time.time(),
            "system_health": self.get_system_health(),
            "performance_metrics": self.get_performance_metrics(),
            "model_health": self.get_all_model_health(),
            "cache_stats": self.get_cache_stats(),
            "statistics": self.get_statistics()
        }
        
        # Добавляем информацию о компонентах
        report["components"] = {
            "ml_core": bool(self.ml_core),
            "model_manager": bool(self.model_manager),
            "text_processor": bool(self.text_processor),
            "response_generator": bool(self.response_generator),
            "hybrid_cache": bool(self.hybrid_cache),
            "token_streamer": bool(self.token_streamer)
        }
        
        return report
    
    def save_state(self, file_path: str) -> bool:
        """
        Сохраняет состояние MLUnit в файл.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            bool: Успешно ли сохранено
        """
        try:
            state = {
                "timestamp": time.time(),
                "system_health": self.get_system_health(),
                "statistics": self.stats.copy(),
                "model_health": self.get_all_model_health(),
                "cache_stats": self.get_cache_stats()
            }
            
            # Сохраняем состояние кэша
            if self.hybrid_cache and hasattr(self.hybrid_cache, 'export_ethics_data'):
                try:
                    self.hybrid_cache.export_ethics_data(file_path + ".cache")
                except Exception as e:
                    logger.error(f"Ошибка экспорта кэша: {e}")
            
            # Сохраняем основное состояние
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Состояние MLUnit сохранено в {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка сохранения состояния MLUnit: {e}", exc_info=True)
            return False
    
    def load_state(self, file_path: str) -> bool:
        """
        Загружает состояние MLUnit из файла.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            bool: Успешно ли загружено
        """
        try:
            # Загружаем основное состояние
            with open(file_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # Восстанавливаем статистику
            if "statistics" in state:
                self.stats = state["statistics"]
            
            # Загружаем состояние кэша
            cache_file = file_path + ".cache"
            if os.path.exists(cache_file) and self.hybrid_cache and hasattr(self.hybrid_cache, 'import_ethics_data'):
                try:
                    self.hybrid_cache.import_ethics_data(cache_file)
                except Exception as e:
                    logger.error(f"Ошибка импорта кэша: {e}")
            
            logger.info(f"Состояние MLUnit загружено из {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки состояния MLUnit: {e}", exc_info=True)
            return False
    
    def get_component_versions(self) -> Dict[str, str]:
        """
        Возвращает версии компонентов.
        
        Returns:
            Dict[str, str]: Версии компонентов
        """
        versions = {
            "ml_unit": "2.0",
            "ml_core": "2.0" if self.ml_core else "N/A",
            "model_manager": "2.0" if self.model_manager else "N/A",
            "text_processor": "2.0" if self.text_processor else "N/A",
            "response_generator": "2.0" if self.response_generator else "N/A",
            "hybrid_cache": "2.0" if self.hybrid_cache else "N/A",
            "token_streamer": "2.0" if self.token_streamer else "N/A"
        }
        
        # Пытаемся получить версии из компонентов
        try:
            if self.ml_core and hasattr(self.ml_core, 'get_version'):
                versions["ml_core"] = self.ml_core.get_version()
        except:
            pass
        
        try:
            if self.model_manager and hasattr(self.model_manager, 'get_version'):
                versions["model_manager"] = self.model_manager.get_version()
        except:
            pass
        
        try:
            if self.text_processor and hasattr(self.text_processor, 'get_version'):
                versions["text_processor"] = self.text_processor.get_version()
        except:
            pass
        
        try:
            if self.response_generator and hasattr(self.response_generator, 'get_version'):
                versions["response_generator"] = self.response_generator.get_version()
        except:
            pass
        
        return versions
    
    def get_extended_system_info(self) -> Dict[str, Any]:
        """
        Возвращает расширенную информацию о системе.
        
        Returns:
            Dict[str, Any]: Расширенная информация
        """
        info = {
            "version": self.get_component_versions(),
            "system_health": self.get_system_health(),
            "performance_metrics": self.get_performance_metrics(),
            "available_tasks": self.get_available_tasks(),
            "supported_tasks": self.get_supported_tasks(),
            "model_statistics": self.get_model_statistics(),
            "cache_stats": self.get_cache_stats(),
            "statistics": self.get_statistics()
        }
        
        # Добавляем информацию о доступных моделях
        try:
            info["available_models"] = self.get_available_models()
        except Exception as e:
            info["available_models_error"] = str(e)
        
        return info
    
    def warmup_models(self, tasks: Optional[List[str]] = None):
        """
        Прогревает модели для указанных задач.
        
        Args:
            tasks: Список задач для прогрева
        """
        if not self.model_manager:
            logger.warning("ModelManager недоступен, прогрев моделей невозможен")
            return
        
        tasks = tasks or self.get_supported_tasks()
        
        for task in tasks:
            try:
                logger.info(f"Прогрев модели для задачи: {task}")
                model, tokenizer, model_name = self.model_manager.get_model_for_task(task)
                
                # Создаем тестовый ввод в зависимости от задачи
                if task == "text-generation":
                    test_input = "Привет, как дела?"
                elif task == "sentiment-analysis":
                    test_input = "Я очень доволен этой системой!"
                else:
                    test_input = "Тестовый текст для прогрева модели."
                
                # Прогреваем модель
                self.model_manager.warmup_model(task, test_input)
                
            except Exception as e:
                logger.error(f"Ошибка прогрева модели для задачи {task}: {e}")
    
    def get_model_capabilities_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Возвращает сводку возможностей моделей.
        
        Returns:
            Dict[str, Dict[str, Any]]: Сводка возможностей
        """
        summary = {}
        
        if not self.model_manager:
            return {"error": "model_manager_unavailable"}
        
        try:
            available_models = self.model_manager.get_available_models()
            
            for model in available_models:
                model_name = model["alias"]
                capabilities = self.get_model_capabilities(model_name)
                summary[model_name] = capabilities
            
            return summary
            
        except Exception as e:
            logger.error(f"Ошибка получения сводки возможностей моделей: {e}")
            return {"error": str(e)}
    
    def get_tokenization_info(self, text: str, task: str = "text-generation") -> Dict[str, Any]:
        """
        Возвращает информацию о токенизации текста для указанной задачи.
        
        Args:
            text: Текст для токенизации
            task: Задача
            
        Returns:
            Dict[str, Any]: Информация о токенизации
        """
        try:
            # Получаем модель и токенизатор
            model, tokenizer, model_name = self.get_model_for_task(task)
            
            # Токенизируем текст
            if hasattr(tokenizer, "tokenize"):
                tokens = tokenizer.tokenize(text)
            elif hasattr(tokenizer, "encode"):
                tokens = [str(i) for i in tokenizer.encode(text, add_special_tokens=False)]
            else:
                tokens = text.split()
            
            # Получаем информацию о токенах
            token_info = {
                "token_count": len(tokens),
                "tokens": tokens[:50],  # Первые 50 токенов
                "model": model_name,
                "task": task,
                "truncated": len(tokens) > tokenizer.model_max_length if hasattr(tokenizer, "model_max_length") else False
            }
            
            return token_info
            
        except Exception as e:
            logger.error(f"Ошибка получения информации о токенизации: {e}")
            return {
                "error": str(e),
                "token_count": 0,
                "tokens": [],
                "truncated": False
            }
    
    def get_optimal_parameters(self, task: str, text_length: int) -> Dict[str, Any]:
        """
        Возвращает оптимальные параметры для генерации на основе длины текста.
        
        Args:
            task: Задача
            text_length: Длина текста
            
        Returns:
            Dict[str, Any]: Оптимальные параметры
        """
        # Базовые параметры
        params = {
            "max_length": min(512, text_length * 2),
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
        
        # Настройки в зависимости от задачи
        if task == "text-generation":
            params["max_length"] = min(1024, max(200, text_length * 3))
        elif task == "summarization":
            params["max_length"] = min(300, max(100, text_length // 3))
        elif task == "translation":
            params["max_length"] = min(512, max(200, text_length * 2))
        elif task == "sentiment-analysis":
            params["max_length"] = 128
            params["do_sample"] = False
        
        return params
    
    def generate_with_optimal_params(self, prompt: str, task: str = "text-generation") -> Dict[str, Any]:
        """
        Генерирует ответ с оптимальными параметрами для задачи.
        
        Args:
            prompt: Текст запроса
            task: Задача
            
        Returns:
            Dict[str, Any]: Структурированный ответ
        """
        # Получаем оптимальные параметры
        text_length = len(prompt.split())
        params = self.get_optimal_parameters(task, text_length)
        
        # Генерируем ответ
        return self.generate_response(
            prompt=prompt,
            max_length=params["max_length"],
            temperature=params["temperature"],
            top_p=params["top_p"],
            task=task,
            do_sample=params["do_sample"]
        )
    
    def add_model(self, model_id: str, model_instance: Any):
        """Добавляет модель в MLUnit."""
        if self.model_manager:
            self.model_manager.models[model_id] = model_instance
            logger.debug(f"Модель {model_id} добавлена в MLUnit")
            
            # Проверяем, можем ли мы использовать эту модель
            if not hasattr(self, 'active_model_id') or not self.active_model_id:
                self.active_model_id = model_id
                logger.info(f"Установлена активная модель: {model_id}")
    
    def analyze_model_performance(self, model_name: str, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Анализирует производительность модели на тестовых случаях.
        
        Args:
            model_name: Имя модели
            test_cases: Тестовые случаи
            
        Returns:
            Dict[str, Any]: Результаты анализа
        """
        if not self.model_manager:
            return {"error": "model_manager_unavailable"}
        
        results = {
            "model": model_name,
            "test_cases": len(test_cases),
            "success_count": 0,
            "failure_count": 0,
            "avg_processing_time": 0.0,
            "tokenization_stats": {
                "avg_tokens": 0,
                "max_tokens": 0,
                "min_tokens": float("inf")
            },
            "detailed_results": []
        }
        
        total_time = 0.0
        total_tokens = 0
        token_counts = []
        
        for i, test_case in enumerate(test_cases):
            start_time = time.time()
            try:
                # Генерируем ответ
                response = self.generate_response(
                    prompt=test_case["input"],
                    task=test_case.get("task", "text-generation"),
                    max_length=test_case.get("max_length", 200)
                )
                
                # Анализируем результат
                processing_time = time.time() - start_time
                token_count = len(response.get("tokens", []))
                
                # Сохраняем детали
                result = {
                    "case_id": i,
                    "input": test_case["input"][:50] + "..." if len(test_case["input"]) > 50 else test_case["input"],
                    "output": response.get("text", "")[:50] + "..." if response.get("text") and len(response.get("text", "")) > 50 else response.get("text", ""),
                    "processing_time": processing_time,
                    "token_count": token_count,
                    "success": bool(response.get("text")),
                    "expected_output": test_case.get("expected_output", "")[:50] + "..." if test_case.get("expected_output") and len(test_case.get("expected_output", "")) > 50 else test_case.get("expected_output", "")
                }
                
                results["detailed_results"].append(result)
                
                # Обновляем статистику
                if response.get("text"):
                    results["success_count"] += 1
                    total_time += processing_time
                    total_tokens += token_count
                    token_counts.append(token_count)
                else:
                    results["failure_count"] += 1
                
                logger.debug(f"Тестовый случай {i+1}/{len(test_cases)}: {'успех' if response.get('text') else 'ошибка'} за {processing_time:.2f}с")
                
            except Exception as e:
                results["failure_count"] += 1
                logger.error(f"Ошибка в тестовом случае {i+1}: {e}")
        
        # Вычисляем итоговые метрики
        if results["success_count"] > 0:
            results["avg_processing_time"] = total_time / results["success_count"]
            results["tokenization_stats"]["avg_tokens"] = total_tokens / results["success_count"]
            results["tokenization_stats"]["max_tokens"] = max(token_counts) if token_counts else 0
            results["tokenization_stats"]["min_tokens"] = min(token_counts) if token_counts else 0
        
        # Добавляем общий статус
        results["status"] = "success" if results["success_count"] > results["failure_count"] else "warning" if results["success_count"] > 0 else "failure"
        
        return results
    
    def get_token_streamer(self):
        """Возвращает потоковый токенизатор."""
        return self.token_streamer
    
    def get_hybrid_cache(self):
        """Возвращает гибридный кэш."""
        return self.hybrid_cache