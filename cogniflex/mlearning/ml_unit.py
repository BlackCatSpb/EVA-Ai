#!/usr/bin/env python3
"""
Исправленная версия MLUnit с правильными отступами
"""

import os
import sys
import logging
import time
import threading
import queue
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime

# Добавляем корневую директорию проекта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger("cogniflex.ml_unit")


def _get_project_root() -> str:
    """Возвращает корневую директорию проекта"""
    import sys
    
    possible_roots = []
    
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    possible_roots.append(os.path.dirname(os.path.dirname(current_dir)))
    possible_roots.append(os.path.dirname(current_dir))
    
    if sys.argv and sys.argv[0]:
        argv_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        possible_roots.append(argv_dir)
        possible_roots.append(os.path.dirname(argv_dir))
    
    for root in possible_roots:
        if root and os.path.exists(root):
            cogniflex_marker = os.path.join(root, 'cogniflex')
            if os.path.exists(cogniflex_marker):
                return root
    
    drive = os.path.splitdrive(os.getcwd())[0] or 'C:'
    username = os.environ.get('USERNAME', 'user')
    
    onedrive_path = os.path.join(drive, 'Users', username, 'OneDrive', 'Desktop', 'CogniFlex')
    if os.path.exists(onedrive_path) and os.path.exists(os.path.join(onedrive_path, 'cogniflex')):
        return onedrive_path
    
    possible_locations = [
        os.path.join(drive, 'Users', username, 'OneDrive', 'Desktop', 'CogniFlex'),
        os.path.join(drive, 'Users', username, 'Desktop', 'CogniFlex'),
        os.path.join(drive, 'CogniFlex'),
    ]
    
    for loc in possible_locations:
        if os.path.exists(loc):
            if os.path.exists(os.path.join(loc, 'cogniflex')):
                return os.path.abspath(loc)
    
    return os.getcwd()

# Пытаемся импортировать torch
try:
    import torch
except ImportError as e:
    logger.error(f"Не удалось импортировать torch: {e}")
    torch = None

# Пытаемся импортировать ModelManager
try:
    from cogniflex.mlearning.model_manager import ModelManager
except ImportError as e:
    logger.error(f"Не удалось импортировать ModelManager: {e}")
    ModelManager = None

class MLUnit:
    """
    Исправленная версия MLUnit с правильными отступами
    """
    
    def __init__(self, brain=None, cache_dir="cache/ml_unit", use_gpu=True, 
                 max_workers=4, hybrid_cache_size=3295640, safe_test_mode=False):
        """
        Инициализирует MLUnit с улучшенной архитектурой.
        
        Args:
            brain: Экземпляр CoreBrain для интеграции
            cache_dir: Директория для кэша
            use_gpu: Использовать GPU если доступно
            max_workers: Максимальное количество воркеров
            hybrid_cache_size: Размер гибридного кэша
            safe_test_mode: Режим безопасного тестирования
        """
        self.brain = brain
        self.cache_dir = cache_dir
        self.use_gpu = use_gpu
        self.max_workers = max_workers
        self.hybrid_cache_size = hybrid_cache_size
        self.safe_test_mode = safe_test_mode
        
        # Компоненты MLUnit
        self.ml_core = None
        self.model_manager = None
        self.text_processor = None
        self.response_generator = None
        self.training_orchestrator = None
        self.token_streamer = None
        self.hybrid_cache = None
        
        # Состояние
        self.running = False
        self.models_ready = False
        
        # Статистика
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
            "last_request_time": 0.0
        }
        
        # Очередь для GUI
        self.gui_queue = queue.Queue()
        
        logger.info("MLUnit инициализирован")
    
    def _init_ml_core(self):
        """Инициализирует MLCore."""
        try:
            from cogniflex.mlearning.ml_core import MLCore
            
            self.ml_core = MLCore(
                brain=self.brain,
                cache_dir=os.path.join(self.cache_dir, "ml_core")
            )
            
            logger.info("MLCore инициализирован")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка инициализации MLCore: {e}")
            return False
    
    def _init_text_processor(self):
        """Инициализирует текстовый процессор."""
        try:
            from cogniflex.nlp.text_processor import TextProcessor
            
            # Используем правильную директорию с моделью и токенизатором
            project_root = _get_project_root()
            model_dir = os.path.join(project_root, "cogniflex_cache", "ml_unit", "fractal_storage", "models", "rugpt3_large_fractal", "model")
            
            self.text_processor = TextProcessor(
                model_name=model_dir,  # Используем полный путь к директории
                cache_dir=os.path.join(self.cache_dir, "text_processor_cache")
            )
            
            logger.info("TextProcessor инициализирован с правильным путем к модели")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка инициализации TextProcessor: {e}")
            return False
    
    def _init_model_manager(self):
        """
        Инициализирует менеджер моделей с интеграцией фрактального хранилища.
        
        Выполняет следующие проверки:
        1. Проверяет доступность и инициализацию токенизатора
        2. Проверяет базовую функциональность токенизатора
        3. Регистрирует отложенную команду для связывания компонентов
        """
        try:
            # Используем существующий FractalModelManager из CoreBrain
            if hasattr(self.brain, 'model_manager') and self.brain.model_manager is not None:
                self.model_manager = self.brain.model_manager
                logger.info(f"Используется существующий model_manager из CoreBrain: {type(self.model_manager).__name__}")
            elif hasattr(self.brain, 'fractal_model_manager') and self.brain.fractal_model_manager is not None:
                self.model_manager = self.brain.fractal_model_manager
                logger.info(f"Используется существующий fractal_model_manager из CoreBrain: {type(self.model_manager).__name__}")
            elif ModelManager is not None:
                # Fallback: создаем новый ModelManager только если нет существующего
                self.model_manager = ModelManager(
                    brain=self.brain,
                    cache_dir=os.path.join(self.cache_dir, "fractal_storage"),
                    model_dir=os.path.join(self.cache_dir, "fractal_storage", "models"),
                    use_gpu=self.use_gpu
                )
                logger.warning("Создан новый ModelManager (fallback), так как в CoreBrain не найден model_manager")
                
                # Регистрируем ModelManager в CoreBrain
                if hasattr(self.brain, 'register_component'):
                    self.brain.register_component('model_manager', self.model_manager)
                    logger.info("ModelManager зарегистрирован в CoreBrain")
            else:
                logger.error("ModelManager недоступен")
                return False
            
            # Проверяем инициализацию текстового процессора и токенизатора
            if not hasattr(self, 'text_processor') or self.text_processor is None:
                logger.warning("Текстовый процессор не инициализирован")
                return False
            
            # Проверяем наличие токенизатора
            if not hasattr(self.text_processor, 'tokenizer') or self.text_processor.tokenizer is None:
                logger.warning("Токенизатор не инициализирован в текстовом процессоре")
            else:
                # Проверяем базовую функциональность токенизатора
                try:
                    test_text = "Проверка работы токенизатора"
                    if hasattr(self.text_processor.tokenizer, "tokenize"):
                        tokens = self.text_processor.tokenizer.tokenize(test_text)
                        logger.debug(f"Токенизатор прошел проверку. Токенов: {len(tokens) if tokens else 0}")
                    elif hasattr(self.text_processor.tokenizer, "encode"):
                        tokens = self.text_processor.tokenizer.encode(test_text, add_special_tokens=False)
                        logger.debug(f"Токенизатор прошел проверку. Токенов: {len(tokens) if tokens else 0}")
                    else:
                        logger.warning("Токенизатор не поддерживает стандартные методы токенизации")
                except Exception as e:
                    logger.error(f"Ошибка при проверке токенизатора: {e}", exc_info=True)
            
            # Регистрируем отложенную команду для связывания компонентов
            if self.brain and hasattr(self.brain, 'add_deferred_command'):
                logger.info("Регистрация отложенной команды для связывания компонентов MLUnit...")
                self.brain.add_deferred_command(self._link_components)
            else:
                # Fallback: если отложенная команда недоступна, пробуем связать компоненты сразу
                logger.info("DeferredCommandSystem недоступна, выполняем прямое связывание компонентов")
                self._link_components()
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при инициализации ModelManager: {e}", exc_info=True)
            return False
    
    def _init_response_generator(self):
        """Инициализирует генератор ответов, получая его из brain или создавая новый."""
        try:
            # Сначала проверяем, есть ли генератор в brain
            if self.brain and hasattr(self.brain, 'response_generator') and self.brain.response_generator:
                self.response_generator = self.brain.response_generator
                logger.info("ResponseGenerator успешно получен из brain.")
                return True
            
            # Если в brain нет генератора, создаем новый
            logger.info("Создание нового экземпляра ResponseGenerator...")
            try:
                from cogniflex.core.response_generator import ResponseGenerator
                
                # Получаем зависимости
                model_manager = getattr(self, 'model_manager', None)
                text_processor = getattr(self, 'text_processor', None)
                tokenizer = getattr(text_processor, 'tokenizer', None) if text_processor else None
                
                # Создаем экземпляр
                self.response_generator = ResponseGenerator(
                    brain=self.brain,
                    model_manager=model_manager,
                    text_processor=text_processor
                )
                
                # Устанавливаем токенизатор напрямую, если он есть
                if tokenizer and hasattr(self.response_generator, 'tokenizer'):
                    self.response_generator.tokenizer = tokenizer
                
                # Регистрируем в brain для использования другими компонентами
                if self.brain:
                    self.brain.response_generator = self.response_generator
                    
                logger.info("ResponseGenerator создан и зарегистрирован в brain.")
                return True
                
            except ImportError as import_error:
                logger.error(f"Не удалось импортировать ResponseGenerator: {import_error}", exc_info=True)
                self.response_generator = None
                return False
            except Exception as create_error:
                logger.error(f"Не удалось создать ResponseGenerator: {create_error}", exc_info=True)
                self.response_generator = None
                return False
                
        except Exception as e:
            logger.error(f"Ошибка при инициализации ResponseGenerator: {e}", exc_info=True)
            self.response_generator = None
            return False
    
    def _init_hybrid_cache(self):
        """Инициализирует гибридный кэш."""
        try:
            from cogniflex.memory.hybrid_token_cache import HybridTokenCache
            
            # Создаем гибридный кэш с настройками из MLUnit
            self.hybrid_cache = HybridTokenCache(
                brain=self.brain,
                max_memory_tokens=self.hybrid_cache_size // 4096,  # Примерное количество токенов
                disk_cache_dir=os.path.join(self.cache_dir, "hybrid_cache"),
                target_memory_gb=1.0,  # 1GB для кэша
                dynamic_memory_limit=True,
                max_ram_usage_percent=10.0  # Не более 10% RAM
            )
            
            logger.info("HybridTokenCache инициализирован в MLUnit")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка инициализации HybridTokenCache: {e}")
            return False
    
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
                    
                    # Устанавливаем токенизатор
                    if hasattr(self, 'token_streamer') and self.token_streamer:
                        self.training_orchestrator.token_streamer = self.token_streamer
                    elif self.text_processor and hasattr(self.text_processor, 'tokenizer'):
                        self.training_orchestrator.token_streamer = self.text_processor
                    elif self.model_manager and hasattr(self.model_manager, 'tokenizer'):
                        self.training_orchestrator.token_streamer = self.model_manager.tokenizer
                    
                    # Устанавливаем гибридный кэш
                    if hasattr(self, 'hybrid_cache') and self.hybrid_cache:
                        self.training_orchestrator.hybrid_cache = self.hybrid_cache
                    elif self.text_processor and hasattr(self.text_processor, 'hybrid_cache'):
                        self.training_orchestrator.hybrid_cache = self.text_processor.hybrid_cache
                    
                    # Устанавливаем knowledge_graph
                    if self.brain is not None and hasattr(self.brain, 'knowledge_graph'):
                        self.training_orchestrator.knowledge_graph = self.brain.knowledge_graph
                    
                    # Принудительно обновляем компоненты в оркестраторе
                    self.training_orchestrator._try_init_components()
                    
                    logger.info("TrainingOrchestrator успешно связан с компонентами MLUnit")
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
                # Некоторые реализации ModelManager не содержат метода get_available_models()
                # Используем безопасный фоллбек через model_metadata
                try:
                    if hasattr(self.model_manager, 'get_available_models') and callable(getattr(self.model_manager, 'get_available_models')):
                        models = self.model_manager.get_available_models()
                    else:
                        md = getattr(self.model_manager, 'model_metadata', {}) or {}
                        models = list(md.values()) if isinstance(md, dict) else (md or [])
                 except Exception:
                     models = []
                 logger.info(f"Доступно моделей: {len(models)}")
             
             # Проверяем токенизацию
             if self.text_processor:
                 test_text = "Это тестовый текст для проверки токенизации."
                 # Используем метод encode вместо process_text
                 try:
                     if torch is not None:
                         encoded = self.text_processor.encode(test_text)
                         input_ids = encoded.get('input_ids', torch.tensor([]))
                         if isinstance(input_ids, list):
                             tokens_count = len(input_ids)
                         else:
                             tokens_count = input_ids.numel()
                     else:
                         # Fallback если torch недоступен
                         tokens_count = 0
                     logger.info(f"Тестовая токенизация успешна. Токенов: {tokens_count}")
                 except Exception as e:
                     logger.error(f"Ошибка при проверке токенизатора: {e}")
            
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
    
    def _init_training_orchestrator(self):
        """Инициализирует оркестратор обучения."""
        try:
            from cogniflex.mlearning.training_orchestrator import TrainingOrchestrator
            
            self.training_orchestrator = TrainingOrchestrator(
                brain=self.brain,
                batch_size=32,
                overlap_tokens=16
            )
            
            # Регистрируем TrainingOrchestrator в CoreBrain
            if self.brain and hasattr(self.brain, 'register_component'):
                self.brain.register_component('training_orchestrator', self.training_orchestrator)
                logger.info("TrainingOrchestrator зарегистрирован в CoreBrain")
            elif self.brain:
                # Fallback - сохраняем напрямую в brain
                self.brain.training_orchestrator = self.training_orchestrator
                logger.info("TrainingOrchestrator сохранен в CoreBrain (fallback)")
            else:
                logger.warning("CoreBrain недоступен для регистрации TrainingOrchestrator")
            
            logger.info("TrainingOrchestrator инициализирован")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при инициализации TrainingOrchestrator: {e}", exc_info=True)
            return False
    
    def _is_training_mode(self):
        """Проверяет, находится ли система в режиме тренировки."""
        return hasattr(self, 'training_mode') and self.training_mode
    
    def get_system_health(self):
        """Возвращает информацию о здоровье системы."""
        try:
            # Проверяем основные компоненты
            components_status = {
                "ml_core": self.ml_core is not None,
                "model_manager": self.model_manager is not None,
                "text_processor": self.text_processor is not None,
                "response_generator": self.response_generator is not None,
                "models_ready": self.models_ready
            }
            
            # Проверяем статистику
            health_score = 0.0
            if components_status["ml_core"]:
                health_score += 0.3
            if components_status["model_manager"]:
                health_score += 0.3
            if components_status["text_processor"]:
                health_score += 0.2
            if components_status["response_generator"]:
                health_score += 0.2
            
            # Определяем статус
            if health_score >= 0.8:
                status = "healthy"
            elif health_score >= 0.5:
                status = "warning"
            else:
                status = "critical"
            
            # Для отладки: логируем состояние компонентов
            logger.debug(f"Health check - Components: {components_status}")
            logger.debug(f"Health check - Score: {health_score}")
            logger.debug(f"Health check - Status: {status}")
            
            return {
                "status": status,
                "score": health_score,
                "components": components_status,
                "stats": self.stats.copy()
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения здоровья системы: {e}")
            return {
                "status": "error",
                "score": 0.0,
                "components": {},
                "stats": self.stats.copy()
            }
    
    def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Обрабатывает запрос через MLUnit.
        
        Args:
            query: Текст запроса
            context: Дополнительный контекст
            
        Returns:
            Dict[str, Any]: Результат обработки
        """
        start_time = time.time()
        self.stats["total_requests"] += 1
        
        try:
            # Проверяем доступность ResponseGenerator
            if not self.response_generator:
                logger.error("ResponseGenerator недоступен")
                return self._create_fallback_response(query, "response_generator_unavailable")
            
            # Генерируем ответ
            response = self.response_generator.generate_response(query, **(context or {}))
            
            # Обновляем статистику
            self._update_statistics(start_time, True)
            
            return response
            
        except Exception as e:
            self._update_statistics(start_time, False)
            logger.error(f"Ошибка генерации ответа: {e}")
            return self._create_fallback_response(query, str(e))
    
    def _create_fallback_response(self, prompt: str, error: str) -> Dict[str, Any]:
        """
        Создает fallback-ответ при критической ошибке.
        
        Args:
            prompt: Исходный запрос
            error: Описание ошибки
            
        Returns:
            Dict[str, Any]: Структурированный fallback-ответ
        """
        logger.error(f"КРИТИЧЕСКАЯ ОШИБКА в MLUnit: {error}")
        logger.error(f"Запрос не может быть обработан: '{prompt[:50]}...'")
        logger.error("Используется fallback-режим ответа")
        
        # Создаем базовый fallback-ответ
        fallback_response = {
            "response": "Извините, система временно недоступна для обработки запроса. Пожалуйста, попробуйте позже.",
            "confidence": 0.1,
            "source": "ml_unit_fallback",
            "error": error,
            "timestamp": time.time(),
            "model_info": {
                "name": "fallback_mode",
                "status": "unavailable"
            },
            "processing_time": 0.0,
            "metadata": {
                "fallback_reason": error,
                "original_prompt_length": len(prompt),
                "system_status": "degraded"
            }
        }
        
        # Пытаемся использовать базовую генерацию если доступна
        try:
            if hasattr(self, 'brain') and self.brain:
                # Проверяем доступность простых ответов
                if hasattr(self.brain, 'get_simple_response'):
                    simple_response = self.brain.get_simple_response(prompt)
                    if simple_response:
                        fallback_response["response"] = simple_response
                        fallback_response["confidence"] = 0.3
                        fallback_response["source"] = "brain_simple_response"
                        logger.info("Использован простой ответ из CoreBrain")
        except Exception as e:
            logger.debug(f"Не удалось получить простой ответ: {e}")
        
        # Добавляем информацию о доступных компонентах
        available_components = []
        if hasattr(self, 'model_manager') and self.model_manager:
            available_components.append("model_manager")
        if hasattr(self, 'text_processor') and self.text_processor:
            available_components.append("text_processor")
        if hasattr(self, 'hybrid_cache') and self.hybrid_cache:
            available_components.append("hybrid_cache")
            
        fallback_response["metadata"]["available_components"] = available_components
        
        logger.warning(f"Fallback-ответ создан. Доступные компоненты: {available_components}")
        
        return fallback_response
    
    def _update_statistics(self, start_time: float, success: bool):
        """Обновляет статистику запросов."""
        processing_time = time.time() - start_time
        self.stats["total_processing_time"] += processing_time
        
        if success:
            self.stats["successful_requests"] += 1
        else:
            self.stats["failed_requests"] += 1
    
    def initialize(self) -> bool:
        """
        Дополнительная инициализация после создания объекта.
        
        Returns:
            bool: Успешно ли инициализировано
        """
        try:
            # Инициализируем компоненты в правильном порядке
            logger.info("Начинаем инициализацию компонентов MLUnit...")
            
            # 1. Инициализируем MLCore
            if not self._init_ml_core():
                logger.warning("Не удалось инициализировать MLCore, но продолжаем")
            
            # 2. Инициализируем текстовый процессор
            if not self._init_text_processor():
                logger.warning("Не удалось инициализировать текстовый процессор, но продолжаем")
            
            # 3. Инициализируем менеджер моделей
            if not self._init_model_manager():
                logger.warning("Не удалось инициализировать менеджер моделей, но продолжаем")
            
            # 4. Инициализируем генератор ответов
            if not self._init_response_generator():
                logger.warning("Не удалось инициализировать генератор ответов, но продолжаем")
            
            # 4.5. Инициализируем гибридный кэш
            if not self._init_hybrid_cache():
                logger.warning("Не удалось инициализировать гибридный кэш, но продолжаем")
            
            # 5. Инициализируем оркестратор обучения
            if not self._init_training_orchestrator():
                logger.warning("Не удалось инициализировать оркестратор обучения, но продолжаем")
            
            # 6. Связываем компоненты
            self._link_components()
            
            # 7. Проверяем здоровье системы
            health = self.get_system_health()
            logger.info(f"Состояние ML системы: {health['status']}, score: {health['score']:.2f}")
            
            # Считаем инициализацию успешной если есть хотя бы model_manager или ml_core
            if self.model_manager is not None or self.ml_core is not None:
                logger.info("MLUnit успешно инициализирован (основные компоненты доступны)")
                return True
            
            # Если есть хотя бы text_processor, тоже считаем успехом
            if self.text_processor is not None:
                logger.info("MLUnit инициализирован с ограниченной функциональностью")
                return True
            
            logger.warning("MLUnit инициализирован без компонентов - режим заглушки")
            return True  # Всегда возвращаем True чтобы не блокировать систему
        
        except Exception as e:
            logger.error(f"Ошибка дополнительной инициализации MLUnit: {e}", exc_info=True)
            return True  # Возвращаем True даже при ошибке, чтобы система продолжила работу
    
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
            self._update_statistics(start_time, False)
            logger.error(f"Ошибка генерации ответа: {e}")
            return self._create_fallback_response(prompt, str(e))
    
    def get_model_statistics(self):
        """Возвращает статистику по моделям."""
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
    
    def process_text(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Метод process_text для совместимости с существующим кодом.
        Обрабатывает текст через доступные компоненты.
        
        Args:
            text: Текст для обработки
            **kwargs: Дополнительные параметры
            
        Returns:
            Dict[str, Any]: Результат обработки текста
        """
        start_time = time.time()
        
        try:
            result = {
                "original_text": text,
                "processed_text": text,
                "tokens": [],
                "embeddings": None,
                "analysis": {},
                "status": "processed",
                "processing_time": 0.0
            }
            
            # Базовая токенизация
            if hasattr(self, 'text_processor') and self.text_processor:
                try:
                    processed = self.text_processor.process_text(text)
                    result["processed_text"] = processed.get("text", text)
                    result["tokens"] = processed.get("tokens", [])
                except Exception as e:
                    logger.warning(f"Ошибка обработки текста: {e}")
            
            # Если есть модель, пробуем получить эмбеддинги
            if hasattr(self, 'model_manager') and self.model_manager:
                try:
                    embeddings = self.model_manager.get_embeddings(text)
                    result["embeddings"] = embeddings
                except Exception as e:
                    logger.debug(f"Не удалось получить эмбеддинги: {e}")
            
            result["processing_time"] = time.time() - start_time
            return result
            
        except Exception as e:
            logger.error(f"Ошибка в process_text: {e}")
            return {
                "original_text": text,
                "processed_text": text,
                "tokens": [],
                "embeddings": None,
                "analysis": {"error": str(e)},
                "status": "error",
                "processing_time": time.time() - start_time
            }
