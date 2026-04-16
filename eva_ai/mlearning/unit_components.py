"""Component management and coordination for MLUnit"""
from __future__ import annotations

import os
import sys
import json
import logging
import time
import threading
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime

logger = logging.getLogger("eva_ai.ml_unit")

try:
    import torch
except ImportError:
    torch = None


def _init_ml_core(self):
    """Инициализирует MLCore."""
    try:
        from eva_ai.mlearning.ml_core import MLCore
        
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
        from eva_ai.nlp.text_processor import TextProcessor
        
        project_root = self._get_project_root()
        model_dir = os.path.join(project_root, "eva", "mlearning", "eva_models", "qwen3.5-0.8b")
        
        if not os.path.exists(model_dir):
            logger.warning(f"Модель не найдена: {model_dir}")
            return False
        
        self.text_processor = TextProcessor(
            model_name=model_dir,
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
    """
    try:
        if hasattr(self.brain, 'model_manager') and self.brain.model_manager is not None:
            self.model_manager = self.brain.model_manager
            logger.info(f"Используется существующий model_manager из CoreBrain: {type(self.model_manager).__name__}")
            return True
        elif hasattr(self.brain, 'fractal_model_manager') and self.brain.fractal_model_manager is not None:
            self.model_manager = self.brain.fractal_model_manager
            logger.info(f"Используется существующий fractal_model_manager из CoreBrain: {type(self.model_manager).__name__}")
            return True
        else:
            logger.warning("ModelManager not available - using UnifiedGenerator pipeline")
            self.model_manager = None
            return False
            
            if hasattr(self.brain, 'register_component'):
                self.brain.register_component('model_manager', self.model_manager)
                logger.info("ModelManager зарегистрирован в CoreBrain")
        
        if not hasattr(self, 'text_processor') or self.text_processor is None:
            logger.warning("Текстовый процессор не инициализирован")
            return False
        
        if not hasattr(self.text_processor, 'tokenizer') or self.text_processor.tokenizer is None:
            logger.warning("Токенизатор не инициализирован в текстовом процессоре")
        else:
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
        
        if self.brain and hasattr(self.brain, 'add_deferred_command'):
            logger.info("Регистрация отложенной команды для связывания компонентов MLUnit...")
            self.brain.add_deferred_command(self._link_components)
        else:
            logger.info("DeferredCommandSystem недоступна, выполняем прямое связывание компонентов")
            self._link_components()
            return True
        
    except Exception as e:
        logger.error(f"Ошибка при инициализации ModelManager: {e}", exc_info=True)
        return False


def _init_response_generator(self):
    """Инициализирует генератор ответов, получая его из brain или создавая новый."""
    try:
        if self.brain and hasattr(self.brain, 'response_generator') and self.brain.response_generator:
            self.response_generator = self.brain.response_generator
            logger.info("ResponseGenerator успешно получен из brain.")
            return True
        
        logger.info("Создание нового экземпляра ResponseGenerator...")
        try:
            from eva_ai.core.response_generator import ResponseGenerator
            
            model_manager = getattr(self, 'model_manager', None)
            text_processor = getattr(self, 'text_processor', None)
            tokenizer = getattr(text_processor, 'tokenizer', None) if text_processor else None
            
            self.response_generator = ResponseGenerator(
                brain=self.brain,
                model_manager=model_manager,
                text_processor=text_processor
            )
            
            if tokenizer and hasattr(self.response_generator, 'tokenizer'):
                self.response_generator.tokenizer = tokenizer
            
            if self.brain:
                self.brain.response_generator = self.response_generator
                
            logger.info("ResponseGenerator создан и зарегистрирован в brain.")
            return True
            
        except ImportError as import_error:
            logger.error(f"Не удалось импортировать ResponseGenerator: {import_error}", exc_info=True)
            if not hasattr(self, 'response_generator') or self.response_generator is None:
                self.response_generator = None
            return False
        except Exception as create_error:
            logger.error(f"Не удалось создать ResponseGenerator: {create_error}", exc_info=True)
            if not hasattr(self, 'response_generator') or self.response_generator is None:
                self.response_generator = None
            return False
            
    except Exception as e:
        logger.error(f"Ошибка при инициализации ResponseGenerator: {e}", exc_info=True)
        if not hasattr(self, 'response_generator') or self.response_generator is None:
            self.response_generator = None
        return False


def _init_hybrid_cache(self):
    """Инициализирует гибридный кэш."""
    try:
        from eva_ai.memory.hybrid_token_cache import HybridTokenCache
        
        self.hybrid_cache = HybridTokenCache(
            brain=self.brain,
            max_memory_tokens=self.hybrid_cache_size // 4096,
            disk_cache_dir=os.path.join(self.cache_dir, "hybrid_cache"),
            target_memory_gb=1.0,
            dynamic_memory_limit=True,
            max_ram_usage_percent=10.0
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
            if self.model_manager and not self.response_generator.model_manager:
                self.response_generator.model_manager = self.model_manager
            if self.text_processor and not self.response_generator.text_processor:
                self.response_generator.text_processor = self.text_processor
            if self.ml_core:
                self.response_generator.ml_core = self.ml_core
        
        logger.info("Компоненты MLUnit успешно связаны.")
        
        if not self._is_training_mode():
            self._verify_basic_functionality()
        
    except Exception as e:
        logger.error(f"Ошибка при отложенной компоновке компонентов: {e}", exc_info=True)


def _verify_basic_functionality(self):
    """Проверяет базовую функциональность компонентов."""
    if self._is_training_mode():
        logger.debug("Пропуск _verify_basic_functionality в режиме тренировки")
        return
    try:
        if self.model_manager:
            try:
                if hasattr(self.model_manager, 'get_available_models') and callable(getattr(self.model_manager, 'get_available_models')):
                    models = self.model_manager.get_available_models()
                else:
                    md = getattr(self.model_manager, 'model_metadata', {}) or {}
                    models = list(md.values()) if isinstance(md, dict) else (md or [])
            except Exception:
                models = []
            logger.info(f"Доступно моделей: {len(models)}")
        
        if self.text_processor:
            test_text = "Это тестовый текст для проверки токенизации."
            try:
                if torch is not None:
                    encoded = self.text_processor.encode(test_text)
                    input_ids = encoded.get('input_ids')
                    if input_ids is None:
                        tokens_count = 0
                    elif isinstance(input_ids, list):
                        tokens_count = len(input_ids) if input_ids else 0
                    elif hasattr(input_ids, 'numel'):
                        tokens_count = input_ids.numel()
                    else:
                        tokens_count = 0
                else:
                    tokens_count = 0
                logger.info(f"Тестовая токенизация успешна. Токенов: {tokens_count}")
            except Exception as e:
                logger.error(f"Ошибка при проверке токенизатора: {e}")
        
        if self.response_generator:
            test_prompt = "Кратко опиши, что такое искусственный интеллект."
            response = self.response_generator.generate_response(
                prompt=test_prompt,
                max_length=256,
                max_new_tokens=5,
                temperature=0.7,
                top_p=0.9,
                task="text-generation"
            )
            logger.info(f"Тестовая генерация ответа успешна. Длина: {len(response.get('text', ''))}")
    
    except Exception as e:
        logger.error(f"Ошибка проверки базовой функциональности: {e}", exc_info=True)


def _init_training_orchestrator(self):
    """Обучение теперь через SelfDialogLearning, TrainingOrchestrator не используется."""
    logger.info("TrainingOrchestrator отключен - обучение через SelfDialogLearning")
    return True


def _is_training_mode(self):
    """Проверяет, находится ли система в режиме тренировки."""
    return hasattr(self, 'training_mode') and self.training_mode


def get_system_health(self):
    """Возвращает информацию о здоровье системы."""
    try:
        components_status = {
            "ml_core": self.ml_core is not None,
            "model_manager": self.model_manager is not None,
            "text_processor": self.text_processor is not None,
            "response_generator": self.response_generator is not None,
            "models_ready": self.models_ready
        }
        
        health_score = 0.0
        if components_status["ml_core"]:
            health_score += 0.3
        if components_status["model_manager"]:
            health_score += 0.3
        if components_status["text_processor"]:
            health_score += 0.2
        if components_status["response_generator"]:
            health_score += 0.2
        
        if health_score >= 0.8:
            status = "healthy"
        elif health_score >= 0.5:
            status = "warning"
        else:
            status = "critical"
        
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
    """
    start_time = time.time()
    self.stats["total_requests"] += 1
    
    try:
        if not self.response_generator:
            logger.error("ResponseGenerator недоступен")
            return self._create_fallback_response(query, "response_generator_unavailable")
        
        response = self.response_generator.generate_response(query, **(context or {}))
        
        self._update_statistics(start_time, True)
        
        return response
        
    except Exception as e:
        self._update_statistics(start_time, False)
        logger.error(f"Ошибка генерации ответа: {e}")
        return self._create_fallback_response(query, str(e))


def _create_fallback_response(self, prompt: str, error: str) -> Dict[str, Any]:
    """Создает fallback-ответ при критической ошибке."""
    logger.error(f"КРИТИЧЕСКАЯ ОШИБКА в MLUnit: {error}")
    logger.error(f"Запрос не может быть обработан: '{prompt[:50]}...'")
    logger.error("Используется fallback-режим ответа")
    
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
    
    try:
        if hasattr(self, 'brain') and self.brain:
            if hasattr(self.brain, 'get_simple_response'):
                simple_response = self.brain.get_simple_response(prompt)
                if simple_response:
                    fallback_response["response"] = simple_response
                    fallback_response["confidence"] = 0.3
                    fallback_response["source"] = "brain_simple_response"
                    logger.info("Использован простой ответ из CoreBrain")
    except Exception as e:
        logger.debug(f"Не удалось получить простой ответ: {e}")
    
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
    """
    try:
        logger.info("Начинаем инициализацию компонентов MLUnit...")
        
        if not self._init_ml_core():
            logger.warning("Не удалось инициализировать MLCore, но продолжаем")
        
        if not self._init_text_processor():
            logger.warning("Не удалось инициализировать текстовый процессор, но продолжаем")
        
        if not self._init_model_manager():
            logger.warning("Не удалось инициализировать менеджер моделей, но продолжаем")
        
        if not self._init_response_generator():
            logger.warning("Не удалось инициализировать генератор ответов, но продолжаем")
        
        if not self._init_hybrid_cache():
            logger.warning("Не удалось инициализировать гибридный кэш, но продолжаем")
        
        if not self._init_training_orchestrator():
            logger.warning("Не удалось инициализировать оркестратор обучения, но продолжаем")
        
        if not (self.brain and hasattr(self.brain, 'add_deferred_command')):
            self._link_components()
        
        health = self.get_system_health()
        logger.info(f"Состояние ML системы: {health['status']}, score: {health['score']:.2f}")
        
        if self.model_manager is not None or self.ml_core is not None:
            logger.info("MLUnit успешно инициализирован (основные компоненты доступны)")
            return True
        
        if self.text_processor is not None:
            logger.info("MLUnit инициализирован с ограниченной функциональностью")
            return True
        
        logger.warning("MLUnit инициализирован без компонентов - режим заглушки")
        return False
    
    except Exception as e:
        logger.error(f"Ошибка дополнительной инициализации MLUnit: {e}", exc_info=True)
        return False


def start(self):
    """Запускает фоновые процессы MLUnit."""
    if self.running:
        return
        
    self.running = True
    
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
    
    if self.text_processor and hasattr(self.text_processor, 'stop'):
        self.text_processor.stop()
    
    if self.response_generator and hasattr(self.response_generator, 'stop'):
        self.response_generator.stop()
    
    if self.model_manager and hasattr(self.model_manager, 'stop'):
        self.model_manager.stop()
    
    if self.hybrid_cache and hasattr(self.hybrid_cache, 'stop'):
        self.hybrid_cache.stop()
    
    logger.info("MLUnit остановлен")


def generate_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
    """Генерирует ответ на запрос с управлением памятью."""
    start_time = time.time()
    self.stats["total_requests"] += 1
    
    try:
        self._maybe_cleanup_memory()
        
        if not self.response_generator:
            logger.error("ResponseGenerator недоступен")
            return self._create_fallback_response(prompt, "response_generator_unavailable")
        
        try:
            response = self.response_generator.generate_response(prompt, **kwargs)
        except Exception as gen_error:
            logger.error(f"Response generator error: {gen_error}")
            return self._create_fallback_response(prompt, f"generation_error: {str(gen_error)}")
        
        self._update_statistics(start_time, True)
        
        return response
        
    except Exception as e:
        self._update_statistics(start_time, False)
        logger.error(f"Ошибка генерации ответа: {e}", exc_info=True)
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
    """
    start_time = time.time()
    
    try:
        self._maybe_cleanup_memory()
        
        result = {
            "original_text": text,
            "processed_text": text,
            "tokens": [],
            "embeddings": None,
            "analysis": {},
            "status": "processed",
            "processing_time": 0.0
        }
        
        if hasattr(self, 'text_processor') and self.text_processor:
            try:
                processed = self.text_processor.process_text(text)
                result["processed_text"] = processed.get("text", text)
                result["tokens"] = processed.get("tokens", [])
            except Exception as e:
                logger.warning(f"Ошибка обработки текста: {e}")
        
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
