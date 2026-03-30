"""
Интегрированный менеджер обучения ЕВА
Поддерживает BaseComponent и EventBus
"""

import logging
import time
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

logger = logging.getLogger("eva.learning")

from eva.core.base_component import BaseComponent, ComponentState
from eva.core.event_bus import get_event_bus, Event, EventTypes

# Импортируем оригинальные менеджеры обучения
try:
    from eva.learning.learning_manager import LearningManager
    ORIGINAL_MANAGER_AVAILABLE = True
except ImportError:
    ORIGINAL_MANAGER_AVAILABLE = False
    logger.warning("Оригинальный LearningManager недоступен")

try:
    from eva.learning.integrated_learning_manager import IntegratedLearningManager as OriginalIntegratedLearningManager
    ORIGINAL_INTEGRATED_AVAILABLE = True
except ImportError:
    ORIGINAL_INTEGRATED_AVAILABLE = False
    logger.warning("Оригинальный IntegratedLearningManager недоступен")


class IntegratedLearningManager(BaseComponent):
    """Интегрированный менеджер обучения с поддержкой событий"""
    
    def __init__(self, event_bus=None, brain=None, cache_dir: Optional[str] = None):
        super().__init__("learning_manager", event_bus)
        
        self.brain = brain
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), 'learning_cache')
        
        # Инициализируем оригинальные компоненты если доступны
        self._original_manager = None
        self._original_integrated = None
        
        if ORIGINAL_MANAGER_AVAILABLE:
            try:
                self._original_manager = LearningManager(brain=brain, config={"cache_dir": cache_dir})
                logger.info("Оригинальный LearningManager инициализирован")
            except Exception as e:
                logger.error(f"Ошибка инициализации оригинального менеджера: {e}")
        
        if ORIGINAL_INTEGRATED_AVAILABLE:
            try:
                self._original_integrated = OriginalIntegratedLearningManager(brain, cache_dir)
                logger.info("Оригинальный IntegratedLearningManager инициализирован")
            except Exception as e:
                logger.error(f"Ошибка инициализации оригинального интегрированного менеджера: {e}")
        
        # Статистика обучения
        self.stats = {
            "learning_sessions": 0,
            "models_trained": 0,
            "knowledge_acquired": 0,
            "adaptations_performed": 0,
            "errors": 0
        }
        
        # База знаний обучения
        self.learning_database = []
        
        logger.info(f"IntegratedLearningManager {self.name} инициализирован")
    
    def _do_initialize(self) -> bool:
        """Инициализация компонента"""
        try:
            logger.info("Инициализация менеджера обучения...")
            
            # Инициализируем оригинальные компоненты
            if self._original_manager and hasattr(self._original_manager, 'initialize'):
                self._original_manager.initialize()
            
            if self._original_integrated and hasattr(self._original_integrated, 'initialize'):
                self._original_integrated.initialize()
            
            # Создаем директорию кэша
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Загружаем базу знаний обучения
            self._load_learning_database()
            
            # Публикуем событие инициализации
            self._emit_event("learning_manager.initialized", {
                'component': self.name,
                'cache_dir': self.cache_dir,
                'knowledge_count': len(self.learning_database)
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка инициализации менеджера обучения: {e}")
            return False
    
    def _do_start(self) -> bool:
        """Запуск компонента"""
        try:
            logger.info("Запуск менеджера обучения...")
            
            # Запускаем оригинальные компоненты
            if self._original_manager and hasattr(self._original_manager, 'start'):
                self._original_manager.start()
            
            if self._original_integrated and hasattr(self._original_integrated, 'start'):
                self._original_integrated.start()
            
            # Публикуем событие запуска
            self._emit_event("learning_manager.started", {
                'component': self.name
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка запуска менеджера обучения: {e}")
            return False
    
    def _do_stop(self) -> bool:
        """Остановка компонента"""
        try:
            logger.info("Остановка менеджера обучения...")
            
            # Останавливаем оригинальные компоненты
            if self._original_manager and hasattr(self._original_manager, 'stop'):
                self._original_manager.stop()
            
            if self._original_integrated and hasattr(self._original_integrated, 'stop'):
                self._original_integrated.stop()
            
            # Сохраняем базу знаний обучения
            self._save_learning_database()
            
            # Публикуем событие остановки
            self._emit_event("learning_manager.stopped", {
                'component': self.name,
                'stats': self.stats,
                'knowledge_count': len(self.learning_database)
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка остановки менеджера обучения: {e}")
            return False
    
    def start_learning_session(self, session_type: str, data: Dict, parameters: Optional[Dict] = None) -> Dict[str, Any]:
        """Начинает сессию обучения"""
        start_time = time.time()
        
        try:
            # Используем оригинальный менеджер если доступен
            if self._original_manager and hasattr(self._original_manager, 'start_learning_session'):
                result = self._original_manager.start_learning_session(session_type, data, parameters)
            elif self._original_integrated and hasattr(self._original_integrated, 'start_learning_session'):
                result = self._original_integrated.start_learning_session(session_type, data, parameters)
            else:
                # Базовая сессия обучения
                result = self._basic_learning_session(session_type, data, parameters)
            
            # Обновляем статистику
            self.stats["learning_sessions"] += 1
            
            if result.get("success", False):
                # Сохраняем в базу знаний
                knowledge_entry = {
                    "id": len(self.learning_database) + 1,
                    "session_type": session_type,
                    "data": data,
                    "parameters": parameters or {},
                    "result": result,
                    "learning_time": datetime.now().isoformat(),
                    "processing_time": time.time() - start_time
                }
                self.learning_database.append(knowledge_entry)
                
                self.stats["knowledge_acquired"] += 1
            
            # Публикуем событие обучения
            self._emit_event("learning_manager.session_started", {
                'session_type': session_type,
                'success': result.get("success", False),
                'processing_time': time.time() - start_time
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка начала сессии обучения: {e}")
            self.stats["errors"] += 1
            return {"success": False, "error": str(e)}
    
    def _basic_learning_session(self, session_type: str, data: Dict, parameters: Optional[Dict] = None) -> Dict[str, Any]:
        """Базовая сессия обучения"""
        # Простая логика обучения в зависимости от типа
        if session_type == "pattern_recognition":
            # Обучение распознаванию паттернов
            patterns = data.get("patterns", [])
            learned_patterns = []
            
            for pattern in patterns:
                # Простая обработка паттерна
                processed_pattern = {
                    "original": pattern,
                    "processed": f"processed_{pattern}",
                    "confidence": 0.8
                }
                learned_patterns.append(processed_pattern)
            
            return {
                "success": True,
                "session_type": session_type,
                "learned_patterns": learned_patterns,
                "total_patterns": len(learned_patterns)
            }
        
        elif session_type == "knowledge_acquisition":
            # Приобретение знаний
            facts = data.get("facts", [])
            acquired_knowledge = []
            
            for fact in facts:
                knowledge_item = {
                    "fact": fact,
                    "confidence": 0.9,
                    "source": "learning_session",
                    "timestamp": datetime.now().isoformat()
                }
                acquired_knowledge.append(knowledge_item)
            
            return {
                "success": True,
                "session_type": session_type,
                "acquired_knowledge": acquired_knowledge,
                "total_facts": len(acquired_knowledge)
            }
        
        elif session_type == "adaptation":
            # Адаптация
            adaptation_data = data.get("adaptation_data", {})
            adaptations = []
            
            for key, value in adaptation_data.items():
                adaptation = {
                    "parameter": key,
                    "old_value": value,
                    "new_value": f"adapted_{value}",
                    "improvement": 0.1
                }
                adaptations.append(adaptation)
            
            return {
                "success": True,
                "session_type": session_type,
                "adaptations": adaptations,
                "total_adaptations": len(adaptations)
            }
        
        else:
            return {
                "success": False,
                "error": f"Неизвестный тип сессии обучения: {session_type}"
            }
    
    def train_model(self, model_name: str, training_data: List, training_config: Optional[Dict] = None) -> Dict[str, Any]:
        """Обучает модель"""
        start_time = time.time()
        
        try:
            # Используем оригинальный менеджер если доступен
            if self._original_manager and hasattr(self._original_manager, 'train_model'):
                result = self._original_manager.train_model(model_name, training_data, training_config)
            elif self._original_integrated and hasattr(self._original_integrated, 'train_model'):
                result = self._original_integrated.train_model(model_name, training_data, training_config)
            else:
                # Базовое обучение модели
                result = self._basic_model_training(model_name, training_data, training_config)
            
            if result.get("success", False):
                self.stats["models_trained"] += 1
                
                # Публикуем событие обучения модели
                self._emit_event("learning_manager.model_trained", {
                    'model_name': model_name,
                    'training_samples': len(training_data),
                    'success': True,
                    'processing_time': time.time() - start_time
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка обучения модели: {e}")
            self.stats["errors"] += 1
            return {"success": False, "error": str(e)}
    
    def _basic_model_training(self, model_name: str, training_data: List, training_config: Optional[Dict] = None) -> Dict[str, Any]:
        """Базовое обучение модели"""
        # Простая симуляция обучения
        epochs = training_config.get("epochs", 10) if training_config else 10
        learning_rate = training_config.get("learning_rate", 0.001) if training_config else 0.001
        
        # Симуляция процесса обучения
        training_progress = []
        for epoch in range(epochs):
            # Симуляция улучшения метрики
            accuracy = 0.5 + (epoch / epochs) * 0.4  # от 0.5 до 0.9
            loss = 1.0 - (epoch / epochs) * 0.8  # от 1.0 до 0.2
            
            training_progress.append({
                "epoch": epoch + 1,
                "accuracy": accuracy,
                "loss": loss
            })
        
        return {
            "success": True,
            "model_name": model_name,
            "training_samples": len(training_data),
            "epochs": epochs,
            "learning_rate": learning_rate,
            "final_accuracy": training_progress[-1]["accuracy"],
            "final_loss": training_progress[-1]["loss"],
            "training_progress": training_progress
        }
    
    def adapt_behavior(self, adaptation_type: str, context: Dict, feedback: Optional[Dict] = None) -> Dict[str, Any]:
        """Адаптирует поведение на основе контекста и обратной связи"""
        try:
            # Используем оригинальный менеджер если доступен
            if self._original_manager and hasattr(self._original_manager, 'adapt_behavior'):
                result = self._original_manager.adapt_behavior(adaptation_type, context, feedback)
            elif self._original_integrated and hasattr(self._original_integrated, 'adapt_behavior'):
                result = self._original_integrated.adapt_behavior(adaptation_type, context, feedback)
            else:
                # Базовая адаптация поведения
                result = self._basic_behavior_adaptation(adaptation_type, context, feedback)
            
            if result.get("success", False):
                self.stats["adaptations_performed"] += 1
                
                # Публикуем событие адаптации
                self._emit_event("learning_manager.behavior_adapted", {
                    'adaptation_type': adaptation_type,
                    'success': True
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка адаптации поведения: {e}")
            self.stats["errors"] += 1
            return {"success": False, "error": str(e)}
    
    def _basic_behavior_adaptation(self, adaptation_type: str, context: Dict, feedback: Optional[Dict] = None) -> Dict[str, Any]:
        """Базовая адаптация поведения"""
        adaptations = []
        
        if adaptation_type == "response_style":
            # Адаптация стиля ответов
            user_preference = feedback.get("style_preference", "neutral") if feedback else "neutral"
            
            adaptations.append({
                "parameter": "response_style",
                "old_value": "default",
                "new_value": user_preference,
                "reason": "User preference detected"
            })
        
        elif adaptation_type == "complexity":
            # Адаптация сложности
            user_level = feedback.get("complexity_level", "medium") if feedback else "medium"
            
            adaptations.append({
                "parameter": "response_complexity",
                "old_value": "medium",
                "new_value": user_level,
                "reason": "User level assessment"
            })
        
        elif adaptation_type == "domain_focus":
            # Адаптация фокуса на домен
            domain = context.get("domain", "general")
            
            adaptations.append({
                "parameter": "domain_focus",
                "old_value": "general",
                "new_value": domain,
                "reason": "Domain context detected"
            })
        
        return {
            "success": True,
            "adaptation_type": adaptation_type,
            "adaptations": adaptations,
            "total_adaptations": len(adaptations)
        }
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику обучения"""
        stats = self.stats.copy()
        
        # Добавляем детальную статистику
        stats.update({
            "total_knowledge": len(self.learning_database),
            "recent_sessions": len([k for k in self.learning_database if k.get("result", {}).get("success", False)]),
            "session_types": list(set(k.get("session_type", "unknown") for k in self.learning_database)),
            "average_processing_time": sum(k.get("processing_time", 0) for k in self.learning_database) / max(1, len(self.learning_database))
        })
        
        # Добавляем статистику из оригинальных компонентов
        if self._original_manager and hasattr(self._original_manager, 'get_statistics'):
            original_stats = self._original_manager.get_statistics()
            stats.update(original_stats)
        
        return stats
    
    def _load_learning_database(self):
        """Загружает базу знаний обучения"""
        try:
            db_file = os.path.join(self.cache_dir, 'learning_database.json')
            if os.path.exists(db_file):
                import json
                with open(db_file, 'r', encoding='utf-8') as f:
                    self.learning_database = json.load(f)
                logger.info(f"Загружено {len(self.learning_database)} записей обучения")
        except Exception as e:
            logger.error(f"Ошибка загрузки базы знаний обучения: {e}")
            self.learning_database = []
    
    def _save_learning_database(self):
        """Сохраняет базу знаний обучения"""
        try:
            db_file = os.path.join(self.cache_dir, 'learning_database.json')
            import json
            with open(db_file, 'w', encoding='utf-8') as f:
                json.dump(self.learning_database, f, ensure_ascii=False, indent=2)
            logger.info(f"Сохранено {len(self.learning_database)} записей обучения")
        except Exception as e:
            logger.error(f"Ошибка сохранения базы знаний обучения: {e}")
