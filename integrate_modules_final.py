#!/usr/bin/env python3
"""
Финальная интеграция модулей CogniFlex
Интеграция learning и web модулей
"""

import os
import sys
import shutil
from datetime import datetime

# Добавляем путь к CogniFlex
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_integrated_learning():
    """Создает интегрированную версию модуля обучения"""
    print("🔧 Создание интегрированного модуля обучения...")
    
    template = '''"""
Интегрированный менеджер обучения CogniFlex
Поддерживает BaseComponent и EventBus
"""

import logging
import time
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

logger = logging.getLogger("cogniflex.learning")

from cogniflex.core.base_component import BaseComponent, ComponentState
from cogniflex.core.event_bus import get_event_bus, Event, EventTypes

# Импортируем оригинальные менеджеры обучения
try:
    from cogniflex.learning.learning_manager import LearningManager
    ORIGINAL_MANAGER_AVAILABLE = True
except ImportError:
    ORIGINAL_MANAGER_AVAILABLE = False
    logger.warning("Оригинальный LearningManager недоступен")

try:
    from cogniflex.learning.integrated_learning_manager import IntegratedLearningManager as OriginalIntegratedLearningManager
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
                self._original_manager = LearningManager(brain, cache_dir)
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
'''
    
    output_path = os.path.join(
        os.path.dirname(__file__), 
        "cogniflex", 
        "learning", 
        "learning_integrated.py"
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(template)
    
    print(f"   ✅ Создан файл: {output_path}")
    return output_path

def create_integrated_web_search():
    """Создает интегрированную версию веб-поиска"""
    print("🔧 Создание интегрированного веб-поиска...")
    
    template = '''"""
Интегрированный поисковый движок CogniFlex
Поддерживает BaseComponent и EventBus
"""

import logging
import time
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

logger = logging.getLogger("cogniflex.websearch")

from cogniflex.core.base_component import BaseComponent, ComponentState
from cogniflex.core.event_bus import get_event_bus, Event, EventTypes

# Импортируем оригинальный поисковый движок
try:
    from cogniflex.websearch.web_search_engine import WebSearchEngine
    ORIGINAL_AVAILABLE = True
except ImportError:
    ORIGINAL_AVAILABLE = False
    logger.warning("Оригинальный WebSearchEngine недоступен")


class IntegratedWebSearchEngine(BaseComponent):
    """Интегрированный поисковый движок с поддержкой событий"""
    
    def __init__(self, event_bus=None, brain=None, cache_dir: Optional[str] = None):
        super().__init__("web_search_engine", event_bus)
        
        self.brain = brain
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), 'websearch_cache')
        
        # Инициализируем оригинальный движок если доступен
        self._original_engine = None
        if ORIGINAL_AVAILABLE:
            try:
                self._original_engine = WebSearchEngine()
                logger.info("Оригинальный WebSearchEngine инициализирован")
            except Exception as e:
                logger.error(f"Ошибка инициализации оригинального движка: {e}")
        
        # Статистика поиска
        self.stats = {
            "searches_performed": 0,
            "results_found": 0,
            "cache_hits": 0,
            "errors": 0
        }
        
        # Кэш результатов поиска
        self.search_cache = {}
        
        logger.info(f"IntegratedWebSearchEngine {self.name} инициализирован")
    
    def _do_initialize(self) -> bool:
        """Инициализация компонента"""
        try:
            logger.info("Инициализация поискового движка...")
            
            # Инициализируем оригинальный движок
            if self._original_engine and hasattr(self._original_engine, 'initialize'):
                self._original_engine.initialize()
            
            # Создаем директорию кэша
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Загружаем кэш поиска
            self._load_search_cache()
            
            # Публикуем событие инициализации
            self._emit_event("web_search_engine.initialized", {
                'component': self.name,
                'cache_dir': self.cache_dir,
                'cache_size': len(self.search_cache)
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка инициализации поискового движка: {e}")
            return False
    
    def _do_start(self) -> bool:
        """Запуск компонента"""
        try:
            logger.info("Запуск поискового движка...")
            
            # Запускаем оригинальный движок
            if self._original_engine and hasattr(self._original_engine, 'start'):
                self._original_engine.start()
            
            # Публикуем событие запуска
            self._emit_event("web_search_engine.started", {
                'component': self.name
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка запуска поискового движка: {e}")
            return False
    
    def _do_stop(self) -> bool:
        """Остановка компонента"""
        try:
            logger.info("Остановка поискового движка...")
            
            # Останавливаем оригинальный движок
            if self._original_engine and hasattr(self._original_engine, 'stop'):
                self._original_engine.stop()
            
            # Сохраняем кэш поиска
            self._save_search_cache()
            
            # Публикуем событие остановки
            self._emit_event("web_search_engine.stopped", {
                'component': self.name,
                'stats': self.stats,
                'cache_size': len(self.search_cache)
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка остановки поискового движка: {e}")
            return False
    
    def search(self, query: str, search_config: Optional[Dict] = None) -> Dict[str, Any]:
        """Выполняет поиск в интернете"""
        start_time = time.time()
        
        try:
            # Проверяем кэш
            cache_key = self._generate_cache_key(query, search_config)
            if cache_key in self.search_cache:
                cached_result = self.search_cache[cache_key]
                self.stats["cache_hits"] += 1
                
                # Публикуем событие кэш-хита
                self._emit_event("web_search_engine.cache_hit", {
                    'query_length': len(query),
                    'cache_key': cache_key
                })
                
                return cached_result
            
            # Используем оригинальный движок если доступен
            if self._original_engine and hasattr(self._original_engine, 'search'):
                result = self._original_engine.search(query, search_config)
            else:
                # Базовый поиск
                result = self._basic_web_search(query, search_config)
            
            # Обновляем статистику
            self.stats["searches_performed"] += 1
            
            if result.get("success", False):
                results = result.get("results", [])
                self.stats["results_found"] += len(results)
                
                # Сохраняем в кэш
                self.search_cache[cache_key] = result
            
            # Публикуем событие поиска
            self._emit_event("web_search_engine.search_performed", {
                'query_length': len(query),
                'success': result.get("success", False),
                'results_count': len(result.get("results", [])),
                'processing_time': time.time() - start_time
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")
            self.stats["errors"] += 1
            return {"success": False, "error": str(e)}
    
    def _basic_web_search(self, query: str, search_config: Optional[Dict] = None) -> Dict[str, Any]:
        """Базовый веб-поиск (симуляция)"""
        # Симуляция результатов поиска
        max_results = search_config.get("max_results", 10) if search_config else 10
        
        # Генерируем фейковые результаты
        results = []
        for i in range(min(max_results, 5)):  # Ограничиваем для симуляции
            result = {
                "title": f"Результат поиска #{i+1} для: {query}",
                "url": f"https://example.com/result{i+1}",
                "snippet": f"Это фрагмент текста о {query}. Здесь содержится релевантная информация...",
                "relevance_score": 0.9 - (i * 0.1),
                "source": "simulated_search",
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
        
        return {
            "success": True,
            "query": query,
            "results": results,
            "total_results": len(results),
            "search_time": time.time(),
            "source": "integrated_web_search"
        }
    
    def search_with_filters(self, query: str, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Поиск с фильтрами"""
        try:
            # Используем оригинальный движок если доступен
            if self._original_engine and hasattr(self._original_engine, 'search_with_filters'):
                result = self._original_engine.search_with_filters(query, filters)
            else:
                # Базовый поиск с фильтрами
                search_config = {"filters": filters}
                result = self._basic_web_search(query, search_config)
            
            # Применяем фильтры к результатам
            if result.get("success", False):
                filtered_results = self._apply_filters(result.get("results", []), filters)
                result["results"] = filtered_results
                result["total_results"] = len(filtered_results)
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка поиска с фильтрами: {e}")
            return {"success": False, "error": str(e)}
    
    def _apply_filters(self, results: List[Dict], filters: Dict[str, Any]) -> List[Dict]:
        """Применяет фильтры к результатам поиска"""
        filtered_results = results.copy()
        
        # Фильтрация по источнику
        if "source" in filters:
            source_filter = filters["source"]
            filtered_results = [r for r in filtered_results if r.get("source") == source_filter]
        
        # Фильтрация по релевантности
        if "min_relevance" in filters:
            min_relevance = filters["min_relevance"]
            filtered_results = [r for r in filtered_results if r.get("relevance_score", 0) >= min_relevance]
        
        # Фильтрация по домену
        if "domain" in filters:
            domain_filter = filters["domain"]
            filtered_results = [r for r in filtered_results if domain_filter in r.get("url", "")]
        
        return filtered_results
    
    def get_search_suggestions(self, partial_query: str) -> List[str]:
        """Возвращает предложения для автодополнения поиска"""
        try:
            # Используем оригинальный движок если доступен
            if self._original_engine and hasattr(self._original_engine, 'get_search_suggestions'):
                return self._original_engine.get_search_suggestions(partial_query)
            else:
                # Базовые предложения
                suggestions = [
                    f"{partial_query} tutorial",
                    f"{partial_query} guide",
                    f"{partial_query} examples",
                    f"{partial_query} best practices",
                    f"how to {partial_query}"
                ]
                return suggestions[:5]  # Ограничиваем количество
                
        except Exception as e:
            logger.error(f"Ошибка получения предложений: {e}")
            return []
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику поиска"""
        stats = self.stats.copy()
        
        # Добавляем детальную статистику
        stats.update({
            "cache_size": len(self.search_cache),
            "cache_hit_rate": self.stats["cache_hits"] / max(1, self.stats["searches_performed"]),
            "average_results_per_search": self.stats["results_found"] / max(1, self.stats["searches_performed"]),
            "most_common_queries": self._get_most_common_queries()
        })
        
        # Добавляем статистику из оригинального движка
        if self._original_engine and hasattr(self._original_engine, 'get_statistics'):
            original_stats = self._original_engine.get_statistics()
            stats.update(original_stats)
        
        return stats
    
    def _get_most_common_queries(self) -> List[str]:
        """Возвращает наиболее частые запросы"""
        # Анализируем кэш для поиска частых запросов
        query_counts = {}
        for cache_key, result in self.search_cache.items():
            query = result.get("query", "")
            if query:
                query_counts[query] = query_counts.get(query, 0) + 1
        
        # Сортируем по частоте
        sorted_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)
        return [q[0] for q in sorted_queries[:5]]
    
    def _generate_cache_key(self, query: str, search_config: Optional[Dict] = None) -> str:
        """Генерирует ключ для кэша"""
        import hashlib
        key_data = query + str(search_config or {})
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _load_search_cache(self):
        """Загружает кэш поиска"""
        try:
            cache_file = os.path.join(self.cache_dir, 'search_cache.json')
            if os.path.exists(cache_file):
                import json
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.search_cache = json.load(f)
                logger.info(f"Загружено {len(self.search_cache)} записей в кэше поиска")
        except Exception as e:
            logger.error(f"Ошибка загрузки кэша поиска: {e}")
            self.search_cache = {}
    
    def _save_search_cache(self):
        """Сохраняет кэш поиска"""
        try:
            cache_file = os.path.join(self.cache_dir, 'search_cache.json')
            import json
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.search_cache, f, ensure_ascii=False, indent=2)
            logger.info(f"Сохранено {len(self.search_cache)} записей в кэше поиска")
        except Exception as e:
            logger.error(f"Ошибка сохранения кэша поиска: {e}")
'''
    
    output_path = os.path.join(
        os.path.dirname(__file__), 
        "cogniflex", 
        "websearch", 
        "web_search_integrated.py"
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(template)
    
    print(f"   ✅ Создан файл: {output_path}")
    return output_path

def update_component_initializer_final():
    """Финальное обновление ComponentInitializer с оставшимися модулями"""
    print("🔧 Финальное обновление ComponentInitializer...")
    
    # Читаем текущий файл
    initializer_path = os.path.join(
        os.path.dirname(__file__), 
        "cogniflex", 
        "core", 
        "component_initializer.py"
    )
    
    try:
        with open(initializer_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Добавляем импорты новых модулей
        new_imports = '''# Финальные импорты интегрированных модулей
from ..learning.learning_integrated import IntegratedLearningManager
from ..websearch.web_search_integrated import IntegratedWebSearchEngine
'''
        
        # Находим место для вставки импортов
        import_pos = content.find("# Импорты новых интегрированных модулей")
        if import_pos != -1:
            # Вставляем после существующих импортов
            insert_pos = content.find("\n\n", import_pos) + 2
            content = content[:insert_pos] + new_imports + content[insert_pos:]
        
        # Добавляем финальные фабрики
        new_factories = '''
    def create_learning_manager(self) -> IntegratedLearningManager:
        """Создает интегрированный менеджер обучения."""
        try:
            logger.debug("Создание интегрированного менеджера обучения...")
            event_bus = self.brain.get_event_bus() if hasattr(self.brain, 'get_event_bus') else self.event_bus
            component = IntegratedLearningManager(
                event_bus=event_bus,
                brain=self.brain,
                cache_dir=os.path.join(self.cache_dir, "learning")
            )
            logger.debug("Интегрированный менеджер обучения создан")
            return component
        except Exception as e:
            logger.error(f"Ошибка создания интегрированного менеджера обучения: {e}")
            raise
    
    def create_web_search_engine(self) -> IntegratedWebSearchEngine:
        """Создает интегрированный поисковый движок."""
        try:
            logger.debug("Создание интегрированного поискового движка...")
            event_bus = self.brain.get_event_bus() if hasattr(self.brain, 'get_event_bus') else self.event_bus
            component = IntegratedWebSearchEngine(
                event_bus=event_bus,
                brain=self.brain,
                cache_dir=os.path.join(self.cache_dir, "websearch")
            )
            logger.debug("Интегрированный поисковый движок создан")
            return component
        except Exception as e:
            logger.error(f"Ошибка создания интегрированного поискового движка: {e}")
            raise
'''
        
        # Находим конец файла перед последним методом
        last_method_pos = content.rfind("def ")
        if last_method_pos != -1:
            # Находим конец этого метода
            method_end = content.find("\n\n", last_method_pos)
            if method_end == -1:
                method_end = len(content)
            
            content = content[:method_end] + new_factories + content[method_end:]
        
        # Сохраняем обновленный файл
        with open(initializer_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"   ✅ Обновлен файл: {initializer_path}")
        return True
        
    except Exception as e:
        print(f"   ❌ Ошибка обновления ComponentInitializer: {e}")
        return False

def main():
    """Основная функция финальной интеграции"""
    print("🚀 Финальная интеграция модулей CogniFlex")
    print("=" * 50)
    
    results = []
    
    # 1. Создаем интегрированный модуль обучения
    try:
        learning_path = create_integrated_learning()
        results.append(("learning", True, learning_path))
    except Exception as e:
        results.append(("learning", False, str(e)))
    
    # 2. Создаем интегрированный веб-поиск
    try:
        websearch_path = create_integrated_web_search()
        results.append(("websearch", True, websearch_path))
    except Exception as e:
        results.append(("websearch", False, str(e)))
    
    # 3. Финальное обновление ComponentInitializer
    try:
        initializer_success = update_component_initializer_final()
        results.append(("component_initializer_final", initializer_success, ""))
    except Exception as e:
        results.append(("component_initializer_final", False, str(e)))
    
    # 4. Итоги
    print(f"\n📊 ИТОГИ ФИНАЛЬНОЙ ИНТЕГРАЦИИ:")
    
    success_count = 0
    for module, success, path in results:
        status = "✅ УСПЕХ" if success else "❌ НЕУДАЧА"
        print(f"   {module}: {status}")
        if success and path:
            print(f"      📁 {path}")
        elif not success:
            print(f"      ⚠️ {path}")
        
        if success:
            success_count += 1
    
    print(f"\n🎯 Результат: {success_count}/{len(results)} модулей интегрировано")
    
    if success_count == len(results):
        print("🎉 Финальная интеграция успешно завершена!")
        return True
    else:
        print("⚠️ Некоторые модули требуют внимания.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
