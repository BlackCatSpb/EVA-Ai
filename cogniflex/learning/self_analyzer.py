"""Главный модуль самоанализа для CogniFlex - объединяет все компоненты"""
import os
import logging
import time
from typing import Dict, Any, Optional, List, Callable

logger = logging.getLogger("cogniflex.self_analyzer")

# Импортируем компоненты
from .analyzer_core import AnalyzerCore
from cogniflex.system.health_monitor import HealthMonitor
from cogniflex.learning.learning_opportunity_manager import LearningOpportunityManager
from cogniflex.learning.performance_analyzer import PerformanceAnalyzer

class SelfAnalyzer:
    """Модуль самоанализа для CogniFlex - объединяет все компоненты."""
    
    def __init__(self, brain=None, cache_dir: Optional[str] = None):
        """
        Инициализирует модуль самоанализа.
        
        Args:
            brain: Ссылка на ядро CogniFlex (опционально)
            cache_dir: Путь к директории кэша (опционально)
        """
        self.brain = brain
        self.cache_dir = cache_dir
        
        self.is_initialized = False
        self.init_error = None
        
        try:
            # Проверяем состояние моделей
            if brain and hasattr(brain, 'model_manager') and hasattr(brain.model_manager, 'model_states'):
                # Проверяем, загружены ли все необходимые модели
                required_models = ['analyzer', 'ethics', 'knowledge']
                for model_id in required_models:
                    state = brain.model_manager.model_states.get(model_id)
                    if state == "loading":
                        logger.warning(f"Модель {model_id} ещё загружается, часть функциональности может быть недоступна")
                        continue
                    elif state == "error":
                        error = getattr(brain.model_manager, 'loading_errors', {}).get(model_id, "неизвестная ошибка")
                        logger.warning(f"Ошибка загрузки модели {model_id}: {error}")
                    elif not state:
                        logger.warning(f"Модель {model_id} не загружена, часть функциональности будет недоступна")
                        
            # Создаем компоненты
            self.analyzer_core = AnalyzerCore(brain, cache_dir)
            self.health_monitor = HealthMonitor(brain, self.analyzer_core)
            self.learning_opportunity_manager = LearningOpportunityManager(brain, self.analyzer_core)
            self.performance_analyzer = PerformanceAnalyzer(brain, self.analyzer_core)
            
            # Инициализируем тренер графа памяти
            from .memory_graph_trainer import MemoryGraphTrainer
            self.memory_graph_trainer = MemoryGraphTrainer(brain)
            
            # Устанавливаем ссылки между компонентами
            if brain:
                brain.analyzer_core = self.analyzer_core
                brain.health_monitor = self.health_monitor
                brain.learning_opportunity_manager = self.learning_opportunity_manager
                brain.performance_analyzer = self.performance_analyzer
                brain.memory_graph_trainer = self.memory_graph_trainer
            
            self.is_initialized = True
            logger.info("SelfAnalyzer полностью инициализирован с тренером графа памяти")
            
        except Exception as e:
            self.init_error = str(e)
            logger.warning(f"Самоанализ отложен: {e}")
            logger.error(f"Ошибка инициализации SelfAnalyzer: {e}")
            # Создаем заглушки для отсутствующих компонентов
            self.analyzer_core = None
            self.health_monitor = None
            self.learning_opportunity_manager = None
            self.performance_analyzer = None
            self.memory_graph_trainer = None
    
    def _check_ready(self) -> bool:
        """Проверяет готовность модуля к работе"""
        if not self.is_initialized:
            if self.init_error:
                logger.warning(f"Самоанализ недоступен: {self.init_error}")
            else:
                logger.warning("Самоанализ ещё не инициализирован")
            return False
        return True

    def start(self):
        """Запускает фоновые процессы модуля самоанализа."""
        if not self._check_ready():
            return False
            
        try:
            # Не запускать, пока модели не готовы
            if not self._models_ready():
                logger.warning("Самоанализ отложен: модели ещё не загружены")
                try:
                    if hasattr(self.brain, 'status_queue') and self.brain.status_queue:
                        self.brain.status_queue.put(("self_learning", 0, "Ожидание загрузки моделей"))
                except Exception:
                    pass
                return False
            if not getattr(self.analyzer_core, 'running', False):
                self.analyzer_core.start_background_analysis()
                logger.info("Фоновые процессы модуля самоанализа запущены")
                return True
            logger.warning("Попытка запуска уже активного процесса самоанализа")
            return False
        except Exception as e:
            logger.error(f"Ошибка запуска самоанализа: {e}")
            return False
    
    def stop(self):
        """Останавливает фоновые процессы модуля самоанализа."""
        try:
            if getattr(self.analyzer_core, 'running', False):
                self.analyzer_core.stop_background_analysis()
                logger.info("Фоновые процессы модуля самоанализа остановлены")
                return True
            logger.warning("Попытка остановки неактивного процесса самоанализа")
            return False
        except Exception as e:
            logger.error(f"Ошибка остановки самоанализа: {e}")
            return False
    
    def analyze_system(self) -> Dict[str, Any]:
        """Анализирует состояние всей системы."""
        if not self._check_ready():
            return {
                "error": "Самоанализ недоступен",
                "state": "error",
                "timestamp": time.time()
            }
            
        try:
            # Анализируем здоровье
            health_report = self.health_monitor.analyze_system_health()
            
            # Анализируем производительность
            performance_report = self.performance_analyzer.analyze_performance()
            
            # Анализируем эволюцию
            evolution_report = self.health_monitor.analyze_evolution()
            
            return {
                "health_report": health_report,
                "performance_report": performance_report,
                "evolution_report": evolution_report,
                "state": "ready",
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Ошибка анализа системы: {e}")
            return {
                "error": str(e),
                "timestamp": time.time()
            }
    
    def get_learning_opportunities(self, min_priority: float = 0.0, 
                                 executed: Optional[bool] = None,
                                 limit: int = 100,
                                 domain: Optional[str] = None,
                                 resolved: Optional[bool] = None) -> List:
        """
        Возвращает список возможностей для обучения с фильтрацией.
        
        Args:
            min_priority: Минимальный приоритет (0.0-1.0)
            executed: Фильтр по выполненным возможностям (True, False, None)
            limit: Максимальное количество результатов
            domain: Фильтр по домену
            resolved: Фильтр по разрешенным возможностям
            
        Returns:
            List: Список возможностей для обучения
        """
        if not self._check_ready():
            return [{
                "id": "system_not_ready",
                "concept": "system_state",
                "type": "error",
                "priority": 1.0,
                "domain": "system",
                "status": "pending",
                "message": f"Самоанализ недоступен: {self.init_error if self.init_error else 'не инициализирован'}",
                "timestamp": time.time()
            }]
            
        try:
            opportunities = self.analyzer_core.get_learning_opportunities(
                min_priority, executed, limit, domain, resolved)
            if not opportunities:
                # Если нет возможностей, добавляем системное состояние
                opportunities.append({
                    "id": "system_ready",
                    "concept": "system_state",
                    "type": "info",
                    "priority": 0.0,
                    "domain": "system",
                    "status": "ready",
                    "message": "Система готова к работе",
                    "timestamp": time.time()
                })
            return opportunities
        except Exception as e:
            logger.error(f"Ошибка получения возможностей обучения: {e}")
            return [{
                "id": "system_error",
                "concept": "system_state",
                "type": "error",
                "priority": 1.0,
                "domain": "system",
                "status": "error",
                "message": f"Ошибка: {str(e)}",
                "timestamp": time.time()
            }]

    def clear_learning_opportunities(self) -> Dict[str, Any]:
        """Очищает все возможности для обучения через AnalyzerCore.

        Возвращает отчёт: { ok: bool, deleted: int, error?: str }
        """
        if not self._check_ready():
            return {
                "ok": False, 
                "deleted": 0, 
                "error": f"Самоанализ недоступен: {self.init_error if self.init_error else 'не инициализирован'}"
            }
            
        try:
            if not self.analyzer_core or not hasattr(self.analyzer_core, 'clear_learning_opportunities'):
                return {"ok": False, "deleted": 0, "error": "analyzer_core недоступен"}
            return self.analyzer_core.clear_learning_opportunities()
        except Exception as e:
            logger.error(f"Ошибка очистки возможностей для обучения: {e}", exc_info=True)
            return {"ok": False, "deleted": 0, "error": str(e)}
    
    def add_learning_opportunity(self, concept: str, opportunity_type: str, 
                               priority: float, domain: str, 
                               evidence: List[str], suggested_actions: List[str],
                               callback: Optional[Callable] = None) -> bool:
        """
        Добавляет новую возможность для обучения.
        
        Args:
            concept: Концепт, связанный с возможностью
            opportunity_type: Тип возможности (expansion, refinement, updating, integration)
            priority: Приоритет (0.0-1.0)
            domain: Домен знаний
            evidence: Доказательства необходимости
            suggested_actions: Предлагаемые действия
            callback: Функция обратного вызова
            
        Returns:
            bool: Успешно ли добавлено
        """
        if not concept or not opportunity_type or not domain:
            logger.warning("Недостаточно данных для добавления возможности обучения")
            return False
            
        if not 0.0 <= priority <= 1.0:
            logger.warning(f"Неверный приоритет: {priority}. Должен быть между 0.0 и 1.0")
            return False
            
        try:
            return self.analyzer_core.add_learning_opportunity(
                concept, opportunity_type, priority, domain, evidence, suggested_actions, callback)
        except Exception as e:
            logger.error(f"Ошибка добавления возможности обучения: {e}")
            return False
    
    def execute_learning_opportunity(self, opportunity_id: str) -> bool:
        """
        Выполняет указанную возможность для обучения.
        
        Args:
            opportunity_id: ID возможности для обучения
            
        Returns:
            bool: Успешно ли выполнено
        """
        if not opportunity_id:
            return False
            
        try:
            return self.learning_opportunity_manager.execute_learning_opportunity(opportunity_id)
        except Exception as e:
            logger.error(f"Ошибка выполнения возможности обучения: {e}")
            return False
    
    def get_fixes(self) -> List[Dict[str, Any]]:
        """
        Возвращает список возможных исправлений для проблем системы.
        
        Returns:
            List[Dict[str, Any]]: Список исправлений
        """
        try:
            return self.learning_opportunity_manager.get_fixes()
        except Exception as e:
            logger.error(f"Ошибка получения списка исправлений: {e}")
            return []
    
    def analyze_user_feedback(self) -> Dict[str, Any]:
        """Анализирует пользовательский фидбэк для выявления проблем."""
        if not self._check_ready():
            return {
                "error": "Самоанализ недоступен",
                "state": "error",
                "message": self.init_error if self.init_error else "не инициализирован",
                "timestamp": time.time()
            }
            
        try:
            if not self._models_ready():
                return {
                    "error": "Модели не готовы",
                    "state": "loading",
                    "message": "Ожидание загрузки моделей",
                    "timestamp": time.time()
                }
                
            result = self.performance_analyzer.analyze_user_feedback()
            result["state"] = "ready"
            return result
        except Exception as e:
            logger.error(f"Ошибка анализа пользовательского фидбэка: {e}")
            return {
                "error": str(e), 
                "state": "error",
                "message": "Ошибка анализа фидбэка",
                "timestamp": time.time()
            }
    
    def analyze_knowledge_gaps(self) -> Dict[str, Any]:
        """Анализирует пробелы в знаниях для выявления возможностей."""
        # В реальной реализации здесь будет код для анализа пробелов в знаниях
        # Для упрощения возвращаем заглушку
        logger.warning("Анализ пробелов в знаниях не реализован в текущей версии")
        return {
            "status": "not_implemented",
            "message": "Метод не реализован",
            "timestamp": time.time()
        }
    
    def analyze_contradictions(self) -> Dict[str, Any]:
        """Анализирует противоречия для выявления возможностей."""
        # В реальной реализации здесь будет код для анализа противоречий
        # Для упрощения возвращаем заглушку
        logger.warning("Анализ противоречий не реализован в текущей версии")
        return {
            "status": "not_implemented",
            "message": "Метод не реализован",
            "timestamp": time.time()
        }
    
    def close(self):
        """Закрывает модуль самоанализа и освобождает ресурсы."""
        logger.info("Закрытие модуля самоанализа...")
        
        try:
            # Останавливаем фоновый анализ
            self.stop()
            
            # Сохраняем данные
            if hasattr(self.analyzer_core, '_save_data'):
                self.analyzer_core._save_data()
                
        except Exception as e:
            logger.error(f"Ошибка при закрытии модуля самоанализа: {e}")
        
        logger.info("Модуль самоанализа закрыт")
    
    def _models_ready(self) -> bool:
        """Проверяет готовность всех необходимых моделей"""
        if not self.brain or not hasattr(self.brain, 'model_manager'):
            logger.warning("ModelManager недоступен")
            return False
            
        required_models = ['analyzer', 'ethics', 'knowledge']
        for model_id in required_models:
            state = self.brain.model_manager.model_states.get(model_id)
            if state == "loading":
                logger.warning(f"Модель {model_id} ещё загружается")
                return False
            elif state == "error":
                error = self.brain.model_manager.loading_errors.get(model_id, "недоступно")
                logger.warning(f"Ошибка загрузки модели {model_id}: {error}")
                return False
            elif not state:
                logger.warning(f"Модель {model_id} не загружена")
                return False
        return True
        
    def start_learning_process(self):
        """Запускает процесс обучения."""
        if not self._check_ready():
            return False
            
        try:
            # Не начинать обучение до готовности моделей
            if not self._models_ready():
                logger.warning("Обучение отложено: модели ещё не загружены")
                try:
                    if hasattr(self.brain, 'status_queue') and self.brain.status_queue:
                        self.brain.status_queue.put(("self_learning", 0, "Ожидание загрузки моделей"))
                except Exception:
                    pass
                return False
            # Пытаемся запустить обучение графа памяти
            if self.memory_graph_trainer:
                return self.memory_graph_trainer.start_learning_process()
            elif hasattr(self.analyzer_core, 'start_learning_process'):
                return self.analyzer_core.start_learning_process()
            else:
                logger.warning("Методы обучения недоступны")
                return False
        except Exception as e:
            logger.error(f"Ошибка запуска процесса обучения: {e}")
            return False
    
    def pause_learning_process(self):
        """Приостанавливает процесс обучения."""
        try:
            # Пытаемся приостановить обучение графа памяти
            if self.memory_graph_trainer:
                return self.memory_graph_trainer.pause_learning_process()
            elif hasattr(self.analyzer_core, 'pause_learning_process'):
                return self.analyzer_core.pause_learning_process()
            else:
                logger.warning("Методы обучения недоступны")
                return False
        except Exception as e:
            logger.error(f"Ошибка приостановки процесса обучения: {e}")
            return False

    # ----------------------------
    # Readiness helpers
    # ----------------------------
    def _models_ready(self) -> bool:
        """Проверяет, загружена ли хотя бы одна ML-модель или фрактальное хранилище."""
        try:
            if not self.brain:
                return False
            # Проверяем fractal_ready - если активно, считаем что модели готовы
            if bool(getattr(self.brain, 'fractal_ready', False)):
                return True
            if bool(getattr(self.brain, 'models_ready', False)):
                return True
            # Попробовать спросить у model_manager
            mm = getattr(self.brain, 'model_manager', None)
            if mm and hasattr(mm, 'models') and isinstance(mm.models, dict) and len(mm.models) > 0:
                return True
            return False
        except Exception:
            return False