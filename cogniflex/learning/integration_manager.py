"""
Learning Integration Manager - Применение результатов обучения к целевым модулям
"""

import time
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import json
import logging
from enum import Enum
import copy


class IntegrationStrategy(Enum):
    """Стратегии интеграции"""
    DIRECT_UPDATE = "direct_update"
    GRADUAL_UPDATE = "gradual_update"
    VALIDATION_FIRST = "validation_first"
    ROLLBACK_ON_ERROR = "rollback_on_error"
    BATCH_UPDATE = "batch_update"


class IntegrationResult:
    """Результат интеграции"""
    
    def __init__(self, task_id: str, target_module: str, strategy: IntegrationStrategy):
        self.task_id = task_id
        self.target_module = target_module
        self.strategy = strategy
        self.start_time = datetime.now()
        self.end_time = None
        self.status = "pending"  # pending, processing, completed, failed, rolled_back
        self.result = None
        self.error = None
        self.performance_before = None
        self.performance_after = None
        self.rollback_available = False
        self.rollback_data = None
    
    def complete(self, result: Dict[str, Any], performance_after: Optional[Dict] = None) -> None:
        """Завершение интеграции успешно"""
        self.end_time = datetime.now()
        self.status = "completed"
        self.result = result
        self.performance_after = performance_after
    
    def fail(self, error: str) -> None:
        """Завершение интеграции с ошибкой"""
        self.end_time = datetime.now()
        self.status = "failed"
        self.error = error
    
    def rollback(self) -> None:
        """Откат изменений"""
        self.end_time = datetime.now()
        self.status = "rolled_back"
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            'task_id': self.task_id,
            'target_module': self.target_module,
            'strategy': self.strategy.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'status': self.status,
            'result': self.result,
            'error': self.error,
            'performance_before': self.performance_before,
            'performance_after': self.performance_after,
            'rollback_available': self.rollback_available,
            'processing_time': (self.end_time - self.start_time).total_seconds() if self.end_time else None
        }


class LearningIntegrationManager:
    """Менеджер интеграции результатов обучения"""
    
    def __init__(self, brain=None):
        self.brain = brain
        self.logger = logging.getLogger(__name__)
        self.integration_queue = []
        self.completed_integrations = []
        self.integration_lock = threading.Lock()
        self.is_running = False
        self.integration_thread = None
        self.integration_stop_event = threading.Event()
        self.integration_strategies = {
            IntegrationStrategy.DIRECT_UPDATE: self._direct_update_strategy,
            IntegrationStrategy.GRADUAL_UPDATE: self._gradual_update_strategy,
            IntegrationStrategy.VALIDATION_FIRST: self._validation_first_strategy,
            IntegrationStrategy.ROLLBACK_ON_ERROR: self._rollback_on_error_strategy,
            IntegrationStrategy.BATCH_UPDATE: self._batch_update_strategy
        }
        
        # Валидаторы для разных типов модулей
        self.validators = {
            'ml_unit': self._validate_ml_unit_integration,
            'knowledge_graph': self._validate_knowledge_graph_integration,
            'ethics_framework': self._validate_ethics_integration,
            'contradiction_manager': self._validate_contradiction_integration,
            'adaptation_manager': self._validate_adaptation_integration,
            'memory_manager': self._validate_memory_integration,
            'query_processor': self._validate_query_integration
        }
        
        # Метрики производительности для валидации
        self.performance_collectors = {
            'ml_unit': self._collect_ml_unit_performance,
            'knowledge_graph': self._collect_knowledge_graph_performance,
            'ethics_framework': self._collect_ethics_performance,
            'contradiction_manager': self._collect_contradiction_performance,
            'adaptation_manager': self._collect_adaptation_performance,
            'memory_manager': self._collect_memory_performance,
            'query_processor': self._collect_query_performance
        }
    
    def initialize(self) -> bool:
        """Инициализация менеджера интеграции"""
        try:
            self.logger.info("Инициализация LearningIntegrationManager...")
            
            # Запуск обработки очереди интеграции
            self.start_integration_processing()
            
            self.logger.info(f"LearningIntegrationManager инициализирован")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при инициализации LearningIntegrationManager: {e}", exc_info=True)
            return False
    
    def start_integration_processing(self) -> None:
        """Запуск обработки интеграции"""
        if self.is_running:
            return
            
        self.integration_stop_event.clear()
        self.is_running = True
        self.integration_thread = threading.Thread(target=self._integration_loop, daemon=True)
        self.integration_thread.start()
        self.logger.info("Обработка интеграции запущена")
    
    def stop_integration_processing(self) -> None:
        """Остановка обработки интеграции"""
        self.integration_stop_event.set()
        self.is_running = False
        if self.integration_thread:
            self.integration_thread.join(timeout=5)
        self.logger.info("Обработка интеграции остановлена")
    
    def _integration_loop(self) -> None:
        """Основной цикл обработки интеграции"""
        while self.is_running and not self.integration_stop_event.is_set():
            try:
                self.process_next_integration()
                time.sleep(1)  # Проверка каждую секунду
            except Exception as e:
                self.logger.error(f"Ошибка в цикле интеграции: {e}")
                time.sleep(5)  # Пауза при ошибке
    
    def queue_integration(self, task_id: str, target_module: str, learning_result: Dict[str, Any], 
                         strategy: IntegrationStrategy = IntegrationStrategy.VALIDATION_FIRST) -> bool:
        """Добавление интеграции в очередь"""
        try:
            integration_result = IntegrationResult(task_id, target_module, strategy)
            
            # Сбор производительности до интеграции
            integration_result.performance_before = self._collect_performance(target_module)
            
            # Подготовка данных для интеграции
            integration_data = {
                'task_id': task_id,
                'target_module': target_module,
                'learning_result': learning_result,
                'strategy': strategy,
                'integration_result': integration_result
            }
            
            with self.integration_lock:
                self.integration_queue.append(integration_data)
            
            self.logger.info(f"Интеграция добавлена в очередь: {task_id} -> {target_module}")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка добавления интеграции в очередь: {e}")
            return False
    
    def process_next_integration(self) -> Optional[IntegrationResult]:
        """Обработка следующей интеграции"""
        with self.integration_lock:
            if not self.integration_queue:
                return None
            
            integration_data = self.integration_queue.pop(0)
        
        task_id = integration_data['task_id']
        target_module = integration_data['target_module']
        learning_result = integration_data['learning_result']
        strategy = integration_data['strategy']
        integration_result = integration_data['integration_result']
        
        try:
            integration_result.status = "processing"
            
            # Выполнение стратегии интеграции
            strategy_func = self.integration_strategies.get(strategy)
            if not strategy_func:
                raise ValueError(f"Неизвестная стратегия интеграции: {strategy}")
            
            result = strategy_func(target_module, learning_result, integration_result)
            
            # Валидация результата
            if self._validate_integration(target_module, result):
                integration_result.complete(result)
                self._collect_performance_after(target_module, integration_result)
                
                with self.integration_lock:
                    self.completed_integrations.append(integration_result)
                
                self.logger.info(f"Интеграция успешно завершена: {task_id} -> {target_module}")
            else:
                raise ValueError("Валидация интеграции не пройдена")
            
        except Exception as e:
            self.logger.error(f"Ошибка интеграции {task_id} -> {target_module}: {e}")
            integration_result.fail(str(e))
            
            # Попытка отката если стратегия поддерживает
            if strategy == IntegrationStrategy.ROLLBACK_ON_ERROR and integration_result.rollback_available:
                self._rollback_integration(integration_result)
            
            with self.integration_lock:
                self.completed_integrations.append(integration_result)
        
        return integration_result
    
    def _validate_integration(self, target_module: str, result: Dict[str, Any]) -> bool:
        """Валидация интеграции"""
        validator = self.validators.get(target_module)
        if validator:
            return validator(result)
        return True  # По умолчанию считаем успешным
    
    def _collect_performance(self, target_module: str) -> Optional[Dict[str, Any]]:
        """Сбор метрик производительности"""
        collector = self.performance_collectors.get(target_module)
        if collector:
            return collector()
        return None
    
    def _collect_performance_after(self, target_module: str, integration_result: IntegrationResult) -> None:
        """Сбор метрик после интеграции"""
        integration_result.performance_after = self._collect_performance(target_module)
    
    # Стратегии интеграции
    def _direct_update_strategy(self, target_module: str, learning_result: Dict[str, Any], 
                               integration_result: IntegrationResult) -> Dict[str, Any]:
        """Прямое обновление"""
        if not self.brain:
            raise ValueError("Brain недоступен")
        
        target_component = getattr(self.brain, target_module, None)
        if not target_component:
            raise ValueError(f"Компонент {target_module} недоступен")
        
        # Применение результатов обучения
        if hasattr(target_component, 'apply_learning_result'):
            result = target_component.apply_learning_result(learning_result)
        else:
            # Базовое применение результатов
            for key, value in learning_result.items():
                if hasattr(target_component, key):
                    setattr(target_component, key, value)
            result = {"updated_keys": list(learning_result.keys())}
        
        return result
    
    def _gradual_update_strategy(self, target_module: str, learning_result: Dict[str, Any], 
                                integration_result: IntegrationResult) -> Dict[str, Any]:
        """Постепенное обновление"""
        if not self.brain:
            raise ValueError("Brain недоступен")
        
        target_component = getattr(self.brain, target_module, None)
        if not target_component:
            raise ValueError(f"Компонент {target_module} недоступен")
        
        # Постепенное применение изменений
        applied_changes = []
        for key, value in learning_result.items():
            try:
                if hasattr(target_component, key):
                    old_value = getattr(target_component, key)
                    # Постепенное изменение
                    if isinstance(old_value, (int, float)) and isinstance(value, (int, float)):
                        new_value = old_value + (value - old_value) * 0.1  # 10% изменения
                        setattr(target_component, key, new_value)
                        applied_changes.append(key)
                    else:
                        setattr(target_component, key, value)
                        applied_changes.append(key)
                
                time.sleep(0.1)  # Небольшая пауза между изменениями
            except Exception as e:
                self.logger.warning(f"Ошибка применения изменения {key}: {e}")
        
        return {"gradual_updates": applied_changes}
    
    def _validation_first_strategy(self, target_module: str, learning_result: Dict[str, Any], 
                                  integration_result: IntegrationResult) -> Dict[str, Any]:
        """Сначала валидация"""
        # Сначала проверяем результат на тестовых данных
        if not self._validate_learning_result(target_module, learning_result):
            raise ValueError("Результат обучения не прошел валидацию")
        
        # Если валидация пройдена, применяем прямое обновление
        return self._direct_update_strategy(target_module, learning_result, integration_result)
    
    def _rollback_on_error_strategy(self, target_module: str, learning_result: Dict[str, Any], 
                                   integration_result: IntegrationResult) -> Dict[str, Any]:
        """Откат при ошибке"""
        if not self.brain:
            raise ValueError("Brain недоступен")
        
        target_component = getattr(self.brain, target_module, None)
        if not target_component:
            raise ValueError(f"Компонент {target_module} недоступен")
        
        # Сохранение текущего состояния для отката
        backup_state = {}
        for key, value in learning_result.items():
            if hasattr(target_component, key):
                backup_state[key] = getattr(target_component, key)
        
        integration_result.rollback_data = backup_state
        integration_result.rollback_available = True
        
        try:
            # Применение изменений
            result = self._direct_update_strategy(target_module, learning_result, integration_result)
            
            # Проверка стабильности
            time.sleep(1)  # Даем время на стабилизацию
            
            if not self._check_component_stability(target_module):
                raise ValueError("Компонент нестабилен после обновления")
            
            return result
            
        except Exception as e:
            # Автоматический откат при ошибке
            self._rollback_to_state(target_component, backup_state)
            raise e
    
    def _batch_update_strategy(self, target_module: str, learning_result: Dict[str, Any], 
                              integration_result: IntegrationResult) -> Dict[str, Any]:
        """Пакетное обновление"""
        if not self.brain:
            raise ValueError("Brain недоступен")
        
        target_component = getattr(self.brain, target_module, None)
        if not target_component:
            raise ValueError(f"Компонент {target_module} недоступен")
        
        # Группировка изменений по типам
        batch_updates = {}
        for key, value in learning_result.items():
            update_type = type(value).__name__
            if update_type not in batch_updates:
                batch_updates[update_type] = {}
            batch_updates[update_type][key] = value
        
        # Применение пакетами
        applied_batches = []
        for update_type, updates in batch_updates.items():
            try:
                for key, value in updates.items():
                    if hasattr(target_component, key):
                        setattr(target_component, key, value)
                applied_batches.append(update_type)
                time.sleep(0.1)  # Пауза между пакетами
            except Exception as e:
                self.logger.warning(f"Ошибка применения пакета {update_type}: {e}")
        
        return {"batch_updates": applied_batches}
    
    # Валидаторы
    def _validate_ml_unit_integration(self, result: Dict[str, Any]) -> bool:
        """Валидация интеграции ML юнита"""
        if not self.brain or not hasattr(self.brain, 'ml_unit'):
            return False
        
        ml_unit = self.brain.ml_unit
        
        # Проверка базовой функциональности
        if hasattr(ml_unit, 'validate_model'):
            return ml_unit.validate_model()
        
        return True
    
    def _validate_knowledge_graph_integration(self, result: Dict[str, Any]) -> bool:
        """Валидация интеграции графа знаний"""
        if not self.brain or not hasattr(self.brain, 'knowledge_graph'):
            return False
        
        kg = self.brain.knowledge_graph
        
        # Проверка целостности графа
        if hasattr(kg, 'validate_integrity'):
            return kg.validate_integrity()
        
        return True
    
    def _validate_ethics_integration(self, result: Dict[str, Any]) -> bool:
        """Валидация интеграции этического фреймворка"""
        if not self.brain or not hasattr(self.brain, 'ethics_framework'):
            return False
        
        ethics = self.brain.ethics_framework
        
        # Проверка этических норм
        if hasattr(ethics, 'validate_ethics'):
            return ethics.validate_ethics()
        
        return True
    
    def _validate_contradiction_integration(self, result: Dict[str, Any]) -> bool:
        """Валидация интеграции менеджера противоречий"""
        if not self.brain or not hasattr(self.brain, 'contradiction_manager'):
            return False
        
        manager = self.brain.contradiction_manager
        
        # Проверка логической целостности
        if hasattr(manager, 'validate_logic'):
            return manager.validate_logic()
        
        return True
    
    def _validate_adaptation_integration(self, result: Dict[str, Any]) -> bool:
        """Валидация интеграции менеджера адаптации"""
        if not self.brain or not hasattr(self.brain, 'adaptation_manager'):
            return False
        
        adaptation = self.brain.adaptation_manager
        
        # Проверка адаптивности
        if hasattr(adaptation, 'validate_adaptation'):
            return adaptation.validate_adaptation()
        
        return True
    
    def _validate_memory_integration(self, result: Dict[str, Any]) -> bool:
        """Валидация интеграции менеджера памяти"""
        if not self.brain or not hasattr(self.brain, 'memory_manager'):
            return False
        
        memory = self.brain.memory_manager
        
        # Проверка целостности памяти
        if hasattr(memory, 'validate_memory'):
            return memory.validate_memory()
        
        return True
    
    def _validate_query_integration(self, result: Dict[str, Any]) -> bool:
        """Валидация интеграции процессора запросов"""
        if not self.brain or not hasattr(self.brain, 'query_processor'):
            return False
        
        processor = self.brain.query_processor
        
        # Проверка обработки запросов
        if hasattr(processor, 'validate_processing'):
            return processor.validate_processing()
        
        return True
    
    # Вспомогательные методы
    def _validate_learning_result(self, target_module: str, learning_result: Dict[str, Any]) -> bool:
        """Валидация результата обучения"""
        # Базовая проверка
        if not isinstance(learning_result, dict):
            return False
        
        # Проверка наличия необходимых полей
        required_fields = ['model_updates', 'performance_metrics']
        for field in required_fields:
            if field not in learning_result:
                self.logger.warning(f"Отсутствует поле {field} в результате обучения")
                return False
        
        return True
    
    def _check_component_stability(self, target_module: str) -> bool:
        """Проверка стабильности компонента"""
        try:
            # Базовая проверка - компонент должен отвечать
            if not self.brain:
                return False
            
            component = getattr(self.brain, target_module, None)
            if not component:
                return False
            
            # Проверка базовых методов
            if hasattr(component, 'health_check'):
                return component.health_check()
            
            return True
        except Exception as e:
            self.logger.error(f"Ошибка проверки стабильности {target_module}: {e}")
            return False
    
    def _rollback_to_state(self, component: Any, backup_state: Dict[str, Any]) -> None:
        """Откат к сохраненному состоянию"""
        try:
            for key, value in backup_state.items():
                if hasattr(component, key):
                    setattr(component, key, value)
            self.logger.info("Откат выполнен успешно")
        except Exception as e:
            self.logger.error(f"Ошибка отката: {e}")
    
    def _rollback_integration(self, integration_result: IntegrationResult) -> None:
        """Откат интеграции"""
        try:
            target_module = integration_result.target_module
            backup_state = integration_result.rollback_data
            
            if not self.brain or not backup_state:
                integration_result.rollback()
                return
            
            component = getattr(self.brain, target_module, None)
            if component:
                self._rollback_to_state(component, backup_state)
            
            integration_result.rollback()
            self.logger.info(f"Интеграция откачена: {integration_result.task_id}")
            
        except Exception as e:
            self.logger.error(f"Ошибка отката интеграции: {e}")
            integration_result.rollback()
    
    # Сборщики метрик производительности
    def _collect_ml_unit_performance(self) -> Dict[str, Any]:
        """Сбор метрик ML юнита"""
        if not self.brain or not hasattr(self.brain, 'ml_unit'):
            return {}
        
        ml_unit = self.brain.ml_unit
        return {
            'accuracy': getattr(ml_unit, 'accuracy', 0.0),
            'predictions_count': getattr(ml_unit, 'predictions_count', 0),
            'model_version': getattr(ml_unit, 'model_version', 'unknown')
        }
    
    def _collect_knowledge_graph_performance(self) -> Dict[str, Any]:
        """Сбор метрик графа знаний"""
        if not self.brain or not hasattr(self.brain, 'knowledge_graph'):
            return {}
        
        kg = self.brain.knowledge_graph
        return {
            'nodes_count': getattr(kg, 'nodes_count', 0),
            'edges_count': getattr(kg, 'edges_count', 0),
            'query_performance': getattr(kg, 'query_performance', 0.0)
        }
    
    def _collect_ethics_performance(self) -> Dict[str, Any]:
        """Сбор метрик этического фреймворка"""
        if not self.brain or not hasattr(self.brain, 'ethics_framework'):
            return {}
        
        ethics = self.brain.ethics_framework
        return {
            'ethical_score': getattr(ethics, 'ethical_score', 0.0),
            'evaluation_count': getattr(ethics, 'evaluation_count', 0)
        }
    
    def _collect_contradiction_performance(self) -> Dict[str, Any]:
        """Сбор метрик менеджера противоречий"""
        if not self.brain or not hasattr(self.brain, 'contradiction_manager'):
            return {}
        
        manager = self.brain.contradiction_manager
        return {
            'resolution_rate': getattr(manager, 'resolution_rate', 0.0),
            'active_contradictions': getattr(manager, 'active_count', 0)
        }
    
    def _collect_adaptation_performance(self) -> Dict[str, Any]:
        """Сбор метрик менеджера адаптации"""
        if not self.brain or not hasattr(self.brain, 'adaptation_manager'):
            return {}
        
        adaptation = self.brain.adaptation_manager
        return {
            'improvement_rate': getattr(adaptation, 'improvement_rate', 0.0),
            'adaptation_cycles': getattr(adaptation, 'cycle_count', 0)
        }
    
    def _collect_memory_performance(self) -> Dict[str, Any]:
        """Сбор метрик менеджера памяти"""
        if not self.brain or not hasattr(self.brain, 'memory_manager'):
            return {}
        
        memory = self.brain.memory_manager
        return {
            'memory_size': getattr(memory, 'memory_size', 0),
            'access_count': getattr(memory, 'access_count', 0),
            'hit_rate': getattr(memory, 'hit_rate', 0.0)
        }
    
    def _collect_query_performance(self) -> Dict[str, Any]:
        """Сбор метрик процессора запросов"""
        if not self.brain or not hasattr(self.brain, 'query_processor'):
            return {}
        
        processor = self.brain.query_processor
        return {
            'processed_queries': getattr(processor, 'query_count', 0),
            'success_rate': getattr(processor, 'success_rate', 0.0),
            'average_response_time': getattr(processor, 'avg_response_time', 0.0)
        }
    
    def get_integration_statistics(self) -> Dict[str, Any]:
        """Получение статистики интеграции"""
        with self.integration_lock:
            stats = {
                'pending_integrations': len(self.integration_queue),
                'completed_integrations': len(self.completed_integrations),
                'success_rate': 0.0,
                'strategy_distribution': {},
                'module_distribution': {},
                'average_processing_time': 0.0,
                'rollback_count': 0
            }
            
            if not self.completed_integrations:
                return stats
            
            successful_count = 0
            total_time = 0.0
            
            for integration in self.completed_integrations:
                # Успешность
                if integration.status == "completed":
                    successful_count += 1
                
                # Стратегии
                strategy = integration.strategy.value
                stats['strategy_distribution'][strategy] = stats['strategy_distribution'].get(strategy, 0) + 1
                
                # Модули
                module = integration.target_module
                stats['module_distribution'][module] = stats['module_distribution'].get(module, 0) + 1
                
                # Время обработки
                if integration.processing_time:
                    total_time += integration.processing_time
                
                # Откаты
                if integration.status == "rolled_back":
                    stats['rollback_count'] += 1
            
            stats['success_rate'] = successful_count / len(self.completed_integrations)
            
            if self.completed_integrations:
                stats['average_processing_time'] = total_time / len(self.completed_integrations)
        
        return stats
