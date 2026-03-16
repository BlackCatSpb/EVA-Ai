"""
Learning Processor - Координация всех компонентов процессора обучения
"""

import time
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import logging
from enum import Enum

from .data_processor import DataProcessor
from .task_generator import LearningTaskGenerator, LearningTask, TaskType, TaskPriority
from .integration_manager import LearningIntegrationManager, IntegrationStrategy


class ProcessorStatus(Enum):
    """Статусы процессора обучения"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


class LearningProcessor:
    """Основной процессор обучения"""
    
    def __init__(self, brain=None):
        self.brain = brain
        self.logger = logging.getLogger(__name__)
        
        # Компоненты процессора
        self.data_processor = None
        self.task_generator = None
        self.integration_manager = None
        
        # Управление процессором
        self.status = ProcessorStatus.STOPPED
        self.start_time = None
        self.stop_time = None
        self.processing_lock = threading.Lock()
        self.main_thread = None
        
        # Статистика и метрики
        self.statistics = {
            'total_cycles': 0,
            'successful_cycles': 0,
            'failed_cycles': 0,
            'total_tasks_generated': 0,
            'total_tasks_processed': 0,
            'total_integrations': 0,
            'average_cycle_time': 0.0,
            'last_cycle_time': None,
            'uptime': 0.0
        }
        
        # Настройки
        self.cycle_interval = 300  # 5 минут
        self.max_concurrent_tasks = 5
        self.health_check_interval = 60  # 1 минута
        self.auto_restart_on_error = True
        
        # События и обратные вызовы
        self.event_callbacks = {}
        self.health_status = {
            'data_processor': True,
            'task_generator': True,
            'integration_manager': True,
            'overall': True
        }
    
    def initialize(self) -> bool:
        """Инициализация процессора обучения"""
        try:
            self.logger.info("Инициализация LearningProcessor...")
            self.status = ProcessorStatus.STARTING
            
            # Инициализация компонентов
            if not self._initialize_components():
                self.status = ProcessorStatus.ERROR
                return False
            
            # Запуск потоков
            if not self._start_processing_threads():
                self.status = ProcessorStatus.ERROR
                return False
            
            self.start_time = datetime.now()
            self.status = ProcessorStatus.RUNNING
            
            self.logger.info(f"LearningProcessor инициализирован и запущен")
            self._trigger_event('processor_started', {'timestamp': self.start_time.isoformat()})
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при инициализации LearningProcessor: {e}", exc_info=True)
            self.status = ProcessorStatus.ERROR
            return False
    
    def _initialize_components(self) -> bool:
        """Инициализация компонентов процессора"""
        try:
            # Инициализация DataProcessor
            self.data_processor = DataProcessor(brain=self.brain)
            if not self.data_processor.initialize():
                self.logger.error("Ошибка инициализации DataProcessor")
                return False
            
            # Инициализация LearningTaskGenerator
            self.task_generator = LearningTaskGenerator(brain=self.brain)
            if not self.task_generator.initialize():
                self.logger.error("Ошибка инициализации LearningTaskGenerator")
                return False
            
            # Инициализация LearningIntegrationManager
            self.integration_manager = LearningIntegrationManager(brain=self.brain)
            if not self.integration_manager.initialize():
                self.logger.error("Ошибка инициализации LearningIntegrationManager")
                return False
            
            self.logger.info(f"Все компоненты процессора инициализированы")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка инициализации компонентов: {e}")
            return False
    
    def _start_processing_threads(self) -> bool:
        """Запуск потоков обработки"""
        try:
            # Основной поток обработки
            self.main_thread = threading.Thread(target=self._main_processing_loop, daemon=True)
            self.main_thread.start()
            
            # Поток проверки здоровья
            health_thread = threading.Thread(target=self._health_check_loop, daemon=True)
            health_thread.start()
            
            self.logger.info(f"Потоки обработки запущены")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка запуска потоков обработки: {e}")
            return False
    
    def _main_processing_loop(self) -> None:
        """Основной цикл обработки"""
        self.logger.info(f"Основной цикл обработки {'запущен' if self.status == ProcessorStatus.RUNNING else 'завершен'}")
        
        while self.status in [ProcessorStatus.RUNNING, ProcessorStatus.PAUSED]:
            try:
                if self.status == ProcessorStatus.RUNNING:
                    cycle_start = time.time()
                    
                    # Выполнение цикла обучения
                    success = self._execute_learning_cycle()
                    
                    # Обновление статистики
                    cycle_time = time.time() - cycle_start
                    self._update_statistics(success, cycle_time)
                    
                    # Пауза между циклами
                    if self.status == ProcessorStatus.RUNNING:
                        time.sleep(self.cycle_interval)
                else:
                    # Пауза при статусе PAUSED
                    time.sleep(5)
                    
            except Exception as e:
                self.logger.error(f"Ошибка в основном цикле обработки: {e}")
                self.statistics['failed_cycles'] += 1
                
                if self.auto_restart_on_error:
                    self.logger.info("Попытка автоматического перезапуска...")
                    time.sleep(30)  # Пауза перед перезапуском
                else:
                    self.status = ProcessorStatus.ERROR
                    break
        
        self.logger.info(f"Основной цикл обработки завершен")
    
    def _execute_learning_cycle(self) -> bool:
        """Выполнение цикла обучения"""
        try:
            self.logger.debug("Начало цикла обучения")
            
            # Шаг 1: Сбор данных
            data = self.data_processor.collect_all_data()
            if not data:
                self.logger.warning("Нет данных для анализа")
                return True  # Не считаем ошибкой
            
            # Шаг 2: Генерация заданий
            tasks = self.task_generator.generate_tasks_from_data(data)
            if not tasks:
                self.logger.debug("Задания обучения не сгенерированы")
                return True
            
            # Шаг 3: Обработка заданий
            processed_tasks = self._process_tasks(tasks)
            
            # Шаг 4: Интеграция результатов
            integrations = self._integrate_results(processed_tasks)
            
            self.logger.info(f"Цикл обучения завершен: {len(tasks)} заданий, {len(integrations)} интеграций")
            self._trigger_event('cycle_completed', {
                'tasks_generated': len(tasks),
                'tasks_processed': len(processed_tasks),
                'integrations': len(integrations)
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка выполнения цикла обучения: {e}")
            return False
    
    def _process_tasks(self, tasks: List[LearningTask]) -> List[Dict[str, Any]]:
        """Обработка заданий обучения"""
        processed_tasks = []
        
        for task in tasks[:self.max_concurrent_tasks]:  # Ограничение concurrent задач
            try:
                # Получение задания из генератора
                task_to_process = self.task_generator.get_next_task()
                if not task_to_process:
                    break
                
                # Симуляция обработки задания (в реальности здесь был бы ML процесс)
                learning_result = self._simulate_learning_process(task_to_process)
                
                if learning_result:
                    processed_tasks.append({
                        'task': task_to_process,
                        'result': learning_result
                    })
                    self.task_generator.complete_task(task_to_process.task_id, learning_result)
                else:
                    self.task_generator.complete_task(task_to_process.task_id, error="Обработка не удалась")
                
            except Exception as e:
                self.logger.error(f"Ошибка обработки задания {task.task_id}: {e}")
                self.task_generator.complete_task(task.task_id, error=str(e))
        
        return processed_tasks
    
    def _simulate_learning_process(self, task: LearningTask) -> Optional[Dict[str, Any]]:
        """Симуляция процесса обучения (заглушка)"""
        try:
            # В реальной реализации здесь был бы процесс обучения
            # Сейчас возвращаем симулированный результат
            
            time.sleep(0.1)  # Симуляция времени обработки
            
            return {
                'model_updates': {
                    'weights_updated': True,
                    'bias_adjusted': True,
                    'learning_rate': 0.001
                },
                'performance_metrics': {
                    'accuracy_improvement': 0.05,
                    'loss_reduction': 0.1,
                    'convergence_rate': 0.8
                },
                'metadata': {
                    'task_type': task.task_type.value,
                    'target_module': task.target_module,
                    'processing_time': 0.1
                }
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка симуляции обучения: {e}")
            return None
    
    def _integrate_results(self, processed_tasks: List[Dict[str, Any]]) -> List[Any]:
        """Интеграция результатов обучения"""
        integrations = []
        
        for task_data in processed_tasks:
            task = task_data['task']
            result = task_data['result']
            
            try:
                # Выбор стратегии интеграции на основе типа задания
                strategy = self._select_integration_strategy(task)
                
                # Добавление в очередь интеграции
                success = self.integration_manager.queue_integration(
                    task_id=task.task_id,
                    target_module=task.target_module,
                    learning_result=result,
                    strategy=strategy
                )
                
                if success:
                    integrations.append(task.task_id)
                
            except Exception as e:
                self.logger.error(f"Ошибка интеграции результата задания {task.task_id}: {e}")
        
        return integrations
    
    def _select_integration_strategy(self, task: LearningTask) -> IntegrationStrategy:
        """Выбор стратегии интеграции на основе типа задания"""
        strategy_map = {
            TaskType.SUPERVISED_LEARNING: IntegrationStrategy.VALIDATION_FIRST,
            TaskType.REINFORCEMENT_LEARNING: IntegrationStrategy.GRADUAL_UPDATE,
            TaskType.KNOWLEDGE_REASONING: IntegrationStrategy.DIRECT_UPDATE,
            TaskType.QUERY_UNDERSTANDING: IntegrationStrategy.BATCH_UPDATE,
            TaskType.KNOWLEDGE_GRAPH_LEARNING: IntegrationStrategy.VALIDATION_FIRST,
            TaskType.ETHICAL_REASONING: IntegrationStrategy.ROLLBACK_ON_ERROR,
            TaskType.ADAPTATION_LEARNING: IntegrationStrategy.GRADUAL_UPDATE,
            TaskType.CONTRADICTION_RESOLUTION: IntegrationStrategy.VALIDATION_FIRST
        }
        
        return strategy_map.get(task.task_type, IntegrationStrategy.VALIDATION_FIRST)
    
    def _health_check_loop(self) -> None:
        """Цикл проверки здоровья компонентов"""
        self.logger.info(f"Цикл проверки здоровья {'запущен' if self.status == ProcessorStatus.RUNNING else 'завершен'}")
        
        while self.status in [ProcessorStatus.RUNNING, ProcessorStatus.PAUSED]:
            try:
                self._perform_health_check()
                time.sleep(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"Ошибка в цикле проверки здоровья: {e}")
                time.sleep(30)
        
        self.logger.info(f"Цикл проверки здоровья завершен")
    
    def _perform_health_check(self) -> None:
        """Выполнение проверки здоровья"""
        previous_health = self.health_status.copy()
        
        # Проверка DataProcessor
        self.health_status['data_processor'] = (
            self.data_processor is not None and 
            hasattr(self.data_processor, 'is_running') and 
            self.data_processor.is_running
        )
        
        # Проверка TaskGenerator
        self.health_status['task_generator'] = (
            self.task_generator is not None and 
            hasattr(self.task_generator, 'is_running') and 
            self.task_generator.is_running
        )
        
        # Проверка IntegrationManager
        self.health_status['integration_manager'] = (
            self.integration_manager is not None and 
            hasattr(self.integration_manager, 'is_running') and 
            self.integration_manager.is_running
        )
        
        # Общий статус здоровья
        self.health_status['overall'] = all(self.health_status.values())
        
        # Проверка изменений здоровья
        if previous_health != self.health_status:
            self.logger.warning(f"Изменение статуса здоровья: {self.health_status}")
            self._trigger_event('health_status_changed', {
                'previous': previous_health,
                'current': self.health_status
            })
    
    def _update_statistics(self, success: bool, cycle_time: float) -> None:
        """Обновление статистики"""
        self.statistics['total_cycles'] += 1
        
        if success:
            self.statistics['successful_cycles'] += 1
        else:
            self.statistics['failed_cycles'] += 1
        
        self.statistics['last_cycle_time'] = cycle_time
        
        # Обновление среднего времени цикла
        if self.statistics['total_cycles'] > 0:
            total_time = (self.statistics.get('average_cycle_time', 0) * 
                         (self.statistics['total_cycles'] - 1) + cycle_time)
            self.statistics['average_cycle_time'] = total_time / self.statistics['total_cycles']
        
        # Обновление времени работы
        if self.start_time:
            self.statistics['uptime'] = (datetime.now() - self.start_time).total_seconds()
        
        # Обновление статистики компонентов
        if self.task_generator:
            task_stats = self.task_generator.get_task_statistics()
            self.statistics['total_tasks_generated'] = task_stats.get('completed_tasks', 0)
        
        if self.integration_manager:
            integration_stats = self.integration_manager.get_integration_statistics()
            self.statistics['total_integrations'] = integration_stats.get('completed_integrations', 0)
    
    def _trigger_event(self, event_name: str, data: Dict[str, Any]) -> None:
        """Trigger события"""
        try:
            if event_name in self.event_callbacks:
                callback = self.event_callbacks[event_name]
                callback(data)
            
            # Отправка события в brain если доступно
            if self.brain and hasattr(self.brain, 'events'):
                self.brain.events.trigger(f'learning_processor_{event_name}', data)
                
        except Exception as e:
            self.logger.error(f"Ошибка trigger события {event_name}: {e}")
    
    # Публичные API методы
    def start(self) -> bool:
        """Запуск процессора"""
        if self.status == ProcessorStatus.RUNNING:
            self.logger.warning("Процессор уже запущен")
            return True
        
        return self.initialize()
    
    def stop(self) -> bool:
        """Остановка процессора"""
        try:
            self.logger.info("Остановка LearningProcessor...")
            self.status = ProcessorStatus.STOPPING
            
            # Остановка компонентов
            if self.data_processor:
                self.data_processor.stop_collection()
            
            if self.task_generator:
                self.task_generator.stop_generation()
            
            if self.integration_manager:
                self.integration_manager.stop_integration_processing()
            
            # Ожидание завершения потоков
            if self.main_thread:
                self.main_thread.join(timeout=10)
            
            self.status = ProcessorStatus.STOPPED
            self.stop_time = datetime.now()
            
            self.logger.info(f"LearningProcessor остановлен")
            self._trigger_event('processor_stopped', {'timestamp': self.stop_time.isoformat()})
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка остановки процессора: {e}")
            return False
    
    def pause(self) -> bool:
        """Пауза процессора"""
        if self.status == ProcessorStatus.RUNNING:
            self.status = ProcessorStatus.PAUSED
            self.logger.info("LearningProcessor поставлен на паузу")
            self._trigger_event('processor_paused', {'timestamp': datetime.now().isoformat()})
            return True
        return False
    
    def resume(self) -> bool:
        """Возобновление работы процессора"""
        if self.status == ProcessorStatus.PAUSED:
            self.status = ProcessorStatus.RUNNING
            self.logger.info("Работа LearningProcessor возобновлена")
            self._trigger_event('processor_resumed', {'timestamp': datetime.now().isoformat()})
            return True
        return False
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Получение полной статистики обучения"""
        stats = self.statistics.copy()
        
        # Добавление статистики компонентов
        if self.data_processor:
            stats['data_processor'] = self.data_processor.get_data_statistics()
        
        if self.task_generator:
            stats['task_generator'] = self.task_generator.get_task_statistics()
        
        if self.integration_manager:
            stats['integration_manager'] = self.integration_manager.get_integration_statistics()
        
        # Добавление информации о статусе
        stats['processor_status'] = self.status.value
        stats['health_status'] = self.health_status.copy()
        stats['start_time'] = self.start_time.isoformat() if self.start_time else None
        stats['stop_time'] = self.stop_time.isoformat() if self.stop_time else None
        
        return stats
    
    def manual_data_collection(self) -> Dict[str, Any]:
        """Ручной сбор данных"""
        if not self.data_processor:
            return {}
        
        return self.data_processor.collect_all_data()
    
    def manual_task_generation(self, data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Ручная генерация заданий"""
        if not self.task_generator:
            return []
        
        # Если данные не предоставлены, получаем из DataProcessor
        if data is None and self.data_processor:
            data = self.data_processor.get_recent_data(hours=1)
        
        tasks = self.task_generator.generate_tasks_from_data(data)
        return [task.to_dict() for task in tasks]
    
    def manual_integration(self, task_id: str, target_module: str, 
                         learning_result: Dict[str, Any], 
                         strategy: str = "validation_first") -> bool:
        """Ручная интеграция"""
        if not self.integration_manager:
            return False
        
        try:
            strategy_enum = IntegrationStrategy(strategy)
            return self.integration_manager.queue_integration(
                task_id, target_module, learning_result, strategy_enum
            )
        except ValueError:
            self.logger.error(f"Неизвестная стратегия интеграции: {strategy}")
            return False
    
    def register_event_callback(self, event_name: str, callback) -> None:
        """Регистрация обратного вызова для событий"""
        self.event_callbacks[event_name] = callback
    
    def unregister_event_callback(self, event_name: str) -> None:
        """Удаление обратного вызова"""
        if event_name in self.event_callbacks:
            del self.event_callbacks[event_name]
    
    def get_status(self) -> Dict[str, Any]:
        """Получение текущего статуса"""
        return {
            'status': self.status.value,
            'health': self.health_status,
            'uptime': self.statistics.get('uptime', 0),
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'last_cycle': self.statistics.get('last_cycle_time'),
            'total_cycles': self.statistics.get('total_cycles', 0)
        }
