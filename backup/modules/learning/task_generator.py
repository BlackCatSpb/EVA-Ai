"""
Learning Task Generator - Генерация заданий обучения на основе анализа данных
"""

import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import logging
from enum import Enum
import random


class TaskType(Enum):
    """Типы заданий обучения"""
    SUPERVISED_LEARNING = "supervised_learning"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    KNOWLEDGE_REASONING = "knowledge_reasoning"
    QUERY_UNDERSTANDING = "query_understanding"
    KNOWLEDGE_GRAPH_LEARNING = "knowledge_graph_learning"
    ETHICAL_REASONING = "ethical_reasoning"
    ADAPTATION_LEARNING = "adaptation_learning"
    CONTRADICTION_RESOLUTION = "contradiction_resolution"


class TaskPriority(Enum):
    """Приоритеты заданий"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class LearningTask:
    """Задание обучения"""
    
    def __init__(self, task_id: str, task_type: TaskType, priority: TaskPriority, 
                 data: Dict[str, Any], target_module: str, description: str):
        self.task_id = task_id
        self.task_type = task_type
        self.priority = priority
        self.data = data
        self.target_module = target_module
        self.description = description
        self.created_at = datetime.now()
        self.status = "pending"  # pending, processing, completed, failed
        self.result = None
        self.error = None
        self.processing_time = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type.value,
            'priority': self.priority.value,
            'data': self.data,
            'target_module': self.target_module,
            'description': self.description,
            'created_at': self.created_at.isoformat(),
            'status': self.status,
            'result': self.result,
            'error': self.error,
            'processing_time': self.processing_time
        }


class LearningTaskGenerator:
    """Генератор заданий обучения"""
    
    def __init__(self, brain=None):
        self.brain = brain
        self.logger = logging.getLogger(__name__)
        self.task_queue = []
        self.completed_tasks = []
        self.task_lock = threading.Lock()
        self.is_running = False
        self.generation_thread = None
        self.generation_interval = 600  # 10 минут
        self.max_queue_size = 50
        self.task_counter = 0
        self.generation_stop_event = threading.Event()
        
        # Паттерны для генерации заданий
        self.pattern_analyzers = {
            'performance_degradation': self._analyze_performance_degradation,
            'knowledge_gaps': self._analyze_knowledge_gaps,
            'contradiction_patterns': self._analyze_contradiction_patterns,
            'ethical_dilemmas': self._analyze_ethical_patterns,
            'adaptation_opportunities': self._analyze_adaptation_patterns,
            'query_patterns': self._analyze_query_patterns,
            'memory_patterns': self._analyze_memory_patterns
        }
    
    def initialize(self) -> bool:
        """Инициализация генератора заданий"""
        try:
            self.logger.info("Инициализация LearningTaskGenerator...")
            
            # Запуск автоматической генерации
            self.start_generation()
            
            self.logger.info(f"LearningTaskGenerator инициализирован")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при инициализации LearningTaskGenerator: {e}", exc_info=True)
            return False
    
    def start_generation(self) -> None:
        """Запуск автоматической генерации заданий"""
        if self.is_running:
            return
            
        self.generation_stop_event.clear()
        self.is_running = True
        self.generation_thread = threading.Thread(target=self._generation_loop, daemon=True)
        self.generation_thread.start()
        self.logger.info("Автоматическая генерация заданий запущена")
    
    def stop_generation(self) -> None:
        """Остановка автоматической генерации заданий"""
        self.generation_stop_event.set()
        self.is_running = False
        if self.generation_thread:
            self.generation_thread.join(timeout=5)
        self.logger.info("Автоматическая генерация заданий остановлена")
    
    def _generation_loop(self) -> None:
        """Основной цикл генерации заданий"""
        while self.is_running and not self.generation_stop_event.is_set():
            try:
                self.generate_tasks_from_data()
                time.sleep(self.generation_interval)
            except Exception as e:
                self.logger.error(f"Ошибка в цикле генерации заданий: {e}")
                time.sleep(60)  # Пауза при ошибке
    
    def generate_tasks_from_data(self, data: Optional[Dict[str, Any]] = None) -> List[LearningTask]:
        """Генерация заданий на основе данных"""
        generated_tasks = []
        
        # Если данные не предоставлены, получаем из DataProcessor
        if data is None and self.brain and hasattr(self.brain, 'data_processor'):
            data = self.brain.data_processor.get_recent_data(hours=1)
        
        if not data:
            return generated_tasks
        
        # Анализ паттернов и генерация заданий
        for pattern_name, analyzer in self.pattern_analyzers.items():
            try:
                tasks = analyzer(data)
                if tasks:
                    generated_tasks.extend(tasks)
            except Exception as e:
                self.logger.warning(f"Ошибка анализа паттерна {pattern_name}: {e}")
        
        # Приоритизация и добавление в очередь
        self._queue_tasks(generated_tasks)
        
        self.logger.info(f"Сгенерировано {len(generated_tasks)} заданий обучения")
        return generated_tasks
    
    def _queue_tasks(self, tasks: List[LearningTask]) -> None:
        """Добавление заданий в очередь с приоритизацией"""
        with self.task_lock:
            # Сортировка по приоритету
            tasks.sort(key=lambda t: t.priority.value, reverse=True)
            
            # Добавление в очередь с учетом размера
            for task in tasks:
                if len(self.task_queue) >= self.max_queue_size:
                    # Удаление старых заданий с низким приоритетом
                    self.task_queue = [t for t in self.task_queue if t.priority.value > TaskPriority.LOW.value]
                    if len(self.task_queue) >= self.max_queue_size:
                        self.task_queue.pop(0)  # Удаление самого старого
                
                self.task_queue.append(task)
    
    def get_next_task(self) -> Optional[LearningTask]:
        """Получение следующего задания из очереди"""
        with self.task_lock:
            if not self.task_queue:
                return None
            
            # Сортировка по приоритету
            self.task_queue.sort(key=lambda t: (t.priority.value, t.created_at), reverse=True)
            return self.task_queue.pop(0)
    
    def complete_task(self, task_id: str, result: Optional[Dict[str, Any]] = None, 
                     error: Optional[str] = None) -> bool:
        """Завершение задания"""
        with self.task_lock:
            for task in self.completed_tasks:
                if task.task_id == task_id:
                    task.status = "completed" if result else "failed"
                    task.result = result
                    task.error = error
                    task.processing_time = (datetime.now() - task.created_at).total_seconds()
                    return True
        return False
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """Получение статистики заданий"""
        with self.task_lock:
            stats = {
                'pending_tasks': len(self.task_queue),
                'completed_tasks': len(self.completed_tasks),
                'task_types': {},
                'priority_distribution': {},
                'status_distribution': {},
                'average_processing_time': 0.0,
                'success_rate': 0.0
            }
            
            # Анализ завершенных заданий
            completed_count = 0
            total_time = 0.0
            successful_count = 0
            
            for task in self.completed_tasks:
                # Типы заданий
                task_type = task.task_type.value
                stats['task_types'][task_type] = stats['task_types'].get(task_type, 0) + 1
                
                # Распределение приоритетов
                priority = task.priority.name
                stats['priority_distribution'][priority] = stats['priority_distribution'].get(priority, 0) + 1
                
                # Распределение статусов
                status = task.status
                stats['status_distribution'][status] = stats['status_distribution'].get(status, 0) + 1
                
                # Время обработки
                if task.processing_time:
                    completed_count += 1
                    total_time += task.processing_time
                
                # Успешность
                if task.status == "completed":
                    successful_count += 1
            
            if completed_count > 0:
                stats['average_processing_time'] = total_time / completed_count
            
            if len(self.completed_tasks) > 0:
                stats['success_rate'] = successful_count / len(self.completed_tasks)
        
        return stats
    
    # Методы анализа паттернов
    def _analyze_performance_degradation(self, data: Dict[str, Any]) -> List[LearningTask]:
        """Анализ деградации производительности"""
        tasks = []
        
        # Проверка производительности ML юнита
        if 'ml_unit' in data:
            ml_data = data['ml_unit']['data']
            accuracy = ml_data.get('accuracy', 0.0)
            
            if accuracy < 0.7:  # Порог точности
                task = self._create_task(
                    TaskType.SUPERVISED_LEARNING,
                    TaskPriority.HIGH,
                    ml_data,
                    'ml_unit',
                    f"Низкая точность модели: {accuracy:.2f}"
                )
                tasks.append(task)
        
        # Проверка успешности обработки запросов
        if 'query_processor' in data:
            query_data = data['query_processor']['data']
            success_rate = query_data.get('success_rate', 0.0)
            
            if success_rate < 0.8:  # Порог успешности
                task = self._create_task(
                    TaskType.QUERY_UNDERSTANDING,
                    TaskPriority.MEDIUM,
                    query_data,
                    'query_processor',
                    f"Низкая успешность обработки запросов: {success_rate:.2f}"
                )
                tasks.append(task)
        
        return tasks
    
    def _analyze_knowledge_gaps(self, data: Dict[str, Any]) -> List[LearningTask]:
        """Анализ пробелов в знаниях"""
        tasks = []
        
        if 'knowledge_graph_new' in data:
            kg_data = data['knowledge_graph_new']['data']
            recent_additions = kg_data.get('recent_additions', [])
            
            # Если мало добавлений, нужно больше знаний
            if len(recent_additions) < 5:
                task = self._create_task(
                    TaskType.KNOWLEDGE_REASONING,
                    TaskPriority.MEDIUM,
                    kg_data,
                    'knowledge_graph',
                    "Недостаточно новых знаний в графе"
                )
                tasks.append(task)
        
        return tasks
    
    def _analyze_contradiction_patterns(self, data: Dict[str, Any]) -> List[LearningTask]:
        """Анализ паттернов противоречий"""
        tasks = []
        
        if 'contradiction_manager' in data:
            contradiction_data = data['contradiction_manager']['data']
            resolution_rate = contradiction_data.get('resolution_rate', 0.0)
            
            if resolution_rate < 0.6:  # Порог разрешения
                task = self._create_task(
                    TaskType.CONTRADICTION_RESOLUTION,
                    TaskPriority.HIGH,
                    contradiction_data,
                    'contradiction_manager',
                    f"Низкая скорость разрешения противоречий: {resolution_rate:.2f}"
                )
                tasks.append(task)
        
        return tasks
    
    def _analyze_ethical_patterns(self, data: Dict[str, Any]) -> List[LearningTask]:
        """Анализ этических паттернов"""
        tasks = []
        
        if 'ethics_framework' in data:
            ethics_data = data['ethics_framework']['data']
            ethical_score = ethics_data.get('ethical_score', 0.0)
            
            if ethical_score < 0.7:  # Порог этической оценки
                task = self._create_task(
                    TaskType.ETHICAL_REASONING,
                    TaskPriority.HIGH,
                    ethics_data,
                    'ethics_framework',
                    f"Низкая этическая оценка: {ethical_score:.2f}"
                )
                tasks.append(task)
        
        return tasks
    
    def _analyze_adaptation_patterns(self, data: Dict[str, Any]) -> List[LearningTask]:
        """Анализ паттернов адаптации"""
        tasks = []
        
        if 'adaptation_manager' in data:
            adaptation_data = data['adaptation_manager']['data']
            improvement_rate = adaptation_data.get('improvement_rate', 0.0)
            
            if improvement_rate < 0.3:  # Порог улучшения
                task = self._create_task(
                    TaskType.ADAPTATION_LEARNING,
                    TaskPriority.MEDIUM,
                    adaptation_data,
                    'adaptation_manager',
                    f"Низкая скорость адаптации: {improvement_rate:.2f}"
                )
                tasks.append(task)
        
        return tasks
    
    def _analyze_query_patterns(self, data: Dict[str, Any]) -> List[LearningTask]:
        """Анализ паттернов запросов"""
        tasks = []
        
        if 'query_processor' in data:
            query_data = data['query_processor']['data']
            recent_queries = query_data.get('recent_queries', [])
            
            # Анализ типов запросов
            query_types = {}
            for query in recent_queries:
                q_type = query.get('type', 'unknown')
                query_types[q_type] = query_types.get(q_type, 0) + 1
            
            # Если есть доминирующий тип, создать задание для улучшения
            if query_types:
                dominant_type = max(query_types, key=query_types.get)
                if query_types[dominant_type] > 10:  # Порог частоты
                    task = self._create_task(
                        TaskType.QUERY_UNDERSTANDING,
                        TaskPriority.MEDIUM,
                        {'query_types': query_types, 'dominant_type': dominant_type},
                        'query_processor',
                        f"Доминирующий тип запросов: {dominant_type}"
                    )
                    tasks.append(task)
        
        return tasks
    
    def _analyze_memory_patterns(self, data: Dict[str, Any]) -> List[LearningTask]:
        """Анализ паттернов памяти"""
        tasks = []
        
        if 'memory_manager' in data:
            memory_data = data['memory_manager']['data']
            access_count = memory_data.get('access_count', 0)
            memory_size = memory_data.get('memory_size', 0)
            
            # Если много обращений, но мало памяти, нужно оптимизировать
            if access_count > 100 and memory_size < 1000:
                task = self._create_task(
                    TaskType.REINFORCEMENT_LEARNING,
                    TaskPriority.LOW,
                    memory_data,
                    'memory_manager',
                    "Оптимизация использования памяти"
                )
                tasks.append(task)
        
        return tasks
    
    def _create_task(self, task_type: TaskType, priority: TaskPriority, 
                    data: Dict[str, Any], target_module: str, description: str) -> LearningTask:
        """Создание задания"""
        self.task_counter += 1
        task_id = f"task_{self.task_counter}_{int(time.time())}"
        
        return LearningTask(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            data=data,
            target_module=target_module,
            description=description
        )
