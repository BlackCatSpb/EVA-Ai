"""Модуль распределенных задач для ЕВА - распределение задач между узлами"""
import os
import logging
import time
import threading
import queue
import json
import requests
from .cluster_manager import ClusterNode
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
import random
import hashlib

logger = logging.getLogger("eva_ai.distributed.tasks")

class DistributedTask:
    """Представляет задачу для распределенной обработки."""
    
    def __init__(self, task_id: str, task_type: str, payload: Dict[str, Any], 
                 priority: float = 0.5, timeout: float = 30.0,
                 created_at: float = time.time(),
                 status: str = "pending", result: Optional[Dict[str, Any]] = None,
                 error: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
                 assigned_node: Optional[str] = None, started_at: Optional[float] = None,
                 completed_at: Optional[float] = None):
        """
        Инициализирует задачу.
        
        Args:
            task_id: Уникальный ID задачи
            task_type: Тип задачи
            payload: Данные задачи
            priority: Приоритет задачи (0.0-1.0)
            timeout: Таймаут выполнения (сек)
            created_at: Время создания
            status: Статус задачи
            result: Результат выполнения
            error: Описание ошибки
            metadata: Дополнительные метаданные
            assigned_node: Узел, которому назначена задача
            started_at: Время начала выполнения
            completed_at: Время завершения
        """
        self.task_id = task_id
        self.task_type = task_type
        self.payload = payload
        self.priority = priority
        self.timeout = timeout
        self.created_at = created_at
        self.status = status
        self.result = result
        self.error = error
        self.metadata = metadata or {}
        self.assigned_node = assigned_node
        self.started_at = started_at
        self.completed_at = completed_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует задачу в словарь."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "payload": self.payload,
            "priority": self.priority,
            "timeout": self.timeout,
            "created_at": self.created_at,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
            "assigned_node": self.assigned_node,
            "started_at": self.started_at,
            "completed_at": self.completed_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DistributedTask':
        """Создает задачу из словаря."""
        return cls(
            task_id=data["task_id"],
            task_type=data["task_type"],
            payload=data["payload"],
            priority=data["priority"],
            timeout=data["timeout"],
            created_at=data["created_at"],
            status=data["status"],
            result=data["result"],
            error=data["error"],
            metadata=data["metadata"],
            assigned_node=data.get("assigned_node"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at")
        )


class TaskScheduler:
    """Планирует и распределяет задачи между узлами кластера."""
    
    def __init__(self, distributed_system, cluster_manager):
        """
        Инициализирует планировщик задач.
        
        Args:
            distributed_system: Ссылка на распределенную систему
            cluster_manager: Менеджер кластера
        """
        self.distributed_system = distributed_system
        self.cluster_manager = cluster_manager
        
        # Очередь задач
        self.task_queue = queue.PriorityQueue()
        
        # Активные задачи
        self.active_tasks: Dict[str, DistributedTask] = {}
        
        # Блокировка ресурсов
        self.lock = threading.Lock()
        
        # Настройки планировщика
        self.scheduler_settings = {
            "max_concurrent_tasks": 10,  # Максимальное количество одновременных задач
            "task_timeout": 60.0,        # Таймаут задачи (сек)
            "retry_count": 3,            # Количество попыток
            "retry_delay": 5.0,          # Задержка между попытками (сек)
            "enabled": True              # Включен ли планировщик
        }
        
        # Статистика
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "avg_processing_time": 0.0,
            "last_task": 0
        }
        
        # Фоновый процесс
        self.running = False
        self.scheduler_thread = None
        
        logger.info("Планировщик задач инициализирован")
    
    def start(self):
        """Запускает фоновые процессы планировщика задач."""
        if self.running or not self.scheduler_settings["enabled"]:
            return
        
        self.running = True
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="TaskScheduler",
            daemon=True
        )
        self.scheduler_thread.start()
        logger.info("Фоновый процесс планировщика задач запущен")
    
    def stop(self):
        """Останавливает фоновые процессы планировщика задач."""
        if not self.running:
            return
        
        self.running = False
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5.0)
        
        logger.info("Фоновый процесс планировщика задач остановлен")
    
    def _scheduler_loop(self):
        """Цикл планировщика задач."""
        while self.running:
            try:
                # Обрабатываем задачи из очереди
                while not self.task_queue.empty() and \
                      len(self.active_tasks) < self.scheduler_settings["max_concurrent_tasks"]:
                    self._process_next_task()
                
                # Проверяем зависшие задачи
                self._check_stuck_tasks()
                
                # Проверяем здоровье системы
                self._check_system_health()
                
                # Ждем перед следующей проверкой
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Ошибка в цикле планировщика задач: {e}")
                time.sleep(5)
    
    def _process_next_task(self):
        """Обрабатывает следующую задачу из очереди."""
        try:
            # Получаем задачу с наивысшим приоритетом
            priority, task = self.task_queue.get_nowait()
            
            # Отмечаем начало обработки
            task.status = "processing"
            task.started_at = time.time()
            self.active_tasks[task.task_id] = task
            
            # Распределяем задачу
            self._assign_task_to_node(task)
            
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"Ошибка обработки задачи: {e}")
    
    def _assign_task_to_node(self, task: 'DistributedTask'):
        """
        Назначает задачу узлу кластера.
        
        Args:
            task: Задача для назначения
        """
        # Получаем подходящие узлы
        suitable_nodes = self._get_suitable_nodes(task)
        
        if not suitable_nodes:
            logger.warning(f"Нет подходящих узлов для задачи {task.task_id} ({task.task_type})")
            # Возвращаем задачу в очередь с задержкой
            self._requeue_task(task)
            return
        
        # Выбираем узел с наименьшей нагрузкой
        suitable_nodes.sort(key=lambda x: x.load)
        target_node = suitable_nodes[0]
        
        # Отправляем задачу на узел
        self._send_task_to_node(task, target_node)
    
    def _get_suitable_nodes(self, task: 'DistributedTask') -> List[ClusterNode]:
        """
        Возвращает подходящие узлы для выполнения задачи.
        
        Args:
            task: Задача для выполнения
            
        Returns:
            List: Список подходящих узлов
        """
        # Определяем требования к узлу
        required_capabilities = self._get_required_capabilities(task.task_type)
        
        # Получаем активные узлы с необходимыми возможностями
        suitable_nodes = []
        for node in self.cluster_manager.get_active_nodes():
            # Проверяем возможности
            has_capabilities = all(
                node.capabilities.get(cap, False) 
                for cap in required_capabilities
            )
            
            if not has_capabilities:
                continue
            
            # Проверяем нагрузку
            if node.load >= 0.8:  # Не назначаем задачи на перегруженные узлы
                continue
            
            suitable_nodes.append(node)
        
        return suitable_nodes
    
    def _get_required_capabilities(self, task_type: str) -> List[str]:
        """
        Возвращает необходимые возможности для типа задачи.
        
        Args:
            task_type: Тип задачи
            
        Returns:
            List[str]: Список необходимых возможностей
        """
        capabilities_map = {
            "nlp_processing": ["nlp_processing"],
            "web_search": ["web_search", "task_processing"],
            "knowledge_update": ["knowledge_sync"],
            "neural_simulation": ["neural_simulation"],
            "ethics_analysis": ["nlp_processing"]
        }
        
        return capabilities_map.get(task_type, ["task_processing"])
    
    def _send_task_to_node(self, task: 'DistributedTask', node: ClusterNode):
        """
        Отправляет задачу на узел кластера.
        
        Args:
            task: Задача для отправки
            node: Целевой узел
        """
        task.assigned_node = node.node_id
        
        try:
            # Формируем запрос
            url = f"http://{node.address}:{node.port}/api/tasks/execute"
            headers = {"Content-Type": "application/json"}
            payload = {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "payload": task.payload,
                "priority": task.priority,
                "timeout": task.timeout,
                "metadata": task.metadata
            }
            
            # Отправляем запрос
            response = requests.post(
                url, 
                json=payload, 
                headers=headers, 
                timeout=self.scheduler_settings["task_timeout"]
            )
            
            if response.status_code == 200:
                result = response.json()
                self._handle_task_result(task, result)
            else:
                logger.error(f"Ошибка выполнения задачи {task.task_id}: {response.status_code}")
                self._handle_task_error(task, f"HTTP error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Ошибка отправки задачи {task.task_id} на узел {node.node_id}: {e}")
            self.cluster_manager.mark_node_error(node.node_id)
            self._handle_task_error(task, str(e))
    
    def _handle_task_result(self, task: 'DistributedTask', result: Dict[str, Any]):
        """
        Обрабатывает результат выполнения задачи.
        
        Args:
            task: Задача
            result: Результат выполнения
        """
        task.status = "completed"
        task.completed_at = time.time()
        task.result = result.get("result")
        
        # Обновляем статистику
        self._update_statistics(task)
        
        # Удаляем из активных задач
        if task.task_id in self.active_tasks:
            del self.active_tasks[task.task_id]
        
        logger.debug(f"Задача {task.task_id} завершена успешно")
        
        # Вызываем callback, если он указан
        if "callback" in task.metadata:
            try:
                callback = task.metadata["callback"]
                if callable(callback):
                    callback(task.result)
            except Exception as e:
                logger.error(f"Ошибка вызова callback для задачи {task.task_id}: {e}")
    
    def _handle_task_error(self, task: 'DistributedTask', error: str):
        """
        Обрабатывает ошибку выполнения задачи.
        
        Args:
            task: Задача
            error: Описание ошибки
        """
        task.error = error
        
        # Проверяем количество попыток
        retry_count = task.metadata.get("retry_count", 0)
        if retry_count < self.scheduler_settings["retry_count"]:
            # Повторяем задачу
            task.metadata["retry_count"] = retry_count + 1
            time.sleep(self.scheduler_settings["retry_delay"])
            self._requeue_task(task)
            logger.info(f"Задача {task.task_id} будет повторена ({retry_count + 1}/{self.scheduler_settings['retry_count']})")
        else:
            # Отмечаем как неудачную
            task.status = "failed"
            task.completed_at = time.time()
            
            # Обновляем статистику
            self._update_statistics(task)
            
            # Удаляем из активных задач
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            
            logger.error(f"Задача {task.task_id} завершена с ошибкой: {error}")
    
    def _requeue_task(self, task: 'DistributedTask'):
        """
        Возвращает задачу в очередь с учетом приоритета.
        
        Args:
            task: Задача для возврата в очередь
        """
        # Увеличиваем приоритет для повторных попыток
        retry_count = task.metadata.get("retry_count", 0)
        adjusted_priority = task.priority - (retry_count * 0.1)
        
        # Возвращаем в очередь
        self.task_queue.put((adjusted_priority, task))
    
    def _check_stuck_tasks(self):
        """Проверяет зависшие задачи и обрабатывает их."""
        current_time = time.time()
        timeout = self.scheduler_settings["task_timeout"]
        
        for task_id, task in list(self.active_tasks.items()):
            if current_time - task.started_at > timeout:
                logger.warning(f"Обнаружена зависшая задача {task_id}, перезапуск")
                self._handle_task_error(task, "Task timeout")
    
    def _update_statistics(self, task: 'DistributedTask'):
        """Обновляет статистику выполнения задач."""
        self.stats["total_tasks"] += 1
        
        if task.status == "completed":
            self.stats["completed_tasks"] += 1
        else:
            self.stats["failed_tasks"] += 1
        
        # Обновляем среднее время
        if task.completed_at and task.started_at:
            processing_time = task.completed_at - task.started_at
            self.stats["avg_processing_time"] = (
                (self.stats["avg_processing_time"] * (self.stats["completed_tasks"] - 1) + processing_time) / 
                self.stats["completed_tasks"] if self.stats["completed_tasks"] > 0 else processing_time
            )
        
        self.stats["last_task"] = time.time()
    
    def _check_system_health(self):
        """Проверяет здоровье системы планировщика."""
        health = self.get_system_health()
        
        # Логируем состояние системы
        logger.debug(
            f"Состояние планировщика задач: всего {health['stats']['total_tasks']}, "
            f"завершено {health['stats']['completed_tasks']}, "
            f"ошибок {health['stats']['failed_tasks']}, "
            f"здоровье: {health['health_score']:.2f}/1.0"
        )
    
    def submit_task(self, task_type: str, payload: Dict[str, Any], 
                   priority: float = 0.5, timeout: Optional[float] = None,
                   callback: Optional[Callable] = None) -> str:
        """
        Добавляет задачу в очередь на выполнение.
        
        Args:
            task_type: Тип задачи
            payload: Данные задачи
            priority: Приоритет задачи
            timeout: Таймаут выполнения (опционально)
            callback: Функция обратного вызова (опционально)
            
        Returns:
            str: ID задачи
        """
        # Генерируем уникальный ID
        task_id = f"task_{hashlib.md5(f'{task_type}_{time.time()}'.encode()).hexdigest()[:12]}"
        
        # Создаем задачу
        task = DistributedTask(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
            priority=priority,
            timeout=timeout or self.scheduler_settings["task_timeout"]
        )
        
        # Добавляем callback в метаданные
        if callback:
            task.metadata["callback"] = callback
        
        # Добавляем в очередь
        with self.lock:
            self.task_queue.put((priority, task))
            self.stats["total_tasks"] += 1
        
        logger.info(f"Задача добавлена в очередь: {task_id} ({task_type})")
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Возвращает статус задачи.
        
        Args:
            task_id: ID задачи
            
        Returns:
            Optional[Dict[str, Any]]: Статус задачи или None
        """
        # Проверяем активные задачи
        if task_id in self.active_tasks:
            return self.active_tasks[task_id].to_dict()
        
        # В реальной системе здесь будет проверка завершенных задач
        return None
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Возвращает отчет о здоровье планировщика задач.
        
        Returns:
            Dict: Отчет о здоровье
        """
        # Рассчитываем общий показатель здоровья
        health_score = 100.0
        
        # Учитываем успешность задач
        success_rate = (
            self.stats["completed_tasks"] / self.stats["total_tasks"] 
            if self.stats["total_tasks"] > 0 else 0.0
        )
        if success_rate < 0.7:
            health_score -= min(40, (0.7 - success_rate) * 100)
        elif success_rate < 0.85:
            health_score -= min(15, (0.85 - success_rate) * 50)
        
        # Учитываем время обработки
        if self.stats["avg_processing_time"] > 10.0:  # 10 секунд
            health_score -= min(20, (self.stats["avg_processing_time"] - 10.0) * 2)
        
        # Анализируем проблемы
        problem_areas = []
        if success_rate < 0.7:
            problem_areas.append("Низкий процент успешных задач")
        
        if self.stats["avg_processing_time"] > 15.0:
            problem_areas.append("Высокое время обработки задач")
        
        # Формируем рекомендации
        recommendations = []
        if success_rate < 0.7:
            recommendations.append(
                "Низкий процент успешных задач. Проверьте доступность узлов "
                "и корректность обработки задач."
            )
        
        if self.stats["avg_processing_time"] > 10.0:
            recommendations.append(
                f"Среднее время обработки {self.stats['avg_processing_time']:.2f} секунд превышает норму. "
                "Рассмотрите оптимизацию задач или увеличение мощности узлов."
            )
        
        if not recommendations:
            recommendations.append(
                "Планировщик задач работает стабильно. Продолжайте мониторинг для "
                "раннего выявления потенциальных проблем."
            )
        
        return {
            "health_score": max(0, min(100, health_score)),
            "stats": self.stats,
            "problem_areas": problem_areas,
            "recommendations": recommendations,
            "timestamp": time.time()
        }
    
    def get_scheduler_dashboard_data(self) -> Dict[str, Any]:
        """
        Возвращает данные для дашборда планировщика задач.
        
        Returns:
            Dict[str, Any]: Данные для дашборда
        """
        # Получаем данные за последние 24 часа
        end_time = time.time()
        start_time = end_time - (24 * 3600)
        
        # Формируем временные ряды
        trends = {
            "tasks_over_time": [],
            "success_rate_over_time": [],
            "avg_processing_time_over_time": []
        }
        
        # В реальной системе здесь будет сбор данных из истории
        # Для демонстрации используем заглушки
        
        # Генерируем фиктивные данные для демонстрации
        current_time = start_time
        while current_time <= end_time:
            date_str = datetime.fromtimestamp(current_time).strftime("%Y-%m-%d %H:%M")
            
            # Добавляем данные о количестве задач
            tasks_count = random.randint(5, 20)
            trends["tasks_over_time"].append({
                "date": date_str,
                "value": tasks_count
            })
            
            # Добавляем данные об успешности
            success_rate = max(0.5, min(1.0, random.gauss(0.85, 0.1)))
            trends["success_rate_over_time"].append({
                "date": date_str,
                "value": success_rate
            })
            
            # Добавляем данные о времени обработки
            processing_time = max(1.0, random.gauss(5.0, 2.0))
            trends["avg_processing_time_over_time"].append({
                "date": date_str,
                "value": processing_time
            })
            
            current_time += 3600  # Каждый час
        
        # Получаем текущий отчет о здоровье
        health_report = self.get_system_health()
        
        return {
            "health_report": health_report,
            "trends": trends,
            "stats": self.stats,
            "settings": self.scheduler_settings,
            "timestamp": time.time()
        }
    
    def generate_scheduler_visualization(self, view_type: str = "overview") -> str:
        """
        Генерирует визуализацию данных планировщика задач.
        
        Args:
            view_type: Тип визуализации
            
        Returns:
            str: Изображение в формате base64
        """
        try:
            # Получаем данные для визуализации
            dashboard_data = self.get_scheduler_dashboard_data()
            
            # Создаем изображение в зависимости от типа
            if view_type == "overview":
                return self._generate_overview_visualization(dashboard_data)
            elif view_type == "success":
                return self._generate_success_visualization(dashboard_data)
            elif view_type == "performance":
                return self._generate_performance_visualization(dashboard_data)
            
            return self._generate_overview_visualization(dashboard_data)
            
        except Exception as e:
            logger.error(f"Ошибка генерации визуализации планировщика: {e}")
            return ""
    
    def _generate_overview_visualization(self, data: Dict[str, Any]) -> str:
        """Генерирует визуализацию обзора планировщика задач."""
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import base64
        from io import BytesIO
        
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Создаем круговую диаграмму статусов задач
        labels = ['Завершено', 'Неудачно']
        sizes = [
            data["stats"]["completed_tasks"],
            data["stats"]["failed_tasks"]
        ]
        colors = ['green', 'red']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        ax.set_title('Результаты выполнения задач')
        
        # Сохраняем в буфер
        buf = BytesIO()
        fig.tight_layout()
        canvas = FigureCanvasAgg(fig)
        canvas.print_png(buf)
        
        # Преобразуем в base64
        buf.seek(0)
        img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"image/png;base64,{img_data}"
    
    def _generate_success_visualization(self, data: Dict[str, Any]) -> str:
        """Генерирует визуализацию успешности выполнения задач."""
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import base64
        from io import BytesIO
        
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Получаем данные об успешности
        success_rates = [entry["value"] for entry in data["trends"]["success_rate_over_time"]]
        dates = [entry["date"] for entry in data["trends"]["success_rate_over_time"]]
        
        if success_rates:
            ax.plot(dates, success_rates, marker='o', color='blue')
            ax.set_ylim(0, 1)
            ax.set_ylabel('Процент успешных задач')
            ax.set_title('Успешность выполнения задач')
            plt.xticks(rotation=45)
        
        # Сохраняем в буфер
        buf = BytesIO()
        fig.tight_layout()
        canvas = FigureCanvasAgg(fig)
        canvas.print_png(buf)
        
        # Преобразуем в base64
        buf.seek(0)
        img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"image/png;base64,{img_data}"
    
    def _generate_performance_visualization(self, data: Dict[str, Any]) -> str:
        """Генерирует визуализацию производительности планировщика."""
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import base64
        from io import BytesIO
        import numpy as np
        
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Получаем данные о времени обработки
        processing_times = [entry["value"] for entry in data["trends"]["avg_processing_time_over_time"]]
        dates = [entry["date"] for entry in data["trends"]["avg_processing_time_over_time"]]
        
        if processing_times:
            ax.plot(dates, processing_times, marker='o', color='purple')
            ax.set_ylabel('Среднее время (сек)')
            ax.set_title('Производительность планировщика задач')
            plt.xticks(rotation=45)
        
        # Сохраняем в буфер
        buf = BytesIO()
        fig.tight_layout()
        canvas = FigureCanvasAgg(fig)
        canvas.print_png(buf)
        
        # Преобразуем в base64
        buf.seek(0)
        img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"image/png;base64,{img_data}"
    
    def export_scheduler_data(self, file_path: str) -> bool:
        """
        Экспортирует данные планировщика в файл.
        
        Args:
            file_path: Путь к файлу для экспорта
            
        Returns:
            bool: Успешно ли экспортировано
        """
        try:
            # Собираем данные для экспорта
            export_data = {
                "metadata": {
                    "export_time": time.time(),
                    "format_version": "1.0"
                },
                "scheduler_stats": self.stats,
                "scheduler_settings": self.scheduler_settings,
                "system_health": self.get_system_health(),
                "dashboard_data": self.get_scheduler_dashboard_data()
            }
            
            # Сохраняем в JSON
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Данные планировщика экспортированы в {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка экспорта данных планировщика: {e}")
            return False
    
    def import_scheduler_data(self, file_path: str) -> bool:
        """
        Импортирует данные планировщика из файла.
        
        Args:
            file_path: Путь к файлу для импорта
            
        Returns:
            bool: Успешно ли импортировано
        """
        try:
            # Загружаем данные из JSON
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Импортируем статистику
            if "scheduler_stats" in data:
                self.stats = data["scheduler_stats"]
            
            # Импортируем настройки
            if "scheduler_settings" in data:
                self.scheduler_settings = data["scheduler_settings"]
            
            logger.info(f"Данные планировщика импортированы из {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка импорта данных планировщика: {e}")
            return False
    
    def get_system_summary(self) -> str:
        """
        Возвращает краткую сводку о системе планировщика.
        
        Returns:
            str: Сводка о состоянии
        """
        health = self.get_system_health()
        
        summary = (
            f"Планировщик задач\n"
            f"{'=' * 30}\n\n"
            f"Задачи: всего {health['stats']['total_tasks']}, "
            f"завершено {health['stats']['completed_tasks']}, "
            f"ошибок {health['stats']['failed_tasks']}\n"
            f"Среднее время: {health['stats']['avg_processing_time']:.2f} сек\n"
            f"Здоровье: {health['health_score']:.2f}/100\n"
        )
        
        if health["health_score"] < 70:
            summary += "\nРекомендации:\n"
            for i, rec in enumerate(health["recommendations"][:2], 1):
                summary += f"{i}. {rec}\n"
        
        return summary