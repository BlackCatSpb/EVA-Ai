"""
Async Generation Pipeline - Асинхронная архитектура генерации

Обеспечивает:
- Неблокирующую обработку запросов
- Очередь задач с приоритетами
- Пул воркеров для параллельной генерации
- Streaming responses через очередь
- Graceful shutdown
"""

import asyncio
import time
import logging
import uuid
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger("eva_ai.async_pipeline")


class TaskPriority(Enum):
    """Приоритеты задач."""
    HIGH = 1      # Пользовательские запросы
    NORMAL = 2    # Стандартные запросы
    LOW = 3       # Фоновые задачи
    BACKGROUND = 4  # Обучение, индексация


class TaskStatus(Enum):
    """Статусы задач."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class GenerationTask:
    """Задача на генерацию."""
    id: str
    query: str
    context: str
    mode: str
    priority: TaskPriority
    stream: bool
    created_at: float
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    tokens_generated: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь."""
        return {
            'id': self.id,
            'query': self.query[:100],
            'mode': self.mode,
            'priority': self.priority.name,
            'status': self.status.value,
            'stream': self.stream,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'duration_ms': self.get_duration_ms(),
            'tokens_generated': self.tokens_generated
        }
    
    def get_duration_ms(self) -> Optional[float]:
        """Получить длительность выполнения в мс."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at) * 1000
        return None


class AsyncGenerationPipeline:
    """
    Асинхронный пайплайн для генерации с очередью и воркерами.
    
    Usage:
        pipeline = AsyncGenerationPipeline(dual_generator, max_workers=4)
        await pipeline.start()
        
        # Синхронный запрос
        task = await pipeline.submit("Hello", mode="extended")
        result = await pipeline.wait_for_result(task.id, timeout=30)
        
        # Streaming запрос
        async for chunk in pipeline.submit_streaming("Hello", mode="extended"):
            print(chunk['text'])
        
        await pipeline.stop()
    """
    
    def __init__(
        self,
        dual_generator,
        max_workers: int = None,  # По умолчанию: все ядра CPU
        max_queue_size: int = 100,
        default_timeout: float = None  # Без таймаута
    ):
        import os
        self.dual_generator = dual_generator
        self.max_workers = max_workers or os.cpu_count() or 4
        self.max_queue_size = max_queue_size
        self.default_timeout = default_timeout  # None = бесконечно
        
        # Очередь задач (приоритетная)
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_queue_size)
        
        # Хранилище задач
        self._tasks: Dict[str, GenerationTask] = {}
        self._tasks_lock = asyncio.Lock()
        
        # Пул потоков для блокирующих операций
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="gen_worker")
        
        # Состояние
        self._running = False
        self._worker_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        
        # Метрики
        self._metrics = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'tasks_cancelled': 0,
            'total_tokens': 0,
            'total_duration_ms': 0
        }
    
    async def start(self):
        """Запустить пайплайн и воркеры."""
        if self._running:
            return
        
        self._running = True
        self._shutdown_event.clear()
        
        # Запускаем воркеры
        for i in range(self.max_workers):
            worker = asyncio.create_task(
                self._worker_loop(f"worker-{i}"),
                name=f"gen-worker-{i}"
            )
            self._worker_tasks.append(worker)
        
        logger.info(f"AsyncGenerationPipeline запущен с {self.max_workers} воркерами")
    
    async def stop(self, timeout: float = 30.0):
        """Остановить пайплайн graceful."""
        if not self._running:
            return
        
        logger.info("Остановка AsyncGenerationPipeline...")
        self._running = False
        
        # Сигнализируем о завершении
        self._shutdown_event.set()
        
        # Ждем завершения воркеров
        if self._worker_tasks:
            done, pending = await asyncio.wait(
                self._worker_tasks,
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED
            )
            
            for task in pending:
                task.cancel()
            
            if pending:
                logger.warning(f"{len(pending)} воркеров не завершились вовремя")
        
        # Очищаем очередь
        while not self._queue.empty():
            try:
                _, task = self._queue.get_nowait()
                task.status = TaskStatus.CANCELLED
                self._metrics['tasks_cancelled'] += 1
            except asyncio.QueueEmpty:
                break
        
        # Закрываем executor
        self._executor.shutdown(wait=True)
        
        logger.info("AsyncGenerationPipeline остановлен")
    
    async def submit(
        self,
        query: str,
        context: str = "",
        mode: str = "extended",
        priority: TaskPriority = TaskPriority.NORMAL,
        stream: bool = False
    ) -> GenerationTask:
        """
        Отправить задачу на генерацию.
        
        Args:
            query: Текст запроса
            context: Контекст
            mode: Режим генерации
            priority: Приоритет
            stream: Нужен ли streaming
            
        Returns:
            GenerationTask
        """
        task = GenerationTask(
            id=str(uuid.uuid4()),
            query=query,
            context=context,
            mode=mode,
            priority=priority,
            stream=stream,
            created_at=time.time()
        )
        
        async with self._tasks_lock:
            self._tasks[task.id] = task
        
        # Добавляем в очередь (priority queue: меньше число = выше приоритет)
        # Используем tuple (priority, timestamp, task) для корректного сравнения
        try:
            await asyncio.wait_for(
                self._queue.put((priority.value, task.created_at, task)),
                timeout=5.0
            )
            self._metrics['tasks_submitted'] += 1
            logger.debug(f"Задача {task.id} добавлена в очередь (priority={priority.name})")
        except asyncio.TimeoutError:
            task.status = TaskStatus.FAILED
            task.error = "Queue is full"
            logger.error(f"Очередь переполнена, задача {task.id} отклонена")
        
        return task
    
    async def submit_streaming(
        self,
        query: str,
        context: str = "",
        mode: str = "extended",
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Отправить задачу и получить streaming ответ.
        
        Yields:
            Чанки с токенами
        """
        task = await self.submit(query, context, mode, priority, stream=True)
        
        # Создаем очередь для токенов
        token_queue: asyncio.Queue = asyncio.Queue()
        
        # Регистрируем callback
        async def token_callback(chunk: Dict[str, Any]):
            await token_queue.put(chunk)
        
        # Сохраняем callback в задаче
        task._token_callback = token_callback
        
        # Ждем результатов
        while True:
            try:
                chunk = await asyncio.wait_for(token_queue.get(), timeout=self.default_timeout)
                yield chunk
                
                if chunk.get('type') in ('complete', 'error'):
                    break
            except asyncio.TimeoutError:
                yield {'type': 'error', 'text': 'Timeout', 'elapsed_ms': self.default_timeout * 1000}
                break
    
    async def wait_for_result(
        self,
        task_id: str,
        timeout: Optional[float] = None,
        poll_interval: float = 0.1
    ) -> Optional[GenerationTask]:
        """
        Ожидать результат задачи.
        
        Args:
            task_id: ID задачи
            timeout: Таймаут ожидания
            poll_interval: Интервал проверки
            
        Returns:
            GenerationTask или None если таймаут
        """
        timeout = timeout or self.default_timeout
        start = time.time()
        
        while time.time() - start < timeout:
            async with self._tasks_lock:
                task = self._tasks.get(task_id)
            
            if task and task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                return task
            
            await asyncio.sleep(poll_interval)
        
        return None
    
    def get_task(self, task_id: str) -> Optional[GenerationTask]:
        """Получить задачу по ID."""
        return self._tasks.get(task_id)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Получить метрики пайплайна."""
        return {
            **self._metrics,
            'queue_size': self._queue.qsize(),
            'active_tasks': sum(1 for t in self._tasks.values() if t.status == TaskStatus.RUNNING),
            'pending_tasks': sum(1 for t in self._tasks.values() if t.status == TaskStatus.PENDING)
        }
    
    async def _worker_loop(self, worker_name: str):
        """Цикл воркера."""
        logger.debug(f"Воркер {worker_name} запущен")
        
        while self._running:
            try:
                # Ждем задачу из очереди
                priority, timestamp, task = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0
                )

                if not self._running:
                    break

                # Выполняем задачу
                await self._execute_task(task)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Ошибка в воркере {worker_name}: {e}")
        
        logger.debug(f"Воркер {worker_name} остановлен")
    
    async def _execute_task(self, task: GenerationTask):
        """Выполнить задачу генерации."""
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        
        try:
            # Выполняем генерацию в пуле потоков
            loop = asyncio.get_event_loop()
            
            if task.stream:
                # Streaming генерация
                result_text = ""
                async for chunk in self._generate_streaming(task):
                    result_text += chunk.get('text', '')
                    
                    # Вызываем callback если есть
                    if hasattr(task, '_token_callback'):
                        await task._token_callback(chunk)
                    
                    if chunk.get('type') == 'complete':
                        task.tokens_generated = chunk.get('total_tokens', 0)
                
                task.result = result_text
                task.status = TaskStatus.COMPLETED
                self._metrics['tasks_completed'] += 1
                
            else:
                # Обычная генерация
                result = await loop.run_in_executor(
                    self._executor,
                    self._generate_sync,
                    task
                )
                
                task.result = result.get('response', '')
                task.tokens_generated = result.get('tokens_estimate', 0)
                task.status = TaskStatus.COMPLETED
                self._metrics['tasks_completed'] += 1
            
            task.completed_at = time.time()
            
            # Обновляем метрики
            duration_ms = task.get_duration_ms() or 0
            self._metrics['total_duration_ms'] += duration_ms
            self._metrics['total_tokens'] += task.tokens_generated
            
            logger.debug(f"Задача {task.id} завершена за {duration_ms:.1f}ms")
            
        except Exception as e:
            logger.error(f"Ошибка выполнения задачи {task.id}: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()
            self._metrics['tasks_failed'] += 1
    
    def _generate_sync(self, task: GenerationTask) -> Dict[str, Any]:
        """Синхронная генерация (вызывается в пуле потоков)."""
        return self.dual_generator.generate(
            query=task.query,
            mode=task.mode,
            return_details=True
        )
    
    async def _generate_streaming(self, task: GenerationTask):
        """Streaming генерация."""
        loop = asyncio.get_event_loop()
        
        # Генератор в пуле потоков
        def generate():
            return list(self.dual_generator.generate_streaming(
                query=task.query,
                mode=task.mode,
                context=task.context
            ))
        
        chunks = await loop.run_in_executor(self._executor, generate)
        
        for chunk in chunks:
            yield chunk


# Глобальный экземпляр (singleton)
_pipeline_instance: Optional[AsyncGenerationPipeline] = None


def get_pipeline(dual_generator=None, max_workers=4) -> AsyncGenerationPipeline:
    """Получить глобальный экземпляр пайплайна."""
    global _pipeline_instance
    
    if _pipeline_instance is None and dual_generator is not None:
        _pipeline_instance = AsyncGenerationPipeline(dual_generator, max_workers)
    
    return _pipeline_instance
