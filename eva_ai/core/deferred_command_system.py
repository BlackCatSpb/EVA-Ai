"""
Система отложенных команд для ЕВА.
Обеспечивает восстановление модулей при сбоях и повторную активацию по запросу.
"""

import time
import logging
import threading
import queue
from typing import Dict, Any, Callable, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger("eva_ai.deferred_commands")

# Глобальный EventBus для публикации событий команд
_global_event_bus = None

def set_event_bus(event_bus):
    """Устанавливает глобальный EventBus для DeferredCommandSystem."""
    global _global_event_bus
    _global_event_bus = event_bus

def get_event_bus():
    """Получает глобальный EventBus."""
    return _global_event_bus

class CommandPriority(Enum):
    """Приоритеты команд."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    # Совместимость: некоторые джобы используют MEDIUM как синоним NORMAL
    MEDIUM = NORMAL
    LOW = 3

class CommandStatus(Enum):
    """Статусы команд."""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

@dataclass
class DeferredCommand:
    """Отложенная команда."""
    id: str
    command: Callable
    args: Tuple
    kwargs: Dict[str, Any]
    priority: CommandPriority
    max_retries: int
    retry_delay: float
    timeout: Optional[float]
    created_at: float
    status: CommandStatus = CommandStatus.PENDING
    attempts: int = 0
    last_error: Optional[str] = None
    result: Any = None

class DeferredCommandSystem:
    """Система отложенных команд с поддержкой приоритетов и восстановления."""
    
    def __init__(self, brain, max_workers: int = 4):
        self.brain = brain
        self.max_workers = max_workers
        
        # Очереди команд по приоритетам
        self.command_queues = {
            CommandPriority.CRITICAL: queue.PriorityQueue(),
            CommandPriority.HIGH: queue.PriorityQueue(), 
            CommandPriority.NORMAL: queue.PriorityQueue(),
            CommandPriority.LOW: queue.PriorityQueue()
        }
        
        # Реестр команд
        self.commands: Dict[str, DeferredCommand] = {}
        self.commands_lock = threading.RLock()
        
        # Пул потоков для выполнения команд
        try:
            self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="DeferredCmd")
        except (OSError, RuntimeError) as e:
            logger.error(f"Failed to create ThreadPoolExecutor: {e}")
            self.executor = None
        
        # Система мониторинга и восстановления
        self.monitoring_enabled = True
        self.recovery_strategies: Dict[str, Callable] = {}
        self.module_health_checks: Dict[str, Callable] = {}
        
        # Статистика
        self.stats = {
            "total_commands": 0,
            "completed_commands": 0,
            "failed_commands": 0,
            "retried_commands": 0,
            "avg_execution_time": 0.0
        }
        self.stats_lock = threading.RLock()
        
        # Флаг корректного завершения для предотвращения планирования задач после shutdown
        self._shutting_down = False
        
        # Запускаем обработчик команд
        self.running = True
        self.processor_thread = threading.Thread(target=self._process_commands, daemon=True)
        self.processor_thread.start()
        
        # Запускаем мониторинг модулей
        self.monitor_thread = threading.Thread(target=self._monitor_modules, daemon=True)
        self.monitor_thread.start()
        
        # Реестр коллбеков для сброса нагрузки (load shedding)
        self._ls_callbacks: List[Dict[str, Any]] = []
        self._ls_lock = threading.RLock()
        self._ls_running = True
        self.load_monitor_thread = threading.Thread(target=self._monitor_load_shedding, daemon=True)
        self.load_monitor_thread.start()
        
        logger.info(f"Система отложенных команд инициализирована с {max_workers} воркерами")
    
    def add_command(self, 
                   command: Callable,
                   args: Tuple = (),
                   kwargs: Dict[str, Any] = None,
                   priority: CommandPriority = CommandPriority.NORMAL,
                   max_retries: int = 3,
                   retry_delay: float = 1.0,
                   timeout: Optional[float] = None,
                   command_id: Optional[str] = None) -> str:
        """Добавляет команду в очередь выполнения с подробным логированием."""
        
        if kwargs is None:
            kwargs = {}
            
        if command_id is None:
            command_id = "cmd_{}_{}".format(int(time.time() * 1000), id(command))
        
        command_name = getattr(command, '__name__', str(command))
        
        logger.info("=== DEFERRED COMMAND: Adding command ===")
        logger.info("  Command ID: {}".format(command_id))
        logger.info("  Command name: {}".format(command_name))
        logger.info("  Priority: {} ({})".format(priority.name, priority.value))
        logger.info("  Args: {}".format(args))
        logger.info("  Kwargs: {}".format(kwargs))
        logger.info("  Max retries: {}".format(max_retries))
        
        deferred_cmd = DeferredCommand(
            id=command_id,
            command=command,
            args=args,
            kwargs=kwargs,
            priority=priority,
            max_retries=max_retries,
            retry_delay=retry_delay,
            timeout=timeout,
            created_at=time.time()
        )
        
        with self.commands_lock:
            self.commands[command_id] = deferred_cmd
            
        # Добавляем в соответствующую очередь (приоритет как отрицательное число для правильной сортировки)
        self.command_queues[priority].put((-priority.value, time.time(), command_id))
        
        with self.stats_lock:
            self.stats["total_commands"] += 1
        
        logger.info("DEFERRED COMMAND added: {} (priority: {})".format(command_id, priority.name))
        logger.debug("  Total commands in queue: {}".format(self.command_queues[priority].qsize()))
        
        return command_id
    
    def add_module_recovery_strategy(self, module_name: str, recovery_func: Callable):
        """Добавляет стратегию восстановления для модуля."""
        self.recovery_strategies[module_name] = recovery_func
        logger.info(f"Добавлена стратегия восстановления для модуля {module_name}")
    
    def add_module_health_check(self, module_name: str, health_check_func: Callable):
        """Добавляет проверку здоровья модуля."""
        self.module_health_checks[module_name] = health_check_func
        logger.info(f"Добавлена проверка здоровья для модуля {module_name}")
    
    def _process_commands(self):
        """Основной цикл обработки команд."""
        while self.running:
            try:
                # Проверяем очереди в порядке приоритета
                for priority in [CommandPriority.CRITICAL, CommandPriority.HIGH, 
                               CommandPriority.NORMAL, CommandPriority.LOW]:
                    
                    try:
                        # Неблокирующее получение команды
                        _, _, command_id = self.command_queues[priority].get_nowait()
                        
                        with self.commands_lock:
                            if command_id in self.commands:
                                cmd = self.commands[command_id]
                                if cmd.status == CommandStatus.PENDING:
                                    # Отправляем команду на выполнение (с защитой от shutdown())
                                    try:
                                        if not self._shutting_down and self.running and self.executor is not None:
                                            self.executor.submit(self._execute_command, cmd)
                                        else:
                                            # Система завершает работу — помечаем как неуспешную без запуска
                                            cmd.status = CommandStatus.FAILED
                                            cmd.last_error = "Executor is shutting down"
                                    except RuntimeError as re:
                                        # Пул уже закрыт: не планируем новую future, фиксируем состояние
                                        if "cannot schedule new futures" in str(re):
                                            with self.commands_lock:
                                                cmd.status = CommandStatus.FAILED
                                                cmd.last_error = str(re)
                                        else:
                                            raise
                                    except Exception as e:
                                        logger.error(f"Error submitting command to executor: {e}")
                                        with self.commands_lock:
                                            cmd.status = CommandStatus.FAILED
                                            cmd.last_error = str(e)
                        
                    except queue.Empty:
                        continue
                        
                # Небольшая пауза между циклами
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Ошибка в процессоре команд: {e}", exc_info=True)
                time.sleep(1.0)
    
    def _execute_command(self, cmd: DeferredCommand):
        """Выполняет отложенную команду с подробным логированием."""
        start_time = time.time()
        
        command_name = getattr(cmd.command, '__name__', str(cmd.command))
        
        logger.info("=== DEFERRED COMMAND: Executing ===")
        logger.info("  Command ID: {}".format(cmd.id))
        logger.info("  Command: {}".format(command_name))
        logger.info("  Attempt: {} of {}".format(cmd.attempts + 1, cmd.max_retries))
        logger.info("  Args: {}".format(cmd.args))
        logger.info("  Kwargs: {}".format(cmd.kwargs))
        
        with self.commands_lock:
            cmd.status = CommandStatus.RUNNING
            cmd.attempts += 1
        
        try:
            logger.info("Executing command {} (attempt {})".format(cmd.id, cmd.attempts))
            
            # Выполняем команду с таймаутом если указан
            if cmd.timeout:
                try:
                    if self.executor is None:
                        raise RuntimeError("Executor is not available")
                    future = self.executor.submit(cmd.command, *cmd.args, **cmd.kwargs)
                    result = future.result(timeout=cmd.timeout)
                except RuntimeError as re:
                    if "cannot schedule new futures" in str(re) or self._shutting_down:
                        # Executor закрыт — считаем команду неуспешной без повторов
                        with self.commands_lock:
                            cmd.status = CommandStatus.FAILED
                            cmd.last_error = str(re)
                        self._update_stats(time.time() - start_time, success=False)
                        logger.debug(f"Команда {cmd.id} не запущена: executor shutdown")
                        return
                    else:
                        raise
                except Exception as e:
                    with self.commands_lock:
                        cmd.status = CommandStatus.FAILED
                        cmd.last_error = str(e)
                    self._update_stats(time.time() - start_time, success=False)
                    logger.debug(f"Команда {cmd.id} не выполнена: {e}")
                    return
            else:
                # Выполняем синхронно в текущем потоке
                result = cmd.command(*cmd.args, **cmd.kwargs)
            
            # Команда выполнена успешно
            with self.commands_lock:
                cmd.status = CommandStatus.COMPLETED
                cmd.result = result
                
            execution_time = time.time() - start_time
            self._update_stats(execution_time, success=True)
            
            self._publish_command_event("command.completed", cmd)
            
            logger.debug(f"Команда {cmd.id} выполнена успешно за {execution_time:.3f}с")
            
        except Exception as e:
            error_msg = str(e)
            execution_time = time.time() - start_time
            
            with self.commands_lock:
                cmd.last_error = error_msg
                
                # Проверяем, нужно ли повторить команду
                if cmd.attempts < cmd.max_retries and not self._shutting_down:
                    cmd.status = CommandStatus.RETRYING
                    logger.warning(f"Команда {cmd.id} неудачна (попытка {cmd.attempts}/{cmd.max_retries}): {error_msg}")
                    
                    # Планируем повторное выполнение
                    retry_thread = threading.Thread(
                        target=self._schedule_retry,
                        args=(cmd,),
                        daemon=True
                    )
                    retry_thread.start()
                    
                    with self.stats_lock:
                        self.stats["retried_commands"] += 1
                else:
                    cmd.status = CommandStatus.FAILED
                    logger.error(f"Команда {cmd.id} окончательно неудачна после {cmd.attempts} попыток: {error_msg}")
                    self._update_stats(execution_time, success=False)
                    self._publish_command_event("command.failed", cmd)
    
    def _schedule_retry(self, cmd: DeferredCommand):
        """Планирует повторное выполнение команды."""
        time.sleep(cmd.retry_delay)
        
        # Во время завершения — больше не перекидываем задачи в очередь
        if self._shutting_down or not self.running:
            with self.commands_lock:
                cmd.status = CommandStatus.FAILED
                cmd.last_error = cmd.last_error or "Retry skipped due to shutdown"
            return
        
        with self.commands_lock:
            if cmd.status == CommandStatus.RETRYING:
                cmd.status = CommandStatus.PENDING
                # Возвращаем в очередь
                self.command_queues[cmd.priority].put((-cmd.priority.value, time.time(), cmd.id))
    
    def _monitor_modules(self):
        """Мониторинг здоровья модулей и автоматическое восстановление."""
        while self.running and self.monitoring_enabled:
            try:
                for module_name, health_check in self.module_health_checks.items():
                    try:
                        if not health_check():
                            logger.warning(f"Модуль {module_name} неисправен, запускаем восстановление")
                            self._recover_module(module_name)
                    except Exception as e:
                        logger.error(f"Ошибка проверки здоровья модуля {module_name}: {e}")
                
                # Проверяем каждые 30 секунд
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Ошибка в мониторинге модулей: {e}", exc_info=True)
                time.sleep(60)
    
    def _recover_module(self, module_name: str):
        """Восстанавливает неисправный модуль."""
        if module_name in self.recovery_strategies:
            recovery_func = self.recovery_strategies[module_name]
            
            # Добавляем команду восстановления с высоким приоритетом
            self.add_command(
                command=recovery_func,
                priority=CommandPriority.HIGH,
                max_retries=2,
                retry_delay=5.0,
                command_id=f"recovery_{module_name}_{int(time.time())}"
            )
            
            logger.info(f"Запущено восстановление модуля {module_name}")
        else:
            logger.warning(f"Нет стратегии восстановления для модуля {module_name}")
    
    def _update_stats(self, execution_time: float, success: bool):
        """Обновляет статистику выполнения."""
        with self.stats_lock:
            if success:
                self.stats["completed_commands"] += 1
            else:
                self.stats["failed_commands"] += 1
            
            # Обновляем среднее время выполнения
            total_completed = self.stats["completed_commands"]
            if total_completed > 0:
                current_avg = self.stats["avg_execution_time"]
                self.stats["avg_execution_time"] = (current_avg * (total_completed - 1) + execution_time) / total_completed
    
    def _publish_command_event(self, event_type: str, cmd: DeferredCommand):
        """Публикует событие при выполнении команды."""
        global _global_event_bus
        if _global_event_bus is None:
            return
        
        try:
            from .event_bus import Event, EventPriority
            event = Event(
                event_type=event_type,
                source="deferred_command_system",
                data={
                    "command_id": cmd.id,
                    "status": cmd.status.value,
                    "result": cmd.result,
                    "error": cmd.last_error,
                    "attempts": cmd.attempts,
                    "execution_time": time.time() - cmd.created_at
                },
                priority=EventPriority.NORMAL
            )
            _global_event_bus.publish(event)
        except Exception as e:
            logger.debug(f"Не удалось опубликовать событие команды: {e}")
    
    def create_bridge(self, event_bus, resource_manager):
        """Создает мост EventBus ↔ DeferredCommandSystem с load shedding."""
        global _global_event_bus
        _global_event_bus = event_bus
        
        if not resource_manager:
            logger.warning("ResourceManager не предоставлен, load shedding не будет работать")
            return
        
        # Register CPU load shedding callback (>80%)
        def check_cpu_high():
            try:
                cpu_usage = resource_manager.get_cpu_usage()
                return cpu_usage > 0.8
            except Exception:
                return False
        
        def drop_low_priority_commands():
            dropped = 0
            try:
                low_queue = self.command_queues.get(CommandPriority.LOW)
                if low_queue:
                    while not low_queue.empty():
                        try:
                            low_queue.get_nowait()
                            dropped += 1
                        except queue.Empty:
                            break
                    if dropped > 0:
                        logger.warning(f"Load shedding: сброшено {dropped} LOW приоритет команд")
            except Exception as e:
                logger.error(f"Ошибка при сбросе команд: {e}")
            return dropped
        
        self.register_load_shed_callback(
            condition=check_cpu_high,
            action=drop_low_priority_commands,
            name="cpu_high_load_shedding",
            cooldown_sec=30.0,
            priority=CommandPriority.HIGH
        )
        
        # Register queue overflow load shedding callback (>100 commands)
        def check_queue_overflow():
            try:
                total = sum(q.qsize() for q in self.command_queues.values())
                return total > 100
            except Exception:
                return False
        
        def drop_more_low_priority():
            dropped = 0
            try:
                low_queue = self.command_queues.get(CommandPriority.LOW)
                if low_queue:
                    count = 0
                    temp_items = []
                    while not low_queue.empty() and count < 20:
                        try:
                            item = low_queue.get_nowait()
                            temp_items.append(item)
                            count += 1
                        except queue.Empty:
                            break
                    # Return half back
                    for i, item in enumerate(temp_items):
                        if i % 2 == 0:
                            dropped += 1
                        else:
                            low_queue.put(item)
                    if dropped > 0:
                        logger.warning(f"Queue overflow: сброшено {dropped} LOW приоритет команд")
            except Exception as e:
                logger.error(f"Ошибка при сбросе команд: {e}")
            return dropped
        
        self.register_load_shed_callback(
            condition=check_queue_overflow,
            action=drop_more_low_priority,
            name="queue_overflow_shedding",
            cooldown_sec=15.0,
            priority=CommandPriority.HIGH
        )
        
        logger.info("Мост EventBus ↔ DeferredCommandSystem создан с load shedding")
    
    def get_command_status(self, command_id: str) -> Optional[CommandStatus]:
        """Возвращает статус команды."""
        with self.commands_lock:
            if command_id in self.commands:
                return self.commands[command_id].status
        return None
    
    def get_command_result(self, command_id: str) -> Any:
        """Возвращает результат выполнения команды."""
        with self.commands_lock:
            if command_id in self.commands:
                cmd = self.commands[command_id]
                if cmd.status == CommandStatus.COMPLETED:
                    return cmd.result
                elif cmd.status == CommandStatus.FAILED:
                    raise RuntimeError(f"Команда {command_id} неудачна: {cmd.last_error}")
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику системы."""
        with self.stats_lock:
            stats = self.stats.copy()
        
        # Добавляем информацию об очередях
        queue_stats = {}
        for priority, q in self.command_queues.items():
            queue_stats[priority.name] = q.qsize()
        
        stats["queue_sizes"] = queue_stats
        stats["active_commands"] = len([cmd for cmd in self.commands.values() 
                                      if cmd.status == CommandStatus.RUNNING])
        
        return stats
    
    def shutdown(self):
        """Завершает работу системы отложенных команд."""
        logger.info("Завершение работы системы отложенных команд...")
        
        # Сигнализируем всем потокам/планировщикам о завершении
        self._shutting_down = True
        self.running = False
        self.monitoring_enabled = False
        self._ls_running = False
        
        # Ждем завершения потоков
        if self.processor_thread.is_alive():
            self.processor_thread.join(timeout=5.0)
        
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        try:
            if self.load_monitor_thread.is_alive():
                self.load_monitor_thread.join(timeout=5.0)
        except Exception:
            pass
        
        # Завершаем пул потоков
        self.executor.shutdown(wait=True, timeout=10.0)
        
        logger.info("Система отложенных команд завершена")

    # ----------------------------
    # Load Shedding API
    # ----------------------------
    def register_load_shed_callback(
        self,
        condition: Callable[[], bool],
        action: Callable[[], None],
        *,
        name: Optional[str] = None,
        cooldown_sec: float = 10.0,
        priority: CommandPriority = CommandPriority.HIGH,
    ) -> str:
        """Регистрирует коллбек сброса нагрузки.
        condition: функция без аргументов, возвращает True, когда нужно выполнить действие
        action: функция без аргументов, выполняет снижение нагрузки (например, уменьшение batch_size)
        cooldown_sec: минимальный интервал между срабатываниями одного и того же коллбека
        priority: приоритет команды действия
        Возвращает идентификатор зарегистрированного коллбека
        """
        cb_id = name or f"ls_{int(time.time() * 1000)}_{id(action)}"
        entry = {
            "id": cb_id,
            "name": cb_id,
            "condition": condition,
            "action": action,
            "cooldown": max(0.0, float(cooldown_sec)),
            "priority": priority,
            "last_fired": 0.0,
        }
        with self._ls_lock:
            self._ls_callbacks.append(entry)
        logger.info(f"Зарегистрирован load-shedding коллбек: {cb_id} (cooldown={cooldown_sec}s, prio={priority.name})")
        return cb_id

    def trigger_load_shedding(self, name_or_id: str) -> bool:
        """Принудительно запускает действие для указанного коллбека по имени/ID (игнорируя condition)."""
        with self._ls_lock:
            for cb in self._ls_callbacks:
                if cb.get("id") == name_or_id or cb.get("name") == name_or_id:
                    try:
                        self.add_command(cb["action"], priority=cb.get("priority", CommandPriority.HIGH))
                        cb["last_fired"] = time.time()
                        logger.info(f"Принудительно запущен load-shedding: {name_or_id}")
                        return True
                    except Exception as e:
                        logger.error(f"Ошибка запуска load-shedding {name_or_id}: {e}")
                        return False
        return False

    def _monitor_load_shedding(self):
        """Фоновой мониторинг условий сброса нагрузки и планирование действий."""
        while self._ls_running:
            try:
                now = time.time()
                with self._ls_lock:
                    callbacks = list(self._ls_callbacks)
                for cb in callbacks:
                    try:
                        cond: Callable[[], bool] = cb.get("condition")  # type: ignore
                        if not callable(cond):
                            continue
                        if cond():
                            # Проверяем cooldown
                            if now - float(cb.get("last_fired", 0.0)) >= float(cb.get("cooldown", 0.0)):
                                try:
                                    self.add_command(cb["action"], priority=cb.get("priority", CommandPriority.HIGH))
                                    cb["last_fired"] = now
                                    logger.debug(f"Load-shedding '{cb.get('name')}' запланирован")
                                except Exception as e:
                                    logger.error(f"Не удалось запланировать load-shedding '{cb.get('name')}': {e}")
                    except Exception as e:
                        logger.debug(f"Ошибка проверки условия load-shedding: {e}")
                time.sleep(2.0)
            except Exception as e:
                logger.error(f"Ошибка в мониторе load-shedding: {e}", exc_info=True)
                time.sleep(5.0)
