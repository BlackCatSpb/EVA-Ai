import logging
from typing import Dict, Any, Callable, List, Optional
import time
import threading
from queue import Queue, Empty

logger = logging.getLogger("eva.deferred_commands")

class DeferredCommandSystem:
    """Система для управления отложенными командами."""

    def __init__(self, brain):
        self.brain = brain
        self.health_checks = {}
        self.recovery_strategies = {}
        self.commands = Queue()
        self.worker_thread = None
        self.running = False
        logger.info("DeferredCommandSystem инициализирован")

    def add_module_health_check(self, module_name: str, check_func: Callable):
        """Добавляет проверку здоровья для модуля."""
        self.health_checks[module_name] = check_func
        logger.debug(f"Добавлена проверка здоровья для {module_name}")

    def add_module_recovery_strategy(self, module_name: str, recovery_func: Callable):
        """Добавляет стратегию восстановления для модуля."""
        self.recovery_strategies[module_name] = recovery_func
        logger.debug(f"Добавлена стратегия восстановления для {module_name}")

    def add_command(self, command_func: Callable, priority: str = 'normal', 
                   condition: Optional[Callable] = None, retries: int = 3, 
                   delay: float = 1.0, name: str = None):
        """Добавляет отложенную команду в очередь."""
        command = {
            'func': command_func,
            'priority': priority,
            'condition': condition,
            'retries': retries,
            'delay': delay,
            'name': name or command_func.__name__,
            'attempt': 0,
            'next_attempt': time.time()
        }
        self.commands.put(command)
        logger.debug(f"Добавлена отложенная команда: {command['name']}")

    def _worker(self):
        """Рабочий поток для выполнения отложенных команд."""
        while self.running:
            try:
                command = self.commands.get(timeout=1.0)
                
                # Проверяем условие выполнения
                if command['condition'] and not command['condition']():
                    # Возвращаем в очередь если условие не выполнено
                    command['next_attempt'] = time.time() + command['delay']
                    self.commands.put(command)
                    continue
                
                # Выполняем команду с повторными попытками
                for attempt in range(command['retries']):
                    try:
                        command['func']()
                        logger.debug(f"Команда {command['name']} выполнена успешно")
                        break
                    except Exception as e:
                        command['attempt'] = attempt + 1
                        if attempt < command['retries'] - 1:
                            logger.warning(f"Команда {command['name']} не удалась (попытка {command['attempt']}): {e}")
                            time.sleep(command['delay'])
                        else:
                            logger.error(f"Команда {command['name']} не удалась после {command['retries']} попыток: {e}")
                            
                            # Применяем стратегию восстановления если есть
                            recovery_func = self.recovery_strategies.get(command['name'])
                            if recovery_func:
                                try:
                                    recovery_func(e)
                                except Exception as recovery_e:
                                    logger.error(f"Стратегия восстановления не удалась: {recovery_e}")
                                    
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Ошибка в рабочем потоке: {e}")

    def start(self):
        """Запускает рабочий поток."""
        if self.worker_thread and self.worker_thread.is_alive():
            return
            
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        logger.info("DeferredCommandSystem рабочий поток запущен")

    def stop(self):
        """Останавливает рабочий поток."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        logger.info("DeferredCommandSystem рабочий поток остановлен")

    def reduce_background_tasks(self):
        """Снижает количество фоновых задач."""
        logger.info("Снижение количества фоновых задач")
        # Заглушка - в реальной реализации здесь была бы логика

    def clear(self):
        """Очищает все отложенные команды."""
        self.health_checks.clear()
        self.recovery_strategies.clear()
        # Очищаем очередь команд
        while not self.commands.empty():
            try:
                self.commands.get_nowait()
            except Empty:
                break
        logger.info("Все отложенные команды очищены")
