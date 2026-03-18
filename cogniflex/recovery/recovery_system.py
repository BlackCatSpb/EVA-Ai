#!/usr/bin/env python3
"""
CogniFlex Recovery System
Механизмы восстановления после сбоев и катастрофических ситуаций.
"""

import os
import time
import logging
import threading
import json
import shutil
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import traceback

logger = logging.getLogger("cogniflex.recovery")

@dataclass
class RecoveryCheckpoint:
    """Контрольная точка восстановления."""
    checkpoint_id: str
    component_name: str
    timestamp: datetime
    state_data: Dict[str, Any]
    checksum: str
    priority: int  # 1-10, где 10 - максимальный приоритет

@dataclass
class RecoveryPlan:
    """План восстановления."""
    plan_id: str
    failure_type: str
    component_name: str
    steps: List[Dict[str, Any]]
    estimated_time: int  # секунды
    created_at: datetime
    executed_at: Optional[datetime] = None
    success: Optional[bool] = None

class ComponentStateManager:
    """Менеджер состояний компонентов для восстановления."""

    def __init__(self, checkpoint_dir: str = "recovery_checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("recovery_checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.current_states: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    def save_component_state(self, component_name: str, state_data: Dict[str, Any],
                           priority: int = 5) -> str:
        """Сохраняет состояние компонента."""
        checkpoint_id = f"{component_name}_{int(time.time())}_{os.urandom(4).hex()}"

        checkpoint = RecoveryCheckpoint(
            checkpoint_id=checkpoint_id,
            component_name=component_name,
            timestamp=datetime.now(),
            state_data=state_data.copy(),
            checksum=self._calculate_checksum(state_data),
            priority=priority
        )

        # Сохраняем в памяти
        with self.lock:
            self.current_states[component_name] = state_data.copy()

        # Сохраняем на диск
        self._save_checkpoint_to_disk(checkpoint)

        logger.info(f"Состояние компонента {component_name} сохранено (checkpoint: {checkpoint_id})")
        return checkpoint_id

    def load_component_state(self, component_name: str) -> Optional[Dict[str, Any]]:
        """Загружает последнее сохраненное состояние компонента."""
        # Сначала проверяем в памяти
        with self.lock:
            if component_name in self.current_states:
                return self.current_states[component_name].copy()

        # Затем ищем на диске
        checkpoints = self._load_checkpoints_from_disk(component_name)
        if checkpoints:
            # Возвращаем самый свежий checkpoint
            latest_checkpoint = max(checkpoints, key=lambda c: c.timestamp)
            return latest_checkpoint.state_data.copy()

        return None

    def _save_checkpoint_to_disk(self, checkpoint: RecoveryCheckpoint):
        """Сохраняет checkpoint на диск."""
        try:
            filename = f"{checkpoint.checkpoint_id}.json"
            filepath = self.checkpoint_dir / filename

            data = {
                "checkpoint_id": checkpoint.checkpoint_id,
                "component_name": checkpoint.component_name,
                "timestamp": checkpoint.timestamp.isoformat(),
                "state_data": checkpoint.state_data,
                "checksum": checkpoint.checksum,
                "priority": checkpoint.priority
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        except Exception as e:
            logger.error(f"Ошибка сохранения checkpoint {checkpoint.checkpoint_id}: {e}")

    def _load_checkpoints_from_disk(self, component_name: str) -> List[RecoveryCheckpoint]:
        """Загружает checkpoints компонента с диска."""
        checkpoints = []

        try:
            for filepath in self.checkpoint_dir.glob(f"{component_name}_*.json"):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    checkpoint = RecoveryCheckpoint(
                        checkpoint_id=data["checkpoint_id"],
                        component_name=data["component_name"],
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        state_data=data["state_data"],
                        checksum=data["checksum"],
                        priority=data.get("priority", 5)
                    )

                    checkpoints.append(checkpoint)

                except Exception as e:
                    logger.warning(f"Ошибка загрузки checkpoint {filepath}: {e}")

        except Exception as e:
            logger.error(f"Ошибка сканирования checkpoints для {component_name}: {e}")

        return sorted(checkpoints, key=lambda c: c.timestamp, reverse=True)

    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Вычисляет контрольную сумму для данных."""
        import hashlib
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()

    def cleanup_old_checkpoints(self, max_age_days: int = 7, max_per_component: int = 10):
        """Очищает старые checkpoints."""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)

        try:
            for filepath in self.checkpoint_dir.glob("*.json"):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    checkpoint_date = datetime.fromisoformat(data["timestamp"])
                    component_name = data["component_name"]

                    # Проверяем возраст
                    if checkpoint_date < cutoff_date:
                        os.remove(filepath)
                        logger.debug(f"Удален старый checkpoint: {filepath.name}")
                        continue

                    # Проверяем количество checkpoints на компонент
                    component_checkpoints = list(self.checkpoint_dir.glob(f"{component_name}_*.json"))
                    if len(component_checkpoints) > max_per_component:
                        # Удаляем самые старые
                        component_checkpoints.sort(key=lambda p: p.stat().st_mtime)
                        for old_file in component_checkpoints[:-max_per_component]:
                            os.remove(old_file)
                            logger.debug(f"Удален лишний checkpoint: {old_file.name}")

                except Exception as e:
                    logger.warning(f"Ошибка обработки checkpoint {filepath}: {e}")

        except Exception as e:
            logger.error(f"Ошибка очистки checkpoints: {e}")

class FailureDetector:
    """Детектор сбоев в компонентах."""

    def __init__(self):
        self.failure_patterns: Dict[str, Dict[str, Any]] = {}
        self.failure_history: List[Dict[str, Any]] = []
        self.lock = threading.Lock()

    def register_failure_pattern(self, pattern_name: str, pattern_config: Dict[str, Any]):
        """Регистрирует паттерн сбоя."""
        with self.lock:
            self.failure_patterns[pattern_name] = pattern_config

    def detect_failure(self, component_name: str, error_info: Dict[str, Any]) -> Optional[str]:
        """Обнаруживает сбой и возвращает тип паттерна."""
        with self.lock:
            for pattern_name, pattern in self.failure_patterns.items():
                if self._matches_pattern(error_info, pattern):
                    # Записываем в историю
                    failure_record = {
                        "component": component_name,
                        "pattern": pattern_name,
                        "timestamp": datetime.now(),
                        "error_info": error_info
                    }
                    self.failure_history.append(failure_record)

                    logger.warning(f"Обнаружен сбой {pattern_name} в компоненте {component_name}")
                    return pattern_name

        return None

    def _matches_pattern(self, error_info: Dict[str, Any], pattern: Dict[str, Any]) -> bool:
        """Проверяет соответствие ошибки паттерну."""
        # Отладка: логируем проверку паттерна
        logger.debug(f"Checking pattern {pattern} against error_info {error_info}")

        # Проверяем тип исключения
        if "exception_type" in pattern:
            expected_type = pattern["exception_type"]
            actual_type = error_info.get("exception_type")
            if actual_type != expected_type:
                logger.debug(f"Exception type mismatch: expected {expected_type}, got {actual_type}")
                return False

        # Проверяем сообщение об ошибке
        if "error_message_contains" in pattern:
            search_text = pattern["error_message_contains"]
            error_msg = error_info.get("error_message", "")
            if search_text not in error_msg:
                logger.debug(f"Error message doesn't contain '{search_text}': '{error_msg}'")
                return False

        # Проверяем стек вызовов
        if "stack_contains" in pattern:
            search_text = pattern["stack_contains"]
            stack_trace = error_info.get("stack_trace", "")
            if search_text not in stack_trace:
                logger.debug(f"Stack trace doesn't contain '{search_text}'")
                return False

        logger.debug("Pattern matched successfully")
        return True

    def get_failure_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику сбоев."""
        with self.lock:
            stats = {}

            # Статистика по компонентам
            component_failures = {}
            for failure in self.failure_history:
                component = failure["component"]
                component_failures[component] = component_failures.get(component, 0) + 1

            # Статистика по паттернам
            pattern_failures = {}
            for failure in self.failure_history:
                pattern = failure["pattern"]
                pattern_failures[pattern] = pattern_failures.get(pattern, 0) + 1

            stats["total_failures"] = len(self.failure_history)
            stats["component_failures"] = component_failures
            stats["pattern_failures"] = pattern_failures
            stats["most_recent_failure"] = self.failure_history[-1] if self.failure_history else None

            return stats

class RecoveryManager:
    """Основной менеджер восстановления."""

    def __init__(self, checkpoint_dir: str = "recovery_checkpoints"):
        self.state_manager = ComponentStateManager(checkpoint_dir)
        self.failure_detector = FailureDetector()
        self.recovery_plans: Dict[str, RecoveryPlan] = {}
        self.active_recoveries: Dict[str, threading.Thread] = {}
        self.lock = threading.Lock()

        self._setup_default_failure_patterns()
        self._setup_default_recovery_plans()

    def _setup_default_failure_patterns(self):
        """Настраивает стандартные паттерны сбоев."""

        # CUDA out of memory
        self.failure_detector.register_failure_pattern(
            "cuda_oom",
            {
                "exception_type": "RuntimeError",
                "error_message_contains": "out of memory",
                "stack_contains": "cuda"
            }
        )

        # Network connection error
        self.failure_detector.register_failure_pattern(
            "network_error",
            {
                "exception_type": "ConnectionError",
                "error_message_contains": "connection"
            }
        )

        # Database connection error
        self.failure_detector.register_failure_pattern(
            "database_error",
            {
                "exception_type": "OperationalError",
                "error_message_contains": "database"
            }
        )

        # File system error
        self.failure_detector.register_failure_pattern(
            "filesystem_error",
            {
                "exception_type": "OSError",
                "error_message_contains": "No space left on device"
            }
        )

    def _setup_default_recovery_plans(self):
        """Настраивает стандартные планы восстановления."""

        # План для CUDA OOM
        cuda_recovery_plan = RecoveryPlan(
            plan_id="cuda_oom_recovery",
            failure_type="cuda_oom",
            component_name="ml_unit",
            steps=[
                {
                    "step": 1,
                    "action": "reduce_batch_size",
                    "description": "Уменьшить размер батча до 1",
                    "timeout": 30
                },
                {
                    "step": 2,
                    "action": "clear_gpu_cache",
                    "description": "Очистить GPU кэш",
                    "timeout": 10
                },
                {
                    "step": 3,
                    "action": "restart_component",
                    "description": "Перезапустить ML компонент",
                    "timeout": 60
                }
            ],
            estimated_time=100,
            created_at=datetime.now()
        )

        # План для сетевых ошибок
        network_recovery_plan = RecoveryPlan(
            plan_id="network_recovery",
            failure_type="network_error",
            component_name="web_search",
            steps=[
                {
                    "step": 1,
                    "action": "retry_with_backoff",
                    "description": "Повторить запрос с экспоненциальной задержкой",
                    "timeout": 300
                },
                {
                    "step": 2,
                    "action": "switch_endpoint",
                    "description": "Переключиться на альтернативный endpoint",
                    "timeout": 30
                }
            ],
            estimated_time=330,
            created_at=datetime.now()
        )

        self.recovery_plans["cuda_oom_recovery"] = cuda_recovery_plan
        self.recovery_plans["network_recovery"] = network_recovery_plan

    def handle_failure(self, component_name: str, exception: Exception,
                      context: Optional[Dict[str, Any]] = None) -> bool:
        """Обрабатывает сбой компонента."""
        import traceback

        error_info = {
            "exception_type": type(exception).__name__,
            "error_message": str(exception),
            "stack_trace": traceback.format_exc(),
            "context": context or {},
            "timestamp": datetime.now()
        }

        # Обнаруживаем тип сбоя
        failure_pattern = self.failure_detector.detect_failure(component_name, error_info)

        if failure_pattern and failure_pattern in self.recovery_plans:
            # Запускаем восстановление
            recovery_plan = self.recovery_plans[failure_pattern]
            return self._execute_recovery_plan(component_name, recovery_plan, error_info)

        # Если нет плана восстановления, логируем ошибку
        logger.error(f"Критический сбой в {component_name}: {exception}")
        return False

    def _execute_recovery_plan(self, component_name: str, plan: RecoveryPlan,
                              error_info: Dict[str, Any]) -> bool:
        """Выполняет план восстановления."""
        plan.executed_at = datetime.now()

        logger.info(f"Запуск плана восстановления {plan.plan_id} для {component_name}")

        success = True

        for step in plan.steps:
            try:
                logger.info(f"Выполнение шага {step['step']}: {step['description']}")

                # Выполняем шаг восстановления
                step_success = self._execute_recovery_step(component_name, step, error_info)

                if not step_success:
                    logger.error(f"Шаг {step['step']} не удался")
                    success = False
                    break

            except Exception as e:
                logger.error(f"Ошибка выполнения шага {step['step']}: {e}")
                success = False
                break

        plan.success = success

        if success:
            logger.info(f"План восстановления {plan.plan_id} выполнен успешно")
        else:
            logger.error(f"План восстановления {plan.plan_id} завершился неудачей")

        return success

    def _execute_recovery_step(self, component_name: str, step: Dict[str, Any],
                              error_info: Dict[str, Any]) -> bool:
        """Выполняет шаг восстановления."""
        action = step["action"]
        timeout = step.get("timeout", 30)

        try:
            if action == "reduce_batch_size":
                return self._reduce_batch_size(component_name, timeout)

            elif action == "clear_gpu_cache":
                return self._clear_gpu_cache(timeout)

            elif action == "restart_component":
                return self._restart_component(component_name, timeout)

            elif action == "retry_with_backoff":
                return self._retry_with_backoff(component_name, error_info, timeout)

            elif action == "switch_endpoint":
                return self._switch_endpoint(component_name, timeout)

            else:
                logger.warning(f"Неизвестное действие восстановления: {action}")
                return False

        except Exception as e:
            logger.error(f"Ошибка выполнения действия {action}: {e}")
            return False

    def _reduce_batch_size(self, component_name: str, timeout: int) -> bool:
        """Уменьшает размер батча."""
        try:
            # Находим компонент и уменьшаем batch size
            # Это зависит от конкретной реализации компонента
            logger.info(f"Уменьшение размера батча для {component_name}")
            time.sleep(1)  # Имитация работы
            return True
        except Exception as e:
            logger.error(f"Ошибка уменьшения batch size: {e}")
            return False

    def _clear_gpu_cache(self, timeout: int) -> bool:
        """Очищает GPU кэш."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU кэш очищен")
            return True
        except Exception as e:
            logger.error(f"Ошибка очистки GPU кэша: {e}")
            return False

    def _restart_component(self, component_name: str, timeout: int) -> bool:
        """Перезапускает компонент."""
        try:
            logger.info(f"Перезапуск компонента {component_name}")
            # Здесь должна быть логика перезапуска конкретного компонента
            time.sleep(2)  # Имитация перезапуска
            return True
        except Exception as e:
            logger.error(f"Ошибка перезапуска компонента {component_name}: {e}")
            return False

    def _retry_with_backoff(self, component_name: str, error_info: Dict[str, Any], timeout: int) -> bool:
        """Повторяет операцию с экспоненциальной задержкой."""
        try:
            max_attempts = 3
            base_delay = 1

            for attempt in range(max_attempts):
                delay = base_delay * (2 ** attempt)
                logger.info(f"Попытка {attempt + 1}/{max_attempts} через {delay} сек")

                time.sleep(delay)

                # Имитация повторной попытки
                if attempt == max_attempts - 1:  # Последняя попытка успешна
                    return True

            return False
        except Exception as e:
            logger.error(f"Ошибка повторной попытки: {e}")
            return False

    def _switch_endpoint(self, component_name: str, timeout: int) -> bool:
        """Переключается на альтернативный endpoint."""
        try:
            logger.info(f"Переключение endpoint для {component_name}")
            # Здесь должна быть логика переключения endpoint
            time.sleep(1)  # Имитация работы
            return True
        except Exception as e:
            logger.error(f"Ошибка переключения endpoint: {e}")
            return False

    def create_backup(self, component_name: str, backup_path: Optional[str] = None) -> bool:
        """Создает резервную копию компонента."""
        try:
            if not backup_path:
                backup_path = f"backup_{component_name}_{int(time.time())}"

            # Сохраняем состояние компонента
            state = self.state_manager.load_component_state(component_name)
            if state:
                os.makedirs(backup_path, exist_ok=True)

                backup_file = os.path.join(backup_path, f"{component_name}_state.json")
                with open(backup_file, 'w', encoding='utf-8') as f:
                    json.dump(state, f, ensure_ascii=False, indent=2, default=str)

                logger.info(f"Резервная копия создана: {backup_path}")
                return True
            else:
                logger.warning(f"Не удалось создать резервную копию {component_name}: состояние не найдено")
                return False

        except Exception as e:
            logger.error(f"Ошибка создания резервной копии {component_name}: {e}")
            return False

    def restore_from_backup(self, component_name: str, backup_path: str) -> bool:
        """Восстанавливает компонент из резервной копии."""
        try:
            backup_file = os.path.join(backup_path, f"{component_name}_state.json")

            if not os.path.exists(backup_file):
                logger.error(f"Файл резервной копии не найден: {backup_file}")
                return False

            with open(backup_file, 'r', encoding='utf-8') as f:
                state = json.load(f)

            # Восстанавливаем состояние
            self.state_manager.save_component_state(component_name, state, priority=10)

            logger.info(f"Компонент {component_name} восстановлен из {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Ошибка восстановления {component_name} из {backup_path}: {e}")
            return False

    def get_recovery_status(self) -> Dict[str, Any]:
        """Возвращает статус восстановления системы."""
        return {
            "active_recoveries": len(self.active_recoveries),
            "available_plans": len(self.recovery_plans),
            "failure_statistics": self.failure_detector.get_failure_statistics(),
            "checkpoint_statistics": self._get_checkpoint_statistics()
        }

    def _get_checkpoint_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику checkpoints."""
        try:
            total_checkpoints = len(list(self.state_manager.checkpoint_dir.glob("*.json")))
            return {
                "total_checkpoints": total_checkpoints,
                "checkpoint_directory": str(self.state_manager.checkpoint_dir)
            }
        except Exception:
            return {"total_checkpoints": 0, "error": "Unable to read checkpoints"}

# Глобальный экземпляр менеджера восстановления
recovery_manager = RecoveryManager()

def get_recovery_manager() -> RecoveryManager:
    """Возвращает глобальный менеджер восстановления."""
    return recovery_manager

# Декоратор для автоматического восстановления
def with_recovery(component_name: str, max_retries: int = 3):
    """Декоратор для автоматического восстановления компонента."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries):
                try:
                    logger.debug(f"Executing {func.__name__} attempt {attempt + 1}/{max_retries}")
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Попытка {attempt + 1}/{max_retries} не удалась для {component_name}: {e}")

                    # Пытаемся восстановить компонент (но не прерываем выполнение из-за неудачи)
                    if attempt < max_retries - 1:  # Не на последней попытке
                        try:
                            # Создаем error_info для обнаружения паттерна
                            import traceback
                            error_info = {
                                "exception_type": type(e).__name__,
                                "error_message": str(e),
                                "stack_trace": traceback.format_exc(),
                                "context": {"attempt": attempt + 1, "function": func.__name__}
                            }

                            # Пытаемся восстановить, но не ждем успеха
                            recovery_success = recovery_manager.handle_failure(component_name, e, error_info)

                            if recovery_success:
                                logger.info(f"Компонент {component_name} восстановлен, повторяем попытку {attempt + 1}")
                            else:
                                logger.warning(f"Не удалось восстановить {component_name}, пробуем еще раз")

                            # Небольшая пауза перед повторной попыткой
                            time.sleep(0.1)

                        except Exception as recovery_e:
                            logger.error(f"Ошибка при попытке восстановления {component_name}: {recovery_e}")

                    # Продолжаем цикл для следующей попытки

            # Если все попытки исчерпаны
            logger.error(f"Все {max_retries} попыток выполнения {func.__name__} для {component_name} исчерпаны")
            if last_exception:
                logger.error(f"Последняя ошибка: {last_exception}")
                raise last_exception
            else:
                raise RuntimeError(f"Все {max_retries} попыток выполнения {func.__name__} для {component_name} исчерпаны")

        return wrapper
    return decorator

# Утилита для graceful shutdown
def graceful_shutdown():
    """Выполняет корректное завершение работы системы."""
    logger.info("Начинается graceful shutdown...")

    try:
        # Сохраняем финальные состояния всех компонентов
        components_to_save = [
            "core_brain",
            "memory_manager",
            "knowledge_graph",
            "ml_unit",
            "chat_module"
        ]

        saved_count = 0
        total_count = len(components_to_save)

        for component in components_to_save:
            try:
                # Сохраняем текущее состояние компонента
                current_state = recovery_manager.state_manager.load_component_state(component)
                if current_state is None:
                    current_state = {}

                # Добавляем информацию о shutdown
                shutdown_state = current_state.copy()
                shutdown_state.update({
                    "shutdown_test": True,
                    "shutdown_time": datetime.now().isoformat(),
                    "shutdown_reason": "graceful_shutdown",
                    "shutdown_timestamp": time.time()
                })

                # Сохраняем с высоким приоритетом
                checkpoint_id = recovery_manager.state_manager.save_component_state(
                    component,
                    shutdown_state,
                    priority=10
                )

                if checkpoint_id:
                    saved_count += 1
                    logger.debug(f"Состояние {component} сохранено с checkpoint_id: {checkpoint_id}")
                else:
                    logger.warning(f"Не удалось получить checkpoint_id для {component}")

            except Exception as e:
                logger.warning(f"Не удалось сохранить состояние {component}: {e}")

        logger.info(f"Graceful shutdown завершен: сохранено {saved_count}/{total_count} компонентов")

        # Возвращаем статистику для тестирования
        return {
            "success": True,
            "saved_count": saved_count,
            "total_count": total_count,
            "shutdown_time": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Ошибка при graceful shutdown: {e}")
        return {
            "success": False,
            "error": str(e),
            "shutdown_time": datetime.now().isoformat()
        }

# Автоматическая настройка при импорте
def _setup_auto_recovery():
    """Настраивает автоматическое восстановление."""
    try:
        logger.info("Система восстановления инициализирована")
    except Exception as e:
        logger.error(f"Ошибка инициализации системы восстановления: {e}")

_setup_auto_recovery()
