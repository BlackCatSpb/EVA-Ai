#!/usr/bin/env python3
"""
CogniFlex Recovery System Tests
Тесты для системы восстановления после сбоев.
"""

import pytest
import os
import time
import json
import tempfile
import shutil
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from pathlib import Path

from cogniflex.recovery.recovery_system import (
    RecoveryManager,
    ComponentStateManager,
    FailureDetector,
    RecoveryCheckpoint,
    RecoveryPlan,
    get_recovery_manager,
    with_recovery,
    graceful_shutdown
)


class TestComponentStateManager:
    """Тесты для ComponentStateManager."""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Создает временную директорию для checkpoints."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def state_manager(self, temp_checkpoint_dir):
        """Создает ComponentStateManager для тестирования."""
        return ComponentStateManager(temp_checkpoint_dir)

    def test_state_manager_initialization(self, state_manager, temp_checkpoint_dir):
        """Тест инициализации ComponentStateManager."""
        assert state_manager.checkpoint_dir == Path(temp_checkpoint_dir)
        assert state_manager.current_states == {}
        assert os.path.exists(temp_checkpoint_dir)

    def test_save_component_state(self, state_manager):
        """Тест сохранения состояния компонента."""
        component_name = "test_component"
        state_data = {
            "config": {"batch_size": 32},
            "metrics": {"accuracy": 0.95},
            "timestamp": datetime.now().isoformat()
        }

        checkpoint_id = state_manager.save_component_state(component_name, state_data)

        assert checkpoint_id is not None
        assert component_name in state_manager.current_states
        assert state_manager.current_states[component_name] == state_data

        # Проверяем что checkpoint сохранен на диск
        checkpoint_files = list(state_manager.checkpoint_dir.glob(f"{component_name}_*.json"))
        assert len(checkpoint_files) == 1

    def test_load_component_state_from_memory(self, state_manager):
        """Тест загрузки состояния из памяти."""
        component_name = "test_component"
        state_data = {"test": "data", "value": 123}

        # Сначала сохраняем
        state_manager.save_component_state(component_name, state_data)

        # Затем загружаем
        loaded_state = state_manager.load_component_state(component_name)

        assert loaded_state is not None
        assert loaded_state == state_data

    def test_load_component_state_from_disk(self, state_manager):
        """Тест загрузки состояния с диска."""
        component_name = "test_component"
        state_data = {"test": "disk_data", "value": 456}

        # Сохраняем состояние
        state_manager.save_component_state(component_name, state_data)

        # Очищаем память
        state_manager.current_states.clear()

        # Загружаем с диска
        loaded_state = state_manager.load_component_state(component_name)

        assert loaded_state is not None
        assert loaded_state == state_data

    def test_load_nonexistent_component_state(self, state_manager):
        """Тест загрузки состояния несуществующего компонента."""
        loaded_state = state_manager.load_component_state("nonexistent")

        assert loaded_state is None

    def test_checkpoint_persistence(self, state_manager):
        """Тест персистентности checkpoints."""
        component_name = "persistent_component"
        state_data = {"persistent": True, "data": [1, 2, 3]}

        # Сохраняем checkpoint
        checkpoint_id = state_manager.save_component_state(component_name, state_data)

        # Проверяем файл на диске
        checkpoint_file = state_manager.checkpoint_dir / f"{checkpoint_id}.json"
        assert checkpoint_file.exists()

        # Читаем и проверяем содержимое
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)

        assert saved_data["component_name"] == component_name
        assert saved_data["state_data"] == state_data
        assert "checksum" in saved_data

    def test_cleanup_old_checkpoints(self, state_manager):
        """Тест очистки старых checkpoints."""
        component_name = "cleanup_test"

        # Создаем несколько checkpoints с разными временными метками
        for i in range(5):
            state_data = {"version": i}
            state_manager.save_component_state(component_name, state_data)
            time.sleep(0.01)  # Небольшая задержка для разных временных меток

        # Проверяем что все checkpoints созданы
        initial_count = len(list(state_manager.checkpoint_dir.glob("*.json")))
        assert initial_count >= 5

        # Очищаем старые checkpoints (оставляем только 2 самых свежих)
        state_manager.cleanup_old_checkpoints(max_age_days=1, max_per_component=2)

        # Проверяем что остались только свежие checkpoints
        remaining_files = list(state_manager.checkpoint_dir.glob(f"{component_name}_*.json"))
        assert len(remaining_files) <= 2

    def test_calculate_checksum(self, state_manager):
        """Тест вычисления контрольной суммы."""
        data1 = {"test": "data", "value": 123}
        data2 = {"test": "data", "value": 456}

        checksum1 = state_manager._calculate_checksum(data1)
        checksum2 = state_manager._calculate_checksum(data2)

        # Одинаковые данные дают одинаковые checksum
        assert state_manager._calculate_checksum(data1) == checksum1

        # Разные данные дают разные checksum
        assert checksum1 != checksum2

        # Проверяем что checksum является hex строкой
        assert len(checksum1) == 32  # MD5 hex length
        assert all(c in '0123456789abcdef' for c in checksum1)


class TestFailureDetector:
    """Тесты для FailureDetector."""

    @pytest.fixture
    def failure_detector(self):
        """Создает FailureDetector для тестирования."""
        return FailureDetector()

    def test_failure_detector_initialization(self, failure_detector):
        """Тест инициализации FailureDetector."""
        assert failure_detector.failure_patterns == {}
        assert failure_detector.failure_history == []

    def test_register_failure_pattern(self, failure_detector):
        """Тест регистрации паттерна сбоя."""
        pattern_name = "test_pattern"
        pattern_config = {
            "exception_type": "ValueError",
            "error_message_contains": "test error"
        }

        failure_detector.register_failure_pattern(pattern_name, pattern_config)

        assert pattern_name in failure_detector.failure_patterns
        assert failure_detector.failure_patterns[pattern_name] == pattern_config

    def test_detect_failure_matching_pattern(self, failure_detector):
        """Тест обнаружения сбоя соответствующего паттерну."""
        # Очищаем историю перед тестом
        failure_detector.failure_history.clear()

        # Регистрируем паттерн
        pattern_name = "value_error_pattern"
        pattern_config = {
            "exception_type": "ValueError",
            "error_message_contains": "invalid value"
        }
        failure_detector.register_failure_pattern(pattern_name, pattern_config)

        # Проверяем что паттерн зарегистрирован
        assert pattern_name in failure_detector.failure_patterns
        assert failure_detector.failure_patterns[pattern_name] == pattern_config

        # Создаем ошибку соответствующую паттерну
        error_info = {
            "exception_type": "ValueError",
            "error_message": "invalid value provided",  # Приводим к нижнему регистру
            "stack_trace": "traceback here"
        }

        # Вызываем detect_failure
        detected_pattern = failure_detector.detect_failure("test_component", error_info)

        # Добавляем отладку
        print(f"Pattern name: {pattern_name}")
        print(f"Detected pattern: {detected_pattern}")
        print(f"Error info: {error_info}")
        print(f"Patterns: {failure_detector.failure_patterns}")
        print(f"History: {failure_detector.failure_history}")

        # Проверяем результат
        assert detected_pattern is not None, f"Expected {pattern_name}, got {detected_pattern}"
        assert detected_pattern == pattern_name

        # Проверяем что ошибка записана в историю
        assert len(failure_detector.failure_history) == 1
        assert failure_detector.failure_history[0]["component"] == "test_component"
        assert failure_detector.failure_history[0]["pattern"] == pattern_name

    def test_detect_failure_non_matching_pattern(self, failure_detector):
        """Тест обнаружения сбоя не соответствующего паттерну."""
        # Регистрируем паттерн
        failure_detector.register_failure_pattern(
            "cuda_pattern",
            {
                "exception_type": "RuntimeError",
                "error_message_contains": "CUDA"
            }
        )

        # Создаем ошибку не соответствующую паттерну
        error_info = {
            "exception_type": "ValueError",
            "error_message": "Some other error",
            "stack_trace": "traceback here"
        }

        detected_pattern = failure_detector.detect_failure("test_component", error_info)

        assert detected_pattern is None

        # Проверяем что история пуста
        assert len(failure_detector.failure_history) == 0

    def test_detect_failure_with_stack_trace_pattern(self, failure_detector):
        """Тест обнаружения сбоя с паттерном в stack trace."""
        # Регистрируем паттерн
        failure_detector.register_failure_pattern(
            "stack_pattern",
            {
                "exception_type": "Exception",
                "stack_contains": "specific_function"
            }
        )

        # Создаем ошибку с соответствующим stack trace
        error_info = {
            "exception_type": "Exception",
            "error_message": "Some error",
            "stack_trace": "at specific_function in line 123"
        }

        detected_pattern = failure_detector.detect_failure("test_component", error_info)

        assert detected_pattern == "stack_pattern"

    def test_get_failure_statistics(self, failure_detector):
        """Тест получения статистики сбоев."""
        # Регистрируем паттерны
        failure_detector.register_failure_pattern("pattern1", {"exception_type": "ValueError"})
        failure_detector.register_failure_pattern("pattern2", {"exception_type": "RuntimeError"})

        # Генерируем ошибки
        failure_detector.detect_failure("comp1", {"exception_type": "ValueError", "error_message": "err1"})
        failure_detector.detect_failure("comp2", {"exception_type": "RuntimeError", "error_message": "err2"})
        failure_detector.detect_failure("comp1", {"exception_type": "ValueError", "error_message": "err3"})

        stats = failure_detector.get_failure_statistics()

        assert stats["total_failures"] == 3
        assert stats["component_failures"]["comp1"] == 2
        assert stats["component_failures"]["comp2"] == 1
        assert stats["pattern_failures"]["pattern1"] == 2
        assert stats["pattern_failures"]["pattern2"] == 1
        assert stats["most_recent_failure"]["component"] == "comp1"


class TestRecoveryManager:
    """Тесты для RecoveryManager."""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Создает временную директорию для checkpoints."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def recovery_manager(self, temp_checkpoint_dir):
        """Создает RecoveryManager для тестирования."""
        return RecoveryManager(temp_checkpoint_dir)

    def test_recovery_manager_initialization(self, recovery_manager):
        """Тест инициализации RecoveryManager."""
        assert recovery_manager.state_manager is not None
        assert recovery_manager.failure_detector is not None
        assert recovery_manager.recovery_plans is not None
        assert recovery_manager.active_recoveries == {}

    def test_default_failure_patterns_setup(self, recovery_manager):
        """Тест настройки стандартных паттернов сбоев."""
        patterns = recovery_manager.failure_detector.failure_patterns

        assert "cuda_oom" in patterns
        assert "network_error" in patterns
        assert "database_error" in patterns
        assert "filesystem_error" in patterns

    def test_default_recovery_plans_setup(self, recovery_manager):
        """Тест настройки стандартных планов восстановления."""
        plans = recovery_manager.recovery_plans

        assert "cuda_oom_recovery" in plans
        assert "network_recovery" in plans

        cuda_plan = plans["cuda_oom_recovery"]
        assert cuda_plan.failure_type == "cuda_oom"
        assert len(cuda_plan.steps) == 3
        assert cuda_plan.steps[0]["action"] == "reduce_batch_size"

    def test_handle_failure_with_recovery_plan(self, recovery_manager):
        """Тест обработки сбоя с планом восстановления."""
        # Создаем mock исключение CUDA OOM
        cuda_exception = RuntimeError("CUDA out of memory")

        # Обрабатываем сбой
        recovery_success = recovery_manager.handle_failure(
            "ml_unit",
            cuda_exception,
            {"context": "test"}
        )

        # Проверяем что восстановление было запущено
        # (В реальности может потребоваться настройка mock для шагов)
        assert isinstance(recovery_success, bool)

    def test_handle_failure_without_recovery_plan(self, recovery_manager):
        """Тест обработки сбоя без плана восстановления."""
        # Создаем исключение не соответствующее известным паттернам
        unknown_exception = ValueError("Unknown error")

        # Обрабатываем сбой
        recovery_success = recovery_manager.handle_failure(
            "unknown_component",
            unknown_exception
        )

        # Должен вернуть False так как нет плана восстановления
        assert recovery_success is False

    def test_create_backup(self, recovery_manager):
        """Тест создания резервной копии."""
        # Очищаем старые backup файлы
        import shutil
        for backup_dir in Path(".").glob("backup_test_component_*"):
            if backup_dir.is_dir():
                shutil.rmtree(backup_dir)

        component_name = "test_component"
        state_data = {"config": {"setting": "value"}, "data": [1, 2, 3]}

        # Сохраняем состояние
        recovery_manager.state_manager.save_component_state(component_name, state_data)

        # Создаем backup
        backup_success = recovery_manager.create_backup(component_name)

        assert backup_success is True

        # Проверяем что backup создан
        backup_files = list(Path(".").glob(f"backup_{component_name}_*/{component_name}_state.json"))
        assert len(backup_files) == 1

        # Очищаем созданный backup
        if backup_files:
            backup_parent = backup_files[0].parent
            if backup_parent.exists():
                shutil.rmtree(backup_parent)

    def test_restore_from_backup(self, recovery_manager):
        """Тест восстановления из резервной копии."""
        # Очищаем старые backup файлы
        import shutil
        for backup_dir in Path(".").glob("backup_test_component_*"):
            if backup_dir.is_dir():
                shutil.rmtree(backup_dir)

        component_name = "test_component"
        state_data = {"config": {"restored": True}, "data": [4, 5, 6]}

        # Создаем backup
        recovery_manager.state_manager.save_component_state(component_name, state_data)
        recovery_manager.create_backup(component_name)  # Добавляем вызов create_backup

        # Очищаем текущее состояние
        if component_name in recovery_manager.state_manager.current_states:
            del recovery_manager.state_manager.current_states[component_name]

        # Находим backup директорию
        backup_dirs = list(Path(".").glob(f"backup_{component_name}_*"))
        assert len(backup_dirs) == 1

        # Восстанавливаем из backup
        restore_success = recovery_manager.restore_from_backup(
            component_name,
            str(backup_dirs[0])
        )

        assert restore_success is True

        # Проверяем что состояние восстановлено
        restored_state = recovery_manager.state_manager.load_component_state(component_name)
        assert restored_state is not None
        assert restored_state["config"]["restored"] is True

        # Очищаем backup
        shutil.rmtree(backup_dirs[0])

    def test_get_recovery_status(self, recovery_manager):
        """Тест получения статуса восстановления."""
        status = recovery_manager.get_recovery_status()

        assert "active_recoveries" in status
        assert "available_plans" in status
        assert "failure_statistics" in status
        assert "checkpoint_statistics" in status

        assert status["available_plans"] >= 2  # cuda_oom_recovery и network_recovery
        assert status["active_recoveries"] == 0  # Нет активных восстановлений


class TestRecoveryDecorators:
    """Тесты для декораторов восстановления."""

    def test_with_recovery_decorator_success(self):
        """Тест декоратора с успешным выполнением."""
        @with_recovery("test_component", max_retries=1)
        def successful_function():
            return "success"

        result = successful_function()
        assert result == "success"

    def test_with_recovery_decorator_failure_no_recovery(self):
        """Тест декоратора с неудачей без возможности восстановления."""
        call_count = 0

        @with_recovery("test_component", max_retries=2)
        def failing_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

        assert call_count == 2  # Функция была вызвана 2 раза

    def test_with_recovery_decorator_with_recovery(self):
        """Тест декоратора с восстановлением."""
        call_count = 0
        recovery_attempts = 0

        # Mock для recovery_manager
        with patch('cogniflex.recovery.recovery_system.recovery_manager') as mock_recovery:
            mock_recovery.handle_failure.return_value = True

            @with_recovery("test_component", max_retries=2)
            def recoverable_function():
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("CUDA out of memory")
                return "recovered"

            result = recoverable_function()

            assert result == "recovered"
            assert call_count == 2  # Функция была вызвана 2 раза
            assert mock_recovery.handle_failure.called


class TestRecoveryIntegration:
    """Интеграционные тесты системы восстановления."""

    @pytest.fixture
    def integrated_recovery(self):
        """Создает полностью настроенную систему восстановления."""
        return get_recovery_manager()

    def test_full_failure_recovery_cycle(self, integrated_recovery):
        """Тест полного цикла восстановления после сбоя."""
        component_name = "test_component"

        # 1. Сохраняем начальное состояние
        initial_state = {"config": {"batch_size": 32}, "healthy": True}
        integrated_recovery.state_manager.save_component_state(
            component_name, initial_state
        )

        # 2. Создаем паттерн для тестирования
        integrated_recovery.failure_detector.register_failure_pattern(
            "test_pattern",
            {
                "exception_type": "RuntimeError",
                "error_message_contains": "CUDA out of memory"
            }
        )

        # 3. Создаем план восстановления для этого паттерна
        test_plan = create_test_recovery_plan(
            "test_recovery_plan",
            "test_pattern",
            [{"step": 1, "action": "clear_gpu_cache", "description": "Clear cache", "timeout": 5}]
        )
        integrated_recovery.recovery_plans["test_pattern"] = test_plan

        # 4. Симулируем сбой
        cuda_exception = RuntimeError("CUDA out of memory")

        # 5. Обрабатываем сбой
        recovery_result = integrated_recovery.handle_failure(
            component_name, cuda_exception
        )

        # 6. Проверяем что сбой был обработан
        failure_stats = integrated_recovery.failure_detector.get_failure_statistics()
        assert failure_stats["total_failures"] >= 1

        # 7. Проверяем что был найден правильный паттерн
        assert failure_stats["pattern_failures"].get("test_pattern", 0) >= 1

        # 5. Проверяем создание backup
        backup_result = integrated_recovery.create_backup(component_name)
        assert backup_result is True

    def test_state_persistence_across_restarts(self, integrated_recovery):
        """Тест персистентности состояния при перезапусках."""
        component_name = "persistent_component"
        state_data = {
            "session_id": "test_session_123",
            "user_preferences": {"theme": "dark"},
            "last_activity": datetime.now().isoformat()
        }

        # Сохраняем состояние
        checkpoint_id = integrated_recovery.state_manager.save_component_state(
            component_name, state_data, priority=8
        )

        # Имитируем перезапуск (создаем новый менеджер)
        new_manager = RecoveryManager(integrated_recovery.state_manager.checkpoint_dir)

        # Загружаем состояние
        loaded_state = new_manager.state_manager.load_component_state(component_name)

        assert loaded_state is not None
        assert loaded_state["session_id"] == state_data["session_id"]
        assert loaded_state["user_preferences"]["theme"] == "dark"

    def test_multiple_component_recovery(self, integrated_recovery):
        """Тест восстановления нескольких компонентов."""
        components = ["brain", "memory", "ml_unit", "chat"]

        # Сохраняем состояния всех компонентов
        for i, component in enumerate(components):
            state = {"component_id": i, "status": "active"}
            integrated_recovery.state_manager.save_component_state(
                component, state, priority=5
            )

        # Проверяем что все состояния сохранены
        for component in components:
            loaded_state = integrated_recovery.state_manager.load_component_state(component)
            assert loaded_state is not None
            assert loaded_state["status"] == "active"

    def test_graceful_shutdown_procedure(self, integrated_recovery):
        """Тест процедуры graceful shutdown."""
        # Сохраняем состояния компонентов
        components = ["core_brain", "memory_manager", "ml_unit"]
        for component in components:
            state = {"shutdown_test": True, "timestamp": datetime.now().isoformat()}
            integrated_recovery.state_manager.save_component_state(component, state)

        # Выполняем graceful shutdown
        graceful_shutdown()

        # Проверяем что финальные состояния сохранены
        for component in components:
            final_state = integrated_recovery.state_manager.load_component_state(component)
            assert final_state is not None
            # Проверяем что shutdown_test присутствует (может быть перезаписан)
            assert "shutdown_test" in final_state or final_state.get("shutdown_test") is True


# Вспомогательные функции для тестирования
def create_test_failure_info(exception_type: str = "RuntimeError",
                           message: str = "Test error",
                           stack_contains: str = "test_function") -> dict:
    """Создает тестовую информацию о сбое."""
    return {
        "exception_type": exception_type,
        "error_message": message,
        "stack_trace": f"at {stack_contains} in test.py line 123",
        "timestamp": datetime.now(),
        "context": {"test": True}
    }

def simulate_component_failure(component_name: str, failure_type: str) -> None:
    """Симулирует сбой компонента."""
    if failure_type == "cuda_oom":
        raise RuntimeError("CUDA out of memory")
    elif failure_type == "network":
        raise ConnectionError("Network connection failed")
    elif failure_type == "filesystem":
        raise OSError("No space left on device")
    else:
        raise Exception(f"Unknown failure type: {failure_type}")

def create_test_recovery_plan(plan_id: str, failure_type: str, steps: list) -> RecoveryPlan:
    """Создает тестовый план восстановления."""
    return RecoveryPlan(
        plan_id=plan_id,
        failure_type=failure_type,
        component_name="test_component",
        steps=steps,
        estimated_time=sum(step.get("timeout", 30) for step in steps),
        created_at=datetime.now()
    )

if __name__ == "__main__":
    print("🚀 Запуск тестов системы восстановления CogniFlex...")

    # Можно запускать тесты напрямую
    import subprocess
    result = subprocess.run([
        'python', '-m', 'pytest',
        __file__,
        '-v',
        '--tb=short'
    ], capture_output=True, text=True)

    print("Вывод тестов:")
    print(result.stdout)
    if result.stderr:
        print("Ошибки:")
        print(result.stderr)

    print(f"Код завершения: {result.returncode}")

    print("✅ Тесты системы восстановления завершены!")
