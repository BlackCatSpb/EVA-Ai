"""
Pytest тесты для диагностики проблем инициализации CogniFlex.

Этот модуль содержит тесты для проверки корректной инициализации
ядра системы и всех компонентов.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch

# Добавляем путь к проекту
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCogniFlexInitialization:
    """Тесты для проверки инициализации CogniFlex."""

    @pytest.fixture
    def mock_config(self):
        """Фикстура с тестовой конфигурацией."""
        return {
            'debug_minimal_mode': True,
            'cache_memory_gb': 1.0,
            'use_gpu_if_available': False
        }

    @patch('cogniflex.core.core_brain.get_security_manager')
    @patch('cogniflex.core.core_brain.get_monitoring_manager')
    @patch('cogniflex.core.core_brain.get_recovery_manager')
    @patch('cogniflex.core.core_brain.apply_hardware_optimizations')
    @patch('cogniflex.core.core_brain.get_runtime_diagnostics')
    def test_core_brain_initialization(self, mock_diagnostics, mock_hw_opt, mock_recovery,
                                      mock_monitoring, mock_security, mock_config):
        """Тест инициализации CoreBrain."""
        # Настраиваем моки
        mock_hw_opt.return_value = None
        mock_diagnostics.return_value = {
            'device': 'cpu', 'precision': 'fp32', 'torch_threads': 4,
            'interop_threads': 2, 'pin_memory_default': False
        }

        mock_security_manager = Mock()
        mock_monitoring_manager = Mock()
        mock_recovery_manager = Mock()

        mock_security.return_value = mock_security_manager
        mock_monitoring.return_value = mock_monitoring_manager
        mock_recovery.return_value = mock_recovery_manager

        # Импортируем и создаем экземпляр
        from cogniflex.core.core_brain import CoreBrain

        try:
            brain = CoreBrain(config=mock_config)

            # Проверяем основные атрибуты
            assert hasattr(brain, 'components')
            assert hasattr(brain, 'config')
            assert hasattr(brain, 'query_logger')
            assert hasattr(brain, 'events')

            # Проверяем компоненты
            assert 'security_manager' in brain.components
            assert 'monitoring_manager' in brain.components
            assert 'recovery_manager' in brain.components

            # Проверяем менеджеров
            assert brain.security_manager == mock_security_manager
            assert brain.monitoring_manager == mock_monitoring_manager
            assert brain.recovery_manager == mock_recovery_manager

            # Проверяем, что методы были вызваны
            mock_hw_opt.assert_called_once()
            mock_diagnostics.assert_called_once()
            mock_security.assert_called_once()
            mock_monitoring.assert_called_once()
            mock_recovery.assert_called_once()

            print("✅ Тест инициализации CoreBrain пройден успешно")

        except Exception as e:
            print(f"❌ Ошибка в тесте инициализации CoreBrain: {e}")
            import traceback
            traceback.print_exc()
            raise

    def test_hardware_optimizations_module(self):
        """Тест модуля аппаратных оптимизаций."""
        try:
            from cogniflex.core.hardware_optimizations import (
                apply_hardware_optimizations,
                get_runtime_diagnostics
            )

            # Тест применения оптимизаций
            apply_hardware_optimizations('cpu', {})

            # Тест получения диагностики
            diag = get_runtime_diagnostics('cpu', 'fp32', False)

            assert isinstance(diag, dict)
            assert 'device' in diag
            assert 'precision' in diag

            print("✅ Тест аппаратных оптимизаций пройден успешно")

        except Exception as e:
            print(f"❌ Ошибка в тесте аппаратных оптимизаций: {e}")
            raise

    def test_event_system_module(self):
        """Тест модуля событийной системы."""
        try:
            from cogniflex.core.event_management import SimpleEventSystem

            # Создаем систему событий
            events = SimpleEventSystem()

            # Тест подписки и вызова
            callback_called = False

            def test_callback(*args, **kwargs):
                nonlocal callback_called
                callback_called = True

            events.on('test_event', test_callback)
            events.trigger('test_event')

            assert callback_called
            assert events.get_listeners_count('test_event') == 1

            print("✅ Тест событийной системы пройден успешно")

        except Exception as e:
            print(f"❌ Ошибка в тесте событийной системы: {e}")
            raise

    def test_component_managers_module(self):
        """Тест модуля менеджеров компонентов."""
        try:
            from cogniflex.core.component_managers import (
                get_security_manager,
                get_monitoring_manager,
                get_recovery_manager
            )

            # Тест создания менеджеров
            security = get_security_manager()
            monitoring = get_monitoring_manager()
            recovery = get_recovery_manager()

            assert security is not None
            assert monitoring is not None
            assert recovery is not None

            print("✅ Тест менеджеров компонентов пройден успешно")

        except Exception as e:
            print(f"❌ Ошибка в тесте менеджеров компонентов: {e}")
            raise

    def test_fractal_attention_system(self):
        """Тест системы фрактального внимания."""
        try:
            from cogniflex.core.fractal_attention_system import FractalAttentionSystem

            # Создаем mock core_brain
            mock_brain = Mock()
            mock_brain.memory_manager = None
            mock_brain.knowledge_graph = None

            # Создаем систему
            fas = FractalAttentionSystem(mock_brain)

            assert hasattr(fas, 'dialog_manager')
            assert hasattr(fas, 'contradiction_resolver')
            assert hasattr(fas, 'learning_scheduler')
            assert hasattr(fas, 'system_optimizer')

            print("✅ Тест системы фрактального внимания пройден успешно")

        except Exception as e:
            print(f"❌ Ошибка в тесте системы фрактального внимания: {e}")
            raise


def run_diagnostics():
    """Запуск диагностики системы."""
    print("🚀 ЗАПУСК ДИАГНОСТИКИ COGNIFLEX")
    print("=" * 50)

    # Тест 1: Проверка импортов
    print("\n📦 ТЕСТ 1: ПРОВЕРКА ИМПОРТОВ")
    try:
        from cogniflex.core.core_brain import CoreBrain
        from cogniflex.core.hardware_optimizations import apply_hardware_optimizations
        from cogniflex.core.event_management import SimpleEventSystem
        from cogniflex.core.component_managers import get_security_manager
        from cogniflex.core.fractal_attention_system import FractalAttentionSystem
        print("✅ Все импорты успешны")
    except Exception as e:
        print(f"❌ Ошибка импортов: {e}")
        return False

    # Тест 2: Проверка создания экземпляров
    print("\n🔧 ТЕСТ 2: ПРОВЕРКА СОЗДАНИЯ ЭКЗЕМПЛЯРОВ")
    try:
        config = {'debug_minimal_mode': True, 'use_gpu_if_available': False}
        brain = CoreBrain(config=config)
        print("✅ CoreBrain создан успешно")
        print(f"   Компонентов: {len(brain.components)}")
        print(f"   Менеджеров: security={brain.security_manager is not None}, monitoring={brain.monitoring_manager is not None}, recovery={brain.recovery_manager is not None}")
    except Exception as e:
        print(f"❌ Ошибка создания CoreBrain: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Тест 3: Проверка атрибутов
    print("\n🔍 ТЕСТ 3: ПРОВЕРКА АТРИБУТОВ")
    try:
        required_attrs = ['components', 'config', 'query_logger', 'events', 'security_manager', 'monitoring_manager', 'recovery_manager']
        for attr in required_attrs:
            if hasattr(brain, attr):
                print(f"   ✅ {attr}: {'OK' if getattr(brain, attr) is not None else 'None'}")
            else:
                print(f"   ❌ {attr}: отсутствует")
                return False
    except Exception as e:
        print(f"❌ Ошибка проверки атрибутов: {e}")
        return False

    print("\n🎉 ДИАГНОСТИКА ЗАВЕРШЕНА УСПЕШНО!")
    return True


if __name__ == "__main__":
    run_diagnostics()
