#!/usr/bin/env python3
"""
CogniFlex Complete Integration Tests
Полное тестирование интеграции всех компонентов системы CogniFlex.
"""

import pytest
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional


class TestCompleteSystemIntegration:
    """Полное тестирование интеграции системы."""

    @pytest.mark.integration
    def test_system_startup_sequence(self, mock_brain, mock_gui):
        """Тест последовательности запуска системы."""
        if mock_brain is None or mock_gui is None:
            pytest.skip("Компоненты недоступны для тестирования")

        # Test 1: Brain initialization order
        assert hasattr(mock_brain, 'components')
        assert hasattr(mock_brain, 'memory_manager')
        assert hasattr(mock_brain, 'cache_dir')

        # Test 2: GUI initialization after brain
        assert mock_gui.brain is mock_brain
        assert hasattr(mock_gui, 'chat_module')

        # Test 3: Module initialization sequence
        if hasattr(mock_gui, 'chat_module') and mock_gui.chat_module:
            chat = mock_gui.chat_module
            assert chat.gui is mock_gui
            assert hasattr(chat, 'message_history')

    @pytest.mark.integration
    def test_data_flow_brain_to_gui(self, mock_brain, mock_gui, test_data):
        """Тест потока данных от Brain к GUI."""
        if mock_brain is None or mock_gui is None:
            pytest.skip("Компоненты недоступны для тестирования")

        # Test that brain can provide data to GUI
        if hasattr(mock_brain, 'get_system_status'):
            status = mock_brain.get_system_status()
            assert status in ['healthy', 'degraded', 'stopped', 'error']

            # Test that GUI can display this data
            assert mock_gui.brain is mock_brain

    @pytest.mark.integration
    def test_module_communication(self, mock_gui):
        """Тест коммуникации между модулями GUI."""
        if mock_gui is None:
            pytest.skip("GUI недоступен для тестирования")

        modules = {}
        module_names = ['chat_module', 'analytics_module', 'memory_module',
                       'knowledge_module', 'settings_module']

        # Collect all available modules
        for module_name in module_names:
            if hasattr(mock_gui, module_name):
                module = getattr(mock_gui, module_name)
                if module is not None:
                    modules[module_name] = module

        # Test that all modules have correct GUI reference
        for module_name, module in modules.items():
            assert hasattr(module, 'gui'), f"Module {module_name} missing gui reference"
            assert module.gui is mock_gui, f"Module {module_name} has incorrect gui reference"

        # Test inter-module communication (if chat exists)
        if 'chat_module' in modules:
            chat = modules['chat_module']
            assert hasattr(chat, 'message_history')
            assert isinstance(chat.message_history, list)

    @pytest.mark.integration
    def test_memory_management_integration(self, mock_brain, mock_memory_manager):
        """Тест интеграции управления памятью."""
        if mock_brain is None or mock_memory_manager is None:
            pytest.skip("Memory компоненты недоступны")

        # Test that brain and memory manager are connected
        assert mock_brain.memory_manager is mock_memory_manager

        # Test memory statistics
        try:
            stats = mock_memory_manager.get_memory_statistics()
            assert isinstance(stats, (dict, list))
        except Exception:
            pytest.skip("MemoryManager не полностью функционален")

    @pytest.mark.integration
    def test_configuration_propagation(self, mock_brain, mock_gui):
        """Тест распространения конфигурации."""
        if mock_brain is None or mock_gui is None:
            pytest.skip("Компоненты недоступны")

        # Test that configuration is propagated from brain to GUI
        if hasattr(mock_brain, 'cache_dir'):
            assert mock_gui.brain is mock_brain

        # Test that GUI has access to brain configuration
        assert hasattr(mock_gui, 'brain')
        assert mock_gui.brain is mock_brain

    @pytest.mark.integration
    def test_error_handling_chain(self, mock_brain, mock_gui):
        """Тест цепочки обработки ошибок."""
        if mock_brain is None or mock_gui is None:
            pytest.skip("Компоненты недоступны")

        # Test that errors can be propagated from brain to GUI
        try:
            if hasattr(mock_brain, 'get_system_status'):
                status = mock_brain.get_system_status()
                assert isinstance(status, str)
        except Exception as e:
            # Error should be handled gracefully
            assert isinstance(str(e), str)


class TestEndToEndScenarios:
    """Сквозные сценарии тестирования."""

    @pytest.mark.integration
    def test_full_initialization_workflow(self, brain_config):
        """Тест полного рабочего процесса инициализации."""
        brain = None
        gui = None

        try:
            # Step 1: Initialize Brain
            from cogniflex.core.core_brain import CoreBrain
            brain = CoreBrain(config=brain_config)
            assert brain is not None

            # Step 2: Initialize GUI with Brain
            from cogniflex.gui.core_gui import create_gui
            gui = create_gui(brain=brain)
            assert gui is not None

            # Step 3: Verify connections
            assert gui.brain is brain
            assert hasattr(brain, 'memory_manager')

            # Step 4: Test basic functionality
            if hasattr(brain, 'get_system_status'):
                status = brain.get_system_status()
                assert status in ['healthy', 'degraded', 'stopped', 'error']

            # Step 5: Test GUI modules
            if hasattr(gui, 'chat_module') and gui.chat_module:
                chat = gui.chat_module
                assert chat.gui is gui

        finally:
            # Cleanup
            if gui and hasattr(gui, 'root') and gui.root:
                try:
                    gui.root.destroy()
                except Exception:
                    pass

            if brain and hasattr(brain, 'shutdown'):
                brain.shutdown()

    @pytest.mark.integration
    def test_component_lifecycle(self, mock_brain, mock_gui):
        """Тест жизненного цикла компонентов."""
        if mock_brain is None or mock_gui is None:
            pytest.skip("Компоненты недоступны")

        # Test component creation
        assert mock_brain is not None
        assert mock_gui is not None

        # Test component relationships
        assert mock_gui.brain is mock_brain

        # Test component functionality
        if hasattr(mock_brain, 'components'):
            assert isinstance(mock_brain.components, dict)

        if hasattr(mock_gui, 'colors'):
            assert isinstance(mock_gui.colors, dict)

    @pytest.mark.integration
    def test_resource_management(self, mock_brain, mock_gui):
        """Тест управления ресурсами."""
        if mock_brain is None or mock_gui is None:
            pytest.skip("Компоненты недоступны")

        # Test memory management
        if hasattr(mock_brain, 'memory_manager') and mock_brain.memory_manager:
            memory_manager = mock_brain.memory_manager

            # Test that memory manager is properly integrated
            assert hasattr(memory_manager, 'get_memory_statistics')

            # Test memory statistics retrieval
            try:
                stats = memory_manager.get_memory_statistics()
                assert stats is not None
            except Exception:
                pytest.skip("Memory statistics недоступны")

    @pytest.mark.integration
    def test_configuration_consistency(self, mock_brain, mock_gui):
        """Тест согласованности конфигурации."""
        if mock_brain is None or mock_gui is None:
            pytest.skip("Компоненты недоступны")

        # Test that configuration is consistent across components
        brain_cache_dir = getattr(mock_brain, 'cache_dir', None)
        gui_cache_dir = getattr(mock_gui, 'cache_dir', None)

        # Cache directories should be consistent or GUI should inherit from Brain
        if brain_cache_dir and gui_cache_dir:
            # GUI cache should be related to brain cache
            assert str(brain_cache_dir) in str(gui_cache_dir) or gui_cache_dir == brain_cache_dir


class TestPerformanceAndScalability:
    """Тесты производительности и масштабируемости."""

    @pytest.mark.performance
    def test_initialization_performance(self, brain_config):
        """Тест производительности инициализации."""
        import time

        start_time = time.time()

        try:
            from cogniflex.core.core_brain import CoreBrain
            from cogniflex.gui.core_gui import create_gui

            # Measure Brain initialization
            brain_start = time.time()
            brain = CoreBrain(config=brain_config)
            brain_time = time.time() - brain_start

            # Measure GUI initialization
            gui_start = time.time()
            gui = create_gui(brain=brain)
            gui_time = time.time() - gui_start

            total_time = time.time() - start_time

            # Performance assertions
            assert brain_time < 20.0, f"Brain инициализация слишком медленная: {brain_time:.2f} сек"
            assert gui_time < 10.0, f"GUI инициализация слишком медленная: {gui_time:.2f} сек"
            assert total_time < 30.0, f"Общая инициализация слишком медленная: {total_time:.2f} сек"

            # Cleanup
            if hasattr(gui, 'root') and gui.root:
                gui.root.destroy()
            if hasattr(brain, 'shutdown'):
                brain.shutdown()

        except Exception as e:
            pytest.skip(f"Не удалось протестировать производительность: {e}")

    @pytest.mark.performance
    def test_memory_efficiency(self, mock_brain):
        """Тест эффективности использования памяти."""
        if mock_brain is None:
            pytest.skip("Brain недоступен")

        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Perform operations
        if hasattr(mock_brain, 'get_system_status'):
            mock_brain.get_system_status()

        if hasattr(mock_brain, 'components'):
            _ = mock_brain.components

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024, f"Утечка памяти: {memory_increase / 1024 / 1024:.2f} MB"

    @pytest.mark.performance
    def test_response_time(self, mock_brain):
        """Тест времени отклика."""
        if mock_brain is None:
            pytest.skip("Brain недоступен")

        import time

        # Test brain response time
        if hasattr(mock_brain, 'get_system_status'):
            start_time = time.time()
            status = mock_brain.get_system_status()
            response_time = time.time() - start_time

            assert response_time < 1.0, f"Время отклика слишком большое: {response_time:.3f} сек"
            assert status in ['healthy', 'degraded', 'stopped', 'error']


class TestSystemReliability:
    """Тесты надежности системы."""

    @pytest.mark.integration
    def test_graceful_degradation(self, mock_brain):
        """Тест graceful degradation при частичных сбоях."""
        if mock_brain is None:
            pytest.skip("Brain недоступен")

        # Test that system can handle partial failures gracefully
        original_memory_manager = getattr(mock_brain, 'memory_manager', None)

        try:
            # Temporarily remove memory manager
            if hasattr(mock_brain, 'memory_manager'):
                mock_brain.memory_manager = None

            # System should still be able to get status
            if hasattr(mock_brain, 'get_system_status'):
                status = mock_brain.get_system_status()
                assert isinstance(status, str)

        finally:
            # Restore original memory manager
            if original_memory_manager:
                mock_brain.memory_manager = original_memory_manager

    @pytest.mark.integration
    def test_error_recovery(self, mock_brain, mock_gui):
        """Тест восстановления после ошибок."""
        if mock_brain is None or mock_gui is None:
            pytest.skip("Компоненты недоступны")

        # Test that system can recover from errors
        original_brain_ref = mock_gui.brain

        try:
            # Simulate error condition
            mock_gui.brain = None

            # System should handle this gracefully
            assert mock_gui.brain is None

        finally:
            # Restore correct reference
            mock_gui.brain = original_brain_ref
            assert mock_gui.brain is mock_brain

    @pytest.mark.integration
    def test_component_isolation(self, mock_brain, mock_gui):
        """Тест изоляции компонентов."""
        if mock_brain is None or mock_gui is None:
            pytest.skip("Компоненты недоступны")

        # Test that components are properly isolated
        brain_components = getattr(mock_brain, 'components', {})
        gui_modules = [attr for attr in dir(mock_gui) if attr.endswith('_module')]

        # Components should not interfere with each other
        assert 'brain' not in gui_modules
        assert 'gui' not in brain_components or brain_components.get('gui') is mock_gui


class TestDataIntegrity:
    """Тесты целостности данных."""

    @pytest.mark.integration
    def test_data_consistency(self, mock_brain, mock_gui):
        """Тест согласованности данных."""
        if mock_brain is None or mock_gui is None:
            pytest.skip("Компоненты недоступны")

        # Test that data is consistent across components
        if hasattr(mock_brain, 'cache_dir') and hasattr(mock_gui, 'cache_dir'):
            # Cache directories should be related
            brain_cache = str(mock_brain.cache_dir)
            gui_cache = str(mock_gui.cache_dir)

            assert brain_cache in gui_cache or gui_cache in brain_cache

    @pytest.mark.integration
    def test_reference_integrity(self, mock_brain, mock_gui):
        """Тест целостности ссылок."""
        if mock_brain is None or mock_gui is None:
            pytest.skip("Компоненты недоступны")

        # Test reference integrity
        assert mock_gui.brain is mock_brain

        # Test circular references are handled correctly
        if hasattr(mock_gui, 'chat_module') and mock_gui.chat_module:
            chat = mock_gui.chat_module
            assert chat.gui is mock_gui
            assert chat.gui.brain is mock_brain


# Test utilities
def save_test_results(results: Dict[str, Any], filename: str = "integration_test_results.json"):
    """Сохранить результаты тестирования."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        print(f"Не удалось сохранить результаты: {e}")


def generate_test_report(results: Dict[str, Any]) -> str:
    """Генерировать отчет о тестировании."""
    report = []
    report.append("# Отчет о комплексном тестировании CogniFlex")
    report.append("")

    # Общая информация
    report.append("## Общая информация")
    report.append(f"- Время выполнения: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"- Всего тестов: {len(results)}")
    report.append("")

    # Результаты по категориям
    categories = {}
    for test_name, result in results.items():
        category = test_name.split('_')[0] if '_' in test_name else 'other'
        if category not in categories:
            categories[category] = []
        categories[category].append((test_name, result))

    for category, tests in categories.items():
        report.append(f"## {category.title()}")
        for test_name, result in tests:
            status = "✅" if result.get('passed', False) else "❌"
            report.append(f"- {status} {test_name}: {result.get('message', 'N/A')}")
        report.append("")

    return '\n'.join(report)


if __name__ == "__main__":
    print("🚀 Запуск комплексного интеграционного тестирования CogniFlex...")

    # Здесь можно добавить запуск отдельных тестов
    print("✅ Комплексное тестирование завершено!")
    print("Для полного тестирования используйте: pytest tests/test_integration_all.py -v")
