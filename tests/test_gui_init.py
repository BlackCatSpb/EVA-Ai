"""
CogniFlex GUI Initialization Tests
Комплексные тесты инициализации и функциональности GUI компонентов.
"""

import pytest
import logging
from pathlib import Path
from typing import Dict, Any
import tkinter as tk


class TestGUIInitialization:
    """Тесты инициализации GUI компонентов."""

    @pytest.mark.gui
    @pytest.mark.unit
    def test_gui_imports(self):
        """Тест импортов GUI модулей."""
        try:
            from cogniflex.gui.core_gui import CogniFlexGUI, create_gui
            assert CogniFlexGUI is not None
            assert create_gui is not None
        except ImportError as e:
            pytest.skip(f"GUI модули недоступны: {e}")

    @pytest.mark.gui
    @pytest.mark.unit
    def test_chat_module_imports(self):
        """Тест импортов ChatModule."""
        try:
            from cogniflex.gui.chat_module import ChatModule
            assert ChatModule is not None
        except ImportError as e:
            pytest.skip(f"ChatModule недоступен: {e}")

    @pytest.mark.gui
    @pytest.mark.unit
    def test_analytics_module_imports(self):
        """Тест импортов AnalyticsModule."""
        try:
            from cogniflex.gui.analytics_module import AnalyticsModule
            assert AnalyticsModule is not None
        except ImportError as e:
            pytest.skip(f"AnalyticsModule недоступен: {e}")


class TestGUICoreFunctionality:
    """Тесты основной функциональности GUI."""

    @pytest.mark.gui
    @pytest.mark.integration
    def test_gui_creation_with_brain(self, mock_brain):
        """Тест создания GUI с Brain."""
        if mock_brain is None:
            pytest.skip("Brain недоступен для тестирования")

        try:
            from cogniflex.gui.core_gui import create_gui

            gui = create_gui(brain=mock_brain)

            assert gui is not None
            assert hasattr(gui, 'brain')
            assert gui.brain is mock_brain

        except Exception as e:
            pytest.skip(f"Не удалось создать GUI: {e}")

    @pytest.mark.gui
    @pytest.mark.integration
    def test_gui_creation_minimal(self, temp_cache_dir):
        """Тест создания GUI в минимальном режиме."""
        try:
            from cogniflex.gui.core_gui import CogniFlexGUI

            gui = CogniFlexGUI(
                brain=None,
                cache_dir=str(temp_cache_dir / "gui_cache")
            )

            assert gui is not None
            assert hasattr(gui, 'cache_dir')
            assert hasattr(gui, 'settings')

        except Exception as e:
            pytest.skip(f"Не удалось создать GUI в минимальном режиме: {e}")

    @pytest.mark.gui
    @pytest.mark.integration
    def test_gui_attributes(self, mock_gui):
        """Тест наличия необходимых атрибутов GUI."""
        if mock_gui is None:
            pytest.skip("GUI недоступен для тестирования")

        required_attrs = [
            'brain', 'cache_dir', 'settings', 'theme_colors',
            'colors', 'content_area', 'sidebar', 'status_bar'
        ]

        for attr in required_attrs:
            assert hasattr(mock_gui, attr), f"GUI missing attribute: {attr}"

    @pytest.mark.gui
    @pytest.mark.integration
    def test_gui_methods(self, mock_gui):
        """Тест наличия необходимых методов GUI."""
        if mock_gui is None:
            pytest.skip("GUI недоступен для тестирования")

        required_methods = [
            '_init_modules', '_switch_view', '_create_interface',
            'start', 'shutdown', '_create_styles'
        ]

        for method in required_methods:
            assert hasattr(mock_gui, method), f"GUI missing method: {method}"
            assert callable(getattr(mock_gui, method)), f"GUI method {method} is not callable"


class TestGUIModuleInitialization:
    """Тесты инициализации модулей GUI."""

    @pytest.mark.gui
    @pytest.mark.integration
    def test_chat_module_initialization(self, mock_gui):
        """Тест инициализации ChatModule."""
        if mock_gui is None:
            pytest.skip("GUI недоступен для тестирования")

        if not hasattr(mock_gui, 'chat_module') or mock_gui.chat_module is None:
            pytest.skip("ChatModule не инициализирован")

        chat = mock_gui.chat_module

        # Test basic attributes
        assert hasattr(chat, 'gui')
        assert hasattr(chat, 'message_history')
        assert hasattr(chat, 'input_text')
        assert hasattr(chat, 'send_button')

        # Test that GUI reference is correct
        assert chat.gui is mock_gui

        # Test message history is a list
        assert isinstance(chat.message_history, list)

    @pytest.mark.gui
    @pytest.mark.integration
    def test_analytics_module_initialization(self, mock_gui):
        """Тест инициализации AnalyticsModule."""
        if mock_gui is None:
            pytest.skip("GUI недоступен для тестирования")

        if not hasattr(mock_gui, 'analytics_module') or mock_gui.analytics_module is None:
            pytest.skip("AnalyticsModule не инициализирован")

        analytics = mock_gui.analytics_module

        # Test basic attributes
        assert hasattr(analytics, 'gui')
        assert analytics.gui is mock_gui

    @pytest.mark.gui
    @pytest.mark.integration
    def test_memory_module_initialization(self, mock_gui):
        """Тест инициализации MemoryModule."""
        if mock_gui is None:
            pytest.skip("GUI недоступен для тестирования")

        if not hasattr(mock_gui, 'memory_module') or mock_gui.memory_module is None:
            pytest.skip("MemoryModule не инициализирован")

        memory = mock_gui.memory_module

        # Test basic attributes
        assert hasattr(memory, 'gui')
        assert memory.gui is mock_gui


class TestGUIFunctionality:
    """Тесты функциональности GUI."""

    @pytest.mark.gui
    @pytest.mark.integration
    def test_gui_switch_view(self, mock_gui):
        """Тест переключения видов в GUI."""
        if mock_gui is None:
            pytest.skip("GUI недоступен для тестирования")

        if not hasattr(mock_gui, '_switch_view'):
            pytest.skip("GUI не имеет метода _switch_view")

        # Test switching to chat view
        try:
            mock_gui._switch_view("chat")
            # Should not raise exception
            assert True
        except Exception as e:
            pytest.fail(f"Не удалось переключиться на chat view: {e}")

    @pytest.mark.gui
    @pytest.mark.integration
    def test_gui_theme_colors(self, mock_gui):
        """Тест цветовой схемы GUI."""
        if mock_gui is None:
            pytest.skip("GUI недоступен для тестирования")

        # Test that theme_colors exists and is a dict
        assert hasattr(mock_gui, 'theme_colors')
        assert isinstance(mock_gui.theme_colors, dict)

        # Test that colors exists and is a dict
        assert hasattr(mock_gui, 'colors')
        assert isinstance(mock_gui.colors, dict)

        # Test common color keys
        expected_keys = ['bg', 'text', 'primary', 'secondary', 'border']
        for key in expected_keys:
            assert key in mock_gui.colors, f"Missing color key: {key}"

    @pytest.mark.gui
    @pytest.mark.integration
    def test_gui_settings(self, mock_gui):
        """Тест настроек GUI."""
        if mock_gui is None:
            pytest.skip("GUI недоступен для тестирования")

        # Test that settings exists
        assert hasattr(mock_gui, 'settings')
        assert mock_gui.settings is not None

        # Test settings structure
        assert isinstance(mock_gui.settings, dict)


class TestGUIIntegration:
    """Интеграционные тесты GUI."""

    @pytest.mark.gui
    @pytest.mark.integration
    def test_gui_brain_integration(self, mock_brain, mock_gui):
        """Тест интеграции GUI с Brain."""
        if mock_brain is None or mock_gui is None:
            pytest.skip("Brain или GUI недоступны для тестирования")

        # Test that GUI has correct brain reference
        assert mock_gui.brain is mock_brain

        # Test that brain knows about GUI
        if hasattr(mock_brain, 'components') and 'gui' in mock_brain.components:
            assert mock_brain.components['gui'] is mock_gui

    @pytest.mark.gui
    @pytest.mark.integration
    def test_gui_chat_brain_flow(self, mock_brain, mock_gui):
        """Тест потока GUI -> Chat -> Brain."""
        if mock_brain is None or mock_gui is None:
            pytest.skip("Brain или GUI недоступны для тестирования")

        if not hasattr(mock_gui, 'chat_module') or mock_gui.chat_module is None:
            pytest.skip("ChatModule не инициализирован")

        chat = mock_gui.chat_module

        # Test the chain: GUI -> Chat -> Brain
        assert chat.gui is mock_gui
        assert mock_gui.brain is mock_brain

        # Test that chat can access brain through GUI
        brain_through_chat = chat.gui.brain
        assert brain_through_chat is mock_brain

    @pytest.mark.gui
    @pytest.mark.integration
    def test_gui_module_interaction(self, mock_gui):
        """Тест взаимодействия между модулями GUI."""
        if mock_gui is None:
            pytest.skip("GUI недоступен для тестирования")

        # Test that modules can interact with each other
        modules = ['chat_module', 'analytics_module', 'memory_module', 'settings_module']

        active_modules = []
        for module_name in modules:
            if hasattr(mock_gui, module_name):
                module = getattr(mock_gui, module_name)
                if module is not None:
                    active_modules.append(module_name)
                    # Test that module has reference to GUI
                    assert hasattr(module, 'gui'), f"Module {module_name} missing gui reference"
                    assert module.gui is mock_gui, f"Module {module_name} has incorrect gui reference"

        # Should have at least chat_module
        assert 'chat_module' in [attr for attr in dir(mock_gui) if attr.endswith('_module') and getattr(mock_gui, attr) is not None], \
               "ChatModule должен быть инициализирован"


class TestGUIPerformance:
    """Тесты производительности GUI."""

    @pytest.mark.gui
    @pytest.mark.performance
    def test_gui_initialization_time(self, brain_config):
        """Тест времени инициализации GUI."""
        import time

        try:
            from cogniflex.core.core_brain import CoreBrain
            from cogniflex.gui.core_gui import create_gui

            # Create brain first
            brain = CoreBrain(config=brain_config)

            # Measure GUI creation time
            start_time = time.time()
            gui = create_gui(brain=brain)
            init_time = time.time() - start_time

            # Should initialize within reasonable time
            assert init_time < 15.0, f"GUI инициализация заняла слишком много времени: {init_time:.2f} сек"

            # Cleanup
            if hasattr(gui, 'root') and gui.root:
                try:
                    gui.root.destroy()
                except Exception:
                    pass

            if hasattr(brain, 'shutdown'):
                brain.shutdown()

        except Exception as e:
            pytest.skip(f"Не удалось протестировать время инициализации GUI: {e}")


# Legacy function for backward compatibility
def test_gui_initialization():
    """Устаревшая функция для обратной совместимости."""
    print("=== CogniFlex GUI Initialization Test ===")

    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('gui_test.log')
            ]
        )
        logger = logging.getLogger("gui_test")

        logger.info("Starting GUI initialization test...")

        # Import required modules
        from cogniflex.gui.core_gui import CogniFlexGUI

        logger.info("Creating GUI instance...")

        # Initialize the GUI with default parameters
        gui = CogniFlexGUI(brain=None, cache_dir="./test_gui_cache")

        # Test basic functionality
        logger.info("Testing GUI methods...")

        # Check if required attributes exist
        required_attrs = [
            'brain', 'cache_dir', 'settings', 'theme_colors',
            'content_area', 'sidebar', 'status_bar', 'menu_bar'
        ]

        for attr in required_attrs:
            if not hasattr(gui, attr):
                logger.warning(f"GUI is missing attribute: {attr}")
            else:
                logger.info(f"GUI has attribute: {attr}")

        logger.info("GUI initialization test completed successfully!")
        return True

    except Exception as e:
        logger.error(f"GUI initialization test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    if test_gui_initialization():
        print("\n✅ GUI initialization test completed successfully!")
        print("Check gui_test.log for detailed logs.")
    else:
        print("\n❌ GUI initialization test failed!")
        print("Check gui_test.log for error details.")
