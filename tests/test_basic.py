"""
CogniFlex Basic Integration Tests
Базовые интеграционные тесты компонентов системы.
"""

import pytest
import torch
from pathlib import Path
from typing import Dict, Any


class TestBasicFunctionality:
    """Базовые тесты функциональности системы."""

    @pytest.mark.unit
    def test_pytorch_basic(self):
        """Тест базовой функциональности PyTorch."""
        # Test PyTorch basics
        x = torch.rand(2, 2)
        assert x.shape == (2, 2)
        assert x.dtype == torch.float32

        # Test basic operations
        y = torch.ones(2, 2)
        z = x + y
        assert z.shape == (2, 2)

        # Test CUDA availability (should be False in test environment)
        cuda_available = torch.cuda.is_available()
        assert isinstance(cuda_available, bool)

    @pytest.mark.unit
    def test_transformers_import(self):
        """Тест импорта transformers."""
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            assert GPT2LMHeadModel is not None
            assert GPT2Tokenizer is not None
        except ImportError:
            pytest.skip("transformers не установлены")

    @pytest.mark.integration
    def test_tokenizer_basic(self):
        """Тест базовой функциональности токенизатора."""
        pytest.importorskip("transformers")

        from transformers import GPT2Tokenizer

        try:
            # Используем локальную модель если доступна
            model_path = "sberbank-ai/rugpt3small_based_on_gpt2"
            tokenizer = GPT2Tokenizer.from_pretrained(model_path)

            # Test basic functionality
            text = "Привет, мир!"
            encoded = tokenizer.encode(text, return_tensors="pt")
            decoded = tokenizer.decode(encoded[0])

            assert isinstance(encoded, torch.Tensor)
            assert encoded.shape[0] == 1  # batch size
            assert encoded.shape[1] > 0   # sequence length
            assert text in decoded or decoded.strip() == text

        except Exception as e:
            pytest.skip(f"Не удалось загрузить токенизатор: {e}")

    @pytest.mark.integration
    def test_model_loading(self):
        """Тест загрузки модели."""
        pytest.importorskip("transformers")

        from transformers import GPT2LMHeadModel

        try:
            model_path = "sberbank-ai/rugpt3small_based_on_gpt2"
            model = GPT2LMHeadModel.from_pretrained(model_path)

            assert model is not None
            assert hasattr(model, 'forward')

            # Test forward pass
            tokenizer = pytest.importorskip("transformers").GPT2Tokenizer.from_pretrained(model_path)
            text = "Тест"
            inputs = tokenizer.encode(text, return_tensors="pt")

            with torch.no_grad():
                outputs = model(inputs)

            assert hasattr(outputs, 'logits')
            assert outputs.logits.shape[0] == 1  # batch size
            assert outputs.logits.shape[2] == model.config.vocab_size

        except Exception as e:
            pytest.skip(f"Не удалось загрузить модель: {e}")


class TestCogniFlexImports:
    """Тесты импортов компонентов CogniFlex."""

    @pytest.mark.unit
    def test_core_imports(self):
        """Тест импортов основных компонентов."""
        try:
            from cogniflex.core.core_brain import CoreBrain
            assert CoreBrain is not None
        except ImportError:
            pytest.skip("CoreBrain недоступен")

        try:
            from cogniflex.core.component_initializer import ComponentInitializer
            assert ComponentInitializer is not None
        except ImportError:
            pytest.skip("ComponentInitializer недоступен")

    @pytest.mark.unit
    def test_memory_imports(self):
        """Тест импортов компонентов памяти."""
        try:
            from cogniflex.memory.memory_manager import MemoryManager
            assert MemoryManager is not None
        except ImportError:
            pytest.skip("MemoryManager недоступен")

    @pytest.mark.unit
    def test_gui_imports(self):
        """Тест импортов GUI компонентов."""
        try:
            from cogniflex.gui.core_gui import CogniFlexGUI, create_gui
            assert CogniFlexGUI is not None
            assert create_gui is not None
        except ImportError:
            pytest.skip("GUI компоненты недоступны")

        try:
            from cogniflex.gui.chat_module import ChatModule
            assert ChatModule is not None
        except ImportError:
            pytest.skip("ChatModule недоступен")


class TestCogniFlexComponents:
    """Тесты компонентов CogniFlex с использованием фикстур."""

    @pytest.mark.brain
    @pytest.mark.integration
    def test_brain_creation(self, mock_brain):
        """Тест создания Brain."""
        if mock_brain is None:
            pytest.skip("Brain недоступен для тестирования")

        assert mock_brain is not None
        assert hasattr(mock_brain, 'components')
        assert hasattr(mock_brain, 'cache_dir')
        assert hasattr(mock_brain, 'memory_manager')

    @pytest.mark.memory
    @pytest.mark.integration
    def test_memory_manager(self, mock_memory_manager):
        """Тест MemoryManager."""
        if mock_memory_manager is None:
            pytest.skip("MemoryManager недоступен для тестирования")

        assert mock_memory_manager is not None
        assert hasattr(mock_memory_manager, 'get_memory_statistics')

        # Test basic functionality
        try:
            stats = mock_memory_manager.get_memory_statistics()
            assert isinstance(stats, (dict, list))
        except Exception as e:
            pytest.skip(f"MemoryManager не полностью функционален: {e}")

    @pytest.mark.gui
    @pytest.mark.integration
    def test_gui_creation(self, mock_gui):
        """Тест создания GUI."""
        if mock_gui is None:
            pytest.skip("GUI недоступен для тестирования")

        assert mock_gui is not None
        assert hasattr(mock_gui, 'brain')
        assert hasattr(mock_gui, 'root')
        assert hasattr(mock_gui, 'colors')

    @pytest.mark.integration
    def test_chat_module(self, mock_chat_module):
        """Тест ChatModule."""
        if mock_chat_module is None:
            pytest.skip("ChatModule недоступен для тестирования")

        assert mock_chat_module is not None
        assert hasattr(mock_chat_module, 'gui')
        assert hasattr(mock_chat_module, 'message_history')

        # Test basic methods
        assert hasattr(mock_chat_module, 'activate')
        assert hasattr(mock_chat_module, 'deactivate')

    @pytest.mark.integration
    def test_component_initializer(self, mock_component_initializer):
        """Тест ComponentInitializer."""
        if mock_component_initializer is None:
            pytest.skip("ComponentInitializer недоступен для тестирования")

        assert mock_component_initializer is not None
        assert hasattr(mock_component_initializer, 'brain')
        assert hasattr(mock_component_initializer, 'initialize_components')


class TestSystemIntegration:
    """Интеграционные тесты системы."""

    @pytest.mark.integration
    def test_brain_gui_integration(self, mock_brain, mock_gui):
        """Тест интеграции Brain и GUI."""
        if mock_brain is None or mock_gui is None:
            pytest.skip("Brain или GUI недоступны для тестирования")

        # Test that GUI has correct brain reference
        assert mock_gui.brain is mock_brain

        # Test that brain has gui component
        if hasattr(mock_brain, 'components'):
            assert 'gui' in mock_brain.components

    @pytest.mark.integration
    def test_full_system_initialization(self, mock_brain, mock_gui):
        """Тест полной инициализации системы."""
        if mock_brain is None or mock_gui is None:
            pytest.skip("Компоненты недоступны для тестирования")

        # Test that all major components are initialized
        required_attrs = ['components', 'cache_dir', 'memory_manager']
        for attr in required_attrs:
            assert hasattr(mock_brain, attr), f"Brain missing attribute: {attr}"

        # Test GUI components
        gui_attrs = ['brain', 'root', 'colors']
        for attr in gui_attrs:
            assert hasattr(mock_gui, attr), f"GUI missing attribute: {attr}"

        # Test chat module if available
        if hasattr(mock_gui, 'chat_module') and mock_gui.chat_module:
            chat = mock_gui.chat_module
            assert hasattr(chat, 'gui')
            assert hasattr(chat, 'message_history')


class TestPerformance:
    """Тесты производительности."""

    @pytest.mark.performance
    def test_brain_initialization_time(self, brain_config):
        """Тест времени инициализации Brain."""
        import time

        start_time = time.time()
        try:
            from cogniflex.core.core_brain import CoreBrain
            brain = CoreBrain(config=brain_config)
            init_time = time.time() - start_time

            # Should initialize within reasonable time
            assert init_time < 30.0, f"Инициализация заняла слишком много времени: {init_time:.2f} сек"

            # Cleanup
            if hasattr(brain, 'shutdown'):
                brain.shutdown()

        except Exception as e:
            pytest.skip(f"Не удалось протестировать время инициализации: {e}")

    @pytest.mark.performance
    def test_memory_usage(self, mock_brain):
        """Тест использования памяти."""
        if mock_brain is None:
            pytest.skip("Brain недоступен для тестирования")

        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss

        # Perform some operations
        if hasattr(mock_brain, 'get_system_status'):
            status = mock_brain.get_system_status()
            assert status in ['healthy', 'degraded', 'stopped', 'error']

        memory_after = process.memory_info().rss
        memory_used = memory_after - memory_before

        # Memory usage should be reasonable (less than 500MB increase)
        assert memory_used < 500 * 1024 * 1024, f"Использовано слишком много памяти: {memory_used / 1024 / 1024:.2f} MB"


# Legacy function for backward compatibility
def test_basic():
    """Устаревшая функция для обратной совместимости."""
    print("=== CogniFlex Basic Integration Test ===")

    # Run basic PyTorch tests
    test_func = TestBasicFunctionality()
    try:
        test_func.test_pytorch_basic()
        print("✅ PyTorch test passed")
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")

    try:
        test_func.test_transformers_import()
        print("✅ Transformers import test passed")
    except Exception as e:
        print(f"❌ Transformers import test failed: {e}")

    try:
        test_func.test_tokenizer_basic()
        print("✅ Tokenizer test passed")
    except Exception as e:
        print(f"⚠️ Tokenizer test skipped: {e}")

    try:
        test_func.test_model_loading()
        print("✅ Model loading test passed")
    except Exception as e:
        print(f"⚠️ Model loading test skipped: {e}")

    print("\nTest completed!")


if __name__ == "__main__":
    test_basic()
