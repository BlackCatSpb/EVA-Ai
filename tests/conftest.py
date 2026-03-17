"""
CogniFlex Test Configuration
Общие фикстуры для тестов
"""
import pytest
import os
import sys

# Добавить путь к проекту
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Временная директория для кэша"""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir()
    return str(cache_dir)


@pytest.fixture
def mock_logger():
    """Mock логгер для тестов"""
    class MockLogger:
        def __init__(self):
            self.messages = []
        
        def debug(self, msg, *args, **kwargs):
            self.messages.append(f"DEBUG: {msg}")
        
        def info(self, msg, *args, **kwargs):
            self.messages.append(f"INFO: {msg}")
        
        def warning(self, msg, *args, **kwargs):
            self.messages.append(f"WARNING: {msg}")
        
        def error(self, msg, *args, **kwargs):
            self.messages.append(f"ERROR: {msg}")
    
    return MockLogger()


@pytest.fixture
def sample_text():
    """Пример текста для тестов"""
    return "Это тестовый текст для проверки функций CogniFlex."


@pytest.fixture
def sample_texts():
    """Список примеров текстов"""
    return [
        "Первый тестовый текст",
        "Второй тестовый текст",
        "Третий тестовый текст"
    ]
