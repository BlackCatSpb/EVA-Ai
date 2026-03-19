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
def brain_config(tmp_path):
    """Конфигурация для CoreBrain в тестах"""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return {
        "cache_dir": str(cache_dir),
        "use_gpu": False,
        "max_workers": 1,
        "autoload": False,
        "log_level": "ERROR"
    }


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


@pytest.fixture
def mock_brain():
    """Mock CoreBrain для тестов"""
    class MockBrain:
        def __init__(self):
            self.state = "initializing"
            self.components = {}
            self.cache_dir = "mock_cache"
            self.memory_manager = object()
            self.knowledge_graph = object()
        
        def initialize(self):
            self.state = "ready"
            return True
        
        def process_query(self, query):
            return f"Mock response to: {query}"
        
        def shutdown(self):
            self.state = "shutdown"
    
    brain = MockBrain()
    brain.components["gui"] = True
    return brain


@pytest.fixture
def mock_memory_manager():
    """Mock MemoryManager для тестов"""
    class MockMemoryManager:
        def __init__(self):
            self.memories = []
            self.cache = {}
        
        def add_memory(self, memory):
            self.memories.append(memory)
        
        def get_memory(self, idx):
            return self.memories[idx] if idx < len(self.memories) else None
        
        def clear(self):
            self.memories = []
            self.cache = {}
        
        def get_memory_statistics(self):
            return {"total": len(self.memories), "cache_size": len(self.cache)}
    
    return MockMemoryManager()


@pytest.fixture
def mock_gui(mock_brain, request):
    """Mock GUI для тестов"""
    class MockGUI:
        def __init__(self, brain):
            self.windows = []
            self.visible = False
            self._brain = brain
            self.root = object()
            self.colors = {}
            self.chat_module = None
        
        @property
        def brain(self):
            return self._brain
        
        @brain.setter
        def brain(self, value):
            self._brain = value
        
        def set_brain(self, brain):
            self._brain = brain
        
        def show(self):
            self.visible = True
        
        def hide(self):
            self.visible = False
        
        def destroy(self):
            self.windows = []
            self.visible = False
    
    return MockGUI(mock_brain)


@pytest.fixture
def mock_chat_module():
    """Mock ChatModule для тестов"""
    class MockChatModule:
        def __init__(self):
            self.messages = []
            self.active = False
            self.gui = None
            self.message_history = []
        
        def send_message(self, msg):
            self.messages.append({"role": "user", "content": msg})
        
        def receive_message(self):
            return {"role": "assistant", "content": "Mock response"}
        
        def clear_history(self):
            self.messages = []
        
        def activate(self):
            self.active = True
        
        def deactivate(self):
            self.active = False
    
    return MockChatModule()


@pytest.fixture
def mock_component_initializer():
    """Mock ComponentInitializer для тестов"""
    class MockComponentInitializer:
        def __init__(self):
            self.components = {}
            self.initialized = False
            self.brain = None
        
        def initialize_components(self):
            self.initialized = True
            self.components = {
                "memory": object(),
                "ml": object(),
                "knowledge": object()
            }
        
        def get_component(self, name):
            return self.components.get(name)
    
    return MockComponentInitializer()
