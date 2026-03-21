"""
CogniFlex GUI Package

Графический интерфейс CogniFlex - интеграция с единой фрактальной архитектурой
"""

import logging

logger = logging.getLogger("cogniflex.gui")

# Импорт основных компонентов GUI
def create_gui(brain=None, **kwargs):
    """Фабричная функция для создания экземпляра CogniFlexGUI."""
    logger.info("Создание экземпляра CogniFlexGUI через фабричную функцию.")
    from .core_gui import create_gui as _create_gui
    return _create_gui(brain=brain, **kwargs)

try:
    from .core_gui import CogniFlexGUI
    
    from .chat_module import ChatModule
    from .memory_module import MemoryModule
    from .knowledge_graph_module import KnowledgeGraphModule
    from .contradiction_module import ContradictionModule
    from .analytics_module import AnalyticsModule
    from .learning_module import LearningModule
    from .neuromorphic_module import NeuromorphicModule
    from .settings_module import SettingsModule
    from .base_gui import BaseGUI
    
    MainWindow = CogniFlexGUI
    
except ImportError as e:
    logging.warning(f"Не удалось импортировать компоненты GUI: {e}")
    
    class MainWindow:
        def __init__(self, *args, **kwargs):
            logging.warning("Используется заглушка MainWindow из-за ошибки импорта")

__all__ = [
    'CogniFlexGUI',
    'MainWindow',
    'create_gui',
    'ChatModule',
    'MemoryModule',
    'KnowledgeGraphModule',
    'ContradictionModule',
    'AnalyticsModule',
    'LearningModule',
    'NeuromorphicModule',
    'SettingsModule',
    'BaseGUI',
]
