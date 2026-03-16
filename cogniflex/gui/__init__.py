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
    from .integrated_gui import IntegratedCogniFlexGUI, create_integrated_gui
    from .core_gui import CogniFlexGUI
    
    # Псевдоним для совместимости
    MainWindow = CogniFlexGUI
    
except ImportError as e:
    logging.warning(f"Не удалось импортировать компоненты GUI: {e}")
    
    # Заглушка на случай ошибки импорта
    class MainWindow:
        def __init__(self, *args, **kwargs):
            logging.warning("Используется заглушка MainWindow из-за ошибки импорта")

# Экспорт для обратной совместимости
__all__ = [
    'IntegratedCogniFlexGUI',
    'create_integrated_gui',
    'CogniFlexGUI',
    'MainWindow',
    'create_gui'
]
