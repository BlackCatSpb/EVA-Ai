"""
GUI Factory Module - Фабрика для создания GUI компонентов
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("cogniflex.gui.factory")


class GUIFactory:
    """Фабрика для создания GUI компонентов CogniFlex"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._component_registry: Dict[str, type] = {}
        self._default_theme = "default"
        self._register_default_components()
    
    def _register_default_components(self):
        """Регистрация компонентов по умолчанию"""
        from .base_gui import BaseGUI
        from .window_manager import WindowManager
        from .theme_manager import ThemeManager
        self._component_registry["base"] = BaseGUI
        self._component_registry["window"] = WindowManager
        self._component_registry["theme"] = ThemeManager
    
    def register_component(self, name: str, component_class: type):
        """Регистрация нового компонента"""
        self._component_registry[name] = component_class
        logger.info(f"Зарегистрирован компонент: {name}")
    
    def create_component(self, name: str, **kwargs) -> Any:
        """Создание компонента по имени"""
        if name not in self._component_registry:
            raise ValueError(f"Компонент '{name}' не зарегистрирован")
        return self._component_registry[name](**kwargs)
    
    def create_main_window(self, brain=None, **kwargs):
        """Создание главного окна"""
        return self.create_component("base", brain=brain, **kwargs)
    
    def set_theme(self, theme_name: str):
        """Установка темы по умолчанию"""
        self._default_theme = theme_name
        logger.info(f"Установлена тема по умолчанию: {theme_name}")
    
    def get_registered_components(self) -> Dict[str, type]:
        """Получение списка зарегистрированных компонентов"""
        return self._component_registry.copy()


def create_gui_factory() -> GUIFactory:
    """Фабричная функция для получения экземпляра GUIFactory"""
    return GUIFactory()


def get_gui_factory() -> GUIFactory:
    """Получение глобального экземпляра GUIFactory"""
    return GUIFactory()


__all__ = ['GUIFactory', 'create_gui_factory', 'get_gui_factory']