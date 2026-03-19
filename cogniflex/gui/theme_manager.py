"""
Theme Manager Module - Управление темами оформления
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger("cogniflex.gui.theme")


class ThemeManager:
    """Менеджер тем для CogniFlex GUI"""
    
    DEFAULT_THEMES = {
        "default": {
            "name": "Default",
            "bg_primary": "#1e1e1e",
            "bg_secondary": "#2d2d2d",
            "bg_tertiary": "#3d3d3d",
            "fg_primary": "#ffffff",
            "fg_secondary": "#cccccc",
            "fg_tertiary": "#999999",
            "accent": "#007acc",
            "success": "#4caf50",
            "warning": "#ff9800",
            "error": "#f44336",
            "border": "#404040",
        },
        "light": {
            "name": "Light",
            "bg_primary": "#ffffff",
            "bg_secondary": "#f5f5f5",
            "bg_tertiary": "#e0e0e0",
            "fg_primary": "#000000",
            "fg_secondary": "#333333",
            "fg_tertiary": "#666666",
            "accent": "#007acc",
            "success": "#4caf50",
            "warning": "#ff9800",
            "error": "#f44336",
            "border": "#cccccc",
        },
        "ocean": {
            "name": "Ocean",
            "bg_primary": "#0d1b2a",
            "bg_secondary": "#1b263b",
            "bg_tertiary": "#415a77",
            "fg_primary": "#e0e1dd",
            "fg_secondary": "#778da9",
            "fg_tertiary": "#4a6fa5",
            "accent": "#00b4d8",
            "success": "#2ec4b6",
            "warning": "#ff9f1c",
            "error": "#e63946",
            "border": "#1b263b",
        },
    }
    
    def __init__(self):
        self._themes: Dict[str, Dict[str, Any]] = self.DEFAULT_THEMES.copy()
        self._active_theme: str = "default"
        self._custom_colors: Dict[str, str] = {}
        logger.info("ThemeManager инициализирован")
    
    def get_theme(self, theme_name: Optional[str] = None) -> Dict[str, Any]:
        """Получить тему по имени"""
        name = theme_name or self._active_theme
        if name not in self._themes:
            logger.warning(f"Тема '{name}' не найдена, используется default")
            return self._themes["default"]
        theme = self._themes[name].copy()
        theme.update(self._custom_colors)
        return theme
    
    def set_active_theme(self, theme_name: str):
        """Установить активную тему"""
        if theme_name not in self._themes:
            raise ValueError(f"Тема '{theme_name}' не зарегистрирована")
        self._active_theme = theme_name
        logger.info(f"Активная тема: {theme_name}")
    
    def get_active_theme(self) -> str:
        """Получить имя активной темы"""
        return self._active_theme
    
    def register_theme(self, name: str, theme_data: Dict[str, Any]):
        """Зарегистрировать новую тему"""
        self._themes[name] = theme_data
        logger.info(f"Зарегистрирована тема: {name}")
    
    def unregister_theme(self, name: str):
        """Удалить тему"""
        if name in self._themes and name not in self.DEFAULT_THEMES:
            del self._themes[name]
            logger.info(f"Удалена тема: {name}")
        else:
            logger.warning(f"Нельзя удалить встроенную тему: {name}")
    
    def get_available_themes(self) -> List[str]:
        """Получить список доступных тем"""
        return list(self._themes.keys())
    
    def set_custom_color(self, key: str, value: str):
        """Установить пользовательский цвет (переопределяет тему)"""
        self._custom_colors[key] = value
        logger.debug(f"Установлен пользовательский цвет: {key}={value}")
    
    def clear_custom_colors(self):
        """Очистить пользовательские цвета"""
        self._custom_colors.clear()
        logger.debug("Пользовательские цвета очищены")
    
    def apply_theme_to_widget(self, widget, theme_name: Optional[str] = None):
        """Применить тему к виджету"""
        theme = self.get_theme(theme_name)
        if hasattr(widget, 'configure'):
            widget.configure(
                bg=theme.get("bg_primary"),
                fg=theme.get("fg_primary")
            )
        logger.debug(f"Применена тема к виджету: {theme_name or self._active_theme}")


__all__ = ['ThemeManager']