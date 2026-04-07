"""
Base GUI Module - Базовый класс для всех GUI компонентов
"""

import logging
from typing import Optional, Dict, Any, Callable
import tkinter as tk
from tkinter import ttk

logger = logging.getLogger("eva_ai.gui.base")


class BaseGUI:
    """Базовый класс для GUI компонентов ЕВА"""
    
    def __init__(self, brain=None, parent=None, **kwargs):
        self.brain = brain
        self.parent = parent
        self.widgets: Dict[str, Any] = {}
        self._initialized = False
        self._callbacks: Dict[str, Callable] = {}
        self._state = "idle"
        logger.info(f"BaseGUI инициализирован с brain={brain is not None}")
    
    def initialize(self):
        """Инициализация GUI компонента"""
        if self._initialized:
            return
        self._setup_ui()
        self._bind_events()
        self._initialized = True
        self._state = "ready"
        logger.info("BaseGUI инициализирован")
    
    def _setup_ui(self):
        """Настройка пользовательского интерфейса - переопределить в подклассах"""
        pass
    
    def _bind_events(self):
        """Привязка событий - переопределить в подклассах"""
        pass
    
    def show(self):
        """Отображение компонента"""
        self._state = "visible"
        logger.debug("BaseGUI отображен")
    
    def hide(self):
        """Скрытие компонента"""
        self._state = "hidden"
        logger.debug("BaseGUI скрыт")
    
    def destroy(self):
        """Уничтожение компонента"""
        self._state = "destroyed"
        self.widgets.clear()
        self._callbacks.clear()
        logger.info("BaseGUI уничтожен")
    
    def register_callback(self, event: str, callback: Callable):
        """Регистрация обратного вызова"""
        self._callbacks[event] = callback
        logger.debug(f"Зарегистрирован callback для события: {event}")
    
    def trigger_callback(self, event: str, *args, **kwargs):
        """Вызов обратного вызова"""
        if event in self._callbacks:
            self._callbacks[event](*args, **kwargs)
    
    def get_state(self) -> str:
        """Получение состояния компонента"""
        return self._state
    
    def set_enabled(self, enabled: bool):
        """Включение/отключение компонента"""
        for widget in self.widgets.values():
            if hasattr(widget, 'configure'):
                widget.configure(state=tk.NORMAL if enabled else tk.DISABLED)
        logger.debug(f"BaseGUI {'включен' if enabled else 'отключен'}")
    
    def refresh(self):
        """Обновление компонента"""
        logger.debug("BaseGUI обновлен")
    
    def update_theme(self, theme_name: str):
        """Обновление темы - переопределить в подклассах"""
        logger.info(f"Обновление темы: {theme_name}")


__all__ = ['BaseGUI']