"""
Window Manager Module - Управление окнами приложения
"""

import logging
from typing import Optional, Dict, Any, List
import tkinter as tk
from tkinter import ttk

logger = logging.getLogger("cogniflex.gui.window")


class WindowManager:
    """Менеджер окон для CogniFlex GUI"""
    
    def __init__(self, root: Optional[tk.Tk] = None):
        self.root = root
        self.windows: Dict[str, tk.Toplevel] = {}
        self._active_window: Optional[str] = None
        self._window_configs: Dict[str, Dict[str, Any]] = {}
        logger.info("WindowManager инициализирован")
    
    def create_window(self, window_id: str, title: str = "", 
                      geometry: str = "800x600", **kwargs) -> tk.Toplevel:
        """Создание нового окна"""
        if window_id in self.windows:
            logger.warning(f"Окно '{window_id}' уже существует")
            return self.windows[window_id]
        
        parent = self.root if self.root else kwargs.get('parent')
        window = tk.Toplevel(parent) if parent else tk.Toplevel()
        window.title(title)
        window.geometry(geometry)
        
        self.windows[window_id] = window
        self._window_configs[window_id] = {
            "title": title,
            "geometry": geometry,
            "resizable": kwargs.get("resizable", True),
            "modal": kwargs.get("modal", False)
        }
        
        if kwargs.get("resizable", True) is False:
            window.resizable(False, False)
        
        logger.info(f"Создано окно: {window_id} ({title})")
        return window
    
    def get_window(self, window_id: str) -> Optional[tk.Toplevel]:
        """Получение окна по ID"""
        return self.windows.get(window_id)
    
    def close_window(self, window_id: str):
        """Закрытие окна"""
        if window_id in self.windows:
            window = self.windows[window_id]
            window.destroy()
            del self.windows[window_id]
            if self._active_window == window_id:
                self._active_window = None
            logger.info(f"Закрыто окно: {window_id}")
    
    def show_window(self, window_id: str):
        """Показать окно"""
        if window_id in self.windows:
            window = self.windows[window_id]
            window.deiconify()
            window.lift()
            self._active_window = window_id
            logger.debug(f"Показано окно: {window_id}")
    
    def hide_window(self, window_id: str):
        """Скрыть окно"""
        if window_id in self.windows:
            self.windows[window_id].withdraw()
            logger.debug(f"Скрыто окно: {window_id}")
    
    def set_active_window(self, window_id: str):
        """Установить активное окно"""
        if window_id in self.windows:
            self._active_window = window_id
            self.show_window(window_id)
    
    def get_active_window(self) -> Optional[str]:
        """Получить ID активного окна"""
        return self._active_window
    
    def get_all_windows(self) -> List[str]:
        """Получить список всех окон"""
        return list(self.windows.keys())
    
    def center_window(self, window_id: str):
        """Центрировать окно на экране"""
        if window_id in self.windows:
            window = self.windows[window_id]
            window.update_idletasks()
            width = window.winfo_width()
            height = window.winfo_height()
            x = (window.winfo_screenwidth() // 2) - (width // 2)
            y = (window.winfo_screenheight() // 2) - (height // 2)
            window.geometry(f"{width}x{height}+{x}+{y}")
    
    def close_all(self):
        """Закрыть все окна"""
        for window_id in list(self.windows.keys()):
            self.close_window(window_id)
        logger.info("Все окна закрыты")


__all__ = ['WindowManager']