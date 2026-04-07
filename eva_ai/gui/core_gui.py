"""
Графический интерфейс пользователя для ЕВА - основной модуль с полной функциональностью
"""
import os
import sys
import logging
import threading
import queue
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional, List

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib
matplotlib.use('TkAgg')

# Импортируем утилиты и настройки
from .settings import load_settings, save_settings

# Импортируем миксины из новых модулей
from .gui_main import MainWindowMixin
from .gui_tabs import TabManagerMixin, MemoryTab, SystemTab
from .gui_status import StatusBarMixin
from .gui_events import EventHandlerMixin

logger = logging.getLogger("eva_ai.gui.core")


class ЕВАGUI(MainWindowMixin, TabManagerMixin, StatusBarMixin, EventHandlerMixin):
    """Полнофункциональный графический интерфейс для ЕВА с поддержкой всех расширенных функций."""
    
    def __init__(self, brain=None, integrator=None, cache_dir: Optional[str] = None):
        logger.debug("Инициализация графического интерфейса...")
        self.brain = brain
        self.integrator = integrator
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "eva_gui_cache")
        self.cache_dir = os.path.abspath(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.settings = load_settings(os.path.join(self.cache_dir, "gui_settings.json"))
        self.notification_throttle_seconds = self.settings.get("gui", {}).get("notification_throttle_seconds", 30)
        
        self.theme_colors = {
            "light": {
                'bg': '#f0f0f0', 'card-bg': 'white', 'text': '#333333',
                'text-muted': '#666666', 'border': '#cccccc', 'primary': '#0078d7',
                'success': '#28a745', 'danger': '#dc3545', 'warning': '#ffc107',
            },
            "dark": {
                'bg': '#1e1e1e', 'card-bg': '#2d2d2d', 'text': '#e0e0e0',
                'text-muted': '#a0a0a0', 'border': '#444444', 'primary': '#0091ff',
                'success': '#4caf50', 'danger': '#f44336', 'warning': '#ff9800',
            }
        }
        
        self.theme = self.settings.get("gui", {}).get("theme", "light")
        self.colors = self.theme_colors[self.theme]
        
        self.gui_queue = queue.Queue()
        self.current_view = "chat"
        self.system_status = "initializing"
        self.running = False
        self.stop_event = threading.Event()
        self.compact_mode = self.settings.get("gui", {}).get("compact_mode", False)
        self.model_loading_state = {
            "active": False,
            "model_id": None,
            "name": None,
            "progress": 0,
            "error": None,
            "action": "load",
        }
        
        self.dashboard_data = {}
        self.last_notification_times = {}
        self.gui_queue_job = None
        self.active_notifications = []
        
        self.chat_logger = logging.getLogger("eva_ai.gui.chat")
        self.chat_logger.info("Инициализирован чат-логгер")

        self.root = None
        self.content_area = None
        self.update_job = None
        
        self.chat_module = None
        self.analytics_module = None
        self.knowledge_module = None
        self.contradiction_module = None
        self.memory_module = None
        self.learning_module = None
        self.settings_module = None
        self.neuromorphic_module = None
        
        logger.info("GUI инициализирован")


# Backward-compatible alias
CoreGUI = ЕВАGUI


def create_gui(brain=None, cache_dir: str = None, integrator=None):
    """Создает и возвращает экземпляр GUI с подробным логгированием."""
    logger.info("Создание экземпляра GUI")
    gui = ЕВАGUI(brain, integrator=integrator, cache_dir=cache_dir)
    
    try:
        gui.create_main_window()
        logger.info("Главное окно создано в create_gui()")
    except Exception as e:
        logger.error(f"Ошибка создания главного окна: {e}", exc_info=True)
    
    try:
        gui._create_interface()
        gui._init_modules()
        logger.info("Модули GUI инициализированы в create_gui()")
    except Exception as e:
        logger.error(f"Ошибка инициализации модулей GUI: {e}", exc_info=True)
    
    logger.info("Экземпляр GUI создан")
    logger.debug(f"Путь к кэшу GUI: {gui.cache_dir}")
    
    return gui
