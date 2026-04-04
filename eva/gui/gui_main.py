"""
Main window, layout, and tab management for ЕВА GUI.
"""
import os
import logging
import threading
import queue
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional, List

import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use('TkAgg')

from .settings import load_settings, save_settings

logger = logging.getLogger("eva.gui.core")

ALLOWED_MODULE_PATHS = frozenset([
    "eva.gui.chat_module",
    "eva.gui.memory_module",
])

def _validate_module_path(module_path: str) -> bool:
    if not module_path or ".." in module_path or module_path.startswith("/"):
        return False
    return module_path in ALLOWED_MODULE_PATHS


class MainWindowMixin:
    """Mixin for main window creation, layout, and lifecycle management."""

    def create_main_window(self):
        """Создает главное окно приложения."""
        if self.root:
            return
            
        self.root = tk.Tk()
        self.root.title("ЕВА - Cognitive AI System")
        self.root.geometry("1600x1000")
        self.root.configure(bg=self.colors['bg'])
        
        self.content_area = ttk.Frame(self.root)
        self.content_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        logger.info("Главное окно GUI создано")

    def _create_styles(self):
        style = ttk.Style()
        self.theme = self.settings.get("gui", {}).get("theme", "light")
        self.colors = self.theme_colors[self.theme]

        if self.theme == "dark":
            style.theme_use("clam")
            style.configure(".", background=self.colors['bg'], foreground=self.colors['text'], fieldbackground=self.colors['card-bg'], bordercolor=self.colors['border'])
            style.configure("TFrame", background=self.colors['bg'])
            style.configure("TLabel", background=self.colors['bg'], foreground=self.colors['text'])
            style.configure("TButton", background=self.colors['card-bg'], foreground=self.colors['text'], borderwidth=1)
            style.map("TButton", background=[("active", self.colors['primary'])])
        else:
            style.theme_use("default")
            style.configure(".", background=self.colors['bg'], foreground=self.colors['text'], fieldbackground=self.colors['card-bg'])

        style.configure("Nav.TButton", padding=(10, 5), relief=tk.FLAT)
        style.configure("NavActive.TButton", padding=(10, 5), relief=tk.FLAT, background=self.colors['primary'])
        style.map("NavActive.TButton", background=[("active", self.colors['primary'])])
        
        style.configure("TNotebook", background=self.colors['bg'])
        style.configure("TNotebook.Tab", padding=(10, 5), background=self.colors['card-bg'], foreground=self.colors['text'])
        style.map("TNotebook.Tab", background=[("selected", self.colors['primary'])], foreground=[("selected", "white")])

        if self.root:
            self.root.configure(bg=self.colors['bg'])

    def _create_interface(self):
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        self._create_navbar()
        self._create_notebook()
        self._create_status_bar()

    def _create_navbar(self):
        """Создаёт упрощённую панель навигации."""
        navbar = ttk.Frame(self.main_container, height=40)
        navbar.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(navbar, text="ЕВА", font=("Segoe UI", 14, "bold"), foreground=self.colors['primary']).pack(side=tk.LEFT, padx=10)
        right_frame = ttk.Frame(navbar)
        right_frame.pack(side=tk.RIGHT)
        ttk.Button(right_frame, text="Перезагрузить", command=self._reboot_system).pack(side=tk.LEFT, padx=5)
        ttk.Button(right_frame, text="Горячая перезагрузка", command=self._soft_reload).pack(side=tk.LEFT, padx=5)

    def _create_notebook(self):
        """Создаёт Notebook с 3 вкладками: Чат, Память, Система."""
        self.notebook = ttk.Notebook(self.main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self.notebook.bind('<<NotebookTabChanged>>', self._on_tab_changed)
        
        self.tabs = {}
        tab_names = [
            ("chat", "Чат"),
            ("memory", "Память"),
            ("system", "Система")
        ]
        
        for tab_id, tab_title in tab_names:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=tab_title)
            self.tabs[tab_id] = frame
        
        self.content_area = self.tabs.get("chat")
        self.tab_order = [tab_id for tab_id, _ in tab_names]

    def start_gui(self) -> None:
        """Запуск GUI для CoreBrain."""
        if not self.running:
            self.chat_logger.info("Запуск GUI после полной инициализации системы...")
            self.start()
        else:
            self.chat_logger.info("GUI уже запущен")
    
    def start(self):
        if self.running: 
            self.chat_logger.warning("Попытка запуска GUI, когда он уже запущен")
            return
            
        self.running = True
        self.chat_logger.info("Запуск графического интерфейса...")
        
        try:
            if not hasattr(self, 'analytics_module') or self.analytics_module is None:
                self._init_modules()
            
            self._load_state()
            self._start_background_services()
            
            if self.root is not None:
                self.root.protocol("WM_DELETE_WINDOW", self.on_close)
            
            self.chat_logger.info("Графический интерфейс успешно запущен")
            if self.root is not None:
                self.chat_logger.info(f"Разрешение окна: {self.root.winfo_width()}x{self.root.winfo_height()}")
            else:
                self.chat_logger.warning("Окно root не создано, пропускаем логирование размера")
            
            if self.root is not None:
                self.root.mainloop()
        except Exception as e:
            self.chat_logger.critical(f"Критическая ошибка запуска GUI: {e}", exc_info=True)

    def stop(self):
        if not self.running: 
            self.chat_logger.warning("Попытка остановки GUI, когда он не запущен")
            return
            
        self.running = False
        self.chat_logger.info("Остановка графического интерфейса...")
        
        self.stop_event.set()
        if self.update_job and self.root:
            try:
                self.root.after_cancel(self.update_job)
                self.chat_logger.debug("Отменено запланированное обновление интерфейса")
            except tk.TclError:
                pass
            finally:
                self.update_job = None
        if self.gui_queue_job and self.root:
            try:
                self.root.after_cancel(self.gui_queue_job)
                self.chat_logger.debug("Отменена запланированная обработка GUI очереди")
            except tk.TclError:
                pass
            finally:
                self.gui_queue_job = None
            
        self._save_state()
        self._cleanup_modules()
        
        if self.brain: 
            self.brain.stop()
            self.chat_logger.info("Ядро системы остановлено")
            
        if self.root: 
            self.root.destroy()
            self.chat_logger.info("Окно GUI уничтожено")
        
        self.chat_logger.info("Графический интерфейс остановлен")

    def _load_state(self):
        """Загружает состояние GUI."""
        try:
            state_file = os.path.join(self.cache_dir, "gui_state.json")
            if os.path.exists(state_file):
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                
                self.current_view = state.get("current_view", "chat")
                self.theme = state.get("theme", "light")
                self.compact_mode = state.get("compact_mode", False)
                
                self.chat_logger.info(f"Состояние GUI загружено из {state_file}")
                self.chat_logger.debug(f"Загруженное состояние: {state}")
                
                if "window_size" in state and self.root:
                    width = state["window_size"].get("width", 1280)
                    height = state["window_size"].get("height", 800)
                    self.root.geometry(f"{width}x{height}")
                    self.chat_logger.info(f"Восстановлен размер окна: {width}x{height}")
            else:
                self.chat_logger.info("Файл состояния GUI не найден, используется состояние по умолчанию")
                
        except Exception as e:
            self.chat_logger.error(f"Ошибка при загрузке состояния GUI: {str(e)}", exc_info=True)
            self.chat_logger.info("Используется состояние по умолчанию")

    def _save_state(self):
        """Сохраняет состояние GUI."""
        try:
            state = {
                "current_view": self.current_view,
                "theme": self.theme,
                "compact_mode": self.compact_mode,
                "window_size": {
                    "width": self.root.winfo_width() if self.root else 1280,
                    "height": self.root.winfo_height() if self.root else 800
                },
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            state_file = os.path.join(self.cache_dir, "gui_state.json")
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
                
            self.chat_logger.info(f"Состояние GUI сохранено в {state_file}")
            self.chat_logger.debug(f"Сохраненное состояние: {state}")
            
        except Exception as e:
            self.chat_logger.error(f"Ошибка при сохранении состояния GUI: {str(e)}", exc_info=True)

    def reload(self):
        """Горячая перезагрузка GUI."""
        try:
            if not self.root or not self.running:
                return
            self.chat_logger.info("Выполняется горячая перезагрузка GUI...")
            try:
                self.settings = load_settings(os.path.join(self.cache_dir, "gui_settings.json"))
            except Exception:
                pass
            self.theme = self.settings.get("gui", {}).get("theme", self.theme)
            self.colors = self.theme_colors.get(self.theme, self.colors)
            try:
                self._create_styles()
            except Exception:
                pass
            cur_view = getattr(self, 'current_view', 'chat')
            try:
                if self.content_area:
                    for w in self.content_area.winfo_children():
                        w.destroy()
                self._init_modules()
                self._switch_view(cur_view)
            except Exception:
                pass
            self._update_interface()
            self.show_toast("GUI обновлён (soft-reload)", "info")
            self.chat_logger.info("Горячая перезагрузка GUI завершена")
        except tk.TclError:
            pass
        except Exception as e:
            self.chat_logger.warning(f"Ошибка горячей перезагрузки GUI: {e}")

        if not self.running: return
        self._update_interface()
        if self.root:
            self.update_job = self.root.after(self.settings.get("gui", {}).get("auto_update_interval", 5000), self._schedule_update)
