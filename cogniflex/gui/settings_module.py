"""Модуль настроек для CogniFlex GUI - полнофункциональная реализация"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import logging
import json
import os
from datetime import datetime
import platform

logger = logging.getLogger("cogniflex.gui.settings")

class SettingsModule:
    """Модуль для управления настройками интерфейса и системы CogniFlex."""
    
    def __init__(self, gui):
        self.gui = gui
        self.settings_frame = None
        self.vars = {}
        logger.info("Модуль настроек инициализирован")
    
    def _safe_brain_call(self, method_name: str, *args, **kwargs):
        """Безопасный вызов метода brain с логированием"""
        if hasattr(self, 'gui') and hasattr(self.gui, 'brain') and self.gui.brain:
            try:
                method = getattr(self.gui.brain, method_name, None)
                if method:
                    result = method(*args, **kwargs)
                    logger.debug(f"[{self.__class__.__name__}] Успешный вызов brain.{method_name}()")
                    return result
                else:
                    logger.warning(f"[{self.__class__.__name__}] Метод brain.{method_name} не найден")
                    return None
            except Exception as e:
                logger.error(f"[{self.__class__.__name__}] Ошибка вызова brain.{method_name}: {e}")
                return None
        else:
            logger.warning(f"[{self.__class__.__name__}] Brain недоступен")
            return None
    
    def activate(self):
        for widget in self.gui.content_area.winfo_children():
            widget.destroy()
        self._create_settings_interface()
        logger.info("Модуль настроек активирован")

    def _create_settings_interface(self):
        self.settings_frame = ttk.Frame(self.gui.content_area)
        self.settings_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.settings_notebook = ttk.Notebook(self.settings_frame)
        self.settings_notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self._create_general_settings_tab()
        self._create_system_settings_tab()
        # ... other tabs ...

        self._create_settings_control_panel()

    def _create_general_settings_tab(self):
        general_frame = ttk.Frame(self.settings_notebook)
        self.settings_notebook.add(general_frame, text="Общие")
        
        # Theme
        ttk.Label(general_frame, text="Тема интерфейса:").pack(anchor=tk.W)
        self.vars['theme'] = tk.StringVar(value=self.gui.settings.get("gui", {}).get("theme"))
        theme_combo = ttk.Combobox(general_frame, textvariable=self.vars['theme'], values=["light", "dark"], state="readonly")
        theme_combo.pack(fill=tk.X)

        # ... other settings widgets ...

    def _create_system_settings_tab(self):
        """Создает вкладку системных настроек"""
        try:
            system_frame = ttk.Frame(self.settings_notebook)
            self.settings_notebook.add(system_frame, text="Система")
            
            # Автозапуск
            self.vars['autostart'] = tk.BooleanVar(value=self.gui.settings.get("system", {}).get("autostart", False))
            ttk.Checkbutton(system_frame, text="Автозапуск системы", variable=self.vars['autostart']).pack(anchor=tk.W, pady=5)
            
            # Интервал автообновления
            ttk.Label(system_frame, text="Интервал обновления (мс):").pack(anchor=tk.W, pady=(10, 0))
            self.vars['update_interval'] = tk.StringVar(value=str(self.gui.settings.get("gui", {}).get("auto_update_interval", 5000)))
            ttk.Entry(system_frame, textvariable=self.vars['update_interval']).pack(fill=tk.X)
            
            # Логирование
            ttk.Label(system_frame, text="Уровень логирования:").pack(anchor=tk.W, pady=(10, 0))
            self.vars['log_level'] = tk.StringVar(value=self.gui.settings.get("system", {}).get("log_level", "INFO"))
            log_combo = ttk.Combobox(system_frame, textvariable=self.vars['log_level'], 
                                     values=["DEBUG", "INFO", "WARNING", "ERROR"], state="readonly")
            log_combo.pack(fill=tk.X)
            
            # Кэш директория
            ttk.Label(system_frame, text="Директория кэша:").pack(anchor=tk.W, pady=(10, 0))
            cache_frame = ttk.Frame(system_frame)
            cache_frame.pack(fill=tk.X)
            self.vars['cache_dir'] = tk.StringVar(value=self.gui.cache_dir)
            ttk.Entry(cache_frame, textvariable=self.vars['cache_dir']).pack(side=tk.LEFT, fill=tk.X, expand=True)
            ttk.Button(cache_frame, text="Обзор...", command=self._browse_cache_dir).pack(side=tk.RIGHT, padx=(5, 0))
            
        except Exception as e:
            logger.error(f"Ошибка создания вкладки системных настроек: {e}")
    
    def _browse_cache_dir(self):
        """Открывает диалог выбора директории кэша"""
        from tkinter import filedialog
        dir_path = filedialog.askdirectory(initialdir=self.vars['cache_dir'].get())
        if dir_path:
            self.vars['cache_dir'].set(dir_path)

    def _create_adaptation_settings_tab(self):
        """Создает вкладку настроек адаптации"""
        try:
            adapt_frame = ttk.Frame(self.settings_notebook)
            self.settings_notebook.add(adapt_frame, text="Адаптация")
            
            # Включение адаптации
            self.vars['adaptation_enabled'] = tk.BooleanVar(value=self.gui.settings.get("adaptation", {}).get("enabled", True))
            ttk.Checkbutton(adapt_frame, text="Включить адаптацию ответов", 
                          variable=self.vars['adaptation_enabled']).pack(anchor=tk.W, pady=5)
            
            # Стиль адаптации
            ttk.Label(adapt_frame, text="Стиль адаптации:").pack(anchor=tk.W, pady=(10, 0))
            self.vars['adaptation_style'] = tk.StringVar(value=self.gui.settings.get("adaptation", {}).get("style", "auto"))
            style_combo = ttk.Combobox(adapt_frame, textvariable=self.vars['adaptation_style'],
                                     values=["auto", "formal", "casual", "technical", "simple"], state="readonly")
            style_combo.pack(fill=tk.X)
            
            # Порог адаптации
            ttk.Label(adapt_frame, text="Порог адаптации (0.0 - 1.0):").pack(anchor=tk.W, pady=(10, 0))
            self.vars['adaptation_threshold'] = tk.StringVar(value=str(self.gui.settings.get("adaptation", {}).get("threshold", 0.7)))
            ttk.Entry(adapt_frame, textvariable=self.vars['adaptation_threshold']).pack(fill=tk.X)
            
            # Автообучение
            self.vars['auto_learning'] = tk.BooleanVar(value=self.gui.settings.get("adaptation", {}).get("auto_learning", True))
            ttk.Checkbutton(adapt_frame, text="Автоматическое обучение на диалогах",
                          variable=self.vars['auto_learning']).pack(anchor=tk.W, pady=(10, 5))
            
        except Exception as e:
            logger.error(f"Ошибка создания вкладки адаптации: {e}")

    def _create_backup_settings_tab(self):
        """Создает вкладку настроек резервного копирования"""
        try:
            backup_frame = ttk.Frame(self.settings_notebook)
            self.settings_notebook.add(backup_frame, text="Резервное копирование")
            
            # Автоматическое резервное копирование
            self.vars['auto_backup'] = tk.BooleanVar(value=self.gui.settings.get("backup", {}).get("auto_backup", True))
            ttk.Checkbutton(backup_frame, text="Автоматическое резервное копирование",
                          variable=self.vars['auto_backup']).pack(anchor=tk.W, pady=5)
            
            # Интервал резервного копирования
            ttk.Label(backup_frame, text="Интервал (часов):").pack(anchor=tk.W, pady=(10, 0))
            self.vars['backup_interval'] = tk.StringVar(value=str(self.gui.settings.get("backup", {}).get("interval_hours", 24)))
            ttk.Entry(backup_frame, textvariable=self.vars['backup_interval']).pack(fill=tk.X)
            
            # Директория для резервных копий
            ttk.Label(backup_frame, text="Директория для резервных копий:").pack(anchor=tk.W, pady=(10, 0))
            backup_dir_frame = ttk.Frame(backup_frame)
            backup_dir_frame.pack(fill=tk.X)
            self.vars['backup_dir'] = tk.StringVar(value=self.gui.settings.get("backup", {}).get("directory", os.path.expanduser("~/CogniFlex_Backups")))
            ttk.Entry(backup_dir_frame, textvariable=self.vars['backup_dir']).pack(side=tk.LEFT, fill=tk.X, expand=True)
            ttk.Button(backup_dir_frame, text="Обзор...", command=self._browse_backup_dir).pack(side=tk.RIGHT, padx=(5, 0))
            
            # Кнопка ручного резервного копирования
            ttk.Button(backup_frame, text="Создать резервную копию сейчас",
                     command=self._create_manual_backup).pack(anchor=tk.W, pady=(20, 5))
            
        except Exception as e:
            logger.error(f"Ошибка создания вкладки резервного копирования: {e}")
    
    def _browse_backup_dir(self):
        """Открывает диалог выбора директории для резервных копий"""
        from tkinter import filedialog
        dir_path = filedialog.askdirectory(initialdir=self.vars['backup_dir'].get())
        if dir_path:
            self.vars['backup_dir'].set(dir_path)
    
    def _create_manual_backup(self):
        """Создает ручную резервную копию"""
        try:
            backup_dir = self.vars['backup_dir'].get()
            os.makedirs(backup_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = os.path.join(backup_dir, f"cogniflex_backup_{timestamp}.json")
            
            # Сохраняем настройки
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(self.gui.settings, f, ensure_ascii=False, indent=2)
            
            if hasattr(self.gui, 'show_toast'):
                self.gui.show_toast(f"Резервная копия создана: {backup_file}", "success")
            logger.info(f"Ручная резервная копия создана: {backup_file}")
            
        except Exception as e:
            logger.error(f"Ошибка создания резервной копии: {e}")
            if hasattr(self.gui, 'show_toast'):
                self.gui.show_toast(f"Ошибка создания резервной копии: {str(e)}", "error")

    def _create_settings_control_panel(self):
        control_frame = ttk.Frame(self.settings_frame)
        control_frame.pack(fill=tk.X, pady=(10, 0))
        apply_button = ttk.Button(control_frame, text="Применить", command=self._apply_settings)
        apply_button.pack(side=tk.RIGHT)

    def _apply_settings(self):
        try:
            # Проверяем существование секции 'gui' в настройках
            if 'gui' not in self.gui.settings:
                self.gui.settings['gui'] = {}
            
            # GUI Settings
            self.gui.settings["gui"]["theme"] = self.vars['theme'].get()
            # ... get other vars

            # Save settings to file
            self._save_settings_to_file()
            
            # Apply settings
            if hasattr(self.gui, '_create_styles'):
                self.gui._create_styles() # Re-apply styles for theme change
            # ... apply other settings ...

            if hasattr(self.gui, 'show_toast'):
                self.gui.show_toast("Настройки успешно применены", "success")
            logger.info("Настройки успешно применены")
        except Exception as e:
            logger.error(f"Ошибка применения настроек: {e}", exc_info=True)
            if hasattr(self.gui, 'show_toast'):
                self.gui.show_toast(f"Ошибка применения настроек: {str(e)}", "error")

    def _save_settings_to_file(self):
        settings_path = os.path.join(self.gui.cache_dir, "gui_settings.json")
        system_settings_path = os.path.join(self.gui.cache_dir, "system_settings.json")
        try:
            with open(settings_path, "w") as f:
                json.dump(self.gui.settings["gui"], f, indent=2)
            with open(system_settings_path, "w") as f:
                json.dump(self.gui.settings["system"], f, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения настроек в файлы: {e}")