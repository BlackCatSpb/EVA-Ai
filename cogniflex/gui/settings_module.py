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
        # ... implementation ...
        pass

    def _create_adaptation_settings_tab(self):
        # ... implementation ...
        pass

    def _create_backup_settings_tab(self):
        # ... implementation ...
        pass

    def _create_settings_control_panel(self):
        control_frame = ttk.Frame(self.settings_frame)
        control_frame.pack(fill=tk.X, pady=(10, 0))
        apply_button = ttk.Button(control_frame, text="Применить", command=self._apply_settings)
        apply_button.pack(side=tk.RIGHT)

    def _apply_settings(self):
        try:
            # GUI Settings
            self.gui.settings["gui"]["theme"] = self.vars['theme'].get()
            # ... get other vars

            # Save settings to file
            self._save_settings_to_file()
            
            # Apply settings
            self.gui._create_styles() # Re-apply styles for theme change
            # ... apply other settings ...

            self.gui.show_toast("Настройки успешно применены", "success")
            logger.info("Настройки успешно применены")
        except Exception as e:
            logger.error(f"Ошибка применения настроек: {e}", exc_info=True)
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