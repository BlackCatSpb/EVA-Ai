"""
Вспомогательные функции для графического интерфейса
"""
import queue
import json
import os
import tkinter as tk
from tkinter import ttk
from datetime import datetime
from .gui_modules import switch_view
from .gui_widgets import create_rounded_button



def load_settings(settings_path: str) -> dict:
    """Загружает настройки из файла."""
    try:
        if os.path.exists(settings_path):
            with open(settings_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Ошибка загрузки настроек: {e}")
    
    # Возвращаем настройки по умолчанию
    return {
        "gui": {
            "theme": "light",
            "language": "ru",
            "font_size": 10,
            "show_reasoning": True,
            "compact_mode": False,
            "show_notifications": True,
            "notification_duration": 5000,
            "auto_update_interval": 5000
        },
        "system": {
            "cache_dir": "cogniflex_cache"
        }
    }

def save_settings(settings: dict, settings_path: str):
    """Сохраняет настройки в файл."""
    try:
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Ошибка сохранения настроек: {e}")

def process_gui_queue(gui):
    """Обрабатывает очередь GUI-задач."""
    def _process():
        try:
            while True:
                task = gui.gui_queue.get_nowait()
                task()
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Ошибка обработки очереди GUI: {e}")
        finally:
            gui.root.after(100, _process)
    
    gui.root.after(100, _process)

def show_notification(gui, message: str, level: str = "info", 
                     title: str = None, duration: int = None,
                     actions: list = None):
    """
    Показывает уведомление в интерфейсе.
    
    Args:
        gui: Экземпляр графического интерфейса
        message: Текст уведомления
        level: Уровень уведомления (info, success, warning, error)
        title: Заголовок уведомления
        duration: Продолжительность отображения
        actions: Список действий
    """
    if not gui.settings["gui"].get("show_notifications", True):
        return
    
    # Проверяем существование notification_area
    if not hasattr(gui, 'notification_area') or not gui.notification_area:
        return
    
    # Создаем уведомление
    notification = ttk.Frame(gui.notification_area)
    notification.pack(fill=tk.X, pady=5)
    
    # Определяем цвет в зависимости от уровня
    bg_color = gui.colors.get(level, gui.colors["info"])
    
    # Заголовок
    if title:
        title_label = ttk.Label(
            notification, 
            text=title, 
            font=("Arial", 10, "bold"),
            background=bg_color,
            foreground="white"
        )
        title_label.pack(fill=tk.X, padx=10, pady=(5, 0))
    
    # Текст уведомления
    message_label = ttk.Label(
        notification, 
        text=message, 
        wraplength=400,
        background=bg_color,
        foreground="white"
    )
    message_label.pack(fill=tk.X, padx=10, pady=(0, 5))
    
    # Действия
    if actions:
        actions_frame = ttk.Frame(notification)
        actions_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        for action in actions:
            action_button = create_rounded_button(
                actions_frame,
                action["text"],
                action["command"],
                bg=bg_color,
                fg="white"
            )
            action_button.pack(side=tk.LEFT, padx=2)
    
    # Добавляем в список активных уведомлений
    if not hasattr(gui, 'active_notifications'):
        gui.active_notifications = []
    
    gui.active_notifications.append(notification)
    
    # Ограничиваем количество уведомлений
    if len(gui.active_notifications) > 5:
        oldest = gui.active_notifications.pop(0)
        oldest.destroy()
    
    # Автоматическое исчезновение
    duration = duration or gui.settings["gui"].get("notification_duration", 5000)
    gui.root.after(duration, lambda: hide_notification(gui, notification))

def hide_notification(gui, notification):
    """Скрывает уведомление с анимацией."""
    if not hasattr(gui, 'active_notifications'):
        return
        
    if notification in gui.active_notifications:
        gui.active_notifications.remove(notification)
        notification.destroy()

def show_learning_opportunities(gui, opportunities: list):
    """Отображает возможности для обучения.
    
    Args:
        gui: Экземпляр графического интерфейса
        opportunities: Список возможностей для обучения
    """
    if not opportunities:
        return
    
    # Показываем уведомление только если GUI активен
    if gui.root and gui.running:
        gui.gui_queue.put(lambda: gui.show_notification(
            f"Обнаружено {len(opportunities)} возможностей для обучения",
            "info",
            actions=[
                {
                    "text": "Посмотреть",
                    "command": lambda: switch_view(gui, "learning")
                }
            ]
        ))