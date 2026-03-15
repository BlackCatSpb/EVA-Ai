"""
Модуль для создания виджетов графического интерфейса
"""
import tkinter as tk
import datetime
from tkinter import ttk
from .widgets import create_rounded_button
from .gui_modules import switch_view


def create_main_interface(gui):
    """Создает основной интерфейс приложения."""
    # Основной контейнер
    gui.main_container = ttk.Frame(gui.root)
    gui.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Создаем элементы интерфейса
    create_navbar(gui)
    create_content_area(gui)
    create_status_bar(gui)
    create_notification_area(gui)

def create_navbar(gui):
    """Создает панель навигации."""
    gui.navbar = ttk.Frame(gui.main_container)
    gui.navbar.pack(fill=tk.X, pady=(0, 5))
    
    # Логотип и название
    title_frame = ttk.Frame(gui.navbar)
    title_frame.pack(side=tk.LEFT, padx=10)
    
    ttk.Label(
        title_frame,
        text="CogniFlex",
        font=('Segoe UI', 16, 'bold'),
        foreground=gui.colors["primary"]
    ).pack(side=tk.LEFT)
    
    ttk.Label(
        title_frame,
        text=" v1.0",
        font=('Segoe UI', 10),
        foreground=gui.colors["text-muted"]
    ).pack(side=tk.LEFT, padx=(2, 0))
    
    # Кнопки навигации
    nav_buttons = [
        ("chat", "Чат", "💬"),
        ("knowledge", "Знания", "🧠"),
        ("contradictions", "Противоречия", "⚠️"),
        ("memory", "Память", "💾"),
        ("learning", "Обучение", "🎓"),
        ("analytics", "Аналитика", "📊"),
        ("settings", "Настройки", "⚙️")
    ]
    
    gui.nav_buttons = {}
    for view_id, text, icon in nav_buttons:
        button = ttk.Button(
            gui.navbar,
            text=f"{icon} {text}",
            command=lambda v=view_id: switch_view(gui, v)
        )
        button.pack(side=tk.LEFT, padx=2)
        gui.nav_buttons[view_id] = button

def create_content_area(gui):
    """Создает основную область контента."""
    # Основная область
    gui.content_frame = ttk.Frame(gui.main_container)
    gui.content_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
    
    # Левая панель (содержимое)
    gui.content_area = ttk.Frame(gui.content_frame)
    gui.content_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
    
    # Правая панель (боковая панель)
    gui.sidebar = ttk.Frame(gui.content_frame, width=300)
    gui.sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
    gui.sidebar.pack_propagate(False)

def create_status_bar(gui):
    """Создает панель статуса."""
    gui.status_bar = ttk.Frame(gui.main_container)
    gui.status_bar.pack(fill=tk.X, pady=(5, 0))
    
    # Индикатор состояния
    gui.status_indicator = tk.Canvas(gui.status_bar, width=20, height=20, highlightthickness=0)
    gui.status_indicator.create_oval(2, 2, 18, 18, fill=gui.colors["warning"], tags="indicator")
    gui.status_indicator.pack(side=tk.LEFT, padx=(0, 5))
    
    # Статус соединения
    gui.connection_status = ttk.Label(
        gui.status_bar, 
        text="Соединение: инициализация...",
        font=("Arial", 9)
    )
    gui.connection_status.pack(side=tk.LEFT, padx=(0, 10))
    
    # Метка времени
    gui.timestamp_label = ttk.Label(
        gui.status_bar, 
        text=datetime.now().strftime("%H:%M:%S"),
        font=("Arial", 9)
    )
    gui.timestamp_label.pack(side=tk.RIGHT, padx=(10, 0))

def create_notification_area(gui):
    """Создает область для уведомлений."""
    gui.notification_area = ttk.Frame(gui.root)
    gui.notification_area.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)