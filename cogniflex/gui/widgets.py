# cogniflex/gui/widgets.py
import tkinter as tk
from tkinter import ttk
import logging
from typing import Dict, Any, Callable, Optional, List

logger = logging.getLogger("cogniflex.gui.widgets")

def create_rounded_button(parent, text, command=None, width=15, height=2, 
                         bg="#4a7abc", fg="white", font=("Arial", 10)):
    """Создает кнопку с закругленными краями."""
    button = tk.Canvas(parent, highlightthickness=0)
    
    # Размеры кнопки
    btn_width = width * 10
    btn_height = height * 20
    
    # Рисуем закругленный прямоугольник
    radius = 15
    button.create_rectangle(
        radius, 0, btn_width - radius, btn_height,
        fill=bg, outline=bg
    )
    button.create_rectangle(
        radius, radius, btn_width - radius, btn_height - radius,
        fill=bg, outline=bg
    )
    button.create_oval(
        0, 0, radius * 2, radius * 2,
        fill=bg, outline=bg
    )
    button.create_oval(
        btn_width - radius * 2, 0, btn_width, radius * 2,
        fill=bg, outline=bg
    )
    button.create_oval(
        0, btn_height - radius * 2, radius * 2, btn_height,
        fill=bg, outline=bg
    )
    button.create_oval(
        btn_width - radius * 2, btn_height - radius * 2, btn_width, btn_height,
        fill=bg, outline=bg
    )
    
    # Добавляем текст
    label = tk.Label(
        button, text=text, bg=bg, fg=fg, 
        font=font, cursor="hand2"
    )
    label.place(relx=0.5, rely=0.5, anchor="center")
    
    # Обработчик клика
    def on_click(event):
        if command:
            command()
    
    button.bind("<Button-1>", on_click)
    button.config(width=btn_width, height=btn_height)
    
    return button

def create_gradient_canvas(parent, width, height, color1, color2):
    """Создает холст с градиентом."""
    canvas = tk.Canvas(parent, width=width, height=height, highlightthickness=0)
    
    # Создаем градиент
    r1, g1, b1 = parent.winfo_rgb(color1)
    r2, g2, b2 = parent.winfo_rgb(color2)
    
    for i in range(height):
        # Интерполируем цвета
        r = int(r1 + (r2 - r1) * i / height)
        g = int(g1 + (g2 - g1) * i / height)
        b = int(b1 + (b2 - b1) * i / height)
        
        color = f'#{r//256:02x}{g//256:02x}{b//256:02x}'
        canvas.create_line(0, i, width, i, fill=color)
    
    return canvas

def create_card_frame(parent, **kwargs):
    """Создает карточку с закругленными краями."""
    frame = ttk.Frame(parent, **kwargs)
    frame.configure(relief="solid", borderwidth=1)
    return frame

def create_header_label(parent, text, **kwargs):
    """Создает заголовок с увеличенным шрифтом."""
    label = ttk.Label(parent, text=text, font=("Arial", 14, "bold"), **kwargs)
    return label

def create_secondary_label(parent, text, **kwargs):
    """Создает второстепенный текст."""
    label = ttk.Label(parent, text=text, font=("Arial", 9), foreground="#666666", **kwargs)
    return label

def show_notification(message: str, level: str = "info", 
                     title: Optional[str] = None, duration: Optional[int] = None,
                     actions: Optional[List[Dict[str, Callable]]] = None, **kwargs):
    """
    Показывает уведомление в интерфейсе с анимацией и автоматическим исчезновением.
    
    Args:
        message: Текст уведомления
        level: Уровень уведомления (info, success, warning, error)
        title: Заголовок уведомления
        duration: Продолжительность отображения в миллисекундах
        actions: Список действий (кнопок)
    """
    # Эта функция будет заменена на метод GUI, но оставлена для обратной совместимости
    logger.info(f"[{level.upper()}] {message}")
    if title:
        logger.info(f"  {title}")
    if actions:
        action_names = [action["text"] for action in actions]
        logger.info(f"  Доступные действия: {', '.join(action_names)}")

def show_toast(message: str, duration: int = 3000, **kwargs):
    """Показывает всплывающее уведомление."""
    # Заглушка для всплывающих уведомлений
    logger.info(f"[TOAST] {message}")