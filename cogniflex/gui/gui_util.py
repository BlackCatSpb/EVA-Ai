# cogniflex/gui/gui_utils.py
import tkinter as tk
from tkinter import ttk

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