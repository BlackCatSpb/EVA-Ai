# eva/gui/widgets.py
import tkinter as tk
from tkinter import ttk
import logging
import platform
from typing import Dict, Any, Callable, Optional, List

logger = logging.getLogger("eva_ai.gui.widgets")

# DPI awareness для Windows
def _set_dpi_awareness():
    """Установить DPI awareness для Windows."""
    if platform.system() == 'Windows':
        try:
            import ctypes
            # Try Windows 10+ API first
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
            logger.debug("DPI awareness установлен (Windows 10+)")
        except Exception:
            try:
                # Fallback for older Windows
                ctypes.windll.user32.SetProcessDPIAware()
                logger.debug("DPI awareness установлен (legacy)")
            except Exception as e:
                logger.debug(f"Не удалось установить DPI awareness: {e}")

# Устанавливаем DPI awareness при импорте
_set_dpi_awareness()

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

_toast_tk_root = None

def _get_toast_root():
    global _toast_tk_root
    if _toast_tk_root is None:
        try:
            _toast_tk_root = tk.Tk()
            _toast_tk_root.withdraw()
            _toast_tk_root.attributes('-topmost', True)
        except Exception:
            _toast_tk_root = None
    return _toast_tk_root

def _show_toast_window(message: str, level: str = "info", duration: int = 5000):
    try:
        root = _get_toast_root()
        if root is None:
            return

        colors = {
            'info': ('#0078d7', 'white'),
            'success': ('#28a745', 'white'),
            'warning': ('#ffc107', 'black'),
            'error': ('#dc3545', 'white')
        }
        bg_color, fg_color = colors.get(level, ('#333333', 'white'))

        toast = tk.Toplevel(root)
        toast.withdraw()
        toast.overrideredirect(True)
        toast.configure(bg=bg_color)
        toast.attributes('-topmost', True)

        label = tk.Label(toast, text=message, bg=bg_color, fg=fg_color,
                        font=('Segoe UI', 10), wraplength=300, padx=15, pady=10)
        label.pack()

        toast.update_idletasks()
        w = toast.winfo_width() or 300
        h = toast.winfo_height() or 50
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        x = screen_w - w - 20
        y = screen_h - h - 60
        toast.geometry(f"{w}x{h}+{x}+{y}")
        toast.deiconify()

        def destroy():
            try:
                toast.destroy()
            except Exception:
                pass

        root.after(duration, destroy)
    except Exception:
        pass

def show_notification(message: str, level: str = "info",
                     title: Optional[str] = None, duration: Optional[int] = None,
                     actions: Optional[List[Dict[str, Callable]]] = None, **kwargs):
    logger.info(f"[{level.upper()}] {message}")
    if title:
        logger.info(f"  {title}")
    if actions:
        action_names = [action["text"] for action in actions]
        logger.info(f"  Available actions: {', '.join(action_names)}")
    _show_toast_window(message, level, duration or 5000)

def show_toast(message: str, duration: int = 3000, **kwargs):
    logger.info(f"[TOAST] {message}")
    _show_toast_window(message, kwargs.get('level', 'info'), duration)