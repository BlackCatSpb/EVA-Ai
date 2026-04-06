"""
Модуль для управления темами и стилями графического интерфейса
"""
import tkinter.ttk as ttk

# Цветовые схемы для тем
THEME_COLORS = {
    "light": {
        'bg': '#f0f0f0',
        'card-bg': 'white',
        'primary': '#4a7abc',
        'secondary': '#5cb85c',
        'text': '#333333',
        'text-muted': '#6c757d',
        'border': '#e0e0e0',
        'success': '#28a745',
        'warning': '#ffc107',
        'danger': '#dc3545',
        'info': '#17a2b8',
        'chat-ai-bg': '#e9ecef',
        'contradiction-high': '#ffebee',
        'contradiction-medium': '#fff8e1',
        'contradiction-low': '#e8f5e9'
    },
    "dark": {
        'bg': '#1e1e1e',
        'card-bg': '#2d2d2d',
        'primary': '#5a8bcd',
        'secondary': '#6cc96c',
        'text': '#e0e0e0',
        'text-muted': '#adb5bd',
        'border': '#404040',
        'success': '#4caf50',
        'warning': '#ffeb3b',
        'danger': '#f44336',
        'info': '#2196f3',
        'chat-ai-bg': '#2a2a2a',
        'contradiction-high': '#3d1d1e',
        'contradiction-medium': '#3d331d',
        'contradiction-low': '#1d3d2a'
    }
}

def create_styles(gui):
    """Создает стили для интерфейса на основе текущей темы."""
    style = ttk.Style()
    
    # Устанавливаем базовую тему
    style.theme_use('clam')
    
    # Получаем цвета для текущей темы
    gui.colors = THEME_COLORS.get(gui.theme, THEME_COLORS["light"])
    
    # Основные стили
    style.configure("MainFrame.TFrame", background=gui.colors["bg"])
    style.configure("Card.TFrame", background=gui.colors["card-bg"], relief="solid", borderwidth=1)
    style.configure("Border.TFrame", background=gui.colors["border"])
    style.configure("Status.TLabel", foreground=gui.colors["text"])
    
    # Стили кнопок
    style.configure("Primary.TButton", background=gui.colors["primary"], foreground="white")
    style.configure("Secondary.TButton", background=gui.colors["secondary"], foreground="white")
    style.configure("Nav.TButton", padding=10, font=("Arial", 10, "bold"))
    
    # Стили для чата
    style.configure("UserBubble.TLabel", background=gui.colors["primary"], foreground="white", padding=8, relief="flat")
    style.configure("SystemBubble.TLabel", background=gui.colors["chat-ai-bg"], foreground=gui.colors["text"], padding=8, relief="flat")
    
    # Настройка общих стилей
    base_font = ("Segoe UI", gui.settings["gui"].get("font_size", 10))
    style.configure('.', font=base_font, background=gui.colors["bg"], foreground=gui.colors["text"])
    
    # Добавляем стили для карточек
    style.configure('Card.TFrame', background=gui.colors['card-bg'], 
                  relief='solid', borderwidth=1, bordercolor=gui.colors['border'])
    
    # Стили для уведомлений
    style.configure('Notification.TFrame', background=gui.colors['bg'], 
                  relief='solid', borderwidth=1)