import logging
import os
import sys
import tempfile
import psutil
from tkinter import messagebox
import threading
import tkinter as tk

# Add the project root (parent of this 'tools' directory) to the Python path
TOOLS_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TOOLS_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cogniflex.core.core_brain import CoreBrain
from cogniflex.gui.core_gui import CogniFlexGUI
from cogniflex.core.utils import setup_logging

logger = logging.getLogger(__name__)

def is_gui_running():
    """Проверяет, запущен ли уже экземпляр GUI."""
    current_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if (proc.info['pid'] != current_pid and 
                'python' in proc.info['name'].lower() and 
                any('run_gui.py' in ' '.join(arg) for arg in proc.info['cmdline'] if arg)):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return False

def initialize_core():
    """Инициализирует ядро CogniFlex."""
    try:
        logger.info("Инициализация ядра CogniFlex...")
        core = CoreBrain()
        if not core.initialize():
            logger.error("Не удалось инициализировать ядро CogniFlex")
            return None
        logger.info("Ядро CogniFlex успешно инициализировано")
        return core
    except Exception as e:
        logger.critical(f"Критическая ошибка при инициализации ядра: {e}", exc_info=True)
        return None

def main():
    """Главная функция для запуска CogniFlex с графическим интерфейсом."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Проверяем, не запущен ли уже GUI
    if is_gui_running():
        logger.error("Обнаружен уже запущенный экземпляр CogniFlex GUI")
        messagebox.showerror("Ошибка", "CogniFlex GUI уже запущен!")
        sys.exit(1)
        
    logger.info("Запуск приложения CogniFlex с GUI...")
    
    # Инициализируем ядро
    core = initialize_core()
    if not core:
        messagebox.showerror("Ошибка", "Не удалось инициализировать ядро CogniFlex. Проверьте логи.")
        sys.exit(1)

    gui = None
    try:
        # Создаем и настраиваем GUI
        gui = CogniFlexGUI(brain=core)
        logger.info("GUI создан")
        
        # Создаем root сначала, чтобы _create_styles() мог работать
        gui.root = tk.Tk()
        gui.root.title("CogniFlex - Адаптивная когнитивная система")
        gui.root.geometry("1280x800")
        gui.root.minsize(800, 600)
        
        # Настраиваем стили и интерфейс
        gui._create_styles()
        gui._create_interface()
        gui._init_modules()
        
        # Устанавливаем обработчик закрытия окна
        def on_gui_close():
            """Обработчик закрытия главного окна"""
            logger.info("Завершение работы CogniFlex GUI...")
            if core and hasattr(core, 'shutdown'):
                core.shutdown()
            if gui and hasattr(gui, 'root') and gui.root:
                gui.root.destroy()
        
        gui.root.protocol("WM_DELETE_WINDOW", on_gui_close)
        
        # Запускаем ядро
        core.start()
        logger.info("Ядро CogniFlex запущено")
        
        # Запускаем GUI (это блокирующий вызов)
        logger.info("Запуск основного цикла GUI...")
        gui.root.mainloop()
        
    except Exception as e:
        logger.critical(f"Критическая ошибка при запуске GUI: {e}", exc_info=True)
        messagebox.showerror("Ошибка", f"Не удалось запустить графический интерфейс: {e}")
        if gui and hasattr(gui, 'root') and gui.root:
            gui.root.quit()
    finally:
        # Очистка ресурсов при завершении
        logger.info("Освобождение ресурсов...")
        if core and hasattr(core, 'shutdown'):
            core.shutdown()
        logger.info("Приложение завершено")

if __name__ == "__main__":
    main()