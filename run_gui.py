import sys
import os
import logging
import threading

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from cogniflex.core.core_brain import CoreBrain
from cogniflex.gui.core_gui import CogniFlexGUI
from cogniflex.core.utils import setup_logging

logger = logging.getLogger(__name__)

def main():
    """Главная функция для запуска CogniFlex с графическим интерфейсом."""
    setup_logging()
    logger.info("Запуск приложения CogniFlex с GUI...")

    core = CoreBrain()
    if not core.initialize():
        logger.critical("Не удалось инициализировать ядро CogniFlex. Выход.")
        # We cannot show a messagebox here as it might create a premature Tk root window.
        # The error is logged, and the application will exit.
        return

    # Запускаем ядро в фоновом потоке
    brain_thread = threading.Thread(target=core.start, daemon=True)
    brain_thread.start()
    logger.info("Ядро CogniFlex запускается в фоновом потоке.")

    gui = None
    try:
        # Инициализация и запуск GUI
        gui = CogniFlexGUI(brain=core)
        gui.start()  # Этот вызов блокирующий, он запускает mainloop
    except Exception as e:
        logger.critical(f"Критическая ошибка при запуске GUI: {e}", exc_info=True)
    finally:
        # Этот код выполнится после закрытия окна GUI
        logger.info("GUI был закрыт. Остановка ядра CogniFlex...")
        if core and core.running:
            core.stop()
        logger.info("Приложение CogniFlex завершило работу.")

if __name__ == "__main__":
    main()
