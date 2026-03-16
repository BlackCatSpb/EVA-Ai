import os
import sys
import logging
from cogniflex.core.utils import setup_logging

# Получаем логгер. Настройка будет выполнена либо в `diagnostic_launcher`, либо при прямом запуске.
logger = logging.getLogger("cogniflex.run")

# Важно: задаём конфигурацию аллокатора CUDA до импорта torch/transformers
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Настраиваем TF32 по новому API PyTorch до загрузки остальной системы
try:
    import torch
    try:
        # Для матричных операций и сверток используем TF32 (быстрее, приемлемая точность)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32 ускорение активировано")
        else:
            logger.info("CUDA недоступен, TF32 не активирован")
    except Exception as e:
        logger.warning(f"Не удалось активировать TF32 ускорение: {e}")
        pass
except Exception as e:
    logger.error(f"Ошибка при настройке PyTorch: {e}")

def launch_gui(brain):
    """Запускает графический интерфейс."""
    try:
        from cogniflex.gui import create_gui
        logger.info("Запуск графического интерфейса...")
        gui = create_gui(brain=brain)
        gui.start()
    except Exception as e:
        logger.error(f"Ошибка при запуске графического интерфейса: {e}", exc_info=True)
        raise

def main():
    """Основная функция запуска системы."""
    from cogniflex.core.core_brain import CoreBrain
    
    try:
        logger.info("Инициализация CoreBrain...")
        brain = CoreBrain()
        
        # Инициализация системы
        logger.info("Инициализация компонентов...")
        if not brain.initialize():
            logger.error("Ошибка инициализации системы")
            return False

        # Запуск системы
        logger.info("Запуск системы...")
        if not brain.start():
            logger.error("Ошибка запуска системы")
            return False
            
        logger.info("CogniFlex успешно запущен")
        
        # Запускаем GUI в основном потоке
        launch_gui(brain)
        
        return True
        
    except Exception as e:
        logger.critical(f"Критическая ошибка при запуске: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    # Если файл запускается напрямую, настраиваем логирование.
    # Если он импортируется, предполагается, что логирование уже настроено.
    setup_logging(log_dir="logs")
    main()
