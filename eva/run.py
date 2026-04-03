import os
import sys
import logging
import warnings
import atexit
warnings.filterwarnings('ignore')

# Обход проверки CVE-2025-32434 для torch.load
import torch
torch._inductor.config.triton.cudagraphs = False

from eva.core.utils import setup_logging

# Получаем логгер. Настройка будет выполнена либо в `diagnostic_launcher`, либо при прямом запуске.
logger = logging.getLogger("eva.run")

# Важно: задаём конфигурацию аллокатора CUDA до импорта torch/transformers
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ── Singleton: защита от множественных запусков ──
_PID_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".eva_instance.pid")

def _cleanup_pid():
    """Удаляет PID-файл при завершении."""
    try:
        if os.path.exists(_PID_FILE):
            os.remove(_PID_FILE)
            logger.info("PID-файл удалён")
    except Exception:
        pass

def _check_singleton():
    """Проверяет, не запущен ли уже экземпляр EVA."""
    if os.path.exists(_PID_FILE):
        try:
            with open(_PID_FILE, "r") as f:
                old_pid = int(f.read().strip())
            # Проверяем, жив ли процесс
            import subprocess
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {old_pid}", "/NH", "/FO", "CSV"],
                capture_output=True, text=True
            )
            if str(old_pid) in result.stdout:
                logger.error(f"EVA уже запущена (PID {old_pid}). Завершите старый процесс или удалите {_PID_FILE}")
                sys.exit(1)
            else:
                logger.info(f"Найден stale PID-файл (PID {old_pid}), удаляю")
                os.remove(_PID_FILE)
        except (ValueError, FileNotFoundError):
            os.remove(_PID_FILE)

    with open(_PID_FILE, "w") as f:
        f.write(str(os.getpid()))
    atexit.register(_cleanup_pid)
    logger.info(f"PID-файл создан: PID {os.getpid()}")

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
    """Запускает веб-интерфейс."""
    try:
        logger.info("Запуск веб-интерфейса...")
        
        # Import web GUI server
        import sys
        import os
        web_gui_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gui', 'web_gui')
        sys.path.insert(0, web_gui_dir)
        
        import server
        gui = server.create_app(brain=brain)
        logger.info(f"Веб-интерфейс запущен на http://{gui.host}:{gui.port}")
        
        # Keep running
        while True:
            import time
            time.sleep(1)
            
    except Exception as e:
        logger.error(f"Ошибка при запуске веб-интерфейса: {e}", exc_info=True)
        raise

def main():
    """Основная функция запуска системы."""
    from eva.core.core_brain import CoreBrain
    
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
            
        logger.info("ЕВА успешно запущен")
        
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
    _check_singleton()
    main()
