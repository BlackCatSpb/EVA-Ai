import os
import sys
import logging
import warnings
import atexit
import signal
import threading
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=PendingDeprecationWarning)

import torch
# CVE-2025-32434 mitigation: all torch.load() calls across the codebase
# use explicit weights_only=False since model checkpoints contain full
# state dicts (not just tensor weights). See each call site for details.

from eva_ai.core.utils import setup_logging

# Получаем логгер. Настройка будет выполнена либо в `diagnostic_launcher`, либо при прямом запуске.
logger = logging.getLogger("eva_ai.run")

# Важно: задаём конфигурацию аллокатора CUDA до импорта torch/transformers
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ── Singleton: защита от множественных запусков ──
_PID_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".eva_instance.pid")

# ── Graceful shutdown ──
_shutdown_event = threading.Event()
_brain_instance = None

def _signal_handler(signum, frame):
    """Обработчик сигналов (Ctrl+C, SIGTERM)."""
    logger.info(f"Получен сигнал {signum}, начинаю корректное завершение...")
    _shutdown_event.set()
    _cleanup_brain()
    _cleanup_pid()
    sys.exit(0)

def _cleanup_brain():
    """Останавливает CoreBrain и все фоновые потоки."""
    global _brain_instance
    if _brain_instance is not None:
        try:
            logger.info("Остановка CoreBrain...")
            _brain_instance.stop()
            logger.info("CoreBrain остановлен")
        except Exception as e:
            logger.warning(f"Ошибка при остановке CoreBrain: {e}")
        _brain_instance = None

def _cleanup_pid():
    """Удаляет PID-файл при завершении."""
    try:
        if os.path.exists(_PID_FILE):
            os.remove(_PID_FILE)
            logger.info("PID-файл удалён")
    except Exception:
        pass

def _check_singleton():
    """Проверяет, не запущен ли уже экземпляр eva_ai."""
    if os.path.exists(_PID_FILE):
        try:
            with open(_PID_FILE, "r") as f:
                old_pid = int(f.read().strip())
            # Проверяем, жив ли процесс
            import subprocess
            import re
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {old_pid}", "/NH", "/FO", "CSV"],
                capture_output=True, text=True, timeout=10
            )
            if re.search(r'\b' + str(old_pid) + r'\b', result.stdout):
                logger.error(f"EVA уже запущена (PID {old_pid}). Завершите старый процесс или удалите {_PID_FILE}")
                sys.exit(1)
            else:
                logger.info(f"Найден stale PID-файл (PID {old_pid}), удаляю")
                os.remove(_PID_FILE)
        except (ValueError, FileNotFoundError):
            os.remove(_PID_FILE)

    try:
        fd = os.open(_PID_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(str(os.getpid()))
        except Exception:
            os.close(fd)
            raise
    except FileExistsError:
        logger.error(f"EVA уже запущена (конкурирующий процесс создал PID-файл). Удалите {_PID_FILE} если уверены, что процесс не запущен")
        sys.exit(1)
    except OSError as e:
        logger.error(f"Не удалось создать PID-файл: {e}")
        sys.exit(1)
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
    global _brain_instance
    _brain_instance = brain
    
    try:
        logger.info("Запуск веб-интерфейса...")
        
        web_gui_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gui', 'web_gui')
        sys.path.insert(0, web_gui_dir)
        
        import server
        gui = server.create_app(brain=brain)
        
        # Flask уже запущен в daemon-потоке через WebGUI.start()
        # Ждём сигнал завершения в главном цикле
        logger.info("Система работает. Нажмите Ctrl+C для остановки.")
        try:
            while not _shutdown_event.is_set():
                _shutdown_event.wait(timeout=1)
        except KeyboardInterrupt:
            pass
        
        logger.info("Завершение работы...")
        gui.stop()
        _cleanup_brain()
        
    except Exception as e:
        logger.error(f"Ошибка при запуске веб-интерфейса: {e}", exc_info=True)
        raise

def main():
    """Основная функция запуска системы."""
    global _brain_instance
    from eva_ai.core.core_brain import CoreBrain
    
    try:
        logger.info("Инициализация CoreBrain...")
        brain = CoreBrain()
        _brain_instance = brain
        
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
        _cleanup_brain()
        return False

if __name__ == "__main__":
    # Если файл запускается напрямую, настраиваем логирование.
    # Если он импортируется, предполагается, что логирование уже настроено.
    setup_logging(log_dir="logs")
    _check_singleton()
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
    main()
