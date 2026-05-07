import os
import sys
import logging
import warnings
import atexit
import signal
import threading
import time
import platform

# ============================================================================
# UTF-8 Console Setup for Windows
# ============================================================================
if platform.system() == 'Windows':
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)  # CP_UTF8
        kernel32.SetConsoleCP(65001)
    except Exception:
        pass

# Попытка установить UTF-8 для stdout/stderr
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
# ============================================================================

# CPU OPTIMIZATION: Используем ВСЕ ядра процессора (i5-12450H: 8 ядер, 12 потоков)
# Устанавливаем переменные среды ДО всех импортов библиотек
# ============================================================================
import multiprocessing
_cpu_count = multiprocessing.cpu_count() or 12

os.environ.setdefault("OMP_NUM_THREADS", str(_cpu_count))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_cpu_count))
os.environ.setdefault("MKL_NUM_THREADS", str(_cpu_count))
os.environ.setdefault("VECL_NUM_THREADS", str(_cpu_count))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(_cpu_count))
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

print(f"[EVA-RUN] CPU threads: {_cpu_count} logical processors")
# ============================================================================

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=PendingDeprecationWarning)

import torch

# Применяем аппаратные оптимизации сразу после импорта torch
from eva_ai.core.hardware_optimizations import apply_hardware_optimizations
apply_hardware_optimizations(torch.device('cpu'), config=None)

from eva_ai.core.utils import setup_logging

# Настроить логирование в файл EVA_log.txt
log_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_file_path = os.path.join(log_dir, "EVA_log.txt")
setup_logging(log_dir=".", log_file="EVA_log.txt", level=logging.INFO)

logger = logging.getLogger("eva_ai.run")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_PID_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".eva_instance.pid")

_shutdown_event = threading.Event()
_brain_instance = None
_webgui_instance = None
_cleanup_done = False

def _cleanup_pid():
    """Удаляет PID-файл при завершении."""
    try:
        if os.path.exists(_PID_FILE):
            os.remove(_PID_FILE)
    except Exception:
        pass

def _cleanup_brain():
    """Корректно останавливает CoreBrain и все компоненты."""
    global _brain_instance, _webgui_instance, _cleanup_done
    
    if _cleanup_done:
        return
    _cleanup_done = True
    
    logger.info("=== Начало корректного завершения ===")
    start_time = time.time()
    
    # 1. Останавливаем CoreBrain (включает все компоненты, EventBus, IntegrationCore)
    if _brain_instance is not None:
        try:
            logger.info("Остановка CoreBrain...")
            if hasattr(_brain_instance, 'stop'):
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_brain_instance.stop)
                    try:
                        future.result(timeout=8)  # 8 секунд максимум
                        logger.info("CoreBrain остановлен")
                    except concurrent.futures.TimeoutError:
                        logger.warning("CoreBrain не остановился за 8 сек, пропускаем...")
                    except Exception as ex:
                        logger.warning(f"Ошибка при остановке CoreBrain: {ex}")
        except Exception as e:
            logger.warning(f"Ошибка при остановке CoreBrain: {e}")
    
    # 2. Останавливаем WebGUI (если не был остановлен через _stop_components)
    if _webgui_instance is not None:
        try:
            logger.info("Остановка WebGUI (fallback)...")
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_webgui_instance.stop)
                try:
                    future.result(timeout=5)
                    logger.info("WebGUI остановлен")
                except concurrent.futures.TimeoutError:
                    logger.warning("WebGUI не остановился за 5 сек, пропускаем...")
                except Exception as ex:
                    logger.debug(f"WebGUI уже остановлен: {ex}")
        except Exception as e:
            logger.debug(f"WebGUI cleanup error: {e}")
    
    # 3. Останавливаем GPU/CUDA если используется
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("CUDA кэш очищен")
    except Exception:
        pass
    
    # 4. Принудительно завершаем daemon-потоки
    logger.info("Ожидание завершения потоков...")
    daemon_threads = [t for t in threading.enumerate() if t != threading.current_thread() and t.is_alive()]
    for t in daemon_threads:
        if t.is_alive():
            logger.debug(f"Поток {t.name} завершается...")
    
    # Ждём немного чтобы потоки завершились
    time.sleep(0.5)
    
    elapsed = time.time() - start_time
    logger.info(f"=== Завершение выполнено за {elapsed:.2f} сек ===")
    
    _cleanup_pid()

def _signal_handler(signum, frame):
    """Обработчик сигналов (Ctrl+C, SIGTERM)."""
    signal_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
    logger.info(f"Получен сигнал {signal_name}, начинаю корректное завершение...")
    
    _shutdown_event.set()
    
    # Немедленно запускаем cleanup в отдельном потоке
    cleanup_thread = threading.Thread(target=_cleanup_brain, daemon=True, name="cleanup_thread")
    cleanup_thread.start()
    
    # Даём время на cleanup но не блокируем надолго
    cleanup_thread.join(timeout=8)
    
    if cleanup_thread.is_alive():
        logger.warning("Cleanup не завершился за 8 сек, принудительное завершение...")
        _cleanup_pid()
        sys.exit(1)
    else:
        logger.info("Cleanup успешно завершён")
        sys.exit(0)

def _check_singleton():
    """Проверяет, не запущен ли уже экземпляр EVA."""
    if os.path.exists(_PID_FILE):
        try:
            with open(_PID_FILE, "r") as f:
                old_pid = int(f.read().strip())
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
            try:
                os.remove(_PID_FILE)
            except Exception:
                pass
    
    try:
        fd = os.open(_PID_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(str(os.getpid()))
        except Exception:
            os.close(fd)
            raise
    except FileExistsError:
        logger.error(f"EVA уже запущена. Удалите {_PID_FILE} если уверены, что процесс не запущен")
        sys.exit(1)
    except OSError as e:
        logger.error(f"Не удалось создать PID-файл: {e}")
        sys.exit(1)
    
    atexit.register(_cleanup_pid)
    logger.info(f"PID-файл создан: PID {os.getpid()}")

# Настройка TF32
try:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 ускорение активировано")
except Exception as e:
    logger.warning(f"Не удалось активировать TF32: {e}")

def launch_gui(brain):
    """Запускает веб-интерфейс."""
    global _brain_instance, _webgui_instance
    
    _brain_instance = brain
    
    try:
        logger.info("Запуск веб-интерфейса...")
        
        web_gui_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gui', 'web_gui')
        sys.path.insert(0, web_gui_dir)
        
        import server
        gui = server.create_app(brain=brain)
        _webgui_instance = gui
        
        logger.info("Система работает. Нажмите Ctrl+C для остановки.")
        
        # Ждём сигнала завершения
        try:
            while not _shutdown_event.is_set():
                _shutdown_event.wait(timeout=1)
        except KeyboardInterrupt:
            logger.info("Получен KeyboardInterrupt")
            _shutdown_event.set()
        
        logger.info("Завершение работы...")
        _cleanup_brain()
        
    except Exception as e:
        logger.error(f"Ошибка при запуске веб-интерфейса: {e}", exc_info=True)
        _cleanup_brain()
        sys.exit(1)

def main():
    """Основная функция запуска системы."""
    global _brain_instance
    
    from eva_ai.core.core_brain import CoreBrain
    
    try:
        logger.info("Инициализация CoreBrain...")
        brain = CoreBrain()
        _brain_instance = brain
        
        logger.info("Инициализация компонентов...")
        if not brain.initialize():
            logger.error("Ошибка инициализации системы")
            return False

        logger.info("Запуск системы...")
        if not brain.start():
            logger.error("Ошибка запуска системы")
            return False
        
        # Запустить онлайн-обучение GNN и LoRA после старта системы
        try:
            from eva_ai.fcp_core.online_trainer import integrate_online_trainer
            logger.info("[RUN] About to call integrate_online_trainer...")
            manager = integrate_online_trainer(brain)
            logger.info(f"[RUN] integrate_online_trainer returned: {manager}")
            logger.info("Online training started")
        except Exception as e:
            import traceback
            logger.warning(f"Could not start online training: {e}")
            logger.warning(traceback.format_exc())
            
        logger.info("EVA успешно запущена")
        logger.info(f"BRAIN OBJECT: {brain}")
        logger.info(f"BRAIN fcp_pipeline attr: {getattr(brain, 'fcp_pipeline', 'MISSING')}")
        
        launch_gui(brain)
        
        return True
        
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt в main")
        _cleanup_brain()
        return True
    except Exception as e:
        logger.critical(f"Критическая ошибка при запуске: {e}", exc_info=True)
        _cleanup_brain()
        return False

if __name__ == "__main__":
    setup_logging(log_dir="logs")
    _check_singleton()
    
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
    
    try:
        main()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt в main")
    finally:
        _cleanup_brain()
        logger.info("EVA завершена")
        sys.exit(0)
