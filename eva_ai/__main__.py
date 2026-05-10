#!/usr/bin/env python3
"""
ЕВА System Launcher via python -m eva_ai
"""
import os
import sys
import logging
import io
import signal
import threading

# Fix Windows console Unicode encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)
        kernel32.SetConsoleCP(65001)
    except Exception:
        pass

# ============================================================================
# CPU OPTIMIZATION: Используем ВСЕ ядра процессора (i5-12450H: 8 ядер, 12 потоков)
# Устанавливаем переменные среды ДО всех импортов
# ============================================================================
import multiprocessing
_cpu_count = multiprocessing.cpu_count() or 12

# OpenMP (используется во многих научных библиотеках)
os.environ.setdefault("OMP_NUM_THREADS", str(_cpu_count))
# OpenBLAS (линейная алгебра)
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_cpu_count))
# MKL (Intel Math Kernel Library)
os.environ.setdefault("MKL_NUM_THREADS", str(_cpu_count))
# VECL (Intel oneAPI)
os.environ.setdefault("VECL_NUM_THREADS", str(_cpu_count))
# NumExpr (вычисления выражений)
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(_cpu_count))
# OMP policy
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
# Intel MKL blocktime
os.environ.setdefault("KMP_BLOCKTIME", "0")

print(f"[EVA-STARTUP] CPU threads: {_cpu_count} logical processors (i5-12450H)")
# ============================================================================

# Вычисляем корневую директорию проекта (родитель eva/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Добавляем project_root в начало sys.path ПЕРЕД любыми импортами
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Важно: задаём конфигурацию аллокатора CUDA до импорта torch/transformers
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Локальный кэш для HuggingFace моделей (избегаем HTTP 429 ошибок)
hf_cache_dir = os.path.join(project_root, 'eva_ai', 'core', 'hf_cache')
os.makedirs(hf_cache_dir, exist_ok=True)
os.environ.setdefault("HF_HOME", hf_cache_dir)
os.environ.setdefault("TRANSFORMERS_CACHE", hf_cache_dir)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", os.path.join(hf_cache_dir, 'sentence-transformers'))

# Настраиваем логирование СРАЗУ, перед любыми импортами
log_dir = os.path.join(project_root, 'logs')
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(log_dir, 'eva_ai.log'), encoding='utf-8')
    ]
)

logger = logging.getLogger("eva_ai.__main__")
logger.info(f"Project root: {project_root}")

# Глобальная переменная для хранилища core_brain
_core_instance = None
_shutdown_in_progress = False

def _handle_shutdown(signum, frame):
    """Обработчик сигнала завершения (Ctrl+C, закрытие терминала)."""
    global _shutdown_in_progress
    
    # Предотвращаем повторный вызов
    if _shutdown_in_progress:
        logger.warning("Повторный сигнал завершения — принудительный выход")
        os._exit(1)
    _shutdown_in_progress = True
    
    logger.info(f"Получен сигнал завершения ({signum})...")
    
    # Сигнал приходит в главный поток — запускаем cleanup в отдельном потоке
    # чтобы не блокировать обработку сигнала
    def _do_shutdown():
        global _core_instance
        if _core_instance is not None:
            try:
                logger.info("Останавливаем EVA...")
                _core_instance.stop()
            except Exception as e:
                logger.error(f"Ошибка при остановке: {e}")
        # Сигнализируем run.py о необходимости завершиться
        try:
            from eva_ai.run import _shutdown_event
            _shutdown_event.set()
        except Exception:
            pass
    
    shutdown_thread = threading.Thread(target=_do_shutdown, daemon=True, name="signal_shutdown")
    shutdown_thread.start()
    
    # Не вызываем sys.exit() здесь — пусть run.py обработает через _shutdown_event
    # Это позволяет корректно пройти через finally блок

# Регистрируем обработчики сигналов
signal.signal(signal.SIGINT, _handle_shutdown)
signal.signal(signal.SIGTERM, _handle_shutdown)

def main():
    """Основная функция запуска системы."""
    global _core_instance
    
    try:
        from eva_ai.run import main as run_main
        result = run_main()
        
        # Пробуем получить core_instance для shutdown
        try:
            from eva_ai.core.core_brain import get_core_instance
            _core_instance = get_core_instance()
        except Exception:
            pass
        
        return result
    except ImportError as e:
        logger.error(f"Ошибка импорта: {e}")
        print("Убедитесь, что все зависимости установлены: pip install -r requirements.txt")
        return False
    except KeyboardInterrupt:
        logger.info("Получен KeyboardInterrupt")
        if _core_instance is not None:
            _core_instance.stop()
        return True
    except Exception as e:
        logger.error(f"Ошибка запуска: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
