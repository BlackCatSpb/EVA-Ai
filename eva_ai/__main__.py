#!/usr/bin/env python3
"""
ЕВА System Launcher via python -m eva_ai
"""
import os
import sys
import logging
import io

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

# Вычисляем корневую директорию проекта (родитель eva/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Добавляем project_root в начало sys.path ПЕРЕД любыми импортами
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Важно: задаём конфигурацию аллокатора CUDA до импорта torch/transformers
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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

def main():
    """Основная функция запуска системы."""
    try:
        from eva_ai.run import main as run_main
        return run_main()
    except ImportError as e:
        logger.error(f"Ошибка импорта: {e}")
        print("Убедитесь, что все зависимости установлены: pip install -r requirements.txt")
        return False
    except Exception as e:
        logger.error(f"Ошибка запуска: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
