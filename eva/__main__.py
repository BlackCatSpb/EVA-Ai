#!/usr/bin/env python3
"""
ЕВА System Launcher via python -m eva
"""
import os
import sys
import logging

# Добавляем текущую директорию в путь для импорта
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Важно: задаём конфигурацию аллокатора CUDA до импорта torch/transformers
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def main():
    """Основная функция запуска системы."""
    try:
        from eva.run import main as run_main
        return run_main()
    except ImportError as e:
        print(f"Ошибка импорта: {e}")
        print("Убедитесь, что все зависимости установлены: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"Ошибка запуска: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
