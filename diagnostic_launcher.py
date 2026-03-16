#!/usr/bin/env python3
"""
CogniFlex Diagnostic Launcher
Запуск системы с максимально подробным логгированием для диагностики
"""

import os
import sys
import logging
import time
from pathlib import Path
import platform
import torch

# Настройка логирования до всех остальных операций
log = logging.getLogger(__name__)

def setup_cuda_environment():
    """Настройка CUDA окружения с учетом особенностей Windows."""
    logger = logging.getLogger("cogniflex.cuda_setup")
    if platform.system() == "Windows":
        # Для Windows используем альтернативные настройки
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Для лучшей отладки
        logger.info("Применены специальные настройки CUDA для Windows: max_split_size_mb:128, CUDA_LAUNCH_BLOCKING=1")
    else:
        # Для Linux используем стандартные настройки
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        logger.info("Применены стандартные настройки CUDA для Linux: expandable_segments:True, CUDA_LAUNCH_BLOCKING=1")
    
    # Глобальное отключение TF32 для повышения точности и избежания некоторых проблем
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    logger.info("TF32 ускорение отключено для matmul и cudnn.")

# Применяем настройки окружения до импорта torch и других библиотек
setup_cuda_environment()

def setup_diagnostic_logging():
    """Настраивает максимально подробное логирование для диагностики"""

    # Создаем директорию для диагностических логов
    log_dir = Path("diagnostic_logs")
    log_dir.mkdir(exist_ok=True)

    # Настраиваем максимально подробное логирование
    log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"cogniflex_diagnostic_{timestamp}.log"

    # Настраиваем корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Очищаем существующие обработчики
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Создаем обработчик для файла
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))

    # Создаем обработчик для консоли
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(log_format))

    # Добавляем обработчики
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Настраиваем все дочерние логгеры
    loggers_to_configure = [
        'cogniflex',
        'cogniflex.core',
        'cogniflex.core_brain',
        'cogniflex.mlearning',
        'cogniflex.knowledge',
        'cogniflex.memory',
        'cogniflex.gui',
        'cogniflex.neuromorphic',
        'cogniflex.ethics',
        'cogniflex.adaptation',
        'cogniflex.contradiction',
        'cogniflex.learning',
        'cogniflex.distributed',
        'cogniflex.websearch'
    ]

    for logger_name in loggers_to_configure:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = True

    print(f"🔍 Диагностическое логирование настроено")
    print(f"📁 Лог-файл: {log_file.absolute()}")
    print(f"📊 Уровень: DEBUG (максимально подробный)")
    print(f"🔄 Логгеры настроены: {len(loggers_to_configure)}")
    print("-" * 50)

    return str(log_file)

def main():
    """Основная функция диагностического запуска"""

    print("=" * 60)
    print("🚀 COGNIFLEX DIAGNOSTIC LAUNCHER")
    print("=" * 60)
    print("Запуск системы с максимально подробным логгированием")
    print()

    # Настраиваем диагностическое логирование
    log_file = setup_diagnostic_logging()

    try:
        # Импортируем и запускаем основную систему
        print("📦 Импорт системы CogniFlex...")

        # Добавляем текущую директорию в путь
        current_dir = Path(__file__).parent
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))

        # Импорт основного модуля
        from cogniflex.run import main as run_main

        print("✅ Система импортирована успешно")
        print("🎯 Запуск CogniFlex...")
        print()

        # Запускаем систему
        run_main()

    except KeyboardInterrupt:
        print("\n⏹️  Диагностический запуск прерван пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка при запуске: {e}")
        logging.exception("Критическая ошибка в diagnostic launcher")

    finally:
        print("\n" + "=" * 60)
        print("📋 ДИАГНОСТИКА ЗАВЕРШЕНА")
        print("=" * 60)
        print(f"📁 Лог-файл сохранен: {log_file}")
        print("📊 Для анализа отправьте этот файл разработчику")
        print("🔍 Лог содержит максимально подробную информацию")
        print("=" * 60)

if __name__ == "__main__":
    main()
