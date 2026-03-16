#!/usr/bin/env python3
"""
Скрипт для детального логирования запуска системы CogniFlex с исправленным PYTHONPATH
"""
import os
import sys
import logging
import traceback
from datetime import datetime

# Добавляем текущую директорию в PYTHONPATH для приоритета импорта из worktree
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

print(f"PYTHONPATH: {sys.path[:3]}")  # Показываем первые 3 пути для отладки

# Настройка расширенного логирования
def setup_debug_logging():
    """Настраивает детальное логирование для отладки запуска"""
    log_dir = "debug_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"cogniflex_debug_fixed_{timestamp}.log"
    log_path = os.path.join(log_dir, log_file)
    
    # Получаем корневой логгер
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Максимальный уровень детализации
    
    # Очищаем существующие обработчики
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    
    # Детальный формат сообщений
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(process)d - %(threadName)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Файловый обработчик с детальным логированием
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Консольный обработчик
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger, log_path

def main():
    """Основная функция с детальным логированием"""
    logger, log_path = setup_debug_logging()
    
    logger.info("=" * 80)
    logger.info("НАЧАЛО ДЕТАЛЬНОГО ЗАПУСКА COGNIFLEX (ИСПРАВЛЕННАЯ ВЕРСИЯ)")
    logger.info(f"Лог-файл: {log_path}")
    logger.info(f"Python версия: {sys.version}")
    logger.info(f"Рабочая директория: {os.getcwd()}")
    logger.info(f"Текущая директория скрипта: {current_dir}")
    logger.info(f"PYTHONPATH первые 3 элемента: {sys.path[:3]}")
    logger.info("=" * 80)
    
    try:
        # Важно: задаём конфигурацию аллокатора CUDA до импорта torch/transformers
        logger.debug("Настройка переменных окружения CUDA")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        logger.debug(f"PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
        
        # Настраиваем TF32 по новому API PyTorch до загрузки остальной системы
        logger.debug("Настройка PyTorch и TF32")
        try:
            import torch
            logger.info(f"PyTorch версия: {torch.__version__}")
            logger.info(f"CUDA доступна: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"CUDA версия: {torch.version.cuda}")
                logger.info(f"Количество GPU: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            
            # Пропускаем настройку TF32 для CPU версии PyTorch
            if not torch.cuda.is_available():
                logger.info("Пропуск настройки TF32 (CPU версия PyTorch)")
            else:
                try:
                    # Для матричных операций и сверток используем TF32 (быстрее, приемлемая точность)
                    torch.backends.cuda.matmul.fp32_precision = 'tf32'
                    torch.backends.cudnn.conv.fp32_precision = 'tf32'
                    logger.info("TF32 ускорение активировано")
                except Exception as e:
                    logger.warning(f"Не удалось активировать TF32 ускорение: {e}")
                    logger.debug(traceback.format_exc())
        except Exception as e:
            logger.error(f"Ошибка при настройке PyTorch: {e}")
            logger.debug(traceback.format_exc())
        
        # Импортируем и настраиваем логирование CogniFlex
        logger.debug("Импорт модуля настройки логирования CogniFlex")
        from cogniflex.core.utils import setup_logging
        logger.info("Настройка логирования CogniFlex")
        setup_logging(log_dir="logs", level=logging.DEBUG)
        
        # Импортируем CoreBrain с проверкой пути
        logger.debug("Проверка пути импорта CoreBrain")
        import cogniflex.core.core_brain
        logger.info(f"CoreBrain импортирован из: {cogniflex.core.core_brain.__file__}")
        
        from cogniflex.core.core_brain import CoreBrain
        logger.info("CoreBrain импортирован")
        
        # Инициализация CoreBrain
        logger.info("Начало инициализации CoreBrain...")
        brain = CoreBrain()
        logger.info("CoreBrain создан")
        
        # Инициализация системы
        logger.info("Начало инициализации компонентов...")
        init_result = brain.initialize()
        logger.info(f"Инициализация компонентов завершена: {init_result}")
        
        if not init_result:
            logger.error("Ошибка инициализации системы")
            return False
        
        # Запуск системы
        logger.info("Начало запуска системы...")
        start_result = brain.start()
        logger.info(f"Запуск системы завершен: {start_result}")
        
        if not start_result:
            logger.error("Ошибка запуска системы")
            return False
            
        logger.info("=" * 80)
        logger.info("COGNIFLEX УСПЕШНО ЗАПУЩЕН")
        logger.info("=" * 80)
        
        # Держим систему работающей 30 секунд для сбора логов
        logger.info("Система будет работать 30 секунд для сбора логов...")
        import time
        time.sleep(30)
        
        # Остановка системы
        logger.info("Начало остановки системы...")
        try:
            brain.stop()
            logger.info("Система успешно остановлена")
        except Exception as e:
            logger.error(f"Ошибка при остановке системы: {e}")
            logger.debug(traceback.format_exc())
        
        return True
        
    except Exception as e:
        logger.critical(f"КРИТИЧЕСКАЯ ОШИБКА ПРИ ЗАПУСКЕ: {e}")
        logger.critical(traceback.format_exc())
        return False
    finally:
        logger.info("=" * 80)
        logger.info("ЗАВЕРШЕНИЕ ДЕТАЛЬНОГО ЗАПУСКА COGNIFLEX")
        logger.info(f"Полный лог сохранен в: {log_path}")
        logger.info("=" * 80)

if __name__ == "__main__":
    main()
