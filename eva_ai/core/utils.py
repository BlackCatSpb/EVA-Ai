"""Вспомогательные функции для ЕВА"""
import logging
import os
import sys
import io
import platform
from datetime import datetime
from logging.handlers import RotatingFileHandler


def setup_logging(log_dir: str = "diagnostic_logs", log_file: str = "eva_app.log", level: int = logging.INFO, max_bytes: int = 10*1024*1024, backup_count: int = 5):
    """Настраивает логирование с явным указанием кодировки utf-8."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    logger = logging.getLogger()
    logger.setLevel(level)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter('%(asctime)s - %(process)d - %(threadName)s - %(name)s - %(levelname)s - %(message)s')
    try:
        file_handler = RotatingFileHandler(
            log_path, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except (OSError, IOError, ValueError, RuntimeError) as e:
        print(f"Не удалось настроить файловый логгер: {e}")

    # Настройка UTF-8 для консоли Windows
    if platform.system() == 'Windows':
        try:
            # Попытка включить UTF-8 режим в консоли Windows
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # CP_UTF8 = 65001
            kernel32.SetConsoleOutputCP(65001)
            kernel32.SetConsoleCP(65001)
            # Попытка установить режим вывода UTF-8
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            if hasattr(sys.stderr, 'reconfigure'):
                sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    class UTF8StreamHandler(logging.StreamHandler):
        """Обработчик для корректного вывода UTF-8 в консоль."""
        def __init__(self, stream=None):
            super().__init__(stream)
            self._stream = stream
            
        def emit(self, record):
            try:
                msg = self.format(record)
                # Пробуем использовать utf-8 с заменой проблемных символов
                if hasattr(self.stream, 'encoding') and self.stream.encoding.lower() != 'utf-8':
                    # Консоль не поддерживает UTF-8 - используем ошибки замены
                    msg = msg.encode(self.stream.encoding, errors='replace').decode(self.stream.encoding)
                self.stream.write(msg + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)
    
    console_handler = UTF8StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.getLogger().critical("Необработанное исключение!", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception
    logging.info(f"Логирование настроено. Уровень: {logging.getLevelName(level)}. Логи: {log_path}")
    return logger


class EthicalDecision:
    """Результат этической оценки контента."""
    
    def __init__(self, overall_score, violations, recommendations, principle_scores, 
                 confidence=1.0, timestamp=None):
        self.overall_score = overall_score
        self.violations = violations
        self.recommendations = recommendations
        self.principle_scores = principle_scores
        self.confidence = confidence
        self.timestamp = timestamp or datetime.now().timestamp()