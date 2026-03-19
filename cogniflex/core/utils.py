"""Вспомогательные функции для CogniFlex"""
import logging
import os
import sys
import io
from datetime import datetime
from logging.handlers import RotatingFileHandler

def setup_logging(log_dir: str = "diagnostic_logs", log_file: str = "cogniflex_app.log", level: int = logging.INFO, max_bytes: int = 10*1024*1024, backup_count: int = 5):
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

    class SafeStreamHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                msg = self.format(record)
                stream = self.stream
                if hasattr(stream, 'encoding') and stream.encoding and stream.encoding.lower() in ('cp1251', 'cp866'):
                    msg = msg.encode('utf-8', 'replace').decode('utf-8', 'replace')
                stream.write(msg + self.terminator)
                self.flush()
            except (AttributeError, OSError, IOError, UnicodeError, RuntimeError):
                self.handleError(record)
    
    console_handler = SafeStreamHandler()
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