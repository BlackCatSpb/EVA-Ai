"""Вспомогательные функции для CogniFlex"""
import logging
import os
import sys
from datetime import datetime

def setup_logging(log_dir: str = "logs", log_file: str = "cogniflex.log", level: int = logging.INFO) -> str:
    """
    Настраивает логирование для системы.
    
    Args:
        log_dir: Директория для логов
        log_file: Имя файла лога
        level: Уровень логирования
        
    Returns:
        str: Путь к файлу лога
    """
    # Создаем директорию для логов
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_dir)
    os.makedirs(log_path, exist_ok=True)
    
    # Полный путь к файлу лога
    log_file_path = os.path.join(log_path, log_file)
    
    # Настройка формата лога
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Создаем логгер
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Очищаем существующие обработчики
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Добавляем обработчик для файла
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    
    # Добавляем обработчик для консоли
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    # Логируем информацию о настройке
    logger.info(f"Логирование настроено. Логи будут записываться в: {log_file_path}")
    
    return log_file_path

class EthicalDecision:
    """Результат этической оценки контента."""
    
    def __init__(self, overall_score, violations, recommendations, principle_scores, 
                 confidence=1.0, timestamp=None):
        """
        Инициализирует результат этической оценки.
        
        Args:
            overall_score: Общий балл этичности (0.0-1.0)
            violations: Список выявленных нарушений
            recommendations: Рекомендации по устранению нарушений
            principle_scores: Оценки по отдельным этическим принципам
            confidence: Уровень уверенности в оценке
            timestamp: Временная метка
        """
        self.overall_score = overall_score
        self.violations = violations
        self.recommendations = recommendations
        self.principle_scores = principle_scores
        self.confidence = confidence
        self.timestamp = timestamp or datetime.now().timestamp()