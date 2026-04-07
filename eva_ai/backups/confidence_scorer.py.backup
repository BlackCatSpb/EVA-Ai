"""
Confidence Scorer - оценка уверенности Self-Reasoning Engine
Формула: Confidence = (ethics × 0.30) + (contradiction × 0.30) + (knowledge × 0.40)
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


# Весовая формула из DESIGN.md
ETHICS_WEIGHT = 0.30
CONTRADICTION_WEIGHT = 0.30
KNOWLEDGE_WEIGHT = 0.40

CONFIDENCE_THRESHOLD = 0.75

# Адаптивные веса для coarse-to-fine рассуждения
# От общности к частичности: на ранних итерациях - общие факторы, на поздних - детали
ITERATION_WEIGHTS = {
    1: {'ethics': 0.40, 'contradiction': 0.25, 'knowledge': 0.35},  # Общий анализ - этика важна
    2: {'ethics': 0.30, 'contradiction': 0.35, 'knowledge': 0.35},  # Переход к деталям
    3: {'ethics': 0.20, 'contradiction': 0.40, 'knowledge': 0.40},  # Детальный анализ противоречий
    4: {'ethics': 0.15, 'contradiction': 0.40, 'knowledge': 0.45},   # Глубокие знания
    5: {'ethics': 0.10, 'contradiction': 0.35, 'knowledge': 0.55},  # Финальная проверка знаний
}

# Адаптивные пороги уверенности - снижаем после каждой итерации
ADAPTIVE_THRESHOLDS = {
    1: 0.80,  # Высокий порог для первой итерации
    2: 0.75,  # Стандартный
    3: 0.70,  # Снижаем
    4: 0.65,  # Еще снижаем
    5: 0.60,  # Минимальный порог
}


def get_adaptive_weights(iteration: int) -> Dict[str, float]:
    """
    Получить адаптивные веса для текущей итерации.
    От общности (итерация 1) к частичности (итерации 5+).
    
    Args:
        iteration: Номер текущей итерации (1-indexed)
        
    Returns:
        Dict с весами для ethics, contradiction, knowledge
    """
    if iteration <= 0:
        iteration = 1
    if iteration > 5:
        iteration = 5
    return ITERATION_WEIGHTS[iteration].copy()


def get_adaptive_threshold(iteration: int) -> float:
    """
    Получить адаптивный порог уверенности для текущей итерации.
    Снижаем порог по мере углубления анализа.
    
    Args:
        iteration: Номер текущей итерации (1-indexed)
        
        Returns:
            float: Порог уверенности
    """
    if iteration <= 0:
        iteration = 1
    if iteration > 5:
        iteration = 5
    return ADAPTIVE_THRESHOLDS[iteration]


def calculate_adaptive_confidence(
    ethics_result: Optional[Dict[str, Any]] = None,
    contradiction_result: Optional[Dict[str, Any]] = None,
    knowledge_result: Optional[Dict[str, Any]] = None,
    query: str = "",
    iteration: int = 1
) -> float:
    """
    Расчёт общей уверенности с адаптивными весами.
    От общности к частичности - на ранних итерациях общий анализ,
    на поздних - детальный.
    
    Args:
        ethics_result: Результат проверки этики
        contradiction_result: Результат проверки противоречий
        knowledge_result: Результат проверки знаний
        query: Оригинальный запрос
        iteration: Номер текущей итерации (1-indexed)
        
    Returns:
        float: Общая уверенность 0.0-1.0
    """
    weights = get_adaptive_weights(iteration)
    
    ethics_score = calculate_ethics_score(ethics_result)
    contradiction_score = calculate_contradiction_score(contradiction_result)
    knowledge_score = calculate_knowledge_score(knowledge_result, query)
    
    confidence = (
        ethics_score * weights['ethics'] +
        contradiction_score * weights['contradiction'] +
        knowledge_score * weights['knowledge']
    )
    
    return max(0.0, min(1.0, confidence))


def calculate_ethics_score(ethics_result: Optional[Dict[str, Any]]) -> float:
    """
    Расчёт оценки этичности
    Возвращает 0.0-1.0 где 1.0 = полностью этично
    """
    if ethics_result is None:
        return 0.5  # Нейтрально если нет результата
    
    try:
        overall_score = ethics_result.get('overall_score', 0.5)
        violations = ethics_result.get('violations', [])
        
        # Если есть серьёзные нарушения - снижаем оценку
        if violations:
            violation_count = len(violations)
            if violation_count >= 3:
                return 0.1
            elif violation_count >= 1:
                return 0.5 - (violation_count * 0.1)
        
        return overall_score
    
    except Exception as e:
        logger.warning(f"Ошибка расчёта ethics score: {e}")
        return 0.5


def calculate_contradiction_score(contradiction_result: Optional[Dict[str, Any]]) -> float:
    """
    Расчёт оценки отсутствия противоречий
    Возвращает 0.0-1.0 где 1.0 = нет противоречий
    """
    if contradiction_result is None:
        return 0.5
    
    try:
        # Handle both dict with 'contradictions' key and direct list
        if isinstance(contradiction_result, list):
            contradictions = contradiction_result
        else:
            contradictions = contradiction_result.get('contradictions', [])
        
        if not contradictions:
            return 1.0  # Нет противоречий - максимум
        
        # Чем больше противоречий - тем ниже оценка
        count = len(contradictions)
        if count >= 5:
            return 0.1
        elif count >= 3:
            return 0.3
        elif count >= 1:
            return 0.6
        
        return 1.0
    
    except Exception as e:
        logger.warning(f"Ошибка расчёта contradiction score: {e}")
        return 0.5


def calculate_knowledge_score(knowledge_result: Optional[Dict[str, Any]], 
                               query: str = "") -> float:
    """
    Расчёт оценки полноты знаний
    Возвращает 0.0-1.0 где 1.0 = достаточно знаний для ответа
    """
    if knowledge_result is None:
        # Проверяем есть ли известная информация в запросе
        if query:
            # Базовый评估 - чем короче запрос, тем выше уверенность
            word_count = len(query.split())
            if word_count <= 3:
                return 0.6
            elif word_count <= 10:
                return 0.4
            else:
                return 0.3
        return 0.5
    
    try:
        gaps = knowledge_result.get('gaps', [])
        coverage = knowledge_result.get('coverage', {})
        
        # Если есть значительные пробелы - снижаем оценку
        if gaps:
            gap_count = len(gaps)
            if gap_count >= 5:
                return 0.2
            elif gap_count >= 3:
                return 0.4
            elif gap_count >= 1:
                return 0.6
        
        # Проверяем coverage
        coverage_score = coverage.get('score', 0.5)
        return coverage_score
    
    except Exception as e:
        logger.warning(f"Ошибка расчёта knowledge score: {e}")
        return 0.5


def calculate_overall_confidence(
    ethics_result: Optional[Dict[str, Any]] = None,
    contradiction_result: Optional[Dict[str, Any]] = None,
    knowledge_result: Optional[Dict[str, Any]] = None,
    query: str = ""
) -> float:
    """
    Расчёт общей уверенности по формуле:
    Confidence = (ethics_score × 0.30) + (contradiction_score × 0.30) + (knowledge_score × 0.40)
    
    Returns: float 0.0-1.0
    """
    ethics_score = calculate_ethics_score(ethics_result)
    contradiction_score = calculate_contradiction_score(contradiction_result)
    knowledge_score = calculate_knowledge_score(knowledge_result, query)
    
    confidence = (
        ethics_score * ETHICS_WEIGHT +
        contradiction_score * CONTRADICTION_WEIGHT +
        knowledge_score * KNOWLEDGE_WEIGHT
    )
    
    # Ограничиваем диапазон
    confidence = max(0.0, min(1.0, confidence))
    
    logger.debug(
        f"Confidence: {confidence:.2f} = "
        f"ethics({ethics_score:.2f})×{ETHICS_WEIGHT} + "
        f"contradiction({contradiction_score:.2f})×{CONTRADICTION_WEIGHT} + "
        f"knowledge({knowledge_score:.2f})×{KNOWLEDGE_WEIGHT}"
    )
    
    return confidence


def should_terminate(confidence: float, threshold: float = CONFIDENCE_THRESHOLD, iteration: int = 1) -> bool:
    """
    Определяет нужно ли завершить рассуждение.
    Поддерживает адаптивные пороги для coarse-to-fine режима.
    
    Args:
        confidence: Текущая уверенность
        threshold: Базовый порог (игнорируется если используется адаптивный)
        iteration: Номер текущей итерации для адаптивного порога
        
    Returns:
        bool: True если нужно завершить
    """
    adaptive_threshold = get_adaptive_threshold(iteration)
    return confidence >= adaptive_threshold


def get_confidence_level(confidence: float) -> str:
    """
    Получить текстовый уровень уверенности
    """
    if confidence >= 0.9:
        return "Высокий"
    elif confidence >= 0.75:
        return "Достаточный"
    elif confidence >= 0.5:
        return "Средний"
    elif confidence >= 0.25:
        return "Низкий"
    else:
        return "Критически низкий"
