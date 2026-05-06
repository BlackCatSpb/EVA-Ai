"""
QueryRouter - динамический роутинг по типу запроса.
Классифицирует запросы и направляет по оптимальному пути.
"""
import re
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass

logger = logging.getLogger("eva_ai.query_router")

@dataclass
class QueryIntent:
    """Результат классификации запроса."""
    intent: str  # code, fact, creative, general
    confidence: float
    keywords: List[str]
    is_coding_query: bool
    is_factual_query: bool
    is_creative_query: bool

class QueryRouter:
    """
    Динамический роутинг запросов по типу.
    Использует простые правила + keyword matching для классификации.
    """
    
    CODE_KEYWORDS = [
        'код', 'программ', 'функци', 'класс', 'метод', 'print', 'def ', 'import ',
        'if ', 'for ', 'while ', 'return', 'variable', 'syntax', 'api', 'функция',
        'напиши код', 'напиши программ', 'реализуй', 'создай функцию',
        'debug', 'error', 'exception', 'lambda', 'async', 'await',
        'javascript', 'python', 'java', 'c++', 'sql', 'html', 'css'
    ]
    
    FACT_KEYWORDS = [
        'что такое', 'кто такой', 'какой', 'какая', 'какое', 'когда', 'почему',
        'сколько', 'где', 'определение', 'факт', 'информация', 'данные',
        'история', 'дата', 'число', 'процент', 'название', 'столица',
        'сколько лет', 'чем отличается', 'как работает', 'объясни'
    ]
    
    CREATIVE_KEYWORDS = [
        'стих', 'поэм', 'рассказ', 'истори', 'придумай', 'создай故事',
        'напиши текст', 'сочинени', 'креатив', 'воображение', 'фантази',
        'юмор', 'шутка', 'анекдот', 'метафора', 'аналоги', 'образ',
        'музык', 'песн', 'текст песни', 'сценарий', 'диалог'
    ]
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
    def classify(self, query: str) -> QueryIntent:
        """
        Классифицирует запрос и возвращает тип.
        
        Returns:
            QueryIntent с типом и уверенностью
        """
        query_lower = query.lower()
        
        # Подсчёт совпадений по категориям
        code_score = self._count_matches(query_lower, self.CODE_KEYWORDS)
        fact_score = self._count_matches(query_lower, self.FACT_KEYWORDS)
        creative_score = self._count_matches(query_lower, self.CREATIVE_KEYWORDS)
        
        # Определяем тип
        max_score = max(code_score, fact_score, creative_score)
        
        if max_score == 0:
            intent = "general"
            confidence = 0.5
        elif max_score == code_score:
            intent = "code"
            confidence = min(0.9, 0.5 + code_score * 0.1)
        elif max_score == fact_score:
            intent = "fact"
            confidence = min(0.9, 0.5 + fact_score * 0.1)
        elif max_score == creative_score:
            intent = "creative"
            confidence = min(0.9, 0.5 + creative_score * 0.1)
        else:
            intent = "general"
            confidence = 0.5
        
        # Извлекаем ключевые слова
        keywords = self._extract_keywords(query_lower)
        
        return QueryIntent(
            intent=intent,
            confidence=confidence,
            keywords=keywords,
            is_coding_query=(intent == "code"),
            is_factual_query=(intent == "fact"),
            is_creative_query=(intent == "creative")
        )
    
    def _count_matches(self, query: str, keywords: List[str]) -> int:
        """Подсчитывает количество совпадений с ключевыми словами."""
        count = 0
        for kw in keywords:
            if kw in query:
                count += 1
        return count
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Извлекает ключевые слова из запроса."""
        words = re.findall(r'\b\w{3,}\b', query)
        return list(set(words))[:10]
    
    def get_routing_decision(self, intent: QueryIntent) -> Dict[str, any]:
        """
        Возвращает решение о роутинге на основе классификации.
        
        Returns:
            Dict с параметрами роутинга
        """
        if intent.is_coding_query:
            return {
                "route": "code",
                "skip_model_a": False,
                "use_model_c": True,
                "max_tokens": 2048,
                "temperature": 0.3,
                "description": "Code query - use full pipeline with higher tokens"
            }
        elif intent.is_factual_query:
            return {
                "route": "fact",
                "skip_model_a": True,  # После Model A - финальная проверка
                "use_model_c": False,
                "max_tokens": 512,
                "temperature": 0.2,
                "description": "Factual query - short answer, stop after A"
            }
        elif intent.is_creative_query:
            return {
                "route": "creative",
                "skip_model_a": False,
                "use_model_c": False,
                "max_tokens": 1024,
                "temperature": 0.8,
                "description": "Creative query - full pipeline with high creativity"
            }
        else:
            return {
                "route": "general",
                "skip_model_a": False,
                "use_model_c": False,
                "max_tokens": 1024,
                "temperature": 0.7,
                "description": "General query - standard pipeline"
            }


def create_query_router(config: Optional[Dict] = None) -> QueryRouter:
    """Создаёт инстанс QueryRouter."""
    return QueryRouter(config)
