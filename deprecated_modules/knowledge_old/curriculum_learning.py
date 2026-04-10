"""
Curriculum Learning Scheduler - планировщик обучения с приоритизацией.
Взвешивает запросы на обучение по частоте, релевантности, confidence графа.
"""
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import time

logger = logging.getLogger("eva_ai.curriculum_learning")

@dataclass
class LearningTask:
    """Задача обучения."""
    topic: str
    query: str
    priority: str  # CRITICAL, HIGH, NORMAL, LOW
    frequency_score: float
    relevance_score: float
    confidence_score: float
    total_weight: float
    created_at: float = field(default_factory=time.time)

class CurriculumScheduler:
    """
    Планировщик Curriculum Learning для ConceptMiner.
    Приоритизирует запросы на обучение по:
    - Частоте запросов пользователя
    - Релевантности домена
    - Текущей confidence графа
    """
    
    def __init__(self, brain=None, config: Optional[Dict] = None):
        self.brain = brain
        self.config = config or {}
        
        # Отслеживание частоты запросов по темам
        self._topic_frequency: Dict[str, int] = defaultdict(int)
        self._topic_last_update: Dict[str, float] = {}
        
        # Вес компонентов
        self.frequency_weight = self.config.get('frequency_weight', 0.3)
        self.relevance_weight = self.config.get('relevance_weight', 0.4)
        self.confidence_weight = self.config.get('confidence_weight', 0.3)
        
    def register_query(self, query: str, topics: List[str] = None):
        """
        Зарегистрировать запрос для обновления частотной статистики.
        
        Args:
            query: Текст запроса
            topics: Список тем (если известны)
        """
        if topics:
            for topic in topics:
                self._topic_frequency[topic] += 1
                self._topic_last_update[topic] = time.time()
        else:
            # Извлекаем простые темы из запроса (первые 2-3 слова)
            words = query.lower().split()[:3]
            if words:
                topic = ' '.join(words)
                self._topic_frequency[topic] += 1
                self._topic_last_update[topic] = time.time()
                
    def create_learning_task(self, query: str, unknown_topics: List[str],
                             graph_confidence: float = 0.5) -> LearningTask:
        """
        Создать задачу обучения с приоритетом.
        
        Args:
            query: Запрос пользователя
            unknown_topics: Неизвестные темы из запроса
            graph_confidence: Текущая confidence графа для этой темы
            
        Returns:
            LearningTask с рассчитанным приоритетом
        """
        # 1. Частота (нормализуем 0-1)
        max_freq = max(self._topic_frequency.values()) if self._topic_frequency else 1
        frequency_score = self._topic_frequency.get(unknown_topics[0] if unknown_topics else '', 0) / max_freq
        
        # 2. Релевантность (простая оценка по наличию в конфиге)
        relevance_score = self._calculate_relevance(unknown_topics)
        
        # 3. Confidence графа (низкая confidence = высокий приоритет)
        confidence_score = 1.0 - graph_confidence
        
        # Общий вес
        total_weight = (
            frequency_score * self.frequency_weight +
            relevance_score * self.relevance_weight +
            confidence_score * self.confidence_weight
        )
        
        # Определяем приоритет
        if total_weight > 0.7:
            priority = "CRITICAL"
        elif total_weight > 0.5:
            priority = "HIGH"
        elif total_weight > 0.3:
            priority = "NORMAL"
        else:
            priority = "LOW"
            
        return LearningTask(
            topic=unknown_topics[0] if unknown_topics else query[:50],
            query=query,
            priority=priority,
            frequency_score=frequency_score,
            relevance_score=relevance_score,
            confidence_score=confidence_score,
            total_weight=total_weight
        )
    
    def _calculate_relevance(self, topics: List[str]) -> float:
        """Рассчитать релевантность темы."""
        if not topics:
            return 0.5
            
        # Приоритетные домены (можно расширить через конфиг)
        priority_domains = [
            'технологи', 'программирован', 'искусственн* интеллект',
            'машинн* обучен', 'наука', 'медицин', 'финанс'
        ]
        
        for topic in topics:
            topic_lower = topic.lower()
            for domain in priority_domains:
                if domain.replace('*', '') in topic_lower:
                    return 0.9
                    
        return 0.5
    
    def get_priority_queue(self) -> List[LearningTask]:
        """Получить очередь задач, отсортированную по приоритету."""
        # В реальной реализации здесь был бы доступ к очереди ConceptMiner
        # Пока возвращаем пустой список - интеграция через brain
        return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику планировщика."""
        return {
            "tracked_topics": len(self._topic_frequency),
            "frequency_weights": {
                "frequency": self.frequency_weight,
                "relevance": self.relevance_weight,
                "confidence": self.confidence_weight
            },
            "top_topics": sorted(
                self._topic_frequency.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }


def create_curriculum_scheduler(brain=None, config: Optional[Dict] = None) -> CurriculumScheduler:
    """Создать инстанс планировщика."""
    return CurriculumScheduler(brain, config)