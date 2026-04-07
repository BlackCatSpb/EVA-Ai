"""
Knowledge Rollback - механизм отката знаний при обнаружении противоречий.
"""

import time
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("eva_ai.knowledge_rollback")


class KnowledgeRollback:
    """Механизм отката знаний при обнаружении противоречий."""

    def __init__(self, graph_learning=None, event_bus=None):
        self.graph_learning = graph_learning
        self.event_bus = event_bus
        self.rollback_history = []
        self.logger = logging.getLogger("eva_ai.knowledge_rollback")
        self.min_confidence = 0.1
        self.confidence_penalty = 0.5
    
    def set_graph_learning(self, graph_learning):
        """Установить graph_learning после инициализации."""
        self.graph_learning = graph_learning
    
    def set_event_bus(self, event_bus):
        """Установить event_bus после инициализации."""
        self.event_bus = event_bus
    
    def rollback_knowledge(self, experience_id: str, reason: str) -> bool:
        """Откатить знание при подтверждённом противоречии."""
        try:
            if not self.graph_learning:
                logger.warning("GraphLearning не доступен для rollback")
                return False
            
            if not hasattr(self.graph_learning, 'get_experience'):
                logger.warning("GraphLearning не имеет метода get_experience")
                return False
            
            exp = self.graph_learning.get_experience(experience_id)
            if not exp:
                logger.info(f"Опыт {experience_id} не найден для rollback")
                return False
            
            old_confidence = exp.get('confidence', 1.0)
            exp['confidence'] = max(self.min_confidence, old_confidence - self.confidence_penalty)
            
            if hasattr(self.graph_learning, 'update_experience'):
                self.graph_learning.update_experience(exp)
            elif hasattr(self.graph_learning, 'experiences'):
                self.graph_learning.experiences[experience_id] = exp
            
            self.rollback_history.append({
                'experience_id': experience_id,
                'reason': reason,
                'old_confidence': old_confidence,
                'new_confidence': exp['confidence'],
                'timestamp': time.time()
            })
            
            self._publish_rollback_event(experience_id, reason, exp['confidence'])
            
            logger.info(f"Rollback: {experience_id}, confidence {old_confidence:.2f} -> {exp['confidence']:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка rollback: {e}")
            return False
    
    def _publish_rollback_event(self, experience_id: str, reason: str, new_confidence: float):
        """Публикует событие rollback в EventBus."""
        if not self.event_bus:
            return
        
        try:
            from eva_ai.core.event_bus import Event, EventPriority
            
            event = Event(
                event_type="knowledge.rolled_back",
                source="knowledge_rollback",
                data={
                    'experience_id': experience_id,
                    'reason': reason,
                    'new_confidence': new_confidence
                },
                timestamp=time.time(),
                priority=EventPriority.NORMAL
            )
            self.event_bus.publish(event)
        except Exception as e:
            logger.debug(f"Событие rollback не опубликовано: {e}")
    
    def get_rollback_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Возвращает историю rollback операций."""
        return self.rollback_history[-limit:]
    
    def check_and_rollback(self, experience_id: str, contradiction_score: float, threshold: float = 0.5) -> bool:
        """Проверить и выполнить rollback если score превышает threshold."""
        if contradiction_score > threshold:
            reason = f"Обнаружено семантическое противоречие (score={contradiction_score:.3f})"
            return self.rollback_knowledge(experience_id, reason)
        return False
