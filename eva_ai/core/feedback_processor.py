from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeedbackData:
    rating: int = 3
    explicit_accuracy: float = 0.5
    coherence_score: float = 0.5
    helpfulness: float = 0.5
    toxicity: float = 0.0
    corrected_answer: Optional[str] = None
    preferred_response: Optional[str] = None
    reasoning_quality: float = 0.5
    message_index: int = 0
    timestamp: float = 0
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


class FeedbackProcessor:
    """Обработка многоуровневой обратной связи"""
    
    def __init__(self, graph_learning=None, event_bus=None):
        self.graph_learning = graph_learning
        self.event_bus = event_bus
        self.feedback_history: List[FeedbackData] = []
    
    def process_feedback(self, feedback: Dict[str, Any]) -> bool:
        """Обработать feedback и обновить граф знаний"""
        
        validated = self._validate_feedback(feedback)
        if not validated:
            logger.warning("Feedback validation failed")
            return False
        
        fb_data = self._to_feedback_data(feedback)
        
        self._update_experience_from_feedback(fb_data)
        
        self._update_edge_weights(fb_data)
        
        self.feedback_history.append(fb_data)
        
        if self.event_bus:
            try:
                from eva_ai.core.event_bus import Event, EventTypes
                event = Event(
                    event_type=EventTypes.FEEDBACK_PROCESSED,
                    source="feedback_processor",
                    data={
                        'rating': fb_data.rating,
                        'message_index': fb_data.message_index,
                        'explicit_accuracy': fb_data.explicit_accuracy
                    },
                    timestamp=time.time()
                )
                self.event_bus.publish(event)
            except Exception as e:
                logger.debug(f"Event publish error: {e}")
        
        logger.info(f"Feedback processed: rating={fb_data.rating}, accuracy={fb_data.explicit_accuracy}")
        return True
    
    def _update_experience_from_feedback(self, fb: FeedbackData):
        """Обновить ExperienceNode на основе feedback"""
        if not self.graph_learning:
            return
        
        try:
            experiences = self._get_recent_experiences(limit=fb.message_index + 1)
            if experiences:
                exp = experiences[-1]
                
                if hasattr(exp, 'quality_score'):
                    exp.quality_score = fb.explicit_accuracy
                
                if hasattr(exp, 'usage_count'):
                    if fb.rating >= 4:
                        exp.usage_count = min(exp.usage_count + 1, 100)
                    elif fb.rating <= 1:
                        exp.usage_count = max(exp.usage_count - 1, 0)
                    
                    self._save_experience(exp)
        except Exception as e:
            logger.debug(f"Experience update error: {e}")
    
    def _update_edge_weights(self, fb: FeedbackData):
        """Обновить веса рёбер графа на основе feedback"""
        if not self.graph_learning:
            return
        
        weight_delta = 0.1 if fb.rating >= 4 else -0.1
        
        try:
            if hasattr(self.graph_learning, 'update_edge_weight'):
                self.graph_learning.update_edge_weight(
                    from_node=fb.message_index,
                    delta=weight_delta
                )
            elif hasattr(self.graph_learning, 'coordinator'):
                if hasattr(self.graph_learning.coordinator, 'update_edge_weight'):
                    self.graph_learning.coordinator.update_edge_weight(
                        from_node=fb.message_index,
                        delta=weight_delta
                    )
        except Exception as e:
            logger.debug(f"Edge weight update error: {e}")
    
    def _get_recent_experiences(self, limit: int = 10) -> List:
        """Получить недавние опыты"""
        try:
            if hasattr(self.graph_learning, '_load_experiences'):
                experiences = self.graph_learning._load_experiences()
                return experiences[-limit:] if experiences else []
            elif hasattr(self.graph_learning, 'learning_loop'):
                experiences = self.graph_learning.learning_loop._load_experiences()
                return experiences[-limit:] if experiences else []
        except Exception as e:
            logger.debug(f"Get experiences error: {e}")
        return []
    
    def _save_experience(self, exp):
        """Сохранить опыт"""
        try:
            if hasattr(self.graph_learning, '_save_experience'):
                self.graph_learning._save_experience(exp)
            elif hasattr(self.graph_learning, 'learning_loop'):
                self.graph_learning.learning_loop._save_experience(exp)
        except Exception as e:
            logger.debug(f"Save experience error: {e}")
    
    def _validate_feedback(self, feedback: Dict) -> bool:
        """Валидация feedback данных"""
        required = ['rating']
        if not all(k in feedback for k in required):
            return False
        
        rating = feedback.get('rating', 0)
        if not isinstance(rating, (int, float)) or rating < 0 or rating > 5:
            return False
        
        return True
    
    def _to_feedback_data(self, feedback: Dict) -> FeedbackData:
        """Преобразование в FeedbackData с умолчаниями"""
        return FeedbackData(
            rating=int(feedback.get('rating', 3)),
            explicit_accuracy=float(feedback.get('explicit_accuracy', 0.5)),
            coherence_score=float(feedback.get('coherence_score', 0.5)),
            helpfulness=float(feedback.get('helpfulness', 0.5)),
            toxicity=float(feedback.get('toxicity', 0.0)),
            corrected_answer=feedback.get('corrected_answer'),
            preferred_response=feedback.get('preferred_response'),
            reasoning_quality=float(feedback.get('reasoning_quality', 0.5)),
            message_index=int(feedback.get('message_index', 0))
        )
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Получить статистику по feedback"""
        if not self.feedback_history:
            return {
                'total': 0,
                'avg_rating': 0,
                'avg_accuracy': 0,
                'positive_count': 0,
                'negative_count': 0
            }
        
        ratings = [fb.rating for fb in self.feedback_history]
        accuracies = [fb.explicit_accuracy for fb in self.feedback_history]
        
        return {
            'total': len(self.feedback_history),
            'avg_rating': sum(ratings) / len(ratings),
            'avg_accuracy': sum(accuracies) / len(accuracies),
            'positive_count': sum(1 for r in ratings if r >= 4),
            'negative_count': sum(1 for r in ratings if r <= 1),
            'recent': [fb.rating for fb in self.feedback_history[-10:]]
        }
