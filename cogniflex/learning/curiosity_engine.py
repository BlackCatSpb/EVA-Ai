"""
Curiosity Engine for CogniFlex
Generates self-learning questions based on detected gaps and entities.
"""
import logging
import time
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("cogniflex.curiosity")

class CuriosityType(Enum):
    ENTITY_EXPLORATION = "entity_exploration"
    KNOWLEDGE_GAP = "knowledge_gap"
    PATTERN_DISCOVERY = "pattern_discovery"
    TOPIC_EXPANSION = "topic_expansion"
    UNCERTAINTY = "uncertainty"

@dataclass
class CuriosityTrigger:
    trigger_id: str
    trigger_type: CuriosityType
    topic: str
    confidence: float
    source_text: str
    timestamp: float
    related_entities: List[str]
    learning_questions: List[str]

class CuriosityEngine:
    """
    Engine that drives system curiosity and self-learning.
    
    Detects:
    - Unknown entities in queries
    - Knowledge gaps (things system doesn't know)
    - Patterns for exploration
    - Uncertainty markers
    
    Generates:
    - Self-learning questions
    - Topics for web search
    - Entity exploration tasks
    """
    
    def __init__(self, brain=None, config: Optional[Dict] = None):
        self.brain = brain
        self.config = config or {}
        
        self.unknown_entity_threshold = self.config.get('unknown_entity_threshold', 0.3)
        self.gap_confidence_threshold = self.config.get('gap_confidence_threshold', 0.5)
        self.curiosity_interval = self.config.get('curiosity_interval', 600)
        
        self.explored_entities = set()
        self.explored_topics = set()
        self.curiosity_history = []
        
        # Инициализируем менеджер возможностей обучения
        self.learning_opportunity_manager = None
        if brain and hasattr(brain, 'learning_opportunity_manager'):
            self.learning_opportunity_manager = brain.learning_opportunity_manager
        elif brain and hasattr(brain, 'analyzer_core'):
            from cogniflex.learning.learning_opportunity_manager import LearningOpportunityManager
            self.learning_opportunity_manager = LearningOpportunityManager(brain=brain, analyzer_core=brain.analyzer_core)
        
        self.curiosity_patterns = [
            (r'\b(кто|who)\s+(это|is)\b', CuriosityType.ENTITY_EXPLORATION),
            (r'\b(что|what)\s+(такое|is)\b', CuriosityType.ENTITY_EXPLORATION),
            (r'\b(почему|why)\b', CuriosityType.KNOWLEDGE_GAP),
            (r'\b(как|how)\b', CuriosityType.TOPIC_EXPANSION),
            (r'\b(не\s+знаю|don\'t\s+know|uncertain)', CuriosityType.UNCERTAINTY),
        ]
    
    def detect_curiosity_triggers(self, text: str) -> List[CuriosityTrigger]:
        """
        Detect potential curiosity triggers in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of CuriosityTrigger objects
        """
        triggers = []
        
        entities = self._extract_entities(text)
        
        for entity in entities:
            if entity not in self.explored_entities:
                trigger = self._create_entity_trigger(entity, text)
                if trigger:
                    triggers.append(trigger)
        
        for pattern, trigger_type in self.curiosity_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                topic = self._extract_topic_from_pattern(text, pattern)
                trigger = CuriosityTrigger(
                    trigger_id=f"curiosity_{int(time.time())}_{len(triggers)}",
                    trigger_type=trigger_type,
                    topic=topic,
                    confidence=0.7,
                    source_text=text[:100],
                    timestamp=time.time(),
                    related_entities=entities,
                    learning_questions=self._generate_questions(trigger_type, topic, entities)
                )
                triggers.append(trigger)
        
        if triggers:
            self.curiosity_history.extend(triggers)
            if len(self.curiosity_history) > 100:
                self.curiosity_history = self.curiosity_history[-100:]
        
        return triggers[:5]
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract potential entities from text."""
        if hasattr(self.brain, 'entity_extractor'):
            try:
                result = self.brain.entity_extractor.extract_entities(text)
                if result:
                    return [e.get('text', e.get('entity', str(e))) for e in result]
            except:
                pass
        
        entities = []
        entities.extend(re.findall(r'"([^"]+)"', text))
        entities.extend(re.findall(r'\b[A-ZА-ЯЁ][a-zа-яё]+(?:\s+[A-ZА-ЯЁ][a-zа-яё]+)*\b', text))
        return list(set(entities))
    
    def _create_entity_trigger(self, entity: str, source_text: str) -> Optional[CuriosityTrigger]:
        """Create curiosity trigger for unknown entity."""
        known = self._is_entity_known(entity)
        
        if not known:
            return CuriosityTrigger(
                trigger_id=f"entity_{int(time.time())}_{hash(entity) % 10000}",
                trigger_type=CuriosityType.ENTITY_EXPLORATION,
                topic=entity,
                confidence=0.8,
                source_text=source_text[:100],
                timestamp=time.time(),
                related_entities=[],
                learning_questions=self._generate_questions(
                    CuriosityType.ENTITY_EXPLORATION, entity, []
                )
            )
        return None
    
    def _is_entity_known(self, entity: str) -> bool:
        """Check if entity is already in knowledge graph."""
        if hasattr(self.brain, 'knowledge_graph') and self.brain.knowledge_graph:
            try:
                result = self.brain.knowledge_graph.search(entity, limit=1)
                return bool(result)
            except:
                pass
        return entity in self.explored_entities
    
    def _extract_topic_from_pattern(self, text: str, pattern: str) -> str:
        """Extract topic from matched pattern."""
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return text[max(0, match.start()-20):match.end()+20]
        return text[:50]
    
    def _generate_questions(self, trigger_type: CuriosityType, topic: str, 
                          entities: List[str]) -> List[str]:
        """Generate learning questions based on trigger type."""
        questions = []
        
        if trigger_type == CuriosityType.ENTITY_EXPLORATION:
            questions = [
                f"Что такое {topic}?",
                f"Как {topic} связан с известными концепциями?",
                f"Где найти информацию о {topic}?",
            ]
        elif trigger_type == CuriosityType.KNOWLEDGE_GAP:
            questions = [
                f"Почему возникает вопрос о {topic}?",
                f"Какие факты известны о {topic}?",
            ]
        elif trigger_type == CuriosityType.UNCERTAINTY:
            questions = [
                f"Что именно неопределенно в {topic}?",
                f"Как можно уточнить {topic}?",
            ]
        else:
            questions = [
                f"Исследовать тему: {topic}",
                f"Найти примеры: {topic}",
            ]
        
        return questions[:3]
    
    def assess_knowledge_gap(self, topic: str) -> float:
        """
        Assess how much the system doesn't know about a topic.
        
        Returns:
            float: 0.0 (knows everything) to 1.0 (knows nothing)
        """
        gap_score = 1.0
        
        if hasattr(self.brain, 'knowledge_graph') and self.brain.knowledge_graph:
            try:
                results = self.brain.knowledge_graph.search(topic, limit=5)
                if results:
                    gap_score = max(0.1, 1.0 - len(results) * 0.2)
            except:
                pass
        
        if topic in self.explored_topics:
            gap_score *= 0.5
        
        return gap_score
    
    def trigger_self_learning(self, topic: str) -> Dict[str, Any]:
        """
        Trigger self-learning process for a topic.
        
        Returns:
            Dict with learning task details
        """
        logger.info(f"Triggering self-learning for topic: {topic}")
        
        self.explored_topics.add(topic)
        gap_score = self.assess_knowledge_gap(topic)
        
        # Пытаемся создать возможность для обучения
        learning_task = {
            'topic': topic,
            'gap_score': gap_score,
            'questions': self._generate_questions(CuriosityType.TOPIC_EXPANSION, topic, []),
            'timestamp': time.time(),
            'status': 'pending'
        }
        
        # Пытаемся добавить в менеджер возможностей обучения
        try:
            if hasattr(self, 'learning_opportunity_manager') and self.learning_opportunity_manager:
                gap_assessment = self.assess_knowledge_gap(topic)
                if gap_assessment > 0.3:
                    opportunity_id = self.learning_opportunity_manager.add_learning_opportunity(
                        concept=topic,
                        opportunity_type='expansion',
                        priority=gap_assessment,
                        domain='auto_detected',
                        evidence=[f'Curiosity gap detected: score={gap_assessment}'],
                        suggested_actions=[
                            f'Изучить тему: {topic}',
                            'Найти релевантные источники',
                            'Обновить знания в графе'
                        ]
                    )
                    learning_task['opportunity_id'] = opportunity_id
                    learning_task['status'] = 'learning_opportunity_created'
                    logger.info(f"Создана возможность для обучения: {opportunity_id}")
        except Exception as e:
            logger.warning(f"Не удалось создать возможность для обучения: {e}")
        
        return learning_task
    
    def get_curiosity_report(self) -> Dict[str, Any]:
        """Get summary of curiosity engine state."""
        return {
            'explored_entities': len(self.explored_entities),
            'explored_topics': len(self.explored_topics),
            'curiosity_history': len(self.curiosity_history),
            'recent_triggers': [
                {
                    'type': t.trigger_type.value,
                    'topic': t.topic,
                    'confidence': t.confidence
                }
                for t in self.curiosity_history[-5:]
            ]
        }
