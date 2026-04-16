"""
Context Entity extraction - wrapper around reasoning EntityExtractor
Обеспечивает полную функциональность извлечения сущностей для всей системы
"""
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class AmbiguityType(Enum):
    """Types of ambiguity"""
    LEXICAL = "lexical"
    REFERENTIAL = "referential"
    STRUCTURAL = "structural"
    TEMPORAL = "temporal"


@dataclass
class AmbiguousEntity:
    """Represents an ambiguous entity"""
    text: str
    ambiguity_type: AmbiguityType
    possible_resolutions: List[str]
    context: str = ""


class EntityExtractor:
    """
    Entity extractor - полная функциональность извлечения сущностей
    
    Использует reasoning.entity_extractor для:
    - extract_from_query(): извлечение из запроса пользователя
    - extract_from_response(): извлечение из ответа модели
    - extract_from_contradiction(): извлечение из противоречий
    - extract_all(): полное извлечение со всех источников
    """
    
    def __init__(self, brain=None, fractal_graph=None):
        self.brain = brain
        self._fg = fractal_graph
        
        # EntityExtractor removed with EnhancedReasoningEngine
        self._reasoning_extractor = None
        logger.info("EntityExtractor: using FGv2 fallback only")
        
        # Fallback на FGv2 если reasoning недоступен
        if brain and hasattr(brain, 'fractal_graph_v2'):
            self._fg = brain.fractal_graph_v2
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Извлекает сущности из текста (query или response)"""
        if self._reasoning_extractor:
            # Используем reasoning extractor
            result = self._reasoning_extractor.extract_from_query(text)
            return [
                {
                    'text': e.name,
                    'type': e.entity_type,
                    'confidence': e.confidence,
                    'content': e.content,
                    'properties': e.properties
                }
                for e in result.entities
            ]
        
        # Fallback на semantic search
        if not self._fg:
            return []
        
        results = self._fg.semantic_search(text, top_k=5)
        return [
            {
                'text': r.get('content', '')[:50],
                'type': 'concept',
                'confidence': r.get('score', 0.5)
            }
            for r in results
        ]
    
    def extract_from_query(self, query: str) -> 'ExtractionResult':
        """Извлекает сущности из запроса пользователя"""
        if self._reasoning_extractor:
            return self._reasoning_extractor.extract_from_query(query)
        return self._ExtractionResult() if hasattr(self, '_ExtractionResult') else []
    
    def extract_from_response(self, response: str) -> 'ExtractionResult':
        """Извлекает сущности из ответа модели"""
        if self._reasoning_extractor:
            return self._reasoning_extractor.extract_from_response(response)
        return self._ExtractionResult() if hasattr(self, '_ExtractionResult') else []
    
    def extract_all(self, query: str, response: str, 
                    contradictions: List[Dict[str, Any]] = None,
                    weights: Dict[str, float] = None) -> List[Any]:
        """Извлекает все сущности из запроса, ответа и противоречий"""
        if self._reasoning_extractor:
            return self._reasoning_extractor.extract_all(
                query, response, contradictions or [], weights
            )
        return []
    
    def resolve_ambiguity(self, entity: str, context: str) -> Optional[str]:
        """Разрешает неоднозначность сущности используя контекст"""
        if self._reasoning_extractor:
            # Используем reasoning для извлечения из контекста
            result = self._reasoning_extractor.extract_from_query(context)
            if result.entities:
                # Возвращаем наиболее вероятную сущность
                best = max(result.entities, key=lambda e: e.confidence)
                return best.content
            return entity
        
        # Fallback на semantic search
        if not self._fg:
            return entity
        
        results = self._fg.semantic_search(context, top_k=3)
        if results:
            return results[0].get('content', entity)
        return entity
    
    def find_related_entities(self, entity: str, limit: int = 5) -> List[str]:
        """Находит связанные сущности"""
        if self._reasoning_extractor:
            result = self._reasoning_extractor.extract_from_query(entity)
            return [e.name for e in result.entities[:limit]]
        
        # Fallback
        if not self._fg:
            return []
        
        results = self._fg.semantic_search(entity, top_k=limit)
        return [r.get('content', '')[:50] for r in results]
    
    def save_to_knowledge_graph(self, entities: List[Any], 
                                 knowledge_graph: Any = None) -> int:
        """Сохраняет сущности в knowledge graph"""
        if self._reasoning_extractor:
            return self._reasoning_extractor.save_to_knowledge_graph(
                entities, knowledge_graph
            )
        return 0
    
    def format_for_self_learning(self, entities: List[Any]) -> str:
        """Форматирует сущности для самообучения"""
        if self._reasoning_extractor:
            return self._reasoning_extractor.format_for_self_learning(entities)
        return "Новых сущностей не обнаружено."


__all__ = ['EntityExtractor', 'AmbiguousEntity', 'AmbiguityType']