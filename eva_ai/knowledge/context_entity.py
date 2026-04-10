"""
Context Entity extraction - stub for backwards compatibility
Redirects to FractalGraph v2 functionality
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
    Entity extractor - redirects to FractalGraph v2
    """
    
    def __init__(self, brain=None, fractal_graph=None):
        self.brain = brain
        self._fg = fractal_graph
        if brain and hasattr(brain, 'fractal_graph_v2'):
            self._fg = brain.fractal_graph_v2
        logger.debug("EntityExtractor initialized (FGv2 stub)")
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text"""
        if not self._fg:
            return []
        
        # Use semantic search to find related concepts
        results = self._fg.semantic_search(text, top_k=5)
        return [
            {
                'text': r.get('content', '')[:50],
                'type': 'concept',
                'confidence': r.get('score', 0.5)
            }
            for r in results
        ]
    
    def resolve_ambiguity(self, entity: str, context: str) -> Optional[str]:
        """Resolve entity ambiguity using context"""
        if not self._fg:
            return entity
        
        results = self._fg.semantic_search(context, top_k=3)
        if results:
            return results[0].get('content', entity)
        return entity
    
    def find_related_entities(self, entity: str, limit: int = 5) -> List[str]:
        """Find entities related to the given entity"""
        if not self._fg:
            return []
        
        results = self._fg.semantic_search(entity, top_k=limit)
        return [r.get('content', '')[:50] for r in results]
