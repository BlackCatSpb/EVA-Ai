"""
Ambiguity Resolver - stub for backwards compatibility
Redirects to FractalGraph v2 functionality
"""
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class AmbiguityResolver:
    """
    Resolver for ambiguous entities - uses FGv2 for context
    """
    
    def __init__(self, brain=None, fractal_graph=None):
        self.brain = brain
        self._fg = fractal_graph
        if brain and hasattr(brain, 'fractal_graph_v2'):
            self._fg = brain.fractal_graph_v2
        logger.debug("AmbiguityResolver initialized (FGv2 stub)")
    
    def resolve(self, text: str, context: str = "") -> Optional[str]:
        """Resolve ambiguity in text using context"""
        if not self._fg:
            return text
        
        results = self._fg.semantic_search(context or text, top_k=3)
        if results:
            return results[0].get('content', text)
        return text
    
    def find_disambiguations(self, entity: str) -> List[Dict[str, Any]]:
        """Find possible disambiguations for an entity"""
        if not self._fg:
            return []
        
        results = self._fg.semantic_search(entity, top_k=5)
        return [
            {
                'text': r.get('content', '')[:100],
                'score': r.get('score', 0),
                'type': 'concept'
            }
            for r in results
        ]
