"""
Smart LoRA Router - Auto-select based on query analysis.

Automatically selects the best LoRA adapter based on query content.

Usage:
    router = SmartLoRARouter()
    lora_name = router.route("напиши код")
    # Returns: 'eva_code'
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger("eva_ai.smart_lora_router")


class SmartLoRARouter:
    """Auto-select LoRA based on query analysis."""
    
    # Query patterns for each LoRA type
    ROUTES = {
        'code': [
            'код', 'функц', 'def ', 'class ', 'import ',
            'программ', 'питон', 'python', 'javascript',
            'реализуй', 'напиши функц', 'создай класс'
        ],
        'creative': [
            'придумай', 'напиши', 'сочини', 'истор',
            'стих', 'рассказ', 'поззия', 'роман',
            'сценарий', 'диалог', 'текст'
        ],
        'knowledge': [
            'что такое', 'кто такой', 'объясни',
            'почему', 'как работает', 'опиши',
            'расскажи', 'дай определение', 'какой'
        ],
        'logic': [
            'логика', 'рассужд', 'сравни', 'анализ',
            'докажи', 'обоснуй', 'вывод', 'следствие',
            'если то', 'значит', 'поэтому'
        ],
    }
    
    # Default LoRA if no match
    DEFAULT = 'eva_knowledge'
    
    def __init__(self):
        self._cache: Dict[str, str] = {}
        self._stats = {'total': 0, 'hits': 0}
    
    def route(self, query: str) -> str:
        """
        Analyze query and return best LoRA name.
        
        Args:
            query: User query text
            
        Returns:
            LoRA name (e.g., 'eva_code', 'eva_knowledge')
        """
        if not query:
            return self.DEFAULT
        
        self._stats['total'] += 1
        query_lower = query.lower()
        
        # Check cache first
        cache_key = hash(query_lower)
        if cache_key in self._cache:
            self._stats['hits'] += 1
            return self._cache[cache_key]
        
        # Score each category
        scores = {}
        for lora_name, patterns in self.ROUTES.items():
            score = sum(1 for p in patterns if p in query_lower)
            scores[lora_name] = score
        
        # Find best match
        best = max(scores, key=scores.get)
        result = f'eva_{best}' if scores[best] > 0 else self.DEFAULT
        
        # Cache result
        self._cache[cache_key] = result
        
        logger.info(f"[LoRA Router] Query: {query[:30]}... → {result}")
        return result
    
    def route_with_confidence(self, query: str) -> Dict[str, any]:
        """Return with confidence score."""
        if not query:
            return {'lora': self.DEFAULT, 'confidence': 0.0}
        
        query_lower = query.lower()
        scores = {}
        
        for lora_name, patterns in self.ROUTES.items():
            score = sum(1 for p in patterns if p in query_lower)
            scores[lora_name] = score
        
        total = sum(scores.values())
        if total > 0:
            best = max(scores, key=scores.get)
            confidence = scores[best] / total
            return {
                'lora': f'eva_{best}',
                'confidence': confidence,
                'scores': scores
            }
        
        return {'lora': self.DEFAULT, 'confidence': 0.0, 'scores': scores}
    
    def get_stats(self) -> Dict[str, any]:
        """Get routing statistics."""
        return {
            'total': self._stats['total'],
            'cache_hits': self._stats['hits'],
            'cache_hit_rate': self._stats['hits'] / max(1, self._stats['total'])
        }
    
    def clear_cache(self):
        """Clear routing cache."""
        self._cache.clear()
        self._stats = {'total': 0, 'hits': 0}


def create_router() -> SmartLoRARouter:
    """Factory function."""
    return SmartLoRARouter()


# Standalone test
if __name__ == '__main__':
    router = SmartLoRARouter()
    
    # Test cases
    tests = [
        ('напиши код для функции', 'eva_code'),
        ('что такое искусственный интеллект', 'eva_knowledge'),
        ('придумай историю', 'eva_creative'),
        ('докажи логику', 'eva_logic'),
        ('привет', 'eva_knowledge'),  # default
    ]
    
    print("Testing SmartLoRARouter:")
    for query, expected in tests:
        result = router.route(query)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{query[:25]}...' → {result} (expected: {expected})")
    
    print(f"\nStats: {router.get_stats()}")