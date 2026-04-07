"""Resolution strategies, merging, prioritization."""
import logging
import time
from typing import Dict, List, Optional, Any
from collections import defaultdict

logger = logging.getLogger("eva_ai.contradiction.core.resolution")


class ResolutionMixin:
    """Mixin providing resolution strategies, merging, and prioritization."""
    
    def resolve_contradiction(self, contradiction_id: str, resolution: Dict[str, Any],
                             resolver: str, confidence: float) -> bool:
        """Разрешает противоречие с указанным ID."""
        try:
            if contradiction_id not in self.contradictions:
                logger.error(f"Противоречие {contradiction_id} не найдено")
                return False
            contradiction = self.contradictions[contradiction_id]
            contradiction.add_resolution_history(resolver, resolution, confidence)
            contradiction.update_status("resolved", resolution)
            self._save_contradictions()
            logger.info(f"Противоречие {contradiction_id} разрешено: {resolution}")
            return True
        except Exception as e:
            logger.error(f"Ошибка разрешения противоречия {contradiction_id}: {e}", exc_info=True)
            return False
    
    def get_active_contradictions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Возвращает активные (неразрешенные) противоречия."""
        active = [c.to_dict() for c in self.contradictions.values() if not c.is_resolved()]
        return sorted(active, key=lambda x: x.get("resolution_priority", 0.0), reverse=True)[:limit]
    
    def get_detected_contradictions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Совместимый метод для старого API - возвращает активные противоречия."""
        return self.get_active_contradictions(limit)
    
    def get_all_contradictions(self) -> List[Dict[str, Any]]:
        """Возвращает все противоречия (активные и разрешенные)."""
        return [c.to_dict() for c in self.contradictions.values()]
    
    def merge_contradictions(self, id1: str, id2: str) -> Optional[str]:
        """Объединяет два противоречия в одно."""
        if id1 not in self.contradictions or id2 not in self.contradictions:
            logger.warning("One or both contradictions not found for merging")
            return None
        
        c1 = self.contradictions[id1]
        c2 = self.contradictions[id2]
        
        if c1.concept != c2.concept:
            logger.warning("Cannot merge contradictions with different concepts")
            return None
        
        merged_facts = c1.conflicting_facts + c2.conflicting_facts
        merged_divergence = max(c1.divergence_level, c2.divergence_level)
        merged_id = f"merged_{id1}_{id2}"
        
        from .core_detection import Contradiction
        merged = Contradiction(
            contradiction_id=merged_id,
            concept=c1.concept,
            conflicting_facts=merged_facts,
            divergence_level=merged_divergence,
            metadata={**c1.metadata, **c2.metadata, "merged_from": [id1, id2]}
        )
        
        self.contradictions[merged_id] = merged
        del self.contradictions[id1]
        del self.contradictions[id2]
        self._save_contradictions()
        
        logger.info(f"Merged contradictions {id1} and {id2} into {merged_id}")
        return merged_id
    
    def prioritize_contradictions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Приоритизирует противоречия по серьезности и возрасту."""
        now = time.time()
        def priority_score(c):
            severity_scores = {"critical": 1.0, "high": 0.8, "medium": 0.5, "low": 0.2}
            sev = severity_scores.get(c.severity, 0.3)
            age_days = (now - c.timestamp) / 86400
            age_factor = min(1.0, age_days / 30.0)
            return sev * 0.7 + age_factor * 0.3
        
        sorted_contradictions = sorted(
            [c for c in self.contradictions.values() if not c.is_resolved()],
            key=priority_score, reverse=True
        )
        return [c.to_dict() for c in sorted_contradictions[:limit]]
