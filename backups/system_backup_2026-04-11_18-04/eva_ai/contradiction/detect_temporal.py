"""Temporal contradiction detection (time-based conflicts)."""
import logging
import time
from typing import Dict, List, Any, Optional
from collections import defaultdict

logger = logging.getLogger("eva_ai.contradiction.detection.temporal")


class TemporalDetectionMixin:
    """Mixin providing temporal contradiction detection capabilities."""
    
    def detect_temporal_contradictions(self) -> List[Dict[str, Any]]:
        """
        Обнаруживает временные противоречия (факты, которые верны в разные периоды времени).
        
        Returns:
            List: Список обнаруженных временных противоречий
        """
        contradictions = []
        concepts = self.knowledge_graph.get_all_concepts()
        
        for concept in concepts:
            facts = self.knowledge_graph.get_facts_by_concept(concept)
            facts_by_period = defaultdict(list)
            for fact in facts:
                period = fact.get("temporal_context", "current")
                facts_by_period[period].append(fact)
            
            periods = list(facts_by_period.keys())
            for i in range(len(periods)):
                for j in range(i + 1, len(periods)):
                    period1 = periods[i]
                    period2 = periods[j]
                    
                    for fact1 in facts_by_period[period1]:
                        for fact2 in facts_by_period[period2]:
                            if self._are_facts_equivalent(fact1, fact2):
                                continue
                            divergence = self._calculate_divergence(fact1, fact2)
                            if divergence > self.detection_threshold:
                                contradiction = self._create_contradiction(
                                    concept, [fact1, fact2], divergence,
                                    relation_type="temporal",
                                    temporal_context=f"{period1} vs {period2}"
                                )
                                contradictions.append(contradiction)
        
        return contradictions
    
    def detect_contextual_contradictions(self) -> List[Dict[str, Any]]:
        """
        Обнаруживает контекстуальные противоречия (факты, которые верны в разных контекстах).
        
        Returns:
            List: Список обнаруженных контекстуальных противоречий
        """
        contradictions = []
        concepts = self.knowledge_graph.get_all_concepts()
        
        for concept in concepts:
            facts = self.knowledge_graph.get_facts_by_concept(concept)
            facts_by_context = defaultdict(list)
            for fact in facts:
                context = fact.get("context", "general")
                facts_by_context[context].append(fact)
            
            contexts = list(facts_by_context.keys())
            for i in range(len(contexts)):
                for j in range(i + 1, len(contexts)):
                    context1 = contexts[i]
                    context2 = contexts[j]
                    
                    for fact1 in facts_by_context[context1]:
                        for fact2 in facts_by_context[context2]:
                            if self._are_facts_equivalent(fact1, fact2):
                                continue
                            divergence = self._calculate_divergence(fact1, fact2)
                            if divergence > self.detection_threshold:
                                contradiction = self._create_contradiction(
                                    concept, [fact1, fact2], divergence,
                                    relation_type="contextual",
                                    context=f"{context1} vs {context2}"
                                )
                                contradictions.append(contradiction)
        
        return contradictions
