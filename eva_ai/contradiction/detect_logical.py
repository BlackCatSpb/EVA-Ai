"""Logical contradiction detection (A and not-A)."""
import logging
import time
import hashlib
from typing import Dict, List, Any, Optional

logger = logging.getLogger("eva_ai.contradiction.detection.logical")


class LogicalDetectionMixin:
    """Mixin providing logical contradiction detection capabilities."""
    
    def detect_hierarchy_contradictions(self) -> List[Dict[str, Any]]:
        """
        Обнаруживает противоречия в иерархии концептов.
        
        Returns:
            List: Список обнаруженных иерархических противоречий
        """
        contradictions = []
        concepts = self.knowledge_graph.get_all_concepts()
        
        for concept in concepts:
            parents = self.knowledge_graph.get_relations(concept, "is_a")
            if len(parents) > 1:
                for i in range(len(parents)):
                    for j in range(i + 1, len(parents)):
                        parent1 = parents[i]
                        parent2 = parents[j]
                        if self._are_concepts_mutually_exclusive(parent1, parent2):
                            contradiction = self._create_contradiction(
                                concept,
                                [{"concept": concept, "relation": "is_a", "value": parent1},
                                 {"concept": concept, "relation": "is_a", "value": parent2}],
                                0.8,
                                relation_type="hierarchy"
                            )
                            contradictions.append(contradiction)
            
            if self._has_cyclic_dependency(concept):
                cycle = self._find_cyclic_dependency(concept)
                if cycle:
                    facts = [{"concept": cycle[i], "relation": "is_a", "value": cycle[(i+1) % len(cycle)]} 
                            for i in range(len(cycle))]
                    contradiction = self._create_contradiction(
                        concept, facts, 0.9, relation_type="hierarchy_cycle"
                    )
                    contradictions.append(contradiction)
        
        return contradictions
    
    def _are_concepts_mutually_exclusive(self, concept1: str, concept2: str) -> bool:
        """
        Проверяет, являются ли два концепта взаимоисключающими.
        
        Args:
            concept1: Первый концепт
            concept2: Второй концепт
            
        Returns:
            bool: Являются ли концепты взаимоисключающими
        """
        exclusive_pairs = [
            ("мужчина", "женщина"),
            ("живой", "мертвый"),
            ("истина", "ложь"),
            ("включение", "выключение"),
            ("свет", "тьма"),
            ("тепло", "холод")
        ]
        
        for pair in exclusive_pairs:
            if (concept1.lower() in pair and concept2.lower() in pair) and (pair[0] != pair[1]):
                return True
        return False
    
    def _has_cyclic_dependency(self, concept: str) -> bool:
        """
        Проверяет, есть ли циклическая зависимость для концепта.
        
        Args:
            concept: Концепт
            
        Returns:
            bool: Есть ли циклическая зависимость
        """
        visited = set()
        path = []
        
        def dfs(node):
            if node in visited:
                return False
            if node in path:
                return True
            visited.add(node)
            path.append(node)
            children = self.knowledge_graph.get_relations(node, "is_a")
            for child in children:
                if dfs(child):
                    return True
            path.pop()
            return False
        
        return dfs(concept)
    
    def _find_cyclic_dependency(self, concept: str) -> Optional[List[str]]:
        """
        Находит циклическую зависимость для концепта.
        
        Args:
            concept: Концепт
            
        Returns:
            List: Цикл зависимостей
        """
        def dfs(node, current_path):
            if node in current_path:
                idx = current_path.index(node)
                return current_path[idx:]
            current_path.append(node)
            children = self.knowledge_graph.get_relations(node, "is_a")
            for child in children:
                cycle = dfs(child, current_path.copy())
                if cycle:
                    return cycle
            return None
        
        return dfs(concept, [])
    
    def detect_exclusivity_contradictions(self) -> List[Dict[str, Any]]:
        """
        Обнаруживает противоречия эксклюзивности.
        
        Returns:
            List: Список обнаруженных противоречий эксклюзивности
        """
        contradictions = []
        concepts = self.knowledge_graph.get_all_concepts()
        
        for concept in concepts:
            only_relations = self.knowledge_graph.get_relations(concept, "only_in")
            not_only_relations = self.knowledge_graph.get_relations(concept, "not_only_in")
            
            for only_rel in only_relations:
                for not_only_rel in not_only_relations:
                    if only_rel == not_only_rel:
                        contradiction = self._create_contradiction(
                            concept,
                            [{"concept": concept, "relation": "only_in", "value": only_rel},
                             {"concept": concept, "relation": "not_only_in", "value": not_only_rel}],
                            0.85,
                            relation_type="exclusivity"
                        )
                        contradictions.append(contradiction)
        
        return contradictions
