"""
Graph ML Training - Training methods and pattern learning for MemoryGraphML.
"""

import time
import logging
import numpy as np
from typing import Dict, Any, Optional, List

logger = logging.getLogger("eva_ai.memory_graph_ml")


def generate_training_sample(self, concept_id: str = None) -> Optional[Dict]:
    """Генерация тренировочного примера из графа"""
    try:
        if not concept_id and self.embeddings:
            concept_id = np.random.choice(list(self.embeddings.keys()))
        
        if concept_id not in self.embeddings:
            return None
        
        embedding = self.embeddings[concept_id]
        related = self._get_related_concepts(concept_id, depth=1)
        
        description = f"Концепт {embedding.node_id}"
        if related:
            description += f" связан с: {', '.join(related[:5])}"
        
        sample = {
            'input': embedding.node_id,
            'context': description,
            'related_concepts': related,
            'embedding': embedding.vector.tolist(),
            'timestamp': time.time()
        }
        
        return sample
        
    except Exception as e:
        logger.debug(f"Ошибка генерации примера: {e}")
        return None


def _extract_patterns(self):
    """Извлечение паттернов из графа"""
    try:
        if not hasattr(self, 'brain') or not self.brain:
            return
        kg = getattr(self.brain, 'knowledge_graph', None)
        if not kg:
            return
        
        patterns_found = []
        
        if hasattr(kg, 'get_all_relations'):
            relations = kg.get_all_relations()
            
            from collections import defaultdict
            node_relations = defaultdict(list)
            for rel in relations:
                from_node = rel.get('from') or rel.get('source')
                to_node = rel.get('to') or rel.get('target')
                rel_type = rel.get('type') or rel.get('relation_type')
                
                if from_node and to_node and rel_type:
                    node_relations[from_node].append((to_node, rel_type))
            
            pattern_counts = defaultdict(int)
            for start_node, first_level in node_relations.items():
                for mid_node, rel1 in first_level:
                    if mid_node in node_relations:
                        for end_node, rel2 in node_relations[mid_node]:
                            if end_node != start_node:
                                pattern_key = f"{rel1}->{rel2}"
                                pattern_counts[pattern_key] += 1
            
            from .graph_ml_core import GraphPattern
            for pattern_key, count in pattern_counts.items():
                if count >= self.min_pattern_frequency:
                    parts = pattern_key.split("->")
                    if len(parts) == 2:
                        patterns_found.append(GraphPattern(
                            pattern_id=f"pattern_{hash(pattern_key) % 100000}",
                            nodes=[],
                            relations=[("A", "B", parts[0]), ("B", "C", parts[1])],
                            frequency=count,
                            confidence=min(0.95, 0.5 + count * 0.05),
                            context=f"Frequent pattern: {pattern_key}"
                        ))
        
        patterns_found.sort(key=lambda p: p.frequency, reverse=True)
        self.patterns = patterns_found[:self.max_patterns]
        
        logger.debug(f"Извлечено {len(self.patterns)} паттернов")
        
    except Exception as e:
        logger.debug(f"Ошибка извлечения паттернов: {e}")


def _setup_training(self):
    """Setup training methods on the MemoryGraphML class."""
    pass
