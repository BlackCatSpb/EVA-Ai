"""EntityFractalStore - Multi-level entity storage for CogniFlex.

Stores entities at different abstraction levels:
- Level 0: Raw tokens
- Level 1: Ambiguous terms
- Level 2: Clarified meanings
- Level 3: Concept definitions
- Level 4: Full understanding
"""
from __future__ import annotations
import time
import logging
import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict
import numpy as np

logger = logging.getLogger("cogniflex.fractal.entity_fractal_store")


@dataclass
class EntityLevelData:
    """Entity data at a specific fractal level."""
    level: int
    data: Any
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    confidence: float = 0.5
    parent_level: Optional[int] = None
    child_levels: List[int] = field(default_factory=list)


class EntityFractalStore:
    """Stores entities across 5 fractal abstraction levels."""
    
    LEVEL_NAMES = {
        0: "raw_tokens",
        1: "ambiguous_terms",
        2: "clarified_meanings",
        3: "concept_definitions",
        4: "full_understanding"
    }
    
    def __init__(self, brain=None, cache_dir: Optional[str] = None):
        self.brain = brain
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), 
            "..", 
            "core", 
            "cogniflex_cache", 
            "entity_fractal"
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self._entities: Dict[str, Dict[int, EntityLevelData]] = defaultdict(dict)
        self._entity_index: Dict[str, List[str]] = defaultdict(list)
        self._level_stats: Dict[int, int] = {i: 0 for i in range(5)}
        self._similarity_cache: Dict[str, np.ndarray] = {}
        
        self._embedding_dim = 128
        self._load_from_disk()
        
        logger.info(f"EntityFractalStore initialized with cache at {self.cache_dir}")
    
    def store_entity(self, entity, query_context: str) -> bool:
        """Store entity across all fractal levels."""
        try:
            entity_id = f"entity_{hash(str(entity.term)) % 1000000}"
            
            level_0_data = self._extract_tokens(entity.term)
            self._store_at_level(entity_id, 0, level_0_data, {'raw_tokens': entity.term})
            
            level_1_data = {
                'term': entity.term,
                'possible_meanings': entity.possible_meanings,
                'context': query_context
            }
            self._store_at_level(entity_id, 1, level_1_data, {
                'ambiguous': True,
                'meanings_count': len(entity.possible_meanings)
            })
            
            if hasattr(entity, 'resolved_meaning') and entity.resolved_meaning:
                self._store_at_level(entity_id, 2, {
                    'resolved_meaning': entity.resolved_meaning,
                    'confidence': entity.confidence
                }, {'clarified': True})
            else:
                self._store_at_level(entity_id, 2, None, {'clarified': False})
            
            self._update_level_3(entity_id, entity, query_context)
            
            self._update_level_4(entity_id)
            
            self._entity_index[entity.term.lower()].append(entity_id)
            
            self._save_entity_to_disk(entity_id)
            
            logger.debug(f"Entity stored at all levels: {entity.term}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing entity: {e}")
            return False
    
    def _store_at_level(
        self, 
        entity_id: str, 
        level: int, 
        data: Any, 
        metadata: Dict[str, Any]
    ):
        """Store entity data at specific level."""
        embedding = self._compute_level_embedding(level, data)
        
        level_data = EntityLevelData(
            level=level,
            data=data,
            embedding=embedding,
            metadata=metadata,
            timestamp=time.time(),
            confidence=metadata.get('confidence', 0.5)
        )
        
        self._entities[entity_id][level] = level_data
        self._level_stats[level] = self._level_stats.get(level, 0) + 1
        
        if level > 0 and (level - 1) in self._entities.get(entity_id, {}):
            level_data.parent_level = level - 1
            self._entities[entity_id][level - 1].child_levels.append(level)
    
    def _extract_tokens(self, text: str) -> List[str]:
        """Extract raw tokens from text (Level 0)."""
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _compute_level_embedding(self, level: int, data: Any) -> np.ndarray:
        """Compute embedding for entity at given level."""
        try:
            text = ""
            if isinstance(data, str):
                text = data
            elif isinstance(data, dict):
                text = " ".join(str(v) for v in data.values() if v)
            elif isinstance(data, list):
                text = " ".join(str(v) for v in data)
            
            np.random.seed(hash(text) % 2**32)
            embedding = np.random.randn(self._embedding_dim).astype(np.float32)
            level_weight = 1.0 - (level * 0.15)
            embedding = embedding * level_weight
            return embedding / np.linalg.norm(embedding)
            
        except Exception as e:
            logger.debug(f"Error computing embedding: {e}")
            np.random.seed(int(time.time()) % 2**32)
            return np.random.randn(self._embedding_dim).astype(np.float32) / np.linalg.norm(
                np.random.randn(self._embedding_dim)
            )
    
    def _update_level_3(self, entity_id: str, entity, query_context: str):
        """Update Level 3 with concept definitions."""
        concept_def = {
            'term': entity.term,
            'definition': self._generate_definition(entity, query_context),
            'related_concepts': self._find_related_concepts(entity.term),
            'usage_examples': [query_context] if query_context else []
        }
        
        self._store_at_level(entity_id, 3, concept_def, {
            'concept_defined': True,
            'completeness': 0.7
        })
    
    def _generate_definition(self, entity, context: str) -> str:
        """Generate concept definition from entity data."""
        if hasattr(entity, 'resolved_meaning') and entity.resolved_meaning:
            return f"{entity.term}: {entity.resolved_meaning}"
        elif hasattr(entity, 'possible_meanings') and entity.possible_meanings:
            return f"{entity.term}: One of {', '.join(entity.possible_meanings[:2])}"
        return f"{entity.term}: Undefined concept"
    
    def _find_related_concepts(self, term: str) -> List[str]:
        """Find related concepts from knowledge graph."""
        related = []
        if self.brain and hasattr(self.brain, 'knowledge_graph'):
            try:
                kg = self.brain.knowledge_graph
                nodes = kg.search_nodes(term, limit=3)
                related = [n.name for n in nodes if n.name != term]
            except Exception as e:
                logger.debug(f"Error finding related concepts: {e}")
        return related
    
    def _update_level_4(self, entity_id: str):
        """Update Level 4 with full understanding synthesis."""
        entity_data = self._entities.get(entity_id)
        if not isinstance(entity_data, dict):
            return
        
        level_0 = entity_data.get(0)
        level_1 = entity_data.get(1)
        level_2 = entity_data.get(2)
        level_3 = entity_data.get(3)
        
        if not all([level_0, level_1, level_2, level_3]):
            return
        
        full_understanding = {
            'term': level_1.data.get('term') if level_1.data else None,
            'core_meaning': level_2.data.get('resolved_meaning') if level_2.data else None,
            'concept': level_3.data.get('definition') if level_3.data else None,
            'understanding_level': self._compute_understanding_level(level_2, level_3),
            'confidence': self._compute_confidence(level_0, level_1, level_2, level_3)
        }
        
        self._store_at_level(entity_id, 4, full_understanding, {
            'synthesized': True,
            'levels_integrated': [0, 1, 2, 3]
        })
    
    def _compute_understanding_level(self, level_2, level_3) -> float:
        """Compute how well the entity is understood."""
        base = 0.0
        if level_2 and level_2.metadata.get('clarified'):
            base += 0.4
        if level_3 and level_3.metadata.get('concept_defined'):
            base += 0.3
        return min(1.0, base + 0.3)
    
    def _compute_confidence(
        self, 
        level_0, 
        level_1, 
        level_2, 
        level_3
    ) -> float:
        """Compute overall confidence score."""
        confidences = []
        for level in [level_0, level_1, level_2, level_3]:
            if level:
                confidences.append(level.confidence)
        
        if not confidences:
            return 0.5
        
        return sum(confidences) / len(confidences)
    
    def get_entity_at_level(self, entity_id: str, level: int) -> Optional[EntityLevelData]:
        """Get entity data at specific level."""
        entity_data = self._entities.get(entity_id)
        if isinstance(entity_data, dict):
            return entity_data.get(level)
        return None
    
    def get_entity_full(self, entity_id: str) -> Optional[Dict[int, EntityLevelData]]:
        """Get entity data at all levels."""
        return self._entities.get(entity_id)
    
    def get_entities_by_term(self, term: str) -> List[str]:
        """Get all entity IDs for a given term."""
        return self._entity_index.get(term.lower(), [])
    
    def update_clarification(
        self, 
        entity_term: str, 
        question: str, 
        answer: str
    ):
        """Update entity when clarification is received."""
        entity_ids = self.get_entities_by_term(entity_term)
        
        for entity_id in entity_ids:
            level_1 = self._entities[entity_id].get(1)
            if level_1 and not level_1.metadata.get('clarified'):
                level_1.data['resolved_meaning'] = answer
                level_1.metadata['clarified'] = True
                level_1.confidence = 1.0
                
                clarification_history = level_1.data.setdefault('clarification_history', [])
                clarification_history.append({
                    'question': question,
                    'answer': answer,
                    'timestamp': time.time()
                })
            
            level_2 = self._entities[entity_id].get(2)
            if level_2:
                level_2.data = {
                    'resolved_meaning': answer,
                    'confidence': 1.0
                }
                level_2.metadata['clarified'] = True
                level_2.confidence = 1.0
            
            self._update_level_3(entity_id, type('Entity', (), {
                'term': entity_term,
                'resolved_meaning': answer,
                'possible_meanings': []
            })(), question)
            
            self._update_level_4(entity_id)
            
            self._save_entity_to_disk(entity_id)
            
            logger.debug(f"Entity clarification updated: {entity_term}")
    
    def search_similar_entities(
        self, 
        query: str, 
        level: Optional[int] = None,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Search for similar entities using embeddings."""
        try:
            query_embedding = self._compute_level_embedding(0, query)
            
            similarities = []
            for entity_id, levels_data in self._entities.items():
                target_level = level if level is not None else 4
                
                level_data = levels_data.get(target_level)
                if not level_data or level_data.embedding is None:
                    continue
                
                similarity = float(np.dot(query_embedding, level_data.embedding))
                
                if similarity > 0.3:
                    similarities.append((entity_id, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching similar entities: {e}")
            return []
    
    def get_level_statistics(self) -> Dict[int, int]:
        """Get statistics about entity distribution across levels."""
        stats = {}
        for level in range(5):
            stats[level] = {
                'count': self._level_stats.get(level, 0),
                'name': self.LEVEL_NAMES[level]
            }
        return stats
    
    def _save_entity_to_disk(self, entity_id: str):
        """Save entity to disk for persistence."""
        try:
            entity_data = {}
            entity_levels = self._entities.get(entity_id)
            if not isinstance(entity_levels, dict):
                return
            for level, level_obj in entity_levels.items():
                entity_data[level] = {
                    'data': level_obj.data,
                    'metadata': level_obj.metadata,
                    'timestamp': level_obj.timestamp,
                    'confidence': level_obj.confidence
                }
            
            file_path = os.path.join(self.cache_dir, f"{entity_id}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(entity_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.debug(f"Error saving entity to disk: {e}")
    
    def _load_from_disk(self):
        """Load entities from disk cache."""
        try:
            if not os.path.exists(self.cache_dir):
                return
            
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.cache_dir, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            entity_data = json.load(f)
                        
                        entity_id = filename[:-5]
                        
                        for level_str, level_obj in entity_data.items():
                            level = int(level_str)
                            data = EntityLevelData(
                                level=level,
                                data=level_obj.get('data'),
                                metadata=level_obj.get('metadata', {}),
                                timestamp=level_obj.get('timestamp', time.time()),
                                confidence=level_obj.get('confidence', 0.5)
                            )
                            self._entities[entity_id][level] = data
                            self._level_stats[level] = self._level_stats.get(level, 0) + 1
                        
                        if level_obj.get('data') and isinstance(level_obj['data'], dict):
                            term = level_obj['data'].get('term')
                            if term:
                                self._entity_index[term.lower()].append(entity_id)
                                
                    except Exception as e:
                        logger.debug(f"Error loading entity {filename}: {e}")
                        
            logger.info(f"Loaded entities from disk cache")
            
        except Exception as e:
            logger.debug(f"Error loading from disk: {e}")
    
    def clear_cache(self):
        """Clear all cached entities."""
        self._entities.clear()
        self._entity_index.clear()
        self._level_stats = {i: 0 for i in range(5)}
        self._similarity_cache.clear()
        logger.info("Entity fractal cache cleared")
