"""
NEREL Integration for EVA AI
Named Entities with REations in Russian dataset
NEREL is a Russian dataset for named entity recognition and relation extraction.

Dataset format:
{
  "id": "...",
  "text": "текст предложения",
  "tokens": [...],
  "entities": [{"type": "PER", "start": 0, "end": 10, "text": "имя"}],
  "relations": [{"type": "PER:CITY", "head": 0, "tail": 1}]
}

Usage for training:
- ConceptMiner: Learn entity patterns and semantic gaps
- ContradictionDetector: Detect logical conflicts in entity relations
"""
import json
import logging
import os
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict

logger = logging.getLogger("eva_ai.knowledge.nerel_integration")

class NERELDataset:
    """
    NEREL dataset loader and processor.
    
    Provides:
    - Entity extraction patterns
    - Relation patterns for training
    - Validation data for ConceptMiner
    - Contradiction examples for ContradictionDetector
    """
    
    ENTITY_TYPES = {'PER', 'ORG', 'LOC', 'DATE', 'MONEY', 'PERCENT', 'TIME', 'EVENT'}
    RELATION_TYPES = {
        'PER_ORG', 'PER_LOC', 'PER_DATE', 'ORG_LOC', 'ORG_DATE',
        'PER_PERS', 'ORG_PERS', 'LOC_LOC', 'PER_MONEY', 'PER_EVENT'
    }
    
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or self._get_default_data_dir()
        self.documents: List[Dict] = []
        self.entity_index: Dict[str, List[int]] = defaultdict(list)
        self.relation_index: Dict[str, List[int]] = defaultdict(list)

        # Auto-load if files exist
        self._try_auto_load()

    def _try_auto_load(self):
        """Try to auto-load data from knowledge_data directory"""
        nerel_path = os.path.join(self.data_dir, 'nerel_russian.json')
        if os.path.exists(nerel_path):
            self.load_from_file(nerel_path)

    def _get_default_data_dir(self) -> str:
        return os.path.join(
            os.path.dirname(__file__),
            'knowledge_data'
        )
    
    def load_from_file(self, filepath: str) -> int:
        """
        Load NEREL dataset from JSON file.
        
        Expected format:
        {
          "documents": [
            {"id": "...", "text": "...", "entities": [...], "relations": [...]},
            ...
          ]
        }
        
        Returns: Number of documents loaded
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict) and 'documents' in data:
                self.documents = data['documents']
            elif isinstance(data, list):
                self.documents = data
            else:
                logger.error(f"Unexpected NEREL format")
                return 0
            
            self._build_indices()
            logger.info(f"Loaded {len(self.documents)} NEREL documents")
            return len(self.documents)
            
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            return 0
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return 0
    
    def _build_indices(self):
        """Build indices for fast lookups"""
        self.entity_index.clear()
        self.relation_index.clear()
        
        for i, doc in enumerate(self.documents):
            text = doc.get('text', '')
            
            # Index entities
            for entity in doc.get('entities', []):
                entity_text = entity.get('text', '').lower()
                if entity_text:
                    self.entity_index[entity_text].append(i)
            
            # Index relations
            for rel in doc.get('relations', []):
                rel_type = rel.get('type', '')
                if rel_type:
                    self.relation_index[rel_type].append(i)
    
    def get_entity_mentions(self, entity_name: str) -> List[Dict]:
        """Get all mentions of an entity across documents"""
        entity_lower = entity_name.lower()
        results = []
        
        for idx in self.entity_index.get(entity_lower, []):
            doc = self.documents[idx]
            for entity in doc.get('entities', []):
                if entity.get('text', '').lower() == entity_lower:
                    results.append({
                        'document_id': doc.get('id', ''),
                        'text': doc.get('text', ''),
                        'entity': entity,
                        'start': entity.get('start'),
                        'end': entity.get('end')
                    })
        return results
    
    def get_relation_patterns(self, relation_type: Optional[str] = None) -> List[Dict]:
        """
        Get relation patterns for training.
        
        Args:
            relation_type: Filter by specific relation type (e.g., 'PER_ORG')
        
        Returns: List of {head_entity, tail_entity, context} patterns
        """
        patterns = []
        
        for doc in self.documents:
            text = doc.get('text', '')
            entities = {e.get('start'): e for e in doc.get('entities', [])}
            relations = doc.get('relations', [])
            
            for rel in relations:
                if relation_type and rel.get('type') != relation_type:
                    continue
                
                head_idx = rel.get('head')
                tail_idx = rel.get('tail')
                
                head_entity = None
                tail_entity = None
                
                for ent in doc.get('entities', []):
                    if ent.get('id') == head_idx or (isinstance(head_idx, int) and ent.get('start', -1) == head_idx):
                        head_entity = ent
                    if ent.get('id') == tail_idx or (isinstance(tail_idx, int) and ent.get('start', -1) == tail_idx):
                        tail_entity = ent
                
                if head_entity and tail_entity:
                    patterns.append({
                        'relation_type': rel.get('type'),
                        'head_entity': head_entity.get('text'),
                        'head_type': head_entity.get('type'),
                        'tail_entity': tail_entity.get('text'),
                        'tail_type': tail_entity.get('type'),
                        'context': text[:200],  # First 200 chars as context
                        'document_id': doc.get('id')
                    })
        
        return patterns
    
    def get_entity_type_distribution(self) -> Dict[str, int]:
        """Get count of each entity type"""
        type_counts = defaultdict(int)
        for doc in self.documents:
            for entity in doc.get('entities', []):
                entity_type = entity.get('type', 'UNKNOWN')
                type_counts[entity_type] += 1
        return dict(type_counts)
    
    def get_relation_type_distribution(self) -> Dict[str, int]:
        """Get count of each relation type"""
        type_counts = defaultdict(int)
        for doc in self.documents:
            for rel in doc.get('relations', []):
                rel_type = rel.get('type', 'UNKNOWN')
                type_counts[rel_type] += 1
        return dict(type_counts)
    
    def extract_contradiction_candidates(self) -> List[Dict]:
        """
        Extract potential contradictions from the dataset.
        
        Contradictions can be:
        - Same entity with different types in different contexts
        - Same relation with different tail entities
        - Conflicting facts (e.g., different locations for same entity)
        
        Returns: List of {entity, facts, conflict_type}
        """
        candidates = []
        
        # Group by entity
        entity_facts = defaultdict(list)
        for doc in self.documents:
            for entity in doc.get('entities', []):
                entity_name = entity.get('text', '')
                if entity_name:
                    entity_facts[entity_name].append({
                        'type': entity.get('type'),
                        'document_id': doc.get('id'),
                        'text': doc.get('text', '')
                    })
        
        # Find contradictions
        for entity, facts in entity_facts.items():
            type_groups = defaultdict(list)
            for fact in facts:
                type_groups[fact['type']].append(fact)
            
            # Same entity with different types
            if len(type_groups) > 1:
                candidates.append({
                    'entity': entity,
                    'conflict_type': 'type_conflict',
                    'facts': [{'type': t, 'count': len(f)} for t, f in type_groups.items()]
                })
        
        return candidates
    
    def get_training_pairs_for_miner(self) -> List[Tuple[str, str]]:
        """
        Get (text, expected_concept) pairs for training ConceptMiner.
        
        Returns: List of (input_text, concept_name) training pairs
        """
        pairs = []
        for doc in self.documents:
            text = doc.get('text', '')
            for entity in doc.get('entities', []):
                entity_text = entity.get('text', '')
                if entity_text and len(entity_text) > 2:
                    pairs.append((text, entity_text))
        return pairs


def create_nerel_dataset(data_dir: Optional[str] = None) -> NERELDataset:
    """Factory function"""
    return NERELDataset(data_dir)


def load_sample_data(dataset: NERELDataset) -> bool:
    """
    Load sample NEREL data for testing.
    """
    sample_docs = [
        {
            "id": "doc1",
            "text": "Владимир Путин родился в Ленинграде.",
            "entities": [
                {"id": 0, "type": "PER", "start": 0, "end": 16, "text": "Владимир Путин"},
                {"id": 1, "type": "LOC", "start": 25, "end": 34, "text": "Ленинград"}
            ],
            "relations": [
                {"type": "PER_LOC", "head": 0, "tail": 1}
            ]
        },
        {
            "id": "doc2",
            "text": "Москва - столица России.",
            "entities": [
                {"id": 0, "type": "LOC", "start": 0, "end": 6, "text": "Москва"},
                {"id": 1, "type": "LOC", "start": 18, "end": 24, "text": "Россия"}
            ],
            "relations": [
                {"type": "LOC_LOC", "head": 0, "tail": 1}
            ]
        },
        {
            "id": "doc3",
            "text": "Компания Google была основана в США.",
            "entities": [
                {"id": 0, "type": "ORG", "start": 0, "end": 6, "text": "Google"},
                {"id": 1, "type": "LOC", "start": 26, "end": 29, "text": "США"}
            ],
            "relations": [
                {"type": "ORG_LOC", "head": 0, "tail": 1}
            ]
        }
    ]
    
    dataset.documents = sample_docs
    dataset._build_indices()
    logger.info(f"Loaded {len(sample_docs)} sample NEREL documents")
    return True


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("=== NEREL Integration Test ===")
    dataset = NERELDataset()
    load_sample_data(dataset)
    
    print("\n--- Entity Distribution ---")
    dist = dataset.get_entity_type_distribution()
    for t, c in dist.items():
        print(f"  {t}: {c}")
    
    print("\n--- Relation Patterns (PER_LOC) ---")
    patterns = dataset.get_relation_patterns('PER_LOC')
    for p in patterns:
        print(f"  {p['head_entity']} -> {p['tail_entity']} ({p['context'][:50]}...)")
    
    print("\n--- Contradiction Candidates ---")
    candidates = dataset.extract_contradiction_candidates()
    print(f"  Found {len(candidates)} candidates")
    
    print("\n--- Training Pairs ---")
    pairs = dataset.get_training_pairs_for_miner()
    print(f"  {len(pairs)} training pairs available")