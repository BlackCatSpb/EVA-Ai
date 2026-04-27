"""
Wikidata/RuBQ 2.0 Integration for EVA AI
Extracts Russian triplets from Wikidata for factual knowledge base
"""
import json
import logging
import os
from typing import List, Dict, Optional, Generator
from collections import defaultdict

logger = logging.getLogger("eva_ai.knowledge.wikidata_integration")

class WikidataExtractor:
    """
    Extracts knowledge triplets from Wikidata/ruBQ dataset.

    RuBQ 2.0 is a Russian Question Answering dataset based on Wikidata.
    It provides structured knowledge triples in format:
    (subject, relation, object)

    Usage:
        extractor = WikidataExtractor()
        extractor.load_from_file('path/to/rubq.json')

        # Get triplets related to a concept
        related = extractor.get_related_triplets('Москва')
    """

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or self._get_default_data_dir()
        self.triplets: List[Dict] = []
        self.entity_index: Dict[str, List[int]] = defaultdict(list)
        self.relation_types: set = set()

        # Auto-load if files exist
        self._try_auto_load()

    def _try_auto_load(self):
        """Try to auto-load data from knowledge_data directory"""
        wikidata_path = os.path.join(self.data_dir, 'wikidata_russian.json')
        if os.path.exists(wikidata_path):
            self.load_from_file(wikidata_path)

    def _get_default_data_dir(self) -> str:
        return os.path.join(
            os.path.dirname(__file__),
            'knowledge_data'
        )
    
    def load_from_file(self, filepath: str) -> int:
        """
        Load triplets from JSON file.
        
        Expected format:
        {
            "triplets": [
                {"subject": "...", "relation": "...", "object": "...", "subject_id": "...", "object_id": "..."},
                ...
            ]
        }
        
        Returns: Number of triplets loaded
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict) and 'triplets' in data:
                self.triplets = data['triplets']
            elif isinstance(data, list):
                self.triplets = data
            else:
                logger.error(f"Unexpected data format in {filepath}")
                return 0
            
            self._build_index()
            logger.info(f"Loaded {len(self.triplets)} triplets from {filepath}")
            return len(self.triplets)
            
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            return 0
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return 0
    
    def load_from_wikidata_dump(self, filepath: str) -> int:
        """
        Load from Wikidata JSON dump (simplified format).
        
        Wikidata dump format:
        {"head": {"vars": ["subject", "relation", "object"]},
         "results": {"bindings": [...]}}
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            vars = data.get('head', {}).get('vars', [])
            if 'subject' not in vars or 'relation' not in vars or 'object' not in vars:
                logger.error(f"Wikidata dump missing required vars: {vars}")
                return 0
            
            subj_idx = vars.index('subject')
            rel_idx = vars.index('relation')
            obj_idx = vars.index('object')
            
            self.triplets = []
            for binding in data.get('results', {}).get('bindings', []):
                try:
                    triplet = {
                        'subject': binding.get('subject', {}).get('value', ''),
                        'relation': binding.get('relation', {}).get('value', ''),
                        'object': binding.get('object', {}).get('value', ''),
                    }
                    if triplet['subject'] and triplet['relation'] and triplet['object']:
                        self.triplets.append(triplet)
                except (KeyError, TypeError):
                    continue
            
            self._build_index()
            logger.info(f"Loaded {len(self.triplets)} triplets from Wikidata dump")
            return len(self.triplets)
            
        except Exception as e:
            logger.error(f"Error loading Wikidata dump: {e}")
            return 0
    
    def _build_index(self):
        """Build index for fast lookups"""
        self.entity_index.clear()
        self.relation_types.clear()
        
        for i, triplet in enumerate(self.triplets):
            # Index subject
            subj = triplet.get('subject', '').lower()
            if subj:
                self.entity_index[subj].append(i)
                # Also index without URI prefix
                subj_label = self._extract_label(subj)
                if subj_label:
                    self.entity_index[subj_label].append(i)
            
            # Index object
            obj = triplet.get('object', '').lower()
            if obj:
                self.entity_index[obj].append(i)
                obj_label = self._extract_label(obj)
                if obj_label:
                    self.entity_index[obj_label].append(i)
            
            # Track relation types
            rel = triplet.get('relation', '')
            if rel:
                self.relation_types.add(self._extract_label(rel))
        
        logger.debug(f"Built index with {len(self.entity_index)} entities, {len(self.relation_types)} relation types")
    
    def _extract_label(self, uri_or_text: str) -> str:
        """Extract label from URI or use text as-is"""
        if not uri_or_text:
            return ''
        # Common URI patterns like http://www.wikidata.org/entity/Q123
        if '/' in uri_or_text and 'http' in uri_or_text:
            return uri_or_text.split('/')[-1]
        return uri_or_text.strip()
    
    def get_related_triplets(self, entity: str, limit: int = 50) -> List[Dict]:
        """
        Get all triplets related to an entity.
        
        Args:
            entity: Entity name or URI
            limit: Maximum number of triplets
        
        Returns: List of triplet dicts
        """
        entity_lower = entity.lower()
        indices = set()
        
        # Direct match
        if entity_lower in self.entity_index:
            indices.update(self.entity_index[entity_lower])
        
        # Partial match
        for key in self.entity_index:
            if entity_lower in key or key in entity_lower:
                indices.update(self.entity_index[key])
        
        results = []
        for idx in list(indices)[:limit]:
            if idx < len(self.triplets):
                results.append(self.triplets[idx])
        
        return results
    
    def get_incoming_triplets(self, entity: str) -> List[Dict]:
        """Get triplets where entity is the object (incoming relations)"""
        entity_lower = entity.lower()
        results = []
        
        for triplet in self.triplets:
            obj = triplet.get('object', '').lower()
            if entity_lower == obj or self._extract_label(obj) == self._extract_label(entity_lower):
                results.append(triplet)
        
        return results
    
    def get_outgoing_triplets(self, entity: str) -> List[Dict]:
        """Get triplets where entity is the subject (outgoing relations)"""
        entity_lower = entity.lower()
        results = []
        
        for triplet in self.triplets:
            subj = triplet.get('subject', '').lower()
            if entity_lower == subj or self._extract_label(subj) == self._extract_label(entity_lower):
                results.append(triplet)
        
        return results
    
    def get_relation_types(self) -> List[str]:
        """Get all unique relation types in the dataset"""
        return list(self.relation_types)
    
    def search_by_relation(self, relation: str, limit: int = 100) -> List[Dict]:
        """Get all triplets with a specific relation type"""
        rel_lower = relation.lower()
        results = []
        
        for triplet in self.triplets:
            rel = triplet.get('relation', '')
            if rel_lower in rel.lower() or self._extract_label(rel) == relation:
                results.append(triplet)
                if len(results) >= limit:
                    break
        
        return results
    
    def get_entity_context(self, entity: str, max_triplets: int = 20) -> Dict:
        """
        Get comprehensive context about an entity.
        
        Returns:
            {
                'incoming': [...],
                'outgoing': [...],
                'relations_used': set of relation types
            }
        """
        incoming = self.get_incoming_triplets(entity)[:max_triplets]
        outgoing = self.get_outgoing_triplets(entity)[:max_triplets]
        
        all_relations = set()
        for t in incoming + outgoing:
            rel = t.get('relation', '')
            if rel:
                all_relations.add(self._extract_label(rel))
        
        return {
            'entity': entity,
            'incoming': incoming,
            'outgoing': outgoing,
            'relations_used': list(all_relations),
            'total_incoming': len(incoming),
            'total_outgoing': len(outgoing)
        }


def create_wikidata_extractor(data_dir: Optional[str] = None) -> WikidataExtractor:
    """Factory function"""
    return WikidataExtractor(data_dir)


def load_sample_data(extractor: WikidataExtractor) -> bool:
    """
    Load sample RuBQ data for testing.
    Creates synthetic triplets representing common Russian knowledge.
    """
    sample_triplets = [
        # Geography
        {'subject': 'Москва', 'relation': 'столица', 'object': 'Россия'},
        {'subject': 'Москва', 'relation': 'находится_в', 'object': 'Центральная Россия'},
        {'subject': 'Россия', 'relation': 'имеет_население', 'object': 'около 146 миллионов'},
        {'subject': 'Санкт-Петербург', 'relation': 'находится_в', 'object': 'Россия'},
        {'subject': 'Волга', 'relation': 'длина', 'object': '3530 км'},
        
        # Science
        {'subject': 'Исаак Ньютон', 'relation': 'родился', 'object': '1643'},
        {'subject': 'Исаак Ньютон', 'relation': 'открыл', 'object': 'законы движения'},
        {'subject': 'Альберт Эйнштейн', 'relation': 'создал', 'object': 'теория относительности'},
        {'subject': 'теория относительности', 'relation': 'описывает', 'object': 'гравитация'},
        
        # Technology
        {'subject': 'Python', 'relation': 'язык_программирования', 'object': 'высокоуровневый'},
        {'subject': 'Python', 'relation': 'поддерживает', 'object': 'ООП'},
        {'subject': 'нейронная сеть', 'relation': 'тип', 'object': 'машинное обучение'},
        
        # General knowledge
        {'subject': 'вода', 'relation': 'химическая_формула', 'object': 'H2O'},
        {'subject': 'кислород', 'relation': 'элемент', 'object': ' periodic table'},
    ]
    
    extractor.triplets = sample_triplets
    extractor._build_index()
    logger.info(f"Loaded {len(sample_triplets)} sample triplets")
    return True


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("=== Wikidata/RuBQ Integration Test ===")
    extractor = WikidataExtractor()
    
    # Load sample data
    load_sample_data(extractor)
    
    # Test queries
    print("\n--- Testing 'Москва' context ---")
    ctx = extractor.get_entity_context('Москва')
    print(f"Incoming: {ctx['total_incoming']}, Outgoing: {ctx['total_outgoing']}")
    print(f"Relations used: {ctx['relations_used']}")
    
    print("\n--- Testing relation search 'находится' ---")
    results = extractor.search_by_relation('находится')
    for r in results[:5]:
        print(f"  {r['subject']} -> {r['relation']} -> {r['object']}")
    
    print("\n--- Testing outgoing from 'Python' ---")
    outgoing = extractor.get_outgoing_triplets('Python')
    for t in outgoing:
        print(f"  Python -[{t['relation']}]-> {t['object']}")