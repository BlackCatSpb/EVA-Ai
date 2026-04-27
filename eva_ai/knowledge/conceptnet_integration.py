"""
ConceptNet Integration for EVA AI
Provides semantic relationships for knowledge graph enrichment
"""
import logging
from typing import List, Dict, Optional, Set
from collections import defaultdict

logger = logging.getLogger("eva_ai.knowledge.conceptnet_integration")

class ConceptNetConnector:
    """
    Connects EVA's FractalGraph to ConceptNet for semantic relationships.
    
    ConceptNet provides:
    - Synonyms, antonyms
    - IsA relationships (hypernyms/hyponyms)
    - PartOf relationships (meronyms/holonyms)
    - Various semantic relations (CreatedBy, UsedFor, CapableOf, etc.)
    """
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        self._connected = False
        self._available = False
        
    def connect(self) -> bool:
        """Connect to ConceptNet database"""
        try:
            from conceptnet_lite import connect, CONCEPTNET_DB_NAME
            from conceptnet_lite.db import _generate_db_path

            import os

            if self.db_path is None:
                candidates = [
                    os.path.join(os.path.dirname(__file__), '..', '..', 'knowledge_data', CONCEPTNET_DB_NAME),
                    os.path.join(os.path.dirname(__file__), '..', '..', CONCEPTNET_DB_NAME),
                    os.path.join(os.getcwd(), CONCEPTNET_DB_NAME),
                ]
                for candidate in candidates:
                    if os.path.exists(candidate):
                        self.db_path = candidate
                        break

            if self.db_path is None or not os.path.exists(self.db_path):
                logger.warning(f"ConceptNet DB not found, checked: {candidates}")
                self._available = False
                return False

            connect(self.db_path)
            self._connected = True
            self._available = True
            logger.info(f"Connected to ConceptNet database at {self.db_path}")
            return True

        except ImportError:
            logger.warning("conceptnet-lite not installed - run: pip install conceptnet-lite")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to ConceptNet: {e}")
            self._connected = False
            self._available = False
            return False
    
    def download_if_needed(self) -> bool:
        """Download ConceptNet database if not present"""
        try:
            from conceptnet_lite import download_db
            
            logger.info("Downloading ConceptNet database (this may take time)...")
            download_db()
            self._available = True
            return self.connect()
            
        except ImportError:
            logger.warning("conceptnet-lite not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to download ConceptNet: {e}")
            return False
    
    def get_concept_info(self, concept: str, lang: str = 'en') -> Dict:
        """
        Get all edges for a concept.

        Returns dict with:
        - edges: list of (relation, other_concept, weight)
        - relations: count by relation type
        """
        if not self._connected:
            return {}

        try:
            from conceptnet_lite import Label, Concept, Language, edges_for

            lang_code = lang[:2]

            lang_obj = Language.get(name=lang_code)
            label = Label.get_or_create(text=concept.lower(), language=lang_obj)[0]
            concept_obj = Concept.get(label=label)

            cn_edges = edges_for([concept_obj])
            cn_edges_list = list(cn_edges)

            result = {
                'concept': concept,
                'edges': [],
                'relations': defaultdict(int)
            }

            for edge in cn_edges_list:
                try:
                    rel = edge.relation.name if hasattr(edge.relation, 'name') else str(edge.relation)
                    start_uri = edge.start.uri if hasattr(edge.start, 'uri') else str(edge.start)
                    end_uri = edge.end.uri if hasattr(edge.end, 'uri') else str(edge.end)
                    weight = edge.etc.get('weight', 1.0) if hasattr(edge, 'etc') else 1.0

                    concept_uri = f'/{lang_code}/{concept.lower()}'
                    other = end_uri.split('/')[-1] if concept_uri in start_uri else start_uri.split('/')[-1]
                    direction = 'out' if concept_uri in start_uri else 'in'

                    result['edges'].append({
                        'relation': rel,
                        'other': other,
                        'direction': direction,
                        'weight': weight
                    })
                    result['relations'][rel] += 1

                except Exception as e:
                    logger.debug(f"Error processing edge: {e}")
                    continue

            return result

        except Exception as e:
            logger.error(f"Error getting concept info: {e}")
            return {}
    
    def find_related_concepts(self, concept: str, relation_types: Optional[List[str]] = None,
                             min_weight: float = 0.5, limit: int = 20) -> List[Dict]:
        """
        Find concepts related to given concept.

        Args:
            concept: The concept to search for
            relation_types: Filter by specific relation types (e.g., ['IsA', 'PartOf'])
            min_weight: Minimum edge weight
            limit: Maximum results

        Returns: List of {concept, relation, weight}
        """
        if not self._connected:
            return []

        info = self.get_concept_info(concept)
        related = []

        for edge in info.get('edges', []):
            if edge['weight'] < min_weight:
                continue
            if relation_types and edge['relation'] not in relation_types:
                continue
            related.append({
                'concept': edge['other'],
                'relation': edge['relation'],
                'direction': edge['direction'],
                'weight': edge['weight']
            })

        related.sort(key=lambda x: x['weight'], reverse=True)
        return related[:limit]
    
    def get_hypernyms(self, concept: str, lang: str = 'ru', limit: int = 10) -> List[str]:
        """Get broader concepts (IsA / hypernyms)"""
        return [r['concept'] for r in self.find_related_concepts(
            concept, relation_types=['IsA', 'SimilarTo', 'RelatedTo'], limit=limit
        )]
    
    def get_hyponyms(self, concept: str, lang: str = 'ru', limit: int = 10) -> List[str]:
        """Get narrower concepts"""
        # ConceptNet doesn't directly give hyponyms, so we invert IsA
        return []  # Would need custom logic
    
    def get_synonyms(self, concept: str, lang: str = 'ru') -> List[str]:
        """Get synonyms and similar concepts"""
        related = self.find_related_concepts(
            concept, relation_types=['SimilarTo', 'Synonym'], limit=15
        )
        return [r['concept'] for r in related if r['relation'] in ['SimilarTo', 'Synonym']]


def create_conceptnet_connector(db_path: Optional[str] = None) -> ConceptNetConnector:
    """Factory function for ConceptNet connector"""
    connector = ConceptNetConnector(db_path)
    return connector


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("=== ConceptNet Integration Test ===")
    connector = ConceptNetConnector()
    
    # Try to connect
    if connector.connect():
        print("Connected to ConceptNet!")
        
        # Test with a simple concept
        info = connector.get_concept_info('человек', lang='ru')
        print(f"\nConcept 'человек' (ru):")
        print(f"  Total edges: {len(info.get('edges', []))}")
        print(f"  Relations: {dict(info.get('relations', {}))}")
        
    else:
        print("Not connected - run connector.download_if_needed() to fetch database")
        print("Or install manually: pip install conceptnet-lite")