"""
Unified Knowledge Integration for EVA AI
Combines ConceptNet, Wikidata/RuBQ, and NEREL into a single interface
"""
import logging
from typing import List, Dict, Optional, Set
from collections import defaultdict

from .conceptnet_integration import ConceptNetConnector, create_conceptnet_connector
from .wikidata_integration import WikidataExtractor, load_sample_data as load_wikidata_sample
from .nerel_integration import NERELDataset, load_sample_data as load_nerel_sample

logger = logging.getLogger("eva_ai.knowledge.knowledge_integrator")

class KnowledgeIntegrator:
    """
    Unified interface to all external knowledge sources.
    
    Combines:
    - ConceptNet: Semantic relationships (synonyms, hypernyms, etc.)
    - Wikidata/RuBQ: Factual triplets about the world
    - NEREL: Named entity recognition and relation patterns
    
    Usage:
        integrator = KnowledgeIntegrator()
        integrator.initialize()
        
        # Get comprehensive context for a concept
        context = integrator.get_concept_context('Москва')
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir
        
        # Initialize connectors
        self.conceptnet = create_conceptnet_connector()
        self.wikidata = WikidataExtractor(data_dir)
        self.nerel = NERELDataset(data_dir)
        
        self._conceptnet_available = False
        self._wikidata_loaded = False
        self._nerel_loaded = False
        
    def initialize(self) -> Dict[str, bool]:
        """
        Initialize all knowledge sources.
        
        Returns: Dict of {source: success}
        """
        results = {}
        
        # ConceptNet
        if self.conceptnet.connect():
            self._conceptnet_available = True
            results['conceptnet'] = True
            logger.info("ConceptNet connected")
        else:
            logger.warning("ConceptNet not available - run download_if_needed()")
            results['conceptnet'] = False
        
        # Wikidata - load sample data if no file
        if not self._wikidata_loaded:
            load_wikidata_sample(self.wikidata)
            self._wikidata_loaded = True
            results['wikidata'] = True
            logger.info("Wikidata loaded with sample data")
        
        # NEREL - load sample data
        if not self._nerel_loaded:
            load_nerel_sample(self.nerel)
            self._nerel_loaded = True
            results['nerel'] = True
            logger.info("NEREL loaded with sample data")
        
        return results
    
    def get_concept_context(self, concept: str, lang: str = 'ru') -> Dict:
        """
        Get comprehensive context about a concept from all sources.
        
        Returns:
        {
            'concept': str,
            'wikidata_triplets': [...],
            'conceptnet_relations': [...],
            'nerel_entities': [...],
            'synonyms': [...],
            'hypernyms': [...],
            'related_concepts': [...]
        }
        """
        context = {
            'concept': concept,
            'wikidata_triplets': [],
            'conceptnet_relations': [],
            'nerel_entities': [],
            'synonyms': [],
            'hypernyms': [],
            'related_concepts': []
        }
        
        # Wikidata triplets
        if self._wikidata_loaded:
            triplets = self.wikidata.get_entity_context(concept)
            context['wikidata_triplets'] = triplets.get('outgoing', []) + triplets.get('incoming', [])
        
        # ConceptNet relations
        if self._conceptnet_available:
            cn_info = self.conceptnet.get_concept_info(concept, lang)
            if cn_info:
                context['conceptnet_relations'] = cn_info.get('edges', [])
                context['synonyms'] = self.conceptnet.get_synonyms(concept, lang)
                context['hypernyms'] = self.conceptnet.get_hypernyms(concept, lang)
                related = self.conceptnet.find_related_concepts(concept, min_weight=0.5, limit=20)
                context['related_concepts'] = [r['concept'] for r in related]
        
        # NEREL entities
        if self._nerel_loaded:
            mentions = self.nerel.get_entity_mentions(concept)
            context['nerel_entities'] = mentions
        
        return context
    
    def add_concept_to_graph(self, concept: str, brain, lang: str = 'ru') -> bool:
        """
        Add a concept to EVA's FractalGraph with all available knowledge.
        
        Args:
            concept: Concept name to add
            brain: EVA Brain instance with fractal_graph_v2
            lang: Language code
        
        Returns: True if added successfully
        """
        if not hasattr(brain, 'fractal_graph_v2'):
            logger.error("Brain doesn't have fractal_graph_v2")
            return False
        
        fg = brain.fractal_graph_v2
        
        # Get context from all sources
        context = self.get_concept_context(concept, lang)
        
        # Create concept node
        try:
            # Add main concept node (auto-generated ID)
            existing_ids = list(fg.storage.nodes.keys())
            concept_id = f"concept_{concept.lower().replace(' ', '_')}"

            # Only add if not exists
            if not any(concept_id in eid or getattr(fg.storage.nodes[eid], 'content', '') == concept for eid in existing_ids):
                fg.add_node(
                    content=concept,
                    node_type='concept',
                    level=2,
                    metadata={
                        'source': 'knowledge_integrator',
                        'concept_id': concept_id,
                        'wikidata_triplets': len(context['wikidata_triplets']),
                        'conceptnet_relations': len(context['conceptnet_relations']),
                        'synonyms': context['synonyms'][:5],
                        'hypernyms': context['hypernyms'][:5],
                        'related': context['related_concepts'][:10]
                    }
                )

            # Add synonym nodes
            for syn in context['synonyms'][:5]:
                syn_id = f"concept_{syn.lower().replace(' ', '_')}"
                existing = [eid for eid in fg.storage.nodes.keys() if syn in getattr(fg.storage.nodes[eid], 'content', '')]
                if not existing:
                    fg.add_node(
                        content=syn,
                        node_type='concept',
                        level=1,
                        metadata={'synonym_of': concept}
                    )

            # Add related concepts
            for rel_concept in context['related_concepts'][:10]:
                rel_id = f"concept_{rel_concept.lower().replace(' ', '_')}"
                existing = [eid for eid in fg.storage.nodes.keys() if rel_concept in getattr(fg.storage.nodes[eid], 'content', '')]
                if not existing:
                    fg.add_node(
                        content=rel_concept,
                        node_type='concept',
                        level=1,
                        metadata={'related_to': concept}
                    )

            logger.info(f"Added concept '{concept}' to graph with {len(context['related_concepts'])} related concepts")
            return True
            
        except Exception as e:
            logger.error(f"Error adding concept to graph: {e}")
            return False
    
    def enrich_existing_nodes(self, brain) -> int:
        """
        Enrich all existing concept nodes in the graph with external knowledge.
        
        Returns: Number of nodes enriched
        """
        if not hasattr(brain, 'fractal_graph_v2'):
            return 0
        
        fg = brain.fractal_graph_v2
        enriched = 0
        
        for node_id, node in list(fg.storage.nodes.items()):
            if getattr(node, 'node_type', '') == 'concept':
                concept = getattr(node, 'content', '')
                if concept and len(concept) > 2:
                    context = self.get_concept_context(concept)
                    
                    # Update node metadata
                    if hasattr(node, 'metadata') and isinstance(node.metadata, dict):
                        node.metadata['synonyms'] = context.get('synonyms', [])[:5]
                        node.metadata['hypernyms'] = context.get('hypernyms', [])[:5]
                        node.metadata['related'] = context.get('related_concepts', [])[:10]
                        enriched += 1
        
        logger.info(f"Enriched {enriched} nodes in graph")
        return enriched
    
    def get_statistics(self) -> Dict:
        """Get statistics about all knowledge sources"""
        return {
            'conceptnet_available': self._conceptnet_available,
            'conceptnet_edges': len(self.conceptnet.get_concept_info('test').get('edges', [])) if self._conceptnet_available else 0,
            'wikidata_triplets': len(self.wikidata.triplets),
            'wikidata_entities': len(self.wikidata.entity_index),
            'nerel_documents': len(self.nerel.documents),
            'nerel_entities': len(self.nerel.entity_index)
        }


def create_knowledge_integrator(data_dir: Optional[str] = None) -> KnowledgeIntegrator:
    """Factory function"""
    return KnowledgeIntegrator(data_dir)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("=== Knowledge Integrator Test ===")
    
    integrator = KnowledgeIntegrator()
    results = integrator.initialize()
    
    print("\n--- Initialization Results ---")
    for source, success in results.items():
        print(f"  {source}: {'OK' if success else 'FAILED'}")
    
    print("\n--- Statistics ---")
    stats = integrator.get_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print("\n--- Moscow Context ---")
    ctx = integrator.get_concept_context('Москва')
    print(f"  Wikidata triplets: {len(ctx['wikidata_triplets'])}")
    print(f"  ConceptNet relations: {len(ctx['conceptnet_relations'])}")
    print(f"  Synonyms: {ctx['synonyms'][:3]}")
    print(f"  Hypernyms: {ctx['hypernyms'][:3]}")
    print(f"  Related: {ctx['related_concepts'][:5]}")