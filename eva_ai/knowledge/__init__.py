"""
Knowledge module - обёртка для Knowledge Graph API
Все вызовы перенаправляются на FractalGraph v2
"""
from .kg_adapter import KnowledgeGraphAdapter
from .knowledge_graph import KnowledgeGraph
from .graph_curator import GraphCurator, create_graph_curator
from .ambiguity_resolver import AmbiguityResolver
from .context_entity import EntityExtractor
from .wikipedia_kb import WikipediaKnowledgeBase, get_wikipedia_kb, clear_wikipedia_kb, get_wikipedia_loader
from .qwen_api_enhancer import QwenAPIEnhancer
from .concept_extractor import ConceptExtractor, Concept, create_concept_extractor
from .concept_miner import ConceptMiner, ConceptStatus, PhantomCandidate, create_concept_miner

# Knowledge integration modules
from .conceptnet_integration import ConceptNetConnector, create_conceptnet_connector
from .wikidata_integration import WikidataExtractor, create_wikidata_extractor, load_sample_data as load_wikidata_sample
from .nerel_integration import NERELDataset, create_nerel_dataset, load_sample_data as load_nerel_sample
from .knowledge_integrator import KnowledgeIntegrator, create_knowledge_integrator

__all__ = [
    'KnowledgeGraphAdapter', 'KnowledgeGraph', 'GraphCurator', 'create_graph_curator',
    'AmbiguityResolver', 'EntityExtractor',
    'WikipediaKnowledgeBase', 'get_wikipedia_kb', 'clear_wikipedia_kb', 'get_wikipedia_loader',
    'QwenAPIEnhancer',
    'ConceptExtractor', 'Concept', 'create_concept_extractor',
    'ConceptMiner', 'ConceptStatus', 'PhantomCandidate', 'create_concept_miner',
    # Integration
    'ConceptNetConnector', 'create_conceptnet_connector',
    'WikidataExtractor', 'create_wikidata_extractor', 'load_wikidata_sample',
    'NERELDataset', 'create_nerel_dataset', 'load_nerel_sample',
    'KnowledgeIntegrator', 'create_knowledge_integrator'
]
