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
from .knowledge_analytics import KnowledgeAnalytics
from .qwen_api_enhancer import QwenAPIEnhancer
from .concept_extractor import ConceptExtractor, Concept, create_concept_extractor
from .concept_miner import ConceptMiner, ConceptStatus, PhantomCandidate, create_concept_miner

__all__ = [
    'KnowledgeGraphAdapter', 'KnowledgeGraph', 'GraphCurator', 'create_graph_curator',
    'AmbiguityResolver', 'EntityExtractor',
    'WikipediaKnowledgeBase', 'get_wikipedia_kb', 'clear_wikipedia_kb', 'get_wikipedia_loader',
    'KnowledgeAnalytics', 'QwenAPIEnhancer',
    'ConceptExtractor', 'Concept', 'create_concept_extractor',
    'ConceptMiner', 'ConceptStatus', 'PhantomCandidate', 'create_concept_miner'
]
