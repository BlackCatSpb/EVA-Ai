"""
Knowledge module - обёртка для Knowledge Graph API
Все вызовы перенаправляются на FractalGraph v2
"""
from .kg_adapter import KnowledgeGraphAdapter
from .graph_curator import GraphCurator
from .ambiguity_resolver import AmbiguityResolver
from .context_entity import EntityExtractor

__all__ = ['KnowledgeGraphAdapter', 'GraphCurator', 'AmbiguityResolver', 'EntityExtractor']
