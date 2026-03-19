"""
Модуль управления знаниями CogniFlex
"""

from .knowledge_graph import KnowledgeGraph, KnowledgeNode, KnowledgeEdge
from .knowledge_manager import KnowledgeManager
from .knowledge_analyzer import KnowledgeAnalyzer
from .knowledge_integrator import KnowledgeIntegrator
from .context_entity import (
    AmbiguousEntity,
    ClarificationRequest,
    RefinementQuery,
    AmbiguityType,
    EntityExtractor
)
from .ambiguity_resolver import AmbiguityResolver

__all__ = [
    'KnowledgeGraph',
    'KnowledgeNode',
    'KnowledgeEdge',
    'KnowledgeManager',
    'KnowledgeAnalyzer',
    'KnowledgeIntegrator',
    'AmbiguousEntity',
    'ClarificationRequest',
    'RefinementQuery',
    'AmbiguityType',
    'EntityExtractor',
    'AmbiguityResolver'
]