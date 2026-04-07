"""
Модуль ядра графа знаний для ЕВА
Содержит основной класс KnowledgeGraph, конструктор, методы работы с БД и базовые операции
Refactored: split into graph_core.py, graph_nodes.py, graph_edges.py, graph_operations.py
"""
import logging

logger = logging.getLogger("eva_ai.knowledge_graph")

# Import base class and mixins from new modules
from eva_ai.knowledge.graph_core import (
    KnowledgeGraphCore,
    _get_knowledge_graph_types,
    _get_hybrid_token_cache,
    _get_unified_text_processor,
    _get_ml_unit,
    _get_entity_extractor,
    _ensure_imports,
    KnowledgeNode,
    KnowledgeEdge,
    NodeType,
    RelationType,
    safe_json_loads,
    HybridTokenCache,
    get_shared_cache,
    UnifiedTextProcessor,
    MLUnit,
    EntityExtractor,
)

from eva_ai.knowledge.graph_nodes import KnowledgeGraphNodeMixin
from eva_ai.knowledge.graph_edges import KnowledgeGraphEdgeMixin
from eva_ai.knowledge.graph_operations import KnowledgeGraphOperationsMixin


# Build the full KnowledgeGraph class by combining base + all mixins
class KnowledgeGraph(
    KnowledgeGraphNodeMixin,
    KnowledgeGraphEdgeMixin,
    KnowledgeGraphOperationsMixin,
    KnowledgeGraphCore,
):
    """
    Граф знаний для ЕВА - хранит и управляет знаниями системы.
    Combines functionality from:
    - graph_core: initialization, lifecycle, DB management, background services
    - graph_nodes: node creation, editing, deletion, properties
    - graph_edges: edge creation, relationships, weights
    - graph_operations: traversal, search, queries, analytics, export/import
    """
    pass


__all__ = [
    'KnowledgeGraph',
    'KnowledgeGraphCore',
    'KnowledgeGraphNodeMixin',
    'KnowledgeGraphEdgeMixin',
    'KnowledgeGraphOperationsMixin',
    'KnowledgeNode',
    'KnowledgeEdge',
    'NodeType',
    'RelationType',
    'safe_json_loads',
    '_get_knowledge_graph_types',
    '_get_hybrid_token_cache',
    '_get_unified_text_processor',
    '_get_ml_unit',
    '_get_entity_extractor',
    '_ensure_imports',
]
