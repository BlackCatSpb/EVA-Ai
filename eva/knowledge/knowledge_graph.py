"""
Модуль графа знаний для ЕВА - управление структурой знаний
Обновленная версия с поддержкой гибридного кэша, асинхронной токенизации и версионирования
Разделен на логические модули: knowledge_graph_core.py, knowledge_graph_query.py, knowledge_graph_traversal.py
"""
import logging

logger = logging.getLogger("eva.knowledge_graph")

# Импорт типов из модулей
from eva.knowledge.knowledge_graph_types import (
    NodeType,
    RelationType,
    KnowledgeNode,
    KnowledgeEdge
)

# Импорт основного класса с миксинами
from eva.knowledge.knowledge_graph_core import KnowledgeGraph
from eva.knowledge.knowledge_graph_query import KnowledgeGraphQueryMixin
from eva.knowledge.knowledge_graph_traversal import KnowledgeGraphTraversalMixin

# Применяем миксины к основному классу
class KnowledgeGraph(KnowledgeGraph, KnowledgeGraphQueryMixin, KnowledgeGraphTraversalMixin):
    """
    Граф знаний для ЕВА - хранит и управляет знаниями системы.
    Объединяет функциональность из:
    - knowledge_graph_core: основные операции и БД
    - knowledge_graph_query: методы запросов и поиска
    - knowledge_graph_traversal: обход графа и анализ
    """
    pass

__all__ = [
    'KnowledgeGraph',
    'KnowledgeNode',
    'KnowledgeEdge',
    'NodeType',
    'RelationType'
]
