"""
Knowledge module - обёртка для Knowledge Graph API
Все вызовы перенаправляются на FractalGraph v2
"""
from .kg_adapter import KnowledgeGraphAdapter

__all__ = ['KnowledgeGraphAdapter']
