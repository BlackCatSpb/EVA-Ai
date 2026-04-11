"""
KnowledgeGraph - обёртка для FractalGraph v2
Обеспечивает совместимость с кодом, ожидающим KnowledgeGraph API
"""
import logging
from typing import Dict, List, Any, Optional

from .kg_adapter import KnowledgeGraphAdapter

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """
    Обёртка вокруг FractalGraph v2 для совместимости с API KnowledgeGraph.
    Все вызовы перенаправляются на FractalGraph v2 через KnowledgeGraphAdapter.
    """
    
    def __init__(self, fractal_graph=None):
        """
        Args:
            fractal_graph: Экземпляр FractalMemoryGraph (FGv2)
        """
        if fractal_graph is None:
            from eva_ai.memory.fractal_graph_v2 import FractalMemoryGraph
            fractal_graph = FractalMemoryGraph()
        
        self._adapter = KnowledgeGraphAdapter(fractal_graph)
        self._fg = fractal_graph
        
        logger.info(f"KnowledgeGraph инициализирован (FGv2: {len(fractal_graph.storage.nodes)} узлов)")
    
    def get_all(self) -> Dict:
        """Получить все данные графа."""
        return {
            'nodes': self._adapter.get_nodes_list() if hasattr(self._adapter, 'get_nodes_list') else [],
            'edges': self._adapter.get_edges_list() if hasattr(self._adapter, 'get_edges_list') else []
        }
    
    def get_nodes(self) -> Dict:
        """Получить узлы."""
        return self._adapter.nodes
    
    def get_edges(self) -> Dict:
        """Получить связи."""
        return self._adapter.edges
    
    def add_node(self, *args, **kwargs):
        """Добавить узел."""
        return self._adapter.add_node(*args, **kwargs)
    
    def add_edge(self, *args, **kwargs):
        """Добавить связь."""
        return self._adapter.add_edge(*args, **kwargs)
    
    def search(self, *args, **kwargs):
        """Поиск по графу."""
        return self._adapter.search_nodes(*args, **kwargs)
    
    def get_stats(self) -> Dict:
        """Получить статистику."""
        return self._adapter.get_stats()
    
    def __getattr__(self, name: str):
        """Перенаправление вызовов."""
        if hasattr(self._adapter, name):
            return getattr(self._adapter, name)
        if hasattr(self._fg, name):
            return getattr(self._fg, name)
        logger.debug(f"KnowledgeGraph: метод {name} не найден")
        return lambda *args, **kwargs: None
