"""Main query class, initialization, and lifecycle for EVA knowledge graph queries."""
import os
import logging
import time
import math
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger("eva_ai.knowledge_graph")


def _get_knowledge_graph_types():
    """Lazy import для типов графа знаний."""
    from eva_ai.knowledge.knowledge_graph_types import (
        KnowledgeNode,
        KnowledgeEdge,
        NodeType,
        RelationType,
        safe_json_loads
    )
    return KnowledgeNode, KnowledgeEdge, NodeType, RelationType, safe_json_loads


def _get_unified_text_processor():
    """Lazy import для текстового процессора."""
    try:
        from eva_ai.mlearning.unified_text_processor import UnifiedTextProcessor
        return UnifiedTextProcessor
    except ImportError:
        logger.warning("UnifiedTextProcessor недоступен, токенизация будет ограничена")
        return None


class KnowledgeGraphQueryMixin:
    """Mixin class with query methods for KnowledgeGraph. Core initialization and lifecycle."""

    def get_node(self, node_id: str) -> Optional[Any]:
        """Возвращает узел по ID."""
        return self.nodes.get(node_id)

    def get_edge(self, edge_id: str) -> Optional[Any]:
        """Возвращает связь по ID."""
        return self.edges.get(edge_id)

    def get_all_nodes(self) -> List[Any]:
        """Возвращает все узлы графа."""
        return list(self.nodes.values())

    def get_all_edges(self) -> List[Any]:
        """Возвращает все связи графа."""
        return list(self.edges.values())

    def get_nodes_by_domain(self, domain: str) -> List[Any]:
        """Возвращает узлы указанного домена."""
        return [node for node in self.nodes.values() if node.domain == domain]

    def is_initialized(self) -> bool:
        """Проверяет, инициализирован ли граф."""
        return self.initialized

    def is_running(self) -> bool:
        """Проверяет, запущен ли граф."""
        return self.running

    def start(self):
        """Запускает граф знаний."""
        if not self.running:
            self._start_background_services()

    def stop(self):
        """Останавливает граф знаний."""
        if self.running:
            self.stop_event.set()
            if hasattr(self, 'monitoring_thread'):
                self.monitoring_thread.join(timeout=5)
            if hasattr(self, 'optimization_thread'):
                self.optimization_thread.join(timeout=5)
            self.running = False
            logger.info("Фоновые службы KnowledgeGraph остановлены")

    def close(self):
        """Закрывает граф знаний и освобождает ресурсы."""
        self.stop()

        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

        logger.info("Graph closed")
