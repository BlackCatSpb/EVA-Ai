"""CRUD operations for EVA knowledge graph - add, remove, update knowledge."""
import logging
import time
from typing import Dict, List, Optional, Any
from collections import defaultdict

logger = logging.getLogger("eva_ai.knowledge.core_operations")


class KnowledgeOperations:
    """CRUD operations mixin for KnowledgeGraph."""

    def add_node(self, node) -> bool:
        """Добавляет узел в граф знаний."""
        if node.id in self.nodes:
            logger.warning(f"Узел с ID {node.id} уже существует")
            return False

        try:
            self.nodes[node.id] = node
            self.domains[node.domain].append(node.id)

            if node.context:
                self.contexts[node.context].append(node.id)

            if not self.storage_manager.insert_node(node):
                return False

            if self.hybrid_index is not None:
                try:
                    self.hybrid_index.put_node(node)
                except Exception as e:
                    logger.warning(f"Ошибка индексации узла {node.id} в гибридном индексе: {e}")

            self._update_stats()
            logger.debug(f"Узел {node.id} добавлен в граф знаний")
            return True

        except Exception as e:
            logger.error(f"Ошибка добавления узла {node.id}: {e}")
            return False

    def add_edge(self, edge) -> bool:
        """Добавляет связь между узлами."""
        if edge.id in self.edges:
            logger.warning(f"Связь с ID {edge.id} уже существует")
            return False

        if edge.source not in self.nodes or edge.target not in self.nodes:
            logger.warning(f"Источник или цель связи {edge.id} не существуют в графе")
            return False

        try:
            self.edges[edge.id] = edge
            self.node_edges[edge.source].append(edge.id)
            self.node_edges[edge.target].append(edge.id)

            if not self.storage_manager.insert_edge(edge):
                return False

            if self.hybrid_index is not None:
                try:
                    self.hybrid_index.put_edge(edge)
                except Exception as e:
                    logger.warning(f"Ошибка индексации связи {edge.id} в гибридном индексе: {e}")

            self._update_stats()
            logger.debug(f"Связь {edge.id} добавлена в граф знаний")
            return True

        except Exception as e:
            logger.error(f"Ошибка добавления связи {edge.id}: {e}")
            return False

    def remove_node(self, node_id: str) -> bool:
        """Удаляет узел и все связанные с ним связи."""
        if node_id not in self.nodes:
            logger.warning(f"Узел {node_id} не найден")
            return False

        try:
            node = self.nodes.pop(node_id)

            if node_id in self.domains[node.domain]:
                self.domains[node.domain].remove(node_id)

            if node.context and node_id in self.contexts[node.context]:
                self.contexts[node.context].remove(node_id)

            edges_to_remove = self.node_edges.pop(node_id, [])
            for edge_id in edges_to_remove:
                if edge_id in self.edges:
                    del self.edges[edge_id]

            self.storage_manager.delete_node(node_id)

            if self.hybrid_index is not None:
                try:
                    self.hybrid_index.remove_node(node_id)
                except Exception as e:
                    logger.warning(f"Ошибка удаления узла {node_id} из гибридного индекса: {e}")

            self._update_stats()
            logger.info(f"Узел {node_id} и связанные с ним связи удалены")
            return True

        except Exception as e:
            logger.error(f"Ошибка удаления узла {node_id}: {e}")
            return False

    def remove_edge(self, edge_id: str) -> bool:
        """Удаляет связь между узлами."""
        if edge_id not in self.edges:
            logger.warning(f"Связь {edge_id} не найдена")
            return False

        try:
            edge = self.edges.pop(edge_id)

            if edge_id in self.node_edges[edge.source]:
                self.node_edges[edge.source].remove(edge_id)
            if edge_id in self.node_edges[edge.target]:
                self.node_edges[edge.target].remove(edge_id)

            self.storage_manager.delete_edge(edge_id)

            if self.hybrid_index is not None:
                try:
                    self.hybrid_index.remove_edge(edge_id)
                except Exception as e:
                    logger.warning(f"Ошибка удаления связи {edge_id} из гибридного индекса: {e}")

            self._update_stats()
            logger.info(f"Связь {edge_id} удалена")
            return True

        except Exception as e:
            logger.error(f"Ошибка удаления связи {edge_id}: {e}")
            return False

    def update_node(self, node_id: str, **kwargs) -> bool:
        """Обновляет свойства узла."""
        if node_id not in self.nodes:
            logger.warning(f"Узел {node_id} не найден")
            return False

        try:
            node = self.nodes[node_id]

            if "content" in kwargs:
                node.content = kwargs["content"]
            if "node_type" in kwargs:
                node.node_type = kwargs["node_type"]
            if "domain" in kwargs:
                if node.domain in self.domains and node_id in self.domains[node.domain]:
                    self.domains[node.domain].remove(node_id)
                node.domain = kwargs["domain"]
                self.domains[node.domain].append(node_id)
            if "strength" in kwargs:
                node.strength = max(0.0, min(1.0, kwargs["strength"]))
            if "timestamp" in kwargs:
                node.timestamp = kwargs["timestamp"]
            if "meta" in kwargs:
                node.meta = kwargs["meta"]
                if node.meta is None:
                    node.meta = {}
                node.context = node.meta.get('context', '')

                if "context" in node.meta:
                    if node_id in self.contexts.get(node.context, []):
                        self.contexts[node.context].remove(node_id)
                    self.contexts[node.meta["context"]].append(node_id)

            self.storage_manager.update_node(node)

            logger.info(f"Узел {node_id} обновлен")
            return True

        except Exception as e:
            logger.error(f"Ошибка обновления узла {node_id}: {e}")
            return False

    def get_node(self, node_id: str) -> Optional[Any]:
        """Возвращает узел по ID."""
        node = self.nodes.get(node_id)
        if node is not None:
            return node
        if self.hybrid_index is not None:
            try:
                data = self.hybrid_index.get_node(node_id)
                if data:
                    from .knowledge_core import KnowledgeNode
                    node = KnowledgeNode.from_dict(data)
                    self.nodes[node.id] = node
                    self.domains[node.domain].append(node.id)
                    if node.context:
                        self.contexts[node.context].append(node.id)
                    return node
            except Exception as e:
                logger.warning(f"Ошибка получения узла {node_id} из гибридного индекса: {e}")
        return None

    def get_edge(self, edge_id: str) -> Optional[Any]:
        """Возвращает связь по ID."""
        edge = self.edges.get(edge_id)
        if edge is not None:
            return edge
        if self.hybrid_index is not None:
            try:
                data = self.hybrid_index.get_edge(edge_id)
                if data:
                    from .knowledge_core import KnowledgeEdge
                    edge = KnowledgeEdge.from_dict(data)
                    self.edges[edge.id] = edge
                    self.node_edges[edge.source].append(edge.id)
                    self.node_edges[edge.target].append(edge.id)
                    return edge
            except Exception as e:
                logger.warning(f"Ошибка получения связи {edge_id} из гибридного индекса: {e}")
        return None

    def get_nodes_by_domain(self, domain: str) -> List[Any]:
        """Возвращает узлы указанного домена."""
        node_ids = self.domains.get(domain, [])
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]

    def get_nodes_by_context(self, context: str) -> List[Any]:
        """Возвращает узлы с указанным контекстом."""
        node_ids = self.contexts.get(context, [])
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]

    def get_edges(self, node_id: str, direction: str = "both") -> List[Any]:
        """Возвращает связи узла."""
        if node_id not in self.node_edges:
            return []

        edge_ids = self.node_edges[node_id]
        edges = [self.edges[eid] for eid in edge_ids if eid in self.edges]

        if direction == "source":
            return [e for e in edges if e.source == node_id]
        elif direction == "target":
            return [e for e in edges if e.target == node_id]
        else:
            return edges

    def get_all_nodes(self, limit: Optional[int] = None) -> List[Any]:
        """Возвращает все узлы графа."""
        nodes = list(self.nodes.values())
        return nodes[:limit] if limit else nodes

    def get_all_edges(self, limit: Optional[int] = None) -> List[Any]:
        """Возвращает все связи графа."""
        edges = list(self.edges.values())
        return edges[:limit] if limit else edges
