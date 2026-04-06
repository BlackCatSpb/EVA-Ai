"""Graph traversal, path finding, BFS/DFS for EVA knowledge graph queries."""
import logging
from typing import Dict, List, Optional, Any
from collections import defaultdict

logger = logging.getLogger("eva.knowledge_graph")


class KnowledgeGraphQueryTraversal:
    """Graph traversal mixin for KnowledgeGraph."""

    def get_edges(self, node_id: str, direction: str = "both",
                  relation_type: Optional[str] = None) -> List[Any]:
        """Возвращает связи для узла."""
        edges = []

        for edge in self.edges.values():
            if direction == "source" and edge.source_id == node_id:
                if relation_type is None or edge.relation_type == relation_type:
                    edges.append(edge)
            elif direction == "target" and edge.target_id == node_id:
                if relation_type is None or edge.relation_type == relation_type:
                    edges.append(edge)
            elif direction == "both" and (edge.source_id == node_id or edge.target_id == node_id):
                if relation_type is None or edge.relation_type == relation_type:
                    edges.append(edge)

        return edges

    def get_related_nodes(self, node_id: str, relation_type: Optional[str] = None,
                          direction: str = "both", max_distance: int = 1) -> List[Any]:
        """Возвращает связанные узлы на указанном расстоянии."""
        if max_distance < 1:
            return []

        visited = set([node_id])
        related_nodes = []

        direct_edges = self.get_edges(node_id, direction=direction, relation_type=relation_type)
        for edge in direct_edges:
            neighbor_id = edge.target_id if edge.source_id == node_id else edge.source_id
            if neighbor_id not in visited:
                neighbor = self.get_node(neighbor_id)
                if neighbor:
                    related_nodes.append(neighbor)
                    visited.add(neighbor_id)

        if max_distance > 1:
            for node in related_nodes.copy():
                deeper_nodes = self.get_related_nodes(
                    node.id,
                    relation_type=relation_type,
                    direction=direction,
                    max_distance=max_distance - 1
                )
                for deeper_node in deeper_nodes:
                    if deeper_node.id not in visited:
                        related_nodes.append(deeper_node)
                        visited.add(deeper_node.id)

        return related_nodes

    def get_subgraph(self, center_node_id: str, depth: int = 2) -> Dict[str, Any]:
        """Возвращает подграф с центром в указанном узле."""
        try:
            if center_node_id not in self.nodes:
                logger.warning(f"Центральный узел {center_node_id} не существует")
                return {"nodes": [], "edges": []}

            visited = set([center_node_id])
            nodes = [self.nodes[center_node_id]]
            edges = []

            current_level = [center_node_id]
            for _ in range(depth):
                next_level = []

                for node_id in current_level:
                    for edge in self.get_edges(node_id, direction="source"):
                        if edge.target_id not in visited:
                            visited.add(edge.target_id)
                            next_level.append(edge.target_id)
                            edges.append(edge)

                            if edge.target_id in self.nodes:
                                nodes.append(self.nodes[edge.target_id])

                    for edge in self.get_edges(node_id, direction="target"):
                        if edge.source_id not in visited:
                            visited.add(edge.source_id)
                            next_level.append(edge.source_id)
                            edges.append(edge)

                            if edge.source_id in self.nodes:
                                nodes.append(self.nodes[edge.source_id])

                current_level = next_level

            return {
                "nodes": [node.to_dict() for node in nodes],
                "edges": [edge.to_dict() for edge in edges]
            }

        except Exception as e:
            logger.error(f"Ошибка построения подграфа: {e}", exc_info=True)
            return {"nodes": [], "edges": []}

    def get_node_details(self, node_id: str, max_distance: int = 2) -> Dict[str, Any]:
        """Возвращает детальную информацию об узле и его соседях."""
        try:
            node = self.get_node(node_id)
            if not node:
                return {"error": "Node not found"}

            related = self.get_related_nodes(node_id, max_distance=max_distance)

            return {
                "node": node.to_dict() if hasattr(node, 'to_dict') else {"id": node.id, "name": node.name},
                "related_nodes_count": len(related),
                "related_nodes": [
                    n.to_dict() if hasattr(n, 'to_dict') else {"id": n.id, "name": n.name}
                    for n in related[:10]
                ],
                "edges": [
                    edge.to_dict() if hasattr(edge, 'to_dict') else {"id": edge.id}
                    for edge in self.get_edges(node_id)
                ]
            }
        except Exception as e:
            logger.error(f"Error getting node details: {e}", exc_info=True)
            return {"error": str(e)}

    def get_all_concepts(self) -> List[Dict[str, Any]]:
        """Возвращает все концепты в формате для MemoryGraphML."""
        concepts = []
        for node in self.nodes.values():
            concepts.append({
                'id': node.id,
                'type': node.node_type,
                'description': node.description or node.meta.get('description', ''),
                'domain': node.domain,
                'properties': node.meta
            })

        for edge in self.edges.values():
            concepts.append({
                'id': edge.id,
                'type': 'relation',
                'description': f"{edge.source_id} -> {edge.target_id}: {edge.relation_type}",
                'domain': 'general',
                'properties': edge.meta
            })

        return concepts
