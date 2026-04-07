"""Search functionality, filtering, and ranking for EVA knowledge graph."""
import logging
import time
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque

logger = logging.getLogger("eva_ai.knowledge.core_search")


class KnowledgeSearch:
    """Search and traversal mixin for KnowledgeGraph."""

    def search_nodes(self, query: str, domain: Optional[str] = None, limit: int = 10, min_strength: float = 0.3) -> List[Any]:
        """Поиск узлов по запросу."""
        results = []
        query_lower = query.lower()

        for node in self.nodes.values():
            if domain and node.domain != domain:
                continue
            if node.strength < min_strength:
                continue

            content_str = str(node.content).lower()
            if query_lower in content_str:
                results.append(node)
                if len(results) >= limit:
                    break

        return results

    def get_related_nodes(self, node_id: str, max_distance: int = 2) -> List[Dict]:
        """Возвращает связанные узлы с информацией о связи."""
        if node_id not in self.nodes:
            return []

        visited = {node_id}
        queue = deque([(node_id, 0)])
        related = []

        while queue:
            current_id, distance = queue.popleft()

            if distance >= max_distance:
                continue

            for edge_id in self.node_edges[current_id]:
                edge = self.edges[edge_id]
                neighbor_id = edge.target if edge.source == current_id else edge.source

                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, distance + 1))

                    related.append({
                        "id": neighbor_id,
                        "distance": distance + 1,
                        "relation": edge.relation,
                        "strength": edge.strength
                    })

        return related

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

    def find_concept_nodes(self, concept: str, max_results: int = 10) -> List[Any]:
        """Находит узлы, связанные с концептом."""
        results = []
        concept_lower = concept.lower()

        for node in self.nodes.values():
            content_str = str(node.content).lower()
            if concept_lower in content_str:
                results.append(node)
                if len(results) >= max_results:
                    break

        return results

    def get_influence_depth(self, node_id: str, max_depth: int = 5) -> int:
        """Определяет глубину влияния узла в графе."""
        if node_id not in self.nodes:
            return 0

        visited = {node_id}
        queue = deque([(node_id, 0)])
        max_reached_depth = 0

        while queue:
            current_id, depth = queue.popleft()
            max_reached_depth = max(max_reached_depth, depth)

            if depth >= max_depth:
                continue

            for edge_id in self.node_edges[current_id]:
                edge = self.edges[edge_id]
                neighbor_id = edge.target if edge.source == current_id else edge.source

                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, depth + 1))

        return max_reached_depth

    def get_concept_usage_frequency(self, concept: str) -> int:
        """Определяет частоту использования концепта в графе."""
        count = 0
        concept_lower = concept.lower()

        for node in self.nodes.values():
            content_str = str(node.content).lower()
            if concept_lower in content_str:
                count += 1

        return count

    def get_concept_relations(self, concept: str) -> Dict[str, List[Dict]]:
        """Возвращает все отношения для концепта."""
        relations = defaultdict(list)
        concept_nodes = self.find_concept_nodes(concept)

        for node in concept_nodes:
            for edge_id in self.node_edges[node.id]:
                edge = self.edges[edge_id]
                neighbor_id = edge.target if edge.source == node.id else edge.source
                neighbor = self.nodes.get(neighbor_id)

                if neighbor:
                    relations[edge.relation].append({
                        "node_id": neighbor.id,
                        "content": neighbor.get_content_summary(),
                        "strength": edge.strength,
                        "direction": "out" if edge.source == node.id else "in"
                    })

        return dict(relations)

    def get_concept_contexts(self, concept: str) -> List[str]:
        """Возвращает контексты, в которых используется концепт."""
        contexts = set()
        concept_nodes = self.find_concept_nodes(concept)

        for node in concept_nodes:
            if node.context:
                contexts.add(node.context)

        return list(contexts)

    def get_concept_domains(self, concept: str) -> List[str]:
        """Возвращает домены, в которых используется концепт."""
        domains = set()
        concept_nodes = self.find_concept_nodes(concept)

        for node in concept_nodes:
            domains.add(node.domain)

        return list(domains)
