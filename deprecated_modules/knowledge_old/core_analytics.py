"""Analytics, statistics, metrics, and reporting for EVA knowledge graph."""
import logging
import time
from typing import Dict, List, Optional, Any
from collections import defaultdict

logger = logging.getLogger("eva_ai.knowledge.core_analytics")


class KnowledgeAnalytics:
    """Analytics and reporting mixin for KnowledgeGraph."""

    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает общую статистику по графу знаний (для GUI)."""
        try:
            node_types: Dict[str, int] = {}
            for node in self.nodes.values():
                node_types[node.node_type] = node_types.get(node.node_type, 0) + 1

            return {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "node_types": node_types,
                "domains": list(self.domains.keys()),
                "last_updated": self.stats.get("last_update", time.time()),
            }
        except Exception as e:
            logger.error(f"Ошибка получения статистики графа знаний: {e}")
            return {"error": str(e)}

    def get_domain_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику по доменам (для GUI вкладки знаний)."""
        try:
            domains: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
                "nodes": 0,
                "edges": 0,
                "node_types": {}
            })

            for node in self.nodes.values():
                d = domains[node.domain]
                d["nodes"] += 1
                nt = d["node_types"]
                nt[node.node_type] = nt.get(node.node_type, 0) + 1

            for edge in self.edges.values():
                src = self.nodes.get(edge.source)
                domain = src.domain if src else "unknown"
                domains[domain]["edges"] += 1

            return {k: dict(v) for k, v in domains.items()}
        except Exception as e:
            logger.error(f"Ошибка получения статистики доменов: {e}")
            return {}

    def get_node_strength_history(self, node_id: str) -> List[Dict[str, Any]]:
        """Возвращает историю изменения силы узла."""
        if node_id not in self.nodes:
            return []

        node = self.nodes[node_id]
        return [{
            "timestamp": node.timestamp,
            "strength": node.strength,
            "reason": "initial_creation"
        }]

    def get_concept_influence_score(self, concept: str) -> float:
        """Вычисляет показатель влияния концепта в графе."""
        concept_nodes = self.find_concept_nodes(concept)
        if not concept_nodes:
            return 0.0

        node_count_score = min(1.0, len(concept_nodes) / 50)

        max_depth = 0
        for node in concept_nodes:
            depth = self.get_influence_depth(node.id)
            max_depth = max(max_depth, depth)

        depth_score = min(1.0, max_depth / 10)

        total_strength = 0
        total_relations = 0

        for node in concept_nodes:
            for edge_id in self.node_edges[node.id]:
                edge = self.edges[edge_id]
                total_strength += edge.strength
                total_relations += 1

        strength_score = min(1.0, total_strength / max(1, total_relations)) if total_relations > 0 else 0.0

        influence_score = (node_count_score * 0.4) + (depth_score * 0.3) + (strength_score * 0.3)
        return influence_score

    def get_concept_reputation(self, concept: str) -> float:
        """Возвращает репутацию концепта на основе источников."""
        concept_nodes = self.find_concept_nodes(concept)
        if not concept_nodes:
            return 0.5

        total_reputation = 0
        count = 0

        for node in concept_nodes:
            source_reputation = node.meta.get('source_reputation', 0.5)
            total_reputation += source_reputation
            count += 1

            for edge_id in self.node_edges[node.id]:
                edge = self.edges[edge_id]
                neighbor_id = edge.target if edge.source == node.id else edge.source
                neighbor = self.nodes.get(neighbor_id)

                if neighbor:
                    neighbor_reputation = neighbor.meta.get('source_reputation', 0.5)
                    total_reputation += neighbor_reputation
                    count += 1

        return total_reputation / max(1, count)

    def get_concept_time_relevance(self, concept: str) -> float:
        """Оценивает актуальность концепта во времени."""
        concept_nodes = self.find_concept_nodes(concept)
        if not concept_nodes:
            return 0.0

        newest_timestamp = max(node.timestamp for node in concept_nodes)
        time_diff = time.time() - newest_timestamp
        days_diff = time_diff / 86400

        if days_diff <= 7:
            return 1.0
        else:
            decay_factor = 0.95 ** ((days_diff - 7) / 7)
            return max(0.1, decay_factor)

    def get_concept_analysis(self, concept: str) -> Dict[str, Any]:
        """Проводит комплексный анализ концепта."""
        return {
            "concept": concept,
            "influence_score": self.get_concept_influence_score(concept),
            "reputation": self.get_concept_reputation(concept),
            "time_relevance": self.get_concept_time_relevance(concept),
            "domains": self.get_concept_domains(concept),
            "contexts": self.get_concept_contexts(concept),
            "relation_types": list(self.get_concept_relations(concept).keys()),
            "node_count": len(self.find_concept_nodes(concept)),
            "depth": max((self.get_influence_depth(node.id) for node in self.find_concept_nodes(concept)), default=0),
            "timestamp": time.time()
        }

    def _update_stats(self):
        """Обновляет статистику графа знаний."""
        self.stats["total_nodes"] = len(self.nodes)
        self.stats["total_edges"] = len(self.edges)
        self.stats["domains"] = set(self.domains.keys())
        self.stats["last_update"] = time.time()
