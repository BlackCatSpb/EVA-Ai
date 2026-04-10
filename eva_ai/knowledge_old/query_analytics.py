"""Query analytics, profiling, and optimization for EVA knowledge graph queries."""
import os
import logging
import time
from typing import Dict, List, Optional, Any
from collections import defaultdict

logger = logging.getLogger("eva_ai.knowledge_graph")


class KnowledgeGraphQueryAnalytics:
    """Analytics and profiling mixin for KnowledgeGraph queries."""

    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику по графу знаний."""
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "domains": self._get_domain_statistics(),
            "node_types": self._get_node_type_statistics(),
            "last_update": time.time(),
            "stats": self.stats.copy()
        }

    def _get_domain_statistics(self) -> Dict[str, int]:
        """Возвращает статистику по доменам."""
        stats = {}
        for node in self.nodes.values():
            if node.domain in stats:
                stats[node.domain] += 1
            else:
                stats[node.domain] = 1
        return stats

    def _get_node_type_statistics(self) -> Dict[str, int]:
        """Возвращает статистику по типам узлов."""
        stats = {}
        for node in self.nodes.values():
            if node.node_type in stats:
                stats[node.node_type] += 1
            else:
                stats[node.node_type] = 1
        return stats

    def _get_relation_type_statistics(self) -> Dict[str, int]:
        """Возвращает статистику по типам связей."""
        stats = {}
        for edge in self.edges.values():
            if edge.relation_type in stats:
                stats[edge.relation_type] += 1
            else:
                stats[edge.relation_type] = 1
        return stats

    def _get_cache_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику кэша."""
        cache_stats = {
            "hybrid_cache_enabled": self.hybrid_cache is not None,
            "cache_dir": self.cache_dir,
            "cache_size_mb": 0.0,
            "cache_entries": 0,
            "cache_hit_rate": 0.0
        }

        try:
            if os.path.exists(self.cache_dir):
                total_size = 0
                for root, dirs, files in os.walk(self.cache_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            total_size += os.path.getsize(file_path)
                        except OSError:
                            pass
                cache_stats["cache_size_mb"] = round(total_size / (1024 * 1024), 2)
        except Exception as e:
            logger.debug(f"Error calculating cache size: {e}")

        return cache_stats

    def get_system_health(self) -> Dict[str, Any]:
        """Возвращает сводку о состоянии системы графа знаний."""
        try:
            return {
                "status": "healthy" if self.initialized else "not_initialized",
                "nodes_count": len(self.nodes),
                "edges_count": len(self.edges),
                "integrity_score": self._check_graph_integrity_score(),
                "resource_usage": self._check_resource_usage(),
                "background_services": {
                    "running": self.running,
                    "monitoring": hasattr(self, 'monitoring_thread') and self.monitoring_thread.is_alive(),
                    "optimization": hasattr(self, 'optimization_thread') and self.optimization_thread.is_alive()
                }
            }
        except Exception as e:
            logger.error(f"Ошибка получения состояния системы: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}

    def _check_graph_integrity_score(self) -> float:
        """Проверяет оценку целостности графа."""
        try:
            if not self.nodes:
                return 1.0

            valid_edges = 0
            total_edges = len(self.edges)

            for edge in self.edges.values():
                if edge.source_id in self.nodes and edge.target_id in self.nodes:
                    valid_edges += 1

            if total_edges == 0:
                return 1.0

            return valid_edges / total_edges

        except Exception as e:
            logger.debug(f"Error calculating integrity score: {e}")
            return 0.0

    def get_graph_health(self) -> Dict[str, Any]:
        """Возвращает детальную информацию о здоровье графа."""
        try:
            health = {
                "overall_score": self._check_graph_integrity_score(),
                "integrity_report": self._get_graph_integrity_report(),
                "resource_usage": self._check_resource_usage()
            }
            return health
        except Exception as e:
            logger.error(f"Ошибка получения здоровья графа: {e}", exc_info=True)
            return {"error": str(e)}

    def _check_resource_usage(self) -> float:
        """Проверяет использование ресурсов."""
        try:
            cpu_usage = 0.0
            memory_usage = 0.0

            try:
                import psutil
                process = psutil.Process()
                cpu_usage = process.cpu_percent(interval=0.1)
                memory_info = process.memory_info()
                memory_usage = memory_info.rss / (1024 * 1024)
            except ImportError:
                logger.debug("psutil not available, using estimate")
                memory_usage = 0.0

            return {
                "cpu_percent": cpu_usage,
                "memory_mb": memory_usage
            }
        except Exception as e:
            logger.debug(f"Error checking resource usage: {e}")
            return {"cpu_percent": 0.0, "memory_mb": 0.0}

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Возвращает метрики производительности."""
        try:
            total_time = self.stats.get("total_processing_time", 0.0)
            total_queries = self.stats.get("total_queries", 0)

            avg_time = total_time / total_queries if total_queries > 0 else 0.0

            return {
                "total_queries": total_queries,
                "successful_queries": self.stats.get("successful_queries", 0),
                "failed_queries": self.stats.get("failed_queries", 0),
                "total_processing_time": total_time,
                "average_query_time": avg_time,
                "node_creations": self.stats.get("node_creations", 0),
                "node_updates": self.stats.get("node_updates", 0),
                "edge_creations": self.stats.get("edge_creations", 0),
                "edge_updates": self.stats.get("edge_updates", 0)
            }
        except Exception as e:
            logger.error(f"Ошибка получения метрик производительности: {e}", exc_info=True)
            return {"error": str(e)}

    def get_detailed_health_report(self) -> Dict[str, Any]:
        """Возвращает детальный отчет о состоянии графа знаний."""
        try:
            return {
                "system_health": self.get_system_health(),
                "graph_health": self.get_graph_health(),
                "performance_metrics": self.get_performance_metrics(),
                "cache_statistics": self._get_cache_statistics()
            }
        except Exception as e:
            logger.error(f"Ошибка получения детального отчета: {e}", exc_info=True)
            return {"error": str(e)}

    def _get_graph_integrity_report(self) -> Dict[str, Any]:
        """Возвращает отчет о целостности графа."""
        try:
            missing_node_refs = 0
            for edge in self.edges.values():
                if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
                    missing_node_refs += 1

            return {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "missing_node_references": missing_node_refs,
                "orphaned_edges": missing_node_refs
            }
        except Exception as e:
            logger.debug(f"Error generating integrity report: {e}")
            return {"error": str(e)}

    def get_domain_statistics(self, domain: str) -> Dict[str, Any]:
        """Возвращает статистику по домену."""
        try:
            domain_nodes = self.get_nodes_by_domain(domain)

            node_types = defaultdict(int)
            total_strength = 0.0

            for node in domain_nodes:
                node_types[node.node_type] += 1
                total_strength += node.strength

            avg_strength = total_strength / len(domain_nodes) if domain_nodes else 0.0

            return {
                "domain": domain,
                "total_nodes": len(domain_nodes),
                "node_types": dict(node_types),
                "average_strength": avg_strength,
                "recent_updates": self._get_recent_updates_for_domain(domain)
            }
        except Exception as e:
            logger.error(f"Error getting domain statistics: {e}", exc_info=True)
            return {"error": str(e)}

    def _get_recent_updates_for_domain(self, domain: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Возвращает последние обновления для домена."""
        try:
            domain_nodes = self.get_nodes_by_domain(domain)
            sorted_nodes = sorted(
                domain_nodes,
                key=lambda n: getattr(n, 'last_updated', 0),
                reverse=True
            )

            updates = []
            for node in sorted_nodes[:limit]:
                updates.append({
                    "node_id": node.id,
                    "name": node.name,
                    "last_updated": getattr(node, 'last_updated', 0),
                    "changes": len(getattr(node, 'history', []))
                })

            return updates
        except Exception as e:
            logger.debug(f"Error getting recent updates: {e}")
            return []
