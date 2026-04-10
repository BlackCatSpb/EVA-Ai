"""
Graph Operations module for EVA Knowledge Graph
Traversal, search, queries, analytics, export/import.
"""
import os
import logging
import time
import json
import sqlite3
import math
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
from datetime import datetime

logger = logging.getLogger("eva_ai.knowledge_graph")


class KnowledgeGraphOperationsMixin:
    """Mixin class with query, search, traversal, analytics operations for KnowledgeGraph."""
    
    def get_node(self, node_id: str) -> Optional[Any]:
        """Возвращает узел по ID."""
        return self.nodes.get(node_id)
    
    def get_sources_for_node(self, node_id: str) -> List[Dict[str, Any]]:
        """Возвращает источники для узла."""
        node = self.nodes.get(node_id)
        if not node:
            return []
        
        sources = node.meta.get('sources', []) if isinstance(node.meta, dict) else []
        result = []
        for source in sources:
            if isinstance(source, dict):
                result.append({
                    'name': source.get('source', 'unknown'),
                    'reliability': source.get('reliability', 0.5),
                    'timestamp': source.get('timestamp', 0),
                    'user_id': source.get('user_id'),
                    'version': source.get('version', 1)
                })
            elif hasattr(source, 'source'):
                result.append({
                    'name': getattr(source, 'source', 'unknown'),
                    'reliability': getattr(source, 'reliability', 0.5),
                    'timestamp': getattr(source, 'timestamp', 0),
                    'user_id': getattr(source, 'user_id', None),
                    'version': getattr(source, 'version', 1)
                })
        return result
    
    def get_node_by_name(self, name: str) -> Optional[Any]:
        """Возвращает узел по имени."""
        name_lower = name.lower()
        for node in self.nodes.values():
            if node.name.lower() == name_lower or node.name.lower() == name_lower.strip():
                return node
            if name_lower in node.name.lower():
                return node
        return None
    
    def get_related_concepts(self, entity: str) -> List[Dict[str, Any]]:
        """Получить связанные концепты для сущности."""
        related = []
        entity_lower = entity.lower()
        
        node = self.get_node_by_name(entity)
        if node:
            related.append({
                "name": node.name,
                "type": node.node_type,
                "relation": "self"
            })
        
        for edge in self.edges.values():
            source_node = self.nodes.get(edge.source_id)
            target_node = self.nodes.get(edge.target_id)
            
            source_name = source_node.name if source_node else edge.source_id
            target_name = target_node.name if target_node else edge.target_id
            
            if source_name and entity_lower in source_name.lower():
                related.append({
                    "name": target_name,
                    "type": edge.relation_type,
                    "relation": "outgoing"
                })
            elif target_name and entity_lower in target_name.lower():
                related.append({
                    "name": source_name,
                    "type": edge.relation_type,
                    "relation": "incoming"
                })
        
        return related
    
    def get_relevant_nodes(self, query: str, limit: int = 50,
                           domains: Optional[List[str]] = None,
                           node_types: Optional[List[str]] = None,
                           min_strength: float = 0.0) -> List[Any]:
        """Возвращает релевантные запросу узлы."""
        try:
            if hasattr(self, "_search_nodes_internal"):
                nodes = self._search_nodes_internal(
                    query=query, limit=limit * 2,
                    domains=domains, node_types=node_types,
                    min_strength=min_strength
                )
            elif hasattr(self, "search_nodes"):
                nodes = self.search_nodes(query, limit=limit * 2)
                if domains:
                    nodes = [n for n in nodes if n.domain in domains]
                if node_types:
                    nodes = [n for n in nodes if n.node_type in node_types]
                if min_strength > 0.0:
                    nodes = [n for n in nodes if getattr(n, "strength", 0.0) >= min_strength]
            else:
                return []
            return nodes[:limit]
        except Exception:
            return []

    def update_node_weights(self, node_id: str, feedback: Dict[str, Any]):
        """Обновляет веса узла на основе обратной связи."""
        try:
            node = self.get_node(node_id)
            if not node:
                return

            verification = feedback.get("verification")
            if isinstance(verification, (int, float)):
                node.strength = max(0.0, min(1.0, node.strength * (0.5 + 0.5 * float(verification))))

            now = time.time()
            node.timestamp = now
            meta = getattr(node, "meta", {}) or {}
            meta["last_used"] = now
            meta["usage_count"] = int(meta.get("usage_count", 0)) + 1
            node.meta = meta

            if hasattr(self, "_update_node_in_db"):
                self._update_node_in_db(node)
            else:
                self.nodes[node.id] = node
        except Exception as e:
            logger.debug(f"Error updating node: {e}")

    def prioritize_nodes(self, query: str, nodes: List[Any]) -> List[Tuple[Any, float]]:
        """Возвращает узлы, отсортированные по динамическому весу."""
        scored: List[Tuple[Any, float]] = []
        for node in nodes:
            try:
                relevance = self._calculate_relevance(query, node)
                temporal_factor = self._calculate_temporal_factor(node)
                usage_factor = self._calculate_usage_factor(node)
                base_strength = getattr(node, "strength", 0.0) or 0.0

                total_weight = (base_strength * 0.3) + (relevance * 0.4) + (temporal_factor * 0.2) + (usage_factor * 0.1)
                scored.append((node, float(total_weight)))
            except Exception:
                scored.append((node, 0.0))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _calculate_relevance(self, query: str, node: Any) -> float:
        """Приближенная оценка релевантности."""
        try:
            text = f"{getattr(node, 'name', '')} {getattr(node, 'description', '')}".lower()
            q = (query or "").lower().strip()
            if not q:
                return 0.0
            q_tokens = [t for t in q.split() if len(t) > 2]
            if not q_tokens:
                return 0.0
            matches = sum(1 for t in set(q_tokens) if t in text)
            return max(0.0, min(1.0, matches / max(1, len(set(q_tokens)))))
        except Exception:
            return 0.0

    def _calculate_temporal_factor(self, node: Any) -> float:
        """Учет актуальности."""
        try:
            last = getattr(node, "last_updated", None) or getattr(node, "timestamp", None) or 0
            age = max(0.0, time.time() - float(last))
            day = 86400.0
            if age <= day:
                return 1.0
            elif age <= 30 * day:
                return 0.5 + 0.5 * (1 - (age - day) / (29 * day))
            elif age <= 180 * day:
                return 0.1 + 0.4 * (1 - (age - 30 * day) / (150 * day))
            else:
                return 0.05
        except Exception:
            return 0.5

    def _calculate_usage_factor(self, node: Any) -> float:
        """Нормализованная частота использования."""
        try:
            meta = getattr(node, "meta", {}) or {}
            usage = int(meta.get("usage_count", 0))
            if usage <= 0:
                return 0.0
            return max(0.0, min(1.0, math.log10(1 + usage) / 2.0))
        except Exception:
            return 0.0

    def get_edge(self, edge_id: str) -> Optional[Any]:
        """Возвращает связь по ID."""
        return self.edges.get(edge_id)
    
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
    
    def get_all_nodes(self) -> List[Any]:
        """Возвращает все узлы графа."""
        return list(self.nodes.values())
    
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

    def get_nodes_by_domain(self, domain: str) -> List[Any]:
        """Возвращает узлы указанного домена."""
        return [node for node in self.nodes.values() if node.domain == domain]
    
    def get_all_edges(self) -> List[Any]:
        """Возвращает все связи графа."""
        return list(self.edges.values())
    
    def search_nodes(self, query: str, limit: int = 10, 
                   domains: Optional[List[str]] = None,
                   node_types: Optional[List[str]] = None,
                   min_strength: float = 0.0) -> List[Any]:
        """Ищет узлы в графе знаний по запросу с поддержкой гибридного кэша."""
        start_time = time.time()
        
        cache_key = self._generate_cache_key(query, domains, node_types, min_strength, limit)
        
        if self.hybrid_cache:
            cached_result = self.hybrid_cache.get_token(cache_key)
            if cached_result:
                logger.debug(f"Найдены кэшированные результаты поиска для '{query}'")
                self._update_statistics(start_time, True)
                return cached_result
        
        results = self._search_nodes_internal(query, limit, domains, node_types, min_strength)
        
        if self.hybrid_cache and results:
            self.hybrid_cache.add_token(cache_key, results)
        
        self._update_statistics(start_time, True)
        return results

    def search(self, query: str, limit: int = 5, **kwargs) -> List[Dict]:
        return self.search_nodes(query, limit=limit, **kwargs)

    def _generate_cache_key(self, query: str, domains: Optional[List[str]], 
                           node_types: Optional[List[str]], min_strength: float, 
                           limit: int) -> str:
        """Генерирует ключ кэша для поискового запроса."""
        config = {
            "query": query,
            "domains": sorted(domains) if domains else None,
            "node_types": sorted(node_types) if node_types else None,
            "min_strength": min_strength,
            "limit": limit
        }
        config_str = json.dumps(config, sort_keys=True)
        return f"search:{hashlib.md5(config_str.encode()).hexdigest()}"
    
    def _search_nodes_internal(self, query: str, limit: int = 10, 
                              domains: Optional[List[str]] = None,
                              node_types: Optional[List[str]] = None,
                              min_strength: float = 0.0) -> List[Any]:
        """Выполняет внутренний поиск узлов."""
        from eva_ai.knowledge.knowledge_graph_types import KnowledgeNode, safe_json_loads
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query_lower = query.lower()
            params = [f"%{query_lower}%", f"%{query_lower}%", min_strength]
            
            sql = """
            SELECT id, name, description, node_type, domain, strength, timestamp, last_updated, 
                   version, meta, spatial_info, temporal_info, history, contradictions, 
                   keyword_index, concept_index
            FROM nodes
            WHERE (LOWER(name) LIKE ? OR LOWER(description) LIKE ?)
            AND strength >= ?
            """
            
            if domains and len(domains) > 0:
                placeholders = ",".join(["?" for _ in domains])
                sql += f" AND domain IN ({placeholders})"
                params.extend(domains)
            
            if node_types and len(node_types) > 0:
                placeholders = ",".join(["?" for _ in node_types])
                sql += f" AND node_type IN ({placeholders})"
                params.extend(node_types)
            
            sql += " ORDER BY strength DESC LIMIT ?"
            params.append(limit)
            
            placeholders_count = sql.count("?")
            if placeholders_count != len(params):
                logger.error(
                    f"Несоответствие параметров SQL: плейсхолдеров={placeholders_count}, параметров={len(params)}. "
                    f"SQL={sql} | params={params}"
                )
            else:
                logger.debug(
                    f"Выполняем SQL запрос. Плейсхолдеров={placeholders_count}, параметров={len(params)}. "
                    f"SQL={sql} | params={params}"
                )

            try:
                cursor.execute(sql, params)
            except Exception as e:
                logger.error(f"SQL execution failed: {e}. SQL={sql} | params={params}")
                return []

            results = []
            for row in cursor.fetchall():
                node = KnowledgeNode(
                    id=row[0],
                    name=row[1],
                    description=row[2],
                    node_type=row[3],
                    domain=row[4],
                    strength=row[5],
                    timestamp=row[6],
                    meta=safe_json_loads(row[9]) if len(row) > 9 and row[9] else {},
                    version=row[8],
                    spatial_info=safe_json_loads(row[10]) if len(row) > 10 and row[10] else {},
                    temporal_info=safe_json_loads(row[11]) if len(row) > 11 and row[11] else {}
                )
                node.last_updated = row[7]
                node.history = safe_json_loads(row[12]) if len(row) > 12 and row[12] else []
                node.contradictions = safe_json_loads(row[13]) if len(row) > 13 and row[13] else []
                node.keyword_index = safe_json_loads(row[14]) if len(row) > 14 and row[14] else []
                node.concept_index = safe_json_loads(row[15]) if len(row) > 15 and row[15] else []
                
                results.append(node)
            
            logger.debug(f"Найдено {len(results)} узлов по запросу '{query}'")
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Ошибка поиска узлов в графе знаний: {e}", exc_info=True)
            return []
    
    def search_by_concept(self, concept: str, limit: int = 5) -> List[Any]:
        """Ищет узлы по концепту с использованием NLP."""
        try:
            from eva_ai.mlearning.unified_text_processor import UnifiedTextProcessor
        except ImportError:
            UnifiedTextProcessor = None
        
        if not self.text_processor:
            return self.search_nodes(concept, limit)
        
        try:
            analysis = self.text_processor.process_text(concept)
            
            keywords = [kw["word"] for kw in analysis.keywords[:3]] if analysis.keywords else [concept]
            
            all_results = []
            for keyword in keywords:
                results = self.search_nodes(keyword, limit=limit)
                all_results.extend(results)
            
            seen = set()
            unique_results = []
            for result in all_results:
                if result.id not in seen:
                    seen.add(result.id)
                    unique_results.append(result)
            
            unique_results.sort(key=lambda x: x.strength, reverse=True)
            
            return unique_results[:limit]
            
        except Exception as e:
            logger.error(f"Ошибка поиска по концепту: {e}", exc_info=True)
            return self.search_nodes(concept, limit)
    
    def update_node(self, node_id: str, new_description: str, 
                   strength: Optional[float] = None, 
                   source: Optional[str] = None,
                   user_id: Optional[str] = None,
                   spatial_info: Optional[Dict[str, Any]] = None,
                   temporal_info: Optional[Dict[str, Any]] = None) -> bool:
        """Обновляет узел в графе знаний."""
        try:
            node = self.get_node(node_id)
            if not node:
                logger.warning(f"Узел с ID {node_id} не найден")
                return False
            
            node.update(
                new_description,
                strength=strength,
                source=source,
                user_id=user_id,
                spatial_info=spatial_info,
                temporal_info=temporal_info
            )
            
            self._update_node_in_db(node)
            
            self._update_indexes(node=node)
            
            self._record_history(
                "node_updated",
                {
                    "node_id": node_id,
                    "new_description": new_description,
                    "new_strength": strength,
                    "spatial_info": spatial_info,
                    "temporal_info": temporal_info
                },
                node_id=node_id,
                user_id=user_id,
                source=source
            )
            
            logger.info(f"Узел '{node.name}' (ID: {node_id}) успешно обновлен")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка обновления узла: {e}", exc_info=True)
            return False
    
    def update_edge(self, edge_id: str, new_strength: Optional[float] = None,
                  source: Optional[str] = None,
                  user_id: Optional[str] = None,
                  spatial_info: Optional[Dict[str, Any]] = None,
                  temporal_info: Optional[Dict[str, Any]] = None) -> bool:
        """Обновляет связь в графе знаний."""
        try:
            edge = self.get_edge(edge_id)
            if not edge:
                logger.warning(f"Связь с ID {edge_id} не найдена")
                return False
            
            edge.update(
                new_strength=new_strength,
                source=source,
                user_id=user_id,
                spatial_info=spatial_info,
                temporal_info=temporal_info
            )
            
            self._update_edge_in_db(edge)
            
            self._update_indexes(edge=edge)
            
            self._record_history(
                "edge_updated",
                {
                    "edge_id": edge_id,
                    "new_strength": new_strength,
                    "spatial_info": spatial_info,
                    "temporal_info": temporal_info
                },
                edge_id=edge_id,
                user_id=user_id,
                source=source
            )
            
            logger.info(f"Связь (ID: {edge_id}) успешно обновлена")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка обновления связи: {e}", exc_info=True)
            return False
    
    def remove_node(self, node_id: str, user_id: Optional[str] = None) -> bool:
        """Удаляет узел из графа знаний."""
        try:
            node = self.get_node(node_id)
            if not node:
                logger.warning(f"Узел с ID {node_id} не найден")
                return False
            
            edges_to_remove = []
            for edge_id, edge in self.edges.items():
                if edge.source_id == node_id or edge.target_id == node_id:
                    edges_to_remove.append(edge_id)
            
            for edge_id in edges_to_remove:
                del self.edges[edge_id]
                self._remove_edge_from_db(edge_id)
            
            del self.nodes[node_id]
            self._remove_node_from_db(node_id)
            
            self._record_history(
                "node_removed",
                {"node_id": node_id, "name": node.name},
                node_id=node_id,
                user_id=user_id
            )
            
            self.stats["total_nodes"] -= 1
            
            logger.info(f"Узел '{node.name}' (ID: {node_id}) успешно удален")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка удаления узла: {e}", exc_info=True)
            return False
    
    def remove_all_concepts(self, user_id: Optional[str] = None) -> int:
        """Массово удаляет все узлы типа 'concept' из графа знаний."""
        try:
            concept_ids = [n.id for n in list(self.nodes.values()) if getattr(n, "node_type", None) == "concept"]
            removed = 0
            for nid in concept_ids:
                try:
                    if self.remove_node(nid, user_id=user_id):
                        removed += 1
                except Exception:
                    logger.error(f"Не удалось удалить концепт с ID: {nid}", exc_info=True)
            logger.info(f"Массовое удаление концептов завершено: удалено {removed} из {len(concept_ids)}")
            return removed
        except Exception as e:
            logger.error(f"Ошибка массового удаления концептов: {e}", exc_info=True)
            return 0
    
    def remove_edge(self, edge_id: str, user_id: Optional[str] = None) -> bool:
        """Удаляет связь из графа знаний."""
        try:
            edge = self.get_edge(edge_id)
            if not edge:
                logger.warning(f"Связь с ID {edge_id} не найдена")
                return False
            
            del self.edges[edge_id]
            self._remove_edge_from_db(edge_id)
            
            self._record_history(
                "edge_removed",
                {
                    "edge_id": edge_id,
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "relation_type": edge.relation_type
                },
                edge_id=edge_id,
                user_id=user_id
            )
            
            self.stats["total_edges"] -= 1
            
            logger.info(f"Связь (ID: {edge_id}) успешно удалена")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка удаления связи: {e}", exc_info=True)
            return False
    
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
    
    def _get_relation_type_statistics(self) -> Dict[str, int]:
        """Возвращает статистику по типам связей."""
        stats = {}
        for edge in self.edges.values():
            if edge.relation_type in stats:
                stats[edge.relation_type] += 1
            else:
                stats[edge.relation_type] = 1
        return stats
    
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
    
    def is_initialized(self) -> bool:
        """Проверяет, инициализирован ли граф."""
        return self.initialized
    
    def is_running(self) -> bool:
        """Проверяет, запущен ли граф."""
        return self.running
    
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
    
    def find_path(self, start_node_id: str, end_node_id: str, 
                 max_length: int = 5) -> List[List[str]]:
        """Находит пути между двумя узлами в графе."""
        if start_node_id not in self.nodes or end_node_id not in self.nodes:
            return []
        
        paths = []
        queue = deque([(start_node_id, [start_node_id])])
        visited = set([start_node_id])
        
        while queue:
            current_node, path = queue.popleft()
            
            if current_node == end_node_id:
                paths.append(path)
                if len(paths) >= 10:
                    break
                continue
            
            if len(path) >= max_length:
                continue
            
            related_nodes = self.get_related_nodes(current_node)
            for node in related_nodes:
                if node.id not in visited:
                    visited.add(node.id)
                    queue.append((node.id, path + [node.id]))
        
        return paths
    
    def get_temporal_relations(self, start_time: Optional[float] = None, 
                              end_time: Optional[float] = None,
                              limit: int = 20) -> List[Dict[str, Any]]:
        """Возвращает временные отношения в указанном временном интервале."""
        start_time = start_time or 0
        end_time = end_time or time.time()
        
        results = []
        
        for timestamp, item_id, item_type in self.temporal_index:
            if timestamp < start_time:
                continue
            if timestamp > end_time:
                break
            
            if item_type == "node":
                node = self.nodes.get(item_id)
                if node and node.temporal_info:
                    results.append({
                        "type": "node",
                        "id": item_id,
                        "name": node.name,
                        "timestamp": timestamp,
                        "temporal_info": node.temporal_info,
                        "domain": node.domain
                    })
            elif item_type == "edge":
                edge = self.edges.get(item_id)
                if edge and edge.temporal_info:
                    results.append({
                        "type": "edge",
                        "id": item_id,
                        "relation_type": edge.relation_type,
                        "timestamp": timestamp,
                        "temporal_info": edge.temporal_info
                    })
            
            if len(results) >= limit:
                break
        
        return results
    
    def get_spatial_relations(self, location: Dict[str, float], 
                             max_distance: float = 100.0,
                             limit: int = 20) -> List[Dict[str, Any]]:
        """Возвращает пространственные отношения в указанном радиусе."""
        results = []
        
        for node in self.nodes.values():
            if not node.spatial_info or "coordinates" not in node.spatial_info:
                continue
            
            node_coords = node.spatial_info["coordinates"]
            distance = self._calculate_distance(
                location["lat"], location["lon"],
                node_coords["lat"], node_coords["lon"]
            )
            
            if distance <= max_distance:
                results.append({
                    "type": "node",
                    "id": node.id,
                    "name": node.name,
                    "distance": distance,
                    "spatial_info": node.spatial_info,
                    "domain": node.domain
                })
        
        for edge in self.edges.values():
            if not edge.spatial_info or "coordinates" not in edge.spatial_info:
                continue
            
            edge_coords = edge.spatial_info["coordinates"]
            distance = self._calculate_distance(
                location["lat"], location["lon"],
                edge_coords["lat"], edge_coords["lon"]
            )
            
            if distance <= max_distance:
                results.append({
                    "type": "edge",
                    "id": edge.id,
                    "relation_type": edge.relation_type,
                    "distance": distance,
                    "spatial_info": edge.spatial_info
                })
        
        results.sort(key=lambda x: x["distance"])
        
        return results[:limit]
    
    def _calculate_distance(self, lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        """Вычисляет расстояние между двумя точками на сфере с использованием формулы Haversine."""
        R = 6371.0
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        distance = R * c
        return distance
    
    def generate_knowledge_graph(self, concept: str, depth: int = 2) -> Dict[str, Any]:
        """Генерирует граф знаний для визуализации."""
        nodes = self.search_nodes(concept, limit=1)
        if not nodes:
            return {"nodes": [], "edges": []}
        
        center_node = nodes[0]
        
        subgraph = self.get_subgraph(center_node.id, depth)
        
        viz_nodes = []
        for node in subgraph["nodes"]:
            viz_nodes.append({
                "id": node["id"],
                "label": node["name"],
                "title": node["description"],
                "group": node["domain"],
                "shape": "ellipse",
                "font": {"size": 14}
            })
        
        viz_edges = []
        for edge in subgraph["edges"]:
            viz_edges.append({
                "from": edge["source_id"],
                "to": edge["target_id"],
                "label": edge["relation_type"],
                "arrows": "to",
                "font": {"align": "middle"}
            })
        
        return {
            "nodes": viz_nodes,
            "edges": viz_edges
        }
    
    def export_graph(self, format: str = "json") -> Any:
        """Экспортирует граф знаний в указанный формат."""
        if format == "json":
            return {
                "nodes": [node.to_dict() for node in self.nodes.values()],
                "edges": [edge.to_dict() for edge in self.edges.values()]
            }
        elif format == "graphml":
            return self._export_to_graphml()
        elif format == "gexf":
            return self._export_to_gexf()
        else:
            raise ValueError(f"Неподдерживаемый формат экспорта: {format}")
    
    def _export_to_graphml(self) -> str:
        """Экспортирует граф в формат GraphML."""
        from xml.etree.ElementTree import Element, SubElement, tostring
        import xml.dom.minidom
        
        graphml = Element('graphml', {
            'xmlns': 'http://graphml.graphdrawing.org/xmlns',
            'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
            'xsi:schemaLocation': 'http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd'
        })
        
        key_node = SubElement(graphml, 'key', {
            'id': 'd0',
            'for': 'node',
            'attr.name': 'description',
            'attr.type': 'string'
        })
        SubElement(key_node, 'default').text = ''
        
        key_domain = SubElement(graphml, 'key', {
            'id': 'd1',
            'for': 'node',
            'attr.name': 'domain',
            'attr.type': 'string'
        })
        SubElement(key_domain, 'default').text = 'general'
        
        key_strength = SubElement(graphml, 'key', {
            'id': 'd2',
            'for': 'node',
            'attr.name': 'strength',
            'attr.type': 'double'
        })
        SubElement(key_strength, 'default').text = '0.5'
        
        key_relation = SubElement(graphml, 'key', {
            'id': 'd3',
            'for': 'edge',
            'attr.name': 'relation_type',
            'attr.type': 'string'
        })
        SubElement(key_relation, 'default').text = 'related_to'
        
        graph = SubElement(graphml, 'graph', {
            'id': 'G',
            'edgedefault': 'directed'
        })
        
        for node in self.nodes.values():
            node_elem = SubElement(graph, 'node', {'id': node.id})
            SubElement(node_elem, 'data', {'key': 'd0'}).text = node.description
            SubElement(node_elem, 'data', {'key': 'd1'}).text = node.domain
            SubElement(node_elem, 'data', {'key': 'd2'}).text = str(node.strength)
        
        for edge in self.edges.values():
            edge_elem = SubElement(graph, 'edge', {
                'id': edge.id,
                'source': edge.source_id,
                'target': edge.target_id
            })
            SubElement(edge_elem, 'data', {'key': 'd3'}).text = edge.relation_type
        
        xml_str = tostring(graphml, encoding='utf-8')
        dom = xml.dom.minidom.parseString(xml_str)
        return dom.toprettyxml()
    
    def _export_to_gexf(self) -> str:
        """Экспортирует граф в формат GEXF."""
        from xml.etree.ElementTree import Element, SubElement, tostring
        import xml.dom.minidom
        
        gexf = Element('gexf', {
            'xmlns': 'http://www.gexf.net/1.2draft',
            'version': '1.2'
        })
        
        meta = SubElement(gexf, 'meta', {'lastmodified': datetime.now().isoformat()})
        SubElement(meta, 'creator').text = 'ЕВА Knowledge Graph'
        SubElement(meta, 'description').text = 'Graph of knowledge exported from ЕВА'
        
        SubElement(gexf, 'visualization')
        
        graph = SubElement(gexf, 'graph', {
            ' defaultedgetype': 'directed',
            'mode': 'static'
        })
        
        node_attributes = SubElement(graph, 'attributes', {'class': 'node'})
        SubElement(node_attributes, 'attribute', {
            'id': 'description',
            'title': 'Description',
            'type': 'string'
        })
        SubElement(node_attributes, 'attribute', {
            'id': 'domain',
            'title': 'Domain',
            'type': 'string'
        })
        SubElement(node_attributes, 'attribute', {
            'id': 'strength',
            'title': 'Strength',
            'type': 'float'
        })
        
        edge_attributes = SubElement(graph, 'attributes', {'class': 'edge'})
        SubElement(edge_attributes, 'attribute', {
            'id': 'relation_type',
            'title': 'Relation Type',
            'type': 'string'
        })
        
        nodes_elem = SubElement(graph, 'nodes')
        for node in self.nodes.values():
            node_elem = SubElement(nodes_elem, 'node', {
                'id': node.id,
                'label': node.name
            })
            attvalues = SubElement(node_elem, 'attvalues')
            SubElement(attvalues, 'attvalue', {
                'for': 'description',
                'value': node.description
            })
            SubElement(attvalues, 'attvalue', {
                'for': 'domain',
                'value': node.domain
            })
            SubElement(attvalues, 'attvalue', {
                'for': 'strength',
                'value': str(node.strength)
            })
        
        edges_elem = SubElement(graph, 'edges')
        for i, edge in enumerate(self.edges.values()):
            edge_elem = SubElement(edges_elem, 'edge', {
                'id': str(i),
                'source': edge.source_id,
                'target': edge.target_id
            })
            attvalues = SubElement(edge_elem, 'attvalues')
            SubElement(attvalues, 'attvalue', {
                'for': 'relation_type',
                'value': edge.relation_type
            })
        
        xml_str = tostring(gexf, encoding='utf-8')
        dom = xml.dom.minidom.parseString(xml_str)
        return dom.toprettyxml()
    
    def import_graph(self, data: Any, format: str = "json") -> bool:
        """Импортирует граф знаний из указанного формата."""
        try:
            if format == "json":
                return self._import_from_json(data)
            elif format == "graphml":
                return self._import_from_graphml(data)
            elif format == "gexf":
                return self._import_from_gexf(data)
            else:
                raise ValueError(f"Неподдерживаемый формат импорта: {format}")
        except Exception as e:
            logger.error(f"Ошибка импорта графа: {e}", exc_info=True)
            return False
    
    def _import_from_json(self, data: Dict[str, Any]) -> bool:
        """Импортирует граф из JSON."""
        from eva_ai.knowledge.knowledge_graph_types import KnowledgeNode, KnowledgeEdge
        
        nodes = data.get("nodes", [])
        for node_data in nodes:
            node = KnowledgeNode.from_dict(node_data)
            self.nodes[node.id] = node
            self._save_node_to_db(node)
        
        edges = data.get("edges", [])
        for edge_data in edges:
            edge = KnowledgeEdge.from_dict(edge_data)
            self.edges[edge.id] = edge
            self._save_edge_to_db(edge)
        
        self._init_indexes()
        
        logger.info(f"Импортировано {len(nodes)} узлов и {len(edges)} связей")
        return True
    
    def _import_from_graphml(self, data: str) -> bool:
        """Импортирует граф из GraphML."""
        import xml.etree.ElementTree as ET
        
        root = ET.fromstring(data)
        namespace = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}
        
        graph = root.find('graphml:graph', namespace)
        if graph is None:
            raise ValueError("Не найден элемент графа в GraphML")
        
        for node in graph.findall('graphml:node', namespace):
            node_id = node.get('id')
            
            description = ""
            domain = "general"
            strength = 0.5
            
            for data_elem in node.findall('graphml:data', namespace):
                key = data_elem.get('key')
                if key == 'd0':
                    description = data_elem.text or ""
                elif key == 'd1':
                    domain = data_elem.text or "general"
                elif key == 'd2':
                    try:
                        strength = float(data_elem.text or "0.5")
                    except ValueError:
                        strength = 0.5
            
            name = node_id
            self.add_node(name, description, domain=domain, strength=strength)
        
        for edge in graph.findall('graphml:edge', namespace):
            source_id = edge.get('source', edge.get('source_id', ''))
            target_id = edge.get('target', edge.get('target_id', ''))
            edge_id = edge.get('id', f"edge_{hash(edge)}")
            
            relation_type = "related_to"
            for data_elem in edge.findall('graphml:data', namespace):
                if data_elem.get('key') == 'd3':
                    relation_type = data_elem.text or "related_to"
            
            self.add_edge(source_id, target_id, relation_type)
        
        logger.info("Граф успешно импортирован из GraphML")
        return True
    
    def _import_from_gexf(self, data: str) -> bool:
        """Импортирует граф из GEXF."""
        import xml.etree.ElementTree as ET
        
        root = ET.fromstring(data)
        namespace = {'gexf': 'http://www.gexf.net/1.2draft'}
        
        graph = root.find('gexf:graph', namespace)
        if graph is None:
            raise ValueError("Не найден элемент графа в GEXF")
        
        nodes = graph.find('gexf:nodes', namespace)
        if nodes is not None:
            for node in nodes.findall('gexf:node', namespace):
                node_id = node.get('id')
                label = node.get('label', node_id)
                
                description = ""
                domain = "general"
                strength = 0.5
                
                attvalues = node.find('gexf:attvalues', namespace)
                if attvalues is not None:
                    for attvalue in attvalues.findall('gexf:attvalue', namespace):
                        for_attr = attvalue.get('for')
                        value = attvalue.get('value', '')
                        
                        if for_attr == 'description':
                            description = value
                        elif for_attr == 'domain':
                            domain = value
                        elif for_attr == 'strength':
                            try:
                                strength = float(value)
                            except ValueError:
                                strength = 0.5
                
                self.add_node(label, description, domain=domain, strength=strength)
        
        edges = graph.find('gexf:edges', namespace)
        if edges is not None:
            for edge in edges.findall('gexf:edge', namespace):
                source_id = edge.get('source')
                target_id = edge.get('target')
                
                relation_type = "related_to"
                attvalues = edge.find('gexf:attvalues', namespace)
                if attvalues is not None:
                    for attvalue in attvalues.findall('gexf:attvalue', namespace):
                        if attvalue.get('for') == 'relation_type':
                            relation_type = attvalue.get('value', 'related_to')
                
                self.add_edge(source_id, target_id, relation_type)
        
        logger.info("Граф успешно импортирован из GEXF")
        return True
    
    def analyze_knowledge_gaps(self, domain: str, num_samples: int = 10) -> List[Dict[str, Any]]:
        """Анализирует пробелы в знаниях в указанной области."""
        try:
            domain_nodes = self.get_nodes_by_domain(domain)
            
            gaps = []
            node_ids = [n.id for n in domain_nodes]
            
            for node in domain_nodes[:num_samples]:
                connected = set()
                for edge in self.get_edges(node.id):
                    connected.add(edge.source_id)
                    connected.add(edge.target_id)
                
                unconnected = [nid for nid in node_ids if nid != node.id and nid not in connected]
                
                if len(unconnected) > 3:
                    gaps.append({
                        "node_id": node.id,
                        "name": node.name,
                        "potential_connections": len(unconnected),
                        "unconnected_nodes": unconnected[:5]
                    })
            
            return gaps
            
        except Exception as e:
            logger.error(f"Ошибка анализа пробелов в знаниях: {e}", exc_info=True)
            return []
    
    def find_central_nodes(self, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """Находит центральные узлы в графе (по количеству связей)."""
        try:
            node_scores = {}
            
            nodes_to_check = self.nodes.values() if domain is None else self.get_nodes_by_domain(domain)
            
            for node in nodes_to_check:
                edges = self.get_edges(node.id)
                node_scores[node.id] = {
                    "id": node.id,
                    "name": node.name,
                    "degree": len(edges),
                    "domain": node.domain
                }
            
            sorted_nodes = sorted(node_scores.values(), key=lambda x: x["degree"], reverse=True)
            return sorted_nodes[:20]
            
        except Exception as e:
            logger.error(f"Ошибка поиска центральных узлов: {e}", exc_info=True)
            return []
    
    def find_islands(self, min_degree: int = 1) -> List[Dict[str, Any]]:
        """Находит изолированные узлы или группы (островные узлы)."""
        try:
            islands = []
            
            for node in self.nodes.values():
                edges = self.get_edges(node.id)
                if len(edges) <= min_degree:
                    islands.append({
                        "node_id": node.id,
                        "name": node.name,
                        "degree": len(edges),
                        "domain": node.domain
                    })
            
            islands.sort(key=lambda x: x["degree"])
            return islands
            
        except Exception as e:
            logger.error(f"Ошибка поиска изолированных узлов: {e}", exc_info=True)
            return []
    
    def calculate_node_importance(self, node_id: str) -> Dict[str, float]:
        """Вычисляет важность узла различными методами."""
        try:
            node = self.get_node(node_id)
            if not node:
                return {}
            
            degree = len(self.get_edges(node_id))
            
            in_degree = len(self.get_edges(node_id, direction="target"))
            out_degree = len(self.get_edges(node_id, direction="source"))
            
            related = self.get_related_nodes(node_id, max_distance=2)
            indirect_connections = len(related)
            
            centrality = {
                "degree_centrality": degree / max(1, len(self.nodes) - 1),
                "in_degree": in_degree,
                "out_degree": out_degree,
                "indirect_connections": indirect_connections,
                "normalized_centrality": degree / max(1, sum(len(self.get_edges(n.id)) for n in self.nodes.values()))
            }
            
            return centrality
            
        except Exception as e:
            logger.error(f"Ошибка вычисления важности узла: {e}", exc_info=True)
            return {}
    
    def find_communities(self) -> List[List[str]]:
        """Находит сообщества в графе (упрощенная реализация)."""
        try:
            visited = set()
            communities = []
            
            for node_id in self.nodes:
                if node_id in visited:
                    continue
                
                community = []
                queue = deque([node_id])
                
                while queue:
                    current = queue.popleft()
                    if current in visited:
                        continue
                    
                    visited.add(current)
                    community.append(current)
                    
                    for edge in self.get_edges(current):
                        neighbor = edge.target_id if edge.source_id == current else edge.source_id
                        if neighbor not in visited:
                            queue.append(neighbor)
                
                if community:
                    communities.append(community)
            
            return communities
            
        except Exception as e:
            logger.error(f"Ошибка поиска сообществ: {e}", exc_info=True)
            return []
    
    def get_graph_density(self) -> float:
        """Вычисляет плотность графа."""
        try:
            n = len(self.nodes)
            if n < 2:
                return 0.0
            
            max_edges = n * (n - 1)
            actual_edges = len(self.edges)
            
            return actual_edges / max_edges if max_edges > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Ошибка вычисления плотности графа: {e}", exc_info=True)
            return 0.0
    
    def get_diameter(self) -> int:
        """Вычисляет диаметр графа (упрощенно - максимальное кратчайшее расстояние)."""
        try:
            if not self.nodes:
                return 0
            
            max_distance = 0
            
            sample_nodes = list(self.nodes.keys())[:min(50, len(self.nodes))]
            
            for start in sample_nodes:
                visited = {start: 0}
                queue = deque([start])
                
                while queue:
                    current = queue.popleft()
                    current_dist = visited[current]
                    
                    if current_dist > max_distance:
                        max_distance = current_dist
                    
                    if current_dist >= 5:
                        continue
                    
                    for edge in self.get_edges(current):
                        neighbor = edge.target_id if edge.source_id == current else edge.source_id
                        if neighbor not in visited:
                            visited[neighbor] = current_dist + 1
                            queue.append(neighbor)
            
            return max_distance
            
        except Exception as e:
            logger.error(f"Ошибка вычисления диаметра графа: {e}", exc_info=True)
            return 0
