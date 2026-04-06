"""Search queries, full-text, and semantic search for EVA knowledge graph."""
import os
import logging
import time
import sqlite3
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger("eva.knowledge_graph")


class KnowledgeGraphQuerySearch:
    """Search query mixin for KnowledgeGraph."""

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

    def search_by_concept(self, concept: str, limit: int = 5) -> List[Any]:
        """Ищет узлы по концепту с использованием NLP."""
        UnifiedTextProcessor = _get_unified_text_processor()

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
        KnowledgeNode, KnowledgeEdge, NodeType, RelationType, safe_json_loads = _get_knowledge_graph_types()

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
