"""
Graph Edges module for EVA Knowledge Graph
Edge creation, relationships, weights, types.
"""
import os
import logging
import time
import sqlite3
import json
import hashlib
from typing import Dict, List, Optional, Any

logger = logging.getLogger("eva.knowledge_graph")


class KnowledgeGraphEdgeMixin:
    """Mixin class with edge operations for KnowledgeGraph."""
    
    def add_edge(self, source_id: str, target_id: str, relation_type: str,
                strength: float = 0.5, meta: Optional[Dict] = None,
                spatial_info: Optional[Dict] = None,
                temporal_info: Optional[Dict] = None, user_id: Optional[str] = None,
                source: Optional[str] = None) -> str:
        """
        Добавляет связь между узлами в графе знаний.
        
        Args:
            source_id: ID исходного узла
            target_id: ID целевого узла
            relation_type: Тип связи
            strength: Сила связи
            meta: Дополнительные метаданные
            spatial_info: Пространственная информация
            temporal_info: Временная информация
            user_id: ID пользователя
            source: Источник информации
            
        Returns:
            str: ID добавленной связи
        """
        start_time = time.time()
        edge_id = f"edge_{int(time.time())}_{hashlib.md5(f'{source_id}_{target_id}'.encode()).hexdigest()[:8]}"
        
        try:
            if source_id not in self.nodes:
                logger.warning(f"Исходный узел {source_id} не существует")
                self._update_statistics(start_time, False)
                return ""
            
            if target_id not in self.nodes:
                logger.warning(f"Целевой узел {target_id} не существует")
                self._update_statistics(start_time, False)
                return ""
            
            existing_edge = None
            for edge in self.edges.values():
                if (edge.source_id == source_id and edge.target_id == target_id and 
                    edge.relation_type == relation_type):
                    existing_edge = edge
                    break
            
            if existing_edge:
                existing_edge.update(
                    new_strength=strength,
                    source=source,
                    user_id=user_id,
                    spatial_info=spatial_info,
                    temporal_info=temporal_info
                )
                self._update_edge_in_db(existing_edge)
                self._update_indexes(edge=existing_edge)
                
                self.stats["edge_updates"] += 1
                self._update_statistics(start_time, True)
                
                logger.info(f"Обновлена существующая связь в графе знаний: {existing_edge.id}")
                return existing_edge.id
            
            from eva.knowledge.knowledge_graph_types import KnowledgeEdge
            edge = KnowledgeEdge(
                id=edge_id,
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                strength=strength,
                meta=meta or {},
                spatial_info=spatial_info,
                temporal_info=temporal_info
            )
            
            if source:
                if not edge.meta:
                    edge.meta = {}
                if 'sources' not in edge.meta:
                    edge.meta['sources'] = []
                edge.meta['sources'].append({
                    'source': source,
                    'timestamp': time.time(),
                    'user_id': user_id,
                    'version': edge.version
                })
            
            self._save_edge_to_db(edge)
            
            self.edges[edge_id] = edge
            self._update_indexes(edge=edge)
            
            self.stats["total_edges"] += 1
            self.stats["edge_creations"] += 1
            self._update_statistics(start_time, True)
            
            logger.debug(f"Добавлена связь в граф знаний: {edge_id}")
            return edge_id
            
        except Exception as e:
            logger.error(f"Ошибка добавления связи в граф знаний: {e}", exc_info=True)
            self._update_statistics(start_time, False)
            return ""
    
    def _save_edge_to_db(self, edge) -> bool:
        """Сохраняет связь в базу данных с использованием транзакций."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.cursor()
            
            cursor.execute("""
            INSERT OR REPLACE INTO edges 
            (id, source_id, target_id, relation_type, strength, timestamp, 
            last_updated, version, meta, spatial_info, temporal_info, history)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                edge.id,
                edge.source_id,
                edge.target_id,
                edge.relation_type,
                edge.strength,
                edge.timestamp,
                edge.last_updated,
                edge.version,
                json.dumps(edge.meta),
                json.dumps(edge.spatial_info),
                json.dumps(edge.temporal_info),
                json.dumps(edge.history)
            ))
            
            conn.commit()
            return True
            
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Ошибка сохранения связи {edge.id} в БД: {e}", exc_info=True)
            return False
        finally:
            if conn:
                conn.close()
    
    def _update_edge_in_db(self, edge) -> bool:
        """Обновляет связь в базе данных с использованием транзакций."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.cursor()
            
            cursor.execute("""
            UPDATE edges SET
                source_id = ?,
                target_id = ?,
                relation_type = ?,
                strength = ?,
                last_updated = ?,
                version = ?,
                meta = ?,
                spatial_info = ?,
                temporal_info = ?,
                history = ?
            WHERE id = ?
            """, (
                edge.source_id,
                edge.target_id,
                edge.relation_type,
                edge.strength,
                edge.last_updated,
                edge.version,
                json.dumps(edge.meta),
                json.dumps(edge.spatial_info),
                json.dumps(edge.temporal_info),
                json.dumps(edge.history),
                edge.id
            ))
            
            conn.commit()
            return True
            
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Ошибка обновления связи {edge.id} в БД: {e}", exc_info=True)
            return False
        finally:
            if conn:
                conn.close()
    
    def _remove_edge_from_db(self, edge_id: str) -> bool:
        """Удаляет связь из базы данных с использованием транзакций."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.cursor()
            
            with conn:
                cursor.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка удаления связи {edge_id} из БД: {e}", exc_info=True)
            return False
