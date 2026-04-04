"""
Graph Nodes module for EVA Knowledge Graph
Node creation, editing, deletion, properties, types.
"""
import os
import logging
import time
import sqlite3
import json
import hashlib
from typing import Dict, List, Optional, Any

logger = logging.getLogger("eva.knowledge_graph")


class KnowledgeGraphNodeMixin:
    """Mixin class with node operations for KnowledgeGraph."""
    
    def add_ambiguity_metadata(self, node_id: str, ambiguity_info: Dict) -> bool:
        """Add ambiguity metadata to a node."""
        if node_id not in self.nodes:
            return False
        
        node = self.nodes[node_id]
        if node.meta and isinstance(node.meta, dict):
            node.meta['ambiguity'] = ambiguity_info
        else:
            node.meta = {'ambiguity': ambiguity_info}
        
        try:
            self._save_node_to_db(node)
        except Exception as e:
            logger.debug(f"Error saving node to DB: {e}")
        return True
    
    def get_disambiguation_candidates(self, ambiguous_term: str) -> List[Dict]:
        """Get possible disambiguations for an ambiguous term."""
        candidates = []
        
        for node in self.nodes.values():
            node_text = getattr(node, 'name', '') + ' ' + getattr(node, 'description', '')
            if ambiguous_term.lower() in node_text.lower():
                candidates.append({
                    'node_id': getattr(node, 'id', ''),
                    'name': getattr(node, 'name', ''),
                    'description': getattr(node, 'description', ''),
                    'type': getattr(node, 'node_type', '')
                })
        
        return candidates
    
    def search_with_disambiguation(self, query: str, clarified_terms: Dict[str, str] = None) -> List:
        """Search with clarified meanings."""
        refined_query = query
        
        if clarified_terms:
            for ambiguous, clarified in clarified_terms.items():
                refined_query = refined_query.replace(ambiguous, clarified)
        
        return self.search_nodes(refined_query)
    
    def add_node(self, name: str = None, description: str = None, node_type: str = "fact", 
                domain: str = "general", strength: float = 0.5,
                meta: Optional[Dict] = None, spatial_info: Optional[Dict] = None,
                temporal_info: Optional[Dict] = None, user_id: Optional[str] = None,
                source: Optional[str] = None,
                node_id: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
        """
        Добавляет новый узел в граф знаний.
        
        Args:
            name: Название узла
            description: Описание узла
            node_type: Тип узла
            domain: Домен знаний
            strength: Сила знания
            meta: Дополнительные метаданные
            spatial_info: Пространственная информация
            temporal_info: Временная информация
            user_id: ID пользователя
            source: Источник информации
            node_id: Опциональный ID узла
            metadata: Алиас для meta
            
        Returns:
            str: ID добавленного узла
        """
        if description is None:
            description = ""
        if meta is None and metadata is not None:
            meta = metadata
        if meta is None:
            meta = {}
            
        start_time = time.time()
        if node_id is None:
            node_id = f"node_{int(time.time())}_{hashlib.md5((name or '').encode()).hexdigest()[:8]}"
        
        try:
            existing_nodes = self.search_nodes(name, limit=1, domains=[domain])
            if existing_nodes:
                existing_node = existing_nodes[0]
                existing_node.update(
                    description, 
                    strength=strength,
                    source=source,
                    user_id=user_id,
                    spatial_info=spatial_info,
                    temporal_info=temporal_info
                )
                self._update_node_in_db(existing_node)
                self._update_indexes(node=existing_node)
                
                self.stats["node_updates"] += 1
                self._update_statistics(start_time, True)
                
                logger.info(f"Обновлен существующий узел в графе знаний: {existing_node.id} ({name})")
                return existing_node.id
            
            from eva.knowledge.knowledge_graph_types import KnowledgeNode
            node = KnowledgeNode(
                id=node_id,
                name=name,
                description=description,
                node_type=node_type,
                domain=domain,
                strength=strength,
                meta=meta,
                spatial_info=spatial_info,
                temporal_info=temporal_info
            )
            
            if self.entity_extractor:
                full_text = f"{name} {description or ''}"
                ambiguous_entities = self.entity_extractor.extract_ambiguous_terms(full_text)
                if ambiguous_entities:
                    if not node.meta or not isinstance(node.meta, dict):
                        node.meta = {}
                    node.meta['ambiguities'] = [
                        {"text": e.text, "type": e.ambiguity_type.value if hasattr(e, 'ambiguity_type') and hasattr(e.ambiguity_type, 'value') else str(e.ambiguity_type)}
                        for e in ambiguous_entities
                    ]
            
            if source:
                if not node.meta or not isinstance(node.meta, dict):
                    node.meta = {}
                if 'sources' not in node.meta:
                    node.meta['sources'] = []
                node.meta['sources'].append({
                    'source': source,
                    'timestamp': time.time(),
                    'user_id': user_id,
                    'version': node.version
                })
            
            self._save_node_to_db(node)
            
            self.nodes[node_id] = node
            self._update_indexes(node=node)
            
            self.stats["total_nodes"] += 1
            self.stats["node_creations"] += 1
            self._update_statistics(start_time, True)
            
            logger.debug(f"Добавлен узел в граф знаний: {node_id} ({name})")
            return node_id
            
        except Exception as e:
            logger.error(f"Ошибка добавления узла в граф знаний: {e}", exc_info=True)
            self._update_statistics(start_time, False)
            return ""

    def add_concept(self, concept: str, description: str,
                    domain: str = "general", strength: float = 1.0,
                    source: Optional[str] = None, tags: Optional[list] = None,
                    user_id: Optional[str] = None,
                    spatial_info: Optional[Dict] = None,
                    temporal_info: Optional[Dict] = None) -> str:
        """Обертка для обратной совместимости: добавляет концепт как узел типа 'concept'."""
        meta = {"source": source, "tags": (tags or []), "user_id": user_id}
        return self.add_node(
            name=concept,
            description=description,
            node_type="concept",
            domain=domain,
            strength=strength,
            meta=meta,
            spatial_info=spatial_info,
            temporal_info=temporal_info,
            user_id=user_id,
            source=source,
        )
    
    def _save_node_to_db(self, node) -> bool:
        """Сохраняет узел в базу данных с использованием транзакций."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.cursor()
            
            cursor.execute("""
            INSERT OR REPLACE INTO nodes 
            (id, name, description, node_type, domain, strength, timestamp, last_updated, 
            version, meta, spatial_info, temporal_info, history, contradictions, keyword_index, concept_index)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                node.id,
                node.name,
                node.description,
                node.node_type,
                node.domain,
                node.strength,
                node.timestamp,
                node.last_updated,
                node.version,
                json.dumps(node.meta),
                json.dumps(node.spatial_info),
                json.dumps(node.temporal_info),
                json.dumps(node.history),
                json.dumps(node.contradictions),
                json.dumps(node.keyword_index),
                json.dumps(node.concept_index)
            ))
            
            conn.commit()
            return True
            
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Ошибка сохранения узла {node.id} в БД: {e}", exc_info=True)
            return False
        finally:
            if conn:
                conn.close()
    
    def _update_node_in_db(self, node) -> bool:
        """Обновляет узел в базе данных с использованием транзакций."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.cursor()
            
            cursor.execute("""
            UPDATE nodes SET
                name = ?,
                description = ?,
                node_type = ?,
                domain = ?,
                strength = ?,
                last_updated = ?,
                version = ?,
                meta = ?,
                spatial_info = ?,
                temporal_info = ?,
                history = ?,
                contradictions = ?,
                keyword_index = ?,
                concept_index = ?
            WHERE id = ?
            """, (
                node.name,
                node.description,
                node.node_type,
                node.domain,
                node.strength,
                node.last_updated,
                node.version,
                json.dumps(node.meta),
                json.dumps(node.spatial_info),
                json.dumps(node.temporal_info),
                json.dumps(node.history),
                json.dumps(node.contradictions),
                json.dumps(node.keyword_index),
                json.dumps(node.concept_index),
                node.id
            ))
            
            conn.commit()
            return True
            
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Ошибка обновления узла {node.id} в БД: {e}", exc_info=True)
            return False
        finally:
            if conn:
                conn.close()
    
    def _remove_node_from_db(self, node_id: str) -> bool:
        """Удаляет узел из базы данных с использованием транзакций."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.cursor()
            
            with conn:
                cursor.execute("DELETE FROM edges WHERE source_id = ? OR target_id = ?", (node_id, node_id))
                cursor.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка удаления узла {node_id} из БД: {e}", exc_info=True)
            return False
        finally:
            if conn:
                conn.close()
