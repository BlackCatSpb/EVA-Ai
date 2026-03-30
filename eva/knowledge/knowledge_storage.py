"""
Модуль хранения графа знаний для ЕВА
Содержит функции для работы с SQLite базой данных
"""
import os
import sqlite3
import json
import time
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger("eva.knowledge_storage")

from .knowledge_nodes import KnowledgeNode, KnowledgeEdge


def safe_json_loads(value):
    """Безопасная загрузка JSON с обработкой ошибок."""
    if not value:
        return {}
    try:
        if isinstance(value, str):
            return json.loads(value)
        elif isinstance(value, (bytes, bytearray)):
            return json.loads(value.decode('utf-8'))
        else:
            return {}
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        return {}


def safe_json_dumps(value):
    """Безопасная сериализация в JSON."""
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


class KnowledgeStorage:
    """Класс для хранения графа знаний в SQLite."""
    
    def __init__(self, db_path: str):
        """
        Инициализирует хранилище.
        
        Args:
            db_path: Путь к базе данных SQLite
        """
        self.db_path = db_path
        self.initialized = False
        self._initialize_database()
    
    def _initialize_database(self):
        """Инициализирует структуру базы данных."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Создаем таблицу узлов
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    node_type TEXT,
                    domain TEXT,
                    strength REAL,
                    timestamp REAL,
                    last_updated REAL,
                    version INTEGER,
                    meta TEXT,
                    spatial_info TEXT,
                    temporal_info TEXT,
                    history TEXT,
                    contradictions TEXT,
                    keyword_index TEXT,
                    concept_index TEXT
                )
            """)
            
            # Создаем таблицу связей
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation_type TEXT,
                    strength REAL,
                    timestamp REAL,
                    last_updated REAL,
                    version INTEGER,
                    meta TEXT,
                    spatial_info TEXT,
                    temporal_info TEXT,
                    history TEXT,
                    contradictions TEXT,
                    FOREIGN KEY (source_id) REFERENCES nodes(id),
                    FOREIGN KEY (target_id) REFERENCES nodes(id)
                )
            """)
            
            # Создаем таблицу истории
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT,
                    edge_id TEXT,
                    change_type TEXT,
                    changes TEXT,
                    timestamp REAL,
                    user_id TEXT,
                    source TEXT,
                    version INTEGER,
                    FOREIGN KEY (node_id) REFERENCES nodes(id),
                    FOREIGN KEY (edge_id) REFERENCES edges(id)
                )
            """)
            
            # Создаем индексы
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_domain ON nodes(domain)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_strength ON nodes(strength)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_timestamp ON nodes(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(relation_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_timestamp ON history(timestamp)")
            
            conn.commit()
            conn.close()
            
            self.initialized = True
            logger.info(f"База знаний инициализирована: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации базы данных: {e}", exc_info=True)
            raise
    
    def save_node(self, node: KnowledgeNode) -> bool:
        """
        Сохраняет узел в базу данных.
        
        Args:
            node: Узел для сохранения
            
        Returns:
            bool: Успешность сохранения
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO nodes 
                (id, name, description, node_type, domain, strength, timestamp, last_updated, 
                 version, meta, spatial_info, temporal_info, history, contradictions, 
                 keyword_index, concept_index)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                node.id,
                node.name,
                node.description,
                node.node_type,
                node.domain,
                node.strength,
                node.timestamp,
                node.last_updated,
                node.version,
                safe_json_dumps(node.meta),
                safe_json_dumps(node.spatial_info),
                safe_json_dumps(node.temporal_info),
                safe_json_dumps(node.history),
                safe_json_dumps(node.contradictions),
                safe_json_dumps(node.keyword_index),
                safe_json_dumps(node.concept_index)
            ])
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка сохранения узла {node.id}: {e}", exc_info=True)
            return False
    
    def save_nodes(self, nodes: Dict[str, KnowledgeNode]) -> bool:
        """
        Сохраняет множество узлов в базу данных.
        
        Args:
            nodes: Словарь узлов для сохранения
            
        Returns:
            bool: Успешность сохранения
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for node in nodes.values():
                cursor.execute("""
                    INSERT OR REPLACE INTO nodes 
                    (id, name, description, node_type, domain, strength, timestamp, last_updated, 
                     version, meta, spatial_info, temporal_info, history, contradictions, 
                     keyword_index, concept_index)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    node.id,
                    node.name,
                    node.description,
                    node.node_type,
                    node.domain,
                    node.strength,
                    node.timestamp,
                    node.last_updated,
                    node.version,
                    safe_json_dumps(node.meta),
                    safe_json_dumps(node.spatial_info),
                    safe_json_dumps(node.temporal_info),
                    safe_json_dumps(node.history),
                    safe_json_dumps(node.contradictions),
                    safe_json_dumps(node.keyword_index),
                    safe_json_dumps(node.concept_index)
                ])
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Сохранено {len(nodes)} узлов")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка сохранения узлов: {e}", exc_info=True)
            return False
    
    def load_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """
        Загружает узел из базы данных.
        
        Args:
            node_id: ID узла
            
        Returns:
            Optional[KnowledgeNode]: Загруженный узел или None
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, name, description, node_type, domain, strength, timestamp, last_updated, 
                       version, meta, spatial_info, temporal_info, history, contradictions, 
                       keyword_index, concept_index
                FROM nodes WHERE id = ?
            """, [node_id])
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
            
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
            
            return node
            
        except Exception as e:
            logger.error(f"Ошибка загрузки узла {node_id}: {e}", exc_info=True)
            return None
    
    def save_edge(self, edge: KnowledgeEdge) -> bool:
        """
        Сохраняет связь в базу данных.
        
        Args:
            edge: Связь для сохранения
            
        Returns:
            bool: Успешность сохранения
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO edges 
                (id, source_id, target_id, relation_type, strength, timestamp, last_updated, 
                 version, meta, spatial_info, temporal_info, history, contradictions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                edge.id,
                edge.source_id,
                edge.target_id,
                edge.relation_type,
                edge.strength,
                edge.timestamp,
                edge.last_updated,
                edge.version,
                safe_json_dumps(edge.meta),
                safe_json_dumps(edge.spatial_info),
                safe_json_dumps(edge.temporal_info),
                safe_json_dumps(edge.history),
                safe_json_dumps(edge.contradictions)
            ])
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка сохранения связи {edge.id}: {e}", exc_info=True)
            return False
    
    def save_edges(self, edges: Dict[str, KnowledgeEdge]) -> bool:
        """
        Сохраняет множество связей в базу данных.
        
        Args:
            edges: Словарь связей для сохранения
            
        Returns:
            bool: Успешность сохранения
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for edge in edges.values():
                cursor.execute("""
                    INSERT OR REPLACE INTO edges 
                    (id, source_id, target_id, relation_type, strength, timestamp, last_updated, 
                     version, meta, spatial_info, temporal_info, history, contradictions)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    edge.id,
                    edge.source_id,
                    edge.target_id,
                    edge.relation_type,
                    edge.strength,
                    edge.timestamp,
                    edge.last_updated,
                    edge.version,
                    safe_json_dumps(edge.meta),
                    safe_json_dumps(edge.spatial_info),
                    safe_json_dumps(edge.temporal_info),
                    safe_json_dumps(edge.history),
                    safe_json_dumps(edge.contradictions)
                ])
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Сохранено {len(edges)} связей")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка сохранения связей: {e}", exc_info=True)
            return False
    
    def load_edge(self, edge_id: str) -> Optional[KnowledgeEdge]:
        """
        Загружает связь из базы данных.
        
        Args:
            edge_id: ID связи
            
        Returns:
            Optional[KnowledgeEdge]: Загруженная связь или None
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, source_id, target_id, relation_type, strength, timestamp, last_updated, 
                       version, meta, spatial_info, temporal_info, history, contradictions
                FROM edges WHERE id = ?
            """, [edge_id])
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
            
            edge = KnowledgeEdge(
                id=row[0],
                source_id=row[1],
                target_id=row[2],
                relation_type=row[3],
                strength=row[4],
                timestamp=row[5],
                meta=safe_json_loads(row[8]) if len(row) > 8 and row[8] else {},
                version=row[7],
                spatial_info=safe_json_loads(row[9]) if len(row) > 9 and row[9] else {},
                temporal_info=safe_json_loads(row[10]) if len(row) > 10 and row[10] else {}
            )
            
            edge.last_updated = row[6]
            edge.history = safe_json_loads(row[11]) if len(row) > 11 and row[11] else []
            edge.contradictions = safe_json_loads(row[12]) if len(row) > 12 and row[12] else []
            
            return edge
            
        except Exception as e:
            logger.error(f"Ошибка загрузки связи {edge_id}: {e}", exc_info=True)
            return None
    
    def load_all_nodes(self) -> Dict[str, KnowledgeNode]:
        """
        Загружает все узлы из базы данных.
        
        Returns:
            Dict[str, KnowledgeNode]: Словарь узлов
        """
        nodes = {}
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, name, description, node_type, domain, strength, timestamp, last_updated, 
                       version, meta, spatial_info, temporal_info, history, contradictions, 
                       keyword_index, concept_index
                FROM nodes
            """)
            
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
                
                nodes[node.id] = node
            
            conn.close()
            logger.debug(f"Загружено {len(nodes)} узлов")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки узлов: {e}", exc_info=True)
        
        return nodes
    
    def load_all_edges(self) -> Dict[str, KnowledgeEdge]:
        """
        Загружает все связи из базы данных.
        
        Returns:
            Dict[str, KnowledgeEdge]: Словарь связей
        """
        edges = {}
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, source_id, target_id, relation_type, strength, timestamp, last_updated, 
                       version, meta, spatial_info, temporal_info, history, contradictions
                FROM edges
            """)
            
            for row in cursor.fetchall():
                edge = KnowledgeEdge(
                    id=row[0],
                    source_id=row[1],
                    target_id=row[2],
                    relation_type=row[3],
                    strength=row[4],
                    timestamp=row[5],
                    meta=safe_json_loads(row[8]) if len(row) > 8 and row[8] else {},
                    version=row[7],
                    spatial_info=safe_json_loads(row[9]) if len(row) > 9 and row[9] else {},
                    temporal_info=safe_json_loads(row[10]) if len(row) > 10 and row[10] else {}
                )
                
                edge.last_updated = row[6]
                edge.history = safe_json_loads(row[11]) if len(row) > 11 and row[11] else []
                edge.contradictions = safe_json_loads(row[12]) if len(row) > 12 and row[12] else []
                
                edges[edge.id] = edge
            
            conn.close()
            logger.debug(f"Загружено {len(edges)} связей")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки связей: {e}", exc_info=True)
        
        return edges
    
    def delete_node(self, node_id: str) -> bool:
        """
        Удаляет узел из базы данных.
        
        Args:
            node_id: ID узла
            
        Returns:
            bool: Успешность удаления
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Удаляем связанные записи в истории
            cursor.execute("DELETE FROM history WHERE node_id = ?", [node_id])
            
            # Удаляем узел
            cursor.execute("DELETE FROM nodes WHERE id = ?", [node_id])
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка удаления узла {node_id}: {e}", exc_info=True)
            return False
    
    def delete_edge(self, edge_id: str) -> bool:
        """
        Удаляет связь из базы данных.
        
        Args:
            edge_id: ID связи
            
        Returns:
            bool: Успешность удаления
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Удаляем связанные записи в истории
            cursor.execute("DELETE FROM history WHERE edge_id = ?", [edge_id])
            
            # Удаляем связь
            cursor.execute("DELETE FROM edges WHERE id = ?", [edge_id])
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка удаления связи {edge_id}: {e}", exc_info=True)
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Возвращает статистику базы данных.
        
        Returns:
            Dict[str, Any]: Статистика
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Количество узлов
            cursor.execute("SELECT COUNT(*) FROM nodes")
            node_count = cursor.fetchone()[0]
            
            # Количество связей
            cursor.execute("SELECT COUNT(*) FROM edges")
            edge_count = cursor.fetchone()[0]
            
            # Количество записей в истории
            cursor.execute("SELECT COUNT(*) FROM history")
            history_count = cursor.fetchone()[0]
            
            # Размер файла базы данных
            db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            
            # Домены
            cursor.execute("SELECT domain, COUNT(*) FROM nodes GROUP BY domain")
            domains = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                "node_count": node_count,
                "edge_count": edge_count,
                "history_count": history_count,
                "db_size_bytes": db_size,
                "db_size_mb": round(db_size / (1024 * 1024), 2),
                "domains": domains,
                "db_path": self.db_path
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}", exc_info=True)
            return {}
    
    def backup_database(self, backup_path: str) -> bool:
        """
        Создает резервную копию базы данных.
        
        Args:
            backup_path: Путь для резервной копии
            
        Returns:
            bool: Успешность создания копии
        """
        try:
            conn = sqlite3.connect(self.db_path)
            backup_conn = sqlite3.connect(backup_path)
            
            conn.backup(backup_conn)
            
            backup_conn.close()
            conn.close()
            
            logger.info(f"Резервная копия создана: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка создания резервной копии: {e}", exc_info=True)
            return False
    
    def optimize_database(self) -> bool:
        """
        Оптимизирует базу данных.
        
        Returns:
            bool: Успешность оптимизации
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Анализ таблиц
            cursor.execute("ANALYZE")
            
            # Перестроение индексов
            cursor.execute("REINDEX")
            
            # Вакуум
            cursor.execute("VACUUM")
            
            conn.commit()
            conn.close()
            
            logger.info("База данных оптимизирована")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка оптимизации базы данных: {e}", exc_info=True)
            return False
    
    # Note: load_nodes and load_edges methods were removed as they are duplicates 
    # of load_all_nodes and load_all_edges. Using the latter for consistency.
