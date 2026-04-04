"""Storage management, DB operations, and persistence for EVA knowledge graph."""
import os
import logging
import time
import sqlite3
import json
from typing import Dict, List, Optional, Any

logger = logging.getLogger("eva.knowledge.core_storage")


class KnowledgeStorageManager:
    """Manages SQLite database storage for knowledge graph nodes and edges."""

    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.db = None

    def init_database(self) -> sqlite3.Connection:
        """Инициализирует базу данных для хранения графа знаний."""
        try:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()

            cursor.execute('''
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                node_type TEXT NOT NULL,
                domain TEXT NOT NULL,
                strength REAL NOT NULL,
                timestamp REAL NOT NULL,
                meta TEXT
            )
            ''')

            cursor.execute('''
            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                relation TEXT NOT NULL,
                strength REAL NOT NULL,
                timestamp REAL NOT NULL,
                meta TEXT,
                FOREIGN KEY (source) REFERENCES nodes (id),
                FOREIGN KEY (target) REFERENCES nodes (id)
            )
            ''')

            cursor.execute('CREATE INDEX IF NOT EXISTS idx_nodes_domain ON nodes (domain)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_edges_source ON edges (source)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_edges_target ON edges (target)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges (relation)')

            conn.commit()
            logger.debug("Структура базы данных графа знаний инициализирована")
            self.db = conn
            return conn
        except Exception as e:
            logger.error(f"Ошибка инициализации базы данных графа знаний: {e}")
            raise

    def load_nodes(self, nodes_dict: dict, domains: dict, contexts: dict):
        """Загружает узлы из хранилища в переданные словари."""
        try:
            cursor = self.db.cursor()
            cursor.execute("SELECT id, content, node_type, domain, strength, timestamp, meta FROM nodes")
            for row in cursor.fetchall():
                node_id, content, node_type, domain, strength, timestamp, meta = row
                node_data = {
                    "id": node_id,
                    "content": json.loads(content),
                    "node_type": node_type,
                    "domain": domain,
                    "strength": strength,
                    "timestamp": timestamp,
                    "meta": json.loads(meta) if meta else {}
                }
                nodes_dict[node_id] = node_data
                domains[domain].append(node_id)
                ctx = node_data["meta"].get("context", "")
                if ctx:
                    contexts[ctx].append(node_id)
            logger.info(f"Загружено {len(nodes_dict)} узлов из хранилища")
        except Exception as e:
            logger.error(f"Ошибка загрузки узлов графа знаний: {e}")

    def load_edges(self, edges_dict: dict, node_edges: dict):
        """Загружает связи из хранилища в переданные словари."""
        try:
            cursor = self.db.cursor()
            cursor.execute("SELECT id, source, target, relation, strength, timestamp, meta FROM edges")
            for row in cursor.fetchall():
                edge_id, source, target, relation, strength, timestamp, meta = row
                edge_data = {
                    "id": edge_id,
                    "source": source,
                    "target": target,
                    "relation": relation,
                    "strength": strength,
                    "timestamp": timestamp,
                    "meta": json.loads(meta) if meta else {}
                }
                edges_dict[edge_id] = edge_data
                node_edges[source].append(edge_id)
                node_edges[target].append(edge_id)
            logger.info(f"Загружено {len(edges_dict)} связей из хранилища")
        except Exception as e:
            logger.error(f"Ошибка загрузки связей графа знаний: {e}")

    def insert_node(self, node) -> bool:
        """Вставляет узел в базу данных."""
        try:
            cursor = self.db.cursor()
            cursor.execute('''
            INSERT INTO nodes (id, content, node_type, domain, strength, timestamp, meta)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                node.id,
                json.dumps(node.content),
                node.node_type,
                node.domain,
                node.strength,
                node.timestamp,
                json.dumps(node.meta)
            ))
            self.db.commit()
            return True
        except Exception as e:
            logger.error(f"Ошибка добавления узла {node.id} в БД: {e}")
            return False

    def insert_edge(self, edge) -> bool:
        """Вставляет связь в базу данных."""
        try:
            cursor = self.db.cursor()
            cursor.execute('''
            INSERT INTO edges (id, source, target, relation, strength, timestamp, meta)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                edge.id,
                edge.source,
                edge.target,
                edge.relation,
                edge.strength,
                edge.timestamp,
                json.dumps(edge.meta)
            ))
            self.db.commit()
            return True
        except Exception as e:
            logger.error(f"Ошибка добавления связи {edge.id} в БД: {e}")
            return False

    def delete_node(self, node_id: str) -> bool:
        """Удаляет узел и все его связи из базы данных."""
        try:
            cursor = self.db.cursor()
            cursor.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
            cursor.execute("DELETE FROM edges WHERE source = ? OR target = ?", (node_id, node_id))
            self.db.commit()
            return True
        except Exception as e:
            logger.error(f"Ошибка удаления узла {node_id} из БД: {e}")
            return False

    def delete_edge(self, edge_id: str) -> bool:
        """Удаляет связь из базы данных."""
        try:
            cursor = self.db.cursor()
            cursor.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
            self.db.commit()
            return True
        except Exception as e:
            logger.error(f"Ошибка удаления связи {edge_id} из БД: {e}")
            return False

    def update_node(self, node) -> bool:
        """Обновляет узел в базе данных."""
        try:
            cursor = self.db.cursor()
            cursor.execute('''
            UPDATE nodes
            SET content = ?, node_type = ?, domain = ?, strength = ?, timestamp = ?, meta = ?
            WHERE id = ?
            ''', (
                json.dumps(node.content),
                node.node_type,
                node.domain,
                node.strength,
                node.timestamp,
                json.dumps(node.meta),
                node.id
            ))
            self.db.commit()
            return True
        except Exception as e:
            logger.error(f"Ошибка обновления узла {node.id} в БД: {e}")
            return False

    def close(self):
        """Закрывает соединение с базой данных."""
        if self.db:
            self.db.close()
            logger.info("Соединение с базой данных графа знаний закрыто")
            self.db = None
