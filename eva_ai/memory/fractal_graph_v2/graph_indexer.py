"""
Graph Indexer - Быстрый поиск в графе через HNSW индекс

Обеспечивает O(log n) поиск вместо O(n) для семантического поиска.
"""
import os
import sqlite3
import json
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set

logger = logging.getLogger("eva_ai.graph_indexer")

class GraphIndexer:
    """
    Индексатор графа с HNSW для быстрого семантического поиска.
    Работает поверх SQLite - не загружает весь граф в память.
    Поддерживает инкрементальное обновление индекса.
    """
    
    def __init__(self, db_path: str, embedding_dim: int = 768):
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self._hnsw_index = None
        self._index_built = False
        self._vector_count = 0
        self._indexed_node_ids: Set[str] = set()
        
    def _get_connection(self):
        return sqlite3.connect(self.db_path)
    
    def build_index(self, limit: int = None, force: bool = False) -> bool:
        """Построить HNSW индекс из базы данных.
        
        Args:
            limit: Лимит узлов (None = все узлы)
            force: Принудительная перестройка индекса
        """
        if self._index_built and not force:
            if self._vector_count > 0:
                logger.info(f"HNSW индекс уже построен: {self._vector_count} векторов (пропуск)")
                return True
        
        try:
            from .optimizations import create_hnsw_index
            self._hnsw_index = create_hnsw_index(dim=self.embedding_dim)
        except ImportError as e:
            logger.error(f"HNSW import failed: {e}")
            self._hnsw_index = None
            return False
        
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        
        if limit:
            query = "SELECT id, embedding FROM nodes WHERE embedding IS NOT NULL LIMIT ?"
            params = (limit,)
        else:
            query = "SELECT id, embedding FROM nodes WHERE embedding IS NOT NULL"
            params = ()
        
        cursor = conn.execute(query, params)
        
        ids = []
        vectors = []
        indexed = set()
        
        for row in cursor:
            try:
                emb_data = row['embedding']
                if isinstance(emb_data, bytes):
                    import numpy as np
                    emb = np.frombuffer(emb_data, dtype=np.float32)
                    if len(emb) == self.embedding_dim:
                        ids.append(row['id'])
                        vectors.append(emb.tolist())
                        indexed.add(row['id'])
                elif isinstance(emb_data, str) and emb_data != 'null':
                    emb = json.loads(emb_data)
                    if emb and len(emb) == self.embedding_dim:
                        ids.append(row['id'])
                        vectors.append(emb)
                        indexed.add(row['id'])
            except Exception as e:
                logger.debug(f"Failed to parse embedding for {row['id']}: {e}")
        
        conn.close()
        
        if ids and self._hnsw_index:
            try:
                self._hnsw_index.add_items(ids, vectors)
                self._vector_count = len(ids)
                self._indexed_node_ids.update(indexed)
                self._index_built = True
                logger.info(f"HNSW индекс построен: {self._vector_count} векторов")
                return True
            except Exception as e:
                logger.error(f"HNSW add_items failed: {e}")
                return False
        
        logger.warning(f"GraphIndexer: No valid embeddings found (checked nodes)")
        return False
    
    def add_nodes(self, node_ids: List[str]) -> int:
        """
        Добавить новые узлы в существующий индекс (инкрементальное обновление).
        
        Args:
            node_ids: Список ID узлов для добавления
            
        Returns:
            Количество добавленных векторов
        """
        if not node_ids:
            return 0
        
        new_ids = [nid for nid in node_ids if nid not in self._indexed_node_ids]
        
        if not new_ids:
            return 0
        
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        
        placeholders = ','.join(['?'] * len(new_ids))
        cursor = conn.execute(
            f"SELECT id, embedding FROM nodes WHERE id IN ({placeholders}) AND embedding IS NOT NULL",
            new_ids
        )
        
        ids = []
        vectors = []
        
        for row in cursor:
            try:
                emb_data = row['embedding']
                if isinstance(emb_data, bytes):
                    import numpy as np
                    emb = np.frombuffer(emb_data, dtype=np.float32)
                    if len(emb) == self.embedding_dim:
                        ids.append(row['id'])
                        vectors.append(emb.tolist())
                elif isinstance(emb_data, str) and emb_data != 'null':
                    emb = json.loads(emb_data)
                    if emb and len(emb) == self.embedding_dim:
                        ids.append(row['id'])
                        vectors.append(emb)
            except Exception as e:
                logger.debug(f"Failed to add node {row['id']}: {e}")
        
        conn.close()
        
        if ids and self._hnsw_index:
            try:
                self._hnsw_index.add_items(ids, vectors)
                self._indexed_node_ids.update(ids)
                self._vector_count = len(self._indexed_node_ids)
                logger.info(f"HNSW: added {len(ids)} nodes incrementally (total: {self._vector_count})")
                return len(ids)
            except Exception as e:
                logger.error(f"HNSW incremental add failed: {e}")
                return 0
        
        return 0
    
    def get_subgraph_embeddings(self, node_ids: List[str]) -> np.ndarray:
        """
        Получить эмбеддинги для списка узлов.
        
        Returns:
            np.ndarray [num_nodes, embedding_dim] - эмбеддинги узлов
        """
        if not node_ids:
            return np.array([]).reshape(0, self.embedding_dim)
        
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        
        embeddings = []
        valid_ids = []
        
        placeholders = ','.join(['?'] * len(node_ids))
        cursor = conn.execute(
            f"SELECT id, embedding FROM nodes WHERE id IN ({placeholders}) AND embedding IS NOT NULL",
            node_ids
        )
        
        for row in cursor:
            emb_data = row['embedding']
            if isinstance(emb_data, bytes):
                emb = np.frombuffer(emb_data, dtype=np.float32)
                if len(emb) == self.embedding_dim:
                    embeddings.append(emb)
                    valid_ids.append(row['id'])
            elif isinstance(emb_data, str) and emb_data != 'null':
                try:
                    emb = json.loads(emb_data)
                    if emb and len(emb) == self.embedding_dim:
                        embeddings.append(np.array(emb, dtype=np.float32))
                        valid_ids.append(row['id'])
                except Exception:
                    pass
        
        conn.close()
        
        if embeddings:
            return np.array(embeddings, dtype=np.float32)
        return np.array([]).reshape(0, self.embedding_dim)
    
    def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 10,
        min_similarity: float = 0.5,
        include_embeddings: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Семантический поиск - загружает только релевантные узлы.
        
        Args:
            query_embedding: запрос в виде эмбеддинга
            top_k: количество результатов
            min_similarity: минимальная похожесть (0.0 - 1.0 для нормализованной)
            include_embeddings: включать ли эмбеддинги в результат
        """
        results = []
        node_ids = []
        
        # 1. Пробуем HNSW
        if self._hnsw_index and self._index_built:
            try:
                hnsw_results = self._hnsw_index.search(query_embedding, k=top_k * 2)
                for node_id, similarity in hnsw_results:
                    if similarity >= min_similarity:
                        node_ids.append(node_id)
                        results.append({
                            "id": node_id,
                            "similarity": float(similarity),
                            "type": "hnsw"
                        })
                if results:
                    results = results[:top_k]
            except Exception as e:
                logger.debug(f"HNSW failed: {e}")
        
        # 2. Fallback: SQL поиск с векторным вычислением если HNSW не дал результатов
        if not results:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            
            query_str = str(query_embedding)[:50]
            words = [w for w in query_str.split() if len(w) > 3][:3]
            
            if not words:
                words = ['knowledge', 'concept']
            
            for word in words:
                cursor = conn.execute(
                    """SELECT id, content, node_type, level, confidence, embedding 
                       FROM nodes 
                       WHERE content LIKE ? AND embedding IS NOT NULL
                       LIMIT ?""",
                    (f"%{word}%", top_k)
                )
                
                for row in cursor:
                    try:
                        emb = json.loads(row['embedding']) if row['embedding'] else None
                        if emb:
                            q = np.array(query_embedding, dtype=np.float32)
                            e = np.array(emb, dtype=np.float32)
                            q = q / (np.linalg.norm(q) + 1e-8)
                            e = e / (np.linalg.norm(e) + 1e-8)
                            sim = float(np.dot(q, e))
                            
                            if sim >= min_similarity:
                                results.append({
                                    "id": row['id'],
                                    "content": row['content'],
                                    "type": row['node_type'],
                                    "level": row['level'],
                                    "confidence": row['confidence'],
                                    "similarity": sim,
                                    "embedding": emb
                                })
                    except Exception:
                        pass
                    
                    conn.close()
            results.sort(key=lambda x: x['similarity'], reverse=True)
            results = results[:top_k]
        
        # 3. Загрузить эмбеддинги для результатов если нужно
        if include_embeddings and results:
            result_ids = [r['id'] for r in results]
            embeddings = self.get_subgraph_embeddings(result_ids)
            
            # Сопоставить эмбеддинги с результатами
            emb_map = {}
            if len(embeddings) > 0:
                for i, node_id in enumerate(result_ids[:len(embeddings)]):
                    emb_map[node_id] = embeddings[i]
            
            for r in results:
                if r['id'] in emb_map:
                    r['embedding'] = emb_map[r['id']]
        
        return results
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Получить конкретный узел по ID."""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        
        cursor = conn.execute(
            "SELECT * FROM nodes WHERE id = ?",
            (node_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "id": row['id'],
                "content": row['content'],
                "type": row['node_type'],
                "level": row['level'],
                "confidence": row['confidence'],
                "metadata": json.loads(row['metadata']) if row['metadata'] else {},
                "embedding": json.loads(row['embedding']) if row['embedding'] else None
            }
        return None
    
    def sync_from_db(self) -> int:
        """
        Синхронизировать _indexed_node_ids с реальным состоянием БД.
        Полезно при инициализации, если в БД появились новые узлы.
        
        Returns:
            Количество новых (неиндексированных) узлов
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        
        if self._indexed_node_ids:
            placeholders = ','.join(['?'] * len(self._indexed_node_ids))
            cursor = conn.execute(
                f"SELECT id FROM nodes WHERE id IN ({placeholders}) AND embedding IS NOT NULL",
                list(self._indexed_node_ids)
            )
            existing = set(row['id'] for row in cursor)
            removed = self._indexed_node_ids - existing
            if removed:
                logger.info(f"HNSW: {len(removed)} indexed nodes no longer exist in DB")
                self._indexed_node_ids = existing
        
        cursor = conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE embedding IS NOT NULL"
        )
        total_with_emb = cursor.fetchone()[0]
        conn.close()
        
        new_count = total_with_emb - len(self._indexed_node_ids)
        logger.info(f"HNSW sync: {len(self._indexed_node_ids)} already indexed, {new_count} new nodes available")
        return max(0, new_count)
    
    def __len__(self):
        """Return number of vectors in HNSW index if built, else 0."""
        if self._index_built and self._hnsw_index:
            try:
                if hasattr(self._hnsw_index, 'get_current_count'):
                    return self._hnsw_index.get_current_count()
                else:
                    return self._vector_count
            except Exception:
                return 0
        return 0
 
 
def create_indexer(db_path: str) -> GraphIndexer:
    """Фабричная функция."""
    return GraphIndexer(db_path)