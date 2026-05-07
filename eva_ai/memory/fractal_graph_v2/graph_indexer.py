"""
Graph Indexer - Быстрый поиск в графе через HNSW индекс

Обеспечивает O(log n) поиск вместо O(n) для семантического поиска.
"""
import os
import sqlite3
import json
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger("eva_ai.graph_indexer")

class GraphIndexer:
    """
    Индексатор графа с HNSW для быстрого семантического поиска.
    Работает поверх SQLite - не загружает весь граф в память.
    """
    
    def __init__(self, db_path: str, embedding_dim: int = 768):
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self._hnsw_index = None
        self._index_built = False
        self._vector_count = 0
        
    def _get_connection(self):
        return sqlite3.connect(self.db_path)
    
    def build_index(self, limit: int = 50000) -> bool:
        """Построить HNSW индекс из базы данных."""
        try:
            from .optimizations import create_hnsw_index
            self._hnsw_index = create_hnsw_index(dim=self.embedding_dim)
        except ImportError as e:
            logger.error(f"HNSW import failed: {e}")
            self._hnsw_index = None
            return False
        
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        
        cursor = conn.execute(
            "SELECT id, embedding FROM nodes WHERE embedding IS NOT NULL LIMIT ?",
            (limit,)
        )
        
        ids = []
        vectors = []
        
        for row in cursor:
            try:
                emb_data = row['embedding']
                if emb_data:
                    # embeddings stored as BLOB (numpy bytes) or JSON string
                    if isinstance(emb_data, bytes):
                        # BLOB - deserialize numpy array
                        import numpy as np
                        emb = np.frombuffer(emb_data, dtype=np.float32)
                        if len(emb) == self.embedding_dim:
                            ids.append(row['id'])
                            vectors.append(emb.tolist())
                    elif isinstance(emb_data, str) and emb_data != 'null':
                        # JSON string fallback
                        emb = json.loads(emb_data)
                        if emb and len(emb) == self.embedding_dim:
                            ids.append(row['id'])
                            vectors.append(emb)
            except Exception as e:
                logger.debug(f"Failed to parse embedding for {row['id']}: {e}")
                pass
        
        conn.close()
        
        if ids and self._hnsw_index:
            try:
                self._hnsw_index.add_items(ids, vectors)
                self._vector_count = len(ids)
                self._index_built = True
                logger.info(f"HNSW индекс построен: {self._vector_count} векторов")
                return True
            except Exception as e:
                logger.error(f"HNSW add_items failed: {e}")
                return False
        
        logger.warning(f"GraphIndexer: No valid embeddings found (checked nodes)")
        return False
    
    def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 10,
        min_similarity: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Семантический поиск - загружает только релевантные узлы.
        """
        results = []
        
        # 1. Пробуем HNSW
        if self._hnsw_index and self._index_built:
            try:
                hnsw_results = self._hnsw_index.search(query_embedding, k=top_k * 2)
                for node_id, similarity in hnsw_results:
                    if similarity >= min_similarity:
                        results.append({
                            "id": node_id,
                            "similarity": similarity,
                            "type": "hnsw"
                        })
                if results:
                    return results[:top_k]
            except Exception as e:
                logger.debug(f"HNSW failed: {e}")
        
        # 2. Fallback: SQL поиск с векторным вычислением
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        
        # Простой поиск по ключевым словам для начала
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
                        # Вычисляем косинусную схожесть
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
                except:
                    pass
        
        conn.close()
        
        # Сортируем и возвращаем
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
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
     
    def __len__(self):
        """Return number of vectors in HNSW index if built, else 0."""
        if self._index_built and self._hnsw_index:
            try:
                if hasattr(self._hnsw_index, 'get_current_count'):
                    return self._hnsw_index.get_current_count()
                else:
                    return self._vector_count
            except:
                return 0
        return 0
 
 
def create_indexer(db_path: str) -> GraphIndexer:
    """Фабричная функция."""
    return GraphIndexer(db_path)