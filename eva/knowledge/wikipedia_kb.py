"""
Wikipedia Knowledge Base для EVA
Загрузка, индексация и поиск статей русскоязычной Википедии.
Используется как базис знаний для рассуждений и дообучения.
"""
import os
import json
import sqlite3
import hashlib
import logging
import threading
from typing import Optional, List, Dict, Any
from datetime import datetime

logger = logging.getLogger("eva.wikipedia_kb")

class WikipediaKnowledgeBase:
    """
    База знаний на основе русскоязычной Википедии.
    
    Хранит статьи в SQLite с эмбеддингами для семантического поиска.
    """
    
    def __init__(self, data_dir: str = None, max_chunks: int = 500000):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                    'memory', 'wikipedia_kb')
        os.makedirs(data_dir, exist_ok=True)
        
        self.data_dir = data_dir
        self.db_path = os.path.join(data_dir, 'wikipedia.db')
        self.max_chunks = max_chunks
        self._lock = threading.RLock()
        self._embedder = None
        self._init_db()
        
        stats = self.get_stats()
        logger.info(f"WikipediaKnowledgeBase инициализирован: {stats['articles']} статей, "
                   f"{stats['chunks']} чанков")
    
    def _init_db(self):
        """Инициализация SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    url TEXT,
                    text TEXT,
                    chunk_count INTEGER DEFAULT 0,
                    created_at TEXT,
                    category TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    article_id TEXT NOT NULL,
                    title TEXT,
                    text TEXT NOT NULL,
                    embedding TEXT,
                    chunk_index INTEGER,
                    created_at TEXT,
                    FOREIGN KEY (article_id) REFERENCES articles(id)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_article ON chunks(article_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_title ON chunks(title)")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.commit()
    
    def _get_embedder(self):
        """Ленивая загрузка эмбеддера."""
        if self._embedder is None:
            try:
                from eva.mlearning.sentence_transformers_cache import get_sentence_transformer
                self._embedder = get_sentence_transformer('intfloat/multilingual-e5-base', device='cpu')
            except Exception as e:
                logger.warning(f"Эмбеддер недоступен: {e}")
        return self._embedder
    
    def _compute_embedding(self, text: str) -> Optional[List[float]]:
        """Вычисляет эмбеддинг текста."""
        embedder = self._get_embedder()
        if embedder is None or not text.strip():
            return None
        try:
            emb = embedder.encode([text.strip()])[0]
            return emb.tolist() if hasattr(emb, 'tolist') else list(emb)
        except Exception as e:
            logger.debug(f"Ошибка эмбеддинга: {e}")
            return None
    
    def add_article(self, title: str, text: str, url: str = None, category: str = None, 
                    chunk_size: int = 500) -> str:
        """
        Добавляет статью в базу знаний.
        
        Args:
            title: Заголовок статьи
            text: Полный текст статьи
            url: URL статьи
            category: Категория
            chunk_size: Размер чанка в символах
        
        Returns:
            ID статьи
        """
        if not text.strip():
            return ""
        
        article_id = hashlib.sha256(title.encode('utf-8')).hexdigest()[:16]
        now = datetime.now().isoformat()
        
        # Чанкинг текста
        chunks = self._chunk_text(text, chunk_size)
        
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                # Проверяем, есть ли уже статья
                cursor = conn.execute("SELECT id FROM articles WHERE id = ?", (article_id,))
                if cursor.fetchone():
                    return article_id  # Уже существует
                
                # Сохраняем статью
                conn.execute(
                    "INSERT OR IGNORE INTO articles (id, title, url, text, chunk_count, created_at, category) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (article_id, title, url, text[:10000], len(chunks), now, category)
                )
                
                # Сохраняем чанки
                for i, chunk_text in enumerate(chunks):
                    chunk_id = f"{article_id}_{i}"
                    embedding = self._compute_embedding(chunk_text)
                    emb_json = json.dumps(embedding) if embedding else None
                    
                    conn.execute(
                        "INSERT OR IGNORE INTO chunks (id, article_id, title, text, embedding, chunk_index, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (chunk_id, article_id, title, chunk_text, emb_json, i, now)
                    )
                
                conn.commit()
        
        logger.debug(f"Добавлена статья: {title} ({len(chunks)} чанков)")
        return article_id
    
    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Разбивает текст на чанки по предложениям."""
        import re
        
        # Разбиваем по предложениям
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(current_chunk) + len(sentence) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def search(self, query: str, limit: int = 5, min_similarity: float = 0.3) -> List[Dict[str, Any]]:
        """
        Семантический поиск по статьям Википедии.
        
        Args:
            query: Поисковый запрос
            limit: Максимальное количество результатов
            min_similarity: Минимальная схожесть
        
        Returns:
            Список релевантных чанков с метаданными
        """
        query_embedding = self._compute_embedding(query)
        if query_embedding is None:
            return []
        
        import math
        
        results = []
        
        with sqlite3.connect(self.db_path) as conn:
            # Загружаем все чанки с эмбеддингами (оптимизация: можно использовать FAISS для больших баз)
            cursor = conn.execute("SELECT id, article_id, title, text, embedding FROM chunks WHERE embedding IS NOT NULL")
            
            for row in cursor:
                chunk_id, article_id, title, text, emb_json = row
                if not emb_json:
                    continue
                
                stored_embedding = json.loads(emb_json)
                
                # Cosine similarity
                dot = sum(a * b for a, b in zip(query_embedding, stored_embedding))
                norm_q = math.sqrt(sum(a * a for a in query_embedding))
                norm_s = math.sqrt(sum(a * a for a in stored_embedding))
                
                if norm_q == 0 or norm_s == 0:
                    continue
                
                similarity = dot / (norm_q * norm_s)
                
                if similarity >= min_similarity:
                    results.append({
                        'chunk_id': chunk_id,
                        'article_id': article_id,
                        'title': title,
                        'text': text,
                        'similarity': round(similarity, 4),
                    })
                
                if len(results) >= limit * 3:  # Загружаем больше для сортировки
                    break
        
        # Сортируем по схожести и берём top-N
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:limit]
    
    def get_article(self, article_id: str) -> Optional[Dict[str, Any]]:
        """Получает статью по ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT id, title, url, text, chunk_count, category FROM articles WHERE id = ?", 
                                 (article_id,))
            row = cursor.fetchone()
            if row:
                return {
                    'id': row[0],
                    'title': row[1],
                    'url': row[2],
                    'text': row[3],
                    'chunk_count': row[4],
                    'category': row[5],
                }
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика базы знаний."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM articles")
            articles = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM chunks")
            chunks = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
            embedded = cursor.fetchone()[0]
            
            return {
                'articles': articles,
                'chunks': chunks,
                'embedded_chunks': embedded,
                'db_size_mb': round(os.path.getsize(self.db_path) / (1024 * 1024), 2) if os.path.exists(self.db_path) else 0,
            }
    
    def clear(self):
        """Очищает базу знаний."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM chunks")
                conn.execute("DELETE FROM articles")
                conn.commit()
            logger.info("WikipediaKnowledgeBase очищена")


# Singleton
_wikipedia_kb = None
_kb_lock = threading.Lock()


def get_wikipedia_kb(data_dir: str = None) -> WikipediaKnowledgeBase:
    """Возвращает singleton базу знаний Википедии."""
    global _wikipedia_kb
    with _kb_lock:
        if _wikipedia_kb is None:
            _wikipedia_kb = WikipediaKnowledgeBase(data_dir=data_dir)
        return _wikipedia_kb


def clear_wikipedia_kb():
    """Очищает базу знаний."""
    global _wikipedia_kb
    with _kb_lock:
        if _wikipedia_kb:
            _wikipedia_kb.clear()
            _wikipedia_kb = None
