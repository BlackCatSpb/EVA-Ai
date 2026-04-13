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

logger = logging.getLogger("eva_ai.wikipedia_kb")

# Опциональный импорт FAISS для быстрого поиска
try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
    logger.info("FAISS доступен для быстрого поиска")
except ImportError:
    FAISS_AVAILABLE = False
    logger.info("FAISS не установлен, используется fallback поиск")

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
        self._faiss_index = None
        self._faiss_ids = []  # Маппинг индекса FAISS -> chunk_id
        self._init_db()
        
        stats = self.get_stats()
        logger.info(f"WikipediaKnowledgeBase инициализирован: {stats['articles']} статей, "
                   f"{stats['chunks']} чанков")
    
    def _build_faiss_index(self):
        """Построить FAISS индекс для быстрого поиска."""
        if not FAISS_AVAILABLE:
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL")
                rows = cursor.fetchall()
            
            if not rows:
                logger.warning("Нет данных для построения FAISS индекса")
                return
            
            # Собираем эмбеддинги
            embeddings = []
            self._faiss_ids = []
            
            for chunk_id, emb_json in rows:
                if emb_json:
                    emb = json.loads(emb_json)
                    embeddings.append(emb)
                    self._faiss_ids.append(chunk_id)
            
            if not embeddings:
                return
            
            embeddings_array = np.array(embeddings, dtype=np.float32)
            dim = embeddings_array.shape[1]
            
            # Создаём индекс (IVF для больших баз, Flat для маленьких)
            if len(embeddings) > 10000:
                nlist = min(100, len(embeddings) // 10)
                quantizer = faiss.IndexFlatIP(dim)
                self._faiss_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
                self._faiss_index.train(embeddings_array)
            else:
                self._faiss_index = faiss.IndexFlatIP(dim)
            
            self._faiss_index.add(embeddings_array)
            logger.info(f"FAISS индекс построен: {len(embeddings)} векторов, dim={dim}")
            
        except Exception as e:
            logger.error(f"Ошибка построения FAISS индекса: {e}")
            self._faiss_index = None
    
    def _search_with_faiss(self, query_embedding: List[float], limit: int, min_similarity: float) -> List[Dict]:
        """Поиск с использованием FAISS."""
        if self._faiss_index is None:
            self._build_faiss_index()
        
        if self._faiss_index is None:
            return self._search_fallback(query_embedding, limit, min_similarity)
        
        try:
            query_array = np.array([query_embedding], dtype=np.float32)
            scores, indices = self._faiss_index.search(query_array, limit * 3)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._faiss_ids):
                    continue
                
                similarity = float(score)
                if similarity >= min_similarity:
                    chunk_id = self._faiss_ids[idx]
                    # Получаем данные чанка из БД
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.execute(
                            "SELECT article_id, title, text FROM chunks WHERE id = ?", 
                            (chunk_id,)
                        )
                        row = cursor.fetchone()
                        if row:
                            results.append({
                                'chunk_id': chunk_id,
                                'article_id': row[0],
                                'title': row[1],
                                'text': row[2],
                                'similarity': round(similarity, 4),
                            })
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Ошибка FAISS поиска: {e}")
            return self._search_fallback(query_embedding, limit, min_similarity)
    
    def _search_fallback(self, query_embedding: List[float], limit: int, min_similarity: float) -> List[Dict]:
        """Fallback поиск без FAISS (медленный)."""
        import math
        
        results = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT id, article_id, title, text, embedding FROM chunks WHERE embedding IS NOT NULL")
            
            for row in cursor:
                chunk_id, article_id, title, text, emb_json = row
                if not emb_json:
                    continue
                
                stored_embedding = json.loads(emb_json)
                
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
                
                if len(results) >= limit * 3:
                    break
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:limit]
    
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
                from eva_ai.mlearning.sentence_transformers_cache import get_sentence_transformer
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
        
        # Используем FAISS если доступен
        if FAISS_AVAILABLE and self._faiss_index is not None:
            return self._search_with_faiss(query_embedding, limit, min_similarity)
        
        # Fallback на медленный поиск
        return self._search_fallback(query_embedding, limit, min_similarity)
    
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

    def add_to_fractal_graph(self, fractal_graph, top_k: int = 100, node_type: str = "wikipedia") -> Dict[str, Any]:
        """
        Добавляет все статьи из Wikipedia KB в FractalGraph v2.

        Args:
            fractal_graph: Экземпляр FractalMemoryGraph
            top_k: Максимальное число статей для добавления
            node_type: Тип узла в графе

        Returns:
            {added_count, article_ids, errors}
        """
        if fractal_graph is None:
            return {"added_count": 0, "article_ids": [], "errors": ["fractal_graph is None"]}

        added_count = 0
        article_ids = []
        errors = []

        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT id, title, text, url, category FROM articles LIMIT ?",
                        (top_k,)
                    )
                    articles = cursor.fetchall()

            for article_id, title, text, url, category in articles:
                try:
                    if not text:
                        continue

                    node = fractal_graph.add_node(
                        content=f"{title}\n\n{text[:1000]}",
                        node_type=node_type,
                        level=1,
                        confidence=0.7,
                        metadata={
                            "title": title,
                            "url": url,
                            "category": category,
                            "source": "wikipedia"
                        }
                    )
                    article_ids.append(article_id)
                    added_count += 1

                except Exception as e:
                    errors.append(f"{article_id}: {str(e)}")
                    logger.error(f"Ошибка добавления статьи {article_id} в FG: {e}")

            logger.info(f"Добавлено {added_count} статей из Wikipedia в FractalGraph")
            return {"added_count": added_count, "article_ids": article_ids, "errors": errors}

        except Exception as e:
            logger.error(f"Ошибка экспорта Wikipedia в FractalGraph: {e}")
            return {"added_count": 0, "article_ids": [], "errors": [str(e)]}

    def search_and_add_to_graph(self, query: str, fractal_graph, top_k: int = 10) -> Dict[str, Any]:
        """
        Находит релевантные статьи и добавляет их в FractalGraph v2.

        Args:
            query: Поисковый запрос
            fractal_graph: Экземпляр FractalMemoryGraph
            top_k: Число результатов

        Returns:
            {search_results, graph_results}
        """
        search_results = self.search(query, limit=top_k)
        if not search_results:
            return {"search_results": [], "graph_results": {"added_count": 0}}

        if fractal_graph is None:
            return {"search_results": search_results, "graph_results": {"error": "fractal_graph is None"}}

        added_count = 0
        node_ids = []

        for item in search_results:
            try:
                node = fractal_graph.add_node(
                    content=item.get("text", ""),
                    node_type="wikipedia",
                    level=2,
                    confidence=item.get("similarity", 0.5),
                    metadata={
                        "title": item.get("title"),
                        "url": item.get("url"),
                        "source": "wikipedia_search",
                        "query": query
                    }
                )
                node_ids.append(node.id)
                added_count += 1
            except Exception as e:
                logger.error(f"Ошибка добавления в FG: {e}")

        return {
            "search_results": search_results,
            "graph_results": {"added_count": added_count, "node_ids": node_ids}
        }


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


class WikipediaLoader:
    """Загрузчик статей из Википедии через Wikipedia API."""
    
    def __init__(self, wikipedia_kb: WikipediaKnowledgeBase, language: str = 'ru'):
        self.kb = wikipedia_kb
        self.language = language
        self.api_url = f"https://{language}.wikipedia.org/w/api.php"
        self._session = None
    
    def _get_session(self):
        """Получить HTTP сессию."""
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                'User-Agent': 'CogniFlex-EVA/1.0 (Educational AI System)'
            })
        return self._session
    
    def search_articles(self, query: str, limit: int = 10) -> List[Dict]:
        """Поиск статей по запросу."""
        import requests
        try:
            params = {
                'action': 'query',
                'list': 'search',
                'srsearch': query,
                'srlimit': limit,
                'format': 'json'
            }
            session = self._get_session()
            response = session.get(self.api_url, params=params, timeout=10)
            data = response.json()
            
            results = []
            for item in data.get('query', {}).get('search', []):
                results.append({
                    'title': item.get('title'),
                    'pageid': item.get('pageid'),
                    'snippet': item.get('snippet', '')
                })
            return results
        except Exception as e:
            logger.warning(f"Ошибка поиска Wikipedia: {e}")
            return []
    
    def load_article(self, title: str, chunk_size: int = 500) -> str:
        """Загрузить статью по названию."""
        import requests
        try:
            params = {
                'action': 'query',
                'titles': title,
                'prop': 'extracts',
                'explaintext': True,
                'exintro': False,
                'format': 'json'
            }
            session = self._get_session()
            response = session.get(self.api_url, params=params, timeout=30)
            data = response.json()
            
            pages = data.get('query', {}).get('pages', {})
            for page_id, page_data in pages.items():
                if page_id != '-1':
                    text = page_data.get('extract', '')
                    url = f"https://{self.language}.wikipedia.org/wiki/{title.replace(' ', '_')}"
                    article_id = self.kb.add_article(title, text, url)
                    return article_id
            return None
        except Exception as e:
            logger.warning(f"Ошибка загрузки статьи {title}: {e}")
            return None
    
    def search_and_add(self, query: str, limit: int = 5) -> List[str]:
        """Поиск и добавление статей в базу."""
        articles = self.search_articles(query, limit)
        added_ids = []
        for article in articles:
            article_id = self.load_article(article['title'])
            if article_id:
                added_ids.append(article_id)
        return added_ids


def get_wikipedia_loader(wikipedia_kb: WikipediaKnowledgeBase = None, language: str = 'ru') -> WikipediaLoader:
    """Создаёт загрузчик статей из Википедии."""
    if wikipedia_kb is None:
        wikipedia_kb = get_wikipedia_kb()
    return WikipediaLoader(wikipedia_kb, language)
