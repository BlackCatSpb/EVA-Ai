"""
Fast Embedder - оптимизированный модуль эмбеддинга с кэшированием и индексацией.
"""

import os
import time
import hashlib
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from collections import OrderedDict

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Кэш для эмбеддингов."""
    
    def __init__(self, max_size: int = 5000, embedding_dim: int = 128):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.embedding_dim = embedding_dim
        self.hits = 0
        self.misses = 0
    
    def _get_key(self, tokens: List[int]) -> str:
        return hashlib.md5(bytes(tokens)).hexdigest()
    
    def get(self, tokens: List[int]) -> Optional[np.ndarray]:
        key = self._get_key(tokens)
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, tokens: List[int], embedding: np.ndarray):
        key = self._get_key(tokens)
        self.cache[key] = embedding
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def get_stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            'size': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / total * 100 if total > 0 else 0
        }


class ApproximateNearestNeighbor:
    """Приближенный поиск ближайших соседей (ANN) для быстрого lookup."""
    
    def __init__(self, embedding_dim: int = 128, num_clusters: int = 100):
        self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters
        self.centroids: np.ndarray = None
        self.embeddings: List[Tuple[str, np.ndarray]] = []
    
    def build_index(self, embeddings: Dict[str, np.ndarray]):
        """Построение индекса."""
        if not embeddings:
            return
        
        vectors = list(embeddings.values())
        self.embeddings = list(embeddings.items())
        
        # Простое квантование на кластеры (k-means упрощенный)
        self.centroids = np.random.randn(self.num_clusters, self.embedding_dim).astype(np.float32)
        
        # Несколько итераций k-means
        for _ in range(3):
            distances = np.array([
                np.linalg.norm(v - self.centroids, axis=1) 
                for v in vectors
            ])
            labels = np.argmin(distances, axis=1)
            
            for i in range(self.num_clusters):
                mask = labels == i
                if mask.any():
                    self.centroids[i] = vectors[mask].mean(axis=0)
        
        logger.info(f"ANN индекс построен: {len(embeddings)} векторов, {self.num_clusters} кластеров")
    
    def search(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Поиск ближайших соседей."""
        if not self.embeddings:
            return []
        
        # Сначала ищем в ближайшем кластере
        distances_to_centroids = np.linalg.norm(self.centroids - query, axis=1)
        nearest_cluster = np.argmin(distances_to_centroids)
        
        # Ищем среди векторов в этом кластере
        results = []
        for idx, (key, emb) in enumerate(self.embeddings):
            dist = np.linalg.norm(emb - query)
            results.append((key, float(dist)))
        
        results.sort(key=lambda x: x[1])
        return results[:top_k]


class FastEmbedder:
    """Оптимизированный эмбеддер с кэшированием и индексацией."""
    
    def __init__(self, embeddings_dir: str = None, embedding_dim: int = 128):
        self.embeddings_dir = embeddings_dir or 'tests/model_optimization/embeddings'
        self.embedding_dim = embedding_dim
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        self.cache = EmbeddingCache(max_size=5000, embedding_dim=embedding_dim)
        self.ann = ApproximateNearestNeighbor(embedding_dim=embedding_dim)
        self.embedding_index: Dict[str, np.ndarray] = {}
        
        self.use_quantization = True
        self.use_fp16 = True
        
        self._load_index()
    
    def _load_index(self):
        """Загрузка индекса эмбеддингов."""
        index_file = os.path.join(self.embeddings_dir, 'embedding_index.npy')
        keys_file = os.path.join(self.embeddings_dir, 'embedding_keys.json')
        
        if os.path.exists(index_file) and os.path.exists(keys_file):
            try:
                self.embedding_index = np.load(index_file, allow_pickle=True).item()
                with open(keys_file, 'r') as f:
                    keys = json.load(f)
                    self.embedding_index = {k: self.embedding_index[k] for k in keys if k in self.embedding_index}
                
                if len(self.embedding_index) > 0:
                    self.ann.build_index(self.embedding_index)
                    logger.info(f"Загружен индекс: {len(self.embedding_index)} эмбеддингов")
            except Exception as e:
                logger.warning(f"Не удалось загрузить индекс: {e}")
    
    def _save_index(self):
        """Сохранение индекса."""
        if not self.embedding_index:
            return
        
        try:
            index_file = os.path.join(self.embeddings_dir, 'embedding_index.npy')
            keys_file = os.path.join(self.embeddings_dir, 'embedding_keys.json')
            
            np.save(index_file, self.embedding_index)
            with open(keys_file, 'w') as f:
                json.dump(list(self.embedding_index.keys()), f)
            
            logger.info(f"Сохранён индекс: {len(self.embedding_index)} эмбеддингов")
        except Exception as e:
            logger.warning(f"Не удалось сохранить индекс: {e}")
    
    def _compute_embedding(self, tokens: List[int]) -> np.ndarray:
        """Вычисление эмбеддинга (упрощенная версия)."""
        # Упрощенная схема - в реальности здесь была бы модель
        np.random.seed(sum(tokens) % (2**32))
        embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        
        if self.use_fp16:
            embedding = embedding.astype(np.float16)
        
        # Нормализация
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def get_embedding(self, tokens: List[int]) -> np.ndarray:
        """Получение эмбеддинга с кэшированием."""
        cached = self.cache.get(tokens)
        if cached is not None:
            return cached
        
        embedding = self._compute_embedding(tokens)
        
        # Кэшируем
        self.cache.put(tokens, embedding)
        
        # Индексируем
        key = hashlib.md5(bytes(tokens)).hexdigest()
        self.embedding_index[key] = embedding
        
        return embedding
    
    def find_similar(self, tokens: List[int], top_k: int = 5) -> List[Tuple[str, float]]:
        """Поиск похожих токенов."""
        query_embedding = self.get_embedding(tokens)
        
        if self.ann.embeddings:
            return self.ann.search(query_embedding, top_k)
        
        return []
    
    def build_index_from_corpus(self, corpus: List[str], tokenizer_func=None):
        """Построение индекса из корпуса текстов."""
        logger.info(f"Построение индекса из {len(corpus)} текстов...")
        
        for i, text in enumerate(corpus):
            if tokenizer_func:
                tokens = tokenizer_func(text)
            else:
                tokens = list(range(len(text.split())))
            
            embedding = self.get_embedding(tokens)
            
            if i % 1000 == 0:
                logger.info(f"Обработано {i}/{len(corpus)}")
        
        # Перестраиваем ANN
        if len(self.embedding_index) > 0:
            self.ann.build_index(self.embedding_index)
        
        self._save_index()
        logger.info(f"Индекс построен: {len(self.embedding_index)} векторов")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Статистика кэша."""
        return self.cache.get_stats()
    
    def clear_cache(self):
        """Очистка кэша."""
        self.cache.cache.clear()
        logger.info("Кэш эмбеддингов очищен")


class HybridEmbeddingModel:
    """Гибридная модель эмбеддинга - объединяет несколько подходов."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.embedding_dim = self.config.get('embedding_dim', 128)
        
        self.fast_embedder = FastEmbedder(embedding_dim=self.embedding_dim)
        self.context_weight = self.config.get('context_weight', 0.3)
        self.static_weight = self.config.get('static_weight', 0.7)
    
    def get_hybrid_embedding(
        self, 
        tokens: List[int], 
        context: List[List[int]] = None
    ) -> np.ndarray:
        """Получение гибридного эмбеддинга."""
        # Базовый эмбеддинг
        static_emb = self.fast_embedder.get_embedding(tokens)
        
        if context and len(context) > 0:
            # Контекстный эмбеддинг
            context_embs = [self.fast_embedder.get_embedding(ctx) for ctx in context]
            context_emb = np.mean(context_embs, axis=0)
            
            # Гибридная комбинация
            hybrid = self.static_weight * static_emb + self.context_weight * context_emb
        else:
            hybrid = static_emb
        
        return hybrid
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Косинусное сходство."""
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))