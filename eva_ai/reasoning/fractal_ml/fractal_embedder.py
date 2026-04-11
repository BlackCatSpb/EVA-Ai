"""
Fractal Embedder для ЕВА Self-Reasoning
Генерация 384-мерных эмбеддингов для фрактальной адресации
"""

import hashlib
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("eva_ai.reasoning.fractal_embedder")

EMBEDDING_DIM = 384


class FractalEmbedder:
    """Генератор эмбеддингов для фрактального хранилища."""
    
    def __init__(self, embedding_dim: int = EMBEDDING_DIM, use_sentence_transformers: bool = True):
        self.embedding_dim = embedding_dim
        self.cache: Dict[str, List[float]] = {}
        self._model = None
        self._use_st = use_sentence_transformers
        self._init_model()
    
    def _init_model(self):
        """Инициализация модели sentence-transformers."""
        if self._use_st:
            try:
                from eva_ai.mlearning.sentence_transformers_cache import get_sentence_transformer
                self._model = get_sentence_transformer('all-MiniLM-L6-v2', device='cpu')
                if self._model:
                    logger.info("FractalEmbedder инициализирован с sentence-transformers (cached)")
            except ImportError:
                logger.warning("sentence-transformers недоступен, используем hash-based эмбеддинги")
                self._use_st = False
    
    def generate_embedding(self, text: str) -> List[float]:
        """Генерация 384-мерного эмбеддинга для текста."""
        if self._use_st and self._model:
            embedding = self._model.encode(text, normalize_embeddings=True)
            return embedding.tolist()[:self.embedding_dim]
        return self._generate_hash_embedding(text)
    
    def embed_text(self, text: str) -> List[float]:
        """Получить эмбеддинг текста с кэшированием."""
        if text in self.cache:
            return self.cache[text]
        embedding = self.generate_embedding(text)
        self.cache[text] = embedding
        return embedding
    
    def _generate_hash_embedding(self, text: str) -> List[float]:
        """Генерация эмбеддинга на основе хэша (fallback)."""
        hash_bytes = hashlib.sha256(text.encode()).digest()
        numbers = [int.from_bytes(hash_bytes[i:i+4], 'big') for i in range(0, min(len(hash_bytes), 32), 4)]
        while len(numbers) < self.embedding_dim:
            numbers.extend(numbers)
        embedding = [n / (2**32 - 1) for n in numbers[:self.embedding_dim]]
        norm = sum(x**2 for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]
        return embedding
    
    def find_similar(self, query_text: str, nodes: List[Dict[str, Any]], top_k: int = 5) -> List[tuple]:
        """Найти похожие узлы по текстовому запросу."""
        query_embedding = self.embed_text(query_text)
        similarities = []
        
        for node in nodes:
            node_embedding = self.embed_node(node)
            sim = self.compute_similarity(query_embedding, node_embedding)
            similarities.append((node, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def embed_node(self, node: Any) -> List[float]:
        """Получить эмбеддинг узла."""
        if hasattr(node, 'content'):
            content = node.content
            node_type = node.node_type
            context = node.context if hasattr(node, 'context') else {}
        else:
            content = node.get("content", "")
            node_type = node.get("node_type", "")
            context = node.get("context", {})
        
        combined = f"{content} {node_type} {json.dumps(context)}"
        return self.embed_text(combined)
    
    def compute_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Вычислить косинусное сходство между эмбеддингами."""
        if len(emb1) != len(emb2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = sum(x ** 2 for x in emb1) ** 0.5
        norm2 = sum(x ** 2 for x in emb2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def create_address_from_text(self, text: str) -> Dict[str, Any]:
        """Создать фрактальный адрес из текста."""
        embedding = self.embed_text(text)
        
        return {
            "dimensions": embedding,
            "hash": hashlib.md5(str(embedding).encode()).hexdigest()[:16],
            "text": text[:100]
        }
    
    def create_address_from_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Создать фрактальный адрес из запроса с контекстом."""
        base_embedding = self.embed_text(query)
        
        if context:
            context_str = str(context)
            context_embedding = self.embed_text(context_str)
            combined = [(a + b) / 2 for a, b in zip(base_embedding, context_embedding)]
        else:
            combined = base_embedding
        
        return {
            "dimensions": combined,
            "hash": hashlib.md5(str(combined).encode()).hexdigest()[:16],
            "query": query[:100]
        }
    
    def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """Пакетная генерация эмбеддингов."""
        if self._use_st and self._model:
            embeddings = self._model.encode(texts, normalize_embeddings=True)
            return [emb.tolist()[:self.embedding_dim] for emb in embeddings]
        
        return [self._generate_hash_embedding(text) for text in texts]
    
    def save_embeddings(self, path: str) -> bool:
        """Сохранить кэш эмбеддингов на диск."""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2)
            logger.info(f"Сохранено {len(self.cache)} эмбеддингов в {path}")
            return True
        except Exception as e:
            logger.error(f"Ошибка сохранения эмбеддингов: {e}")
            return False
    
    def load_embeddings(self, path: str) -> bool:
        """Загрузить кэш эмбеддингов с диска."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.cache = json.load(f)
            logger.info(f"Загружено {len(self.cache)} эмбеддингов из {path}")
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки эмбеддингов: {e}")
            return False
    
    def embed_reasoning_step(
        self, 
        content: str, 
        parent_step_id: str = None,
        iteration: int = 0
    ) -> List[float]:
        """Эмбеддинг шага рассуждения с учётом позиции в цепочке."""
        base_emb = self.embed_text(content)
        
        iteration_factor = iteration / 100.0
        
        result = base_emb[:]
        for i in range(min(len(result), self.embedding_dim // 10)):
            result[i] = result[i] * (1 - iteration_factor) + iteration_factor * (i / 10)
        
        return result
    
    def compare_reasoning_chains(
        self, 
        chain1: List[Dict], 
        chain2: List[Dict]
    ) -> float:
        """Сравнение двух цепочек рассуждений."""
        if not chain1 or not chain2:
            return 0.0
        
        emb1 = self.embed_text(" ".join([n.get("content", "") for n in chain1]))
        emb2 = self.embed_text(" ".join([n.get("content", "") for n in chain2]))
        
        return self.compute_similarity(emb1, emb2)
