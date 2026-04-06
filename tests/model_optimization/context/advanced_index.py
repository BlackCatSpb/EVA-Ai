"""
Advanced Context Index - оптимизированная индексация контекста.
Кэширует результаты prompt evaluation и ускоряет контекстный поиск.
"""

import os
import time
import hashlib
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from collections import OrderedDict
import numpy as np
import re

logger = logging.getLogger(__name__)


class PromptEvalCache:
    """
    Кэш результатов prompt evaluation для llama.cpp.
    Кэширует уже токенизированные промпты для избежания повторной обработки.
    """
    
    def __init__(self, max_size: int = 1000):
        self.cache: OrderedDict[str, Dict] = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def _hash_prompt(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]
    
    def get_cached_tokens(self, prompt: str) -> Optional[bytes]:
        """Получить токенизированный промпт из кэша."""
        key = self._hash_prompt(prompt)
        
        if key in self.cache:
            self.hits += 1
            # Перемещаем в конец (LRU)
            self.cache.move_to_end(key)
            return self.cache[key].get('tokens')
        
        self.misses += 1
        return None
    
    def store_tokens(self, prompt: str, tokens: bytes):
        """Сохранить токенизированный промпт."""
        key = self._hash_prompt(prompt)
        
        self.cache[key] = {
            'prompt': prompt,
            'tokens': tokens,
            'timestamp': time.time()
        }
        
        # LRU вытеснение
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def get_stats(self) -> Dict:
        total = self.hits + self.misses
        return {
            'size': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / total * 100 if total > 0 else 0
        }


class SemanticContextIndex:
    """
    Семантический индекс контекста для быстрого поиска релевантной информации.
    """
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        
        # Индекс документов
        self.documents: Dict[str, Dict] = {}
        
        # Инвертированный индекс (слово -> документы)
        self.inverted_index: Dict[str, List[str]] = {}
        
        # Семантические эмбеддинги
        self.embeddings: Dict[str, np.ndarray] = {}
        
        # Кэш частых запросов
        self.query_cache: Dict[str, List[str]] = {}
    
    def add_document(self, doc_id: str, content: str, metadata: Dict = None):
        """Добавить документ в индекс."""
        self.documents[doc_id] = {
            'content': content,
            'metadata': metadata or {},
            'timestamp': time.time()
        }
        
        # Индексируем слова
        words = self._tokenize(content)
        for word in set(words):
            if word not in self.inverted_index:
                self.inverted_index[word] = []
            if doc_id not in self.inverted_index[word]:
                self.inverted_index[word].append(doc_id)
        
        # Вычисляем эмбеддинг
        self.embeddings[doc_id] = self._compute_embedding(words)
    
    def _tokenize(self, text: str) -> List[str]:
        """Токенизация текста."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        # Фильтруем стоп-слова
        stop_words = {'и', 'в', 'на', 'с', 'по', 'для', 'о', 'что', 'как', 'это', 'а', 'но', 'не', 'из', 'к', 'за', 'от'}
        return [w for w in words if w not in stop_words and len(w) > 2]
    
    def _compute_embedding(self, words: List[str]) -> np.ndarray:
        """Вычисление эмбеддинга документа."""
        np.random.seed(sum(ord(c) for c in ''.join(words)) % (2**32))
        emb = np.random.randn(self.embedding_dim).astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-8
        return emb
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Поиск релевантных документов."""
        query_words = self._tokenize(query)
        
        # Проверяем кэш
        cache_key = ' '.join(sorted(query_words))
        if cache_key in self.query_cache:
            return self.query_cache[cache_key][:top_k]
        
        # Подсчет баллов
        scores: Dict[str, float] = {}
        
        for word in query_words:
            if word in self.inverted_index:
                for doc_id in self.inverted_index[word]:
                    scores[doc_id] = scores.get(doc_id, 0) + 1
        
        # Добавляем семантическое сходство
        query_emb = self._compute_embedding(query_words)
        for doc_id, doc_emb in self.embeddings.items():
            similarity = float(np.dot(query_emb, doc_emb))
            scores[doc_id] = scores.get(doc_id, 0) + similarity
        
        # Сортировка
        results = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        
        # Кэшируем
        self.query_cache[cache_key] = results
        
        return results
    
    def get_context_for_query(self, query: str, max_chars: int = 500) -> str:
        """Получить контекст для запроса."""
        results = self.search(query, top_k=3)
        
        contexts = []
        for doc_id, score in results:
            if score > 0.1:  # Порог релевантности
                doc = self.documents.get(doc_id, {})
                content = doc.get('content', '')
                if content:
                    contexts.append(content[:max_chars])
        
        return '\n\n'.join(contexts)


class ContextManager:
    """
    Главный менеджер контекста - объединяет все оптимизации.
    """
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or 'tests/model_optimization/cache'
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.prompt_cache = PromptEvalCache(max_size=500)
        self.semantic_index = SemanticContextIndex(embedding_dim=128)
        
        self.stats = {
            'total_queries': 0,
            'context_retrievals': 0,
            'avg_retrieval_time_ms': 0
        }
        
        logger.info("ContextManager инициализирован")
    
    def add_context(self, doc_id: str, content: str, metadata: Dict = None):
        """Добавить контекст для поиска."""
        self.semantic_index.add_document(doc_id, content, metadata)
    
    def get_relevant_context(self, query: str, max_length: int = 500) -> str:
        """Получить релевантный контекст для запроса."""
        start = time.perf_counter()
        
        context = self.semantic_index.get_context_for_query(query, max_chars=max_length)
        
        elapsed = (time.perf_counter() - start) * 1000
        self.stats['context_retrievals'] += 1
        self.stats['avg_retrieval_time_ms'] = (
            (self.stats['avg_retrieval_time_ms'] * (self.stats['context_retrievals'] - 1) + elapsed) 
            / self.stats['context_retrievals']
        )
        
        return context
    
    def preprocess_prompt(self, query: str, conversation_history: List[Dict] = None) -> str:
        """Предобработка промпта с использованием контекста."""
        self.stats['total_queries'] += 1
        
        # Получаем релевантный контекст
        context = self.get_relevant_context(query)
        
        # Формируем итоговый промпт
        if context:
            prompt = f"Контекст:\n{context}\n\nВопрос: {query}"
        else:
            prompt = query
        
        return prompt
    
    def get_stats(self) -> Dict:
        return {
            **self.stats,
            'prompt_cache': self.prompt_cache.get_stats(),
            'semantic_docs': len(self.semantic_index.documents)
        }
    
    def save_state(self):
        """Сохранение состояния."""
        state = {
            'stats': self.stats,
            'doc_count': len(self.semantic_index.documents)
        }
        
        state_file = os.path.join(self.cache_dir, 'context_manager_state.json')
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Состояние сохранено: {state_file}")
    
    def load_state(self):
        """Загрузка состояния."""
        state_file = os.path.join(self.cache_dir, 'context_manager_state.json')
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                state = json.load(f)
                logger.info(f"Загружено состояние: {state.get('doc_count', 0)} документов")


class ContextAwarePipeline:
    """
    Конвейер с учетом контекста для генерации.
    """
    
    def __init__(self, llm=None, context_manager: ContextManager = None):
        self.llm = llm
        self.context_manager = context_manager or ContextManager()
    
    def generate_with_context(
        self,
        query: str,
        max_tokens: int = 50,
        temperature: float = 0.7,
        use_context: bool = True
    ) -> Dict[str, Any]:
        """Генерация с использованием контекста."""
        
        # Предобработка промпта
        if use_context:
            enriched_query = self.context_manager.preprocess_prompt(query)
        else:
            enriched_query = query
        
        # Генерация
        start = time.perf_counter()
        
        if self.llm:
            output = self.llm(enriched_query, max_tokens=max_tokens, temperature=temperature)
            response = output['choices'][0]['text']
        else:
            response = f"Ответ на: {query[:30]}..."
        
        elapsed = time.perf_counter() - start
        
        return {
            'query': query,
            'response': response,
            'enriched_query': enriched_query if use_context else None,
            'time_seconds': elapsed,
            'used_context': use_context,
            'context_stats': self.context_manager.get_stats()
        }


def create_context_manager(cache_dir: str = None) -> ContextManager:
    """Фабричная функция для создания ContextManager."""
    return ContextManager(cache_dir)


# === Интеграция с основной системой ===

class OptimizedPipeline:
    """
    Оптимизированный пайплайн генерации.
    """
    
    def __init__(self, llm_path: str = None, n_ctx: int = 512):
        self.llm = None
        self.llm_path = llm_path
        self.n_ctx = n_ctx
        
        self.context_manager = ContextManager()
        
        # Пример добавления контекста
        self._init_sample_context()
    
    def _init_sample_context(self):
        """Добавление примера контекста."""
        sample_docs = [
            ("ai_russia", "Россия активно развивает технологии искусственного интеллекта. В 2019 году утверждена национальная стратегия развития ИИ. К 2024 году планируется увеличить долю ИИ в экономике до 1% ВВП."),
            ("ml_algorithms", "Основные алгоритмы машинного обучения: нейронные сети, случайный лес, градиентный бустинг, SVM. Для работы с текстом используются трансформеры и LSTM сети."),
            ("quantum_ml", "Квантовые вычисления могут ускорить обучение нейросетей. Алгоритмы квантового машинного обучения находятся на стадии исследований. Ожидается квантовое превосходство к 2030 году."),
        ]
        
        for doc_id, content in sample_docs:
            self.context_manager.add_context(doc_id, content)
        
        logger.info(f"Добавлено {len(sample_docs)} документов в контекст")
    
    def load_model(self):
        """Ленивая загрузка модели."""
        if self.llm:
            return
        
        if not self.llm_path or not os.path.exists(self.llm_path):
            logger.warning("Модель не указана")
            return
        
        try:
            from llama_cpp import Llama
            self.llm = Llama(
                model_path=self.llm_path,
                n_ctx=self.n_ctx,
                n_threads=4
            )
            logger.info(f"Модель загружена: {self.llm_path}")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
    
    def generate(self, query: str, use_context: bool = True, **kwargs) -> Dict:
        """Генерация ответа."""
        return self.context_manager.generate_with_context(
            query=query,
            llm=self.llm,
            use_context=use_context,
            **kwargs
        )


if __name__ == '__main__':
    # Тест
    cm = ContextManager()
    
    # Добавление контекста
    cm.add_context("doc1", "Искусственный интеллект - это область computer science")
    cm.add_context("doc2", "Машинное обучение - подраздел ИИ")
    cm.add_context("doc3", "Нейросети имитируют работу человеческого мозга")
    
    # Поиск
    results = cm.semantic_index.search("ИИ машинное обучение")
    print(f"Найдено: {results}")
    
    # Статистика
    print(f"Статистика: {cm.get_stats()}")