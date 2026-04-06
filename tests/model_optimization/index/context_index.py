"""
Context Index - модуль индексации контекста для ускорения генерации.
Использует LRU кэш, предиктивный словарь, быстрый lookup.
"""

import os
import time
import hashlib
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from collections import OrderedDict
import numpy as np

logger = logging.getLogger(__name__)


class LRUCache:
    """LRU кэш для токенов и их представлений."""
    
    def __init__(self, max_size: int = 10000):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, key: str, value: Any):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def get_stats(self) -> Dict[str, int]:
        return {
            'size': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / (self.hits + self.misses) * 100 if (self.hits + self.misses) > 0 else 0
        }


class TokenPredictor:
    """Предиктор следующих токенов на основе частотного анализа."""
    
    def __init__(self):
        self.ngram_probs: Dict[Tuple, Dict[str, float]] = {}
        self.vocab: Dict[str, int] = {}
    
    def train(self, texts: List[str], n: int = 2):
        """Обучение на текстах."""
        for text in texts:
            words = text.split()
            for i in range(len(words) - n):
                key = tuple(words[i:i+n])
                next_word = words[i+n]
                if key not in self.ngram_probs:
                    self.ngram_probs[key] = {}
                self.ngram_probs[key][next_word] = self.ngram_probs[key].get(next_word, 0) + 1
        
        for key in self.ngram_probs:
            total = sum(self.ngram_probs[key].values())
            for w in self.ngram_probs[key]:
                self.ngram_probs[key][w] /= total
        
        logger.info(f"TokenPredictor обучен на {len(texts)} текстах, n={n}")
    
    def predict(self, prefix: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Предсказание следующих токенов."""
        key = tuple(prefix[-2:]) if len(prefix) >= 2 else tuple(prefix)
        
        if key in self.ngram_probs:
            probs = sorted(self.ngram_probs[key].items(), key=lambda x: -x[1])
            return probs[:top_k]
        
        return []


class FastTokenizer:
    """Оптимизированный токенизатор с быстрым словарём."""
    
    def __init__(self, vocab: Dict[str, int] = None):
        self.vocab = vocab or {}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.special_tokens = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
        
        # Быстрый словарь для частых слов
        self.frequent_words: Dict[str, int] = {}
        self._build_fast_dict()
    
    def _build_fast_dict(self):
        """Построение быстрого словаря."""
        if self.vocab:
            sorted_vocab = sorted(self.vocab.items(), key=lambda x: -x[1])
            for word, idx in sorted_vocab[:5000]:
                self.frequent_words[word] = idx
        logger.info(f"FastTokenizer: {len(self.frequent_words)} частых слов")
    
    def encode_fast(self, text: str) -> List[int]:
        """Быстрое кодирование."""
        words = text.split()
        tokens = []
        
        for word in words:
            if word in self.frequent_words:
                tokens.append(self.frequent_words[word])
            elif word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                tokens.append(self.special_tokens['<unk>'])
        
        return tokens
    
    def decode_fast(self, tokens: List[int]) -> str:
        """Быстрое декодирование."""
        words = []
        for t in tokens:
            if t in self.reverse_vocab:
                words.append(self.reverse_vocab[t])
            elif t in self.special_tokens:
                words.append(list(self.special_tokens.keys())[list(self.special_tokens.values()).index(t)])
            else:
                words.append('<unk>')
        return ' '.join(words)


class ContextIndex:
    """Главный класс индексации контекста."""
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or 'tests/model_optimization/cache'
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.token_cache = LRUCache(max_size=5000)
        self.predictor = TokenPredictor()
        self.tokenizer = FastTokenizer()
        
        self.stats = {
            'tokenize_calls': 0,
            'cache_hits': 0,
            'predictions': 0
        }
        
        self._load_cache()
    
    def _get_text_hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    
    def _load_cache(self):
        """Загрузка кэша с диска."""
        cache_file = os.path.join(self.cache_dir, 'token_cache.json')
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    for k, v in data.items():
                        self.token_cache.put(k, v)
                logger.info(f"Загружен кэш: {len(self.token_cache.cache)} записей")
            except Exception as e:
                logger.warning(f"Не удалось загрузить кэш: {e}")
    
    def _save_cache(self):
        """Сохранение кэша на диск."""
        cache_file = os.path.join(self.cache_dir, 'token_cache.json')
        try:
            data = dict(self.token_cache.cache)
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Не удалось сохранить кэш: {e}")
    
    def tokenize_with_cache(self, text: str) -> Tuple[List[int], bool]:
        """Токенизация с использованием кэша."""
        self.stats['tokenize_calls'] += 1
        
        text_hash = self._get_text_hash(text)
        cached_tokens = self.token_cache.get(text_hash)
        
        if cached_tokens is not None:
            self.stats['cache_hits'] += 1
            return cached_tokens, True
        
        tokens = self.tokenizer.encode_fast(text)
        self.token_cache.put(text_hash, tokens)
        
        if self.stats['tokenize_calls'] % 100 == 0:
            self._save_cache()
        
        return tokens, False
    
    def predict_next_tokens(self, prefix: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Предсказание следующих токенов."""
        self.stats['predictions'] += 1
        return self.predictor.predict(prefix, top_k)
    
    def add_training_data(self, texts: List[str]):
        """Добавление данных для обучения предиктора."""
        self.predictor.train(texts, n=2)
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику."""
        cache_stats = self.token_cache.get_stats()
        return {
            **self.stats,
            'cache': cache_stats
        }
    
    def clear_cache(self):
        """Очистка кэша."""
        self.token_cache.cache.clear()
        self._save_cache()
        logger.info("Кэш очищен")