"""
Hybrid Cache Bridge - интеграция гибридного кэша с llama.cpp.
Объединяет наш semantic index + prompt cache с встроенным prefix-matching GGUF.
"""

import os
import time
import hashlib
import logging
from typing import Dict, List, Optional, Any, Tuple
import json

logger = logging.getLogger(__name__)


class HybridCacheBridge:
    """
    Мост между гибридным кэшем и llama.cpp prefix-matching.
    """
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or 'tests/model_optimization/cache'
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Наш гибридный кэш
        self._semantic_index: Dict[str, Any] = {}
        self._prompt_cache: Dict[str, Tuple[bytes, bytes]] = {}  # prompt_hash -> (tokens, attn)
        
        # Статистика
        self.stats = {
            'semantic_hits': 0,
            'semantic_misses': 0,
            'prompt_cache_hits': 0,
            'prefix_match_from_llama': 0,
            'total_queries': 0
        }
        
        self._load_state()
        
        logger.info("HybridCacheBridge инициализирован")
    
    def _load_state(self):
        """Загрузка состояния из диска."""
        state_file = os.path.join(self.cache_dir, 'hybrid_bridge_state.json')
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    self.stats = data.get('stats', self.stats)
                logger.info("Состояние загружено")
            except Exception as e:
                logger.warning(f"Не загрузили состояние: {e}")
    
    def _save_state(self):
        """Сохранение состояния."""
        state_file = os.path.join(self.cache_dir, 'hybrid_bridge_state.json')
        try:
            with open(state_file, 'w') as f:
                json.dump({'stats': self.stats}, f, indent=2)
        except Exception as e:
            logger.warning(f"Не сохранили состояние: {e}")
    
    def _get_prompt_hash(self, prompt: str) -> str:
        """Хэш промпта для кэширования."""
        return hashlib.sha256(prompt.encode()).hexdigest()[:24]
    
    def add_to_semantic_index(self, doc_id: str, content: str):
        """Добавить документ в семантический индекс."""
        self._semantic_index[doc_id] = content
    
    def get_relevant_context(self, query: str) -> Optional[str]:
        """Получить релевантный контекст из семантического индекса."""
        self.stats['total_queries'] += 1
        
        # Простой поиск по ключевым словам
        query_words = set(query.lower().split())
        
        best_doc = None
        best_score = 0
        
        for doc_id, content in self._semantic_index.items():
            content_words = set(content.lower().split())
            score = len(query_words & content_words)
            
            if score > best_score:
                best_score = score
                best_doc = content
        
        if best_score > 0:
            self.stats['semantic_hits'] += 1
            return best_doc[:500]  # Ограничиваем длину
        
        self.stats['semantic_misses'] += 1
        return None
    
    def cache_prompt(self, prompt: str, tokens: bytes, attn_mask: bytes = None):
        """Кэшировать токенизированный промпт."""
        key = self._get_prompt_hash(prompt)
        
        # LRU вытеснение
        if len(self._prompt_cache) > 500:
            oldest = next(iter(self._prompt_cache))
            del self._prompt_cache[oldest]
        
        self._prompt_cache[key] = (tokens, attn_mask or b'')
    
    def get_cached_prompt(self, prompt: str) -> Optional[Tuple[bytes, bytes]]:
        """Получить кэшированный промпт."""
        key = self._get_prompt_hash(prompt)
        
        if key in self._prompt_cache:
            self.stats['prompt_cache_hits'] += 1
            return self._prompt_cache[key]
        
        return None
    
    def record_prefix_match(self, count: int):
        """Зафиксировать prefix-match от llama.cpp."""
        self.stats['prefix_match_from_llama'] += count
    
    def get_combined_cache_stats(self) -> Dict:
        """Получить объединенную статистику кэшей."""
        total = self.stats['semantic_hits'] + self.stats['semantic_misses']
        
        return {
            **self.stats,
            'semantic_hit_rate': self.stats['semantic_hits'] / total * 100 if total > 0 else 0,
            'prompt_cache_size': len(self._prompt_cache),
            'semantic_index_size': len(self._semantic_index)
        }
    
    def build_context_for_prompt(self, prompt: str) -> str:
        """Построить обогащенный контекст для промпта."""
        # 1. Семантический поиск
        context = self.get_relevant_context(prompt)
        
        if context:
            # Формируем обогащенный промпт
            enriched = f"""Контекст из базы знаний:
{context}

Вопрос пользователя: {prompt}

Ответь на основе контекста, если он релевантен."""
            return enriched
        
        return prompt
    
    def save(self):
        """Сохранить состояние."""
        self._save_state()


class LlamaCppCacheIntegrator:
    """
    Интегратор для связи с llama.cpp.
    """
    
    def __init__(self, bridge: HybridCacheBridge, llm=None):
        self.bridge = bridge
        self.llm = llm
        
        # llm.cpp internally caches KV for prefix matches
        # Мы можем:1) Обогащать промпты через semantic index
        # 2) Предварительно токенизировать через наш кэш
    
    def generate_with_bridge(
        self,
        prompt: str,
        max_tokens: int = 50,
        use_bridge: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Генерация с использованием моста кэшей."""
        
        start = time.perf_counter()
        
        # Обогащаем промпт через semantic index
        if use_bridge:
            enriched_prompt = self.bridge.build_context_for_prompt(prompt)
        else:
            enriched_prompt = prompt
        
        # Проверяем кэш токенов
        cached_tokens = self.bridge.get_cached_prompt(enriched_prompt)
        
        if cached_tokens:
            logger.info(f"Используем кэшированные токены ({len(cached_tokens[0])} байт)")
        
        # Генерация
        if self.llm:
            output = self.llm(enriched_prompt, max_tokens=max_tokens, **kwargs)
            response = output['choices'][0]['text']
        else:
            response = f"Ответ: {prompt[:30]}..."
        
        elapsed = time.perf_counter() - start
        
        # Статистика от llama.cpp (если доступна)
        # В реальном llm.cpp она в output['timings']
        
        return {
            'original_prompt': prompt,
            'enriched_prompt': enriched_prompt if use_bridge else None,
            'used_semantic_context': use_bridge and enriched_prompt != prompt,
            'response': response,
            'time_seconds': elapsed,
            'cache_stats': self.bridge.get_combined_cache_stats()
        }


def create_cache_bridge(cache_dir: str = None) -> HybridCacheBridge:
    """Фабричная функция."""
    return HybridCacheBridge(cache_dir)


# === Интеграция с pipeline ===

class CachedPipeline:
    """
    Пайплайн с интегрированным гибридным кэшем.
    """
    
    def __init__(self, model_path: str = None, n_ctx: int = 512):
        self.llm = None
        self.model_path = model_path
        self.n_ctx = n_ctx
        
        # Инициализируем мост
        self.cache_bridge = create_cache_bridge()
        
        # Добавляем базовые знания
        self._init_knowledge_base()
    
    def _init_knowledge_base(self):
        """Добавление базы знаний."""
        knowledge = [
            ("ai_russia", "Россия развивает ИИ: нацстратегия 2019, нацпрограмма 2024, Сколково, Иннополис, МФТИ, Физтех"),
            ("ml_algos", "ML алгоритмы: нейросети, трансформеры, LSTM, CNN, RNN, случайный лес, градиентный бустинг"),
            ("quantum_ml", "Квантовое ML: квантовые алгоритмы ускорят обучение, квантовое превосходство ожидается к 2030"),
            ("fractal_memory", "Фрактальная память: самоподобные структуры для эффективного хранения и извлечения информации"),
            ("cogniflex", "CogniFlex (EVA) - когнитивная система с рекурсивным рассуждением и самодиалогом"),
        ]
        
        for doc_id, content in knowledge:
            self.cache_bridge.add_to_semantic_index(doc_id, content)
        
        logger.info(f"База знаний: {len(knowledge)} документов")
    
    def load_model(self):
        """Ленивая загрузка модели."""
        if self.llm:
            return
        
        if not self.model_path or not os.path.exists(self.model_path):
            logger.warning("Модель не указана")
            return
        
        from llama_cpp import Llama
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=4
        )
        
        self.integrator = LlamaCppCacheIntegrator(self.cache_bridge, self.llm)
        logger.info(f"Модель загружена: {self.model_path}")
    
    def generate(self, prompt: str, use_context: bool = True, **kwargs) -> Dict:
        """Генерация с гибридным кэшированием."""
        if not self.llm:
            self.load_model()
        
        return self.integrator.generate_with_bridge(
            prompt=prompt,
            use_bridge=use_context,
            **kwargs
        )


if __name__ == '__main__':
    # Тест
    bridge = create_cache_bridge()
    
    # Добавляем знания
    bridge.add_to_semantic_index("test", "Тестовая информация о системе")
    
    # Проверяем работу
    context = bridge.get_relevant_context("информация о системе")
    print(f"Контекст: {context}")
    
    # Статистика
    print(f"Статистика: {bridge.get_combined_cache_stats()}")