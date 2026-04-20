"""
ThoughtContextExtractor - Извлечение релевантных рассуждений для Model A.

Извлекает из предыдущих рассуждений Model B семантически и логически 
релевантные данные к текущему запросу и подаёт в system_prompt Model A.

Использование:
```
extractor = ThoughtContextExtractor(brain)
relevant = extractor.extract_relevant_thoughts(
    query="Как работает X?",
    max_thoughts=3
)
# → "Контекст из предыдущих рассуждений: посылки: A, B | выводы: C"
```
"""

import logging
import re
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger("eva_ai.core.thought_context")

try:
    from eva_ai.memory.fractal_graph_v2.embeddings import get_embeddings_manager
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    get_embeddings_manager = None

@dataclass
class ThoughtFragment:
    """Фрагмент рассуждения."""
    text: str
    thought_type: str  # premise, conclusion, uncertainty, gap, step
    relevance_score: float = 0.0
    source_iteration: int = 0


class ThoughtContextExtractor:
    """
    Извлекатель релевантных рассуждений для контекста Model A.
    """
    
    def __init__(self, brain=None):
        self.brain = brain
        
        self._thought_cache: List[Dict[str, Any]] = []
        self._max_cache = 100
        
        self._embeddings_manager = None
        if EMBEDDINGS_AVAILABLE:
            try:
                self._embeddings_manager = get_embeddings_manager()
            except Exception as e:
                logger.warning(f"Embeddings manager not available: {e}")
        
        self._keyword_weights = {
            'потому что': 0.8,
            'значит': 0.7,
            'следовательно': 0.8,
            'вывод': 0.9,
            'однако': 0.6,
            'но': 0.5,
            'возможно': 0.4,
            'вероятно': 0.4,
            'если': 0.6,
            'то': 0.5,
        }
    
    def store_thought(
        self,
        thought_text: str,
        query: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Сохраняет рассуждение в кэш для будущего использования.
        
        Args:
            thought_text: Текст рассуждения.
            query: Запрос, на который это рассуждение.
            metadata: Дополнительные метаданные.
        """
        fragments = self._extract_fragments(thought_text)
        
        embedding = None
        if self._embeddings_manager and self._embeddings_manager.model:
            try:
                embedding = self._embeddings_manager.encode_single(thought_text, normalize=True)
                if embedding is not None:
                    embedding = embedding.tolist()
            except Exception as e:
                logger.debug(f"Embeddings encode failed: {e}")
        
        entry = {
            'query': query,
            'thought': thought_text,
            'fragments': fragments,
            'embedding': embedding,
            'metadata': metadata or {}
        }
        
        self._thought_cache.append(entry)
        
        if len(self._thought_cache) > self._max_cache:
            self._thought_cache.pop(0)
        
        logger.debug(f"Stored thought with {len(fragments)} fragments, embedding={embedding is not None}")
    
    def extract_relevant_thoughts(
        self,
        query: str,
        max_thoughts: int = 3,
        min_relevance: float = 0.3
    ) -> str:
        """
        Извлекает релевантные рассуждения для текущего запроса.
        Использует KEMBEDDINGS для семантического сходства + keyword matching.
        
        Args:
            query: Текущий запрос пользователя.
            max_thoughts: Максимум фрагментов.
            min_relevance: Минимальный порог релевантности.
            
        Returns:
            Форматированная строка для system_prompt или пустая строка.
        """
        if not self._thought_cache:
            return ""
        
        query_emb = None
        if self._embeddings_manager and self._embeddings_manager.model:
            try:
                query_emb = self._embeddings_manager.encode_single(query, normalize=True)
            except Exception as e:
                logger.debug(f"Query embedding failed: {e}")
        
        query_lower = query.lower()
        query_keywords = set(re.findall(r'\w+', query_lower))
        
        scored_fragments: List[Tuple[ThoughtFragment, float]] = []
        
        for entry in reversed(self._thought_cache[-20:]):
            thought_emb = entry.get('embedding')
            fragments = entry.get('fragments', [])
            
            for frag in fragments:
                score = self._calculate_relevance(
                    query, query_keywords, frag, 
                    query_emb=query_emb, 
                    thought_emb=thought_emb
                )
                
                if score >= min_relevance:
                    scored_fragments.append((frag, score))
        
        scored_fragments.sort(key=lambda x: x[1], reverse=True)
        
        top_fragments = scored_fragments[:max_thoughts]
        
        if not top_fragments:
            return ""
        
        return self._format_for_prompt(top_fragments)
    
    def _extract_fragments(self, thought_text: str) -> List[ThoughtFragment]:
        """Извлекает фрагменты из текста рассуждения."""
        fragments = []
        
        lines = thought_text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or len(line) < 20:
                continue
            
            line_lower = line.lower()
            frag_type = 'step'
            
            if any(k in line_lower for k in ['значит', 'следовательно', 'вывод', '→']):
                frag_type = 'conclusion'
            elif any(k in line_lower for k in ['потому что', 'так как', 'исходя']):
                frag_type = 'premise'
            elif any(k in line_lower for k in ['возможно', 'вероятно', 'может']):
                frag_type = 'uncertainty'
            elif any(k in line_lower for k in ['но', 'однако', 'пропуск']):
                frag_type = 'gap'
            elif re.search(r'\d+\.', line):
                frag_type = 'step'
            
            fragments.append(ThoughtFragment(
                text=line[:200],
                thought_type=frag_type,
                source_iteration=i
            ))
        
        return fragments
    
    def _calculate_relevance(
        self,
        query: str,
        query_keywords: set,
        fragment: ThoughtFragment,
        query_emb: Optional[np.ndarray] = None,
        thought_emb: Optional[List[float]] = None
    ) -> float:
        """
        Вычисляет релевантность фрагмента к запросу.
        Комбинирует keyword matching + semantic similarity через embeddings.
        
        Args:
            query: Запрос пользователя.
            query_keywords: Ключевые слова запроса.
            fragment: Фрагмент рассуждения.
            query_emb: Эмбеддинг запроса (опционально).
            thought_emb: Эмбеддинг мысли (опционально).
            
        Returns:
            Оценка релевантности 0.0 - 1.0.
        """
        text_lower = fragment.text.lower()
        frag_keywords = set(re.findall(r'\w+', text_lower))
        
        if not frag_keywords:
            return 0.0
        
        keyword_overlap = len(query_keywords & frag_keywords)
        
        base_score = keyword_overlap / max(len(query_keywords), 1)
        
        keyword_bonus = 0.0
        for kw, weight in self._keyword_weights.items():
            if kw in text_lower:
                keyword_bonus += weight
        
        type_bonus = {
            'conclusion': 0.2,
            'premise': 0.15,
            'step': 0.1,
            'uncertainty': 0.05,
            'gap': 0.1
        }.get(fragment.thought_type, 0.0)
        
        semantic_score = 0.0
        if query_emb is not None and thought_emb is not None:
            try:
                thought_emb_arr = np.array(thought_emb)
                query_norm = np.linalg.norm(query_emb)
                thought_norm = np.linalg.norm(thought_emb_arr)
                if query_norm > 0 and thought_norm > 0:
                    semantic_score = np.dot(query_emb, thought_emb_arr) / (query_norm * thought_norm)
            except Exception:
                pass
        
        keyword_part = base_score * 0.5 + min(keyword_bonus, 0.4) + type_bonus
        semantic_part = semantic_score * 0.4 if semantic_score > 0 else 0.0
        
        total = keyword_part + semantic_part
        
        return min(1.0, max(0.0, total))
    
    def _format_for_prompt(
        self,
        scored_fragments: List[Tuple[ThoughtFragment, float]]
    ) -> str:
        """
        Форматирует фр��гменты для system_prompt.
        
        Args:
            scored_fragments: Список (фрагмент, оценка).
            
        Returns:
            Строка для system_prompt.
        """
        parts = []
        
        by_type = defaultdict(list)
        for frag, score in scored_fragments:
            by_type[frag.thought_type].append((frag.text, score))
        
        if 'premise' in by_type:
            premises = [f[0] for f in by_type['premise'][:2]]
            if premises:
                parts.append(f"Посылки: {'; '.join(premises)}")
        
        if 'conclusion' in by_type:
            conclusions = [f[0] for f in by_type['conclusion'][:2]]
            if conclusions:
                parts.append(f"Выводы: {'; '.join(conclusions)}")
        
        if 'step' in by_type:
            steps = [f[0] for f in by_type['step'][:2]]
            if steps:
                parts.append(f"Шаги: {'; '.join(steps)}")
        
        if 'uncertainty' in by_type:
            unc = [f[0] for f in by_type['uncertainty'][:1]]
            if unc:
                parts.append(f"Неопределённости: {'; '.join(unc)}")
        
        if 'gap' in by_type:
            gaps = [f[0] for f in by_type['gap'][:1]]
            if gaps:
                parts.append(f"Пропуски: {'; '.join(gaps)}")
        
        result = " | ".join(parts)
        
        if len(result) > 500:
            result = result[:497] + "..."
        
        return f"[Из предыдущих рассуждений: {result}]"
    
    def get_thought_history(
        self,
        query: str = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Получает историю рассуждений.
        
        Args:
            query: Опциональный фильтр по запросу.
            limit: Максимум записей.
            
        Returns:
            Список записей.
        """
        if not query:
            return self._thought_cache[-limit:]
        
        query_lower = query.lower()
        filtered = [
            e for e in self._thought_cache
            if query_lower in e.get('query', '').lower()
        ]
        
        return filtered[-limit:]
    
    def clear_cache(self):
        """Очищает кэш рассуждений."""
        self._thought_cache.clear()
        logger.info("Thought cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Получает статистику."""
        return {
            'cached_thoughts': len(self._thought_cache),
            'max_cache': self._max_cache,
        }


def create_thought_extractor(brain=None) -> ThoughtContextExtractor:
    """Фабричная функция."""
    return ThoughtContextExtractor(brain=brain)