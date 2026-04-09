"""
DualGenerator - Физически разделённые генераторы для разных задач

Использует 2 физических инстанса модели:
1. CondensedGenerator - быстрые краткие ответы
2. ExtendedGenerator - развёрнутые ответы с расширением контекста

Преимущества:
- Параллельная загрузка моделей
- Независимые параметры генерации
- Разные промты для разных задач
"""

import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .hybrid_tokenizer import HybridTokenizer
from .gguf_shadow import GGUFShadowProfiler
from .semantic_context_cache import SemanticContextCache

logger = logging.getLogger("eva_ai.fractal_graph_v2.dual_generator")


CONDENSED_PROMPT = """Ты — краткий ассистент. Дай ответ в 1-2 предложениях.

Вопрос: {query}

Ответ:"""

EXTENDED_PROMPT = """Дай развёрнутый и подробный ответ на вопрос. НЕ повторяй уже написанное.

Вопрос: {query}

Контекст: {graph_context}

Подробный ответ (без повторений):"""


@dataclass
class GeneratorStats:
    """Статистика генератора."""
    total_calls: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    total_tokens: int = 0


class CondensedGenerator:
    """Генератор кратких ответов - быстрый, один вызов модели."""
    
    def __init__(
        self,
        llama_model,
        graph=None,
        max_tokens: int = 512,
        temperature: float = 0.1
    ):
        self.llama = llama_model
        self.graph = graph
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stats = GeneratorStats()
        logger.info(f"CondensedGenerator инициализирован: max_tokens={max_tokens}")
    
    def generate(self, query: str, context: str = "") -> str:
        """Генерация краткого ответа."""
        start = time.time()
        self.stats.total_calls += 1
        
        prompt = CONDENSED_PROMPT.format(query=query)
        
        try:
            output = self.llama(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                repeat_penalty=1.3,
                stop=["</s>", "User:", "user:", "Human:", "Вопрос:", "Контекст:"],
                echo=False
            )
            
            response = ""
            if isinstance(output, dict):
                response = output.get('choices', [{}])[0].get('text', '')
            else:
                response = str(output)
            
            response = self._clean_response(response)
            
        except Exception as e:
            logger.error(f"CondensedGenerator error: {e}")
            response = "Не удалось сгенерировать ответ."
        
        elapsed = time.time() - start
        self.stats.total_time += elapsed
        self.stats.avg_time = self.stats.total_time / self.stats.total_calls
        self.stats.total_tokens += len(response.split())
        
        return response
    
    def _clean_response(self, text: str) -> str:
        """Очистка ответа."""
        text = text.strip()
        
        fillers = ['хорошо,', 'конечно,', 'вот,', 'отлично,']
        for f in fillers:
            if text.lower().startswith(f):
                text = text[len(f):].strip()
        
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()][:2]
        
        result = '. '.join(sentences)
        if result and not result.endswith('.'):
            result += '.'
        
        return result


class ExtendedGenerator:
    """Генератор развёрнутых ответов - с анализом и примерами."""
    
    def __init__(
        self,
        llama_model,
        graph=None,
        max_tokens: int = 2048,
        temperature: float = 0.4
    ):
        self.llama = llama_model
        self.graph = graph
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stats = GeneratorStats()
        logger.info(f"ExtendedGenerator инициализирован: max_tokens={max_tokens}")
    
    def generate(self, query: str, context: str = "") -> str:
        """Генерация развёрнутого ответа."""
        start = time.time()
        self.stats.total_calls += 1
        
        graph_context = self._get_context(query) if self.graph else context
        prompt = EXTENDED_PROMPT.format(query=query, graph_context=graph_context)
        
        try:
            output = self.llama(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                repeat_penalty=1.3,
                stop=["</s>", "User:", "user:", "Human:", "Вопрос:", "Контекст:"],
                echo=False
            )
            
            response = ""
            if isinstance(output, dict):
                response = output.get('choices', [{}])[0].get('text', '')
            else:
                response = str(output)
            
            response = self._clean_response(response)
            
        except Exception as e:
            logger.error(f"ExtendedGenerator error: {e}")
            response = "Не удалось сгенерировать развёрнутый ответ."
        
        elapsed = time.time() - start
        self.stats.total_time += elapsed
        self.stats.avg_time = self.stats.total_time / self.stats.total_calls
        self.stats.total_tokens += len(response.split())
        
        return response
    
    def _get_context(self, query: str) -> str:
        """Получить релевантный контекст из графа."""
        if not self.graph or not hasattr(self.graph, 'nodes'):
            return "Нет контекста"
        
        query_lower = query.lower()
        relevant = []
        
        for node_id, node in list(self.graph.nodes.items())[:50]:
            content = getattr(node, 'content', '')
            if content and any(kw in content.lower() for kw in query_lower.split()[:3]):
                relevant.append(content[:200])
        
        if relevant:
            return ' | '.join(relevant[:3])
        return "Нет релевантного контекста"
    
    def _clean_response(self, text: str) -> str:
        """Очистка ответа."""
        text = text.strip()
        
        lines = text.split('\n')
        lines = [l.strip() for l in lines if l.strip()]
        
        return '\n'.join(lines[:10])


class DualGenerator:
    """
    Объединённый генератор с 2 физическими моделями.
    
    Использование:
    ```python
    dual = DualGenerator(llama_condensed, llama_extended, graph)
    
    brief = dual.generate_condensed("Что такое Python?")
    extended = dual.generate_extended("Объясни разницу между ML и DL")
    ```
    """
    
    def __init__(
        self,
        llama_condensed: Any,
        llama_extended: Any,
        graph=None,
        condensed_max_tokens: int = 512,
        extended_max_tokens: int = 2048
    ):
        self.condensed = CondensedGenerator(
            llama_model=llama_condensed,
            graph=graph,
            max_tokens=condensed_max_tokens
        )
        
        self.extended = ExtendedGenerator(
            llama_model=llama_extended,
            graph=graph,
            max_tokens=extended_max_tokens
        )
        
        self.graph = graph
        logger.info("DualGenerator инициализирован с двумя физическими моделями")
    
    def generate_condensed(self, query: str) -> str:
        """Генерация краткого ответа."""
        return self.condensed.generate(query)
    
    def generate_extended(self, query: str) -> str:
        """Генерация развёрнутого ответа."""
        return self.extended.generate(query)
    
    def generate(
        self,
        query: str,
        mode: str = "auto"
    ) -> str:
        """
        Умная генерация.
        
        Args:
            query: Текст запроса
            mode: 'condensed', 'extended', 'auto'
                - 'condensed': всегда краткий
                - 'extended': всегда развёрнутый
                - 'auto': определяет по ключевым словам
        """
        if mode == "condensed":
            return self.generate_condensed(query)
        elif mode == "extended":
            return self.generate_extended(query)
        else:
            return self._auto_generate(query)
    
    def _auto_generate(self, query: str) -> str:
        """Автоматическое определение режима."""
        query_lower = query.lower()
        
        brief_keywords = ['кратко', 'вкратце', 'суть', 'кто такой', 'перечисли', 'назови']
        for kw in brief_keywords:
            if kw in query_lower:
                return self.generate_condensed(query)
        
        return self.generate_extended(query)
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику обоих генераторов."""
        return {
            'condensed': {
                'calls': self.condensed.stats.total_calls,
                'avg_time': self.condensed.stats.avg_time,
                'total_tokens': self.condensed.stats.total_tokens
            },
            'extended': {
                'calls': self.extended.stats.total_calls,
                'avg_time': self.extended.stats.avg_time,
                'total_tokens': self.extended.stats.total_tokens
            }
        }
