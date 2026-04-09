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

EXTENDED_PROMPT = """Дай развёрнутый и подробный ответ на вопрос. НЕ повторяй уже написанное. Каждый факт упоминай только ОДИН раз.

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
        
        system_patterns = [
            'Модель B:', 'Модель A:', 'Модель C:', 
            'Model B:', 'Model A:', 'Model C:',
            'Ответ модели B:', 'Ответ модели A:', 'Ответ модели C:',
            'модель B:', 'модель A:'
        ]
        
        for pattern in system_patterns:
            text = text.replace(pattern, '')
        
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
        max_tokens: int = 4096,
        temperature: float = 0.35,
        repeat_penalty: float = 1.8,
        frequency_penalty: float = 0.3,
        presence_penalty: float = 0.2
    ):
        self.llama = llama_model
        self.graph = graph
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.repeat_penalty = repeat_penalty
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stats = GeneratorStats()
        self._seen_ngrams = set()
        logger.info(f"ExtendedGenerator инициализирован: max_tokens={max_tokens}, repeat_penalty={repeat_penalty}")
    
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
                repeat_penalty=self.repeat_penalty,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                stop=["</s>", "User:", "user:", "Human:", "Вопрос:", "Контекст:", "Повтор:", "повтор"],
                echo=False
            )
            
            response = ""
            if isinstance(output, dict):
                response = output.get('choices', [{}])[0].get('text', '')
            else:
                response = str(output)
            
            response = self._remove_repetitions(response)
            
        except Exception as e:
            logger.error(f"ExtendedGenerator error: {e}")
            response = "Не удалось сгенерировать развёрнутый ответ."
        
        elapsed = time.time() - start
        self.stats.total_time += elapsed
        self.stats.avg_time = self.stats.total_time / self.stats.total_calls
        self.stats.total_tokens += len(response.split())
        
        return response
    
    def _remove_repetitions(self, text: str) -> str:
        """Удаление повторяющихся фрагментов из текста."""
        if not text:
            return text
        
        text = text.strip()
        lines = text.split('\n')
        unique_lines = []
        seen_sentences = set()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            normalized = line.lower()[:100]
            if normalized in seen_sentences:
                continue
            
            sentence_set = set()
            for sentence in line.split('.')[:2]:
                s = sentence.strip().lower()[:80]
                if s and s not in sentence_set:
                    sentence_set.add(s)
                    
                    if len(sentence_set) > 1:
                        break
            
            is_duplicate = False
            for prev_line in unique_lines[-3:]:
                common_words = set(line.lower().split()) & set(prev_line.lower().split())
                if len(common_words) >= 5 and len(line) < 150:
                    overlap_ratio = len(common_words) / max(len(set(line.lower().split())), 1)
                    if overlap_ratio > 0.6:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                seen_sentences.add(normalized)
                unique_lines.append(line)
        
        result = '\n'.join(unique_lines[:15])
        
        if len(result) < 50 and len(unique_lines) > 0:
            result = '. '.join([l.rstrip('.') for l in unique_lines[:5] if l])
        
        return result
    
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
        
        system_patterns = [
            'Модель B:', 'Модель A:', 'Модель C:', 
            'Model B:', 'Model A:', 'Model C:',
            'модель B:', 'модель A:', 'модель C:'
        ]
        
        answer_patterns = [
            'Ответ модели B:', 'Ответ модели A:', 'Ответ модели C:',
            'ответ модели B:', 'ответ модели A:', 'ответ модели C:',
            'ответ модели:', 'Ответ модели:'
        ]
        
        lines = text.split('\n')
        cleaned_lines = []
        stop_processing = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            for pattern in answer_patterns:
                if pattern in line:
                    stop_processing = True
                    break
            
            if stop_processing:
                break
            
            for pattern in system_patterns:
                if line.startswith(pattern):
                    continue
                line = line.replace(pattern, '')
            
            if line.strip():
                cleaned_lines.append(line.strip())
        
        result = '\n'.join(cleaned_lines[:10])
        return result


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
        extended_max_tokens: int = 4096,
        extended_temperature: float = 0.35,
        extended_repeat_penalty: float = 1.8
    ):
        self.condensed = CondensedGenerator(
            llama_model=llama_condensed,
            graph=graph,
            max_tokens=condensed_max_tokens
        )
        
        self.extended = ExtendedGenerator(
            llama_model=llama_extended,
            graph=graph,
            max_tokens=extended_max_tokens,
            temperature=extended_temperature,
            repeat_penalty=extended_repeat_penalty
        )
        
        self.graph = graph
        logger.info(f"DualGenerator инициализирован: condensed={condensed_max_tokens}, extended={extended_max_tokens}")
    
    def generate_condensed(self, query: str) -> Dict[str, Any]:
        """Генерация краткого ответа."""
        start = time.time()
        response = self.condensed.generate(query)
        elapsed = time.time() - start
        
        return {
            'response': response,
            'mode': 'condensed',
            'time': elapsed,
            'length': len(response),
            'tokens_estimate': len(response.split())
        }
    
    def generate_extended(self, query: str) -> Dict[str, Any]:
        """Генерация развёрнутого ответа."""
        start = time.time()
        response = self.extended.generate(query)
        elapsed = time.time() - start
        
        return {
            'response': response,
            'mode': 'extended',
            'time': elapsed,
            'length': len(response),
            'tokens_estimate': len(response.split())
        }
    
    def generate(
        self,
        query: str,
        mode: str = "auto",
        return_details: bool = False
    ) -> Any:
        """
        Умная генерация.
        
        Args:
            query: Текст запроса
            mode: 'condensed', 'extended', 'auto'
                - 'condensed': всегда краткий
                - 'extended': всегда развёрнутый
                - 'auto': определяет по ключевым словам
            return_details: возвращать детали (Dict) или только response (str)
        """
        if mode == "condensed":
            result = self.generate_condensed(query)
        elif mode == "extended":
            result = self.generate_extended(query)
        else:
            result = self._auto_generate(query)
        
        if return_details:
            return result
        return result.get('response', result) if isinstance(result, dict) else result
    
    def _auto_generate(self, query: str) -> Dict[str, Any]:
        """Автоматическое определение режима."""
        query_lower = query.lower()
        
        brief_keywords = ['кратко', 'вкратце', 'суть', 'кто такой', 'перечисли', 'назови']
        for kw in brief_keywords:
            if kw in query_lower:
                result = self.generate_condensed(query)
                result['auto_selected'] = True
                return result
        
        result = self.generate_extended(query)
        result['auto_selected'] = True
        return result
    
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
