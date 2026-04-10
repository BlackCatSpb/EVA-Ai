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
        temperature: float = 0.1,
        repeat_penalty: float = 1.8
    ):
        self.llama = llama_model
        self.graph = graph
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.repeat_penalty = repeat_penalty
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
                repeat_penalty=self.repeat_penalty,
                stop=["</s>", "User:", "user:", "Human:", "Assistant:", "System:"],
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
            'модель B:', 'модель A:', 'модель C:',
            '(Model B)', '(Model A)', '(Model C)',
            'модель B развивает', 'ответ модели B развивает',
            'В контексте развития мысли (Model B)',
            'Это позволяет мне', 'Это позволяет',
        ]
        
        for pattern in system_patterns:
            text = text.replace(pattern, '')
        
        # Remove repetition patterns
        lines = text.split('\n')
        unique_lines = []
        seen = set()
        for line in lines:
            line_lower = line.lower().strip()
            if line_lower and line_lower not in seen:
                seen.add(line_lower)
                unique_lines.append(line)
            elif not line.strip():
                unique_lines.append(line)
        text = '\n'.join(unique_lines)
        
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
                stop=["</s>", "User:", "user:", "Human:", "Assistant:", "System:"],
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
    
    def generate_chunked(
        self,
        query: str,
        context: str = "",
        max_total_tokens: int = 4096,
        chunk_size: int = 1024,
        max_chunks: int = 4
    ) -> Dict[str, Any]:
        """
        Генерация большого ответа блоками с использованием стоп-токенов.
        
        Args:
            query: Запрос пользователя
            context: Контекст для генерации
            max_total_tokens: Максимальное количество токенов всего
            chunk_size: Размер одного блока (токенов)
            max_chunks: Максимальное количество блоков
            
        Returns:
            Dict с полями:
                - text: полный текст ответа
                - chunks: список отдельных блоков
                - total_tokens: общее количество токенов
                - chunk_count: количество блоков
                - generation_time: время генерации
        """
        start_time = time.time()
        chunks = []
        total_tokens = 0
        
        # Подготавливаем начальный промпт
        graph_context = self._get_context(query) if self.graph else context
        base_prompt = EXTENDED_PROMPT.format(query=query, graph_context=graph_context)
        
        # Первый блок с маркером начала
        current_prompt = f"{base_prompt}\n\n[CHUNK_START]"
        
        logger.info(f"Начата chunked генерация: max_total={max_total_tokens}, chunk_size={chunk_size}")
        
        for chunk_num in range(max_chunks):
            try:
                # Генерируем блок до стоп-токена
                output = self.llama(
                    current_prompt,
                    max_tokens=min(chunk_size, max_total_tokens - total_tokens),
                    temperature=self.temperature,
                    repeat_penalty=self.repeat_penalty,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    stop=["[CHUNK_BREAK]", "[CHUNK_END]", "</s>"],
                    echo=False
                )
                
                # Извлекаем текст
                chunk_text = ""
                if isinstance(output, dict):
                    chunk_text = output.get('choices', [{}])[0].get('text', '')
                else:
                    chunk_text = str(output)
                
                # Очищаем от маркеров
                chunk_text = chunk_text.replace("[CHUNK_START]", "").replace("[CHUNK_BREAK]", "").strip()
                
                if not chunk_text or len(chunk_text) < 20:
                    logger.info(f"Блок {chunk_num+1} пустой, завершаем генерацию")
                    break
                
                # Удаляем повторы с предыдущими блоками
                chunk_text = self._deduplicate_chunk(chunk_text, chunks)
                
                chunks.append(chunk_text)
                total_tokens += len(chunk_text.split())
                
                logger.info(f"Сгенерирован блок {chunk_num+1}: {len(chunk_text)} символов, {len(chunk_text.split())} токенов")
                
                # Проверяем условия остановки
                if total_tokens >= max_total_tokens:
                    logger.info(f"Достигнут лимит токенов: {total_tokens}")
                    break
                
                # Проверяем на естественное завершение
                if any(marker in chunk_text.lower() for marker in 
                       ["в заключение", "итак,", "таким образом", "подведем итог", "вывод:"]):
                    logger.info(f"Обнаружено естественное завершение в блоке {chunk_num+1}")
                    break
                
                # Готовим промпт для следующего блока
                context_summary = self._create_context_summary(chunks)
                current_prompt = (
                    f"Продолжи развёрнутый ответ на вопрос, сохраняя связность с предыдущим текстом.\n\n"
                    f"Вопрос: {query}\n\n"
                    f"Уже написано ({len(context_summary)} символов):\n{context_summary}\n\n"
                    f"[CHUNK_CONTINUE]"
                )
                
            except Exception as e:
                logger.error(f"Ошибка генерации блока {chunk_num+1}: {e}")
                break
        
        # Собираем полный текст
        full_text = "\n\n".join(chunks)
        
        elapsed = time.time() - start_time
        
        result = {
            'text': full_text,
            'chunks': chunks,
            'total_tokens': total_tokens,
            'chunk_count': len(chunks),
            'generation_time': elapsed,
            'tokens_per_second': total_tokens / elapsed if elapsed > 0 else 0
        }
        
        logger.info(f"Chunked генерация завершена: {len(chunks)} блоков, {total_tokens} токенов, {elapsed:.2f}с")
        
        return result
    
    def _deduplicate_chunk(self, chunk: str, previous_chunks: List[str]) -> str:
        """Удаляет дублирующийся контент с предыдущими блоками."""
        if not previous_chunks:
            return chunk
        
        # Берём последние 200 символов предыдущего блока
        prev_end = previous_chunks[-1][-200:].lower()
        chunk_start = chunk[:200].lower()
        
        # Ищем пересечение
        for length in range(min(len(chunk_start), 100), 20, -1):
            if prev_end.endswith(chunk_start[:length]):
                # Нашли пересечение, удаляем его из начала текущего блока
                logger.debug(f"Удалено пересечение длиной {length} символов")
                return chunk[length:].strip()
        
        return chunk
    
    def _create_context_summary(self, chunks: List[str], max_length: int = 500) -> str:
        """Создаёт краткое резюме предыдущих блоков для контекста."""
        if not chunks:
            return ""
        
        # Берём последние 2 блока
        recent = chunks[-2:]
        summary_parts = []
        
        for i, chunk in enumerate(recent):
            # Извлекаем первое и последнее предложение
            sentences = chunk.split('.')
            if len(sentences) >= 2:
                summary_parts.append(f"[Блок {len(chunks)-len(recent)+i+1}] {sentences[0]}. ... {sentences[-2]}.")
            else:
                summary_parts.append(f"[Блок {len(chunks)-len(recent)+i+1}] {chunk[:150]}...")
        
        summary = " ".join(summary_parts)
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        
        return summary
    
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
        """Получить релевантный контекст из графа с динамическим размером и компактификацией."""
        if not self.graph:
            return "Нет контекста"
        
        # Проверяем, есть ли уже скомпактированный контекст для этого запроса
        cache_key = f"compacted_context:query_{hash(query) % 10000000}"
        if hasattr(self, 'brain') and self.brain and hasattr(self.brain, 'hybrid_cache'):
            try:
                cached = self.brain.hybrid_cache.get(cache_key)
                if cached and cached.get('compacted_context'):
                    logger.debug(f"Использован кэшированный компактный контекст для: {query[:50]}...")
                    return cached['compacted_context']
            except:
                pass
        
        # Параметры контекста
        MAX_CONTEXT_CHARS = 20000
        TARGET_COMPACTED_SIZE = 2000  # Целевой размер после компактификации
        AVG_NODE_CHARS = 150
        ESTIMATED_TOKENS_PER_CHAR = 0.25
        
        query_tokens = len(query) * ESTIMATED_TOKENS_PER_CHAR
        available_tokens = min(6000 - query_tokens, MAX_CONTEXT_CHARS * ESTIMATED_TOKENS_PER_CHAR)
        max_nodes = min(int(available_tokens / AVG_NODE_CHARS), 50)
        max_nodes = max(max_nodes, 5)
        
        # Собираем контекстные элементы
        context_items = []
        
        # Используем semantic_search если доступен
        if hasattr(self.graph, 'semantic_search'):
            try:
                results = self.graph.semantic_search(query, top_k=max_nodes)
                if results:
                    for r in results:
                        content = r.get('content', '')
                        if content:
                            context_items.append({
                                'content': content,
                                'type': r.get('type', 'unknown'),
                                'score': r.get('score', 0)
                            })
            except Exception as e:
                logger.debug(f"Semantic search error: {e}")
        
        # Fallback на простой поиск если ничего не нашли
        if not context_items:
            query_words = query.lower().split()[:5]
            for node_id, node in list(getattr(self.graph, 'nodes', {}).items())[:100]:
                content = getattr(node, 'content', '')
                if content:
                    matches = sum(1 for w in query_words if w in content.lower())
                    if matches > 0:
                        context_items.append({
                            'content': content,
                            'type': getattr(node, 'node_type', 'unknown'),
                            'score': matches
                        })
        
        # Компактифицируем контекст если он большой
        total_size = sum(len(str(item.get('content', ''))) for item in context_items)
        if total_size > TARGET_COMPACTED_SIZE * 1.5 and context_items:
            # Вызываем компактификацию через SelfDialogLearning если доступен
            if hasattr(self, 'brain') and self.brain and hasattr(self.brain, 'self_dialog_learning'):
                try:
                    sdl = self.brain.self_dialog_learning
                    if hasattr(sdl, 'compact_context'):
                        result = sdl.compact_context(
                            context_items,
                            query,
                            target_size=TARGET_COMPACTED_SIZE,
                            method="semantic_extraction"
                        )
                        compacted = result.get('compacted_context', '')
                        
                        # Сохраняем в кэш
                        if hasattr(self.brain, 'hybrid_cache'):
                            self.brain.hybrid_cache.put(cache_key, result, ttl=1800)
                        
                        logger.info(f"Контекст компактифицирован: {result['compression_ratio']:.1%} "
                                   f"(сохранность {result['semantic_preserved']:.1%})")
                        
                        # Также сохраняем сырой контекст для фоновой обработки
                        raw_key = f"raw_context:query_{hash(query) % 10000000}"
                        if not self.brain.hybrid_cache.get(raw_key):
                            self.brain.hybrid_cache.put(raw_key, {
                                'items': context_items,
                                'query': query,
                                'target_size': TARGET_COMPACTED_SIZE
                            }, ttl=3600)
                        
                        return compacted
                except Exception as e:
                    logger.warning(f"Ошибка компактификации контекста: {e}")
        
        # Возвращаем несжатый контекст если компактификация не удалась или не нужна
        if context_items:
            context_parts = []
            current_chars = 0
            for item in context_items:
                content = str(item.get('content', ''))[:300]
                if current_chars + len(content) < MAX_CONTEXT_CHARS:
                    context_parts.append(content)
                    current_chars += len(content)
            return ' | '.join(context_parts)
        
        # Fallback на простой поиск если ничего не сработало
        query_words = query.lower().split()[:5]
        relevant = []
        current_chars = 0
        
        for node_id, node in list(getattr(self.graph, 'nodes', {}).items())[:100]:
            content = getattr(node, 'content', '')
            if content and current_chars < MAX_CONTEXT_CHARS:
                if any(kw in content.lower() for kw in query_words):
                    relevant.append(content[:250])
                    current_chars += len(content)
        
        if relevant:
            return ' | '.join(relevant[:max_nodes])
        return "Нет релевантного контекста"
    
    def _clean_response(self, text: str) -> str:
        """Очистка ответа."""
        text = text.strip()
        
        system_patterns = [
            'Модель B:', 'Модель A:', 'Модель C:', 
            'Model B:', 'Model A:', 'Model C:',
            'модель B:', 'модель A:', 'модель C:',
            '(Model B)', '(Model A)', '(Model C)',
            'модель B развивает', 'ответ модели B развивает',
            'Это позволяет мне', 'Это позволяет',
        ]
        
        answer_patterns = [
            'Ответ модели B:', 'Ответ модели A:', 'Ответ модели C:',
            'ответ модели B:', 'ответ модели A:', 'ответ модели C:',
            'ответ модели:', 'Ответ модели:',
            'В контексте развития мысли (Model B)',
            'модель B действительно', 'ответ модели B действительно',
            'ответ модели B', 'Ответ модели B',
            'Это позволяет мне', 'Это позволяет',
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
            
            should_skip_line = False
            for pattern in system_patterns:
                if line.startswith(pattern) or f'\n{pattern}' in line:
                    should_skip_line = True
                    break
                line = line.replace(pattern, '')
            
            if should_skip_line:
                continue
            
            if line.strip():
                cleaned_lines.append(line.strip())
        
        # Remove duplicate consecutive lines
        result_lines = []
        prev_line = None
        for line in cleaned_lines[:10]:
            if line != prev_line:
                result_lines.append(line)
                prev_line = line
        
        result = '\n'.join(result_lines)
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
        condensed_max_tokens: int = 1024,
        extended_max_tokens: int = 4096,
        extended_temperature: float = 0.35,
        extended_repeat_penalty: float = 1.8,
        brain=None  # Добавляем ссылку на brain для компактификации
    ):
        self.condensed = CondensedGenerator(
            llama_model=llama_condensed,
            graph=graph,
            max_tokens=condensed_max_tokens,
            repeat_penalty=1.8
        )
        
        self.extended = ExtendedGenerator(
            llama_model=llama_extended,
            graph=graph,
            max_tokens=extended_max_tokens,
            temperature=extended_temperature,
            repeat_penalty=extended_repeat_penalty
        )
        
        self.graph = graph
        self.brain = brain  # Сохраняем ссылку на brain
        
        # Передаем brain в под-генераторы для доступа к компактификации
        if hasattr(self.condensed, 'brain'):
            self.condensed.brain = brain
        if hasattr(self.extended, 'brain'):
            self.extended.brain = brain
        
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
            mode: 'condensed', 'extended', 'large', 'auto'
                - 'condensed': всегда краткий (до 1024 токенов)
                - 'extended': всегда развёрнутый (до 4096 токенов)
                - 'large': большой ответ блоками (до 4096+ токенов)
                - 'auto': определяет по ключевым словам
            return_details: возвращать детали (Dict) или только response (str)
        """
        if mode == "condensed":
            result = self.generate_condensed(query)
        elif mode == "extended":
            result = self.generate_extended(query)
        elif mode == "large":
            result = self.generate_large(query)
        else:
            result = self._auto_generate(query)
        
        if return_details:
            return result
        return result.get('response', result) if isinstance(result, dict) else result
    
    def generate_large(self, query: str, context: str = "") -> Dict[str, Any]:
        """
        Генерация большого ответа с использованием chunked generation.
        
        Args:
            query: Запрос пользователя
            context: Дополнительный контекст
            
        Returns:
            Dict с полями:
                - response: полный текст ответа
                - chunks: список блоков
                - total_tokens: общее количество токенов
                - generation_time: время генерации
                - mode: 'large'
        """
        logger.info(f"Начата large генерация для: {query[:50]}...")
        
        # Используем chunked генерацию из ExtendedGenerator
        result = self.extended.generate_chunked(
            query=query,
            context=context,
            max_total_tokens=4096,
            chunk_size=1024,
            max_chunks=4
        )
        
        # Добавляем поле response для совместимости
        result['response'] = result['text']
        result['mode'] = 'large'
        
        return result
    
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
