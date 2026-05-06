"""
ChunkedContextProcessor - Разбиение большого контекста на чанки для поэтапной обработки.
"""
import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ContextChunk:
    """Чанк контекста с метаданными."""
    id: str
    content: str
    chunk_type: str  # 'paragraph', 'list', 'code', 'header', 'conversation'
    relevance_score: float = 0.5
    position: int = 0  # Позиция в исходном тексте
    keywords: List[str] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
    
    def __len__(self) -> int:
        return len(self.content)
    
    def to_prompt(self, include_marker: bool = True) -> str:
        """Форматировать чанк для промпта."""
        if include_marker:
            return f"[Чанк {self.id}] {self.content}"
        return self.content


class ChunkedContextProcessor:
    """
    Процессор контекста с чанкованием и приоритизацией.
    
    Разбивает большой контекст на семантические чанки,
    маркирует их и позволяет обрабатывать поэтапно.
    """
    
    def __init__(
        self,
        max_chunk_size: int = 500,
        overlap_tokens: int = 50,
        min_chunk_size: int = 100,
        enable_semantic_split: bool = True
    ):
        self.max_chunk_size = max_chunk_size
        self.overlap_tokens = overlap_tokens
        self.min_chunk_size = min_chunk_size
        self.enable_semantic_split = enable_semantic_split
    
    def process(self, context: str, query: str = "") -> List[ContextChunk]:
        """
        Обработать контекст и разбить на чанки.
        
        Args:
            context: Исходный контекст
            query: Запрос для оценки релевантности
            
        Returns:
            Список чанков с метаданными
        """
        if not context or len(context.strip()) < 50:
            return []
        
        chunks = []
        
        # 1. Семантическое разбиение
        if self.enable_semantic_split:
            semantic_chunks = self._semantic_split(context)
        else:
            semantic_chunks = self._simple_split(context)
        
        # 2. Оценка релевантности и маркировка
        for i, chunk_content in enumerate(semantic_chunks):
            chunk_type = self._detect_chunk_type(chunk_content)
            keywords = self._extract_keywords(chunk_content)
            relevance = self._calculate_relevance(chunk_content, query)
            
            chunk = ContextChunk(
                id=f"chunk_{i+1}",
                content=chunk_content.strip(),
                chunk_type=chunk_type,
                relevance_score=relevance,
                position=i,
                keywords=keywords
            )
            chunks.append(chunk)
        
        # 3. Сортировка по релевантности
        chunks.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return chunks
    
    def _semantic_split(self, text: str) -> List[str]:
        """Семантическое разбиение по абзацам, спискам, коду."""
        chunks = []
        
        # Разбиваем по двойным переносам строк (абзацы)
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_size = len(para)
            
            # Определяем тип параграфа
            is_list = bool(re.match(r'^[\d•\-\*]+[\s\)」\.]', para))
            is_code = bool(re.match(r'^```|`', para))
            is_header = bool(re.match(r'^#{1,6}\s', para))
            
            # Особый размер для разных типов
            type_multiplier = 1.5 if (is_list or is_code) else 1.0
            adjusted_size = para_size * type_multiplier
            
            # Если чанк слишком большой - разбиваем further
            if adjusted_size > self.max_chunk_size * 1.5:
                # Разбиваем по предложениям
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    if current_size + len(sentence) > self.max_chunk_size and current_chunk:
                        chunks.append('\n'.join(current_chunk))
                        current_chunk = [sentence]
                        current_size = len(sentence)
                    else:
                        current_chunk.append(sentence)
                        current_size += len(sentence) + 1
                continue
            
            # Добавляем параграф к текущему чанку
            if current_size + para_size > self.max_chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size + 1
        
        # Добавляем последний чанк
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def _simple_split(self, text: str) -> List[str]:
        """Простое разбиение по токенам с перекрытием."""
        words = text.split()
        chunks = []
        
        start = 0
        while start < len(words):
            end = min(start + self.max_chunk_size // 4, len(words))
            chunk = ' '.join(words[start:end])
            
            if len(chunk) >= self.min_chunk_size:
                chunks.append(chunk)
            
            start = end - self.overlap_tokens // 4
            if start >= end:
                start = end
        
        return chunks
    
    def _detect_chunk_type(self, chunk: str) -> str:
        """Определить тип чанка."""
        chunk = chunk.strip()
        
        if re.match(r'^```|`', chunk):
            return 'code'
        if re.match(r'^#{1,6}\s', chunk):
            return 'header'
        if re.match(r'^[\d•\-\*]+[\s\)」\.]', chunk):
            return 'list'
        if re.match(r'^(В|Q:|A:|Вопрос:|Ответ:)', chunk, re.IGNORECASE):
            return 'conversation'
        if re.match(r'^\([^\)]+\)\s*:', chunk):
            return 'dialogue'
        
        return 'paragraph'
    
    def _extract_keywords(self, chunk: str) -> List[str]:
        """Извлечь ключевые слова из чанка."""
        # Убираем стоп-слова
        stop_words = {
            'это', 'что', 'как', 'и', 'в', 'на', 'с', 'по', 'для', 'к', 'о',
            'из', 'за', 'от', 'до', 'при', 'так', 'то', 'не', 'но', 'да', 'или',
            'а', 'же', 'ли', 'быть', 'был', 'была', 'были', 'будет', 'есть'
        }
        
        # Извлекаем слова
        words = re.findall(r'\b[а-яА-Яa-zA-Z]{4,}\b', chunk.lower())
        keywords = [w for w in words if w not in stop_words]
        
        # Берем top-5 по частоте
        from collections import Counter
        counter = Counter(keywords)
        return [w for w, _ in counter.most_common(5)]
    
    def _calculate_relevance(self, chunk: str, query: str) -> float:
        """Рассчитать релевантность чанка к запросу."""
        if not query:
            return 0.5
        
        query_words = set(query.lower().split())
        chunk_words = set(re.findall(r'\b[а-яА-Яa-zA-Z]{3,}\b', chunk.lower()))
        
        if not query_words:
            return 0.5
        
        # Пересечение
        intersection = query_words & chunk_words
        if not intersection:
            return 0.1
        
        # Jaccard similarity
        union = query_words | chunk_words
        relevance = len(intersection) / len(union)
        
        # Бонус за точное совпадение фраз
        query_lower = query.lower()
        if query_lower in chunk.lower():
            relevance += 0.3
        
        return min(relevance + 0.2, 1.0)
    
    def format_for_prompt(
        self,
        chunks: List[ContextChunk],
        max_chunks: int = 5,
        include_markers: bool = True
    ) -> str:
        """
        Форматировать чанки для промпта.
        
        Args:
            chunks: Список чанков
            max_chunks: Максимум чанков для включения
            include_markers: Включать ли маркеры [Чанк N]
            
        Returns:
            Отформатированная строка для промпта
        """
        if not chunks:
            return ""
        
        # Берем топ чанков
        top_chunks = sorted(chunks, key=lambda x: x.relevance_score, reverse=True)[:max_chunks]
        
        # Сортируем по позиции для сохранения порядка
        top_chunks.sort(key=lambda x: x.position)
        
        parts = []
        for chunk in top_chunks:
            marker = f"[{chunk.chunk_type.upper()}] " if include_markers else ""
            parts.append(f"{marker}{chunk.content}")
        
        return '\n\n---\n\n'.join(parts)
    
    def get_chunk_summary(self, chunks: List[ContextChunk]) -> str:
        """Получить краткую сводку о чанках."""
        if not chunks:
            return "Пустой контекст"
        
        type_counts = {}
        for chunk in chunks:
            type_counts[chunk.chunk_type] = type_counts.get(chunk.chunk_type, 0) + 1
        
        types_str = ', '.join([f"{t}: {c}" for t, c in sorted(type_counts.items())])
        
        total_size = sum(len(c) for c in chunks)
        avg_relevance = sum(c.relevance_score for c in chunks) / len(chunks)
        
        return f"{len(chunks)} чанков ({types_str}), {total_size} символов, релевантность: {avg_relevance:.2f}"


class StreamingGenerator:
    """
    Генератор с постепенной выдачей токенов.
    
    Позволяет получать текст чанками по мере генерации,
    что даёт более натуральный UX.
    """
    
    def __init__(
        self,
        generator,
        chunk_size: int = 50,
        min_pause_chars: int = 100,
        paragraph_delimiters: str = '.!?\n'
    ):
        """
        Args:
            generator: Базовый генератор с методом generate()
            chunk_size: Размер чанка для выдачи (символов)
            min_pause_chars: Минимум символов перед паузой
            paragraph_delimiters: Разделители абзацев
        """
        self.generator = generator
        self.chunk_size = chunk_size
        self.min_pause_chars = min_pause_chars
        self.paragraph_delimiters = paragraph_delimiters
    
    def generate_streaming(
        self,
        query: str,
        context: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7
    ):
        """
        Генерация с постепенной выдачей.
        
        Yields:
            Dict с ключами: 'type', 'text', 'is_final', 'tokens_count', 'elapsed_ms'
        """
        import time
        from typing import Generator
        
        start_time = time.time()
        full_text = ""
        buffer = ""
        tokens_count = 0
        
        # 1. Получаем полный ответ
        result = self.generator.generate(
            query=query,
            context=context,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        if not result or not result.text:
            yield {
                'type': 'error',
                'text': 'Empty response',
                'is_final': True,
                'tokens_count': 0,
                'elapsed_ms': int((time.time() - start_time) * 1000)
            }
            return
        
        full_text = result.text
        tokens_count = result.tokens_generated or len(full_text.split())
        
        # 2. Разбиваем на чанки для стриминга
        chunks = self._split_into_chunks(full_text)
        
        for i, chunk in enumerate(chunks):
            is_final = (i == len(chunks) - 1)
            
            yield {
                'type': 'chunk' if not is_final else 'complete',
                'text': chunk,
                'is_final': is_final,
                'tokens_count': tokens_count,
                'elapsed_ms': int((time.time() - start_time) * 1000),
                'chunk_index': i,
                'total_chunks': len(chunks)
            }
            
            # Небольшая задержка для естественности (5-20ms)
            if not is_final:
                time.sleep(0.01)
    
    def _split_into_chunks(self, text: str) -> List[str]:
        """Разбить текст на чанки для постепенной выдачи."""
        if len(text) <= self.chunk_size:
            return [text] if text else []
        
        chunks = []
        remaining = text
        
        while remaining:
            # Ищем точку паузы
            chunk_end = min(self.chunk_size, len(remaining))
            
            if chunk_end < len(remaining):
                # Ищем последний разделитель
                for delimiter in self.paragraph_delimiters:
                    last_delimiter = remaining[:chunk_end].rfind(delimiter)
                    if last_delimiter > self.min_pause_chars:
                        chunk_end = last_delimiter + 1
                        break
            
            chunk = remaining[:chunk_end].strip()
            if chunk:
                chunks.append(chunk)
            
            remaining = remaining[chunk_end:]
        
        return chunks if chunks else [text]
    
    def generate_with_fractal_context(
        self,
        query: str,
        fractal_graph,
        hybrid_cache,
        max_context_chunks: int = 3,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ):
        """
        Генерация с фрактальным контекстом из чанков.
        
        Обрабатывает контекст частями, начиная с самых релевантных чанков.
        """
        import time
        from typing import List
        
        start_time = time.time()
        
        # 1. Получаем контекст
        context_text = self._build_context(query, fractal_graph, hybrid_cache)
        
        # 2. Чанкуем контекст
        processor = ChunkedContextProcessor()
        chunks = processor.process(context_text, query)
        
        if not chunks:
            # Нет контекста - обычная генерация
            return list(self.generate_streaming(query, "", max_tokens, temperature))
        
        # 3. Обрабатываем чанки поэтапно
        all_responses = []
        used_chunks = []
        
        for i in range(min(max_context_chunks, len(chunks))):
            chunk = chunks[i]
            used_chunks.append(chunk)
            
            # Добавляем контекст от предыдущих чанков
            previous_context = '\n\n'.join(c.content for c in used_chunks[:-1])
            current_context = chunk.content
            
            # Формируем расширенный промпт
            extended_prompt = self._build_prompt_with_chunk(
                query, current_context, previous_context
            )
            
            # Генерируем для этого чанка
            result = self.generator.generate(
                query=extended_prompt,
                context=previous_context,
                max_tokens=max_tokens // max_context_chunks,
                temperature=temperature
            )
            
            if result and result.text:
                all_responses.append({
                    'chunk_id': chunk.id,
                    'text': result.text,
                    'relevance': chunk.relevance_score,
                    'chunk_type': chunk.chunk_type
                })
        
        # 4. Синтезируем финальный ответ
        final_text = self._synthesize_responses(all_responses, query)
        
        # 5. Стримим как обычно
        return list(self.generate_streaming(
            query=f"Резюме: {final_text[:500]}",
            context="",
            max_tokens=max_tokens,
            temperature=temperature
        ))
    
    def _build_context(
        self,
        query: str,
        fractal_graph,
        hybrid_cache
    ) -> str:
        """Построить контекст из графов и кэша."""
        contexts = []
        
        # FractalGraph
        if fractal_graph:
            try:
                if hasattr(fractal_graph, 'semantic_search'):
                    results = fractal_graph.semantic_search(query, top_k=10)
                    for r in results[:5]:
                        content = r.get('content', '')
                        if content:
                            contexts.append(content)
            except:
                pass
        
        # HybridCache
        if hybrid_cache:
            try:
                if hasattr(hybrid_cache, 'search'):
                    results = hybrid_cache.search(query, top_k=5)
                    for r in results:
                        if isinstance(r, dict):
                            text = r.get('text', '') or r.get('content', '')
                        else:
                            text = str(r)
                        if text:
                            contexts.append(text)
            except:
                pass
        
        return '\n\n'.join(contexts)
    
    def _build_prompt_with_chunk(
        self,
        query: str,
        current_context: str,
        previous_context: str
    ) -> str:
        """Построить промпт с текущим чанком контекста."""
        prompt_parts = [
            f"Вопрос: {query}",
            "",
            "Текущий контекст:",
            current_context
        ]
        
        if previous_context:
            prompt_parts.extend([
                "",
                "Предыдущий анализ:",
                previous_context[:1000]
            ])
        
        return '\n'.join(prompt_parts)
    
    def _synthesize_responses(
        self,
        responses: List[Dict],
        query: str
    ) -> str:
        """Синтезировать финальный ответ из частичных."""
        if not responses:
            return ""
        
        if len(responses) == 1:
            return responses[0]['text']
        
        # Сортируем по релевантности
        responses.sort(key=lambda x: x['relevance'], reverse=True)
        
        # Берем самый релевантный как основу
        main_response = responses[0]['text']
        
        # Дополняем из других
        additions = []
        for r in responses[1:]:
            if r['text'] not in main_response:
                # Извлекаем уникальные части
                sentences = r['text'].split('. ')
                for sent in sentences[:2]:  # Берём первые 2 предложения
                    if len(sent) > 20:
                        additions.append(sent.strip())
        
        if additions:
            main_response += "\n\n" + ". ".join(additions[:3])
        
        return main_response
