"""
BatchSplitter - Разделение промптов на сегменты с маркерами.

Маркеры:
- [SEG_START] - начало сегмента
- [SEG_BREAK] - разделитель между сегментами  
- [SEG_END] - конец сегмента

Используется для:
- Обхода лимитов контекста
- Параллельной обработки на Model B
- Оптимизации CPU загрузки
"""

import re
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


SEG_START = "[SEG_START]"
SEG_BREAK = "[SEG_BREAK]"  
SEG_END = "[SEG_END]"


class BatchSplitter:
    """
    Разбивает большие промпты на меньшие сегменты.
    
    Стратегии:
    - by_tokens: по количеству токенов
    - by_sentences: по предложениям
    - by_paragraphs: по абзацам
    """
    
    def __init__(
        self, 
        max_chunk_tokens: int = 2048,
        strategy: str = "by_sentences",
        tokenizer=None
    ):
        """
        Args:
            max_chunk_tokens: Макс. токенов на чанк
            strategy: Стратегия разделения
            tokenizer: Токенизатор для точного подсчёта
        """
        self.max_chunk_tokens = max_chunk_tokens
        self.strategy = strategy
        self.tokenizer = tokenizer
        
        # Оценка: 1 токен ≈ 4 символа (грубая)
        self.chars_per_token = 4
    
    def split(self, text: str, max_segments: int = 4) -> List[str]:
        """
        Разбивает текст на сегменты.
        
        Args:
            text: Входной текст
            max_segments: Макс. количество сегментов
            
        Returns:
            Список сегментов с маркерами
        """
        if not text or len(text) < 100:
            return [text]
        
        if self.strategy == "by_tokens":
            return self._split_by_tokens(text, max_segments)
        elif self.strategy == "by_sentences":
            return self._split_by_sentences(text, max_segments)
        elif self.strategy == "by_paragraphs":
            return self._split_by_paragraphs(text, max_segments)
        else:
            return [text]
    
    def _split_by_tokens(self, text: str, max_segments: int) -> List[str]:
        """Разделение по токенам."""
        if self.tokenizer:
            tokens = self.tokenizer.encode(text)
            total_tokens = len(tokens)
        else:
            total_tokens = len(text) // self.chars_per_token
        
        if total_tokens <= self.max_chunk_tokens:
            return [text]
        
        segments = []
        segment_size = min(
            self.max_chunk_tokens,
            (total_tokens // max_segments) + 1
        )
        
        if self.tokenizer:
            for i in range(0, total_tokens, segment_size):
                chunk_tokens = tokens[i:i + segment_size]
                chunk_text = self.tokenizer.decode(chunk_tokens)
                segments.append(chunk_text)
        else:
            chars_per_segment = segment_size * self.chars_per_token
            for i in range(0, len(text), chars_per_segment):
                segments.append(text[i:i + chars_per_segment])
        
        return self._add_markers(segments)
    
    def _split_by_sentences(self, text: str, max_segments: int) -> List[str]:
        """Разделение по предложениям."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        segments = []
        current = ""
        current_tokens = 0
        
        for sent in sentences:
            sent_tokens = len(sent) // self.chars_per_token
            
            if current_tokens + sent_tokens > self.max_chunk_tokens and current:
                segments.append(current.strip())
                current = sent
                current_tokens = sent_tokens
            else:
                current += " " + sent if current else sent
                current_tokens += sent_tokens
        
        if current.strip():
            segments.append(current.strip())
        
        if len(segments) > max_segments:
            # Объединяем если слишком много
            merged = []
            for i in range(0, len(segments), 2):
                if i + 1 < len(segments):
                    merged.append(segments[i] + " " + segments[i + 1])
                else:
                    merged.append(segments[i])
            segments = merged[:max_segments]
        
        return self._add_markers(segments)
    
    def _split_by_paragraphs(self, text: str, max_segments: int) -> List[str]:
        """Разделение по абзацам."""
        paragraphs = text.split('\n\n')
        
        segments = []
        current = ""
        
        for para in paragraphs:
            para_tokens = len(para) // self.chars_per_token
            
            if para_tokens > self.max_chunk_tokens:
                if current:
                    segments.append(current)
                segments.extend(self._split_by_sentences(para, max_segments - len(segments)))
                current = ""
            elif (len(current) + len(para)) // self.chars_per_token > self.max_chunk_tokens:
                if current:
                    segments.append(current)
                current = para
            else:
                current += "\n\n" + para if current else para
        
        if current.strip():
            segments.append(current.strip())
        
        return self._add_markers(segments[:max_segments])
    
    def _add_markers(self, segments: List[str]) -> List[str]:
        """Добавляет маркеры к сегментам."""
        if len(segments) <= 1:
            return segments
        
        marked = []
        for i, seg in enumerate(segments):
            if i == 0:
                marked.append(f"{SEG_START}{seg}{SEG_BREAK}")
            elif i == len(segments) - 1:
                marked.append(f"{SEG_END}{seg}")
            else:
                marked.append(f"{SEG_START}{seg}{SEG_BREAK}")
        
        return marked
    
    def assemble(self, responses: List[str]) -> str:
        """
        Собирает ответы, удаляя маркеры.
        
        Args:
            responses: Список ответов от модели
            
        Returns:
            Очищенный текст
        """
        cleaned = []
        
        for resp in responses:
            if isinstance(resp, Exception):
                logger.warning(f"Segment error: {resp}")
                continue
            
            text = str(resp)
            
            text = text.replace(SEG_START, "")
            text = text.replace(SEG_BREAK, " ")
            text = text.replace(SEG_END, "")
            
            cleaned.append(text.strip())
        
        result = " ".join(cleaned)
        
        result = re.sub(r'\s+', ' ', result)
        
        return result.strip()
    
    def needs_splitting(self, text: str) -> bool:
        """Проверяет нужен ли сплит."""
        if not text:
            return False
        
        tokens = len(text) // self.chars_per_token
        if self.tokenizer:
            try:
                tokens = len(self.tokenizer.encode(text))
            except Exception:
                pass
        
        return tokens > self.max_chunk_tokens


def create_batch_splitter(
    max_tokens: int = 2048,
    strategy: str = "by_sentences",
    tokenizer=None
) -> BatchSplitter:
    """Factory."""
    return BatchSplitter(
        max_chunk_tokens=max_tokens,
        strategy=strategy,
        tokenizer=tokenizer
    )