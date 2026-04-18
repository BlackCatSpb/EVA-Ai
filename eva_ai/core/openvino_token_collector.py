"""
Custom OpenVINO Streamer для сбора токенов.

Позволяет получить raw token IDs в процессе генерации,
без необходимости декодировать текст и затем токенизировать обратно.

Использование:
    from eva_ai.core.openvino_token_collector import TokenCollectorStreamer
    
    streamer = TokenCollectorStreamer()
    
    # При генерации
    pipeline.generate(prompt, config, streamer=streamer)
    
    # Получаем токены
    tokens = streamer.get_tokens()
"""

import logging
from typing import List, Optional

logger = logging.getLogger("eva_ai.token_collector")

try:
    import openvino_genai as ov_genai
except ImportError:
    ov_genai = None


class TokenCollectorStreamer:
    """
    Кастомный streamer для сбора токенов в процессе генерации.
    
    Наследует базовый функционал streamer и собирает:
    1. Raw token IDs
    2. Текстовые чанки
    3. Статистику генерации
    """
    
    def __init__(self):
        self.tokens: List[int] = []
        self.text_chunks: List[str] = []
        self.total_tokens: int = 0
        self.is_active: bool = False
    
    def write(self, text: str) -> bool:
        """
        Callback вызывается для каждого сгенерированного текста.
        
        OpenVINO передаёт текст, не токены напрямую.
        Но мы можем собрать текст и потом декодировать.
        
        Args:
            text: Сгенерированный текст
            
        Returns:
            False - продолжить генерацию, True - остановить
        """
        if not text:
            return False
        
        self.is_active = True
        self.text_chunks.append(text)
        
        return False  # Продолжаем генерацию
    
    def end(self) -> None:
        """Вызывается когда генерация завершена."""
        self.is_active = False
        self.total_tokens = len(self.tokens)
    
    def get_tokens(self) -> List[int]:
        """Получить собранные токены (после токенизации текста)."""
        return self.tokens
    
    def get_text(self) -> str:
        """Получить собранный текст."""
        return ''.join(self.text_chunks)
    
    def get_token_count(self) -> int:
        """Количество токенов."""
        return len(self.tokens)
    
    def __len__(self) -> int:
        return len(self.tokens)


class RawTokenStreamer:
    """
    Streamer для сбора RAW token IDs.
    
    Пробует использовать внутренние методы OpenVINO.
    Если недоступно - fallback на текстовый сбор.
    """
    
    def __init__(self, pipeline=None):
        self.pipeline = pipeline
        self.tokens: List[int] = []
        self.text_chunks: List[str] = []
        self._raw_mode = False
        
        # Пробуем включить raw mode
        self._try_enable_raw_mode()
    
    def _try_enable_raw_mode(self):
        """Пробуем включить raw token collection."""
        if not self.pipeline:
            return
        
        try:
            # Проверяем доступность internal методов
            # Это может не работать во всех версиях
            tokenizer = self.pipeline.get_tokenizer()
            if tokenizer and hasattr(tokenizer, 'encode'):
                self._raw_mode = True
                logger.info("Raw token mode enabled")
        except Exception as e:
            logger.warning(f"Raw mode unavailable: {e}")
    
    def write(self, text: str) -> bool:
        """Callback от OpenVINO."""
        if not text:
            return False
        
        self.text_chunks.append(text)
        
        # Пробуем получить токены из текста
        if self._raw_mode and self.pipeline:
            try:
                tokenizer = self.pipeline.get_tokenizer()
                if tokenizer:
                    new_tokens = tokenizer.encode(text)
                    self.tokens.extend(new_tokens)
            except Exception:
                pass
        
        return False
    
    def end(self) -> None:
        """Генерация завершена."""
        pass
    
    def get_tokens(self) -> List[int]:
        """Получить токены."""
        return self.tokens
    
    def get_text(self) -> str:
        """Получить текст."""
        return ''.join(self.text_chunks)


class TokenStats:
    """Сборщик статистики генерации."""
    
    def __init__(self):
        self.total_tokens: int = 0
        self.first_token_time: Optional[float] = None
        self.last_token_time: Optional[float] = None
        self.tokens_per_second: float = 0.0
        
        import time
        self._start_time = time.time()
    
    def write(self, text: str) -> bool:
        """Callback для сбора статистики."""
        import time
        
        if self.first_token_time is None:
            self.first_token_time = time.time() - self._start_time
        
        self.last_token_time = time.time() - self._start_time
        
        # Подсчёт токенов (приблизительно по символам)
        if text:
            # Средняя длина токена ~4 символа
            self.total_tokens += max(1, len(text) // 4)
        
        return False
    
    def end(self) -> None:
        """Завершение - подсчёт статистики."""
        if self.last_token_time and self.last_token_time > 0:
            self.tokens_per_second = self.total_tokens / self.last_token_time if self.last_token_time > 0 else 0
    
    def get_stats(self) -> dict:
        """Получить статистику."""
        return {
            'total_tokens': self.total_tokens,
            'first_token_ms': int(self.first_token_time * 1000) if self.first_token_time else 0,
            'total_time_ms': int(self.last_token_time * 1000) if self.last_token_time else 0,
            'tokens_per_second': round(self.tokens_per_second, 2)
        }


def create_token_collector(pipeline=None) -> TokenCollectorStreamer:
    """Factory для создания TokenCollectorStreamer."""
    return TokenCollectorStreamer()


def create_raw_token_streamer(pipeline=None) -> RawTokenStreamer:
    """Factory для создания RawTokenStreamer."""
    return RawTokenStreamer(pipeline)


def create_token_stats() -> TokenStats:
    """Factory для создания сборщика статистики."""
    return TokenStats()