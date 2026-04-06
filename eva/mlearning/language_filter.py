"""
Language Filter Tokenizer - обёртка-фильтр для токенизатора с поддержкой:
- Фильтрации иностранных языков (китайский, английский и др.)
- Динамического переключения режимов
- Поддержки русского языка по умолчанию
"""

import re
import logging
from typing import List, Optional, Set, Union, Dict, Any

logger = logging.getLogger(__name__)


class LanguageFilterTokenizer:
    """
    Обёртка-фильтр для токенизатора, фильтрующая нежелательные языки/токены.
    Может работать в нескольких режимах:
    - 'russian_only': только русский язык
    - 'no_chinese': без китайских иероглифов  
    - 'no_foreign': без иностранных символов (латиница)
    - 'full': без фильтрации
    """
    
    RUSSIAN_CHARS = re.compile(r'[а-яА-ЯёЁ]+')
    LATIN_CHARS = re.compile(r'[a-zA-Z]+')
    CHINESE_CHARS = re.compile(r'[\u4e00-\u9fff]')
    JAPANESE_CHARS = re.compile(r'[\u3040-\u309f\u30a0-\u30ff]')
    KOREAN_CHARS = re.compile(r'[\uac00-\ud7af]')
    
    def __init__(
        self,
        base_tokenizer: Any,
        mode: str = 'russian_only',
        custom_blocked_patterns: Optional[List[str]] = None
    ):
        self.base_tokenizer = base_tokenizer
        self.mode = mode
        self.custom_blocked_patterns = custom_blocked_patterns or []
        self._blocked_tokens: Set[str] = set()
        self._update_blocked_tokens()
        
        logger.info(f"LanguageFilterTokenizer инициализирован в режиме: {mode}")
    
    def _update_blocked_tokens(self):
        """Обновляет набор блокируемых токенов на основе режима."""
        self._blocked_tokens.clear()
        
        if self.mode == 'russian_only':
            self._blocked_tokens.update([' китайск', ' chinese', ' english', ' japan'])
        elif self.mode == 'no_chinese':
            self._blocked_tokens.update([' китайск', ' chinese'])
        elif self.mode == 'no_foreign':
            self._blocked_tokens.update([' chinese', ' english', ' japan', ' korean'])
    
    def set_mode(self, mode: str):
        """Динамическое переключение режима работы."""
        valid_modes = ['russian_only', 'no_chinese', 'no_foreign', 'full']
        if mode not in valid_modes:
            logger.warning(f"Неизвестный режим: {mode}, используем 'full'")
            mode = 'full'
        
        self.mode = mode
        self._update_blocked_tokens()
        logger.info(f"Переключен режим LanguageFilterTokenizer: {mode}")
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        **kwargs
    ) -> List[int]:
        """Токенизация с фильтрацией входного текста."""
        filtered_text = self._filter_input_text(text)
        return self.base_tokenizer.encode(filtered_text, add_special_tokens, **kwargs)
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
        **kwargs
    ) -> str:
        """Декодирование токенов в текст."""
        return self.base_tokenizer.decode(token_ids, skip_special_tokens, **kwargs)
    
    def _filter_input_text(self, text: str) -> str:
        """Фильтрация входного текста на основе режима."""
        if self.mode == 'full':
            return text
        
        filtered = text
        
        if self.mode == 'russian_only':
            filtered = self._keep_only_russian(filtered)
        elif self.mode == 'no_chinese':
            filtered = self._remove_chinese(filtered)
        elif self.mode == 'no_foreign':
            filtered = self._remove_foreign_latin(filtered)
        
        for pattern in self.custom_blocked_patterns:
            filtered = re.sub(pattern, '', filtered, flags=re.IGNORECASE)
        
        return filtered
    
    def _keep_only_russian(self, text: str) -> str:
        """Оставляет только русские символы."""
        russian_words = self.RUSSIAN_CHARS.findall(text)
        return ' '.join(russian_words)
    
    def _remove_chinese(self, text: str) -> str:
        """Удаляет китайские, японские, корейские символы."""
        text = self.CHINESE_CHARS.sub('', text)
        text = self.JAPANESE_CHARS.sub('', text)
        text = self.KOREAN_CHARS.sub('', text)
        return text
    
    def _remove_foreign_latin(self, text: str) -> str:
        """Удаляет латинские буквы (кроме случаев в коде)."""
        lines = text.split('\n')
        result_lines = []
        
        for line in lines:
            if any(kw in line.lower() for kw in ['def ', 'class ', 'import ', 'function ', 'const ', 'var ', 'let ']):
                result_lines.append(line)
            else:
                result_lines.append(self.LATIN_CHARS.sub('', line))
        
        return '\n'.join(result_lines)
    
    def filter_output(self, text: str) -> str:
        """Фильтрация выходного текста (после генерации модели)."""
        if self.mode == 'full':
            return text
        
        filtered = text
        
        if self.mode in ['russian_only', 'no_foreign']:
            filtered = self._remove_foreign_latin(filtered)
        
        if self.mode in ['russian_only', 'no_chinese']:
            filtered = self._remove_chinese(filtered)
        
        return filtered
    
    def __getattr__(self, name: str):
        """Делегирование не переопределённых методов базовому токенизатору."""
        return getattr(self.base_tokenizer, name)


class DynamicQuantizationManager:
    """
    Управление динамическим переключением режимов квантования.
    Под контролем CoreBrain.
    """
    
    QUANTIZATION_MODES = {
        'q4_k_m': {'memory_factor': 1.0, 'quality': 1.0, 'description': 'Q4_K_M - баланс'},
        'q5_k_m': {'memory_factor': 1.25, 'quality': 1.15, 'description': 'Q5_K_M - выше качество'},
        'q2_k': {'memory_factor': 0.5, 'quality': 0.7, 'description': 'Q2_K - экономия памяти'},
        'q8_0': {'memory_factor': 2.0, 'quality': 1.3, 'description': 'Q8_0 - максимальное качество'},
    }
    
    def __init__(self, brain=None):
        self.brain = brain
        self.current_mode = 'q4_k_m'
        self.auto_switch_enabled = True
        self.memory_threshold_gb = 2.0
        
        logger.info("DynamicQuantizationManager инициализирован")
    
    def set_mode(self, mode: str):
        """Установить режим квантования."""
        if mode not in self.QUANTIZATION_MODES:
            logger.warning(f"Неизвестный режим квантования: {mode}")
            return False
        
        self.current_mode = mode
        logger.info(f"Режим квантования изменён на: {mode}")
        
        if self.brain and hasattr(self.brain, '_publish_event'):
            self.brain._publish_event('quantization.mode_changed', {
                'mode': mode,
                'description': self.QUANTIZATION_MODES[mode]['description']
            })
        
        return True
    
    def check_memory_and_switch(self, available_memory_gb: float):
        """Автоматическое переключение на основе доступной памяти."""
        if not self.auto_switch_enabled:
            return
        
        if available_memory_gb < self.memory_threshold_gb:
            if self.current_mode != 'q2_k':
                logger.warning(f"Мало памяти ({available_memory_gb:.1f}GB), переключаю на Q2_K")
                self.set_mode('q2_k')
        else:
            if self.current_mode == 'q2_k':
                logger.info("Память доступна, возвращаюсь на Q4_K_M")
                self.set_mode('q4_k_m')
    
    def get_current_config(self) -> Dict[str, Any]:
        """Получить текущую конфигурацию."""
        return {
            'mode': self.current_mode,
            **self.QUANTIZATION_MODES[self.current_mode]
        }


class ModelModeController:
    """
    Центральный контроллер режимов модели под управлением CoreBrain.
    Объединяет LanguageFilterTokenizer и DynamicQuantizationManager.
    """
    
    def __init__(self, brain=None):
        self.brain = brain
        self.language_filter = None
        self.quantization_manager = DynamicQuantizationManager(brain)
        self.current_language_mode = 'russian_only'
        
        logger.info("ModelModeController инициализирован")
    
    def initialize_with_tokenizer(self, tokenizer: Any):
        """Инициализация с базовым токенизатором."""
        self.language_filter = LanguageFilterTokenizer(
            base_tokenizer=tokenizer,
            mode=self.current_language_mode
        )
        logger.info("ModelModeController настроен с токенизатором")
    
    def set_language_mode(self, mode: str):
        """Установить режим языка (russian_only, no_chinese, no_foreign, full)."""
        self.current_language_mode = mode
        if self.language_filter:
            self.language_filter.set_mode(mode)
        
        logger.info(f"Языковой режим: {mode}")
        
        if self.brain and hasattr(self.brain, '_publish_event'):
            self.brain._publish_event('model.mode_changed', {
                'language_mode': mode,
                'quantization_mode': self.quantization_manager.current_mode
            })
    
    def set_quantization_mode(self, mode: str):
        """Установить режим квантования."""
        self.quantization_manager.set_mode(mode)
    
    def get_status(self) -> Dict[str, Any]:
        """Получить статус контроллера."""
        return {
            'language_mode': self.current_language_mode,
            'quantization': self.quantization_manager.get_current_config(),
            'filter_active': self.language_filter is not None
        }
    
    def process_input(self, text: str) -> str:
        """Обработка входного текста через фильтр."""
        if self.language_filter:
            return self.language_filter._filter_input_text(text)
        return text
    
    def process_output(self, text: str) -> str:
        """Обработка выходного текста через фильтр."""
        if self.language_filter:
            return self.language_filter.filter_output(text)
        return text