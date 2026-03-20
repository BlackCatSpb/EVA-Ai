"""
Модуль для обработки текста в CogniFlex.

Содержит класс TextProcessor, который предоставляет функциональность
для токенизации, нормализации и предобработки текста.
"""

import re
import os
import logging
from typing import List, Dict, Any, Optional, Union

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

class TextProcessor:
    """Класс для обработки текста с поддержкой различных токенизаторов.
    
    Предоставляет унифицированный интерфейс для работы с разными токенизаторами
    и выполняет предобработку текста.
    """
    
    def __init__(self, 
                 model_name: str = "rugpt3_small_fractal",
                 **tokenizer_kwargs):
        """Инициализация процессора текста.
        
        Args:
            model_name: Имя предобученной модели или путь к ней.
            **tokenizer_kwargs: Дополнительные аргументы для инициализации токенизатора.
        """
        self.model_name = model_name
        self.tokenizer_kwargs = {
            'max_length': 2048,  # Увеличено с 512 до 2048
            'truncation': True,
            'padding': 'max_length',
            'return_tensors': 'pt',
            'add_special_tokens': True,
            **tokenizer_kwargs
        }
        
        # Определяем путь к локальному токенизатору
        if model_name == "rugpt3_small_fractal":
            # Используем абсолютный путь от текущей рабочей директории
            # ВАЖНО: модель находится в models/rugpt3_small_fractal/model/, не в tokenizers/
            current_dir = os.getcwd()
            self.tokenizer_path = os.path.join(current_dir, "cogniflex_cache", "ml_unit", "fractal_storage", "models", "rugpt3_small_fractal", "model")
        elif os.path.isdir(model_name):
            # Если model_name - это путь к директории, используем его напрямую
            self.tokenizer_path = model_name
        else:
            self.tokenizer_path = model_name
        
        self._tokenizer = None
        self._initialize_tokenizer()
    
    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """Возвращает инициализированный токенизатор."""
        if self._tokenizer is None:
            self._initialize_tokenizer()
        return self._tokenizer
    
    def _initialize_tokenizer(self):
        """Инициализирует токенизатор с указанными параметрами."""
        try:
            # Сначала пробуем через Localrugpt3largeLoader
            try:
                from cogniflex.mlearning.local_rugpt3_loader import LocalRuGPT3Loader
                loader = LocalRuGPT3Loader(storage_path=self.tokenizer_path)
                self._tokenizer = loader.create_tokenizer()
                if self._tokenizer:
                    logger.info(f"Токенизатор успешно загружен через Localrugpt3largeLoader из: {self.tokenizer_path}")
                    return
            except Exception as fallback_error:
                logger.warning(f"Localrugpt3largeLoader не удался: {fallback_error}")
            
            # Если Localrugpt3largeLoader не сработал, пробуем AutoTokenizer с абсолютным путем
            init_params = {
                'local_files_only': True,  # Принудительно только локальные файлы
                'use_fast': False,  # Используем медленную но более совместимую версию
            }
            
            logger.info(f"Пробуем загрузить токенизатор из: {self.tokenizer_path}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                **init_params
            )
            logger.info(f"Токенизатор успешно загружен через AutoTokenizer из: {self.tokenizer_path}")
            
        except Exception as e:
            logger.error(f"Не удалось загрузить токенизатор {self.tokenizer_path}: {str(e)}")
            # В случае ошибки создаем базовый токенизатор
            try:
                from transformers import GPT2Tokenizer
                self._tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                logger.warning("Используем базовый GPT2 токенизатор как fallback")
            except Exception as fallback_error:
                logger.error(f"Не удалось загрузить даже базовый токенизатор: {fallback_error}")
                raise Exception(f"Не удалось инициализировать токенизатор: {e}")

    
    def tokenize(self, text: str, **kwargs) -> List[str]:
        """Токенизирует текст.
        
        Args:
            text: Входной текст для токенизации.
            **kwargs: Дополнительные аргументы для токенизатора.
            
        Returns:
            Список токенов.
        """
        try:
            return self.tokenizer.tokenize(text, **kwargs)
        except Exception as e:
            logger.error(f"Ошибка токенизации: {str(e)}")
            return text.split()  # Fallback на простой сплит по пробелам
    
    def encode(self, text: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        """Кодирует текст в числовые идентификаторы.
        
        Args:
            text: Входной текст или список текстов.
            **kwargs: Дополнительные аргументы для кодирования.
            
        Returns:
            Словарь с закодированными входами (input_ids, attention_mask и т.д.)
        """
        # Удаляем cache_dir из параметров, если он есть
        params = {
            k: v for k, v in {**self.tokenizer_kwargs, **kwargs}.items()
            if k not in ('cache_dir', 'local_files_only', 'use_fast', 'use_auth_token', 'trust_remote_code')
        }
        
        try:
            if isinstance(text, str):
                return self.tokenizer(text, **params)
            return self.tokenizer(text, **params)
        except Exception as e:
            logger.error(f"Ошибка кодирования: {str(e)}")
            raise
    
    def decode(self, token_ids: Union[torch.Tensor, List[int], List[List[int]]], **kwargs) -> str:
        """Декодирует идентификаторы токенов обратно в текст.
        
        Args:
            token_ids: Тензор или список идентификаторов токенов.
            **kwargs: Дополнительные аргументы для декодирования.
            
        Returns:
            Декодированный текст.
        """
        try:
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
            return self.tokenizer.decode(token_ids, **kwargs)
        except Exception as e:
            logger.error(f"Ошибка декодирования: {str(e)}")
            return ""
    
    def preprocess_text(self, text: str, **kwargs) -> str:
        """Предобрабатывает текст перед подачей в модель.
        
        Выполняет очистку, нормализацию и другие преобразования текста.
        
        Args:
            text: Входной текст.
            **kwargs: Дополнительные параметры предобработки.
            
        Returns:
            Обработанный текст.
        """
        # Базовая предобработка
        text = text.strip()
        
        # Нормализация пробелов
        text = re.sub(r'\s+', ' ', text)
        
        # Удаление спецсимволов (опционально)
        if kwargs.get('remove_special_chars', False):
            text = re.sub(r'[^\w\s\.\!\?\,\;\:]', '', text)
        
        # Приведение к нижнему регистру
        text = text.lower()
        
        return text.strip()
    
    def process_text(self, text: str, **kwargs) -> Dict[str, Any]:
        """Обрабатывает текст и возвращает словарь с результатами.
        
        Args:
            text: Входной текст.
            **kwargs: Дополнительные параметры.
            
        Returns:
            Словарь с результатами обработки.
        """
        try:
            # Токенизация
            tokens = self.tokenize(text, **kwargs)
            
            # Кодирование
            encoded = self.encode(text, **kwargs)
            
            # Предобработка
            processed = self.preprocess_text(text, **kwargs)
            
            return {
                "text": processed,
                "tokens": tokens,
                "token_ids": encoded.get("input_ids", []),
                "attention_mask": encoded.get("attention_mask", []),
                "vocab_size": self.get_vocab_size(),
                "status": "processed"
            }
            
        except Exception as e:
            logger.error(f"Ошибка в process_text: {e}")
            return {
                "text": text,
                "tokens": text.split(),
                "token_ids": [],
                "attention_mask": [],
                "vocab_size": 0,
                "status": "error",
                "error": str(e)
            }
    
    def batch_process(self, texts: List[str], **kwargs) -> Dict[str, torch.Tensor]:
        """Обрабатывает пакет текстов.
        
        Args:
            texts: Список текстов для обработки.
            **kwargs: Дополнительные аргументы для обработки.
            
        Returns:
            Словарь с обработанными данными для пакета.
        """
        processed_texts = [self.preprocess_text(text, **kwargs) for text in texts]
        return self.encode(processed_texts, **kwargs)
    
    def get_vocab_size(self) -> int:
        """Возвращает размер словаря токенизатора."""
        return len(self.tokenizer) if self._tokenizer else 0
