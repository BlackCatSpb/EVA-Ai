"""
Модуль для обработки текста в ЕВА.

Содержит класс TextProcessor, который предоставляет функциональность
для токенизации, нормализации и предобработки текста.
Теперь использует Qwen2.5-0.5B токенизатор.
"""

import re
import os
import logging
from typing import List, Dict, Any, Optional, Union

from functools import lru_cache
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def _get_project_root() -> str:
    """Возвращает корневую директорию проекта"""
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    
    grandparent = os.path.dirname(os.path.dirname(current_dir))
    eva_dir = os.path.join(grandparent, 'eva')
    
    if os.path.exists(eva_dir) and os.path.isdir(eva_dir):
        return grandparent
    
    drive = os.path.splitdrive(os.getcwd())[0] or 'C:'
    username = os.environ.get('USERNAME', 'user')
    onedrive_path = os.path.join(drive, 'Users', username, 'OneDrive', 'Desktop', 'ЕВА')
    if os.path.exists(os.path.join(onedrive_path, 'eva')):
        return onedrive_path
    
    return os.getcwd()


class TextProcessor:
    """Класс для обработки текста с поддержкой различных токенизаторов.
    
    Предоставляет унифицированный интерфейс для работы с разными токенизаторами
    и выполняет предобработку текста. Теперь использует Qwen2.5-0.5B.
    """
    
    def __init__(self, 
                 model_name: str = "qwen2.5-0.5b",
                 **tokenizer_kwargs):
        """Инициализация процессора текста.
        
        Args:
            model_name: Имя предобученной модели или путь к ней.
            **tokenizer_kwargs: Дополнительные аргументы для инициализации токенизатора.
        """
        self.model_name = model_name
        self.tokenizer_kwargs = {
            'max_length': 32768,
            'truncation': True,
            'padding': 'max_length',
            'return_tensors': 'pt',
            'add_special_tokens': True,
            **tokenizer_kwargs
        }
        
        project_root = _get_project_root()
        
        if model_name == "qwen2.5-0.5b":
            self.tokenizer_path = "Qwen/Qwen2.5-0.5B"
        elif model_name == "qwen3.5-2b":
            self.tokenizer_path = os.path.join(project_root, "eva", "mlearning", "eva_models", "qwen3.5-2b")
        elif os.path.isdir(model_name):
            self.tokenizer_path = model_name
        else:
            self.tokenizer_path = os.path.join(project_root, "eva", "mlearning", "eva_models", model_name) if not os.path.isabs(model_name) else model_name
        
        self._tokenizer = None
        # Lazy loading - do not call _initialize_tokenizer() here
    
    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """Возвращает инициализированный токенизатор."""
        if self._tokenizer is None:
            self._initialize_tokenizer()
        return self._tokenizer
    
    def _initialize_tokenizer(self):
        """Initializes tokenizer via TokenizerRegistry (singleton)."""
        try:
            from eva.mlearning.tokenizer_registry import TokenizerRegistry
            
            # Try to get from registry (singleton)
            self._tokenizer = TokenizerRegistry.get_tokenizer(self.tokenizer_path)
            
            if self._tokenizer is not None:
                logger.info(f"Tokenizer loaded via TokenizerRegistry: {self.tokenizer_path}")
                return
            
            # Fallback: direct load
            logger.warning(f"TokenizerRegistry failed, trying direct load: {self.tokenizer_path}")
            from transformers import AutoTokenizer
            
            init_params = {
                'local_files_only': True,
                'use_fast': True,
                'trust_remote_code': True,
            }
            
            if os.path.isabs(self.tokenizer_path) and os.name == 'nt':
                try:
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self.tokenizer_path,
                        **init_params
                    )
                except Exception as path_error:
                    logger.warning(f"Failed to load from absolute path: {path_error}")
                    project_root = _get_project_root()
                    rel_path = os.path.relpath(self.tokenizer_path, project_root)
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        rel_path,
                        **init_params
                    )
                    self.tokenizer_path = rel_path
            else:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_path,
                    **init_params
                )
            
            # Set pad_token
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token or '</pad>'
            
            # Register in registry for other components
            TokenizerRegistry._tokenizer = self._tokenizer
            TokenizerRegistry._model_path = self.tokenizer_path
            
            logger.info(f"Tokenizer loaded directly: {self.tokenizer_path}")
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer {self.tokenizer_path}: {e}")
            self._tokenizer = None


    
    def tokenize(self, text: str, **kwargs) -> List[str]:
        """Токенизирует текст.
        
        Args:
            text: Входной текст для токенизации.
            **kwargs: Дополнительные аргументы для токенизатора.
            
        Returns:
            Список токенов.
        """
        try:
            tok = self.tokenizer
            if tok is None:
                return text.split()
            return tok.tokenize(text, **kwargs)
        except Exception as e:
            logger.error(f"Tokenization error: {e}")
            return text.split()
    

    @lru_cache(maxsize=2048)
    def _encode_cached(self, text: str):
        """Cached encoding for repeated queries."""
        params = {k: v for k, v in self.tokenizer_kwargs.items()
                  if k not in ('cache_dir', 'local_files_only', 'use_fast', 'use_auth_token', 'trust_remote_code')}
        tok = self._tokenizer
        if tok is None:
            return {"input_ids": torch.tensor([]), "attention_mask": torch.tensor([])}
        return tok(text, **params)

    def clear_cache(self):
        """Clears the encoding cache."""
        self._encode_cached.cache_clear()

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
            tok = self.tokenizer
            if tok is None:
                return {"input_ids": torch.tensor([]), "attention_mask": torch.tensor([])}
            return tok(text, **params)
        except Exception as e:
            logger.error(f"Encoding error: {e}")
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
            tok = self.tokenizer
            if tok is None:
                return ""
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
            return tok.decode(token_ids, **kwargs)
        except Exception as e:
            logger.error(f"Decoding error: {e}")
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
        tok = self.tokenizer
        return len(tok) if tok else 0
