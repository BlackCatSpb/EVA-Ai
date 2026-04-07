"""
TokenizerRegistry — единый синглтон токенизатора для всей системы EVA

Вместо того чтобы каждый компонент загружал свой AutoTokenizer (4-6 копий,
150-400MB RAM, 5-15 секунд startup), этот реестр загружает токенизатор
один раз и раздаёт ссылку всем компонентам.

Также поддерживает адаптер для llama.cpp встроенного токенизатора.
"""

import os
import json
import threading
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TokenizerRegistry:
    """
    Синглтон-реестр токенизаторов.
    
    Использование:
        tokenizer = TokenizerRegistry.get_tokenizer()
        tokens = tokenizer.encode("Привет мир")
    """
    _instance = None
    _tokenizer = None
    _model_path = None
    _lock = threading.Lock()
    _initialized = False
    
    @classmethod
    def get_tokenizer(cls, model_path: str = None):
        """Получает токенизатор (ленивая загрузка, синглтон)"""
        with cls._lock:
            if cls._tokenizer is not None:
                # Если запрошен другой путь — перезагружаем
                if model_path and model_path != cls._model_path:
                    logger.info(f"TokenizerRegistry: смена модели {cls._model_path} -> {model_path}")
                    cls._tokenizer = None
                    cls._model_path = model_path
                else:
                    return cls._tokenizer
            
            if model_path:
                cls._model_path = model_path
            elif not cls._model_path:
                # Пробуем найти конфиг
                cls._model_path = cls._find_model_path()
            
            if cls._model_path:
                cls._tokenizer = cls._load_tokenizer(cls._model_path)
            
            return cls._tokenizer
    
    @classmethod
    def _find_model_path(cls) -> Optional[str]:
        """Ищет путь к модели в brain_config.json"""
        config_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "brain_config.json"),
            os.path.join(os.getcwd(), "brain_config.json"),
        ]
        
        for config_path in config_paths:
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    # Пробуем разные пути
                    for key in ['model.path', 'model']:
                        if '.' in key:
                            parts = key.split('.')
                            val = config
                            for p in parts:
                                val = val.get(p, {})
                            if val and isinstance(val, str):
                                return val
                        elif key in config:
                            val = config[key]
                            if isinstance(val, dict) and 'path' in val:
                                return val['path']
                except Exception:
                    pass
        
        # Fallback путь
        fallback = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                "mlearning", "eva_models", "qwen3.5-0.8b")
        if os.path.exists(fallback):
            return fallback
        
        logger.warning("TokenizerRegistry: не найден путь к модели")
        return None
    
    @classmethod
    def _load_tokenizer(cls, model_path: str):
        """Загружает токенизатор с оптимальными настройками"""
        try:
            from transformers import AutoTokenizer
            
            # Проверяем существование пути
            if not os.path.exists(model_path):
                logger.warning(f"TokenizerRegistry: путь не существует: {model_path}")
                return None
            
            logger.info(f"TokenizerRegistry: загрузка токенизатора из {model_path}")
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,    # Не ходить в сеть
                use_fast=True,            # Быстрый Rust-токенизатор
                trust_remote_code=True,
                padding_side='left'
            )
            
            # Устанавливаем pad_token если отсутствует
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            logger.info(f"TokenizerRegistry: токенизатор загружен (vocab_size={tokenizer.vocab_size})")
            return tokenizer
            
        except Exception as e:
            logger.error(f"TokenizerRegistry: ошибка загрузки токенизатора: {e}")
            return None
    
    @classmethod
    def reset(cls):
        """Сбрасывает реестр (для тестов)"""
        with cls._lock:
            cls._tokenizer = None
            cls._model_path = None
    
    @classmethod
    def get_stats(cls) -> dict:
        """Статистика реестра"""
        stats = {
            'initialized': cls._tokenizer is not None,
            'model_path': cls._model_path,
        }
        if cls._tokenizer:
            stats['vocab_size'] = getattr(cls._tokenizer, 'vocab_size', 0)
        return stats


class LlamaCppTokenizerAdapter:
    """
    Адаптер встроенного токенизатора llama.cpp.
    
    Позволяет использовать токенизатор GGUF модели без загрузки
    отдельного PyTorch AutoTokenizer.
    
    Использование:
        adapter = LlamaCppTokenizerAdapter(llama_model)
        tokens = adapter.encode("Привет мир")
    """
    
    def __init__(self, llama_model):
        """
        Args:
            llama_model: Экземпляр llama_cpp.Llama
        """
        self.model = llama_model
        self.vocab_size = getattr(llama_model, 'n_vocab', 0)
        self.pad_token = None
        self.eos_token = None
        self.bos_token = None
    
    def tokenize(self, text: str, **kwargs):
        """Токенизирует текст в список строк"""
        if isinstance(text, str):
            text = text.encode('utf-8')
        token_ids = self.model.tokenize(text)
        # Detokenize to get string tokens
        tokens = []
        for tid in token_ids:
            try:
                token = self.model.detokenize([tid]).decode('utf-8', errors='replace')
                tokens.append(token)
            except Exception:
                tokens.append(f"<token_{tid}>")
        return tokens
    
    def encode(self, text: str, **kwargs):
        """Кодирует текст в список token IDs"""
        if isinstance(text, str):
            text = text.encode('utf-8')
        return self.model.tokenize(text)
    
    def decode(self, token_ids, skip_special_tokens: bool = True, **kwargs):
        """Декодирует token IDs в текст"""
        if isinstance(token_ids, list) and len(token_ids) > 0 and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return self.model.detokenize(token_ids).decode('utf-8', errors='replace')
    
    def __call__(self, text: str, **kwargs):
        """Совместимость с transformers API"""
        import torch
        if isinstance(text, str):
            text_bytes = text.encode('utf-8')
        else:
            text_bytes = text
        input_ids = self.model.tokenize(text_bytes)
        return {
            "input_ids": torch.tensor([input_ids]),
            "attention_mask": torch.tensor([[1] * len(input_ids)]),
        }
    
    def batch_decode(self, sequences, **kwargs):
        """Пакетный декодинг"""
        return [self.decode(seq, **kwargs) for seq in sequences]
