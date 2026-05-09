"""
Transformers Backend - Реализация для HuggingFace Transformers

Поддержка PyTorch моделей через transformers library.
"""

import logging
import abc
from typing import Dict, Any, List, Optional, Iterator

from .base import BaseBackend, GenerationResult, GenerationConfig

logger = logging.getLogger("eumi.backends.transformers")


class TransformersBackend(BaseBackend):
    """
    Бэкенд для HuggingFace Transformers моделей.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        logger.info("TransformersBackend initialized")
    
    @abc.abstractmethod
    def load_model(self, path: str, **kwargs) -> None:
        """Загрузить модель Transformers."""
        pass
    
    @abc.abstractmethod
    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> GenerationResult:
        """Сгенерировать текст."""
        pass
    
    @abc.abstractmethod
    def generate_stream(self, prompt: str, config: Optional[GenerationConfig] = None) -> Iterator[str]:
        """Сгенерировать потоком."""
        pass
    
    @abc.abstractmethod
    def tokenize(self, text: str) -> List[int]:
        """Токенизировать."""
        pass
    
    @abc.abstractmethod
    def detokenize(self, tokens: List[int]) -> str:
        """Детокенизировать."""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Информация о модели."""
        return {"backend": "transformers", "status": "initialized", "loaded": self.is_loaded}
    
    def unload(self) -> None:
        """Выгрузить модель."""
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
