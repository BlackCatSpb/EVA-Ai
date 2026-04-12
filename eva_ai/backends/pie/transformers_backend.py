"""
Transformers Backend - Реализация для HuggingFace Transformers

Поддержка PyTorch моделей через transformers library.
"""

import logging
from typing import Dict, Any, List, Optional, Iterator

from .base import BaseBackend, GenerationResult, GenerationConfig

logger = logging.getLogger("eumi.backends.transformers")


class TransformersBackend(BaseBackend):
    """
    Бэкенд для HuggingFace Transformers моделей.
    
    Заглушка для будущей реализации.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        logger.warning("TransformersBackend is a stub. Not fully implemented yet.")
    
    def load_model(self, path: str, **kwargs) -> None:
        """Загрузить модель Transformers."""
        # TODO: Implement using transformers.AutoModelForCausalLM
        raise NotImplementedError("Transformers backend not implemented yet")
    
    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> GenerationResult:
        """Сгенерировать текст."""
        raise NotImplementedError()
    
    def generate_stream(self, prompt: str, config: Optional[GenerationConfig] = None) -> Iterator[str]:
        """Сгенерировать потоком."""
        raise NotImplementedError()
    
    def tokenize(self, text: str) -> List[int]:
        """Токенизировать."""
        raise NotImplementedError()
    
    def detokenize(self, tokens: List[int]) -> str:
        """Детокенизировать."""
        raise NotImplementedError()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Информация о модели."""
        return {"backend": "transformers", "status": "not_implemented"}
    
    def unload(self) -> None:
        """Выгрузить модель."""
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
