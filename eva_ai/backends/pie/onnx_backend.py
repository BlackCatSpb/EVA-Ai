"""
ONNX Backend - Реализация для ONNX Runtime

Поддержка ONNX моделей.
"""

import logging
from typing import Dict, Any, List, Optional, Iterator

from .base import BaseBackend, GenerationResult, GenerationConfig

logger = logging.getLogger("eumi.backends.onnx")


class ONNXBackend(BaseBackend):
    """
    Бэкенд для ONNX моделей через ONNX Runtime.
    
    Заглушка для будущей реализации.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.session = None
        logger.warning("ONNXBackend is a stub. Not fully implemented yet.")
    
    def load_model(self, path: str, **kwargs) -> None:
        """Загрузить ONNX модель."""
        # TODO: Implement using onnxruntime
        raise NotImplementedError("ONNX backend not implemented yet")
    
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
        return {"backend": "onnx", "status": "not_implemented"}
    
    def unload(self) -> None:
        """Выгрузить модель."""
        self.session = None
        self.is_loaded = False
