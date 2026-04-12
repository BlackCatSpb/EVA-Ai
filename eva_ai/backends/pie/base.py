"""
Base Backend - Абстрактный класс для всех бэкендов

Определяет интерфейс для загрузки и генерации моделей
различных форматов (GGUF, Transformers, ONNX).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GenerationResult:
    """Результат генерации."""
    text: str
    tokens: List[int]
    num_tokens: int
    generation_time: float
    finish_reason: str = "stop"  # stop, length, error
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class GenerationConfig:
    """Конфигурация генерации."""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.1
    stop_sequences: Optional[List[str]] = None
    stream: bool = False
    
    def __post_init__(self):
        if self.stop_sequences is None:
            self.stop_sequences = []


class BaseBackend(ABC):
    """
    Абстрактный базовый класс для всех бэкендов.
    
    Все бэкенды (GGUF, Transformers, ONNX) должны реализовывать
    этот интерфейс для совместимости с EvaPipeline.
    
    Example:
        >>> backend = GGUFBackend()
        >>> backend.load_model("model.gguf")
        >>> result = backend.generate("Привет!", config)
        >>> print(result.text)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация бэкенда.
        
        Args:
            config: Конфигурация бэкенда
        """
        self.config = config or {}
        self.is_loaded = False
        self.model_path: Optional[Path] = None
        self.model_info: Dict[str, Any] = {}
    
    @abstractmethod
    def load_model(self, path: str, **kwargs) -> None:
        """
        Загрузить модель.
        
        Args:
            path: Путь к файлу модели
            **kwargs: Дополнительные параметры загрузки
            
        Raises:
            FileNotFoundError: Если модель не найдена
            RuntimeError: Если не удалось загрузить
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """
        Сгенерировать текст.
        
        Args:
            prompt: Промпт для генерации
            config: Конфигурация генерации
            
        Returns:
            Результат генерации
        """
        pass
    
    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> Iterator[str]:
        """
        Сгенерировать текст потоком.
        
        Args:
            prompt: Промпт для генерации
            config: Конфигурация генерации
            
        Yields:
            Части сгенерированного текста
        """
        pass
    
    @abstractmethod
    def tokenize(self, text: str) -> List[int]:
        """
        Токенизировать текст.
        
        Args:
            text: Текст для токенизации
            
        Returns:
            Список ID токенов
        """
        pass
    
    @abstractmethod
    def detokenize(self, tokens: List[int]) -> str:
        """
        Детокенизировать.
        
        Args:
            tokens: Список ID токенов
            
        Returns:
            Текст
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Получить информацию о модели.
        
        Returns:
            Словарь с метаданными модели
        """
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """Выгрузить модель и освободить память."""
        pass
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Получить использование памяти.
        
        Returns:
            Словарь с {ram_mb, vram_mb}
        """
        return {"ram_mb": 0.0, "vram_mb": 0.0}
    
    def is_model_loaded(self) -> bool:
        """Проверить, загружена ли модель."""
        return self.is_loaded
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload()
