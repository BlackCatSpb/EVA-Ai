"""
GGUF Backend - Реализация бэкенда для GGUF моделей через llama-cpp-python
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterator

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    logging.warning("llama-cpp-python not installed. GGUF backend will not work.")

from .base import BaseBackend, GenerationResult, GenerationConfig

logger = logging.getLogger("eumi.backends.gguf")


class GGUFBackend(BaseBackend):
    """
    Бэкенд для GGUF моделей через llama-cpp-python.
    
    Поддерживает:
    - Загрузку GGUF моделей
    - Генерацию текста
    - Стриминг генерации
    - Квантизированные модели (Q4_K_M, Q5_K_M, etc.)
    - CPU и GPU инференс
    
    Example:
        >>> backend = GGUFBackend()
        >>> backend.load_model("model.gguf", n_ctx=4096, n_threads=8)
        >>> result = backend.generate("Привет!", GenerationConfig(max_tokens=100))
        >>> print(result.text)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация GGUF бэкенда.
        
        Args:
            config: Конфигурация с параметрами:
                - n_ctx: Размер контекста (по умолчанию 4096)
                - n_threads: Количество потоков (-1 = auto)
                - n_gpu_layers: Количество слоёв на GPU (0 = только CPU)
                - verbose: Логирование
        """
        super().__init__(config)
        
        if not LLAMA_AVAILABLE:
            raise RuntimeError("llama-cpp-python not installed. Install: pip install llama-cpp-python")
        
        self.model: Optional[Llama] = None
        self.n_ctx = self.config.get("n_ctx", 4096)
        self.n_threads = self.config.get("n_threads", -1)
        self.n_gpu_layers = self.config.get("n_gpu_layers", 0)
        self.verbose = self.config.get("verbose", False)
        
        # Auto threads
        if self.n_threads == -1:
            import os
            self.n_threads = os.cpu_count() or 4
    
    def load_model(self, path: str, **kwargs) -> None:
        """
        Загрузить GGUF модель.
        
        Args:
            path: Путь к .gguf файлу
            **kwargs: Переопределение параметров загрузки
            
        Raises:
            FileNotFoundError: Если файл не найден
            RuntimeError: Если не удалось загрузить
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"GGUF model not found: {path}")
        
        # Переопределение параметров
        n_ctx = kwargs.get("n_ctx", self.n_ctx)
        n_threads = kwargs.get("n_threads", self.n_threads)
        n_gpu_layers = kwargs.get("n_gpu_layers", self.n_gpu_layers)
        verbose = kwargs.get("verbose", self.verbose)
        
        logger.info(f"Loading GGUF model: {path}")
        logger.info(f"  Context size: {n_ctx}")
        logger.info(f"  Threads: {n_threads}")
        logger.info(f"  GPU layers: {n_gpu_layers}")
        
        try:
            self.model = Llama(
                model_path=str(path),
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                verbose=verbose
            )
            
            self.model_path = path
            self.is_loaded = True
            
            # Извлекаем метаданные
            self._extract_model_info()
            
            logger.info(f"Model loaded successfully")
            logger.info(f"  Vocab size: {self.model_info.get('vocab_size', 'unknown')}")
            logger.info(f"  Context size: {self.model_info.get('context_size', 'unknown')}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load GGUF model: {e}")
    
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """
        Сгенерировать текст через GGUF модель.
        
        Args:
            prompt: Промпт для генерации
            config: Конфигурация генерации
            
        Returns:
            Результат генерации
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        config = config or GenerationConfig()
        start_time = time.time()
        
        try:
            output = self.model(
                prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repeat_penalty=config.repetition_penalty,
                stop=config.stop_sequences or [],
                echo=False
            )
            
            generation_time = time.time() - start_time
            
            # Извлекаем текст
            if isinstance(output, dict):
                text = output.get("choices", [{}])[0].get("text", "")
                finish_reason = output.get("choices", [{}])[0].get("finish_reason", "stop")
            else:
                text = str(output)
                finish_reason = "stop"
            
            # Токенизация для подсчёта
            tokens = self.tokenize(text)
            
            return GenerationResult(
                text=text,
                tokens=tokens,
                num_tokens=len(tokens),
                generation_time=generation_time,
                finish_reason=finish_reason,
                metadata={
                    "prompt_tokens": len(self.tokenize(prompt)),
                    "total_tokens": len(self.tokenize(prompt)) + len(tokens)
                }
            )
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return GenerationResult(
                text="",
                tokens=[],
                num_tokens=0,
                generation_time=time.time() - start_time,
                finish_reason="error",
                metadata={"error": str(e)}
            )
    
    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> Iterator[str]:
        """
        Сгенерировать текст потоком.
        
        Args:
            prompt: Промпт
            config: Конфигурация
            
        Yields:
            Части текста
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded")
        
        config = config or GenerationConfig()
        
        try:
            stream = self.model(
                prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repeat_penalty=config.repetition_penalty,
                stop=config.stop_sequences or [],
                stream=True,
                echo=False
            )
            
            for chunk in stream:
                if isinstance(chunk, dict):
                    text = chunk.get("choices", [{}])[0].get("text", "")
                else:
                    text = str(chunk)
                
                if text:
                    yield text
                    
        except Exception as e:
            logger.error(f"Stream generation error: {e}")
            yield ""
    
    def tokenize(self, text: str) -> List[int]:
        """Токенизировать текст."""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded")
        
        return self.model.tokenize(text.encode("utf-8"))
    
    def detokenize(self, tokens: List[int]) -> str:
        """Детокенизировать."""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded")
        
        # llama_cpp не имеет прямого detokenize, используем generate с echo
        try:
            # Временно отключаем логирование
            import logging
            old_level = logging.getLogger("llama_cpp").level
            logging.getLogger("llama_cpp").setLevel(logging.ERROR)
            
            text = self.model.detokenize(tokens).decode("utf-8", errors="ignore")
            
            logging.getLogger("llama_cpp").setLevel(old_level)
            return text
        except:
            # Fallback: просто объединяем
            return " ".join(map(str, tokens))
    
    def get_model_info(self) -> Dict[str, Any]:
        """Получить информацию о модели."""
        return self.model_info.copy()
    
    def unload(self) -> None:
        """Выгрузить модель."""
        if self.model is not None:
            # llama_cpp не имеет явного unload, просто удаляем ссылку
            del self.model
            self.model = None
            self.is_loaded = False
            logger.info("Model unloaded")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Получить использование памяти."""
        # llama_cpp не предоставляет прямого API для этого
        # Можно использовать psutil для оценки
        try:
            import psutil
            process = psutil.Process()
            ram_mb = process.memory_info().rss / 1024 / 1024
            return {"ram_mb": ram_mb, "vram_mb": 0.0}  # VRAM сложно измерить
        except:
            return {"ram_mb": 0.0, "vram_mb": 0.0}
    
    def _extract_model_info(self) -> None:
        """Извлечь метаданные модели."""
        if self.model is None:
            return
        
        try:
            self.model_info = {
                "vocab_size": self.model.n_vocab(),
                "context_size": self.model.n_ctx(),
                "embedding_size": self.model.n_embd(),
                "num_layers": self.model.n_layer(),
                "num_heads": self.model.n_head(),
                "path": str(self.model_path) if self.model_path else None,
                "backend": "gguf",
                "type": "llama_cpp"
            }
        except Exception as e:
            logger.warning(f"Could not extract model info: {e}")
            self.model_info = {"backend": "gguf", "error": str(e)}
