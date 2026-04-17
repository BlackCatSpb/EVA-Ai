"""
EVA OpenVINO Tokenizer - Адаптер токенизатора для OpenVINO моделей с фрактальной интеграцией.

Загружает токенизатор из OpenVINO модели и добавляет фрактальные специальные токены
для интеграции с FractalGraphV2 и HybridTokenCache.
"""

import os
import json
import time
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field

logger = logging.getLogger("eva_ai.openvino_tokenizer")

# Глобальный синглтон
_tokenizer_instance: Optional['EVAOpenVinoTokenizer'] = None
_tokenizer_lock = threading.Lock()


@dataclass
class FractalTokenizerConfig:
    """Конфигурация фрактального токенизатора."""
    fractal_levels: int = 4
    block_size: int = 64
    hybrid_cache_size: int = 10000
    max_length: int = 32768
    language: str = "ru"
    padding_side: str = "left"


class EVAOpenVinoTokenizer:
    """
    Токенизатор для OpenVINO моделей с фрактальной интеграцией.
    
    Особенности:
    - Загрузка токенизатора из файлов OpenVINO модели
    - Фрактальные специальные токены для интеграции с FractalGraph
    - Гибридный кэш токенов через HybridTokenCache
    - Синхронный и асинхронный API
    """
    
    def __init__(
        self,
        model_path: str,
        fractal_config: Optional[FractalTokenizerConfig] = None,
        brain=None
    ):
        self.model_path = Path(model_path) if model_path else None
        self.brain = brain
        self.config = fractal_config or FractalTokenizerConfig()
        
        self._tokenizer = None
        self._is_initialized = False
        self._load_time = 0.0
        
        # Фрактальные компоненты
        self.fractal_levels = self.config.fractal_levels
        self.block_size = self.config.block_size
        self.hybrid_cache_size = self.config.hybrid_cache_size
        self.hybrid_cache = None
        
        # Пути
        self.cache_dir = Path(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'memory', 'tokenizer_cache'
        ))
        self.metadata_dir = self.cache_dir / "metadata"
        
        self._initialize()
    
    def _initialize(self):
        """Инициализация токенизатора."""
        if not self.model_path or not self.model_path.exists():
            logger.error(f"Model path not found: {self.model_path}")
            self._create_fallback_tokenizer()
            return
        
        try:
            self._load_tokenizer()
            self._initialize_fractal_components()
            self._is_initialized = True
            logger.info(f"EVAOpenVinoTokenizer initialized from: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize EVAOpenVinoTokenizer: {e}")
            self._create_fallback_tokenizer()
    
    def _load_tokenizer(self):
        """Загружает токенизатор из файлов модели."""
        start_time = time.time()
        
        tokenizer_json = self.model_path / "tokenizer.json"
        tokenizer_config = self.model_path / "tokenizer_config.json"
        
        if not tokenizer_json.exists():
            raise FileNotFoundError(f"tokenizer.json not found in {self.model_path}")
        
        logger.info(f"Loading tokenizer from {self.model_path}")
        
        try:
            from transformers import AutoTokenizer
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                local_files_only=True,
                trust_remote_code=False,
                use_fast=True
            )
            
            self._load_time = time.time() - start_time
            logger.info(f"Tokenizer loaded in {self._load_time:.2f}s, vocab_size={len(self._tokenizer)}")
            
        except Exception as e:
            logger.warning(f"Failed to load with transformers: {e}, trying tokenizers library")
            self._load_with_tokenizers_lib()
    
    def _load_with_tokenizers_lib(self):
        """Загрузка через tokenizers library (более низкоуровневый)."""
        try:
            from tokenizers import Tokenizer
            from tokenizers.decoders import Questyafier
            
            tokenizer_json = self.model_path / "tokenizer.json"
            
            self._tokenizer = Tokenizer.from_file(str(tokenizer_json))
            logger.info("Tokenizer loaded via tokenizers library")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer: {e}")
    
    def _create_fallback_tokenizer(self):
        """Резервный токенизатор."""
        logger.warning("Using fallback tokenizer")
        self._tokenizer = None
        self._is_initialized = True
    
    def _initialize_fractal_components(self):
        """Инициализирует фрактальные компоненты."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.metadata_dir.mkdir(exist_ok=True)
            
            self._add_fractal_special_tokens()
            self._initialize_fractal_metadata()
            self._initialize_hybrid_cache()
            
            logger.info("Fractal components initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize fractal components: {e}")
    
    def _add_fractal_special_tokens(self):
        """Добавляет фрактальные специальные токены.
        
        Токены уже добавлены в tokenizer.json модели при конвертации.
        ID: 151665-151677
        """
        if self._tokenizer is None:
            return
        
        try:
            # Проверяем что токены уже есть
            fractal_tokens = [
                "<fractal_start>", "<fractal_end>", "<fractal_node>", "<fractal_edge>",
                "<fractal_level>", "<fractal_block>", "<fractal_compress>",
                "<fractal_expand>", "<fractal_memory>", "<fractal_cache>",
                "<fractal_reconstruct>", "<fractal_optimize>", "<fractal_stream>"
            ]
            
            # Проверяем наличие токенов
            missing = []
            for token in fractal_tokens:
                if token not in self._tokenizer.get_vocab():
                    missing.append(token)
            
            if missing:
                logger.warning(f"Missing fractal tokens: {missing}")
                # Добавляем если не хватает
                special_tokens_dict = {"additional_special_tokens": missing}
                added = self._tokenizer.add_special_tokens(special_tokens_dict)
                logger.info(f"Added {added} missing fractal tokens")
            else:
                logger.info("Fractal tokens already present in vocabulary")
            
        except Exception as e:
            logger.warning(f"Failed to add fractal tokens: {e}")
    
    def _initialize_fractal_metadata(self):
        """Инициализирует фрактальные метаданные."""
        metadata = {
            "tokenizer_type": "EVAOpenVinoTokenizer",
            "base_model": str(self.model_path),
            "vocab_size": len(self._tokenizer) if self._tokenizer else 0,
            "fractal_levels": self.fractal_levels,
            "block_size": self.block_size,
            "special_tokens": {
                "fractal_start": "<fractal_start>",
                "fractal_end": "<fractal_end>",
                "fractal_node": "<fractal_node>",
                "fractal_edge": "<fractal_edge>"
            }
        }
        
        try:
            metadata_file = self.metadata_dir / "fractal_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save metadata: {e}")
    
    def _initialize_hybrid_cache(self):
        """Инициализирует гибридный кэш."""
        try:
            from eva_ai.memory.hybrid_token_cache import HybridTokenCache
            
            self.hybrid_cache = HybridTokenCache(
                brain=self.brain,
                max_memory_tokens=self.hybrid_cache_size,
                disk_cache_dir=str(self.cache_dir / "hybrid_cache"),
                target_memory_gb=2.0,
                dynamic_memory_limit=True,
                max_ram_usage_percent=75.0,
                vram_threshold=0.2,
            )
            logger.info("Hybrid token cache initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize hybrid cache: {e}")
    
    # =========================================================================
    # Публичный API
    # =========================================================================
    
    @property
    def is_initialized(self) -> bool:
        return self._is_initialized
    
    def tokenize(self, text: str, **kwargs) -> List[str]:
        """Токенизирует текст."""
        if self._tokenizer is None:
            return text.split()
        return self._tokenizer.tokenize(text, **kwargs)
    
    def encode(self, text: str, add_special_tokens: bool = True, **kwargs) -> List[int]:
        """Кодирует текст в ID токенов."""
        if self._tokenizer is None:
            return [0] * len(text.split())
        
        if hasattr(self._tokenizer, 'encode'):
            return self._tokenizer.encode(text, add_special_tokens=add_special_tokens, **kwargs)
        return self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(text))
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True, **kwargs) -> str:
        """Декодирует ID токенов в текст."""
        if self._tokenizer is None:
            return ""
        
        if hasattr(self._tokenizer, 'decode'):
            return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)
        return self._tokenizer.convert_ids_to_tokens(token_ids)
    
    def __call__(self, text: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """Прямой вызов токенизатора."""
        if self._tokenizer is None:
            return {"input_ids": [], "attention_mask": []}
        return self._tokenizer(text, **kwargs)
    
    def get_fractal_tokens(self) -> List[str]:
        """Возвращает список фрактальных токенов."""
        return [
            "<fractal_start>", "<fractal_end>", "<fractal_node>", "<fractal_edge>",
            "<fractal_level>", "<fractal_block>", "<fractal_compress>",
            "<fractal_expand>", "<fractal_memory>", "<fractal_cache>",
            "<fractal_reconstruct>", "<fractal_optimize>", "<fractal_stream>"
        ]
    
    def encode_fractal_path(self, path_data: Dict[str, Any]) -> List[int]:
        """Кодирует фрактальный путь в токены."""
        import json
        path_str = json.dumps(path_data)
        fractal_start_id = self.encode("<fractal_start>")[0]
        fractal_end_id = self.encode("<fractal_end>")[0]
        
        content_ids = self.encode(path_str, add_special_tokens=False)
        return [fractal_start_id] + content_ids + [fractal_end_id]


# =============================================================================
# Фабричные функции
# =============================================================================

def create_openvino_tokenizer(
    model_path: str,
    fractal_config: Optional[FractalTokenizerConfig] = None,
    brain=None
) -> EVAOpenVinoTokenizer:
    """
    Создаёт токенизатор из OpenVINO модели.
    
    Использует синглтон для шаринга между компонентами.
    """
    global _tokenizer_instance
    
    with _tokenizer_lock:
        if _tokenizer_instance is None:
            logger.info(f"Creating EVAOpenVinoTokenizer singleton for: {model_path}")
            _tokenizer_instance = EVAOpenVinoTokenizer(
                model_path=model_path,
                fractal_config=fractal_config,
                brain=brain
            )
        else:
            logger.debug("Reusing existing EVAOpenVinoTokenizer singleton")
        return _tokenizer_instance


def get_openvino_tokenizer() -> Optional[EVAOpenVinoTokenizer]:
    """Получить текущий экземпляр токенизатора."""
    return _tokenizer_instance


def reset_openvino_tokenizer():
    """Сбросить синглтон (для тестирования)."""
    global _tokenizer_instance
    _tokenizer_instance = None
