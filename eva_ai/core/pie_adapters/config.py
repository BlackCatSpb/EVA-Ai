"""
EvaConfig - Конфигурация EUMI

Управление конфигурацией моделей, бэкендов и компонентов.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict


@dataclass
class ModelConfig:
    """Конфигурация модели."""
    name: str = "unnamed"
    type: str = "gguf"
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    
    # Архитектура
    architecture: str = "transformer"
    num_layers: int = 32
    num_heads: int = 32
    hidden_size: int = 2048
    vocab_size: int = 32000
    
    # Контекст
    max_position_embeddings: int = 8192
    max_new_tokens: int = 2048


@dataclass
class BackendConfig:
    """Конфигурация бэкенда."""
    type: str = "gguf"
    device: str = "cpu"
    
    # GGUF специфичные
    n_ctx: int = 2048
    n_threads: int = -1  # -1 = auto (все ядра)
    n_batch: int = 512
    n_gpu_layers: int = 0
    
    # Квантизация
    quantization: str = "Q4_K_M"
    
    # Пути к моделям
    condensed_path: Optional[str] = None
    extended_path: Optional[str] = None
    coder_path: Optional[str] = None


@dataclass
class TokenizerConfig:
    """Конфигурация токенизатора."""
    type: str = "hybrid"
    vocab_size: int = 32000
    
    # Виртуальные токены
    virtual_tokens_enabled: bool = True
    virtual_token_start_id: int = 100000
    virtual_token_count: int = 50000
    
    # Специальные токены
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    
    # Параметры
    add_bos_token: bool = True
    add_eos_token: bool = False


@dataclass
class QuantizationConfig:
    """Конфигурация квантизации."""
    enabled: bool = True
    method: str = "layer_wise"
    
    # Layer-wise
    attention_bits: int = 16  # fp16
    ffn_bits: int = 8
    embedding_bits: int = 8
    
    # Mixed precision
    use_mixed_precision: bool = True
    important_layers: List[int] = field(default_factory=list)
    
    # Адаптивная
    adaptive_enabled: bool = False
    target_memory_mb: int = 2048


@dataclass
class KnowledgeConfig:
    """Конфигурация интеграции знаний."""
    enabled: bool = True
    
    # Virtual tokens
    inject_virtual_tokens: bool = True
    max_virtual_tokens: int = 10
    
    # Graph attention
    use_graph_attention: bool = False
    graph_attention_weight: float = 0.1
    
    # LoRA
    lora_enabled: bool = False
    lora_path: Optional[str] = None
    lora_alpha: int = 16
    lora_dropout: float = 0.05


@dataclass
class EvaConfig:
    """
    Полная конфигурация EUMI.
    
    Attributes:
        eva_format_version: Версия формата .eva
        model: Конфигурация модели
        backend: Конфигурация бэкенда
        tokenizer: Конфигурация токенизатора
        quantization: Конфигурация квантизации
        knowledge: Конфигурация интеграции знаний
    """
    eva_format_version: str = "1.0.0"
    
    model: ModelConfig = field(default_factory=ModelConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaConfig":
        """Создать конфиг из словаря."""
        return cls(
            eva_format_version=data.get("eva_format_version", "1.0.0"),
            model=ModelConfig(**data.get("model", {})),
            backend=BackendConfig(**data.get("backend", {})),
            tokenizer=TokenizerConfig(**data.get("tokenizer", {})),
            quantization=QuantizationConfig(**data.get("quantization", {})),
            knowledge=KnowledgeConfig(**data.get("knowledge", {}))
        )
    
    @classmethod
    def from_json(cls, path: str) -> "EvaConfig":
        """Загрузить из JSON файла."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_yaml(cls, path: str) -> "EvaConfig":
        """Загрузить из YAML файла."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертировать в словарь."""
        return asdict(self)
    
    def to_json(self, path: str, indent: int = 2) -> None:
        """Сохранить в JSON файл."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent, ensure_ascii=False)
    
    def to_yaml(self, path: str) -> None:
        """Сохранить в YAML файл."""
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
    
    def validate(self) -> List[str]:
        """
        Валидировать конфигурацию.
        
        Returns:
            Список ошибок (пустой если всё ок)
        """
        errors = []
        
        # Проверка версии
        if self.eva_format_version not in ["1.0.0", "1.1.0"]:
            errors.append(f"Unsupported version: {self.eva_format_version}")
        
        # Проверка бэкенда
        if self.backend.type not in ["gguf", "transformers", "onnx"]:
            errors.append(f"Unsupported backend: {self.backend.type}")
        
        # Проверка квантизации
        if self.quantization.enabled:
            if self.quantization.attention_bits not in [8, 16, 32]:
                errors.append("attention_bits must be 8, 16, or 32")
        
        return errors
