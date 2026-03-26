"""
Machine learning subpackage for CogniFlex.

This module provides the core machine learning components for the CogniFlex system,
including the Fractal Transformer architecture and related utilities.
"""
from . import storage
from .model_manager import ModelManager
from .cogniflex_tokenizer import CogniFlexTokenizer
from .async_text_generator import AsyncTextGenerator
from .unified_text_processor import UnifiedTextProcessor

# Fractal Transformer components
from .fractal_transformer import FractalTransformer, FractalConfig
from .tokenization_fractal import ExtendedFractalTokenizer
from .neuromorphic_memory import NeuromorphicMemoryLayer
from .fractal_trainer import FractalKnowledgeTrainer

# Qwen model support
from .qwen_model_manager import QwenModelManager, QWEN_MODELS, get_qwen_model_path, is_qwen_available

__all__ = [
    # Existing components
    "ModelManager",
    "CogniFlexTokenizer",
    "AsyncTextGenerator",
    "UnifiedTextProcessor",
    
    # Fractal Transformer components
    "FractalTransformer",
    "FractalConfig",
    "ExtendedFractalTokenizer",
    "NeuromorphicMemoryLayer",
    "FractalKnowledgeTrainer",
    "storage",
    
    # Qwen model
    "QwenModelManager",
    "QWEN_MODELS",
    "get_qwen_model_path",
    "is_qwen_available",
]

# Оптимизированные менеджеры
from .unified_fractal_manager import UnifiedFractalManager

# Для обратной совместимости
def get_fractal_manager(model_path=None, config_path=None, use_optimized=True):
    """Возвращает лучший доступный менеджер"""
    return UnifiedFractalManager(
        model_path=model_path, 
        config_path=config_path, 
        force_optimized=use_optimized
    )
