"""
Machine learning subpackage for ЕВА.

This module provides the core machine learning components for the ЕВА system,
including the Fractal Transformer architecture and related utilities.
"""
from . import storage
from .ml_unit import MLUnit
from .eva_tokenizer import ЕВАTokenizer
from .async_text_generator import AsyncTextGenerator
from .unified_text_processor import UnifiedTextProcessor

# Fractal Transformer components
from .fractal_transformer import FractalTransformer, FractalConfig
from .tokenization_fractal import ExtendedFractalTokenizer
from .neuromorphic_memory import NeuromorphicMemoryLayer
from .fractal_trainer import FractalKnowledgeTrainer

# FractalModelManager - GGUF/fallback support (safe import)
try:
    from .fractal_model_manager import FractalModelManager
    _fractal_model_available = True
except (ImportError, Exception) as e:
    FractalModelManager = None
    _fractal_model_available = False
    import logging
    logger = logging.getLogger("eva_ai.mlearning")
    logger.debug(f"FractalModelManager not available: {e}")

__all__ = [
    "MLUnit",
    "ЕВАTokenizer",
    "AsyncTextGenerator",
    "UnifiedTextProcessor",
    "FractalTransformer",
    "FractalConfig",
    "ExtendedFractalTokenizer",
    "NeuromorphicMemoryLayer",
    "FractalKnowledgeTrainer",
    "storage",
]
if _fractal_model_available:
    __all__.append("FractalModelManager")
