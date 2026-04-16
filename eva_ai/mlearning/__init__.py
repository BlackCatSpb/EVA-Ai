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

# FractalModelManager - GGUF/fallback support
from .fractal_model_manager import FractalModelManager

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
    "FractalModelManager",
]
