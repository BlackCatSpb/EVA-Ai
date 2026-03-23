"""
FractalML - Фрактальное хранилище для CogniFlex
"""

from .fractal_base import (
    FractalNode,
    FractalNodeType,
    FractalEdge,
    FractalRelationType,
    FractalAddress,
    FractalIndex,
    FractalMetadata,
    create_fractal_id
)

from .fractal_tokenizer import FractalTokenizer, FractalTokenizerWrapper

from .fractal_storage import FractalStorage

__all__ = [
    'FractalNode',
    'FractalNodeType', 
    'FractalEdge',
    'FractalRelationType',
    'FractalAddress',
    'FractalIndex',
    'FractalMetadata',
    'FractalTokenizer',
    'FractalTokenizerWrapper',
    'FractalStorage',
    'create_fractal_id',
]

MAX_LEVELS = 4
BRANCHING_FACTOR = 16
EMBEDDING_DIM = 384
