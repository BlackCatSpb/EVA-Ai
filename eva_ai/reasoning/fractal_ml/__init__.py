"""
FractalML - Фрактальное хранилище для ЕВА
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
from .fractal_retriever import FractalRetriever
from .fractal_embedder import FractalEmbedder

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
    'FractalRetriever',
    'FractalEmbedder',
    'create_fractal_id',
]

MAX_LEVELS = 4
BRANCHING_FACTOR = 16
EMBEDDING_DIM = 384
