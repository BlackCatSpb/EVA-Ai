"""
EUMI Embeddings - Система эмбеддингов

Layer-wise embedding system with parallel processing:
- LayerWiseEmbedder: Параллельная многоуровневая система
- EmbeddingLayer: Отдельный слой с кэшированием
- Projection: Проекции между пространствами
"""

from .layer_wise import (
    LayerWiseEmbedder,
    EmbeddingLayer,
    LayerConfig,
    LayerCache
)

__all__ = [
    "LayerWiseEmbedder",
    "EmbeddingLayer",
    "LayerConfig",
    "LayerCache"
]
