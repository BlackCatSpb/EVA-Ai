"""
FCP Input Layer - токенизация, эмбеддинги, RoPE

Заимствовано из FCP/src/fcp_core/input_layer.py
"""
import math
import numpy as np
from typing import Optional
from dataclasses import dataclass


class InputLayer:
    """
    Входной слой FCP:
    - Токенизация (BPE)
    - Embeddings table
    - Rotary Positional Encoding (RoPE)
    """

    def __init__(
        self,
        tokenizer,
        embedding_dim: int = 2048,
        max_seq_len: int = 2048
    ):
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        vocab_size = tokenizer.vocab_size()
        self.embeddings = None

        self._rope_cache = self._build_rope_cache(max_seq_len, embedding_dim)

    def _build_rope_cache(self, max_seq_len: int, dim: int) -> np.ndarray:
        """Precompute Rotary Positional Embeddings."""
        half_dim = dim // 2
        freqs = np.exp(
            -np.log(1e9) * np.arange(half_dim) / half_dim
        ).reshape(1, 1, -1)

        positions = np.arange(max_seq_len).reshape(-1, 1, 1)
        angles = positions * freqs

        cache = np.zeros((max_seq_len, dim))
        cache[:, ::2] = np.sin(angles).squeeze()
        cache[:, 1::2] = np.cos(angles).squeeze()

        return cache

    def get_positional_embedding(self, position: int) -> np.ndarray:
        """Get RoPE embedding for position."""
        if position >= self.max_seq_len:
            raise ValueError(f"Position {position} >= max_seq_len {self.max_seq_len}")
        return self._rope_cache[position].copy()

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Convert token IDs to embeddings with RoPE.
        """
        batch_size, seq_len = token_ids.shape

        embeddings = self._get_token_embeddings(token_ids)
        embeddings = self._apply_rope(embeddings, token_ids)

        return embeddings

    def _get_token_embeddings(self, token_ids: np.ndarray) -> np.ndarray:
        """Get embeddings for token IDs."""
        batch_size, seq_len = token_ids.shape
        embeddings = np.random.randn(batch_size, seq_len, self.embedding_dim) * 0.1
        return embeddings

    def _apply_rope(self, embeddings: np.ndarray, token_ids: np.ndarray) -> np.ndarray:
        """Apply Rotary Positional Encoding."""
        batch_size, seq_len, dim = embeddings.shape

        positions = np.arange(seq_len)

        half_dim = dim // 2
        x1 = embeddings[:, :, :half_dim]
        x2 = embeddings[:, :, half_dim:]

        rope = self._rope_cache[positions]
        rope1 = rope[:, :half_dim]
        rope2 = rope[:, half_dim:]

        x1_rot = x1 * rope1 - x2 * rope2
        x2_rot = x1 * rope2 + x2 * rope1

        embeddings = np.concatenate([x1_rot, x2_rot], axis=-1)

        return embeddings


class LayerState:
    """Состояние гибридного слоя для передачи между слоями."""

    def __init__(
        self,
        cluster_assignments: Optional[np.ndarray] = None,
        master_tokens: Optional[np.ndarray] = None,
        accumulated_confidence: Optional[np.ndarray] = None
    ):
        self.cluster_assignments = cluster_assignments
        self.master_tokens = master_tokens
        self.accumulated_confidence = accumulated_confidence

    def is_empty(self) -> bool:
        return self.master_tokens is None


@dataclass
class GraphContext:
    """Контекст из графа знаний."""
    nodes: list
    edges: list
    embeddings: Optional[np.ndarray] = None
    node_mask: Optional[np.ndarray] = None
    routing_decisions: Optional[np.ndarray] = None


@dataclass
class LayerOutput:
    """Выход гибридного слоя."""
    hidden_states: np.ndarray
    layer_state: LayerState
    stop_mask: np.ndarray
    layer_confidence: float
