"""
Semantic Cache Evictor - Семантическое вытеснение кэша

Заимствовано из FCP/src/fcp_tools/semantic_cache_evictor.py

Выбирает блоки для вытеснения на основе важности.
"""
import numpy as np
from typing import List, Optional


class SemanticCacheEvictor:
    """
    Вытеснение на основе семантической важности.

    Оценивает важность токенов через связь с графом.
    """

    def __init__(
        self,
        gnn,
        graph
    ):
        self.gnn = gnn
        self.graph = graph
        self.node_importance: dict = {}

    def _token_importance(self, token_hidden: np.ndarray) -> float:
        """
        Оценить важность токена.
        """
        if self.gnn is None:
            return 0.5

        try:
            sub = self.gnn.retrieve_subgraph(token_hidden, k=1)

            if not sub.get("node_ids"):
                return 0.5

            node_id = sub["node_ids"][0]
            distance = sub.get("distances", [0.5])[0]

            temporal_weight = self.graph.get_node(node_id).get("temporal_weight", 1.0) if self.graph else 1.0

            importance = (1.0 / (1.0 + distance)) * temporal_weight

            return importance

        except Exception:
            return 0.5

    def select_blocks_to_evict(
        self,
        kv_blocks: List[any],
        num_to_free: int
    ) -> List[int]:
        """
        Выбрать блоки для вытеснения.
        """
        if not kv_blocks:
            return []

        scores = []

        for i, block in enumerate(kv_blocks):
            last_hidden = self._get_block_hidden(block)

            if last_hidden is not None:
                importance = self._token_importance(last_hidden)
            else:
                importance = 0.5

            scores.append((i, importance))

        scores.sort(key=lambda x: x[1])

        return [idx for idx, _ in scores[:num_to_free]]

    def _get_block_hidden(self, block) -> Optional[np.ndarray]:
        """Получить hidden state из блока."""
        if hasattr(block, 'last_hidden_state'):
            return block.last_hidden_state
        elif hasattr(block, 'hidden'):
            return block.hidden
        else:
            return None

    def should_evict(
        self,
        current_size: int,
        max_size: int,
        threshold: float = 0.7
    ) -> bool:
        """Определить нужно ли вытеснение."""
        return current_size >= max_size * threshold


class CacheEvictionPolicy:
    """Политика вытеснения кэша."""

    def __init__(
        self,
        evictor: SemanticCacheEvictor,
        min_size: int = 10,
        max_size: int = 100
    ):
        self.evictor = evictor
        self.min_size = min_size
        self.max_size = max_size

    def evict(
        self,
        cache_blocks: List[any]
    ) -> List[any]:
        """
        Вытеснить блоки.
        """
        if len(cache_blocks) <= self.min_size:
            return cache_blocks

        num_to_free = len(cache_blocks) - self.min_size

        indices = self.evictor.select_blocks_to_evict(
            cache_blocks,
            num_to_free
        )

        result = [
            block for i, block in enumerate(cache_blocks)
            if i not in indices
        ]

        return result
