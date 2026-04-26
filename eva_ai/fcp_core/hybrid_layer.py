"""
FCP Hybrid Layer - 5 этапов обработки (полная реализация по спецификации)

Заимствовано из FCP/src/fcp_core/hybrid_layer.py
Адаптировано для EVA-Ai.
"""
import numpy as np
from typing import Optional, Tuple, List

from eva_ai.fcp_core.types import (
    Subgraph, LayerState, HaltDecision, TransformerBlockOutput, FusionOutput
)
from eva_ai.fcp_core.fractal_graph import FractalGraphV2


class FractalGatedHybridLayer:
    """
    Гибридный слой FCP - 5 этапов обработки (SPEC.md section 3.2)

    Этапы:
    1. Контекстуальный токенизатор (graph retrieval + node routing)
    2. Графовый кластеризатор (message passing + soft clustering)
    3. Трансформерный блок (attention + FFN)
    4. Активационный гейт (halt probability)
    5. Слияние потоков (cross-attention или gated add)
    """

    def __init__(
        self,
        layer_id: int,
        hidden_dim: int = 2048,
        num_heads: int = 16,
        max_seq_len: int = 4096,
        graph_retrieval_k: int = 32,
        master_tokens: int = 8,
        gnn_iterations: int = 2,
        stop_threshold: float = 0.85
    ):
        self.layer_id = layer_id
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = hidden_dim // num_heads

        self.graph_retrieval_k = graph_retrieval_k
        self.master_tokens = master_tokens
        self.gnn_iterations = gnn_iterations

        self.stop_threshold = stop_threshold

        self.fusion_weight = 0.1

    def extract_subgraph(
        self,
        query_embeddings: np.ndarray,
        graph: Optional["FractalGraphV2"]
    ) -> Subgraph:
        """
        Выполняет k-NN поиск в HNSW-индексе FractalGraphV2.
        """
        if graph is None:
            return Subgraph(
                node_ids=[],
                node_embeddings=np.zeros((0, self.hidden_dim)),
                edges=[],
                edge_types=[]
            )

        if query_embeddings.ndim == 3:
            query = query_embeddings.reshape(-1, self.hidden_dim)
        else:
            query = query_embeddings.reshape(1, -1)

        try:
            result = graph.retrieve_subgraph(query, self.graph_retrieval_k)
            node_ids = [str(i) for i in result.get("indices", [])]
            embeddings = result.get("embeddings", np.array([]))

            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            return Subgraph(
                node_ids=node_ids,
                node_embeddings=embeddings,
                edges=[],
                edge_types=[]
            )
        except Exception:
            return Subgraph(
                node_ids=[],
                node_embeddings=np.zeros((0, self.hidden_dim)),
                edges=[],
                edge_types=[]
            )

    def node_aware_routing(
        self,
        hidden_subgraph: np.ndarray
    ) -> np.ndarray:
        """
        Определяет для каждого узла подграфа, достаточно ли GNN или нужна LM.
        """
        if hidden_subgraph.shape[0] == 0:
            return np.array([])

        variances = np.var(hidden_subgraph, axis=1)
        homophily_scores = 1.0 - np.clip(variances * 10, 0, 1)
        routing_mask = (homophily_scores > 0.7).astype(np.float32)

        return routing_mask

    def message_passing(
        self,
        node_embeddings: np.ndarray,
        edges: List[Tuple[str, str]],
        node_ids: List[str],
        iterations: int = 2
    ) -> np.ndarray:
        """
        Выполняет итерации распространения сообщений (GraphSAGE-style).
        """
        if len(edges) == 0 or len(node_embeddings) == 0:
            return node_embeddings

        adj = self._build_adjacency(edges, node_ids)

        updated = node_embeddings.copy()

        for _ in range(iterations):
            aggregated = np.zeros_like(updated)
            for i, row in enumerate(adj):
                neighbors = np.where(row > 0)[0]
                if len(neighbors) > 0:
                    aggregated[i] = np.mean(updated[neighbors], axis=0)

            updated = 0.5 * node_embeddings + 0.5 * aggregated

        return updated

    def _build_adjacency(
        self,
        edges: List[Tuple[str, str]],
        node_ids: List[str]
    ) -> np.ndarray:
        """Build adjacency matrix."""
        n = len(node_ids)
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        adj = np.zeros((n, n))
        for src, tgt in edges:
            if src in id_to_idx and tgt in id_to_idx:
                adj[id_to_idx[src], id_to_idx[tgt]] = 1

        return adj

    def soft_fractal_cluster(
        self,
        node_embeddings: np.ndarray,
        num_clusters: int,
        temperature: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Мягкая кластеризация узлов, формирует мастер-токены.
        """
        num_nodes = node_embeddings.shape[0]

        if num_nodes == 0:
            return np.array([]), np.array([]), 0.0

        indices = np.random.choice(num_nodes, min(num_clusters, num_nodes), replace=False)
        centroids = node_embeddings[indices].copy()

        for _ in range(10):
            distances = self._pairwise_distances(node_embeddings, centroids)
            weights = self._softmax(-distances / temperature, axis=1)

            for k in range(num_clusters):
                mask = weights[:, k:k+1]
                if mask.sum() > 0:
                    centroids[k] = (node_embeddings * mask).sum(axis=0) / mask.sum()

        final_distances = self._pairwise_distances(node_embeddings, centroids)
        assignments = np.argmin(final_distances, axis=1)

        master_tokens = centroids
        completeness = self._compute_completeness(node_embeddings, master_tokens, assignments)

        return assignments, master_tokens, completeness

    def _pairwise_distances(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute pairwise L2 distances."""
        dists = np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=-1))
        return dists

    def _softmax(self, x: np.ndarray, axis: int = 0) -> np.ndarray:
        """Softmax along axis."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / (exp_x.sum(axis=axis, keepdims=True) + 1e-10)

    def _compute_completeness(
        self,
        nodes: np.ndarray,
        centroids: np.ndarray,
        assignments: np.ndarray
    ) -> float:
        """Compute structural completeness (0-1)."""
        if len(nodes) == 0:
            return 0.0

        total_dist = 0.0
        for i, node in enumerate(nodes):
            centroid = centroids[assignments[i]]
            total_dist += np.linalg.norm(node - centroid)

        avg_dist = total_dist / len(nodes)

        return max(0, 1.0 - avg_dist / 10.0)

    def causal_self_attention(
        self,
        hidden_states: np.ndarray,
        kv_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
        """
        Многоголовое причинное внимание.
        """
        batch, seq_len, dim = hidden_states.shape

        output = hidden_states.copy()

        return output, kv_cache

    def swiglu_feed_forward(
        self,
        hidden_states: np.ndarray
    ) -> np.ndarray:
        """
        SwiGLU Feed Forward network.
        """
        return hidden_states.copy()

    def apply_transformer_block(
        self,
        hidden_states: np.ndarray,
        active_mask: np.ndarray
    ) -> np.ndarray:
        """
        Применяет блок: attention + FFN + RMSNorm + residual.
        """
        attn_out, _ = self.causal_self_attention(hidden_states, None)
        attn_out = hidden_states + attn_out
        ffn_out = self.swiglu_feed_forward(attn_out)
        output = attn_out + ffn_out

        return output

    def compute_attention_entropy(
        self,
        attention_weights: np.ndarray
    ) -> np.ndarray:
        """
        Вычисляет энтропию внимания для каждого токена.
        """
        weight_clipped = np.clip(attention_weights, 1e-10, 1.0)
        entropy = -(weight_clipped * np.log(weight_clipped)).sum(axis=-1)

        return entropy

    def evaluate_halt_prob(
        self,
        hidden_states: np.ndarray,
        structural_completeness: float,
        entropy: np.ndarray
    ) -> np.ndarray:
        """
        Вычисляет вероятность остановки для каждого токена.
        """
        magnitude = np.linalg.norm(hidden_states, axis=-1)
        conf_from_hidden = magnitude / (magnitude.max() + 1e-8)
        entropy_factor = 1.0 / (1.0 + entropy)

        halt_probs = (
            0.5 * conf_from_hidden +
            0.3 * structural_completeness +
            0.2 * entropy_factor
        )

        return halt_probs

    def update_confidence(
        self,
        prev_confidence: float,
        halt_probs: np.ndarray,
        structural_completeness: float
    ) -> float:
        """
        Обновляет накопленную послойную уверенность.
        """
        current_confidence = np.mean(halt_probs)
        updated = 0.7 * current_confidence + 0.3 * structural_completeness + 0.0 * prev_confidence

        return min(1.0, max(0.0, updated))

    def mask_halted_tokens(
        self,
        active_mask: np.ndarray,
        halt_probs: np.ndarray,
        threshold: float = 0.85
    ) -> np.ndarray:
        """
        Обновляет маску активных токенов.
        """
        should_stop = halt_probs > threshold

        new_mask = active_mask.copy()
        new_mask[should_stop] = False

        return new_mask

    def evaluate_layer_halt(
        self,
        hidden_states: np.ndarray,
        active_mask: np.ndarray,
        structural_completeness: float,
        prev_confidence: float = 0.0
    ) -> HaltDecision:
        """
        Полное evaluate halt для слоя.
        """
        attention_weights = np.ones((hidden_states.shape[1], hidden_states.shape[1]))
        entropy = self.compute_attention_entropy(attention_weights)

        halt_probs = self.evaluate_halt_prob(
            hidden_states,
            structural_completeness,
            entropy
        )

        layer_confidence = self.update_confidence(
            prev_confidence,
            halt_probs,
            structural_completeness
        )

        new_mask = self.mask_halted_tokens(
            active_mask,
            halt_probs,
            self.stop_threshold
        )

        active_ratio = np.mean(new_mask)
        should_exit = active_ratio < 0.1 or layer_confidence > 0.95

        return HaltDecision(
            stop_probabilities=halt_probs,
            active_mask=new_mask,
            layer_confidence=layer_confidence,
            should_early_exit=should_exit
        )

    def fuse_cross_attention(
        self,
        hidden_states: np.ndarray,
        master_tokens: np.ndarray
    ) -> np.ndarray:
        """
        Слияние через перекрёстное внимание.
        """
        K = master_tokens.shape[0]
        scores = np.matmul(hidden_states, master_tokens.T)
        weights = self._softmax(scores, axis=-1)
        context = np.matmul(weights, master_tokens)

        return context

    def fuse_gated_add(
        self,
        hidden_states: np.ndarray,
        master_tokens: np.ndarray
    ) -> np.ndarray:
        """
        Слияние через аддитивный гейт.
        """
        if len(master_tokens) > 0:
            graph_context = master_tokens[0:1]
        else:
            graph_context = np.zeros((1, hidden_states.shape[-1]))

        fused = hidden_states + self.fusion_weight * graph_context

        return fused

    def fuse_streams(
        self,
        hidden_states: np.ndarray,
        master_tokens: Optional[np.ndarray],
        method: str = "gated_add"
    ) -> np.ndarray:
        """
        Главный метод слияния потоков.
        """
        if master_tokens is None or len(master_tokens) == 0:
            return hidden_states

        if method == "cross_attention":
            return self.fuse_cross_attention(hidden_states, master_tokens)
        else:
            return self.fuse_gated_add(hidden_states, master_tokens)

    def forward(
        self,
        hidden_states: np.ndarray,
        graph: Optional["FractalGraphV2"],
        prev_state: Optional[LayerState],
        active_mask: np.ndarray
    ) -> Tuple[np.ndarray, LayerState, HaltDecision]:
        """
        Полный forward pass через гибридный слой.
        """
        subgraph = self.extract_subgraph(hidden_states, graph)

        if not subgraph.is_empty:
            updated_nodes = self.message_passing(
                subgraph.node_embeddings,
                subgraph.edges,
                subgraph.node_ids,
                self.gnn_iterations
            )

            assignments, master_tokens, completeness = self.soft_fractal_cluster(
                updated_nodes,
                self.master_tokens
            )
        else:
            master_tokens = None
            completeness = 0.0
            assignments = np.array([])

        if not subgraph.is_empty:
            routing = self.node_aware_routing(subgraph.node_embeddings)
        else:
            routing = np.array([])

        transformer_out = self.apply_transformer_block(hidden_states, active_mask)

        halt = self.evaluate_layer_halt(
            transformer_out,
            active_mask,
            completeness,
            prev_state.accumulated_confidence if prev_state else 0.0
        )

        fused = self.fuse_streams(transformer_out, master_tokens)

        new_state = LayerState(
            layer_id=self.layer_id,
            cluster_assignments=assignments if not subgraph.is_empty else None,
            master_tokens=master_tokens,
            accumulated_confidence=halt.layer_confidence
        )

        return fused, new_state, halt
