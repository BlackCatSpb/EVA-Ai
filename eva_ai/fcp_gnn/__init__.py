"""
FCP GNN Integration - Гибридная инъекция для OpenVINO

Интеграция:
- FractalGraphEncoder (SAGEConv + HNSW) - из graph_encoder.py
- AdaptiveFusionInjector
- KCA cycles
- SRG decision
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger("eva_ai.fcp_gnn")

try:
    import hnswlib
    HAS_HNSWLIB = True
except ImportError:
    HAS_HNSWLIB = False

from eva_ai.fcp_gnn.hybrid_integration import (
    HybridLayerConfig,
    HybridLayerState,
    HybridLayerProcessor,
    HybridLayerManager
)
from eva_ai.fcp_gnn.graph_encoder import (
    FractalGraphEncoder,
    GraphEncoderRuntime
)
from eva_ai.fcp_gnn.hybrid_transformer_layer import (
    HybridTransformerLayer,
    HybridTransformerStack,
    HybridModelWithGNN,
    GNNInjectorHook
)
from eva_ai.fcp_gnn.convert_gnn_to_ov import (
    convert_gnn_to_ov,
    GNNEncoderOV,
    GNNExporter
)


class AdaptiveFusionInjector:
    """
    Адаптивный инъектор графового вектора в скрытые состояния.

    Для OpenVINO - используем текстовую инъекцию вместо скрытой.
    """

    def __init__(
        self,
        hidden_dim: int = 2560,
        injection_scale: float = 0.1
    ):
        self.hidden_dim = hidden_dim
        self.injection_scale = injection_scale
        self.gate_weights: Optional[np.ndarray] = None

    def inject_to_hidden(
        self,
        hidden_states: np.ndarray,
        graph_vec: np.ndarray
    ) -> np.ndarray:
        """Инъекция в скрытые состояния (для PyTorch моделей)."""
        if graph_vec.ndim == 1:
            graph_vec = graph_vec.reshape(1, 1, -1)
        elif graph_vec.ndim == 2:
            graph_vec = graph_vec.reshape(1, 1, -1)

        h_last = hidden_states[:, -1:, :]
        injection = self.injection_scale * graph_vec
        hidden_states[:, -1:, :] = h_last + injection

        return hidden_states

    def set_gate_weights(self, weights: np.ndarray):
        """Установить gate weights."""
        self.gate_weights = weights

    def reset(self):
        """Сбросить gate weights."""
        self.gate_weights = None


class TextFusionInjector:
    """
    Текстовый инъектор - преобразует подграф в текст для OpenVINO.
    """

    def __init__(
        self,
        max_nodes: int = 5,
        format_template: str = "\n📚 Контекст из графа знаний:\n{context}\n"
    ):
        self.max_nodes = max_nodes
        self.format_template = format_template

    def format_subgraph(self, subgraph: dict) -> str:
        """Форматировать подграф в текст."""
        contents = subgraph.get('contents', [])

        if not contents:
            return ""

        limited = contents[:self.max_nodes]
        lines = []
        for i, content in enumerate(limited):
            lines.append(f"  {i+1}. {content}")

        context_str = "\n".join(lines)
        return self.format_template.format(context=context_str)

    def inject_to_prompt(
        self,
        prompt: str,
        subgraph: dict,
        position: str = "prefix"
    ) -> str:
        """Инъектировать подграф в промпт."""
        context = self.format_subgraph(subgraph)

        if not context:
            return prompt

        if position == "prefix":
            return f"{context}\n{prompt}"
        else:
            return f"{prompt}\n{context}"


class HybridFusionInjector:
    """
    Гибридный инъектор - комбинирует текстовую и скрытую инъекцию.
    """

    def __init__(
        self,
        hidden_dim: int = 2560,
        use_text: bool = True,
        use_hidden: bool = True
    ):
        self.hidden_dim = hidden_dim
        self.text_injector = TextFusionInjector() if use_text else None
        self.hidden_injector = AdaptiveFusionInjector(hidden_dim) if use_hidden else None

    def inject(
        self,
        prompt: str,
        hidden_states: Optional[np.ndarray],
        subgraph: dict,
        inject_mode: str = "text"
    ) -> Tuple[str, np.ndarray]:
        """
        Инъектировать данные графа.

        Args:
            prompt: текстовый промпт
            hidden_states: скрытые состояния (для hidden injection)
            subgraph: данные подграфа
            inject_mode: 'text', 'hidden', 'both'

        Returns:
            (modified_prompt, modified_hidden_states or None)
        """
        result_prompt = prompt
        result_hidden = hidden_states

        if inject_mode in ("text", "both") and self.text_injector:
            result_prompt = self.text_injector.inject_to_prompt(prompt, subgraph)

        if inject_mode in ("hidden", "both") and self.hidden_injector and hidden_states is not None:
            graph_vec = self._subgraph_to_vec(subgraph)
            result_hidden = self.hidden_injector.inject_to_hidden(hidden_states, graph_vec)

        return result_prompt, result_hidden

    def _subgraph_to_vec(self, subgraph: dict) -> np.ndarray:
        """Преобразовать подграф в вектор (mean pool)."""
        x = subgraph.get('x')

        if x is None or len(x) == 0:
            return np.zeros(self.hidden_dim)

        if isinstance(x, list):
            x = np.array(x)

        vec = np.mean(x, axis=0)

        if vec.shape[-1] != self.hidden_dim:
            if vec.shape[-1] < self.hidden_dim:
                padding = np.zeros(self.hidden_dim - vec.shape[-1])
                vec = np.concatenate([vec, padding])
            else:
                vec = vec[:self.hidden_dim]

        return vec


class GNNInjectorHook:
    """
    Хук для автоматической GNN инъекции в модель.
    """

    def __init__(
        self,
        gnn_encoder: FractalGraphEncoder,
        injection_scale: float = 0.1
    ):
        self.gnn_encoder = gnn_encoder
        self.injection_scale = injection_scale
        self.current_graph_vec = None

    def set_graph_vector(self, graph_vec: np.ndarray):
        """Установить graph vector для текущего запроса."""
        self.current_graph_vec = graph_vec

    def clear(self):
        """Очистить graph vector."""
        self.current_graph_vec = None