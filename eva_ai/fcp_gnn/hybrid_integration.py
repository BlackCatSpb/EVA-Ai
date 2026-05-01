"""
Hybrid Layer Integration - Связь всех компонентов: LLM, GNN, LoRA, KCA, SRG

Каждый слой модели становится гибридным:
- LLM: базовый forward
- GNN: graph_vec injection
- LoRA: адаптер для тонкой настройки
- KCA: коррекция через лакуны и противоречия
- SRG: decision gate для режима генерации
"""
import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger("eva_ai.fcp_gnn")

from eva_ai.fcp_core import (
    FCPConfig,
    KnowledgeConsciousAttention,
    SemanticRelevanceGate,
    ConvergenceController
)


class FractalGraphEncoderLocal:
    """
    Графовый энкодер с полными матричными вычислениями.

    SAGEConv-style aggregation с HNSW для быстрого поиска.
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 512,
        output_dim: int = 2560
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        np.random.seed(42)

        # SAGEConv-style веса: W_agg для агрегации, W_update для обновления
        self.W_agg = np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.02
        self.W_update = np.random.randn(hidden_dim, output_dim).astype(np.float32) * 0.02

        # Гейт для контроля инъекции graph_vec
        self.W_gate = np.random.randn(2 * output_dim, output_dim).astype(np.float32) * 0.02

        # Матрица проекции для выходного graph_vec
        self.W_proj = np.random.randn(output_dim, output_dim).astype(np.float32) * 0.02

        # LoRA-style адаптер
        self.lora_A = np.random.randn(output_dim, 8).astype(np.float32) * 0.02
        self.lora_B = np.random.randn(8, output_dim).astype(np.float32) * 0.02

        self._hnsw = None
        self._graph_nodes: List[Dict] = []

    def encode(self, x: np.ndarray, edge_index: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Полное GNN кодирование с матричными вычислениями.

        SAGEConv-style:
        h_neighbors = mean(neighbor embeddings)
        h_self = self embedding
        h_new = ReLU(W_agg @ [h_self || h_neighbors])
        """
        batch_size = 1 if x.ndim == 2 else x.shape[0]

        if x.ndim == 2:
            x = x.reshape(1, -1) if len(x) == 0 else x

        # Если есть рёбра - используем SAGEConv агрегацию
        if edge_index is not None and len(edge_index[0]) > 1:
            # SAGEConv: neighbor aggregation
            neighbor_embs = []
            for i in range(len(edge_index[0])):
                src, dst = edge_index[0][i], edge_index[1][i]
                if src < x.shape[0] and dst < x.shape[0]:
                    neighbor_embs.append(x[src])

            if neighbor_embs:
                neighbor_matrix = np.stack(neighbor_embs)  # [num_edges, hidden]
                h_agg = np.mean(neighbor_matrix, axis=0, keepdims=True)  # [1, hidden_dim]
            else:
                h_agg = np.mean(x, axis=0, keepdims=True)
        else:
            # Без рёбер - просто pooling
            h_agg = np.mean(x, axis=0, keepdims=True)

        # SAGEConv step: h = ReLU(W_agg @ h_agg)
        h = np.tanh(h_agg @ self.W_agg)

        # Project to output dimension
        graph_vec = h @ self.W_update  # [1, output_dim]

        # LoRA-style adaptation
        lora_out = (graph_vec @ self.lora_A) @ self.lora_B
        lora_out = lora_out * (8 ** 0.5 / 8)  # scaling
        graph_vec = graph_vec + lora_out

        # Compute gate weights for injection control
        concat_vec = np.concatenate([graph_vec, graph_vec], axis=-1)
        gate_weights = 1.0 / (1.0 + np.exp(-(concat_vec @ self.W_gate)))  # sigmoid

        # Final projection
        graph_vec = graph_vec @ self.W_proj

        return graph_vec, gate_weights

    def build_hnsw_index(self, graph_nodes: List[Dict], dim: int = 2560):
        try:
            import hnswlib
        except ImportError:
            logger.warning("hnswlib not available")
            return

        self._graph_nodes = graph_nodes
        self._hnsw = hnswlib.Index(space='cosine', dim=dim)
        self._hnsw.init_index(max_elements=len(graph_nodes), ef_construction=200, M=16)

        embeddings = np.array([n.get('embedding', np.zeros(dim)) for n in graph_nodes])
        if len(embeddings) > 0:
            self._hnsw.add_items(embeddings, np.arange(len(embeddings)))
            self._hnsw.set_ef(50)

    def retrieve_subgraph(self, query_embedding: np.ndarray, k: int = 10) -> Dict[str, Any]:
        if self._hnsw is None:
            return {'x': np.array([]), 'edge_index': np.array([[], []]),
                    'node_ids': [], 'contents': [], 'distances': []}

        labels, distances = self._hnsw.knn_query(query_embedding.reshape(1, -1), k=k)
        labels = labels[0]
        distances = distances[0]
        nodes = [self._graph_nodes[i] for i in labels]

        return {
            'x': np.array([n.get('embedding', np.zeros(self.output_dim)) for n in nodes]),
            'edge_index': np.array([np.arange(len(nodes)), np.roll(np.arange(len(nodes)), 1)]),
            'node_ids': [n.get('id', str(i)) for i in labels],
            'contents': [n.get('content', '') for n in nodes],
            'distances': distances.tolist()
        }


class AdaptiveFusionInjectorLocal:
    """
    Адаптивный инъектор с матричными вычислениями для управления потоком информации.

    Graph vector injection через learned gate mechanism:
    injection = scale * gate(hidden) * W_proj(graph)
    """

    def __init__(self, hidden_dim: int = 2560, injection_scale: float = 0.1):
        self.hidden_dim = hidden_dim
        self.injection_scale = injection_scale

        # Learned проекции для инъекции
        self.W_graph_proj = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.02
        self.W_hidden_proj = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.02

        # Gate network: [hidden || graph] -> scalar
        self.W_gate = np.random.randn(2 * hidden_dim, hidden_dim).astype(np.float32) * 0.02
        self.b_gate = np.zeros(hidden_dim).astype(np.float32)

        # LoRA адаптер для fine-tuning
        self.lora_A = np.random.randn(hidden_dim, 4).astype(np.float32) * 0.02
        self.lora_B = np.random.randn(4, hidden_dim).astype(np.float32) * 0.02

        self.gate_weights: Optional[np.ndarray] = None

    def inject_to_hidden(
        self,
        hidden_states: np.ndarray,
        graph_vec: np.ndarray
    ) -> np.ndarray:
        """
        Матричная инъекция graph_vec в hidden_states.

        Args:
            hidden_states: [batch, seq_len, hidden_dim] или [seq_len, hidden_dim]
            graph_vec: [hidden_dim] или [1, hidden_dim]

        Returns:
            modified hidden_states
        """
        # Reshape hidden_states
        original_shape = hidden_states.shape
        if hidden_states.ndim == 2:
            hidden_states = hidden_states.unsqueeze(0) if hasattr(hidden_states, 'unsqueeze') else hidden_states.reshape(1, *hidden_states.shape)

        batch_size, seq_len, dim = hidden_states.shape

        # Reshape graph_vec
        if graph_vec.ndim == 1:
            graph_vec = graph_vec.reshape(1, 1, -1)
        elif graph_vec.ndim == 2:
            graph_vec = graph_vec.reshape(1, 1, -1)

        # Project graph_vec: [1, 1, hidden_dim] -> [1, 1, hidden_dim]
        graph_proj = np.tanh(graph_vec @ self.W_graph_proj)

        # Get last token hidden state: [batch, 1, hidden_dim]
        last_hidden = hidden_states[:, -1:, :]

        # Compute injection gate: sigmoid(W @ [last_hidden || graph_proj])
        concat = np.concatenate([last_hidden, graph_proj], axis=-1)
        gate_input = concat @ self.W_gate + self.b_gate
        gate = 1.0 / (1.0 + np.exp(-gate_input))  # sigmoid -> [0, 1]

        # Compute LoRA adaptation
        lora_update = (last_hidden @ self.lora_A) @ self.lora_B
        lora_update = lora_update * (4 ** 0.5 / 4)  # scaling

        # Injection: scale * gate * graph_proj + lora
        injection = self.injection_scale * gate * graph_proj
        injection = injection + lora_update

        # Apply to last token position
        hidden_states[:, -1:, :] = hidden_states[:, -1:, :] + injection

        return hidden_states.reshape(original_shape)

    def set_gate_weights(self, weights: np.ndarray):
        """Установить предвычисленные gate weights."""
        self.gate_weights = weights

    def reset(self):
        """Сбросить gate weights."""
        self.gate_weights = None


class TextFusionInjectorLocal:
    """Локальная копия для hybrid_integration"""

    def __init__(self, max_nodes: int = 5,
                 format_template: str = "\n📚 Контекст из графа знаний:\n{context}\n"):
        self.max_nodes = max_nodes
        self.format_template = format_template

    def format_subgraph(self, subgraph: dict) -> str:
        contents = subgraph.get('contents', [])
        if not contents:
            return ""
        limited = contents[:self.max_nodes]
        lines = []
        for i, content in enumerate(limited):
            lines.append(f"  {i+1}. {content}")
        context_str = "\n".join(lines)
        return self.format_template.format(context=context_str)

    def inject_to_prompt(self, prompt: str, subgraph: dict, position: str = "prefix") -> str:
        context = self.format_subgraph(subgraph)
        if not context:
            return prompt
        if position == "prefix":
            return f"{context}\n{prompt}"
        else:
            return f"{prompt}\n{context}"


@dataclass
class HybridLayerConfig:
    """Конфигурация гибридного слоя"""
    hidden_dim: int = 2560
    num_layers: int = 32
    use_gnn: bool = True
    use_lora: bool = True
    use_kca: bool = True
    use_srg: bool = True
    injection_scale: float = 0.1
    lora_rank: int = 8


class HybridLayerState:
    """
    Состояние гибридного слоя.
    
    Отслеживает все компоненты для каждого запроса.
    """
    
    def __init__(self, config: HybridLayerConfig):
        self.config = config
        
        # LLM State
        self.hidden_states: Optional[np.ndarray] = None
        self.query_text: str = ""
        self.response_text: str = ""
        
        # GNN State
        self.graph_encoder: Optional[FractalGraphEncoder] = None
        self.current_subgraph: Dict = {}
        self.graph_vec: Optional[np.ndarray] = None
        
        # KCA State
        self.kca: Optional[KnowledgeConsciousAttention] = None
        self.kca_corrected_states: Optional[np.ndarray] = None
        self.kca_history: Dict = {}
        
        # SRG State
        self.srg: Optional[SemanticRelevanceGate] = None
        self.srg_mode: str = "direct"
        self.srg_metrics: Dict = {}
        
        # LoRA State
        self.lora_adapter: str = "default"
        self.lora_active: bool = False
        
        # Injection tracking
        self.injection_count: int = 0
        self.last_injection_layer: int = -1
        
    def reset(self):
        """Сброс состояния"""
        self.hidden_states = None
        self.response_text = ""
        self.current_subgraph = {}
        self.graph_vec = None
        self.kca_corrected_states = None
        self.kca_history = {}
        self.srg_mode = "direct"
        self.srg_metrics = {}
        self.lora_active = False
        self.injection_count = 0
        self.last_injection_layer = -1


class HybridLayerProcessor:
    """
    Обработчик гибридного слоя.
    
    Интегрирует все компоненты в единый forward pass:
    1. LLM forward → hidden_states
    2. GNN retrieval → subgraph + graph_vec
    3. SRG evaluation → mode (direct/reasoning/variational)
    4. KCA correction (if reasoning mode)
    5. LoRA injection (if enabled)
    6. Return corrected states + text
    """
    
    def __init__(self, config: HybridLayerConfig = None):
        self.config = config or HybridLayerConfig()
        
        # Инициализация компонентов
        self.fcp_config = FCPConfig()
        
        self.kca = KnowledgeConsciousAttention(self.fcp_config) if self.config.use_kca else None
        self.srg = SemanticRelevanceGate(self.fcp_config) if self.config.use_srg else None
        self.convergence_controller = ConvergenceController(self.fcp_config)
        
        self.graph_encoder = FractalGraphEncoderLocal(
            input_dim=384,
            hidden_dim=512,
            output_dim=self.config.hidden_dim
        ) if self.config.use_gnn else None

        self.text_injector = TextFusionInjectorLocal()
        self.hidden_injector = AdaptiveFusionInjectorLocal(
            hidden_dim=self.config.hidden_dim,
            injection_scale=self.config.injection_scale
        )
        
        # Состояние
        self.state = HybridLayerState(self.config)
        
        logger.info(f"[HLP] HybridLayerProcessor initialized: "
                   f"gnn={self.config.use_gnn}, kca={self.config.use_kca}, "
                   f"srg={self.config.use_srg}, lora={self.config.use_lora}")
    
    def process(
        self,
        query_text: str,
        hidden_states: np.ndarray,
        knowledge_nodes: Optional[List[Dict]] = None
    ) -> Tuple[str, np.ndarray, Dict]:
        """
        Основной метод обработки.
        
        Args:
            query_text: текст запроса
            hidden_states: [seq_len, hidden_dim] скрытые состояния
            knowledge_nodes: список узлов знаний для графа
            
        Returns:
            (enriched_prompt, corrected_hidden_states, metadata)
        """
        self.state.reset()
        self.state.query_text = query_text
        self.state.hidden_states = hidden_states.copy()
        
        metadata = {
            "srg_mode": "unknown",
            "kca_cycles": 0,
            "injections": 0,
            "lora_applied": False
        }
        
        # 1. Build GNN graph from knowledge nodes
        if knowledge_nodes and self.graph_encoder:
            self._build_graph(knowledge_nodes)
        
        # 2. Get subgraph if we have nodes
        if self.graph_encoder and self.graph_encoder._hnsw:
            query_emb = self._get_query_embedding(query_text)
            self.state.current_subgraph = self.graph_encoder.retrieve_subgraph(
                query_emb, k=self.fcp_config.graph_top_k
            )
            
            if len(self.state.current_subgraph.get('x', [])) > 0:
                self.state.graph_vec, _ = self.graph_encoder.encode(
                    self.state.current_subgraph['x'],
                    self.state.current_subgraph.get('edge_index')
                )
        
        # 3. SRG Evaluation
        if self.srg and self.state.graph_vec is not None:
            response_vec = hidden_states.mean(axis=0)
            self.state.srg_mode, self.state.srg_metrics = self.srg.evaluate(
                query_vec=self.state.graph_vec.squeeze(0),
                response_vec=response_vec,
                logits=np.zeros(100)
            )
            metadata["srg_mode"] = self.state.srg_mode
            metadata["srg_metrics"] = self.state.srg_metrics
        
        # 4. KCA Correction (if reasoning mode)
        if self.state.srg_mode == "reasoning" and self.kca and self.state.current_subgraph:
            if len(self.state.current_subgraph.get('x', [])) > 0:
                self.state.kca_corrected_states, self.state.kca_history = self.kca.forward(
                    hidden_states,
                    self.state.current_subgraph
                )
                metadata["kca_cycles"] = self.state.kca_history.get("cycles", 0)
                metadata["kca_status"] = self.state.kca_history.get("status", "unknown")
                
                # Use corrected states
                hidden_states = self.state.kca_corrected_states
        
        # 5. Text injection for prompt enrichment
        enriched_prompt = self._enrich_prompt(query_text)
        
        return enriched_prompt, hidden_states, metadata
    
    def _build_graph(self, nodes: List[Dict]):
        """Построить граф из узлов знаний."""
        if not self.graph_encoder:
            return
        
        self.graph_encoder._graph_nodes = []
        
        embeddings = []
        for node in nodes:
            emb = node.get('embedding')
            if emb is not None:
                embeddings.append(emb)
                self.graph_encoder._graph_nodes.append(node)
        
        if embeddings:
            embeddings = np.array(embeddings)
            dim = embeddings.shape[1] if len(embeddings) > 0 else self.config.hidden_dim
            self.graph_encoder.build_hnsw_index(
                self.graph_encoder._graph_nodes,
                dim=dim
            )
    
    def _get_query_embedding(self, text: str) -> np.ndarray:
        """Получить эмбеддинг запроса (заглушка - нужен encoder)."""
        # TODO: Использовать реальный encoder
        return np.random.randn(self.config.hidden_dim).astype(np.float32)
    
    def _enrich_prompt(self, prompt: str) -> str:
        """Обогатить промпт текстовым контекстом из графа."""
        if not self.state.current_subgraph:
            return prompt
        
        return self.text_injector.inject_to_prompt(
            prompt,
            self.state.current_subgraph,
            position="prefix"
        )
    
    def set_lora_adapter(self, adapter_name: str):
        """Установить LoRA адаптер."""
        self.state.lora_adapter = adapter_name
        self.state.lora_active = True
    
    def get_state(self) -> HybridLayerState:
        """Получить текущее состояние."""
        return self.state
    
    def get_status(self) -> Dict:
        """Получить статус всех компонентов."""
        return {
            "initialized": True,
            "use_gnn": self.config.use_gnn,
            "use_kca": self.config.use_kca,
            "use_srg": self.config.use_srg,
            "use_lora": self.config.use_lora,
            "graph_nodes": len(self.graph_encoder._graph_nodes) if self.graph_encoder else 0,
            "kca_ready": self.kca is not None,
            "srg_ready": self.srg is not None,
            "last_srg_mode": self.state.srg_mode,
            "last_kca_status": self.state.kca_history.get("status", "none") if self.kca else "disabled"
        }

    def load_trained_encoder(self, encoder_path: str) -> bool:
        """
        Загрузить обученный GNN энкодер из файла.

        Args:
            encoder_path: Путь к файлу graph_encoder.pt

        Returns:
            True если успешно, False если нет
        """
        if not os.path.exists(encoder_path):
            logger.warning(f"[HLP] Encoder not found: {encoder_path}")
            return False

        try:
            import torch
            state_dict = torch.load(encoder_path, map_location='cpu', weights_only=False)

            if self.graph_encoder is None:
                self.graph_encoder = FractalGraphEncoderLocal(
                    input_dim=384,
                    hidden_dim=512,
                    output_dim=self.config.hidden_dim
                )

            # Маппинг атрибутов из файла -> в класс (MLP энкодер)
            attr_mapping = {
                'lin1.weight': 'W_agg',
                'lin2.weight': 'W_update', 
                'lin3.weight': 'W_gate',
                'proj.weight': 'W_proj'
            }
            
            # Загружаем bias если есть
            bias_mapping = {
                'lin1.bias': 'b_agg',
                'lin2.bias': 'b_update',
                'lin3.bias': 'b_gate',
                'proj.bias': 'b_proj'
            }

            loaded_count = 0
            for old_attr, new_attr in attr_mapping.items():
                if old_attr in state_dict and hasattr(self.graph_encoder, new_attr):
                    value = state_dict[old_attr]
                    # Конвертируем из torch в numpy если нужно
                    if hasattr(value, 'numpy'):
                        value = value.numpy()
                    setattr(self.graph_encoder, new_attr, value)
                    loaded_count += 1
            
            # Загружаем bias
            for old_attr, new_attr in bias_mapping.items():
                if old_attr in state_dict:
                    value = state_dict[old_attr]
                    if hasattr(value, 'numpy'):
                        value = value.numpy()
                    # Добавляем атрибут если нет
                    if not hasattr(self.graph_encoder, new_attr):
                        setattr(self.graph_encoder, new_attr, value)
                    loaded_count += 1

            if loaded_count > 0:
                logger.info(f"[HLP] Loaded {loaded_count} trained weights from {encoder_path}")
            else:
                logger.warning(f"[HLP] No matching weights found, using random init")
            
            return True

        except Exception as e:
            logger.error(f"[HLP] Failed to load encoder: {e}")
            return False


class HybridLayerManager:
    """
    Менеджер гибридных слоёв для всей модели.
    
    Управляет состоянием и инъекциями на каждом слое.
    """
    
    def __init__(self, config: HybridLayerConfig = None):
        self.config = config or HybridLayerConfig()
        self.processors: Dict[int, HybridLayerProcessor] = {}
        
        # Глобальный GNN encoder (шарится между слоями)
        self.global_graph_encoder: Optional[FractalGraphEncoder] = None
        
        # Кеш состояний для каждого запроса
        self._state_cache: Dict[str, HybridLayerState] = {}
        
        logger.info(f"[HLM] HybridLayerManager initialized: {self.config.num_layers} layers")
    
    def get_processor(self, layer_idx: int) -> HybridLayerProcessor:
        """Получить процессор для слоя."""
        if layer_idx not in self.processors:
            self.processors[layer_idx] = HybridLayerProcessor(self.config)
        return self.processors[layer_idx]
    
    def set_global_graph(self, nodes: List[Dict]):
        """Установить глобальный граф знаний."""
        if not self.global_graph_encoder:
            self.global_graph_encoder = FractalGraphEncoderLocal(
                input_dim=384,
                hidden_dim=512,
                output_dim=self.config.hidden_dim
            )

    def process_layer(
        self,
        layer_idx: int,
        query_text: str,
        hidden_states: np.ndarray
    ) -> Tuple[str, np.ndarray, Dict]:
        """Обработать запрос на конкретном слое."""
        processor = self.get_processor(layer_idx)

        return processor.process(query_text, hidden_states)

    def inject_to_prompt(
        self,
        prompt: str,
        subgraph: Optional[Dict] = None
    ) -> str:
        """Инъектировать контекст в промпт (TextFusionInjector)."""
        if subgraph:
            return TextFusionInjectorLocal().inject_to_prompt(prompt, subgraph, "prefix")
        return prompt
    
    def clear_cache(self):
        """Очистить кеш состояний."""
        self._state_cache.clear()
    
    def get_statistics(self) -> Dict:
        """Получить статистику."""
        stats = {
            "total_layers": self.config.num_layers,
            "processors_initialized": len(self.processors),
            "graph_nodes": len(self.global_graph_encoder._graph_nodes) if self.global_graph_encoder else 0
        }
        
        for idx, proc in self.processors.items():
            stats[f"layer_{idx}_srg_mode"] = proc.state.srg_mode
            stats[f"layer_{idx}_kca_cycles"] = proc.state.kca_history.get("cycles", 0)
        
        return stats