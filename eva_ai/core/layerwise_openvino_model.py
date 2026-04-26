"""
LayerwiseOpenVINOModel - Послойный доступ к OpenVINO Model

Позволяет:
- Получать hidden_states после каждого слоя
- Модифицировать hidden_states между слоями
- Интегрировать HybridLayerProcessor для KCA/GNN коррекции

Это низкоуровневый API в отличие от high-level LLMPipeline.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging

logger = logging.getLogger("eva_ai.openvino_layerwise")

try:
    import openvino as ov
    import openvino.genai as ov_genai
    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False
    logger.warning("OpenVINO not available")


class LayerwiseOpenVINOModel:
    """
    Низкоуровневый доступ к OpenVINO Model для послойной обработки.

    Использует ov.Model.forward() вместо ov_genai.LLMPipeline.generate().

    Паттерн работы:
    1. tokenize(input) -> input_ids, attention_mask
    2. embed(input_ids) -> embeddings
    3. for layer_idx in range(num_layers):
           layer_output = self._run_layer(layer_idx, layer_input, attention_mask)
           # ЗДЕСЬ: перехват hidden_states для snapshot/correction
           layer_input = layer_output  # для следующего слоя
    4. final_norm(layer_output) -> logits
    5. decode(logits) -> tokens
    """

    def __init__(
        self,
        model_path: str,
        device: str = "CPU",
        num_layers: int = 32,
        hidden_dim: int = 2560
    ):
        """
        Args:
            model_path: Путь к OpenVINO модели (.xml)
            device: Устройство (CPU, GPU)
            num_layers: Количество слоёв
            hidden_dim: Размерность скрытых состояний
        """
        if not HAS_OPENVINO:
            raise RuntimeError("OpenVINO not available")

        self.model_path = model_path
        self.device = device
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self._core = ov.Core()
        self._model: Optional[ov.Model] = None
        self._compiled_model: Optional[ov.CompiledModel] = None
        self._tokenizer: Optional[Any] = None

        self._layer_outputs: Dict[int, np.ndarray] = {}

        self._is_loaded = False

    def load(self) -> bool:
        """Загрузить модель"""
        try:
            self._model = self._core.read_model(self.model_path)
            self._compiled_model = self._core.compile_model(
                self._model,
                self.device
            )
            self._is_loaded = True
            logger.info(f"[Layerwise] Model loaded: {self.model_path}, layers={self.num_layers}")
            return True
        except Exception as e:
            logger.error(f"[Layerwise] Failed to load model: {e}")
            return False

    def set_tokenizer(self, tokenizer):
        """Установить токенизатор"""
        self._tokenizer = tokenizer

    def tokenize(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        """Токенизировать текст"""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not set")
        tokens = self._tokenizer.encode(text)
        input_ids = np.array(tokens).reshape(1, -1)
        attention_mask = np.ones_like(input_ids)
        return input_ids, attention_mask

    def detokenize(self, tokens: np.ndarray) -> str:
        """Декодировать токены в текст"""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not set")
        return self._tokenizer.decode(tokens)

    def get_embedding_layer(self) -> ov.Model:
        """Получить embedding подмодель"""
        # Для Qwen моделей embedding называется "embedding"
        # Это нужно адаптировать под конкретную модель
        try:
            return self._model.get_subtensor('embedding')
        except:
            return None

    def _run_submodel(
        self,
        submodel: ov.Model,
        inputs: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Запустить подмодель (embedding, layer, final_norm)"""
        infer_request = self._compiled_model.create_infer_request()

        # Создаём infer request для подмодели если возможно
        # Иначе используем основной compiled model
        try:
            result = infer_request.infer(inputs)
            return result[0] if result else None
        except:
            # Fallback - делаем partial infer через основной request
            return None

    def forward_embedding(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Первый шаг: получить эмбеддинги из input_ids.

        Returns:
            embeddings: [1, seq_len, hidden_dim]
        """
        # В OpenVINO модели эмбеддинги обычно идут вместе с первым слоем
        # Здесь нужно реализовать специфичную логику для Qwen
        raise NotImplementedError("Embedding extraction requires model-specific implementation")

    def forward_layer(
        self,
        layer_idx: int,
        hidden_states: np.ndarray,
        attention_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward pass через один слой.

        Args:
            layer_idx: Индекс слоя (0-31)
            hidden_states: [batch, seq_len, hidden_dim] - входные states
            attention_mask: [batch, seq_len] - attention mask

        Returns:
            output_states: [batch, seq_len, hidden_dim] - выходные states
        """
        raise NotImplementedError("Layer forward requires model-specific implementation")

    def forward_all_layers(
        self,
        hidden_states: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
        layer_callback: Optional[Callable[[int, np.ndarray], None]] = None
    ) -> np.ndarray:
        """
        Forward pass через все слои с возможностью перехвата.

        Args:
            hidden_states: [batch, seq_len, hidden_dim] - входные states
            attention_mask: [batch, seq_len] - attention mask
            layer_callback: Вызывается после каждого слоя (layer_idx, hidden_states)

        Returns:
            output_states после всех слоёв
        """
        self._layer_outputs = {}
        current_states = hidden_states

        for layer_idx in range(self.num_layers):
            current_states = self.forward_layer(
                layer_idx,
                current_states,
                attention_mask
            )
            self._layer_outputs[layer_idx] = current_states.copy()

            if layer_callback:
                layer_callback(layer_idx, current_states.copy())

        return current_states

    def get_layer_outputs(self) -> Dict[int, np.ndarray]:
        """Получить все outputs слоёв после forward_all_layers()"""
        return self._layer_outputs

    def generate_with_layer_hooks(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        layer_callback: Optional[Callable[[int, np.ndarray], None]] = None
    ) -> str:
        """
        Генерация с перехватом на всех слоях.

        Args:
            prompt: Текстовый промпт
            max_new_tokens: Максимум новых токенов
            layer_callback: (layer_idx, hidden_states) -> None
                           Вызывается после каждого слоя для snapshot/correction

        Returns:
            Сгенерированный текст
        """
        if not self._is_loaded:
            self.load()

        # Токенизация
        input_ids, attention_mask = self.tokenize(prompt)

        # Embedding
        hidden_states = self.forward_embedding(input_ids)

        # Послойный проход
        output_states = self.forward_all_layers(
            hidden_states,
            attention_mask,
            layer_callback
        )

        # Final norm + lm_head -> logits
        logits = self._apply_lm_head(output_states)

        # Sampling (top-k)
        next_token = self._sample_token(logits[:, -1, :])

        # Декодирование
        return self.detokenize(next_token)

    def _apply_lm_head(self, hidden_states: np.ndarray) -> np.ndarray:
        """Применить LM head (final linear)"""
        # Это нужно реализовать в зависимости от модели
        raise NotImplementedError("LM head requires model-specific implementation")

    def _sample_token(self, logits: np.ndarray, temperature: float = 0.2) -> np.ndarray:
        """Сэмплировать следующий токен"""
        # Top-k sampling
        top_k = 40
        probs = np.exp(logits) / np.sum(np.exp(logits))
        top_k_indices = np.argsort(probs)[-top_k:]
        top_k_probs = probs[top_k_indices]
        top_k_probs /= np.sum(top_k_probs)
        chosen_idx = np.random.choice(top_k, p=top_k_probs)
        return top_k_indices[chosen_idx:chosen_idx+1]

    def is_loaded(self) -> bool:
        """Проверить загружена ли модель"""
        return self._is_loaded


class HybridLayerForwardHook:
    """
    Hook для перехвата и модификации hidden_states между слоями.

    Используется с LayerwiseOpenVINOModel для:
    - Memory Snapshot (сохранение состояний в граф)
    - KCA correction (коррекция через знания)
    - GNN retrieval (получение релевантных знаний)
    """

    def __init__(
        self,
        hybrid_processor,  # HybridLayerProcessor
        memory_snapshot_integration,  # MemorySnapshotIntegration
        fractal_graph  # FractalGraphV2
    ):
        self.hybrid_processor = hybrid_processor
        self.memory_snapshot = memory_snapshot_integration
        self.graph = fractal_graph

    def __call__(
        self,
        layer_idx: int,
        hidden_states: np.ndarray
    ) -> np.ndarray:
        """
        Вызывается после каждого слоя.

        1. Сохраняем snapshot в MemorySnapshotIntegration
        2. Получаем коррекцию из HybridLayerProcessor
        3. Возвращаем (возможно модифицированные) hidden_states

        Returns:
            Модифицированные hidden_states
        """
        # 1. Memory Snapshot
        if self.memory_snapshot:
            try:
                self.memory_snapshot.on_layer_forward(
                    layer_idx=layer_idx,
                    hidden_states=hidden_states,
                    layer_confidence=0.0  # TODO: compute confidence
                )
            except Exception as e:
                logger.warning(f"[Hook] Snapshot failed layer {layer_idx}: {e}")

        # 2. Get correction from hybrid processor
        if self.hybrid_processor and self.graph:
            try:
                # Prepare knowledge nodes from graph
                nodes = self._get_graph_nodes()

                # Get query text context (would need to be passed somehow)
                query_text = ""  # TODO: pass query context

                # Process through hybrid layer
                corrected, metadata = self.hybrid_processor.process(
                    query_text=query_text,
                    hidden_states=hidden_states,
                    knowledge_nodes=nodes
                )

                return corrected
            except Exception as e:
                logger.warning(f"[Hook] Hybrid processing failed layer {layer_idx}: {e}")

        return hidden_states

    def _get_graph_nodes(self) -> List[Dict]:
        """Получить узлы графа для retrieval"""
        if not self.graph or self.graph.node_count == 0:
            return []

        nodes = []
        for i in range(min(self.graph.node_count, 100)):  # Limit to 100
            node_emb = self.graph.get_node(i)
            if node_emb is not None:
                nodes.append({
                    'id': str(i),
                    'embedding': node_emb,
                    'content': f'Node {i}',
                    'metadata': {}
                })
        return nodes
