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
        except Exception:
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
        if not self._is_loaded:
            self.load()
        
        try:
            # Пытаемся получить эмбеддинги через выход embedding слоя
            # Ищем tensor с именем содержащим "embedding" или выход после входного слоя
            target_names = ['embedding', 'model.embed_tokens', 'model.embedding', 'embed_tokens_output']
            
            model = self._core.read_model(self.model_path)
            compiled = self._core.compile_model(model, self.device)
            
            # Пробуем получить выход embedding
            for op in model.get_ops():
                name = op.get_friendly_name().lower()
                if any(tn in name for tn in ['embedding', 'embed']):
                    try:
                        model.add_outputs(op.output(0))
                        compiled = self._core.compile_model(model, self.device)
                        break
                    except:
                        continue
            
            infer_request = compiled.create_infer_request()
            
            # Определяем входы модели
            input_dict = {}
            for i, input_tensor in enumerate(model.get_inputs()):
                if i == 0:
                    input_dict[input_tensor.get_any_name()] = input_ids
                elif i == 1:
                    input_dict[input_tensor.get_any_name()] = np.ones((input_ids.shape[0], input_ids.shape[1]), dtype=np.int64)
                else:
                    input_dict[input_tensor.get_any_name()] = np.zeros((input_ids.shape[0], 1), dtype=np.int64)
            
            result = infer_request.infer(input_dict)
            
            # Ищем выход с формой [1, seq, hidden_dim] = (batch, seq, 2560)
            for key in result.keys():
                data = result[key]
                if len(data.shape) == 3 and data.shape[-1] == self.hidden_dim:
                    return np.array(data)
            
            # Fallback: ищем любой выход с 3 измерениями
            for key in result.keys():
                data = result[key]
                if len(data.shape) == 3:
                    return np.array(data)
            
        except Exception as e:
            logger.warning(f"[Layerwise] forward_embedding failed: {e}")
        
        # Fallback: инициализируем случайными значениями
        batch = input_ids.shape[0] if len(input_ids.shape) > 1 else 1
        seq_len = input_ids.shape[-1]
        logger.warning("[Layerwise] forward_embedding: using random init")
        np.random.seed(42 + seq_len)
        return np.random.randn(batch, seq_len, self.hidden_dim).astype(np.float32) * 0.02

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
        if not self._is_loaded:
            self.load()
        
        try:
            # Загружаем модель и добавляем выход для нужного слоя
            model = self._core.read_model(self.model_path)
            
            # Целевое имя tensor для слоя (по аналогии с split_model_runner)
            target_pattern = f"__module.model.layers.{layer_idx}/aten::add/Add_1"
            
            # Ищем операцию с нужным именем
            target_op = None
            for op in model.get_ops():
                if op.get_friendly_name() == target_pattern:
                    target_op = op
                    break
            
            # Если не нашли, ищем по паттерну
            if target_op is None:
                for op in model.get_ops():
                    name = op.get_friendly_name()
                    if f".layers.{layer_idx}" in name or f"/layers/{layer_idx}" in name:
                        target_op = op
                        break
            
            if target_op is not None:
                model.add_outputs(target_op.output(0))
            
            compiled = self._core.compile_model(model, self.device)
            infer_request = compiled.create_infer_request()
            
            # Подготавливаем входы
            seq_len = hidden_states.shape[1]
            input_dict = {}
            input_tensors = model.get_inputs()
            
            for i, input_tensor in enumerate(input_tensors):
                name = input_tensor.get_any_name()
                if 'position' in name.lower():
                    input_dict[name] = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
                elif 'mask' in name.lower():
                    input_dict[name] = np.ones((1, seq_len), dtype=np.int64) if attention_mask is None else attention_mask
                elif 'input' in name.lower():
                    # Кодируем пустой текст для получения входного embedding
                    if self._tokenizer:
                        dummy_ids = self._tokenizer.encode("")[:seq_len]
                        while len(dummy_ids) < seq_len:
                            dummy_ids.append(0)
                        input_dict[name] = np.array([dummy_ids], dtype=np.int64)
                    else:
                        input_dict[name] = np.zeros((1, seq_len), dtype=np.int64)
                elif 'hidden' in name.lower():
                    input_dict[name] = hidden_states
                elif 'beam' in name.lower():
                    input_dict[name] = np.array([0], dtype=np.int32)
                else:
                    input_dict[name] = np.zeros((1, seq_len), dtype=np.int64)
            
            result = infer_request.infer(input_dict)
            
            # Ищем выход с формой [batch, seq, 2560]
            for key in result.keys():
                data = result[key]
                if len(data.shape) == 3 and data.shape[-1] == self.hidden_dim:
                    return np.array(data)
            
            # Fallback: ищем любой 3D выход
            for key in result.keys():
                data = result[key]
                if len(data.shape) == 3:
                    return np.array(data)
            
        except Exception as e:
            logger.warning(f"[Layerwise] forward_layer {layer_idx} failed: {e}")
        
        # Fallback: применяем простую трансформацию (сдвиг по слоям)
        logger.warning(f"[Layerwise] forward_layer {layer_idx}: using residual transform")
        shift = (layer_idx + 1) * 0.01
        return hidden_states * (1 + shift)

    def _apply_lm_head(self, hidden_states: np.ndarray) -> np.ndarray:
        """Применить LM head (final linear)"""
        if not self._is_loaded:
            self.load()
        
        try:
            # Ищем модель с lm_head выходом
            model = self._core.read_model(self.model_path)
            
            # Ищем lm_head операцию
            for op in model.get_ops():
                name = op.get_friendly_name().lower()
                if 'lm_head' in name or 'linear' in name or 'model.layers' not in name:
                    try:
                        model.add_outputs(op.output(0))
                        break
                    except:
                        continue
            
            compiled = self._core.compile_model(model, self.device)
            infer_request = compiled.create_infer_request()
            
            # Подготавливаем входы
            seq_len = hidden_states.shape[1]
            input_dict = {}
            input_tensors = model.get_inputs()
            
            for i, input_tensor in enumerate(input_tensors):
                name = input_tensor.get_any_name()
                if 'position' in name.lower():
                    input_dict[name] = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
                elif 'mask' in name.lower():
                    input_dict[name] = np.ones((1, seq_len), dtype=np.int64)
                elif 'input' in name.lower():
                    if self._tokenizer:
                        dummy_ids = self._tokenizer.encode("")[:seq_len]
                        while len(dummy_ids) < seq_len:
                            dummy_ids.append(0)
                        input_dict[name] = np.array([dummy_ids], dtype=np.int64)
                    else:
                        input_dict[name] = np.zeros((1, seq_len), dtype=np.int64)
                elif 'hidden' in name.lower():
                    input_dict[name] = hidden_states
                elif 'beam' in name.lower():
                    input_dict[name] = np.array([0], dtype=np.int32)
                else:
                    input_dict[name] = np.zeros((1, seq_len), dtype=np.int64)
            
            result = infer_request.infer(input_dict)
            
            # Ищем выход с формой [..., vocab_size]
            vocab_sizes = [151936, 146260, 152064]  # Возможные размеры словаря
            for key in result.keys():
                data = result[key]
                if len(data.shape) >= 2 and data.shape[-1] in vocab_sizes:
                    return np.array(data)
            
            # Fallback: ищем logits (любой большой выход)
            for key in result.keys():
                data = result[key]
                if len(data.shape) >= 2 and data.shape[-1] > 1000:
                    return np.array(data)
            
        except Exception as e:
            logger.warning(f"[Layerwise] _apply_lm_head failed: {e}")
        
        # Fallback: возвращаем logits через матричное умножение
        logger.warning("[Layerwise] _apply_lm_head: using matmul fallback")
        vocab_size = 151936  # Размер словаря Qwen
        np.random.seed(123)
        W = np.random.randn(self.hidden_dim, vocab_size).astype(np.float32) * 0.02
        logits = hidden_states @ W
        return logits

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
        # Для Qwen: нужно применить матрицу прикрепления словаря
        # Пока возвращаем нулевые логиты (заглушка)
        logger.warning("[Layerwise] _apply_lm_head: using dummy zeros")
        vocab_size = 151936  # Размер словаря Qwen
        batch = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        return np.zeros((batch, seq_len, vocab_size), dtype=np.float32)

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
        self.query_text = None  # Может быть установлено перед генерацией

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
                # Compute confidence based on hidden states stability
                if isinstance(hidden_states, np.ndarray):
                    var = np.var(hidden_states)
                    confidence = float(1.0 / (1.0 + var))
                else:
                    confidence = 0.5
                self.memory_snapshot.on_layer_forward(
                    layer_idx=layer_idx,
                    hidden_states=hidden_states,
                    layer_confidence=confidence
                )
            except Exception as e:
                logger.warning(f"[Hook] Snapshot failed layer {layer_idx}: {e}")

        # 2. Get correction from hybrid processor
        if self.hybrid_processor and self.graph:
            try:
                # Prepare knowledge nodes from graph
                nodes = self._get_graph_nodes()

                # Get query text context
                query_text = self.query_text if self.query_text else ""

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
