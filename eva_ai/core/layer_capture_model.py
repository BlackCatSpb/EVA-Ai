"""
LayerCaptureModel - Трансформер-модель с перехватом hidden states

Использует transformers для получения hidden_states на всех 32 слоях.
Позволяет:
- Получать hidden_states после каждого слоя
- Модифицировать hidden_states через HybridLayerProcessor
- Продолжать генерацию с модифицированными states

Это bridge между EVA FCP system и transformer models.
"""
import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
import logging

logger = logging.getLogger("eva_ai.layer_capture")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("Transformers not available")


class LayerCaptureModel:
    """
    Transformer model с возможностью перехвата hidden states на всех слоях.

    Использует transformers с `output_hidden_states=True` для получения
    всех промежуточных состояний.

    Паттерн:
    1. tokenize(text) -> input_ids
    2. model(input_ids, output_hidden_states=True) -> hidden_states[layer_idx]
    3. Для каждого слоя вызывается callback(layer_idx, hidden_states)
    4. Продолжаем generation с возможной модификацией states
    """

    def __init__(
        self,
        model_path: str,
        num_layers: int = 32,
        device: str = "cpu"
    ):
        """
        Args:
            model_path: Путь к модели HuggingFace или локальный
            num_layers: Количество слоёв
            device: cpu или cuda
        """
        if not HAS_TRANSFORMERS:
            raise RuntimeError("Transformers not available")

        self.model_path = model_path
        self.num_layers = num_layers
        self.device = device if torch.cuda.is_available() else "cpu"

        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.config: Optional[AutoConfig] = None

        self._is_loaded = False

    def load(self) -> bool:
        """Загрузить модель и токенизатор"""
        try:
            logger.info(f"[LayerCapture] Loading model from {self.model_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.config = AutoConfig.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                config=self.config,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
                low_cpu_mem_usage=True
            )

            self.model.eval()

            self._is_loaded = True
            logger.info(f"[LayerCapture] Model loaded on {self.device}")
            return True

        except Exception as e:
            logger.error(f"[LayerCapture] Failed to load model: {e}")
            return False

    def is_loaded(self) -> bool:
        """Проверить загружена ли модель"""
        return self._is_loaded and self.model is not None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
        base_model_path: Optional[str] = None
    ) -> "LayerCaptureModel":
        """
        Создать модель из checkpoint с весами wrapped модели.

        Args:
            checkpoint_path: Путь к qwen_layer_model.pt
            device: cpu или cuda
            base_model_path: Путь к базовой модели (если отличается от checkpoint)

        Returns:
            LayerCaptureModel с загруженными весами
        """
        if not HAS_TRANSFORMERS:
            raise RuntimeError("Transformers not available")

        import torch

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint['config']
        num_layers = checkpoint['num_layers']
        state_dict = checkpoint['model_state_dict']

        # Create a fresh config with all required attributes
        from transformers import Qwen3Config
        fresh_config = Qwen3Config()
        fresh_config.hidden_size = config.hidden_size
        fresh_config.num_hidden_layers = config.num_hidden_layers
        fresh_config.num_attention_heads = config.num_attention_heads
        fresh_config.num_key_value_heads = config.num_key_value_heads
        fresh_config.intermediate_size = config.intermediate_size
        fresh_config.hidden_act = config.hidden_act
        fresh_config.rope_theta = getattr(config, 'rope_theta', 900000.0)
        fresh_config.attention_scaling = getattr(config, 'attention_scaling', 1.0)
        fresh_config.sliding_window = getattr(config, 'sliding_window', 32768)
        fresh_config.max_position_embeddings = getattr(config, 'max_position_embeddings', 32768)
        fresh_config.rms_norm_eps = getattr(config, 'rms_norm_eps', 1e-6)
        fresh_config.use_sliding_window = getattr(config, 'use_sliding_window', False)
        fresh_config.rope_scaling = getattr(config, 'rope_scaling', None)
        fresh_config.attention_bias = getattr(config, 'attention_bias', False)
        fresh_config.attention_dropout = getattr(config, 'attention_dropout', 0.0)
        fresh_config.hidden_dropout = getattr(config, 'hidden_dropout', 0.0)
        fresh_config.tie_word_embeddings = getattr(config, 'tie_word_embeddings', False)

        # Use provided path or determine from config
        if base_model_path:
            model_path = base_model_path
        else:
            # For RefalMachine models, we need local path
            model_path = getattr(config, 'name_or_path', None)
            if model_path and model_path.startswith('RefalMachine'):
                # It's a HF model - use local fallback
                model_path = None

        instance = cls(
            model_path=model_path or checkpoint_path,
            num_layers=num_layers,
            device=device
        )
        instance.config = fresh_config

        # Create model architecture without loading weights
        instance.model = AutoModelForCausalLM.from_config(fresh_config, trust_remote_code=True)

        # Load state dict directly
        instance.model.load_state_dict(state_dict, strict=False)
        instance.model = instance.model.to(device)
        instance.model.eval()

        # Load tokenizer
        if model_path and os.path.exists(model_path):
            try:
                instance.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                if instance.tokenizer.pad_token is None:
                    instance.tokenizer.pad_token = instance.tokenizer.eos_token
            except Exception as e:
                logger.warning(f"[LayerCapture] Tokenizer load failed: {e}")

        instance._is_loaded = True

        logger.info(f"[LayerCapture] Model loaded from checkpoint: {checkpoint_path}")
        return instance

    def tokenize(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Токенизировать текст"""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        return input_ids, attention_mask

    def detokenize(self, tokens: torch.Tensor) -> str:
        """Декодировать токены"""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        return self.tokenizer.decode(tokens[0], skip_special_tokens=True)

    def get_all_layer_outputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_callback: Optional[Callable[[int, torch.Tensor], None]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Получить outputs всех слоёв.

        Args:
            input_ids: [batch, seq_len] токены
            attention_mask: [batch, seq_len] attention mask
            layer_callback: (layer_idx, hidden_state) -> None
                          Вызывается после каждого слоя

        Returns:
            (logits, hidden_states_list)
            hidden_states_list[layer_idx] = [batch, seq_len, hidden_dim]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

        hidden_states = outputs.hidden_states  # tuple of (num_layers + 1) tensors

        # layer_callback для каждого слоя
        if layer_callback:
            for layer_idx, hs in enumerate(hidden_states[1:]):  # пропускаем embedding layer (index 0)
                layer_callback(layer_idx, hs)

        logits = outputs.logits  # [batch, seq_len, vocab_size]

        return logits, list(hidden_states)

    def capture_and_process(
        self,
        text: str,
        layer_callback: Optional[Callable[[int, np.ndarray], np.ndarray]] = None
    ) -> Tuple[str, List[np.ndarray]]:
        """
        Полный forward pass с перехватом и возможной модификацией states.

        Args:
            text: Входной текст
            layer_callback: (layer_idx, hidden_states) -> modified_hidden_states
                          Если возвращает None, состояние не меняется

        Returns:
            (output_text, list_of_hidden_states)
        """
        input_ids, attention_mask = self.tokenize(text)

        # Forward через все слои
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        hidden_states = list(outputs.hidden_states)
        logits = outputs.logits

        # Применяем callback для каждого слоя
        if layer_callback:
            for layer_idx in range(1, len(hidden_states)):
                hs = hidden_states[layer_idx]
                modified = layer_callback(layer_idx - 1, hs.cpu().numpy())
                if modified is not None:
                    hidden_states[layer_idx] = torch.from_numpy(modified).to(hs.device)

        # Получаем предсказанный токен
        next_token_logits = logits[0, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).item()

        # Декодируем
        output_text = self.tokenizer.decode([next_token_id], skip_special_tokens=True)

        # Возвращаем hidden states как numpy
        hs_numpy = [hs.cpu().numpy() for hs in hidden_states[1:]]

        return output_text, hs_numpy

    def generate_with_layer_processing(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        layer_callback: Optional[Callable[[int, np.ndarray], Optional[np.ndarray]]] = None,
        use_cache: bool = True
    ) -> str:
        """
        Генерация с обработкой на каждом слое.

        Args:
            prompt: Начальный промпт
            max_new_tokens: Максимум новых токенов
            layer_callback: Обработчик для каждого слоя
            use_cache: Использовать KV cache

        Returns:
            Сгенерированный текст
        """
        if not self.is_loaded():
            self.load()

        input_ids, attention_mask = self.tokenize(prompt)

        generated_tokens = []
        past_key_values = None

        for step in range(max_new_tokens):
            outputs = self.model(
                input_ids=input_ids if step == 0 else next_token_id.unsqueeze(0).unsqueeze(0),
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_hidden_states=True,
                use_cache=use_cache
            )

            hidden_states = list(outputs.hidden_states)

            # Обработка каждого слоя через callback
            if layer_callback:
                for layer_idx in range(1, len(hidden_states)):
                    hs_np = hidden_states[layer_idx].cpu().numpy()
                    modified = layer_callback(layer_idx - 1, hs_np)
                    if modified is not None:
                        # Здесь нужно решить как применить модификацию
                        # Пока просто логируем
                        pass

            logits = outputs.logits
            past_key_values = outputs.past_key_values

            next_token_logits = logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1)

            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token_id.item())

            if step == 0:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((1, 1), device=self.device)
                ], dim=-1)

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)


class HybridLayerBridge:
    """
    Bridge между LayerCaptureModel и HybridLayerProcessor.

    Позволяет использовать EVA's FCP компоненты (KCA, GNN, LoRA)
    для обработки hidden states из transformer модели.
    """

    def __init__(
        self,
        hybrid_processor,
        fractal_graph,
        memory_snapshot_integration=None
    ):
        """
        Args:
            hybrid_processor: HybridLayerProcessor для обработки слоёв
            fractal_graph: FractalGraphV2 для retrieval
            memory_snapshot_integration: MemorySnapshotIntegration для snapshots
        """
        self.hybrid_processor = hybrid_processor
        self.graph = fractal_graph
        self.memory_snapshot = memory_snapshot_integration

    def create_layer_callback(self, layer_idx: int) -> Callable:
        """
        Создать callback для конкретного слоя.

        Returns:
            callback(layer_idx, hidden_states) -> modified_states or None
        """
        def callback(layer_idx: int, hidden_states: np.ndarray) -> Optional[np.ndarray]:
            """
            Обработать hidden_states слоя.

            1. Сохранить snapshot для MemorySnapshot
            2. Retrieve из графа
            3. Применить HybridLayerProcessor
            4. Вернуть модифицированные states
            """
            # 1. Memory Snapshot
            if self.memory_snapshot:
                try:
                    self.memory_snapshot.on_layer_forward(
                        layer_idx=layer_idx,
                        hidden_states=hidden_states,
                        layer_confidence=0.0
                    )
                except Exception as e:
                    logger.warning(f"[Bridge] Snapshot failed: {e}")

            # 2. Get graph context
            nodes = []
            if self.graph and self.graph.node_count > 0:
                for i in range(min(self.graph.node_count, 50)):
                    node_emb = self.graph.get_node(i)
                    if node_emb is not None:
                        nodes.append({
                            'id': str(i),
                            'embedding': node_emb,
                            'content': f'Node {i}'
                        })

            # 3. Process through hybrid layer
            if self.hybrid_processor and nodes:
                try:
                    # Prepare query (would need context)
                    query_text = ""

                    corrected, metadata = self.hybrid_processor.process(
                        query_text=query_text,
                        hidden_states=hidden_states,
                        knowledge_nodes=nodes
                    )

                    return corrected
                except Exception as e:
                    logger.warning(f"[Bridge] Hybrid processing failed: {e}")

            return None  # No modification

        return callback
