"""
Graph Injection Module - Интеграция GNN с LayerwiseStateInjector

EVA.txt раздел 8.2: "На этапе prefill после первого infer инжектор получает доступ 
ко всем состояниям. Вычисляется графовый вектор (через GNN) и семантическая маска 
(через SQAM). Затем apply_to_keys применяется ко всем 36 слоям для масштабирования 
Key, а apply_to_values — для добавления графовой коррекции."

Реализует полнослойную инъекцию (Layerwise Injection) согласно спецификации.
"""
import numpy as np
import logging
from typing import Optional, Tuple, List, Callable, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger("FCP.GraphInjection")


@dataclass
class InjectionConfig:
    """Конфигурация для инъекции графа в модель."""
    # Сила графовой коррекции (0.0 - 1.0)
    graph_correction_strength: float = 0.15
    # Сила масштабирования Key (0.0 - 1.0)
    key_scaling_strength: float = 0.10
    # Порог активации гейта (если accumulated > threshold - ранний выход)
    activation_gate_threshold: float = 0.85
    # Нормализация graph_vector перед инъекцией
    normalize_graph_vector: bool = True
    # Использовать gate_weights для модуляции инъекции
    use_gate_weights: bool = True
    # Слои для инъекции (36 для Qwen)
    num_layers: int = 36


class GraphStateInjector:
    """
    Полнослойная инъекция графа в KV-кеш модели.
    
    Интегрирует:
    1. GNN энкодер (graph_vector + gate_weights)
    2. LayerwiseStateInjector (apply_to_values/apply_to_keys)
    3. KCA (обнаружение лакун и противоречий)
    
    Согласно EVA.txt раздел 8.2:
    - apply_to_keys масштабирует Key для всех 36 слоёв
    - apply_to_values добавляет графовую коррекцию
    - На каждом шаге декодирования операция повторяется
    """
    
    def __init__(
        self,
        state_injector,  # LayerwiseStateInjector instance
        gnn_encoder,     # MiniGNN encoder
        kca_detector,    # KCA detector for lacunas/contradictions
        config: Optional[InjectionConfig] = None
    ):
        """
        Args:
            state_injector: LayerwiseStateInjector для доступа к KV-кешу
            gnn_encoder: MiniGNN энкодер для получения graph_vector
            kca_detector: KCA для обнаружения лакун/противоречий
            config: InjectionConfig с настройками
        """
        self.state_injector = state_injector
        self.gnn_encoder = gnn_encoder
        self.kca_detector = kca_detector
        self.config = config or InjectionConfig()
        
        # Нормализатор для graph_vector
        self._graph_mean = None
        self._graph_std = None
        
        # Статистика для отладки
        self._injection_count = 0
        self._last_gate_values = None
        
        logger.info(f"[GraphStateInjector] Initialized with {self.config.num_layers} layers")
        logger.info(f"  Graph correction: {self.config.graph_correction_strength}")
        logger.info(f"  Key scaling: {self.config.key_scaling_strength}")
        logger.info(f"  Activation gate threshold: {self.config.activation_gate_threshold}")
    
    def set_normalization_stats(self, mean: np.ndarray, std: np.ndarray):
        """Установить статистику нормализации graph_vector."""
        self._graph_mean = mean.astype(np.float32)
        self._graph_std = std.astype(np.float32)
    
    def compute_graph_vector(
        self,
        subgraph_embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Вычислить graph_vector и gate_weights из subgraph embeddings.
        
        Args:
            subgraph_embeddings: [num_nodes, embedding_dim] - эмбеддинги узлов подграфа
            
        Returns:
            (graph_vector, gate_weights)
            - graph_vector: [256] нормализованный вектор графа
            - gate_weights: [36] веса гейтов для каждого слоя
        """
        if subgraph_embeddings is None or len(subgraph_embeddings) == 0:
            # Пустой подграф - возврат нулевых векторов
            return np.zeros(256, dtype=np.float32), np.zeros(36, dtype=np.float32)
        
        # Если subgraph содержит только один узел, дублируем для batching
        if len(subgraph_embeddings.shape) == 1:
            subgraph_embeddings = subgraph_embeddings.reshape(1, -1)
        
        # Убеждаемся что embedding_dim соответствует модели
        expected_dim = 768
        if subgraph_embeddings.shape[1] != expected_dim:
            logger.warning(f"[GraphStateInjector] Unexpected dim {subgraph_embeddings.shape[1]}, expected {expected_dim}")
            # Паддинг или обрезка
            if subgraph_embeddings.shape[1] < expected_dim:
                pad_width = ((0, 0), (0, expected_dim - subgraph_embeddings.shape[1]))
                subgraph_embeddings = np.pad(subgraph_embeddings, pad_width, mode='constant')
            else:
                subgraph_embeddings = subgraph_embeddings[:, :expected_dim]
        
        # Get device from model
        import torch
        encoder_device = None
        if self.gnn_encoder is not None:
            try:
                encoder_device = next(self.gnn_encoder.parameters()).device
            except:
                encoder_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        with torch.no_grad():
            embeddings_tensor = torch.from_numpy(subgraph_embeddings).float()
            # Ensure tensor is on same device as model
            if encoder_device is not None:
                embeddings_tensor = embeddings_tensor.to(encoder_device)
            
            if self.gnn_encoder is not None:
                graph_vec, gates = self.gnn_encoder.encode(embeddings_tensor)
                graph_vector = graph_vec.cpu().numpy()
                gate_weights = gates.cpu().numpy()
            else:
                # Fallback: усреднение embeddings
                graph_vector = subgraph_embeddings.mean(axis=0)
                if len(graph_vector) > 256:
                    graph_vector = graph_vector[:256]
                elif len(graph_vector) < 256:
                    graph_vector = np.pad(graph_vector, (0, 256 - len(graph_vector)))
                gate_weights = np.ones(36, dtype=np.float32) * 0.5
        
        # Нормализация если настроена
        if self.config.normalize_graph_vector and self._graph_mean is not None:
            graph_vector = (graph_vector - self._graph_mean) / (self._graph_std + 1e-8)
        
        self._last_gate_values = gate_weights
        return graph_vector.astype(np.float32), gate_weights.astype(np.float32)
    
    def _create_value_correction(
        self,
        graph_vector: np.ndarray,
        layer_idx: int,
        gate_weight: float
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Создать функцию коррекции Value тензоров.
        
        Согласно EVA.txt: "apply_to_values — для добавления графовой коррекции"
        
        Args:
            graph_vector: [256] графовый вектор
            layer_idx: индекс слоя (0-35)
            gate_weight: вес гейта для этого слоя
            
        Returns:
            Функция f(value_array) -> corrected_value_array
        """
        # Проекция graph_vector в размерность value тензора
        value_dim = 128  # head_dim для Qwen
        if len(graph_vector) != value_dim:
            # Проекция через матрицу (в реальной реализации - обучаемая)
            projection = np.eye(value_dim, len(graph_vector), dtype=np.float32)[:value_dim, :value_dim]
            projected = np.dot(graph_vector[:value_dim], projection)
        else:
            projected = graph_vector[:value_dim]
        
        def correction_fn(value_array: np.ndarray) -> np.ndarray:
            """
            Применить графовую коррекцию к Value тензору.
            
            value_array shape: [batch, kv_heads, seq_len, head_dim] или similar
            
            Формула: value_corrected = value + strength * gate_weight * projected
            """
            if value_array is None or len(value_array) == 0:
                return value_array
            
            # Добавляем измерения для broadcasting
            projected_expanded = projected.reshape(1, 1, 1, -1)
            
            # Применяем коррекцию
            strength = self.config.graph_correction_strength * gate_weight
            corrected = value_array + strength * projected_expanded
            
            return corrected
        
        return correction_fn
    
    def _create_key_scaling(
        self,
        gate_weight: float,
        layer_idx: int
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Создать функцию масштабирования Key тензоров.
        
        Согласно EVA.txt: "apply_to_keys применяется ко всем 36 слоям 
        для масштабирования Key"
        
        Args:
            gate_weight: вес гейта для этого слоя
            layer_idx: индекс слоя
            
        Returns:
            Функция f(key_array) -> scaled_key_array
        """
        def scaling_fn(key_array: np.ndarray) -> np.ndarray:
            """
            Масштабировать Key тензор.
            
            Формула: key_scaled = key * (1.0 + strength * (gate_weight - 0.5))
            
            Гейты около 0.5 не влияют, около 1.0 усиливают, около 0.0 ослабляют.
            """
            if key_array is None or len(key_array) == 0:
                return key_array
            
            strength = self.config.key_scaling_strength
            # gate_weight - 0.5: положительный для >0.5, отрицательный для <0.5
            scale_factor = 1.0 + strength * (gate_weight - 0.5)
            
            return key_array * scale_factor
        
        return scaling_fn
    
    def inject_graph(
        self,
        subgraph_embeddings: Optional[np.ndarray] = None,
        apply_kca: bool = True
    ) -> Dict[str, Any]:
        """
        Полнослойная инъекция графа в KV-кеш.
        
        Согласно EVA.txt раздел 9.3:
        "Графовый вектор вплавляется в Value-тензоры всех 36 слоёв.
        На каждом слое работает KCA: обнаруживает лакуны и противоречия..."
        
        Args:
            subgraph_embeddings: [num_nodes, 768] эмбеддинги подграфа
            apply_kca: применять ли KCA для обнаружения лакун/противоречий
            
        Returns:
            Dict с результатами инъекции и метриками
        """
        results = {
            "injection_count": self._injection_count,
            "layers_processed": 0,
            "gate_weights": None,
            "kca_results": None,
            "early_exit_triggered": False,
            "activation_gate_value": 0.0
        }
        
        # 1. Вычислить graph_vector и gate_weights
        graph_vector, gate_weights = self.compute_graph_vector(subgraph_embeddings)
        results["gate_weights"] = gate_weights.tolist()
        
        # 2. Получить список слоёв
        layer_indices = self.state_injector.get_all_layer_indices()
        if not layer_indices:
            logger.warning("[GraphStateInjector] No layers available in state_injector")
            return results
        
        # 3. KCA: обнаружение лакун и противоречий (если включено)
        kca_corrections = {}
        if apply_kca and self.kca_detector is not None:
            try:
                kca_results = self.kca_detector.detect(
                    graph_vector=graph_vector,
                    layer_indices=layer_indices,
                    gate_weights=gate_weights
                )
                results["kca_results"] = kca_results
                kca_corrections = kca_results.get("corrections", {})
            except Exception as e:
                logger.warning(f"[GraphStateInjector] KCA failed: {e}")
        
        # 4. Применить инъекцию ко всем слоям
        accumulated_gate = 0.0
        layers_to_process = min(len(layer_indices), self.config.num_layers)
        
        for i, layer_idx in enumerate(layer_indices[:layers_to_process]):
            gate = gate_weights[i] if i < len(gate_weights) else 0.5
            
            # Apply to Keys (масштабирование)
            if self.config.use_gate_weights:
                key_scaling_fn = self._create_key_scaling(gate, layer_idx)
                try:
                    current_key = self.state_injector.get_key(layer_idx)
                    if len(current_key) > 0:
                        scaled_key = key_scaling_fn(current_key)
                        self.state_injector.set_key(layer_idx, scaled_key)
                except Exception as e:
                    logger.debug(f"[GraphStateInjector] Key injection failed at layer {layer_idx}: {e}")
            
            # Apply to Values (графовая коррекция)
            value_correction_fn = self._create_value_correction(graph_vector, layer_idx, gate)
            try:
                current_value = self.state_injector.get_value(layer_idx)
                if len(current_value) > 0:
                    # Применить KCA коррекцию если есть
                    correction = kca_corrections.get(layer_idx, {})
                    if correction:
                        # Добавить корректирующий эмбеддинг от KCA
                        kca_embedding = correction.get("embedding", np.zeros(128))
                        correction_fn = self._create_value_correction(kca_embedding, layer_idx, gate)
                        current_value = correction_fn(current_value)
                    
                    corrected_value = value_correction_fn(current_value)
                    self.state_injector.set_value(layer_idx, corrected_value)
            except Exception as e:
                logger.debug(f"[GraphStateInjector] Value injection failed at layer {layer_idx}: {e}")
            
            # Accumulate gate для Activation Gate
            accumulated_gate += gate
            results["layers_processed"] += 1
        
        # 5. Проверка Activation Gate
        avg_gate = accumulated_gate / max(1, results["layers_processed"])
        results["activation_gate_value"] = avg_gate
        
        if avg_gate >= self.config.activation_gate_threshold:
            results["early_exit_triggered"] = True
            logger.info(f"[GraphStateInjector] Activation Gate triggered: {avg_gate:.3f} >= {self.config.activation_gate_threshold}")
        
        self._injection_count += 1
        return results
    
    def get_injection_stats(self) -> Dict[str, Any]:
        """Получить статистику инъекций."""
        return {
            "injection_count": self._injection_count,
            "last_gate_weights": self._last_gate_values.tolist() if self._last_gate_values is not None else None,
            "num_layers": self.config.num_layers,
            "config": {
                "graph_correction_strength": self.config.graph_correction_strength,
                "key_scaling_strength": self.config.key_scaling_strength,
                "activation_gate_threshold": self.config.activation_gate_threshold
            }
        }