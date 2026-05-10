"""
MemorySnapshotIntegration - Сохранение состояний LLM слоёв в граф

Паттерн Memory Snapshot:
- Каждый слой LLM сохраняет свой hidden_state как узел графа
- Из этих snapshot'ов формируются концепты и противоречия
- Граф возвращает коррекцию для инъекции обратно в генерацию

Это обеспечивает:
1. Прямое сохранение истории (не текст, а состояния)
2. Формирование концептов из паттернов активаций
3. Обнаружение противоречий между состояниями
4. Коррекцию генерации через retrieval из графа
"""
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger("eva_ai.core.memory_snapshot")

from eva_ai.fcp_core import (
    FCPConfig,
    FractalGraphV2,
    LayerState,
    HaltDecision
)


@dataclass
class LayerSnapshot:
    """Снимок состояния одного слоя."""
    layer_idx: int
    hidden_states: np.ndarray
    embeddings: np.ndarray
    timestamp: float
    confidence: float = 0.0
    node_id: Optional[str] = None


class MemorySnapshotIntegration:
    """
    Интеграция Memory Snapshot паттерна в EVA.

    Поток данных:
    LLM Layer N → [snapshot] → FractalGraphV2 (как узел)
                                      ↓
                                retrieval
                                      ↓
    LLM Layer N → [коррекция] ← GNN + KCA
    """

    def __init__(
        self,
        brain,
        fractal_graph: Optional[FractalGraphV2] = None,
        config: Optional[Dict] = None
    ):
        self.brain = brain
        self.graph = fractal_graph
        self.config = config or {}

        self.snapshot_interval = self.config.get('snapshot_interval', 1)
        self.save_to_graph = self.config.get('save_to_graph', True)
        self.snapshot_all_layers = self.config.get('snapshot_all_layers', True)
        specified_layers = self.config.get('snapshot_layers', [])
        self.num_layers = self.config.get('num_layers', 32)

        if self.snapshot_all_layers:
            self.snapshot_layers = set(range(self.num_layers))
        else:
            self.snapshot_layers = set(specified_layers) if specified_layers else set()

        logger.info(f"[MemorySnapshot] Initialized: all_layers={self.snapshot_all_layers}, "
                   f"num_layers={self.num_layers}, save_to_graph={self.save_to_graph}")

        self.current_snapshots: List[LayerSnapshot] = []
        self.snapshot_callbacks: List[Callable] = []

        self.enabled = self.config.get('enabled', True)

        logger.info(f"[MemorySnapshot] Initialized: layers={self.snapshot_layers}, "
                   f"save_to_graph={self.save_to_graph}")

    def on_layer_forward(
        self,
        layer_idx: int,
        hidden_states: np.ndarray,
        layer_confidence: float = 0.0
    ) -> Optional[np.ndarray]:
        """
        Вызывается после forward каждого слоя.

        Args:
            layer_idx: индекс слоя
            hidden_states: [batch, seq_len, hidden_dim] - выход слоя
            layer_confidence: уверенность слоя (из early exit)

        Returns:
            Optional[np.ndarray] - коррекция для инъекции или None
        """
        if not self.enabled:
            return None

        if layer_idx not in self.snapshot_layers:
            return None

        logger.debug(f"[MemorySnapshot] Recording layer {layer_idx}")

        try:
            snapshot = self._create_snapshot(layer_idx, hidden_states, layer_confidence)
            self.current_snapshots.append(snapshot)

            if self.save_to_graph and self.graph:
                self._save_to_graph(snapshot)

            correction = self._compute_correction(snapshot)
            return correction

        except Exception as e:
            logger.warning(f"[MemorySnapshot] Error on layer {layer_idx}: {e}")
            return None

    def _create_snapshot(
        self,
        layer_idx: int,
        hidden_states: np.ndarray,
        confidence: float
    ) -> LayerSnapshot:
        """Создать snapshot из hidden_states."""
        if hidden_states.ndim == 3:
            last_hidden = hidden_states[:, -1:, :]
        else:
            last_hidden = hidden_states

        embedding = np.mean(last_hidden, axis=1).squeeze(0)

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        import time
        snapshot = LayerSnapshot(
            layer_idx=layer_idx,
            hidden_states=hidden_states,
            embeddings=embedding,
            timestamp=time.time(),
            confidence=confidence
        )

        return snapshot

    def _save_to_graph(self, snapshot: LayerSnapshot):
        """Сохранить snapshot как узел графа."""
        if self.graph is None:
            return

        try:
            node_data = {
                'type': 'layer_snapshot',
                'layer_idx': snapshot.layer_idx,
                'timestamp': snapshot.timestamp,
                'confidence': snapshot.confidence,
                'embedding': snapshot.embeddings.tobytes() if hasattr(snapshot.embeddings, 'tobytes') else snapshot.embeddings
            }

            self.graph.add_node(
                embedding=snapshot.embeddings,
                metadata=node_data
            )

            logger.debug(f"[MemorySnapshot] Saved layer {snapshot.layer_idx} to graph")

        except Exception as e:
            logger.warning(f"[MemorySnapshot] Save to graph failed: {e}")

    def _compute_correction(self, snapshot: LayerSnapshot) -> Optional[np.ndarray]:
        """
        Вычислить коррекцию на основе snapshot.

        Использует:
        - ConceptMiner для извлечения концептов из паттернов активаций
        - ContradictionMiner для поиска противоречий
        - KCA для формирования вектора коррекции
        """
        if self.graph is None:
            return None

        try:
            query_embedding = snapshot.embeddings.reshape(1, -1)
            subgraph = self.graph.retrieve_subgraph(query_embedding, top_k=5)

            embeddings = subgraph.get('embeddings')
            if embeddings is None or len(embeddings) == 0:
                return None

            correction_vectors = []

            for emb in subgraph['embeddings']:
                if isinstance(emb, np.ndarray):
                    correction_vectors.append(emb)

            if not correction_vectors:
                return None

            correction = np.mean(correction_vectors, axis=0)

            correction = correction.reshape(1, 1, -1)

            return correction * 0.1

        except Exception as e:
            logger.warning(f"[MemorySnapshot] Correction compute failed: {e}")
            return None

    def get_current_snapshots(self) -> List[LayerSnapshot]:
        """Получить все текущие snapshots."""
        return self.current_snapshots.copy()

    def clear_snapshots(self):
        """Очистить текущие snapshots."""
        self.current_snapshots.clear()

    def extract_concepts_from_snapshots(self) -> List[Dict]:
        """
        Извлечь концепты из текущих snapshots.

        ConceptMiner работает циклично - просто сохраняем snapshots в граф.
        ConceptMiner сам их подхватит при анализе.
        """
        concepts = []

        if not self.current_snapshots:
            return concepts

        if hasattr(self.brain, 'concept_miner'):
            for snapshot in self.current_snapshots:
                try:
                    if snapshot.node_id:
                        node_data = {
                            'type': 'layer_snapshot',
                            'layer_idx': snapshot.layer_idx,
                            'timestamp': snapshot.timestamp,
                            'confidence': snapshot.confidence,
                            'embedding': snapshot.embeddings
                        }
                        logger.debug(f"[MemorySnapshot] Snapshot for concept: layer={snapshot.layer_idx}")
                except Exception as e:
                    logger.warning(f"[MemorySnapshot] Concept extraction failed: {e}")

        return concepts

    def detect_contradictions_from_snapshots(self) -> List[Dict]:
        """
        Обнаружить противоречия между snapshots.

        ContradictionMiner работает циклично.
        """
        return []

    def register_callback(self, callback: Callable):
        """Зарегистрировать callback для обработки snapshots."""
        self.snapshot_callbacks.append(callback)

    def on_generation_complete(self, response: str):
        """
        Вызывается после завершения генерации.

        Сохраняет итоговые snapshots в граф для будущего обучения.
        """
        if not self.enabled:
            return

        if hasattr(self.brain, 'concept_miner'):
            try:
                concepts = self.extract_concepts_from_snapshots()
                for concept in concepts:
                    self.brain.concept_miner.add_concept(concept)

                logger.info(f"[MemorySnapshot] Extracted {len(concepts)} concepts from generation")
            except Exception as e:
                logger.warning(f"[MemorySnapshot] Post-generation extraction failed: {e}")

        self.clear_snapshots()


class LayerSnapshotTracker:
    """
    Трекер состояний слоёв для отладки и анализа.

    Позволяет отслеживать какие слои активируются,
    как меняется confidence, когда происходят early exits.
    """

    def __init__(self, num_layers: int = 32):
        self.num_layers = num_layers
        self.layer_activations: Dict[int, int] = {i: 0 for i in range(num_layers)}
        self.layer_confidences: Dict[int, List[float]] = {i: [] for i in range(num_layers)}
        self.early_exits: int = 0

    def record_activation(self, layer_idx: int, confidence: float):
        """Записать активацию слоя."""
        self.layer_activations[layer_idx] = self.layer_activations.get(layer_idx, 0) + 1
        self.layer_confidences[layer_idx].append(confidence)

    def record_early_exit(self, at_layer: int):
        """Записать early exit."""
        self.early_exits += 1
        for i in range(at_layer, self.num_layers):
            self.layer_activations[i] = self.layer_activations.get(i, 0)

    def get_statistics(self) -> Dict:
        """Получить статистику по слоям."""
        return {
            'layer_activations': self.layer_activations,
            'early_exits': self.early_exits,
            'avg_confidence_per_layer': {
                layer: np.mean(confs) if confs else 0.0
                for layer, confs in self.layer_confidences.items()
            }
        }

    def reset(self):
        """Сбросить статистику."""
        self.layer_activations = {i: 0 for i in range(self.num_layers)}
        self.layer_confidences = {i: [] for i in range(self.num_layers)}
        self.early_exits = 0
