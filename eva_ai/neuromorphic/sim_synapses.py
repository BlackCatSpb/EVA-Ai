"""Нейроморфный симулятор — управление синапсами, пластичность."""
import logging
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger("eva_ai.neuromorphic")


class SynapseManager:
    """Управление синаптическими связями и пластичностью."""

    def __init__(self, num_neurons: int):
        self.num_neurons = num_neurons
        self.weights = np.random.rand(num_neurons, num_neurons) * 0.5
        self.delays = np.ones((num_neurons, num_neurons))
        self.plasticity_enabled = True

    def update_weights(self, pre_spike_times: List[float], post_spike_times: List[float],
                       learning_rate: float = 0.01):
        if not self.plasticity_enabled:
            return

        for pre_t in pre_spike_times:
            for post_t in post_spike_times:
                dt = post_t - pre_t
                if abs(dt) < 0.05:
                    if dt > 0:
                        delta = learning_rate * np.exp(-dt / 0.02)
                    else:
                        delta = -learning_rate * np.exp(dt / 0.02)
                    self.weights = np.clip(self.weights + delta * 0.01, 0.0, 1.0)

    def get_connection_strength(self, source: int, target: int) -> float:
        if 0 <= source < self.num_neurons and 0 <= target < self.num_neurons:
            return float(self.weights[target, source])
        return 0.0

    def prune_weak_connections(self, threshold: float = 0.05):
        mask = self.weights < threshold
        self.weights[mask] = 0.0
        pruned = int(np.sum(mask))
        if pruned > 0:
            logger.debug(f"Удалено {pruned} слабых синаптических связей")
        return pruned

    def get_stats(self) -> Dict[str, Any]:
        non_zero = int(np.count_nonzero(self.weights))
        total = self.num_neurons * self.num_neurons
        return {
            "total_synapses": total,
            "active_synapses": non_zero,
            "connectivity": non_zero / total if total > 0 else 0.0,
            "mean_weight": float(np.mean(self.weights[self.weights > 0])) if non_zero > 0 else 0.0,
            "plasticity_enabled": self.plasticity_enabled
        }
