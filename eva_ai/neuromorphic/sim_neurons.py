"""Нейроморфный симулятор — симуляция нейронов, функции активации."""
import logging
import numpy as np
from typing import Dict, Any, Optional

logger = logging.getLogger("eva_ai.neuromorphic")


class FallbackNeuralNetwork:
    """Альтернативная реализация нейронной сети без NEST."""

    def __init__(self, num_neurons: int, memory_type: str):
        self.num_neurons = num_neurons
        self.memory_type = memory_type
        self.neuron_states = np.random.rand(num_neurons)
        self.connections = np.random.rand(num_neurons, num_neurons)
        self.activity_history = []
        self.simulation_steps = 0
        logger.info(f"Альтернативная нейронная сеть инициализирована для {memory_type} памяти с {num_neurons} нейронами")

    def simulate_step(self, input_stimulus: Optional[np.ndarray] = None) -> np.ndarray:
        if input_stimulus is not None and len(input_stimulus) == self.num_neurons:
            self.neuron_states = np.maximum(self.neuron_states, input_stimulus)

        influence = np.dot(self.connections, self.neuron_states)
        noise = np.random.normal(0, 0.01, self.num_neurons)
        new_states = self.neuron_states + influence * 0.1 + noise
        self.neuron_states = np.clip(new_states, 0.0, 1.0)

        self.activity_history.append(self.neuron_states.copy())
        if len(self.activity_history) > 100:
            self.activity_history.pop(0)
        self.simulation_steps += 1
        return self.neuron_states

    def get_activity_pattern(self) -> list:
        return self.neuron_states.tolist()

    def connect_neurons(self, source: int, target: int, weight: float = 0.5):
        if 0 <= source < self.num_neurons and 0 <= target < self.num_neurons:
            self.connections[target, source] = max(0.0, min(1.0, weight))
        else:
            logger.warning(f"Неверные индексы нейронов для соединения: {source} -> {target}")

    def get_network_info(self) -> Dict[str, Any]:
        return {
            "num_neurons": self.num_neurons,
            "memory_type": self.memory_type,
            "simulation_steps": self.simulation_steps,
            "activity_history_length": len(self.activity_history)
        }
