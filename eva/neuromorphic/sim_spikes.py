"""Нейроморфный симулятор — генерация спайков, распространение."""
import logging
import time
import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass, field

logger = logging.getLogger("eva.neuromorphic")


@dataclass
class SpikeEvent:
    """Представляет событие спайка."""
    timestamp: float = field(default_factory=time.time)
    neuron_id: int = 0
    amplitude: float = 1.0
    source: str = "simulation"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "neuron_id": self.neuron_id,
            "amplitude": self.amplitude,
            "source": self.source,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SpikeEvent':
        return cls(
            timestamp=data.get("timestamp", time.time()),
            neuron_id=data.get("neuron_id", 0),
            amplitude=data.get("amplitude", 1.0),
            source=data.get("source", "simulation"),
            metadata=data.get("metadata", {})
        )


@dataclass
class NeuralActivity:
    """Представляет данные о нейронной активности."""
    timestamp: float = field(default_factory=time.time)
    activity_pattern: List[float] = field(default_factory=list)
    memory_type: str = "working"
    strength: float = 0.5
    importance: float = 0.5
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "activity_pattern": self.activity_pattern,
            "memory_type": self.memory_type,
            "strength": self.strength,
            "importance": self.importance,
            "context": self.context,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NeuralActivity':
        return cls(
            timestamp=data["timestamp"],
            activity_pattern=data["activity_pattern"],
            memory_type=data["memory_type"],
            strength=data["strength"],
            importance=data["importance"],
            context=data["context"],
            metadata=data["metadata"]
        )

    def get_analysis(self) -> Dict[str, Any]:
        if not self.activity_pattern:
            return {"status": "no_data"}

        mean_activity = np.mean(self.activity_pattern)
        std_activity = np.std(self.activity_pattern)
        max_activity = np.max(self.activity_pattern)
        min_activity = np.min(self.activity_pattern)
        coherence = 1.0 - (std_activity / (mean_activity + 1e-8))
        coherence = max(0.0, min(1.0, coherence))

        return {
            "mean_activity": float(mean_activity),
            "std_activity": float(std_activity),
            "max_activity": float(max_activity),
            "min_activity": float(min_activity),
            "coherence": float(coherence),
            "pattern_length": len(self.activity_pattern),
            "strength": self.strength,
            "importance": self.importance
        }


class SpikeGenerator:
    """Генератор спайков на основе активности нейронов."""

    def __init__(self, threshold: float = 0.5, refractory_period: float = 0.005):
        self.threshold = threshold
        self.refractory_period = refractory_period
        self.last_spike_times: Dict[int, float] = {}
        self.spike_history: List[SpikeEvent] = []

    def generate_spikes(self, neuron_states: np.ndarray, current_time: float) -> List[SpikeEvent]:
        spikes = []
        for neuron_id, state in enumerate(neuron_states):
            if state >= self.threshold:
                last_spike = self.last_spike_times.get(neuron_id, 0.0)
                if current_time - last_spike >= self.refractory_period:
                    spike = SpikeEvent(
                        timestamp=current_time,
                        neuron_id=neuron_id,
                        amplitude=state,
                        source="threshold_crossing"
                    )
                    spikes.append(spike)
                    self.last_spike_times[neuron_id] = current_time

        self.spike_history.extend(spikes)
        if len(self.spike_history) > 10000:
            self.spike_history = self.spike_history[-10000:]

        return spikes

    def get_spike_rate(self, window: float = 1.0, current_time: float = None) -> Dict[int, float]:
        if current_time is None:
            current_time = time.time()

        rates = {}
        recent_spikes = [s for s in self.spike_history if current_time - s.timestamp <= window]

        neuron_counts: Dict[int, int] = {}
        for spike in recent_spikes:
            neuron_counts[spike.neuron_id] = neuron_counts.get(spike.neuron_id, 0) + 1

        for neuron_id, count in neuron_counts.items():
            rates[neuron_id] = count / window

        return rates

    def get_stats(self) -> Dict[str, Any]:
        if not self.spike_history:
            return {"total_spikes": 0, "unique_neurons": 0, "mean_rate": 0.0}

        unique_neurons = len(set(s.neuron_id for s in self.spike_history))
        if len(self.spike_history) >= 2:
            duration = self.spike_history[-1].timestamp - self.spike_history[0].timestamp
            mean_rate = len(self.spike_history) / max(duration, 0.001)
        else:
            mean_rate = 0.0

        return {
            "total_spikes": len(self.spike_history),
            "unique_neurons": unique_neurons,
            "mean_rate": mean_rate,
            "threshold": self.threshold
        }


class SpikePropagator:
    """Распространение спайков по нейронной сети."""

    def __init__(self, connection_matrix: np.ndarray, propagation_delay: float = 0.001):
        self.connections = connection_matrix
        self.propagation_delay = propagation_delay
        self.propagation_history = []

    def propagate(self, source_spikes: List[SpikeEvent], current_time: float) -> List[SpikeEvent]:
        propagated = []

        for spike in source_spikes:
            neuron_id = spike.neuron_id
            if neuron_id < self.connections.shape[0]:
                connection_strengths = self.connections[:, neuron_id]
                target_neurons = np.where(connection_strengths > 0.1)[0]

                for target_id in target_neurons:
                    strength = connection_strengths[target_id]
                    propagated_spike = SpikeEvent(
                        timestamp=current_time + self.propagation_delay,
                        neuron_id=int(target_id),
                        amplitude=spike.amplitude * strength,
                        source=f"propagated_from_{neuron_id}",
                        metadata={"original_spike": spike.to_dict(), "strength": float(strength)}
                    )
                    propagated.append(propagated_spike)

        self.propagation_history.extend(propagated)
        if len(self.propagation_history) > 5000:
            self.propagation_history = self.propagation_history[-5000:]

        return propagated

    def get_propagation_stats(self) -> Dict[str, Any]:
        if not self.propagation_history:
            return {"total_propagated": 0, "mean_amplitude": 0.0, "unique_targets": 0}

        amplitudes = [s.amplitude for s in self.propagation_history]
        unique_targets = len(set(s.neuron_id for s in self.propagation_history))

        return {
            "total_propagated": len(self.propagation_history),
            "mean_amplitude": float(np.mean(amplitudes)),
            "max_amplitude": float(np.max(amplitudes)),
            "unique_targets": unique_targets
        }
