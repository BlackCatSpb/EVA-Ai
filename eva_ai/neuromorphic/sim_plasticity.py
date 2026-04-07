"""Нейроморфный симулятор — правила обучения, STDP, адаптация."""
import logging
import numpy as np
from typing import Dict, Any, List, Optional

logger = logging.getLogger("eva_ai.neuromorphic")


class STDPPlasticity:
    """Spike-Timing-Dependent Plasticity (STDP) реализация."""

    def __init__(self, learning_rate: float = 0.01, tau_plus: float = 0.02, tau_minus: float = 0.02):
        self.learning_rate = learning_rate
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.weight_updates = []

    def compute_weight_change(self, pre_spike_time: float, post_spike_time: float,
                               current_weight: float) -> float:
        dt = post_spike_time - pre_spike_time

        if dt > 0:
            delta = self.learning_rate * np.exp(-dt / self.tau_plus)
        elif dt < 0:
            delta = -self.learning_rate * np.exp(dt / self.tau_minus)
        else:
            delta = 0.0

        new_weight = np.clip(current_weight + delta, 0.0, 1.0)
        actual_change = new_weight - current_weight
        self.weight_updates.append({
            'pre_time': pre_spike_time,
            'post_time': post_spike_time,
            'dt': dt,
            'delta': actual_change,
            'new_weight': new_weight
        })

        if len(self.weight_updates) > 1000:
            self.weight_updates = self.weight_updates[-1000:]

        return actual_change

    def get_stats(self) -> Dict[str, Any]:
        if not self.weight_updates:
            return {"total_updates": 0, "mean_delta": 0.0, "potentiation_count": 0, "depression_count": 0}

        deltas = [u['delta'] for u in self.weight_updates]
        return {
            "total_updates": len(self.weight_updates),
            "mean_delta": float(np.mean(deltas)),
            "std_delta": float(np.std(deltas)),
            "potentiation_count": sum(1 for d in deltas if d > 0),
            "depression_count": sum(1 for d in deltas if d < 0)
        }


class AdaptiveThreshold:
    """Адаптивный порог активации нейронов."""

    def __init__(self, base_threshold: float = 0.5, adaptation_rate: float = 0.01,
                 decay_rate: float = 0.001):
        self.base_threshold = base_threshold
        self.current_threshold = base_threshold
        self.adaptation_rate = adaptation_rate
        self.decay_rate = decay_rate
        self.recent_activity = []

    def update(self, activity_level: float):
        self.recent_activity.append(activity_level)
        if len(self.recent_activity) > 100:
            self.recent_activity.pop(0)

        if len(self.recent_activity) >= 10:
            avg_activity = np.mean(self.recent_activity[-10:])
            if avg_activity > 0.7:
                self.current_threshold += self.adaptation_rate
            elif avg_activity < 0.3:
                self.current_threshold -= self.adaptation_rate

        self.current_threshold += (self.base_threshold - self.current_threshold) * self.decay_rate
        self.current_threshold = np.clip(self.current_threshold, 0.1, 0.9)

    def get_threshold(self) -> float:
        return self.current_threshold

    def reset(self):
        self.current_threshold = self.base_threshold
        self.recent_activity.clear()


class HomeostaticPlasticity:
    """Гомеостатическая пластичность для поддержания стабильности сети."""

    def __init__(self, target_activity: float = 0.5, adaptation_timescale: float = 100.0):
        self.target_activity = target_activity
        self.adaptation_timescale = adaptation_timescale
        self.activity_history = []

    def adjust_synaptic_scaling(self, current_activity: float, current_weights: np.ndarray) -> np.ndarray:
        self.activity_history.append(current_activity)
        if len(self.activity_history) > int(self.adaptation_timescale):
            self.activity_history.pop(0)

        if len(self.activity_history) >= 10:
            avg_activity = np.mean(self.activity_history[-10:])
            error = self.target_activity - avg_activity
            scaling_factor = 1.0 + error * 0.1
            new_weights = current_weights * scaling_factor
            return np.clip(new_weights, 0.0, 1.0)

        return current_weights

    def get_stats(self) -> Dict[str, Any]:
        if not self.activity_history:
            return {"target_activity": self.target_activity, "current_avg": 0.0, "samples": 0}

        return {
            "target_activity": self.target_activity,
            "current_avg": float(np.mean(self.activity_history[-10:])),
            "samples": len(self.activity_history),
            "stability": 1.0 - float(np.std(self.activity_history[-10:]))
        }
