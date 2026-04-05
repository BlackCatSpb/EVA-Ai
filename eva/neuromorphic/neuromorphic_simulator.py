"""Модуль нейроморфного симулятора для ЕВА с интеграцией фрактального хранилища.

Модуль был разделён на:
- sim_core.py — Main class, initialization, lifecycle
- sim_neurons.py — Neuron simulation, activation functions
- sim_synapses.py — Synapse management, plasticity
- sim_plasticity.py — Learning rules, STDP, adaptation
- sim_spikes.py — Spike generation, propagation
"""
from .sim_core import NeuromorphicSimulator
from .sim_neurons import FallbackNeuralNetwork
from .sim_spikes import NeuralActivity, SpikeEvent, SpikeGenerator, SpikePropagator
from .sim_synapses import SynapseManager
from .sim_plasticity import STDPPlasticity, AdaptiveThreshold, HomeostaticPlasticity

__all__ = [
    'NeuromorphicSimulator',
    'FallbackNeuralNetwork',
    'NeuralActivity',
    'SpikeEvent',
    'SpikeGenerator',
    'SpikePropagator',
    'SynapseManager',
    'STDPPlasticity',
    'AdaptiveThreshold',
    'HomeostaticPlasticity',
]
