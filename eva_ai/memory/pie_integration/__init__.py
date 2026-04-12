"""
Pie Integration Module for EVA AI

Интеграция Eva Pie Architecture в существующую систему EVA.
Предоставляет fallback механизм когда GGUF pipeline недоступен.
"""

from .fractal_graph_l1_l2 import (
    FractalGraphL1L2,
    create_l1l2_graph,
    ActivationProfileData,
    RoutingRuleData
)
from .activation_profiler import (
    ActivationProfiler,
    create_default_profiler,
    ProfileStats
)
from .routing_engine import (
    RoutingEngine,
    create_default_engine,
    RoutingParams
)
from .pie_adapter import (
    PieIntegration,
    create_pie_integration,
    GenerationMetadata
)

__all__ = [
    # L1/L2 Graph
    'FractalGraphL1L2',
    'create_l1l2_graph',
    'ActivationProfileData',
    'RoutingRuleData',
    # L1: Activation Profiler
    'ActivationProfiler',
    'create_default_profiler',
    'ProfileStats',
    # L2: Routing Engine
    'RoutingEngine',
    'create_default_engine',
    'RoutingParams',
    # Integration Adapter
    'PieIntegration',
    'create_pie_integration',
    'GenerationMetadata',
]
