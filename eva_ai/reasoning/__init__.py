"""
ЕВА Reasoning - Self-Reasoning Engine и Fractal Storage
"""

from .self_reasoning_engine import SelfReasoningEngine, create_reasoning_engine
from .confidence_scorer import calculate_overall_confidence, should_terminate, CONFIDENCE_THRESHOLD, get_confidence_level
from .clarification_generator import ClarificationGenerator
from .integration import ReasoningIntegration, integrate_reasoning

from .fractal_ml import FractalStorage, FractalNode, FractalNodeType, FractalRetriever, FractalEmbedder

__all__ = [
    # Self-Reasoning
    'SelfReasoningEngine',
    'create_reasoning_engine',
    'calculate_overall_confidence',
    'should_terminate',
    'CONFIDENCE_THRESHOLD',
    'get_confidence_level',
    'ClarificationGenerator',
    'ReasoningIntegration',
    'integrate_reasoning',
    
    # Fractal Storage
    'FractalStorage',
    'FractalNode',
    'FractalNodeType',
    'FractalRetriever',
    'FractalEmbedder',
]

# Параметры из DESIGN.md
MAX_ITERATIONS = 5
MAX_LEVELS = 4
BRANCHING_FACTOR = 16
EMBEDDING_DIM = 384
