"""
MemoryGraphML - Re-exports from split modules.

This module now serves as a facade that re-exports the main class
from the refactored sub-modules for backward compatibility.

Modules:
- graph_ml_core.py - Main class MemoryGraphML, initialization, embedding
- graph_ml_training.py - Training methods, pattern learning
- graph_ml_inference.py - Inference, prediction, similarity
- graph_ml_patterns.py - Pattern detection, clustering, analysis
"""

from .graph_ml_core import (
    MemoryGraphML,
    GraphEmbedding,
    AmbiguousEntity,
    ClarificationRequest,
    GraphPattern,
)
from .graph_ml_training import (
    generate_training_sample,
    _extract_patterns,
)

MemoryGraphML._extract_patterns = _extract_patterns
from .graph_ml_inference import (
    predict_relation,
    find_similar_nodes,
    classify_node,
)
from .graph_ml_patterns import (
    detect_clusters,
    find_frequent_patterns,
    analyze_graph_structure,
    get_pattern_insights,
)

__all__ = [
    'MemoryGraphML',
    'GraphEmbedding',
    'AmbiguousEntity',
    'ClarificationRequest',
    'GraphPattern',
    'generate_training_sample',
    '_extract_patterns',
    'predict_relation',
    'find_similar_nodes',
    'classify_node',
    'detect_clusters',
    'find_frequent_patterns',
    'analyze_graph_structure',
    'get_pattern_insights',
]
