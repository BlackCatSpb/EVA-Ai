"""
Re-export stub for backward compatibility.
Main code has been split into:
- store_core.py - Main class FractalWeightStore, initialization, lifecycle
- store_operations.py - CRUD operations, store/retrieve/update
- store_queries.py - Query processing, search, filtering
- store_cache.py - Caching strategies, hot/cold management
"""
from .store_core import (
    FractalWeightStore,
    FractalContainer,
    NodeProxy,
    EdgeProxy,
    KnowledgeGraphProxy,
    repack_model_to_fractal,
    export_hf_model_to_fractal,
)

__all__ = [
    "FractalWeightStore",
    "FractalContainer",
    "NodeProxy",
    "EdgeProxy",
    "KnowledgeGraphProxy",
    "repack_model_to_fractal",
    "export_hf_model_to_fractal",
]
