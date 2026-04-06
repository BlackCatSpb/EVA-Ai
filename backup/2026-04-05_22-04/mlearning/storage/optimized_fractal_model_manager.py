"""
Re-export stub for backward compatibility.
Main code has been split into:
- opt_core.py - Main class OptimizedFractalModelManager, initialization, lifecycle
- opt_models.py - Model management, loading, unloading
- opt_cache.py - Caching, optimization strategies
"""
from .opt_core import OptimizedFractalModelManager

__all__ = ["OptimizedFractalModelManager"]
