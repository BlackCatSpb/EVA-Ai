#!/usr/bin/env python3
"""
Re-export stub for backward compatibility.
Main code has been split into:
- unit_core.py - Main class MLUnit, initialization, lifecycle
- unit_components.py - Component management, coordination
- unit_training.py - Training orchestration, model updates
"""

from .unit_core import MLUnit, _load_brain_config, _get_hybrid_cache_config, _get_project_root

__all__ = [
    "MLUnit",
    "_load_brain_config",
    "_get_hybrid_cache_config",
    "_get_project_root",
]
