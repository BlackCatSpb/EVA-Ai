"""
Recursive Model Pipeline - Re-exports from split modules.

This module now serves as a facade that re-exports the main class
from the refactored sub-modules for backward compatibility.

Modules:
- pipeline_core.py - Main class RecursiveModelPipeline, process_query(), load_models(), unload_models()
- pipeline_quality.py - check_quality(), _sanitize_response, _clean_filler_start, _remove_looping_blocks, _generate_with_timeout
- pipeline_adaptive.py - AdaptiveParameterController, semantic stuck detection, parameter adaptation
- pipeline_models.py - generate_with_model_a, generate_with_model_b
"""

from .pipeline_core import RecursiveModelPipeline, create_recursive_pipeline
from .pipeline_adaptive import AdaptiveParameterController
from .pipeline_quality import (
    check_quality,
    _sanitize_response,
    _clean_filler_start,
    _remove_looping_blocks,
    _generate_with_timeout,
)
from .pipeline_models import (
    generate_with_model_a,
    generate_with_model_b,
)

__all__ = [
    'RecursiveModelPipeline',
    'create_recursive_pipeline',
    'AdaptiveParameterController',
    'check_quality',
    '_sanitize_response',
    '_clean_filler_start',
    '_remove_looping_blocks',
    '_generate_with_timeout',
    'generate_with_model_a',
    'generate_with_model_b',
]
