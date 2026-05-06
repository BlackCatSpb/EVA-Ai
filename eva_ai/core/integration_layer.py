"""
Интеграционный слой ЕВА - Re-exports from split modules.

This module now serves as a facade that re-exports the main class
from the refactored sub-modules for backward compatibility.

Modules:
- integration_core.py - Main class ЕВАIntegrator, initialization, lifecycle
- integration_adapters.py - Adapter classes for different subsystems
- integration_events.py - Event handling, pub/sub integration
- integration_sync.py - Synchronization, data flow management
"""

from .integration_core import ЕВАIntegrator, IntegrationLayer
from .integration_adapters import (
    _handle_query_received,
    _handle_tokenize_request,
    _handle_tokens_ready,
    _handle_hot_window_ready,
    _handle_response_generated,
    _handle_contradiction_detected,
    _handle_learning_opportunity,
    _handle_self_dialog_request,
    _handle_ethical_check_request,
    _tokenize_text,
    _enhance_prompt_with_reasoning,
)
from .integration_events import (
    _setup_event_subscriptions,
    _handle_pipeline_start,
    _handle_pipeline_model_a_complete,
    _handle_pipeline_web_search_complete,
    _handle_pipeline_contradiction_check_complete,
    _handle_pipeline_ethics_check_complete,
    _handle_pipeline_model_b_complete,
    _handle_pipeline_relevance_check_complete,
    _handle_pipeline_refinement_needed,
    _handle_pipeline_refinement_attempt,
    _handle_pipeline_complete,
    _handle_pipeline_failed,
)
from .integration_sync import (
    _learning_scheduler_worker,
    _system_optimizer_worker,
    _health_monitor_worker,
)

__all__ = [
    'ЕВАIntegrator',
    'IntegrationLayer',
]
