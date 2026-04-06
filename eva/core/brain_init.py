"""
Initialization orchestration helpers for CoreBrain.
Extracted to keep core_brain.py thin.
"""
import os
import time
import logging
from typing import Any, Dict, Optional

query_logger = logging.getLogger("eva.core_brain.query_processing")
logger = logging.getLogger("eva.core_brain")


def _init_fractal_final(brain):
    """Finalize fractal model initialization and set models_ready flag."""
    if brain.fractal_model_manager:
        try:
            if hasattr(brain.fractal_model_manager, 'initialized') and brain.fractal_model_manager.initialized:
                brain.fractal_ready = True
            else:
                brain.fractal_ready = brain.fractal_model_manager.initialize()
        except Exception:
            brain.fractal_ready = False
        if brain.fractal_ready and brain.events:
            brain.events.trigger('fractal_model_ready', brain.fractal_model_manager)
    ml_ready = False
    if hasattr(brain, 'ml_unit') and brain.ml_unit:
        ml_ready = getattr(brain.ml_unit, 'models_ready', False) or getattr(brain.ml_unit, 'initialized', False) or getattr(brain.ml_unit, 'running', False)
    if brain.fractal_ready or ml_ready:
        brain.models_ready = True


def _init_gen_coord(brain):
    """Initialize generation coordinator."""
    try:
        from .generation_coordinator import initialize_generation_coordinator
        brain.generation_coordinator = initialize_generation_coordinator(brain)
        brain.components['generation_coordinator'] = brain.generation_coordinator
    except Exception as e:
        query_logger.error(f"Ошибка инициализации координатора генерации: {e}", exc_info=True)
        brain.generation_coordinator = None


def _init_wikipedia(brain):
    """Initialize Wikipedia KB if enabled (search only, no auto-learn)."""
    brain.wikipedia_kb = None
    brain.wikipedia_loader = None
    wiki_config = brain.config.get('wikipedia', {})
    if wiki_config.get('enabled', False):
        try:
            from eva.knowledge.wikipedia_kb import get_wikipedia_kb
            from eva.knowledge.wikipedia_loader import get_wikipedia_loader
            brain.wikipedia_kb = get_wikipedia_kb()
            brain.wikipedia_loader = get_wikipedia_loader(brain.wikipedia_kb)
        except (ImportError, RuntimeError, OSError) as e:
            query_logger.warning(f"Wikipedia KB не инициализирована: {e}")


def _init_reasoning(brain):
    """Initialize reasoning integration."""
    try:
        from eva.reasoning.integration import ReasoningIntegration
        ri = ReasoningIntegration(brain)
        if ri.integrate_with_brain():
            brain.reasoning_integration = ri
            brain.components['reasoning_integration'] = ri
    except Exception:
        pass


def _start_post_init_services(brain):
    """Start services that must run after full initialization."""
    if hasattr(brain, 'self_dialog_learning') and brain.self_dialog_learning and hasattr(brain.self_dialog_learning, 'start'):
        try:
            brain.self_dialog_learning.start()
        except Exception as e:
            query_logger.warning(f"Failed to start SelfDialogLearningSystem: {e}")

    try:
        from eva.knowledge.graph_curator import GraphCurator
        if not (hasattr(brain, 'graph_curator') and brain.graph_curator and hasattr(brain.graph_curator, 'is_running') and brain.graph_curator.is_running()):
            brain.graph_curator = GraphCurator(brain=brain, config=brain.config.get('graph_curator', {}))
            brain.graph_curator.start()
    except Exception as e:
        query_logger.warning(f"Failed to start GraphCurator: {e}")
        brain.graph_curator = None

    try:
        from eva.training.gguf_training_system import GGUFTrainingSystem
        brain.gguf_training = GGUFTrainingSystem(brain=brain, config=brain.config.get('gguf_training', {}))
        if brain.gguf_training.initialize_training_model():
            try:
                brain.gguf_training.auto_start_if_ready()
            except Exception:
                pass
    except Exception as e:
        query_logger.warning(f"Failed to initialize GGUFTrainingSystem: {e}")
        brain.gguf_training = None


def _connect_components(brain):
    """Wire up component references after initialization."""
    if 'model_manager' in brain.components:
        brain.model_manager = brain.components['model_manager']
        if brain.model_manager and brain.events:
            brain.events.trigger('model_manager_ready', brain.model_manager)
    if 'text_processor' in brain.components:
        brain.text_processor = brain.components['text_processor']
    if hasattr(brain, 'response_generator') and brain.response_generator:
        if hasattr(brain, 'model_manager') and brain.model_manager:
            brain.response_generator.model_manager = brain.model_manager
        if hasattr(brain, 'text_processor') and brain.text_processor:
            brain.response_generator.text_processor = brain.text_processor
            brain.response_generator.token_streamer = brain.text_processor
            if hasattr(brain.text_processor, 'hybrid_cache'):
                brain.response_generator.hybrid_cache = brain.text_processor.hybrid_cache
    if brain.events:
        for cn in ['memory_manager', 'text_processor', 'response_generator', 'ethics_framework']:
            if cn in brain.components:
                brain.events.trigger(f'{cn}_ready', brain.components[cn])


def _start_components(brain) -> tuple:
    """Start all startable components. Returns (started, skipped, failed)."""
    try:
        from .base_component import ComponentState
    except ImportError:
        class ComponentState:
            UNINITIALIZED = "uninitialized"; INITIALIZING = "initializing"
            READY = "ready"; STARTING = "starting"; RUNNING = "running"
            STOPPING = "stopping"; STOPPED = "stopped"; ERROR = "error"

    started = skipped = failed = 0
    for name, component in brain.components.items():
        if hasattr(component, 'start'):
            try:
                if name == 'neuromorphic_simulator' and getattr(component, 'use_nest', False):
                    skipped += 1; continue
                if hasattr(component, 'get_state'):
                    state = component.get_state()
                    if state in (ComponentState.RUNNING, ComponentState.STARTING):
                        started += 1; continue
                    if state not in (ComponentState.READY, ComponentState.UNINITIALIZED, ComponentState.STOPPED):
                        failed += 1; continue
                result = component.start()
                if result is not False:
                    started += 1
                else:
                    failed += 1
            except Exception as e:
                query_logger.warning(f"Ошибка при запуске компонента {name}: {e}", exc_info=True)
                if brain.metrics_manager:
                    brain.metrics_manager.record_error(f"component_{name}_start_failed")
                failed += 1
        else:
            skipped += 1
    return started, skipped, failed


def _stop_components(brain):
    """Stop all stoppable components."""
    try:
        if getattr(brain, 'background', None):
            brain.background.stop()
    except Exception as e:
        query_logger.warning(f"Ошибка остановки BackgroundCoordinator: {e}")
    if brain.resource_manager:
        brain.resource_manager.stop_monitoring()
    for name, component in brain.components.items():
        try:
            if hasattr(component, 'stop'):
                component.stop()
        except Exception as e:
            query_logger.error(f"Ошибка остановки компонента {name}: {e}")
