"""
init_connections.py - Dependency injection, post-initialization connections, and component linking.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("eva_ai.component_initializer.connections")


def define_dependencies() -> Dict[str, List[str]]:
    """Defines dependencies between components."""
    return {
        'event_bus': [],
        'resource_manager': [],
        'config_manager': [],
        'memory_manager': [],
        'hybrid_cache': ['memory_manager'],
        'knowledge_graph': ['event_bus'],
        'fractal_graph_v2': ['event_bus'],  # Новый граф памяти
        'text_processor': ['hybrid_cache'],
        'ml_unit': ['memory_manager', 'knowledge_graph'],
        'model_manager': ['ml_unit'],
        'query_processor': ['text_processor', 'knowledge_graph'],
        'response_generator': ['query_processor'],
        'reasoning_engine': ['knowledge_graph'],
        'analytics_manager': ['system_monitor'],
        'system_monitor': ['resource_manager'],
        'metrics_collector': ['system_monitor'],
        'contradiction_manager': ['knowledge_graph'],
        'adaptation_manager': ['analytics_manager'],
        'ethics_framework': ['knowledge_graph'],
        'web_search_engine': ['knowledge_graph'],
        'qwen_api_enhancer': ['knowledge_graph'],
        'gui': [],
        'fractal_storage': [],
        'self_reasoning_engine': ['fractal_storage', 'knowledge_graph'],
        'enhanced_reasoning_engine': ['fractal_storage', 'knowledge_graph', 'contradiction_manager', 'ethics_framework', 'web_search_engine'],
    }


def validate_dependencies(initializer, component_name: str) -> Tuple[bool, List[str]]:
    """
    Validates component dependencies before initialization.

    Args:
        initializer: ComponentInitializer instance
        component_name: Component name to check

    Returns:
        Tuple[bool, List[str]]: (valid, list of issues)
    """
    issues = []

    if component_name not in initializer.component_dependencies:
        issues.append(f"Component {component_name} not found in dependencies")
        return False, issues

    dependencies = initializer.component_dependencies[component_name]

    for dep in dependencies:
        if dep not in initializer.component_factories:
            issues.append(f"Dependency {dep} not registered as factory")
            continue

        # Check if dependency already exists in any form
        dep_exists = (
            dep in initializer.initialized_components or  # Successfully initialized
            dep in initializer.component_instances or  # Has instance
            dep in getattr(initializer.core_brain, 'components', {}) or  # In brain.components
            getattr(initializer.core_brain, dep, None) is not None  # As brain attribute
        )
        
        # If failed but exists elsewhere (brain or component_instances), it's OK
        if dep in initializer.failed_components and dep_exists:
            initializer.logger.debug(f"Dependency {dep} previously failed but exists in other location - allowing")
            initializer.failed_components.discard(dep)  # Clear the failure
            continue
        
        if dep in initializer.failed_components and not dep_exists:
            issues.append(f"Dependency {dep} previously failed to initialize")
            continue

        if not dep_exists:
            # Check if maybe it will be initialized later (try once more)
            if dep not in initializer.initialized_components:
                issues.append(f"Dependency {dep} not yet initialized")
                continue

    is_valid = len(issues) == 0

    if not is_valid:
        initializer.logger.debug(f"Dependency validation for {component_name}: {issues}")

    return is_valid, issues


def check_dependencies(initializer, component_name: str) -> Tuple[bool, List[str]]:
    """
    Checks if all dependencies of a component are initialized.

    Args:
        initializer: ComponentInitializer instance
        component_name: Component name to check

    Returns:
        Tuple[bool, List[str]]: (success, list of missing dependencies)
    """
    dependencies = initializer.component_dependencies.get(component_name, [])
    missing = []

    for dep in dependencies:
        if dep not in initializer.initialized_components:
            missing.append(dep)

    if missing:
        return False, missing
    return True, []


def post_initialize_connections(initializer):
    """Establishes connections between components after initialization."""
    try:
        initializer.logger.info("Установка связей между компонентами...")

        initializer.logger.info(f"Проверка гибридного кэша в brain.components: {hasattr(initializer.core_brain, 'components')}")
        if hasattr(initializer.core_brain, 'components'):
            initializer.logger.info(f"Ключи в brain.components: {list(initializer.core_brain.components.keys())}")

        hybrid_cache = initializer.core_brain.components.get('hybrid_cache')
        initializer.logger.info(f"hybrid_cache из brain.components: {hybrid_cache}")

        if hybrid_cache is not None:
            initializer.core_brain.hybrid_cache = hybrid_cache
            initializer.logger.info("Установка гибридного кэша в brain.hybrid_cache")

            components_to_connect = [
                ('memory_manager', 'memory_manager'),
                ('ml_unit', 'ml_unit'),
                ('text_processor', 'text_processor'),
                ('model_manager', 'model_manager'),
                ('query_processor', 'query_processor'),
                ('response_generator', 'response_generator')
            ]

            for comp_name, comp_key in components_to_connect:
                component = initializer.component_instances.get(comp_key)
                if component is not None:
                    if hasattr(component, 'hybrid_cache'):
                        old_cache = component.hybrid_cache
                        component.hybrid_cache = hybrid_cache

                        if old_cache is hybrid_cache:
                            initializer.logger.info(f"   [OK] {comp_name}: уже был подключен")
                        elif old_cache is not None:
                            initializer.logger.info(f"   [UPD] {comp_name}: заменен старый кэш")
                        else:
                            initializer.logger.info(f"   └─ hybrid_cache → {comp_name}")
                    else:
                        try:
                            setattr(component, 'hybrid_cache', hybrid_cache)
                            initializer.logger.info(f"   └─ hybrid_cache → {comp_name} (через setattr)")
                        except Exception as e:
                            initializer.logger.debug(f"   [WARN] {comp_name}: не удалось установить hybrid_cache: {e}")
                else:
                    initializer.logger.debug(f"   [WARN] {comp_name}: компонент не найден")

        ml_unit = initializer.component_instances.get('ml_unit')
        if ml_unit is not None:
            model_manager = initializer.component_instances.get('model_manager')
            if model_manager is not None and hasattr(ml_unit, 'model_manager'):
                ml_unit.model_manager = model_manager
                initializer.logger.info("   └─ ml_unit → model_manager")

            text_processor = initializer.component_instances.get('text_processor')
            if text_processor is not None and hasattr(ml_unit, 'text_processor'):
                ml_unit.text_processor = text_processor
                initializer.logger.info("   └─ ml_unit → text_processor")

        response_generator = initializer.component_instances.get('response_generator')
        if response_generator is not None:
            model_manager = initializer.component_instances.get('model_manager')
            if model_manager is not None and hasattr(response_generator, 'model_manager'):
                response_generator.model_manager = model_manager
                initializer.logger.info("   └─ response_generator → model_manager")

            reasoning_engine = initializer.component_instances.get('reasoning_engine')
            if reasoning_engine is not None and hasattr(response_generator, 'reasoning_engine'):
                response_generator.reasoning_engine = reasoning_engine
                initializer.logger.info("   └─ response_generator → reasoning_engine")

        initializer.logger.info("[OK] Все связи установлены")

    except Exception as e:
        initializer.logger.error(f"[FAIL] Ошибка установки связей: {e}", exc_info=True)
        raise
