"""
init_validation.py - Validation, health checks, readiness checks, and error handling.
"""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("eva_ai.component_initializer.validation")


def get_component_health(initializer, component_name: str) -> Dict[str, Any]:
    """
    Returns health information for a component.

    Args:
        initializer: ComponentInitializer instance
        component_name: Component name

    Returns:
        Dict: Health information
    """
    component = initializer.component_instances.get(component_name)

    if component is None:
        return {
            'name': component_name,
            'status': 'not_initialized',
            'healthy': False
        }

    health_info = {
        'name': component_name,
        'status': 'initialized',
        'healthy': True,
        'dependencies': initializer.component_dependencies.get(component_name, [])
    }

    for dep in health_info['dependencies']:
        if dep not in initializer.initialized_components:
            health_info['healthy'] = False
            health_info['missing_dependency'] = dep
            break

    if hasattr(component, 'is_healthy'):
        health_info['healthy'] = component.is_healthy()
    elif hasattr(component, 'health_check'):
        health_info['health_status'] = component.health_check()

    return health_info


def get_all_component_health(initializer) -> Dict[str, Dict[str, Any]]:
    """
    Returns health information for all components.

    Returns:
        Dict: Health information for all components
    """
    return {
        name: get_component_health(initializer, name)
        for name in initializer.component_factories.keys()
    }


def get_initialization_status(initializer) -> Dict[str, Any]:
    """
    Returns initialization status of components.

    Returns:
        Dict: Initialization status
    """
    total = len(initializer.component_factories)
    initialized = len(initializer.initialized_components)
    failed = len(initializer.failed_components)

    return {
        'initialized': list(initializer.initialized_components),
        'failed': list(initializer.failed_components),
        'total_factories': total,
        'success_count': initialized,
        'failed_count': failed,
        'success_rate': initialized / max(1, total),
        'components': {
            name: {
                'initialized': name in initializer.initialized_components,
                'instance': initializer.component_instances.get(name),
                'dependencies': initializer.component_dependencies.get(name, [])
            }
            for name in initializer.component_factories.keys()
        }
    }


def retry_failed_components(initializer) -> Dict[str, bool]:
    """
    Retries initialization of failed components.

    Returns:
        Dict: Results of retry initialization
    """
    results = {}

    for component_name in list(initializer.failed_components):
        if component_name not in initializer.component_factories:
            continue

        try:
            initializer.logger.info(f"Повторная инициализация: {component_name}")

            from eva_ai.core.init_connections import check_dependencies
            deps_ok, _ = check_dependencies(initializer, component_name)
            if not deps_ok:
                initializer.logger.warning(f"[WARN] Зависимости не готовы для {component_name}")
                results[component_name] = False
                continue

            factory = initializer.component_factories[component_name]
            component = factory()

            if component:
                initializer.initialized_components.add(component_name)
                initializer.failed_components.discard(component_name)
                initializer.component_instances[component_name] = component
                results[component_name] = True
                initializer.logger.info(f"[OK] {component_name} инициализирован")
            else:
                results[component_name] = False

        except Exception as e:
            initializer.logger.error(f"[FAIL] Ошибка повторной инициализации {component_name}: {e}")
            results[component_name] = False

    return results
