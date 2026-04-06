"""
ComponentInitializer - Единый инициализатор компонентов системы ЕВА
Версия: 2.0.0
Поддерживаемые компоненты: 21

Refactored: split into 4 focused modules:
  - init_core.py: Main ComponentInitializer class, lifecycle management
  - init_factories.py: Factory functions for creating components
  - init_connections.py: Dependency injection, post-initialization connections
  - init_validation.py: Validation, health checks, error handling
"""
from eva.core.init_core import ComponentInitializer, create_component_initializer
from eva.core.init_factories import (
    create_event_bus, create_resource_manager, create_config_manager,
    create_memory_manager, create_hybrid_cache,
    create_knowledge_graph, create_qwen_api_enhancer, create_text_processor,
    create_ml_unit, create_model_manager,
    create_query_processor, create_response_generator, create_reasoning_engine,
    create_analytics_manager, create_system_monitor, create_metrics_collector,
    create_contradiction_manager, create_adaptation_manager,
    create_ethics_framework, create_web_search_engine,
    create_fractal_storage, create_self_reasoning_engine,
    create_enhanced_reasoning_engine,
    register_all_factories,
)
from eva.core.init_connections import (
    define_dependencies, validate_dependencies, check_dependencies,
    post_initialize_connections,
)
from eva.core.init_validation import (
    get_component_health, get_all_component_health,
    get_initialization_status, retry_failed_components,
)

__all__ = [
    'ComponentInitializer',
    'create_component_initializer',
    'create_event_bus', 'create_resource_manager', 'create_config_manager',
    'create_memory_manager', 'create_hybrid_cache',
    'create_knowledge_graph', 'create_qwen_api_enhancer', 'create_text_processor',
    'create_ml_unit', 'create_model_manager',
    'create_query_processor', 'create_response_generator', 'create_reasoning_engine',
    'create_analytics_manager', 'create_system_monitor', 'create_metrics_collector',
    'create_contradiction_manager', 'create_adaptation_manager',
    'create_ethics_framework', 'create_web_search_engine',
    'create_fractal_storage', 'create_self_reasoning_engine',
    'create_enhanced_reasoning_engine',
    'register_all_factories',
    'define_dependencies', 'validate_dependencies', 'check_dependencies',
    'post_initialize_connections',
    'get_component_health', 'get_all_component_health',
    'get_initialization_status', 'retry_failed_components',
]

if __name__ == "__main__":
    """Тестирование ComponentInitializer."""

    print("=" * 60)
    print("ТЕСТИРОВАНИЕ ComponentInitializer")
    print("=" * 60)

    class MockCoreBrain:
        def __init__(self):
            self.components = {}
            self.cache_dir = './test_cache'

    mock_brain = MockCoreBrain()
    initializer = ComponentInitializer(mock_brain)

    print(f"\n[INFO] Зарегистрировано фабрик: {len(initializer.component_factories)}")
    print(f"[INFO] Компонентов в списке: {len(initializer.COMPONENT_LIST)}")
    print(f"[INFO] Зависимостей определено: {len(initializer.component_dependencies)}")

    print("\n[OK] ComponentInitializer готов к работе")
    print("=" * 60)
