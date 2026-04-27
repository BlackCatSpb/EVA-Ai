"""
init_core.py - Main ComponentInitializer class, initialize_components(), lifecycle management.
"""
import os
import sys
import logging
import time
from typing import Dict, Any, List, Set, Optional, Callable, Tuple

logger = logging.getLogger("eva_ai.component_initializer.core")

def _ensure_eva_path():
    """Ensure EVA path is in sys.path using dynamic path detection."""
    current_file = os.path.abspath(__file__)
    eva_core_dir = os.path.dirname(current_file)
    eva_dir = os.path.dirname(eva_core_dir)
    eva_root = os.path.dirname(eva_dir)
    eva_root = os.path.normpath(eva_root)

    if eva_root not in sys.path:
        sys.path.insert(0, eva_root)

    eva_parent = os.path.dirname(eva_root)
    if eva_parent not in sys.path:
        sys.path.insert(0, eva_parent)

    return eva_root

_ensure_eva_path()


class ComponentInitializer:
    """
    Полный инициализатор компонентов системы ЕВА.

    Управляет жизненным циклом всех компонентов системы:
    - Регистрация фабрик компонентов
    - Проверка зависимостей
    - Последовательная инициализация
    - Установка связей между компонентами
    - Мониторинг состояния
    """

    COMPONENT_LIST = [
        'event_bus',
        'resource_manager',
        'config_manager',
        'memory_manager',
        'hybrid_cache',
        'fractal_graph_v2',
        'knowledge_graph',
        'text_processor',
        'ml_unit',
        'query_processor',
        'response_generator',
        'system_monitor',
        'metrics_collector',
        'analytics_manager',
        'contradiction_manager',
        'ethics_framework',
        'qwen_api_enhancer',
        'adaptation_manager',
        'web_search_engine',
        'fractal_storage',
        'self_reasoning_engine',
        'fcp_pipeline',
        'closed_cognitive_loop',
    ]

    OPTIONAL_COMPONENTS = {
        'qwen_api_enhancer',
        'web_search_engine',
        'gui',
    }

    def __init__(self, core_brain):
        """
        Инициализирует ComponentInitializer.

        Args:
            core_brain: Экземпляр CoreBrain для связи с ядром системы
        """
        self.core_brain = core_brain
        self.initialized_components: Set[str] = set()
        self.failed_components: Set[str] = set()
        self.component_dependencies: Dict[str, List[str]] = {}
        self.component_factories: Dict[str, Callable[[], Any]] = {}
        self.component_instances: Dict[str, Any] = {}
        self.component_states: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger("eva_ai.component_initializer")
        self.component_configs: Dict[str, Any] = {}

        from eva_ai.core.init_connections import define_dependencies
        self.component_dependencies = define_dependencies()

        from eva_ai.core.init_factories import register_all_factories
        register_all_factories(self)

        self.logger.info("ComponentInitializer инициализирован")

    def initialize_components(self, component_list: Optional[List[str]] = None) -> bool:
        """
        Инициализирует все компоненты в правильном порядке зависимостей.

        Args:
            component_list: Опциональный список компонентов для инициализации
                           (по умолчанию все 21 компонент)

        Returns:
            bool: True если все обязательные компоненты инициализированы успешно
        """
        try:
            _ensure_eva_path()

            self.logger.info("=" * 60)
            self.logger.info("НАЧАЛО ИНИЦИАЛИЗАЦИИ КОМПОНЕНТОВ")
            self.logger.info("=" * 60)
            
            self.logger.info(f"Available factories: {list(self.component_factories.keys())}")

            components_to_init = component_list or self.COMPONENT_LIST

            success_count = 0
            failed_count = 0
            skipped_count = 0

            for component_name in components_to_init:
                if component_name not in self.component_factories:
                    self.logger.warning(f"[WARN] Фабрика для {component_name} не найдена - пропущено")
                    skipped_count += 1
                    continue
                
                # Log fcp_pipeline initialization specially
                if component_name == 'fcp_pipeline':
                    self.logger.info("[INIT] === fcp_pipeline initialization START ===")

                from eva_ai.core.init_connections import validate_dependencies
                deps_ok, missing_deps = validate_dependencies(self, component_name)
                if not deps_ok:
                    self.logger.error(f"[FAIL] {component_name}: зависимости не готовы - {missing_deps}")
                    self.failed_components.add(component_name)
                    failed_count += 1
                    continue
                
                if component_name == 'fcp_pipeline':
                    self.logger.info("[INIT] Dependencies validated, calling factory...")

                try:
                    factory = self.component_factories[component_name]
                    component = factory()

                    if component is not None:
                        event_integration_ok = True
                        if hasattr(component, 'setup_event_subscriptions'):
                            try:
                                component.setup_event_subscriptions()
                                self.logger.debug(f"   └─ Event subscriptions set up for {component_name}")
                            except Exception as e:
                                self.logger.warning(f"   [WARN] Failed to set up event subscriptions for {component_name}: {e}")
                                event_integration_ok = False
                        elif hasattr(component, 'register_event_handlers'):
                            try:
                                component.register_event_handlers()
                                self.logger.debug(f"   └─ Event handlers registered for {component_name}")
                            except Exception as e:
                                self.logger.warning(f"   [WARN] Failed to register event handlers for {component_name}: {e}")
                                event_integration_ok = False
                        elif hasattr(component, '_setup_event_subscriptions'):
                            try:
                                component._setup_event_subscriptions()
                                self.logger.debug(f"   └─ Base event subscriptions set up for {component_name}")
                            except Exception as e:
                                self.logger.warning(f"   [WARN] Failed to set up base event subscriptions for {component_name}: {e}")
                                event_integration_ok = False
                        elif hasattr(component, 'initialize') and not getattr(component, '_event_setup_done', False):
                            try:
                                if hasattr(component, 'is_initialized'):
                                    if not component.is_initialized:
                                        component.initialize()
                            except Exception:
                                pass

                        if not event_integration_ok:
                            self.logger.warning(f"   [WARN] Неполная интеграция с системой событий для {component_name}")

                        try:
                            event_bus = getattr(self.core_brain, 'event_bus', None)
                            if event_bus is not None:
                                from eva_ai.core.event_bus import EventTypes, Event
                                event = Event(
                                    event_type=EventTypes.COMPONENT_INITIALIZED,
                                    source=component_name,
                                    data={'component_name': component_name}
                                )
                                event_bus.publish_sync(event)
                        except Exception as e:
                            self.logger.debug(f"   └─ Failed to publish COMPONENT_INITIALIZED event for {component_name}: {e}")

                            self.initialized_components.add(component_name)
                        self.component_instances[component_name] = component
                        self.core_brain.components[component_name] = component
                        self.logger.info(f"[OK] {component_name} инициализирован")
                        success_count += 1
                    else:
                        # Check if component already exists in brain.components (created elsewhere)
                        if component_name in getattr(self.core_brain, 'components', {}):
                            self.logger.info(f"[SKIP] {component_name} уже существует в brain.components (создан ранее)")
                            self.initialized_components.add(component_name)
                            self.component_instances[component_name] = self.core_brain.components[component_name]
                            success_count += 1
                        else:
                            self.logger.error(f"[FAIL] {component_name} не был создан")
                            self.failed_components.add(component_name)
                            failed_count += 1

                except Exception as e:
                    self.logger.error(f"[FAIL] {component_name}: {e}", exc_info=True)
                    self.failed_components.add(component_name)
                    failed_count += 1

            if success_count > 0:
                try:
                    self.post_initialize_connections()
                    self.logger.info("[OK] Пост-инициализация связей выполнена")
                except Exception as e:
                    self.logger.error(f"[FAIL] Ошибка пост-инициализации: {e}")

            self.logger.info("=" * 60)
            self.logger.info("ИТОГИ ИНИЦИАЛИЗАЦИИ")
            self.logger.info("=" * 60)
            self.logger.info(f"[OK] Успешно: {success_count}")
            self.logger.info(f"[FAIL] Ошибки: {failed_count}")
            self.logger.info(f"[WARN] Пропущено: {skipped_count}")
            self.logger.info(f"[STAT] Всего: {len(components_to_init)}")

            if failed_count > 0:
                self.logger.warning(f"Не инициализированы: {self.failed_components}")

            success_rate = success_count / max(1, len(components_to_init)) * 100
            self.logger.info(f"[STAT] Успешность: {success_rate:.1f}%")

            mandatory_failed = self.failed_components - self.OPTIONAL_COMPONENTS
            if mandatory_failed:
                self.logger.error(f"[FAIL] Не инициализированы обязательные компоненты: {mandatory_failed}")
            elif self.failed_components:
                self.logger.info(f"[WARN] Не инициализированы опциональные компоненты (система продолжит работу)")

            self.logger.info("=" * 60)

            return len(mandatory_failed) == 0

        except Exception as e:
            self.logger.error(f"[CRITICAL] Критическая ошибка инициализации: {e}", exc_info=True)
            return False

    def post_initialize_connections(self):
        """Устанавливает связи между компонентами после инициализации."""
        from eva_ai.core.init_connections import post_initialize_connections as _setup
        _setup(self)

    def register_component(self, name: str, component: Any,
                          dependencies: Optional[List[str]] = None) -> bool:
        """
        Регистрирует компонент в системе.

        Args:
            name: Имя компонента
            component: Экземпляр компонента
            dependencies: Список зависимостей

        Returns:
            bool: True если регистрация успешна
        """
        try:
            if dependencies is None:
                dependencies = []

            self.component_dependencies[name] = dependencies
            self.component_instances[name] = component

            if hasattr(self.core_brain, 'components'):
                self.core_brain.components[name] = component

            self.logger.info(f"[OK] Компонент {name} зарегистрирован")
            return True

        except Exception as e:
            self.logger.error(f"[FAIL] Ошибка регистрации {name}: {e}")
            return False

    def get_component(self, name: str) -> Any:
        """Получает компонент по имени."""
        return self.component_instances.get(name)

    def get_initialization_status(self) -> Dict[str, Any]:
        """Возвращает статус инициализации компонентов."""
        from eva_ai.core.init_validation import get_initialization_status
        return get_initialization_status(self)

    def retry_failed_components(self) -> Dict[str, bool]:
        """Повторяет инициализацию неудачных компонентов."""
        from eva_ai.core.init_validation import retry_failed_components
        return retry_failed_components(self)

    def shutdown_components(self):
        """Корректно завершает работу всех компонентов."""
        if getattr(self, '_shutdown_complete', False):
            self.logger.debug("Компоненты уже завершили работу")
            return

        self.logger.info("=" * 60)
        self.logger.info("ЗАВЕРШЕНИЕ РАБОТЫ КОМПОНЕНТОВ")
        self.logger.info("=" * 60)

        component_order = list(self.component_factories.keys())

        for component_name in reversed(component_order):
            if component_name in self.initialized_components:
                try:
                    component = self.component_instances.get(component_name)
                    if component:
                        if hasattr(component, 'shutdown'):
                            component.shutdown()
                            self.logger.info(f"[OK] {component_name} завершил работу")
                        elif hasattr(component, 'stop'):
                            component.stop()
                            self.logger.info(f"[OK] {component_name} остановлен")
                except Exception as e:
                    self.logger.error(f"[FAIL] Ошибка завершения {component_name}: {e}")

        self.initialized_components.clear()
        self._shutdown_complete = True
        self.logger.info("[OK] Все компоненты завершили работу")
        self.logger.info("=" * 60)

    def get_component_health(self, component_name: str) -> Dict[str, Any]:
        """Возвращает информацию о здоровье компонента."""
        from eva_ai.core.init_validation import get_component_health
        return get_component_health(self, component_name)

    def get_all_component_health(self) -> Dict[str, Dict[str, Any]]:
        """Возвращает информацию о здоровье всех компонентов."""
        from eva_ai.core.init_validation import get_all_component_health
        return get_all_component_health(self)


def create_component_initializer(core_brain) -> ComponentInitializer:
    """Factory function to create a ComponentInitializer."""
    return ComponentInitializer(core_brain)
