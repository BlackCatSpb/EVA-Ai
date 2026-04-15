"""
Единое ядро системы ЕВА - тонкий координатор всех компонентов
"""
import os
import logging
import time
import threading
import queue
import collections
import weakref
from typing import Dict, Any, Optional

from .brain_config import load_brain_config, mask_secrets, ConfigMixin
from .brain_components import ComponentMixin, _init_managers, _init_fractal_model, _init_llama_cpp, _init_two_model_pipeline, _init_unified_generator, _init_preprocessing, _init_qwen_config, _init_background, _init_mode_controller, _init_hybrid_dialog_manager
from .brain_init import _init_fractal_final, _init_gen_coord, _init_wikipedia, _init_reasoning, _init_performance_monitor, _start_post_init_services, _connect_components, _start_components, _stop_components
from .brain_query import QueryMixin, FALLBACK_RESPONSES, FALLBACK_RESPONSE_DEFAULT
from .brain_monitoring import MonitoringMixin
from .brain_memory import MemoryMixin
from .brain_memory_manager import MemoryManagerMixin
from .brain_state import SystemState, SystemStateManager, StateMixin
from .brain_coordination import EventSubscriptionMixin, CommandIssuerMixin, ProcessTrackerMixin

logger = logging.getLogger("eva_ai.core_brain")
query_logger = logging.getLogger("eva_ai.core_brain.query_processing")

_SENSITIVE_PATTERNS = {'secret', 'password', 'api_key', 'token', 'credentials', 'auth', 'key', 'private'}
_global_brain_instance_ref = None

def _get_global_brain():
    global _global_brain_instance_ref
    return _global_brain_instance_ref() if _global_brain_instance_ref else None

def _set_global_brain(brain):
    global _global_brain_instance_ref
    _global_brain_instance_ref = weakref.ref(brain)

try:
    from .query_processor import QueryProcessor
except Exception:
    QueryProcessor = None

try:
    from .base_component import ComponentState
except ImportError:
    class ComponentState:
        UNINITIALIZED = "uninitialized"; INITIALIZING = "initializing"; READY = "ready"
        STARTING = "starting"; RUNNING = "running"; STOPPING = "stopping"
        STOPPED = "stopped"; ERROR = "error"


class CoreBrain(ConfigMixin, ComponentMixin, QueryMixin, MonitoringMixin, MemoryMixin, MemoryManagerMixin, StateMixin, EventSubscriptionMixin, CommandIssuerMixin, ProcessTrackerMixin):
    """Центральный координатор системы ЕВА."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        logger.debug("Инициализация ЕВАCore...")
        if config is None:
            config = self._load_brain_config()
        try:
            from .event_system import EventSystem
            self.events = EventSystem()
        except ImportError:
            self.events = None

        try:
            from .event_bus import EventBus, get_event_bus
            self._new_event_bus = get_event_bus()
        except ImportError:
            self._new_event_bus = None

        self._event_bridge = None
        try:
            if self.events is not None and self._new_event_bus is not None:
                from .event_bus_bridge import EventBusBridge
                self._event_bridge = EventBusBridge(self.events.event_bus, self._new_event_bus)
                self._event_bridge.link()
        except Exception as e:
            logger.warning(f"Не удалось связать шины событий: {e}")
        if config:
            query_logger.debug(f"Получена конфигурация с {len(config)} параметрами")
            if any(k.lower() in _SENSITIVE_PATTERNS for k in config.keys()):
                query_logger.debug(f"Конфигурация (с маскировкой): {mask_secrets(config)}")

        self.config = config or {}
        self.components: Dict[str, Any] = {}
        self.module_activity: Dict[str, Dict[str, Any]] = {}
        self.module_access_log = collections.deque(maxlen=1000)
        self.activity_lock = threading.Lock()
        self.module_control: Dict[str, Dict[str, Any]] = {}
        self.initialized = False
        self.running = False
        self._shutting_down = False
        self._shutdown_lock = threading.Lock()
        self.status_queue = queue.Queue()
        self.deferred_commands = []
        self._deferred_commands_lock = threading.Lock()
        self.log_throttle_seconds = int(self.config.get("system", {}).get("log_throttle_seconds", 30))
        self.query_timeout = float(self.config.get("system", {}).get("query_timeout", 30))
        self._log_throttle: Dict[str, float] = {}
        self._log_throttle_lock = threading.Lock()
        self._model_load_lock = threading.Lock()

        try:
            from .deferred_command_system import DeferredCommandSystem, CommandPriority, set_event_bus
            self.deferred_system = DeferredCommandSystem(self, max_workers=6)
            # Устанавливаем EventBus для публикации событий команд
            if hasattr(self, '_new_event_bus') and self._new_event_bus:
                set_event_bus(self._new_event_bus)
            self._register_deferred_system_handlers()
        except ImportError:
            self.deferred_system = None

        self.cache_dir = os.path.join(os.path.dirname(__file__), "eva_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        try:
            mode = str(self.config.get('mode') or os.environ.get('COGNIFLEX_MODE') or '').lower()
            if mode == 'context_first':
                from .context_first_policy import ContextFirstPolicy
                ContextFirstPolicy(self).apply()
        except Exception:
            pass

        _init_managers(self)
        # FractalModelManager and ML unit placeholders - using UnifiedGenerator only
        self.fractal_model_manager = None
        self.ml_unit = None
        # _init_fractal_model(self)
        # _init_llama_cpp(self)
        _init_unified_generator(self)
        _init_hybrid_dialog_manager(self)  # HybridKnowledgeDialogManager
        _init_preprocessing(self)
        # QwenModelManager disabled - using UnifiedGenerator only
        # _init_qwen_config(self)
        _init_background(self)
        _init_mode_controller(self)
        self._init_memory_manager()
        _set_global_brain(self)
        
        # Initialize ProcessTrackerMixin
        ProcessTrackerMixin.__init__(self)
        
        query_logger.debug("ЕВАCore инициализирован")

    @property
    def event_bus(self):
        """Возвращает унифицированную шину событий."""
        if self._new_event_bus is not None:
            return self._new_event_bus
        if self.events is not None:
            return self.events.event_bus
        return None
    
    @event_bus.setter
    def event_bus(self, value):
        """Устанавливает шину событий."""
        self._new_event_bus = value

    def initialize(self) -> bool:
        self._initialize_detailed_logging()
        start_time = time.time()
        query_logger.info("НАЧАЛО ИНИЦИАЛИЗАЦИИ ЯДРА COGNIFLEX")
        try:
            self._subscribe_to_system_events()
            self._update_state(SystemState.INITIALIZING, "Инициализация компонентов")
            if self.resource_manager:
                self.resource_manager.start_monitoring()
                # Активируем load shedding мост
                if self.deferred_system and self.event_bus:
                    self.deferred_system.create_bridge(self.event_bus, self.resource_manager)
            if self.metrics_manager:
                self.metrics_manager.start_tracking()
            try:
                from eva_ai.core.fractal_attention_system import FractalAttentionSystem
                self.attention_system = FractalAttentionSystem(self)
            except Exception:
                pass
            if self.component_initializer:
                init_result = self.component_initializer.initialize_components()
                if not init_result:
                    query_logger.warning("Failed: %s" % self.component_initializer.failed_components)
            if not self._initialize_memory_manager():
                query_logger.warning("Не удалось инициализировать MemoryManager")
            _connect_components(self)
            _init_fractal_final(self)
            _init_gen_coord(self)
            _init_wikipedia(self)
            self.initialized = True
            _init_reasoning(self)
            _init_performance_monitor(self)
            self._update_state(SystemState.READY, "Инициализация завершена успешно")
            total_time = time.time() - start_time
            if self.metrics_manager:
                self.metrics_manager.record_system_startup(total_time)
            query_logger.info(f"Ядро ЕВА успешно инициализировано за {total_time:.4f} сек")
            self._execute_deferred_commands()
            _start_post_init_services(self)
            return True
        except Exception as e:
            query_logger.error(f"Ошибка инициализации ядра за {time.time() - start_time:.4f} сек: {e}", exc_info=True)
            if self.metrics_manager:
                self.metrics_manager.record_error("core_initialization_failed")
            return False

    def start(self) -> bool:
        if not self.initialized:
            query_logger.error("Невозможно запустить неинициализированное ядро"); return False
        start_time = time.time()
        query_logger.info("Запуск ядра ЕВА...")
        try:
            started, skipped, failed = _start_components(self)
            total = len(self.components)
            query_logger.info(f"Итоги запуска: {started}/{total} запущено, {skipped} пропущено, {failed} неудачно")
            active = started + failed
            if active > 0 and failed > active * 0.5:
                query_logger.warning(f"ВНИМАНИЕ: Запущено только {started}/{active} активных компонентов")
                if self.metrics_manager:
                    self.metrics_manager.record_warning("insufficient_components_started")
            self.running = True
            try:
                if 'gui' in self.components and hasattr(self.components['gui'], 'start_gui'):
                    self.components['gui'].start_gui()
            except Exception as e:
                query_logger.warning(f"Ошибка при запуске GUI: {e}")
            try:
                if getattr(self, 'background', None):
                    self.background.start()
            except Exception as e:
                query_logger.warning(f"Не удалось запустить BackgroundCoordinator: {e}")
            query_logger.info(f"Ядро ЕВА успешно запущено за {time.time() - start_time:.4f} сек")
            return True
        except Exception as e:
            query_logger.error(f"Ошибка запуска ядра: {e}", exc_info=True)
            if self.metrics_manager:
                self.metrics_manager.record_error("core_start_failed")
            if self.state_manager:
                self.state_manager.set_state(SystemState.ERROR, f"Ошибка запуска: {e}")
            return False

    def stop(self):
        with self._shutdown_lock:
            if self._shutting_down or not self.running:
                return
            self._shutting_down = True
        stop_time = time.time()
        query_logger.info("Остановка ядра ЕВА...")
        try:
            if self.state_manager:
                self.state_manager.set_state(SystemState.SHUTTING_DOWN, "Начало остановки системы")
            if self.deferred_system:
                try:
                    self.deferred_system.shutdown()
                except Exception:
                    pass
            _stop_components(self)
            self.running = False
            if self.state_manager:
                self.state_manager.set_state(SystemState.OFFLINE, "Система остановлена")
            total_time = time.time() - stop_time
            if self.metrics_manager:
                self.metrics_manager.record_system_shutdown(total_time)
            query_logger.info(f"Ядро ЕВА остановлено за {total_time:.4f} сек")
        except Exception as e:
            query_logger.error(f"Ошибка остановки ядра: {e}", exc_info=True)
            if self.metrics_manager:
                self.metrics_manager.record_error("core_stop_failed")
            if self.state_manager:
                self.state_manager.set_state(SystemState.ERROR, str(e))

    def start_background_services(self) -> None:
        try:
            if getattr(self, 'background', None):
                self.background.start()
        except Exception as e:
            query_logger.warning(f"start_background_services: {e}")

    def stop_background_services(self) -> None:
        try:
            if getattr(self, 'background', None):
                self.background.stop()
        except Exception as e:
            query_logger.warning(f"stop_background_services: {e}")

    def signal_user_activity(self) -> None:
        try:
            if getattr(self, 'background', None):
                self.background.signal_user_activity()
        except Exception:
            pass

    def reboot(self) -> bool:
        self._log_throttled(query_logger, logging.INFO, "core_reboot_request", "Запрос перезагрузки ядра")
        try:
            if getattr(self, 'running', False):
                try:
                    self.stop()
                except Exception as e:
                    query_logger.error(f"Ошибка при остановке перед перезагрузкой: {e}", exc_info=True)
            self.initialized = False
            if not self.initialize() or not self.start():
                return False
            self._log_throttled(query_logger, logging.INFO, "core_reboot_success", "Перезагрузка ядра успешно завершена")
            return True
        except Exception as e:
            query_logger.error(f"Ошибка перезагрузки ядра: {e}", exc_info=True)
            return False

    def debug_message(self, message: str) -> str:
        msg = message.strip().lower()
        if msg == "status":
            return f"Status: fractal_ready={self.fractal_ready}"
        elif msg == "health":
            return f"Health: fractal_ready={self.fractal_ready}, learning={hasattr(self, 'self_dialog_learning') and self.self_dialog_learning is not None}"
        elif msg == "test":
            return self.fractal_model_manager.generate_response("Привет", max_new_tokens=30) if self.fractal_model_manager else "Model not available"
        elif msg == "memory":
            if hasattr(self, 'memory_manager') and self.memory_manager:
                return f"Memory: initialized={getattr(self.memory_manager, 'initialized', False)}"
            return "Memory manager not available"
        return f"[DEBUG] Received: '{message}'. System ready. Commands: status, health, test, memory"

    def trigger_subjective_correctness(self, message_text: str, rating: int) -> bool:
        try:
            query_logger.info(f"Субъективная корректность: rating={rating}")
            if hasattr(self, 'self_dialog_learning') and self.self_dialog_learning and hasattr(self.self_dialog_learning, 'analyze_interaction'):
                self.self_dialog_learning.analyze_interaction(query="", response=message_text, feedback={'rating': rating, 'text': message_text, 'type': 'subjective_feedback'})
            if rating == -1 and hasattr(self, 'knowledge_graph') and self.knowledge_graph:
                try:
                    self.knowledge_graph.add_node(name=f"user_feedback_dislike", content=f"Пользователь отметил как неверное: {message_text[:100]}", domain="feedback")
                except Exception:
                    pass
            return True
        except Exception as e:
            query_logger.error(f"Error triggering subjective correctness: {e}"); return False


# Backward compatibility alias
ЕВАCore = CoreBrain


def setup_logging():
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main():
    setup_logging()
    core = CoreBrain()
    if core.initialize() and core.start():
        def _cli_exit_listener():
            try:
                while core.running:
                    try:
                        user_input = input()
                    except (EOFError, OSError, KeyboardInterrupt):
                        break
                    if isinstance(user_input, str) and user_input.strip().lower() in ("exit", "quit", "q"):
                        try:
                            core.stop()
                        except Exception:
                            pass
                        break
            except Exception:
                pass
        threading.Thread(target=_cli_exit_listener, name="CLIExitListener", daemon=True).start()
        last_health_check = time.time()
        try:
            while core.running:
                time.sleep(1)
                if time.time() - last_health_check >= 30:
                    last_health_check = time.time()
                    health = core.get_system_health()
                    logger.info(f"Текущее состояние системы: {health['status']}")
        except KeyboardInterrupt:
            pass
        finally:
            core.stop()


if __name__ == "__main__":
    main()
