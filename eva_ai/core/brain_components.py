"""
Component initialization, deferred commands, module management, model management, and CoreBrain init helpers.
"""
import os
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("eva_ai.core_brain")
query_logger = logging.getLogger("eva_ai.core_brain.query_processing")


def _subscribe_components_to_eventbus(brain):
    """Подписать все компоненты brain на EventBus для управления через события."""
    event_bus = getattr(brain, '_new_event_bus', None)
    if event_bus is None:
        logger.info("[brain_components] EventBus not available for component subscriptions")
        return
    
    from eva_ai.core.event_bus import EventTypes, Event
    
    def on_system_ready(event: Event):
        logger.info(f"[brain_components] System ready: {event.data}")
        if brain.state_manager and hasattr(brain.state_manager, 'set_state'):
            try:
                from eva_ai.core.system_state import SystemState
                brain.state_manager.set_state(SystemState.RUNNING, "Система готова")
            except:
                pass
    
    def on_component_initialized(event: Event):
        logger.debug(f"[brain_components] Component initialized: {event.data}")
    
    def on_system_error(event: Event):
        logger.error(f"[brain_components] System error: {event.data}")
        if brain.state_manager and hasattr(brain.state_manager, 'set_state'):
            try:
                from eva_ai.core.system_state import SystemState
                brain.state_manager.set_state(SystemState.ERROR, str(event.data))
            except:
                pass
    
    event_bus.subscribe(EventTypes.SYSTEM_READY, on_system_ready, priority=1)
    event_bus.subscribe(EventTypes.COMPONENT_INITIALIZED, on_component_initialized, priority=5)
    event_bus.subscribe(EventTypes.SYSTEM_ERROR, on_system_error, priority=1)
    
    logger.info("[brain_components] Components subscribed to EventBus")


def _init_mode_controller(brain):
    """Инициализация контроллера режимов модели (язык, квантование)."""
    import sys
    import os
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    
    try:
        from eva_ai.mlearning.language_filter import ModelModeController
        brain.mode_controller = ModelModeController(brain=brain)
        
        default_language = brain.config.get('model', {}).get('language_mode', 'russian_only')
        brain.mode_controller.set_language_mode(default_language)
        
        brain.components['mode_controller'] = brain.mode_controller
        logger.info(f"ModelModeController инициализирован (язык: {default_language})")
    except Exception as e:
        logger.warning(f"Не удалось инициализировать ModelModeController: {e}")
        brain.mode_controller = None


def _init_managers(brain):
    """Initialize all manager components on the brain instance."""
    try:
        from .config_manager import ConfigManager
        brain.config_manager = ConfigManager()
    except ImportError:
        brain.config_manager = None
        query_logger.warning("Менеджер конфигурации недоступен")
    
    try:
        from .system_state import SystemStateManager, SystemState
        brain.state_manager = SystemStateManager()
        if brain.state_manager and hasattr(brain.state_manager, 'set_state'):
            brain.state_manager.set_state(SystemState.INITIALIZING, "Начало инициализации CoreBrain")
    except ImportError:
        brain.state_manager = None
        query_logger.warning("Менеджер состояния системы недоступен")

    try:
        from .resource_manager import ResourceManager
        brain.resource_manager = ResourceManager(brain.config_manager) if brain.config_manager else ResourceManager(None)
    except ImportError:
        brain.resource_manager = None
        query_logger.warning("Менеджер ресурсов недоступен")

    try:
        from eva_ai.learning.self_analyzer import SelfAnalyzer
        brain.self_analyzer = SelfAnalyzer(brain=brain, cache_dir=brain.cache_dir)
    except ImportError as e:
        brain.self_analyzer = None
        query_logger.warning(f"Модуль самоанализа недоступен: {e}")

    try:
        from .system_metrics import SystemMetricsManager
        brain.metrics_manager = SystemMetricsManager()
    except ImportError:
        class SystemMetricsManager:
            def __init__(self): self.metrics = {"error_rate": 0.0}
            def start_tracking(self): pass
            def get_metrics(self): return self.metrics
            def record_error(self, error_type): pass
            def record_warning(self, warning_type): pass
            def record_system_startup(self, time): pass
            def record_system_shutdown(self, time): pass
            def record_query_metrics(self, **kwargs): pass
            def emit(self, metric): pass
            def emit_many(self, metrics): return 0
            def flush(self): return []
        brain.metrics_manager = SystemMetricsManager()
        query_logger.warning("Менеджер системных метрик недоступен, используется заглушка")

    # ОТКЛЮЧЕНО - используем fractal_graph_v2 вместо MemoryGraphML
    # try:
    #     from .memory_graph_ml import MemoryGraphML
    #     brain.memory_graph_ml = MemoryGraphML(brain, config=brain.config.get('memory_graph_ml', {}))
    #     if not brain.memory_graph_ml.initialize():
    #         query_logger.warning("Не удалось инициализировать MemoryGraphML")
    # except ImportError as e:
    #     query_logger.warning(f"MemoryGraphML недоступен: {e}")
    #     brain.memory_graph_ml = None
    brain.memory_graph_ml = None  # Используем fractal_graph_v2

    try:
        from .feedback_processor import FeedbackProcessor
        graph_learning = None
        if hasattr(brain, 'fractal_memory') and brain.fractal_memory:
            graph_learning = brain.fractal_memory
        elif hasattr(brain, 'memory') and brain.memory:
            graph_learning = brain.memory
        
        brain.feedback_processor = FeedbackProcessor(
            graph_learning=graph_learning,
            event_bus=getattr(brain, 'event_bus', None)
        )
        query_logger.info("FeedbackProcessor initialized")
    except ImportError as e:
        query_logger.warning(f"FeedbackProcessor недоступен: {e}")
        brain.feedback_processor = None

    try:
        from eva_ai.learning.self_dialog_learning import SelfDialogLearningSystem
        if SelfDialogLearningSystem:
            brain.self_dialog_learning = SelfDialogLearningSystem(brain=brain, config=brain.config.get('self_dialog_learning', {}))
            query_logger.info("SelfDialogLearningSystem initialized")
        else:
            brain.self_dialog_learning = None
    except Exception as e:
        brain.self_dialog_learning = None
        query_logger.warning(f"SelfDialogLearningSystem initialization failed: {e}")

    try:
        from eva_ai.learning.performance_analyzer import PerformanceAnalyzer
        if PerformanceAnalyzer:
            brain.performance_analyzer = PerformanceAnalyzer(brain=brain)
        else:
            brain.performance_analyzer = None
    except Exception:
        brain.performance_analyzer = None

    try:
        from eva_ai.knowledge.online_knowledge import OnlineKnowledgeAccess
        if OnlineKnowledgeAccess:
            brain.online_knowledge = OnlineKnowledgeAccess(brain=brain, config=brain.config.get('online_knowledge', {}))
        else:
            brain.online_knowledge = None
    except Exception:
        brain.online_knowledge = None

    try:
        from .self_learning_system import initialize_self_learning
        if not initialize_self_learning(brain):
            query_logger.warning("Не удалось инициализировать систему самообучения (legacy)")
    except ImportError:
        query_logger.warning("Система самообучения (legacy) недоступна")

    try:
        from .query_processor import QueryProcessor
        brain.query_processor = QueryProcessor(brain) if QueryProcessor else None
    except Exception:
        brain.query_processor = None
    if brain.query_processor:
        brain.components['query_processor'] = brain.query_processor

    try:
        from .component_initializer import ComponentInitializer
        brain.component_initializer = ComponentInitializer(brain)
        query_logger.info(f"[INIT] ComponentInitializer created: {brain.component_initializer}")
    except Exception as e:
        brain.component_initializer = None
        query_logger.warning(f"Ошибка инициализации компонентного инициализатора: {e}", exc_info=True)

    try:
        get_shared_cache = None
        try:
            from eva_ai.memory.hybrid_token_cache import get_shared_cache
        except ImportError:
            try:
                from eva_ai.memory import get_shared_cache
            except ImportError:
                pass
        
        if get_shared_cache:
            brain.token_cache = get_shared_cache(brain, "default")
        else:
            brain.token_cache = None
    except Exception as e:
        query_logger.warning(f"Ошибка инициализации гибридного кэша: {e}")
        brain.token_cache = None

    brain.fractal_ready = False
    brain.qwen_ready = False
    brain.models_ready = False


def _init_fractal_model(brain):
    try:
        from eva_ai.mlearning.fractal_model_manager import FractalModelManager
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(project_root, "eva", "mlearning", "eva_models", "qwen3.5-0.8b")
        brain.fractal_model_manager = FractalModelManager(model_path=model_path)
        query_logger.debug(f"FractalModelManager инициализирован с путем: {model_path}")
    except (ImportError, Exception) as e:
        query_logger.debug(f"Ошибка инициализации FractalModelManager: {e}")
        brain.fractal_model_manager = None


def _init_llama_cpp(brain):
    brain.llama_cpp_deployment = None
    brain.llama_cpp_ready = False
    try:
        model_config = brain.config.get('model', {})
        if model_config.get('use_llama_cpp', False):
            from eva_ai.mlearning.hot_deployment.llama_cpp_hot import LlamaCppHotDeployment
            brain.llama_cpp_deployment = LlamaCppHotDeployment(
                model_path=model_config.get('gguf_model_path', ''),
                n_ctx=model_config.get('llama_cpp_n_ctx', 2048),
                n_threads=model_config.get('llama_cpp_threads', os.cpu_count() or 12)
            )
            if brain.llama_cpp_deployment.initialize(preload_root=True):
                brain.llama_cpp_ready = True
                query_logger.info("LlamaCpp (GGUF) готов к работе!")
    except Exception as e:
        query_logger.debug(f"LlamaCpp не инициализирован: {e}")
        brain.llama_cpp_deployment = None


def _init_fcp_pipeline(brain):
    """Инициализация FCPipeline - основной и единственный пайплайн генерации.
    
    FCPipeline включает:
    - OpenVINO GenAI с ruadapt_qwen3_4b
    - GNN инъекция на всех 32 слоях
    - LoRA management (ShadowLoRAManagerOV)
    - ToolOrchestrator, ThinkingController
    - LearningGraphManager, GraphCurator
    - ScenarioTCM, ExpertSystem
    """
    brain.fcp_pipeline = None
    brain.fcp_pipeline_ready = False
    
    try:
        # fcp_pipeline config is nested under 'fcp' key
        fcp_config = brain.config.get('fcp', {}).get('fcp_pipeline', {})

        # Fallback to top-level fcp_pipeline for backwards compatibility
        if not fcp_config:
            fcp_config = brain.config.get('fcp_pipeline', {})

        # Check if enabled at fcp level
        fcp_enabled = brain.config.get('fcp', {}).get('enabled', True)
        if not fcp_enabled:
            query_logger.info("FCPipeline отключён в конфигурации (fcp.enabled=false)")
            return

        if not fcp_config.get('enabled', True):
            query_logger.info("FCPipeline отключён в конфигурации (fcp_pipeline.enabled=false)")
            return

        from eva_ai.core.fcp_pipeline import FCPPipelineV15

        model_path = fcp_config.get('model_path')
        if not model_path:
            # Пробуем получить из fcp_pipeline section
            fcp_pipeline_config = fcp_config.get('fcp_pipeline', {})
            model_path = fcp_pipeline_config.get('model_path')
        
        if not model_path:
            # Fallback на brain_config.json корневой model секции
            model_path = brain.config.get('model', {}).get('model_path')
        
        if not model_path:
            # Проверяем стандартный путь
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            default_path = os.path.join(project_root, 'models', 'ruadapt_qwen3_4b_openvino_ModelB')
            if os.path.exists(default_path):
                model_path = default_path
            else:
                query_logger.error("FCPipeline: model_path не указан и стандартный путь не существует")
                brain.fcp_pipeline_ready = False
                return
        
        graph_path = fcp_config.get('graph_path')
        gnn_ov_path = fcp_config.get('gnn_ov_path')
        lora_dir = fcp_config.get('lora_dir')
        
        # Проверяем, не инициализирован ли уже FCPipeline
        if brain.fcp_pipeline is not None:
            logger.info(f"FCPipeline уже инициализирован, используем существующий экземпляр")
            return
        
        pipeline = FCPPipelineV15(
            model_path=model_path,
            graph_path=graph_path,
            gnn_ov_path=gnn_ov_path,
            lora_dir=lora_dir,
            brain=brain
        )
        
        brain.fcp_pipeline = pipeline
        brain.fcp_pipeline_ready = True
        
        query_logger.info(f"FCPipeline инициализирован: model={model_path}")
        
        try:
            from eva_ai.core.event_bus import Event, EventTypes
            if brain.events:
                brain.events.trigger(EventTypes.COMPONENT_INITIALIZED, data={
                    "component": "FCPipeline",
                    "model_path": model_path,
                    "ready": True
                })
        except Exception:
            pass
        
    except Exception as e:
        query_logger.error(f"Ошибка инициализации FCPipeline: {e}")
        brain.fcp_pipeline = None
        brain.fcp_pipeline_ready = False


def _init_two_model_pipeline(brain):
    """DEPRECATED - используюется FCPipeline вместо Two-Model Pipeline.
    
    Эта функция сохранена для обратной совместимости но НЕ инициализирует старый пайплайн.
    """
    brain.two_model_pipeline = None
    brain.two_model_pipeline_ready = False
    query_logger.info("Two-Model Pipeline DEPRECATED - используется FCPipeline")
    
    try:
        model_config = brain.config.get('model', {})
        
        # Получаем режим работы
        # 'fractal' - только FractalPipeline (новый)
        # 'recursive' - только RecursiveModelPipeline (старый)
        # 'hybrid' - FractalPipeline + fallback на RecursiveModelPipeline
        pipeline_mode = model_config.get('pipeline_mode', 'fractal')
        
        # Проверяем нужен ли pipeline
        if not model_config.get('use_two_model_pipeline', False):
            query_logger.info("Two-Model Pipeline отключён в конфигурации")
            return

        # Инициализируем fractal_memory
        try:
            from eva_ai.memory.unified_fractal_memory import UnifiedFractalMemory
            fractal_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "memory", "fractal_torch_storage", "unified_memory"
            )
            brain.fractal_memory = UnifiedFractalMemory(
                storage_dir=fractal_dir,
                config=brain.config.get('fractal_memory', {})
            )
        except Exception as e:
            query_logger.warning(f"UnifiedFractalMemory не инициализирован: {e}")
            brain.fractal_memory = None

        # Получаем пути к моделям
        model_a_path = model_config.get('model_a_gguf_path', '')
        model_b_path = model_config.get('model_b_gguf_path', '')
        n_ctx = model_config.get('llama_cpp_n_ctx', 8192)
        n_threads = model_config.get('llama_cpp_threads', os.cpu_count() or 12)
        
        # Проверяем пути
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if not os.path.isabs(model_a_path):
            model_a_path = os.path.join(project_root, model_a_path)
        if not os.path.isabs(model_b_path):
            model_b_path = os.path.join(project_root, model_b_path)
        
        if not os.path.exists(model_a_path):
            query_logger.error(f"Model A file not found: {model_a_path}")
            return
        if not os.path.exists(model_b_path):
            model_b_path = model_a_path
        
        # Создаём HybridPipelineAdapter
        try:
            from eva_ai.core.hybrid_pipeline_adapter import HybridPipelineAdapter
            
            adapter_kwargs = {
                'model_a_path': model_a_path,
                'model_b_path': model_b_path,
                'n_ctx': n_ctx,
                'n_threads': n_threads,
                'mode': pipeline_mode
            }
            
            if brain.fractal_memory:
                adapter_kwargs['fractal_graph'] = brain.fractal_memory
            
            brain.two_model_pipeline = HybridPipelineAdapter(**adapter_kwargs)
            brain.two_model_pipeline_ready = True
            
            # Загружаем модели если нужен RecursiveModelPipeline
            if pipeline_mode in ['recursive', 'hybrid']:
                brain.two_model_pipeline.recursive_pipeline.load_models()
            
            query_logger.info(f"HybridPipelineAdapter инициализирован: mode={pipeline_mode}")
            
        except Exception as e:
            query_logger.error(f"Ошибка инициализации HybridPipelineAdapter: {e}")
            # Fallback на старый RecursiveModelPipeline
            try:
                from eva_ai.core.recursive_model_pipeline import RecursiveModelPipeline
                pipeline_kwargs = {
                    'model_a_path': model_a_path,
                    'model_b_path': model_b_path,
                    'n_ctx': n_ctx,
                    'n_threads': n_threads
                }
                if brain.fractal_memory:
                    pipeline_kwargs['fractal_memory'] = brain.fractal_memory
                brain.two_model_pipeline = RecursiveModelPipeline(**pipeline_kwargs)
                brain.two_model_pipeline.load_models()
                brain.two_model_pipeline_ready = True
                query_logger.info("Fallback на RecursiveModelPipeline")
            except Exception as e2:
                query_logger.error(f"Fallback тоже не работает: {e2}")
                return

        query_logger.info(f"Two-Model Pipeline готов к работе! (mode={pipeline_mode})")

        try:
            from eva_ai.core.event_bus import Event, EventTypes
            if brain.events:
                brain.events.trigger(EventTypes.COMPONENT_INITIALIZED, data={
                    "component": "TwoModelPipeline",
                    "model_a": model_a_path,
                    "model_b": model_b_path,
                    "n_ctx": n_ctx,
                    "n_threads": n_threads,
                    "mode": pipeline_mode,
                    "ready": True
                })
        except Exception:
            pass
    except Exception as e:
        query_logger.error(f"Ошибка инициализации Two-Model Pipeline: {e}")
        brain.two_model_pipeline = None


def _init_preprocessing(brain):
    brain.preprocessing_pipeline = None
    try:
        from ..preprocess.preprocessing_pipeline import PreprocessingPipeline
        llama_instance = brain.llama_cpp_deployment.llama if brain.llama_cpp_deployment and hasattr(brain.llama_cpp_deployment, 'llama') else None
        brain.preprocessing_pipeline = PreprocessingPipeline(llama_instance=llama_instance, hybrid_cache=brain.hybrid_cache)
    except Exception as e:
        query_logger.debug(f"PreprocessingPipeline не инициализирован: {e}")


def _init_qwen_config(brain):
    """DISABLED - Using UnifiedGenerator instead of QwenModelManager"""
    brain.qwen_model_manager = None
    brain._qwen_config = None
    query_logger.info("QwenModelManager disabled - using UnifiedGenerator")
    return


def _init_background(brain):
    try:
        try:
            from .background_coordinator import BackgroundCoordinator, Policies
        except ImportError:
            query_logger.warning("background_coordinator module not found, skipping")
            return
        brain.background = BackgroundCoordinator(
            brain=brain, deferred_system=getattr(brain, 'deferred_system', None),
            resource_manager=getattr(brain, 'resource_manager', None),
            metrics_manager=getattr(brain, 'metrics_manager', None),
            state_manager=getattr(brain, 'state_manager', None),
            policies=Policies(
                idle_threshold_s=float(brain.config.get('autopilot_idle_threshold_s', 10.0)),
                cpu_threshold_soft=float(brain.config.get('autopilot_cpu_soft', 0.90)),
                cpu_threshold_hard=float(brain.config.get('autopilot_cpu_hard', 0.95))
            )
        )
        brain.components['background_coordinator'] = brain.background
        try:
            from .background_jobs.training_job import TrainingJob
            from .background_jobs.web_index_job import WebIndexJob
            from .background_jobs.module_recovery_job import ModuleRecoveryJob
            brain.background.register_job_type(TrainingJob)
            brain.background.register_job_type(WebIndexJob)
            brain.background.register_job_type(ModuleRecoveryJob)
        except Exception:
            pass
        try:
            from .opportunities.learning_detector import LearningOpportunityDetector
            from .opportunities.web_discovery_detector import WebDiscoveryDetector
            from .opportunities.recovery_detector import ModuleRecoveryDetector
            brain.background.register_detector(LearningOpportunityDetector())
            brain.background.register_detector(WebDiscoveryDetector())
            brain.background.register_detector(ModuleRecoveryDetector())
        except Exception as e:
            logger.warning(f"Не удалось зарегистрировать Detectors: {e}")
    except Exception as e:
        logger.warning(f"Не удалось инициализировать BackgroundCoordinator: {e}")
        brain.background = None


def _init_mode_controller(brain):
    """Инициализация контроллера режимов модели (язык, квантование)."""
    import sys
    import os
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    
    try:
        from eva_ai.mlearning.language_filter import ModelModeController
        brain.mode_controller = ModelModeController(brain=brain)
        
        default_language = brain.config.get('model', {}).get('language_mode', 'russian_only')
        brain.mode_controller.set_language_mode(default_language)
        
        brain.components['mode_controller'] = brain.mode_controller
        logger.info(f"ModelModeController инициализирован (язык: {default_language})")
    except Exception as e:
        logger.warning(f"Не удалось инициализировать ModelModeController: {e}")
        brain.mode_controller = None


class ComponentMixin:
    """Mixin providing component initialization, deferred commands, module management, and model management to CoreBrain."""

    def _register_deferred_system_handlers(self):
        if not getattr(self, 'deferred_system', None):
            return

        if hasattr(self, 'two_model_pipeline') and self.two_model_pipeline:
            def check_pipeline():
                return hasattr(self, 'two_model_pipeline_ready') and self.two_model_pipeline_ready
            self.deferred_system.add_module_health_check('two_model_pipeline', check_pipeline)
            def recover_pipeline():
                try:
                    self.two_model_pipeline.load_models()
                    self.two_model_pipeline_ready = True
                    logger.info("Two-Model Pipeline restored")
                except Exception as e:
                    logger.error(f"Failed to restore Two-Model Pipeline: {e}")
            self.deferred_system.add_module_recovery_strategy('two_model_pipeline', recover_pipeline)

        if hasattr(self, 'self_reasoning_engine') and self.self_reasoning_engine:
            def check_sre():
                return hasattr(self.self_reasoning_engine, 'brain') and self.self_reasoning_engine.brain is not None
            self.deferred_system.add_module_health_check('self_reasoning_engine', check_sre)

        if hasattr(self, 'llama_cpp_deployment') and self.llama_cpp_deployment:
            def check_llama():
                return hasattr(self, 'llama_cpp_ready') and self.llama_cpp_ready
            self.deferred_system.add_module_health_check('llama_cpp_deployment', check_llama)

        if hasattr(self, 'web_search_engine') and self.web_search_engine:
            def check_web_search():
                return True
            self.deferred_system.add_module_health_check('web_search_engine', check_web_search)
        
        if hasattr(self, 'web_gui') and self.web_gui:
            def check_web_gui():
                return hasattr(self, 'web_gui') and self.web_gui is not None and getattr(self.web_gui, 'running', False)
            self.deferred_system.add_module_health_check('web_gui', check_web_gui)
            def recover_web_gui():
                try:
                    if hasattr(self, 'web_gui') and self.web_gui:
                        if not getattr(self.web_gui, 'running', False):
                            self.web_gui.start()
                            logger.info("WebGUI recovered")
                except Exception as e:
                    logger.error("Failed to recover WebGUI: {}".format(e))
            self.deferred_system.add_module_recovery_strategy('web_gui', recover_web_gui)

        logger.info("Health checks and recovery strategies registered for deferred system")

    def _execute_deferred_commands(self):
        with self._deferred_commands_lock:
            commands_to_execute = list(self.deferred_commands)
            self.deferred_commands.clear()
        query_logger.info(f"Executing {len(commands_to_execute)} deferred commands...")
        for command, args, kwargs in commands_to_execute:
            try:
                command(*args, **kwargs)
                query_logger.info(f"Deferred command {getattr(command, '__name__', 'lambda')} executed successfully.")
            except Exception as e:
                query_logger.error(f"Error executing deferred command {getattr(command, '__name__', 'lambda')}: {e}", exc_info=True)
        query_logger.info("All deferred commands executed.")

    def add_deferred_command(self, command: callable, *args, **kwargs):
        query_logger.info(f"Deferred command added: {getattr(command, '__name__', 'lambda')}")
        with self._deferred_commands_lock:
            self.deferred_commands.append((command, args, kwargs))

    def _initialize_memory_manager(self) -> bool:
        try:
            if not self.component_initializer:
                query_logger.warning("ComponentInitializer недоступен")
                return False
            if hasattr(self, 'memory_manager') and self.memory_manager is not None:
                return True
            
            # Проверяем component_initializer и brain.components
            if hasattr(self.component_initializer, 'memory_manager'):
                self.memory_manager = self.component_initializer.memory_manager
            elif hasattr(self.component_initializer, 'components') and 'memory_manager' in self.component_initializer.components:
                self.memory_manager = self.component_initializer.components.get('memory_manager')
            elif hasattr(self, 'components') and 'memory_manager' in self.components:
                self.memory_manager = self.components.get('memory_manager')
            
            if self.memory_manager:
                self.components['memory_manager'] = self.memory_manager
                if hasattr(self.memory_manager, 'initialize'):
                    return self.memory_manager.initialize()
                return True
            else:
                query_logger.warning("memory_manager не найден в component_initializer")
                return False
        except Exception as e:
            query_logger.error(f"Ошибка инициализации MemoryManager: {e}")
            return False

    def _initialize_detailed_logging(self):
        import torch, psutil, sys
        query_logger.debug("ДЕТАЛЬНОЕ ЛОГГИРОВАНИЕ ЗАПУСКА СИСТЕМЫ COGNIFLEX")
        query_logger.debug(f"Python version: {sys.version}")
        query_logger.debug(f"Platform: {sys.platform}")
        query_logger.debug(f"CPU count: {os.cpu_count()}")
        mem = psutil.virtual_memory()
        query_logger.debug(f"Total RAM: {mem.total / (1024**3):.2f} GB")
        query_logger.debug(f"Available RAM: {mem.available / (1024**3):.2f} GB")
        query_logger.debug(f"RAM usage: {mem.percent}%")
        disk = psutil.disk_usage('.')
        query_logger.debug(f"Total disk: {disk.total / (1024**3):.2f} GB")
        query_logger.debug(f"Free disk: {disk.free / (1024**3):.2f} GB")
        query_logger.debug(f"Disk usage: {disk.percent}%")
        if torch.cuda.is_available():
            query_logger.debug(f"CUDA available: Yes")
            query_logger.debug(f"CUDA device count: {torch.cuda.device_count()}")
            query_logger.debug(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        else:
            query_logger.debug("CUDA available: No")
        return True

    def _ensure_module_entry(self, name: str) -> Dict[str, Any]:
        if name not in self.module_control:
            self.module_control[name] = {"enabled": True, "status": "unknown", "last_error": None, "last_change": time.time()}
        return self.module_control[name]

    def enable_module(self, name: str) -> bool:
        entry = self._ensure_module_entry(name)
        entry["enabled"] = True; entry["last_change"] = time.time()
        query_logger.info(f"Модуль '{name}' включен")
        return True

    def disable_module(self, name: str, stop_if_running: bool = True) -> bool:
        entry = self._ensure_module_entry(name)
        entry["enabled"] = False; entry["last_change"] = time.time()
        component = self.components.get(name)
        try:
            if stop_if_running and component and hasattr(component, 'stop'):
                component.stop(); entry["status"] = "stopped"
                query_logger.info(f"Модуль '{name}' остановлен при отключении")
            else:
                query_logger.info(f"Модуль '{name}' отключен")
            return True
        except Exception as e:
            entry["last_error"] = str(e)
            query_logger.error(f"Ошибка при отключении модуля '{name}': {e}", exc_info=True)
            return False

    def start_module(self, name: str) -> bool:
        entry = self._ensure_module_entry(name)
        if not entry.get("enabled", True):
            query_logger.warning(f"Попытка запуска отключенного модуля '{name}'"); return False
        component = self.components.get(name)
        if not component or not hasattr(component, 'start'):
            query_logger.warning(f"Компонент '{name}' не найден или не поддерживает start()"); return False
        try:
            component.start(); entry["status"] = "running"; entry["last_error"] = None; entry["last_change"] = time.time()
            query_logger.info(f"Модуль '{name}' запущен"); return True
        except Exception as e:
            entry["last_error"] = str(e); entry["status"] = "error"
            if hasattr(self.metrics_manager, 'record_error'):
                self.metrics_manager.record_error(f"module_{name}_start_failed")
            query_logger.error(f"Ошибка запуска модуля '{name}': {e}", exc_info=True); return False

    def stop_module(self, name: str) -> bool:
        entry = self._ensure_module_entry(name)
        component = self.components.get(name)
        if not component or not hasattr(component, 'stop'):
            query_logger.warning(f"Компонент '{name}' не найден или не поддерживает stop()"); return False
        try:
            component.stop(); entry["status"] = "stopped"; entry["last_error"] = None; entry["last_change"] = time.time()
            query_logger.info(f"Модуль '{name}' остановлен"); return True
        except Exception as e:
            entry["last_error"] = str(e); entry["status"] = "error"
            query_logger.error(f"Ошибка остановки модуля '{name}': {e}", exc_info=True); return False

    def get_module_status(self, name: str) -> Dict[str, Any]:
        entry = self._ensure_module_entry(name)
        component = self.components.get(name)
        status = {"enabled": entry.get("enabled", True), "status": entry.get("status", "unknown"),
                  "last_error": entry.get("last_error"), "last_change": entry.get("last_change"),
                  "running": bool(getattr(component, 'running', False)) if component else False, "healthy": None}
        try:
            if component and hasattr(component, 'health_check'):
                hc = component.health_check()
                status["healthy"] = bool(hc.get('healthy')) if isinstance(hc, dict) else bool(hc)
        except Exception:
            status["healthy"] = False
        return status

    def list_modules(self) -> Dict[str, Dict[str, Any]]:
        return {name: self.get_module_status(name) for name in sorted(self.components.keys())}

    def check_module_dependencies(self, module_name: str) -> List[str]:
        dependencies = {'model_manager': ['ml_unit'], 'response_generator': ['model_manager', 'text_processor'],
                        'integrated_learning_manager': ['model_manager', 'knowledge_graph'],
                        'analytics_manager': ['learning_manager', 'memory_manager'],
                        'learning_processor': ['model_manager', 'hybrid_cache']}
        missing_deps = []
        if module_name in dependencies:
            for dep in dependencies[module_name]:
                if dep not in self.components:
                    missing_deps.append(dep)
        return missing_deps

    def register_component(self, name: str, component: Any) -> bool:
        try:
            self.components[name] = component
            query_logger.debug(f"Компонент '{name}' зарегистрирован в CoreBrain"); return True
        except Exception as e:
            query_logger.error(f"Ошибка регистрации компонента '{name}': {e}"); return False

    def get_component(self, name: str) -> Any:
        return self.components.get(name)

    def get_available_models(self) -> List[Dict[str, Any]]:
        try:
            ml_unit = self.components.get('ml_unit')
            if ml_unit and hasattr(ml_unit, 'get_available_models'):
                return ml_unit.get_available_models()
            if hasattr(self, 'model_manager') and self.model_manager and hasattr(self.model_manager, 'get_available_models'):
                return self.model_manager.get_available_models()
            return []
        except Exception as e:
            query_logger.warning(f"get_available_models: ошибка получения списка моделей: {e}"); return []

    @property
    def knowledge_graph(self):
        kg = self.components.get('knowledge_graph')
        if kg is None:
            query_logger.debug("knowledge_graph не инициализирован или недоступен")
        return kg

    @knowledge_graph.setter
    def knowledge_graph(self, value):
        if value is not None:
            query_logger.debug(f"Установка компонента knowledge_graph: {type(value).__name__}")
        self.components['knowledge_graph'] = value

    @property
    def qwen_api_enhancer(self):
        return self.components.get('qwen_api_enhancer')

    @qwen_api_enhancer.setter
    def qwen_api_enhancer(self, value):
        query_logger.debug(f"Установка компонента qwen_api_enhancer: {value}")
        self.components['qwen_api_enhancer'] = value

    def is_model_ready(self, model_id: str) -> bool:
        try:
            mm = getattr(self, 'model_manager', None)
            if not mm: return False
            return model_id in getattr(mm, 'models', {})
        except Exception:
            return False

    def ensure_model_available(self, model_id: str, wait: bool = False, timeout_s: float = 0.0) -> Dict[str, Any]:
        info = {"requested": model_id, "started": False, "ready": False, "waited": 0.0, "remaining": None}
        mm = getattr(self, 'model_manager', None)
        if not mm:
            info["error"] = "ModelManager недоступен"; return info
        try:
            if not self.is_model_ready(model_id):
                info["started"] = bool(mm.load_model(model_id)) if hasattr(mm, 'load_model') else False
            if wait:
                t0 = time.time(); deadline = t0 + max(0.0, float(timeout_s))
                while time.time() < deadline:
                    if self.is_model_ready(model_id): info["ready"] = True; break
                    time.sleep(0.2)
                info["waited"] = round(time.time() - t0, 3)
                info["remaining"] = max(0.0, round(deadline - time.time(), 3))
            else:
                info["ready"] = self.is_model_ready(model_id)
            return info
        except Exception as e:
            info["error"] = str(e); return info

    # === GRAPH API ===
    
    def get_graph_tools(self) -> Dict[str, Any]:
        """Получить инструменты для работы с графом знаний."""
        return {
            "curator": self._get_curator_tools(),
            "fractal_graph_v2": self._get_fractal_graph_tools()
        }
    
    def _get_curator_tools(self) -> Dict[str, Any]:
        """Инструменты куратора графа."""
        if not hasattr(self, 'graph_curator') or not self.graph_curator:
            return {"available": False, "reason": "Curator not initialized"}
        
        curator = self.graph_curator
        return {
            "available": True,
            "state": curator.get_state(),
            "metrics": curator.get_metrics(),
            "actions": {
                "start": lambda: curator.start() if not curator.running else None,
                "stop": curator.stop,
                "pause": curator.pause,
                "resume": curator.resume,
                "force_curation": curator.force_curation,
                "get_stats": curator.get_graph_stats
            }
        }
    
    def _get_fractal_graph_tools(self) -> Dict[str, Any]:
        """Инструменты Fractal Graph V2."""
        fg = getattr(self, 'fractal_graph_v2', None)
        if not fg:
            return {"available": False, "reason": "FractalGraphV2 not initialized"}
        
        return {
            "available": True,
            "stats": fg.get_stats(),
            "actions": {
                "add_node": fg.add_node,
                "add_knowledge": fg.add_knowledge,
                "semantic_search": fg.semantic_search,
                "keyword_search": fg.keyword_search,
                "check_contradiction": fg.check_contradiction,
                "self_dialogue": fg.self_dialogue,
                "save_experience": fg.save_experience,
                "get_context_for_query": fg.get_context_for_query,
                "retrieve_knowledge": fg.retrieve_knowledge,
                "vectorize_all": fg.vectorize_all,
                "vectorize_groups": fg.vectorize_groups,
                "auto_cluster": fg.auto_cluster
            }
        }
    
    def execute_graph_command(self, command: str, **kwargs) -> Any:
        """Выполнить команду управления графом."""
        cmd = command.lower().strip()
        
        # Curator commands
        if cmd in ('curator_start', 'curator.start'):
            if hasattr(self, 'graph_curator') and self.graph_curator:
                return self.graph_curator.start()
            return {"error": "Curator not available"}
        
        elif cmd in ('curator_stop', 'curator.stop'):
            if hasattr(self, 'graph_curator') and self.graph_curator:
                self.graph_curator.stop()
                return {"status": "stopped"}
            return {"error": "Curator not available"}
        
        elif cmd in ('curator_status', 'curator.status'):
            if hasattr(self, 'graph_curator') and self.graph_curator:
                return self.graph_curator.get_state()
            return {"error": "Curator not available"}
        
        elif cmd in ('curator_metrics', 'curator.metrics'):
            if hasattr(self, 'graph_curator') and self.graph_curator:
                return self.graph_curator.get_metrics()
            return {"error": "Curator not available"}
        
        elif cmd in ('curator_force', 'curator.force'):
            if hasattr(self, 'graph_curator') and self.graph_curator:
                self.graph_curator.force_curation()
                return {"status": "curation started"}
            return {"error": "Curator not available"}
        
        # Graph stats
        elif cmd in ('graph_stats', 'graph.stats'):
            fg = getattr(self, 'fractal_graph_v2', None)
            if fg:
                return fg.get_stats()
            kg = getattr(self, 'knowledge_graph', None)
            if kg and hasattr(kg, 'get_stats'):
                return kg.get_stats()
            return {"error": "No graph available"}
        
        # Search
        elif cmd.startswith('search '):
            query = command[7:]
            fg = getattr(self, 'fractal_graph_v2', None)
            if fg:
                return fg.semantic_search(query, top_k=kwargs.get('top_k', 5))
            return {"error": "FractalGraphV2 not available"}
        
        # Add knowledge
        elif cmd.startswith('add '):
            parts = command[4:].split(' | ')
            if len(parts) >= 3:
                subject, relation, obj = parts[0], parts[1], parts[2]
                fg = getattr(self, 'fractal_graph_v2', None)
                if fg:
                    return fg.add_knowledge(subject, relation, obj)
            return {"error": "Invalid format. Use: add subject | relation | object"}
        
        return {"error": f"Unknown command: {command}"}



def _init_unified_generator(brain):
    '''DEPRECATED - используется FCPipeline.
    
    FCPipeline - основной и единственный пайплайн генерации.
    '''
    brain.two_model_pipeline = None
    brain.two_model_pipeline_ready = False
    brain.fcp_pipeline = None
    brain.fcp_pipeline_ready = False
    
    try:
        fcp_config = brain.config.get('fcp_pipeline', {})
        query_logger.info(f"FCP config: {fcp_config}")
        
        if not fcp_config:
            query_logger.warning("FCP config empty or not found")
        
        if not fcp_config.get('enabled', True):
            query_logger.info("FCPipeline disabled")
            return
        
        # Debug: проверяем импорт
        query_logger.info("Importing FCPipeline...")
        from eva_ai.core.fcp_pipeline import FCPipeline, get_fcp_pipeline
        query_logger.info("Import OK")
        
        if hasattr(brain, 'fcp_pipeline') and brain.fcp_pipeline is not None:
            existing = get_fcp_pipeline()
            if existing is not None:
                brain.fcp_pipeline = existing
                query_logger.info(f"FCPipeline already initialized, using existing instance")
                return
        
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        model_path = fcp_config.get('model_path')
        if not model_path:
            model_path = fcp_config.get('openvino_path', '')
            if not model_path or not os.path.exists(model_path):
                query_logger.warning("No model_path in fcp_pipeline config, FCP will be initialized later")
                return
        
        query_logger.info(f"Model path: {model_path}, exists: {os.path.exists(model_path)}")
        
        if not os.path.exists(model_path):
            query_logger.error(f"Model not found: {model_path}")
            return
        
        graph_path = fcp_config.get('graph_path')
        gnn_ov_path = fcp_config.get('gnn_ov_path')
        lora_dir = fcp_config.get('lora_dir')
        
        query_logger.info("Creating FCPipeline instance...")
        pipeline = FCPipeline(
            model_path=model_path,
            graph_path=graph_path,
            gnn_ov_path=gnn_ov_path,
            lora_dir=lora_dir
        )
        query_logger.info(f"FCPipeline created: {type(pipeline)}")
        brain.fcp_pipeline = pipeline
        brain.fcp_pipeline_ready = True
        brain.two_model_pipeline = pipeline
        brain.two_model_pipeline_ready = True
        
        query_logger.info(f"FCPipeline ready!")
    except Exception as e:
        import traceback
        query_logger.error(f"ERROR: {e}")
        query_logger.error(traceback.format_exc())
        brain.fcp_pipeline = None
        brain.fcp_pipeline_ready = False
        brain.two_model_pipeline = None
        brain.two_model_pipeline_ready = False


def _init_hybrid_dialog_manager(brain):
    """
    Инициализация HybridKnowledgeDialogManager с двумя физическими моделями.
    
    Этот менеджер обеспечивает:
    - Chat template форматирование для Qwen3-4B
    - Prefix caching для истории диалога
    - Виртуальные токены знаний из FractalGraphV2
    - Интеграция с ConceptExtractor и ContradictionMiner
    - Валидация ответов
    - Две физические модели (Model A + Model B)
    """
    brain.hybrid_dialog_manager = None
    
    try:
        from eva_ai.core.hybrid_dialog_manager import create_hybrid_dialog_manager
        
        model_config = brain.config.get('model', {})
        
        # Check FCP first (новый единый пайплайн)
        if hasattr(brain, 'fcp_pipeline') and brain.fcp_pipeline:
            if hasattr(brain.fcp_pipeline, 'pipeline') and brain.fcp_pipeline.pipeline:
                model_a_path = brain.fcp_pipeline.model_path
                if model_a_path:
                    query_logger.info(f'  FCP Model: {model_a_path}')
                    return
            elif hasattr(brain.fcp_pipeline, 'model_path') and brain.fcp_pipeline.model_path:
                # FCP существует но pipeline ещё не инициализирован - используем его model_path
                model_a_path = brain.fcp_pipeline.model_path
                query_logger.info(f'  FCP Model (pending): {model_a_path}')
                return
        
        # Legacy: two_model_pipeline
        model_a_path = model_config.get('logic_model_path', '')
        model_b_path = model_config.get('model_b_openvino_path', '')
        
        if not model_a_path or not os.path.exists(model_a_path):
            # Пробуем альтернативные пути
            if hasattr(brain, 'two_model_pipeline') and brain.two_model_pipeline:
                try:
                    if hasattr(brain.two_model_pipeline, '_openvino_cpu'):
                        model_a_path = getattr(brain.two_model_pipeline._openvino_cpu, '_model_path', '')
                    elif hasattr(brain.two_model_pipeline, 'model_a_path'):
                        model_a_path = brain.two_model_pipeline.model_a_path
                except:
                    pass
            
            # Fallback: проверяем pipeline напрямую
            if not model_a_path and hasattr(brain, 'pipeline') and brain.pipeline:
                if hasattr(brain.pipeline, '_model_path'):
                    model_a_path = brain.pipeline._model_path
                elif hasattr(brain.pipeline, 'model_path'):
                    model_a_path = brain.pipeline.model_path
        
        if not model_a_path:
            query_logger.warning('Путь к модели не найден для HybridKnowledgeDialogManager')
            return
        
        # Получаем устройство
        device = model_config.get('cpu_device', 'CPU')
        
        # Получаем компоненты графа памяти (fractal_graph или fractal_graph_v2)
        fractal_graph = getattr(brain, 'fractal_graph', None) or getattr(brain, 'fractal_graph_v2', None)
        
        # Получаем concept_extractor
        concept_extractor = None
        if hasattr(brain, 'concept_extractor'):
            concept_extractor = brain.concept_extractor
        elif hasattr(brain, 'knowledge') and hasattr(brain.knowledge, 'concept_extractor'):
            concept_extractor = brain.knowledge.concept_extractor
        
        # Получаем contradiction_manager
        contradiction_manager = None
        if hasattr(brain, 'contradiction_manager'):
            contradiction_manager = brain.contradiction_manager
        elif hasattr(brain, 'knowledge') and hasattr(brain.knowledge, 'contradiction_manager'):
            contradiction_manager = brain.knowledge.contradiction_manager
        
        # Создаём менеджер с двумя моделями (initialize вызывается внутри factory)
        manager = create_hybrid_dialog_manager(
            brain=brain,
            fractal_graph=fractal_graph,
            concept_extractor=concept_extractor,
            contradiction_manager=contradiction_manager,
            model_path=model_a_path,
            model_b_path=model_b_path,
            device=device,
            enable_validation=True,
            max_history=50,
            max_tokens=4096,
            temperature=0.7
        )
        
        if manager.initialized:
            brain.hybrid_dialog_manager = manager
            query_logger.info(f'HybridKnowledgeDialogManager инициализирован:')
            query_logger.info(f'  Model A: {model_a_path}')
            query_logger.info(f'  Model B: {model_b_path}')
            query_logger.info(f'  device: {device}')
            query_logger.info(f'  concept_extractor: {concept_extractor is not None}')
            query_logger.info(f'  contradiction_manager: {contradiction_manager is not None}')
            query_logger.info(f'  fractal_graph: {fractal_graph is not None}')
        else:
            query_logger.error('HybridKnowledgeDialogManager не удалось инициализировать')
            
    except ImportError as e:
        query_logger.warning(f'HybridKnowledgeDialogManager не доступен: {e}')
    except Exception as e:
        query_logger.error(f'Ошибка инициализации HybridKnowledgeDialogManager: {e}')

