"""
Единое ядро системы CogniFlex - координирует работу всех компонентов
"""
import sys
import os
import logging
import time
import threading
import queue
import datetime
import random
import psutil
import torch
from typing import Dict, Any, Optional, List, Tuple

from .background_coordinator import BackgroundCoordinator, Policies
from .opportunities.learning_detector import LearningOpportunityDetector
from .background_jobs.training_job import TrainingJob
from .autopilot_cache import AutopilotCache
from .opportunities.web_discovery_detector import WebDiscoveryDetector
from .opportunities.recovery_detector import ModuleRecoveryDetector
from .background_jobs.web_index_job import WebIndexJob
from .background_jobs.module_recovery_job import ModuleRecoveryJob
from .generation_coordinator import initialize_generation_coordinator, get_generation_coordinator

logger = logging.getLogger("cogniflex.core_brain")
query_logger = logging.getLogger("cogniflex.core_brain.query_processing")

# Глобальная ссылка на текущий экземпляр CoreBrain (для доступа из других модулей)
_global_brain_instance: Optional['CoreBrain'] = None

# Используем реальный QueryProcessor из модуля cogniflex.core.query_processor
try:
    from .query_processor import QueryProcessor
except Exception:
    QueryProcessor = None  # Will be checked at runtime

try:
    from cogniflex.learning.self_dialog_learning import SelfDialogLearningSystem
except ImportError:
    SelfDialogLearningSystem = None

try:
    from cogniflex.knowledge.online_knowledge import OnlineKnowledgeAccess
except ImportError:
    OnlineKnowledgeAccess = None

# Глобальный импорт/фолбэк для SystemState, чтобы использовать его в методах класса
try:
    from .system_state import SystemState  # Enum с состояниями системы
except Exception:
    # Улучшенный фолбэк-класс с полноценной функциональностью
    class SystemState:  # type: ignore
        """Fallback реализация SystemState с дополнительной функциональностью."""
        
        # Базовые состояния системы
        INITIALIZING = "INITIALIZING"
        READY = "READY"
        ERROR = "ERROR"
        OFFLINE = "OFFLINE"
        SHUTTING_DOWN = "SHUTTING_DOWN"
        MAINTENANCE = "MAINTENANCE"
        DEGRADED = "DEGRADED"
        
        # Дополнительные состояния для детализации
        LOADING_MODELS = "LOADING_MODELS"
        INITIALIZING_COMPONENTS = "INITIALIZING_COMPONENTS"
        CONNECTING_SERVICES = "CONNECTING_SERVICES"
        RECOVERING = "RECOVERING"
        
        # Методы для валидации и сравнения
        @classmethod
        def is_valid_state(cls, state: str) -> bool:
            """Проверяет, является ли состояние валидным."""
            return hasattr(cls, state) and isinstance(getattr(cls, state), str)
        
        @classmethod
        def get_all_states(cls) -> list:
            """Возвращает все доступные состояния."""
            return [attr for attr in dir(cls) if not attr.startswith('_') and isinstance(getattr(cls, attr), str)]
        
        @classmethod
        def is_operational_state(cls, state: str) -> bool:
            """Проверяет, является ли состояние рабочим."""
            operational_states = [cls.READY, cls.INITIALIZING, cls.LOADING_MODELS, 
                                 cls.INITIALIZING_COMPONENTS, cls.CONNECTING_SERVICES]
            return state in operational_states
        
        @classmethod
        def is_error_state(cls, state: str) -> bool:
            """Проверяет, является ли состояние ошибочным."""
            error_states = [cls.ERROR, cls.DEGRADED, cls.SHUTTING_DOWN]
            return state in error_states


class CoreBrain:
    """Центральный координатор системы CogniFlex."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Инициализирует ядро CogniFlex."""
        logger.debug("Инициализация CogniFlexCore...")
        
        # Инициализируем query_logger ПЕРЕД загрузкой конфигурации
        self.query_logger = logging.getLogger("cogniflex.core_brain.query_processing")
        self.query_logger.info("Инициализирован логгер обработки запросов")
        
        # Если конфигурация не передана, загружаем из brain_config.json
        if config is None:
            config = self._load_brain_config()
        
        # Инициализация событийной системы
        try:
            from .event_system import EventSystem
            self.events = EventSystem()
            self.query_logger.debug("Событийная система инициализирована")
            
            # Централизованный транспорт метрик через событийную шину
            try:
                self.events.subscribe('metrics', self._on_metrics_event)
                self.query_logger.debug("Подписка на события 'metrics' зарегистрирована")
            except Exception:
                pass
        except ImportError:
            self.events = None
            self.query_logger.warning("Событийная система недоступна")
        
        # Логируем получение конфигурации
        if config:
            self.query_logger.debug(f"Получена конфигурация с {len(config)} параметрами")
            if 'secret' in config or 'password' in config:
                masked_config = {k: '***' if k in ['secret', 'password'] else v for k, v in config.items()}
                self.query_logger.debug(f"Конфигурация (с маскировкой): {masked_config}")
            else:
                self.query_logger.debug(f"Конфигурация: {config}")
        else:
            self.query_logger.debug("Конфигурация не предоставлена, используется конфигурация по умолчанию")
        
        self.config = config or {}
        self.components: Dict[str, Any] = {}
        
        # Мониторинг активности модулей
        self.module_activity: Dict[str, Dict[str, Any]] = {}
        self.module_access_log: List[Dict[str, Any]] = []
        self.activity_lock = threading.Lock()
        
        # Контроллер модулей: хранит флаги включения и состояние
        self.module_control: Dict[str, Dict[str, Any]] = {}
        self.initialized = False
        self.running = False
        self._shutting_down = False
        self._shutdown_lock = threading.Lock()
        self.status_queue = queue.Queue()
        self.deferred_commands = []
        
        # Настройки троттлинга логов
        self.log_throttle_seconds = int(self.config.get("system", {}).get("log_throttle_seconds", 30))
        self._log_throttle: Dict[str, float] = {}
        
        # Инициализация системы отложенных команд
        try:
            from .deferred_command_system import DeferredCommandSystem
            self.deferred_system = DeferredCommandSystem(self, max_workers=6)
            self.query_logger.debug("Система отложенных команд инициализирована")
        except ImportError as e:
            self.deferred_system = None
            self.query_logger.warning(f"Система отложенных команд недоступна: {e}")
        
        # Настройка директории кэша
        self.cache_dir = os.path.join(os.path.dirname(__file__), "cogniflex_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.query_logger.debug(f"Путь к кэшу: {self.cache_dir}")
        
        # Применяем контекст-ориентированную политику при необходимости
        try:
            mode = str(self.config.get('mode') or os.environ.get('COGNIFLEX_MODE') or '').lower()
            if mode == 'context_first':
                try:
                    from .context_first_policy import ContextFirstPolicy
                    ContextFirstPolicy(self).apply()
                    self.query_logger.debug("ContextFirstPolicy применена (mode=context_first)")
                except Exception as e:
                    self.query_logger.warning(f"Не удалось применить ContextFirstPolicy: {e}")
        except Exception:
            pass
        
        # Инициализация новых менеджеров
        try:
            from .config_manager import ConfigManager
            self.config_manager = ConfigManager()
            self.query_logger.debug("Менеджер конфигурации инициализирован")
        except ImportError:
            self.config_manager = None
            self.query_logger.warning("Менеджер конфигурации недоступен")
        
        try:
            from .system_state import SystemStateManager, SystemState
            self.state_manager = SystemStateManager()
            self.query_logger.debug("Менеджер состояния системы инициализирован")
            self.state_manager.set_state(SystemState.INITIALIZING, "Начало инициализации CoreBrain")
        except ImportError:
            self.state_manager = None
            self.query_logger.warning("Менеджер состояния системы недоступен")
        
        try:
            from .resource_manager import ResourceManager
            self.resource_manager = ResourceManager(self.config_manager)
            self.query_logger.debug("Менеджер ресурсов инициализирован")
        except ImportError:
            self.resource_manager = None
            self.query_logger.warning("Менеджер ресурсов недоступен")
        
        # Инициализация модуля самоанализа
        try:
            from cogniflex.learning.self_analyzer import SelfAnalyzer
            self.self_analyzer = SelfAnalyzer(brain=self, cache_dir=self.cache_dir)
            self.query_logger.debug("Модуль самоанализа инициализирован")
        except ImportError as e:
            self.self_analyzer = None
            self.query_logger.warning(f"Модуль самоанализа недоступен: {e}")
        
        # Инициализация менеджера системных метрик
        try:
            from .system_metrics import SystemMetricsManager
            self.metrics_manager = SystemMetricsManager()
            self.query_logger.debug("Менеджер системных метрик инициализирован")
        except ImportError:
            class SystemMetricsManager:
                def __init__(self): self.metrics = {"error_rate": 0.0}
                def start_tracking(self): pass
                def get_metrics(self): return self.metrics
                def record_error(self, error_type): pass
                def record_system_startup(self, time): pass
                def record_query_metrics(self, **kwargs): pass
            self.metrics_manager = SystemMetricsManager()
            self.query_logger.warning("Менеджер системных метрик недоступен, используется заглушка")
        
        # Инициализация расширенной системы самообучения с эпохами
        try:
            from .enhanced_self_learning import EnhancedSelfLearningSystem
            self.enhanced_learning = EnhancedSelfLearningSystem(self, config=self.config.get('learning', {}))
            if self.enhanced_learning.start():
                self.query_logger.debug("EnhancedSelfLearningSystem инициализирована и запущена")
            else:
                self.query_logger.warning("Не удалось запустить EnhancedSelfLearningSystem")
                self.enhanced_learning = None
        except ImportError as e:
            self.query_logger.warning(f"EnhancedSelfLearningSystem недоступна: {e}")
            self.enhanced_learning = None
        
        # Инициализация MemoryGraphML для обучения на графе памяти
        try:
            from .memory_graph_ml import MemoryGraphML
            self.memory_graph_ml = MemoryGraphML(self, config=self.config.get('memory_graph_ml', {}))
            if self.memory_graph_ml.initialize():
                self.query_logger.debug("MemoryGraphML инициализирован")
            else:
                self.query_logger.warning("Не удалось инициализировать MemoryGraphML")
        except ImportError as e:
            self.query_logger.warning(f"MemoryGraphML недоступен: {e}")
            self.memory_graph_ml = None
        
        # Self-Dialog Learning System
        try:
            if SelfDialogLearningSystem:
                self.self_dialog_learning = SelfDialogLearningSystem(
                    brain=self,
                    config=self.config.get('self_dialog_learning', {})
                )
                self.query_logger.info("SelfDialogLearningSystem initialized")
            else:
                self.self_dialog_learning = None
                self.query_logger.debug("SelfDialogLearningSystem not available")
        except Exception as e:
            self.self_dialog_learning = None
            self.query_logger.warning(f"SelfDialogLearningSystem initialization failed: {e}")
        
        # Online Knowledge Access
        try:
            if OnlineKnowledgeAccess:
                self.online_knowledge = OnlineKnowledgeAccess(
                    brain=self,
                    config=self.config.get('online_knowledge', {})
                )
                self.query_logger.debug("OnlineKnowledgeAccess initialized")
            else:
                self.online_knowledge = None
        except Exception as e:
            self.online_knowledge = None
            self.query_logger.debug(f"OnlineKnowledgeAccess initialization failed: {e}")
        
        # Инициализация системы самообучения (устаревшая, для совместимости)
        try:
            from .self_learning_system import initialize_self_learning
            if initialize_self_learning(self):
                self.query_logger.debug("Система самообучения инициализирована (legacy)")
            else:
                self.query_logger.warning("Не удалось инициализировать систему самообучения (legacy)")
        except ImportError as e:
            self.query_logger.warning(f"Система самообучения (legacy) недоступна: {e}")
        
        self.system_metrics_manager = self.metrics_manager  # Alias for compatibility
        
        # Устаревшие менеджеры для совместимости
        self.distributed_system = None
        
        # Инициализация процессора запросов
        self.query_processor = QueryProcessor(self) if QueryProcessor else None
        if self.query_processor:
            self.components['query_processor'] = self.query_processor
            self.query_logger.debug("Процессор запросов инициализирован и зарегистрирован в components")
        
        # Инициализация инициализатора компонентов
        try:
            from .component_initializer import ComponentInitializer
            self.component_initializer = ComponentInitializer(self)
            self.query_logger.debug("Инициализатор компонентов инициализирован")
        except ImportError as e:
            self.component_initializer = None
            self.query_logger.warning(f"Ошибка импорта инициализатора компонентов: {e}")
        except Exception as e:
            self.component_initializer = None
            self.query_logger.warning(f"Ошибка при инициализации компонента: {e}")
            self.query_logger.error(f"Ошибка инициализации компонентного инициализатора: {e}", exc_info=True)
        
        # Инициализация кэша токенов - используем синглтон
        try:
            from ..memory.hybrid_token_cache import get_shared_cache
            self.token_cache = get_shared_cache(self, "default")
            self.hybrid_cache = self.token_cache
            self.query_logger.debug("Гибридный кэш токенов инициализирован (синглтон)")
            if hasattr(self.token_cache, 'get_cache_stats'):
                cache_stats = self.token_cache.get_cache_stats()
        except ImportError as e:
            self.query_logger.warning(f"Ошибка импорта гибридного кэша: {e}")
        
        self.fractal_ready = False  # Флаг готовности фрактальной модели
        self.qwen_ready = False  # Флаг готовности Qwen модели
        self.models_ready = False  # Флаг готовности ML моделей
        
        # Инициализация FractalModelManager для загрузки модели из фрактального хранилища
        try:
            from ..mlearning.fractal_model_manager import FractalModelManager
            # Динамически определяем путь относительно расположения brain_config.json или текущей директории
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(project_root, "cogniflex", "core", "cogniflex_cache", "ml_unit", "fractal_storage", "models", "rugpt3_small_fractal", "model")
            self.fractal_model_manager = FractalModelManager(model_path=model_path)
            self.query_logger.debug(f"FractalModelManager инициализирован с путем: {model_path}")
        except (ImportError, Exception) as e:
            self.query_logger.debug(f"Ошибка инициализации FractalModelManager: {e}")
            self.fractal_model_manager = None
        
        # Инициализация QwenModelManager как предпочтительной модели (LAZY LOADING)
        # Модель загружается только при первом запросе, не блокирует запуск
        self.qwen_model_manager = None
        self.qwen_ready = False
        self._qwen_config = None
        try:
            model_config = self.config.get('model', {})
            model_type = model_config.get('type', '')
            model_name = model_config.get('name', '')
            
            if model_type == 'qwen' or model_name.startswith('qwen'):
                self._qwen_config = model_config
                self.query_logger.info(f"QwenModelManager будет загружен при первом запросе (type={model_type}, name={model_name})")
            else:
                self.query_logger.warning(f"Qwen НЕ ЗАГРУЖЕН: model.type='{model_type}', model.name='{model_name}'. Ожидается type='qwen' или name начинающееся с 'qwen'")
        except Exception as e:
            self.query_logger.warning(f"Qwen config не найден: {e}")
        
        # Устанавливаем глобальную ссылку на текущий экземпляр
        self.query_logger.debug(f"CoreBrain зарегистрирован как глобальный экземпляр: {id(self)}")
        
        # Логируем завершение инициализации
        self.query_logger.debug("CogniFlexCore инициализирован")
        logger.debug("CogniFlexCore инициализирован")
        
        # Подготовка автопилота (фоновый координатор)
        try:
            self.background: Optional[BackgroundCoordinator] = BackgroundCoordinator(
                brain=self,
                deferred_system=getattr(self, 'deferred_system', None),
                resource_manager=getattr(self, 'resource_manager', None),
                metrics_manager=getattr(self, 'metrics_manager', None),
                state_manager=getattr(self, 'state_manager', None),
                policies=Policies(
                    idle_threshold_s=float(self.config.get('autopilot_idle_threshold_s', 10.0)),
                    cpu_threshold_soft=float(self.config.get('autopilot_cpu_soft', 0.90)),
                    cpu_threshold_hard=float(self.config.get('autopilot_cpu_hard', 0.95))
                )
            )
            
            # Регистрируем BackgroundCoordinator как компонент
            if hasattr(self, 'components'):
                self.components['background_coordinator'] = self.background
                self.query_logger.debug("BackgroundCoordinator зарегистрирован как компонент")
            
            # Регистрируем фоновые задачи
            try:
                from .background_jobs.training_job import TrainingJob
                from .background_jobs.web_index_job import WebIndexJob
                from .background_jobs.module_recovery_job import ModuleRecoveryJob
                self.background.register_job_type(TrainingJob)
                self.background.register_job_type(WebIndexJob)
                self.background.register_job_type(ModuleRecoveryJob)
                self.query_logger.debug("Фоновые задачи зарегистрированы")
            except Exception as e:
                self.query_logger.warning(f"Не удалось зарегистрировать фоновые задачи: {e}")
                
        except Exception as e:
            logger.warning(f"Не удалось инициализировать BackgroundCoordinator: {e}")
            self.background = None
    
    def _initialize_memory_manager(self) -> bool:
        """
        Инициализирует менеджер памяти через component_initializer.
        Этот метод вызывается для обеспечения полной инициализации памяти.
        
        Returns:
            bool: True если инициализация успешна, False в противном случае
        """
        try:
            if not self.component_initializer:
                self.query_logger.warning("ComponentInitializer недоступен")
                return False
            
            if hasattr(self, 'memory_manager') and self.memory_manager is not None:
                return True
            
            if hasattr(self.component_initializer, 'memory_manager'):
                self.memory_manager = self.component_initializer.memory_manager
                self.components['memory_manager'] = self.memory_manager
                
                if hasattr(self.memory_manager, 'initialize'):
                    return self.memory_manager.initialize()
                
                return True
            else:
                self.query_logger.warning("memory_manager не найден в component_initializer")
                return False
                
        except Exception as e:
            self.query_logger.error(f"Ошибка инициализации MemoryManager: {e}")
            return False
    
    def _initialize_detailed_logging(self):
        """Включает детальное логгирование для всех компонентов."""
        self.query_logger.debug("ДЕТАЛЬНОЕ ЛОГГИРОВАНИЕ ЗАПУСКА СИСТЕМЫ COGNIFLEX")
        
        # Логируем информацию о системе (debug level)
        self.query_logger.debug(f"Python version: {sys.version}")
        self.query_logger.debug(f"Platform: {sys.platform}")
        self.query_logger.debug(f"CPU count: {os.cpu_count()}")
        
        # Информация о памяти
        mem = psutil.virtual_memory()
        self.query_logger.debug(f"Total RAM: {mem.total / (1024**3):.2f} GB")
        self.query_logger.debug(f"Available RAM: {mem.available / (1024**3):.2f} GB")
        self.query_logger.debug(f"RAM usage: {mem.percent}%")
        
        # Информация о диске
        disk = psutil.disk_usage('.')
        self.query_logger.debug(f"Total disk: {disk.total / (1024**3):.2f} GB")
        self.query_logger.debug(f"Free disk: {disk.free / (1024**3):.2f} GB")
        self.query_logger.debug(f"Disk usage: {disk.percent}%")
        
        # CUDA информация
        if torch.cuda.is_available():
            self.query_logger.debug(f"CUDA available: Yes")
            self.query_logger.debug(f"CUDA device count: {torch.cuda.device_count()}")
            self.query_logger.debug(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        else:
            self.query_logger.debug("CUDA available: No")
        
        return True
    
    def initialize(self) -> bool:
        """Инициализирует все компоненты системы."""
        
        # Включаем детальное логгирование в начале
        self._initialize_detailed_logging()
        
        start_time = time.time()
        self.query_logger.info("НАЧАЛО ИНИЦИАЛИЗАЦИИ ЯДРА COGNIFLEX")
        
        try:
            # Обновляем состояние системы
            if self.state_manager:
                self.state_manager.set_state(SystemState.INITIALIZING, "Инициализация компонентов")
            
            # Запускаем мониторинг ресурсов
            if self.resource_manager:
                self.resource_manager.start_monitoring()
                self.query_logger.debug("Мониторинг ресурсов запущен")
            
            # Начало отслеживания метрик
            self.metrics_manager.start_tracking()
            self.query_logger.debug("Отслеживание системных метрик запущено")
            
            # Инициализация компонентов
            self.query_logger.debug("Запуск инициализации компонентов системы...")
            init_start = time.time()
            
            if self.component_initializer:
                if not self.component_initializer.initialize_components():
                    self.query_logger.error("Не удалось инициализировать все компоненты системы")
                    if self.state_manager:
                        self.state_manager.set_state(SystemState.ERROR, "Ошибка инициализации компонентов")
                    self.metrics_manager.record_error("component_initialization_failed")
                    return False
            else:
                self.query_logger.warning("Инициализатор компонентов недоступен, пропускаем инициализацию")
            
            # Явная инициализация MemoryManager
            self.query_logger.debug("Вызов _initialize_memory_manager()...")
            if not self._initialize_memory_manager():
                self.query_logger.warning("Не удалось инициализировать MemoryManager, продолжаем без него")
            
            # Устанавливаем ссылки на компоненты после инициализации
            if 'model_manager' in self.components:
                self.model_manager = self.components['model_manager']
                self.query_logger.debug("model_manager подключен к brain")
                if self.events:
                    self.events.trigger('model_manager_ready', self.model_manager)
            
            if 'text_processor' in self.components:
                self.text_processor = self.components['text_processor']
                self.query_logger.debug("text_processor подключен к brain")
            
            # Обновляем ResponseGenerator с новыми компонентами
            if hasattr(self, 'response_generator') and self.response_generator:
                if self.model_manager:
                    self.response_generator.model_manager = self.model_manager
                if self.text_processor:
                    self.response_generator.text_processor = self.text_processor
                    self.response_generator.token_streamer = self.text_processor
                if hasattr(self.text_processor, 'hybrid_cache'):
                    self.response_generator.hybrid_cache = self.text_processor.hybrid_cache
                self.query_logger.debug("ResponseGenerator обновлен с компонентами")
            
            # Уведомляем о готовности других компонентов
            if self.events:
                for component_name in ['memory_manager', 'text_processor', 'response_generator', 'ethics_framework']:
                    if component_name in self.components:
                        self.events.trigger(f'{component_name}_ready', self.components[component_name])
            
            # Инициализация фрактальной модели из хранилища
            if self.fractal_model_manager:
                self.query_logger.debug("Загрузка фрактальной модели...")
                
                # Check if model_path attribute exists
                if hasattr(self.fractal_model_manager, 'model_path'):
                    if self.fractal_model_manager.model_path:
                        # Проверяем существование директории модели
                        model_dir = self.fractal_model_manager.model_path
                        if not os.path.isabs(model_dir):
                            model_dir = os.path.abspath(model_dir)
                        model_exists = os.path.exists(model_dir)
                        
                        if model_exists:
                            model_files = ['pytorch_model.bin', 'config.json', 'vocab.json']
                            files_found = [f for f in model_files if os.path.exists(os.path.join(model_dir, f))]
                            if len(files_found) >= 2:
                                self.query_logger.debug("  Структура модели корректна")
                            else:
                                self.query_logger.warning(f"  Неполная структура модели, найдены только: {files_found}")
                        else:
                            self.query_logger.warning(f"  Директория модели не существует: {model_dir}")

            try:
                if hasattr(self.fractal_model_manager, 'initialized'):
                    if self.fractal_model_manager.initialized:
                        fractal_init_result = True
                        self.fractal_ready = True
                        if self.events:
                            self.events.trigger('fractal_model_ready', self.fractal_model_manager)
                    else:
                        fractal_init_result = False
                else:
                    fractal_init_result = self.fractal_model_manager.initialize()

                if fractal_init_result and not self.fractal_ready:
                    self.fractal_ready = True
                    self.query_logger.debug("Фрактальная модель успешно загружена и активирована")
                    if self.events:
                        self.events.trigger('fractal_model_ready', self.fractal_model_manager)
                elif not fractal_init_result:
                    self.query_logger.debug("Не удалось загрузить фрактальную модель")
                    self.fractal_ready = False
                
                # Устанавливаем models_ready если фрактальная модель готова или ml_unit готов
                ml_unit_ready = False
                if hasattr(self, 'ml_unit') and self.ml_unit is not None:
                    ml_unit_ready = getattr(self.ml_unit, 'models_ready', False)
                    if not ml_unit_ready:
                        ml_unit_ready = getattr(self.ml_unit, 'initialized', False)
                        if not ml_unit_ready and hasattr(self.ml_unit, 'running'):
                            ml_unit_ready = getattr(self.ml_unit, 'running', False)
                
                if self.fractal_ready or ml_unit_ready:
                    self.models_ready = True
            except Exception as e:
                self.query_logger.debug(f"Исключение при инициализации фрактальной модели: {e}")
                self.fractal_ready = False
            
            # Инициализация координатора генерации - единая точка входа
            try:
                self.generation_coordinator = initialize_generation_coordinator(self)
                self.query_logger.debug("Координатор генерации инициализирован как единая точка входа")
                self.components['generation_coordinator'] = self.generation_coordinator
                
                coordinator_status = self.generation_coordinator.get_status()
                self.query_logger.info(f"Статус координатора: {coordinator_status}")
            except Exception as e:
                self.query_logger.error(f"Ошибка инициализации координатора генерации: {e}", exc_info=True)
                self.generation_coordinator = None
            
            # Получение информации о системе
            system_info = self._get_system_info()
            self.query_logger.info(f"Информация о системе: {system_info}")
            
            # Установка флага инициализации
            self.initialized = True
            
            # Обновляем состояние системы на готовность
            if self.state_manager:
                self.state_manager.set_state(SystemState.READY, "Инициализация завершена успешно")
            
            # Запись статистики инициализации
            total_time = time.time() - start_time
            self.metrics_manager.record_system_startup(total_time)
            self.query_logger.info(f"Ядро CogniFlex успешно инициализировано за {total_time:.4f} сек")
            
            # Выполнение отложенных команд
            self.query_logger.info(f"Выполнение {len(self.deferred_commands)} отложенных команд...")
            for command, args, kwargs in self.deferred_commands:
                try:
                    command(*args, **kwargs)
                    self.query_logger.info(f"Отложенная команда {getattr(command, '__name__', 'lambda')} выполнена успешно.")
                except Exception as e:
                    self.query_logger.error(f"Ошибка выполнения отложенной команды {getattr(command, '__name__', 'lambda')}: {e}", exc_info=True)
            self.deferred_commands.clear()
            self.query_logger.info("Все отложенные команды выполнены.")
            
            # Настраиваем стратегии восстановления модулей
            if self.deferred_system:
                self._setup_module_recovery_strategies()
            
            # Настраиваем умное вытеснение кэша токенов
            self.setup_smart_cache_eviction()
            
            # Start SelfDialogLearningSystem if enabled
            if hasattr(self, 'self_dialog_learning') and self.self_dialog_learning:
                try:
                    self.self_dialog_learning.start()
                    self.query_logger.info("SelfDialogLearningSystem started")
                except Exception as e:
                    self.query_logger.warning(f"Failed to start SelfDialogLearningSystem: {e}")
            
            return True
            
        except Exception as e:
            error_time = time.time() - start_time
            self.query_logger.error(f"Ошибка инициализации ядра за {error_time:.4f} сек: {e}", exc_info=True)
            self.metrics_manager.record_error("core_initialization_failed")
            return False
    
    def _load_brain_config(self) -> Dict[str, Any]:
        """Загружает конфигурацию из brain_config.json."""
        import json
        
        # Ищем файл в нескольких местах
        possible_paths = [
            "brain_config.json",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "brain_config.json"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "brain_config.json"),
        ]
        
        for config_path in possible_paths:
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    logger.info(f"Конфигурация загружена из {config_path}")
                    self.query_logger.info(f"Загружена конфигурация из {config_path}")
                    return config
                except Exception as e:
                    logger.warning(f"Ошибка загрузки {config_path}: {e}")
        
        logger.error("brain_config.json не найден ни в одном из ожидаемых расположений")
        raise FileNotFoundError(
            f"brain_config.json не найден. Проверены пути: {possible_paths}"
        )
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Возвращает информацию о системе для логгирования."""
        system_info = {
            "python_version": sys.version.split()[0],
            "os": os.name,
            "platform": sys.platform,
            "cpu_count": os.cpu_count(),
            "memory": f"{psutil.virtual_memory().percent}%"
        }
        
        if self.resource_manager:
            system_info.update(self.resource_manager.get_system_info())
        
        if self.state_manager:
            system_info["system_state"] = self.state_manager.get_state().value
        
        return system_info
    
    def log_module_activity(self, module_name: str, activity: str, details: Dict[str, Any] = None):
        """Логирует активность модуля"""
        try:
            with self.activity_lock:
                timestamp = time.time()
                
                # Обновляем активность модуля
                if module_name not in self.module_activity:
                    self.module_activity[module_name] = {
                        'first_access': timestamp,
                        'last_access': timestamp,
                        'access_count': 0,
                        'activities': []
                    }
                
                self.module_activity[module_name]['last_access'] = timestamp
                self.module_activity[module_name]['access_count'] += 1
                
                # Добавляем запись в лог
                activity_record = {
                    'timestamp': timestamp,
                    'module': module_name,
                    'activity': activity,
                    'details': details or {}
                }
                self.module_activity[module_name]['activities'].append(activity_record)
                self.module_access_log.append(activity_record)
                
                # Ограничиваем размер лога
                if len(self.module_access_log) > 1000:
                    self.module_access_log = self.module_access_log[-500:]
                
                self.query_logger.debug(f"Активность модуля {module_name}: {activity}")
                
        except Exception as e:
            self.query_logger.error(f"Ошибка логирования активности модуля {module_name}: {e}")
    
    def get_module_activity(self, module_name: str = None) -> Dict[str, Any]:
        """Возвращает информацию об активности модулей"""
        with self.activity_lock:
            if module_name:
                return self.module_activity.get(module_name, {})
            else:
                return {
                    'modules': dict(self.module_activity),
                    'total_accesses': len(self.module_access_log),
                    'active_modules': len([m for m, a in self.module_activity.items() 
                                          if time.time() - a['last_access'] < 300]),  # Активные последние 5 минут
                    'recent_activities': self.module_access_log[-20:] if self.module_access_log else []
                }
    
    def check_module_dependencies(self, module_name: str) -> List[str]:
        """Проверяет зависимости модуля и возвращает список недоступных"""
        # Карта зависимостей модулей
        dependencies = {
            'model_manager': ['ml_unit'],
            'response_generator': ['model_manager', 'text_processor'],
            'training_orchestrator': ['ml_unit', 'learning_manager'],
            'integrated_learning_manager': ['model_manager', 'knowledge_graph'],
            'analytics_manager': ['learning_manager', 'memory_manager'],
            'learning_processor': ['model_manager', 'hybrid_cache']
        }
        
        missing_deps = []
        if module_name in dependencies:
            for dep in dependencies[module_name]:
                if dep not in self.components:
                    missing_deps.append(dep)
        
        return missing_deps
    
    def _log_throttled(self, logger_obj: logging.Logger, level: int, key: str, message: str) -> None:
        """Логирует сообщение не чаще одного раза в self.log_throttle_seconds для указанного ключа."""
        try:
            now = time.time()
            last = self._log_throttle.get(key, 0.0)
            if (now - last) >= float(self.log_throttle_seconds):
                self._log_throttle[key] = now
                logger_obj.log(level, message)
        except Exception:
            logger_obj.log(level, message)
    
    @property
    def knowledge_graph(self):
        """Возвращает knowledge_graph компонент."""
        return self.components.get('knowledge_graph')
    
    @knowledge_graph.setter
    def knowledge_graph(self, value):
        """Устанавливает knowledge_graph компонент."""
        self.query_logger.debug(f"Установка компонента knowledge_graph: {value}")
        self.components['knowledge_graph'] = value
    
    @property
    def qwen_api_enhancer(self):
        """Возвращает QwenAPIEnhancer компонент."""
        return self.components.get('qwen_api_enhancer')
    
    @qwen_api_enhancer.setter
    def qwen_api_enhancer(self, value):
        """Устанавливает QwenAPIEnhancer компонент."""
        self.query_logger.debug(f"Установка компонента qwen_api_enhancer: {value}")
        self.components['qwen_api_enhancer'] = value
    
    def register_component(self, name: str, component: Any) -> bool:
        """Регистрирует компонент в CoreBrain."""
        try:
            self.components[name] = component
            self.query_logger.debug(f"Компонент '{name}' зарегистрирован в CoreBrain")
            return True
        except Exception as e:
            self.query_logger.error(f"Ошибка регистрации компонента '{name}': {e}")
            return False
    
    def get_component(self, name: str) -> Any:
        """Получает компонент по имени."""
        return self.components.get(name)
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Возвращает список доступных моделей из MLUnit или ModelManager."""
        try:
            ml_unit = self.components.get('ml_unit')
            if ml_unit and hasattr(ml_unit, 'get_available_models'):
                return ml_unit.get_available_models()
            
            if self.model_manager and hasattr(self.model_manager, 'get_available_models'):
                return self.model_manager.get_available_models()
            
            return []
        except Exception as e:
            self.query_logger.warning(f"get_available_models: ошибка получения списка моделей: {e}")
            return []
    
    def get_system_health(self) -> Dict[str, Any]:
        """Возвращает информацию о состоянии здоровья системы."""
        try:
            health_status = {
                "status": "healthy",
                "timestamp": time.time(),
                "components": {},
                "warnings": [],
                "errors": [],
                "resources": {}
            }
            
            if self.state_manager:
                state = self.state_manager.get_state()
                health_status["system_state"] = state.value if hasattr(state, 'value') else str(state)
                if str(state) in ["ERROR", "FAILED"]:
                    health_status["status"] = "unhealthy"
                    health_status["errors"].append(f"System state: {state}")
            
            if self.resource_manager:
                try:
                    resource_info = self.resource_manager.get_system_info() or {}
                    memory_percent = resource_info.get("memory_percent", 0)
                    if memory_percent > 90:
                        health_status["warnings"].append(f"High memory usage: {memory_percent}%")
                        health_status["status"] = "degraded"
                    health_status["components"]["resources"] = "ok"
                    health_status["resources"] = resource_info
                except Exception as e:
                    health_status["errors"].append(f"Resource manager error: {e}")
                    health_status["status"] = "degraded"
            else:
                health_status.setdefault("resources", {})
            
            critical_components = ['ml_unit', 'memory_manager']
            for component in critical_components:
                if component in self.components and self.components[component]:
                    health_status["components"][component] = "ok"
                else:
                    health_status["warnings"].append(f"Component {component} not available")
                    if health_status["status"] == "healthy":
                        health_status["status"] = "degraded"
            
            return health_status
        except Exception as e:
            self.query_logger.error(f"Ошибка получения состояния здоровья системы: {e}", exc_info=True)
            return {
                "status": "error",
                "timestamp": time.time(),
                "error": str(e),
                "components": {},
                "warnings": [],
                "errors": [str(e)]
            }
    
    def process_query(self, query: str, user_context: Optional[Dict] = None, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Обрабатывает пользовательский запрос через унифицированный координатор генерации с многоуровневым fallback."""
        start_time = time.time()
        self.query_logger.info(f"Обработка запроса: {query[:50]}...")
        
        # Если передан context, используем его как user_context для обратной совместимости
        if context is not None and user_context is None:
            user_context = context
        elif context is not None and user_context is not None:
            # Объединяем оба контекста, context имеет приоритет
            user_context = {**user_context, **context}
        
        # === QWEN-ONLY MODE: Load Qwen FIRST before any processing ===
        # Check if Qwen-only mode is enabled
        qwen_only_mode = False
        try:
            model_cfg = self.config.get('model', {}) if hasattr(self, 'config') and self.config else {}
            qwen_only_mode = model_cfg.get('qwen_only_mode', False)
        except:
            pass
        
        # Load Qwen if not loaded yet (before greeting check!)
        if qwen_only_mode and self.qwen_model_manager is None and self._qwen_config is not None:
            self.query_logger.info("Qwen-only mode: Загрузка QwenModelManager...")
            try:
                from cogniflex.mlearning.qwen_model_manager import get_qwen_model_manager
                qwen_device = self._qwen_config.get('device', 'cuda')
                
                self.qwen_model_manager = get_qwen_model_manager(
                    model_size=self._qwen_config.get('name', 'qwen3.5-0.8b'),
                    device=qwen_device,
                    load_in_8bit=True,
                    load_in_4bit=False
                )
                
                if self.qwen_model_manager and self.qwen_model_manager.initialized:
                    self.qwen_ready = True
                    self.query_logger.info("QwenModelManager загружен для обработки запроса")
                else:
                    self.query_logger.error("QwenModelManager НЕ загружен - ошибка конфигурации")
            except Exception as e:
                self.query_logger.error(f"Ошибка загрузки Qwen: {e}")
        
        # === В Qwen-only режиме НЕ используем greeting handler ===
        # Все запросы (включая приветствия) идут через Qwen
        if not qwen_only_mode:
            # БЫСТРЫЙ ОТВЕТ НА ПРИВЕТСТВИЯ - только если НЕ Qwen-only режим
            query_lower = query.lower().strip()
            greeting_keywords = ['привет', 'здравствуй', 'добрый', 'hello', 'hi', 'хай', 'здорово', 'прив']
            if any(g in query_lower for g in greeting_keywords):
                return {
                    "response": "Привет! Я CogniFlex, рад общению. Что хотите обсудить?",
                    "text": "Привет! Я CogniFlex, рад общению. Что хотите обсудить?",
                    "status": "ok",
                    "confidence": 1.0,
                    "source": "greeting_handler",
                    "fallback_level": 0,
                    "processing_time": time.time() - start_time
                }
        
        # === Qwen-only mode: Force Qwen usage ===
        if qwen_only_mode:
            # If Qwen not ready, return error
            if not self.qwen_model_manager or not self.qwen_model_manager.initialized:
                return {
                    "response": "Ошибка: Qwen модель недоступна. Проверьте конфигурацию.",
                    "text": "Ошибка: Qwen модель недоступна. Проверьте конфигурацию.",
                    "status": "error",
                    "confidence": 0.0,
                    "source": "qwen_error",
                    "error": "Qwen model not initialized in qwen_only_mode",
                    "processing_time": time.time() - start_time
                }
            
            # Use Qwen for ALL queries in qwen_only_mode
            self.query_logger.info("Используем QwenModelManager (qwen_only_mode)")
            
            # Get generation params from config
            gen_config = self.config.get('generation', {})
            temperature = gen_config.get('temperature', 0.7)
            top_p = gen_config.get('top_p', 0.9)
            repetition_penalty = gen_config.get('repetition_penalty', 1.1)
            
            messages = [{"role": "user", "content": query}]
            response_text = self.qwen_model_manager.generate(
                messages,
                max_new_tokens=2048,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
            
            if response_text and not response_text.startswith("Ошибка"):
                return {
                    "response": response_text,
                    "text": response_text,
                    "status": "ok",
                    "confidence": 0.9,
                    "source": "qwen_model",
                    "fallback_level": 0,
                    "processing_time": time.time() - start_time
                }
            else:
                return {
                    "response": f"Ошибка генерации: {response_text or 'пустой ответ'}",
                    "text": f"Ошибка генерации: {response_text or 'пустой ответ'}",
                    "status": "error",
                    "confidence": 0.0,
                    "source": "qwen_error",
                    "processing_time": time.time() - start_time
                }
        
        # === NON-QWEN ONLY MODE: Legacy fallback chain ===
        # Only executed if qwen_only_mode is False
        # Уровень 0: QwenModelManager (приоритетная модель для диалогов)
        try:
            # Lazy loading - загружаем модель только при первом запросе
            if self.qwen_model_manager is None and self._qwen_config is not None:
                self.query_logger.info("Загрузка QwenModelManager (lazy)...")
                try:
                    try:
                        from cogniflex.mlearning.qwen_model_manager import get_qwen_model_manager
                    except ImportError:
                        from ..mlearning.qwen_model_manager import get_qwen_model_manager
                    
                    qwen_device = self._qwen_config.get('device', 'cuda')
                    
                    self.qwen_model_manager = get_qwen_model_manager(
                        model_size=self._qwen_config.get('name', 'qwen3.5-0.8b'),
                        device=qwen_device,
                        load_in_8bit=True,
                        load_in_4bit=False
                    )
                    
                    if self.qwen_model_manager.initialized:
                        self.qwen_ready = True
                        if self.events:
                            self.events.trigger('qwen_model_ready', self.qwen_model_manager)
                        self.query_logger.info("QwenModelManager успешно загружен!")
                    else:
                        self.qwen_model_manager = None
                        self.query_logger.warning("QwenModelManager не инициализирован")
                except Exception as e:
                    self.query_logger.warning(f"Ошибка lazy загрузки QwenModelManager: {e}")
                    self.qwen_model_manager = None
            
            if self.qwen_model_manager and self.qwen_model_manager.initialized:
                self.query_logger.info("Используем QwenModelManager для генерации")
                
                # Get generation params from config
                gen_config = self.config.get('generation', {})
                temperature = gen_config.get('temperature', 0.7)
                top_p = gen_config.get('top_p', 0.9)
                repetition_penalty = gen_config.get('repetition_penalty', 1.1)
                
                # Используем chat format для Qwen
                messages = [{"role": "user", "content": query}]
                response_text = self.qwen_model_manager.generate(
                    messages,
                    max_new_tokens=2048,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty
                )
                
                if response_text and not response_text.startswith("Ошибка"):
                    return {
                        "response": response_text,
                        "text": response_text,
                        "status": "ok",
                        "confidence": 0.9,
                        "source": "qwen_model",
                        "fallback_level": 0,
                        "processing_time": time.time() - start_time
                    }
        except Exception as e:
            self.query_logger.warning(f"QwenModelManager недоступен: {e}")
        
        # Проверяем наличие reasoning_engine и используем его для генерации с рассуждением
        if hasattr(self, 'reasoning_engine') and self.reasoning_engine:
            try:
                self.query_logger.info("Используем ReasoningEngine для генерации с рассуждением")
                reasoning_result = self.reasoning_engine.reason(query, user_context)
                
                if reasoning_result.get('answer'):
                    response_dict = {
                        "response": reasoning_result.get('answer', ''),
                        "text": reasoning_result.get('answer', ''),
                        "status": "ok",
                        "confidence": reasoning_result.get('confidence', 0.0),
                        "reasoning": reasoning_result,
                        "source": "reasoning_engine",
                        "fallback_level": 0,
                        "processing_time": time.time() - start_time
                    }
                    self.query_logger.info("Успешно использован reasoning_engine")
                    return response_dict
            except Exception as e:
                self.query_logger.warning(f"Reasoning engine недоступен: {e}")
        
        # Уровень 1: Generation Coordinator
        try:
            if self.generation_coordinator:
                response = self.generation_coordinator.generate_response(
                    prompt=query,
                    max_new_tokens=2048
                )
                if isinstance(response, dict):
                    response_dict = response
                elif hasattr(response, 'to_dict'):
                    response_dict = response.to_dict()
                else:
                    response_dict = {"generated_text": str(response), "status": "success"}
                response_dict["fallback_level"] = 1
                response_dict["source"] = "generation_coordinator"
                self.query_logger.info("Успешно использован generation_coordinator")
                return response_dict
        except Exception as e:
            self.query_logger.warning(f"Generation coordinator недоступен: {e}")
        
        # Уровень 2: Fractal Model Manager
        try:
            if hasattr(self, 'fractal_model_manager') and self.fractal_model_manager:
                response = self.fractal_model_manager.generate(query)
                if response:
                    if isinstance(response, dict):
                        response_dict = response
                    elif hasattr(response, 'to_dict'):
                        response_dict = response.to_dict()
                    else:
                        response_dict = {"generated_text": str(response), "status": "success"}
                    response_dict["fallback_level"] = 2
                    response_dict["source"] = "fractal_model_manager"
                    self.query_logger.info("Успешно использован fractal_model_manager")
                    return response_dict
        except Exception as e:
            self.query_logger.warning(f"Fractal model manager недоступен: {e}")
        
        # Уровень 3: Query Processor
        try:
            if 'query_processor' in self.components and self.components['query_processor']:
                resp = self.components['query_processor'].process_query(query, user_context)
                if isinstance(resp, dict) and 'status' not in resp:
                    status_val = 'error' if resp.get('error') else 'ok'
                    try:
                        resp['status'] = status_val
                    except Exception:
                        resp = {"response": str(resp), "status": status_val}
                
                resp["fallback_level"] = 3
                resp["source"] = "query_processor"
                self.query_logger.info("Успешно использован query_processor")
                return resp
        except Exception as e:
            self.query_logger.warning(f"Query processor недоступен: {e}")
        
        # Уровень 4: MLUnit напрямую
        try:
            if hasattr(self, 'ml_unit') and self.ml_unit:
                response = self.ml_unit.generate_response(query)
                if response:
                    response["fallback_level"] = 4
                    response["source"] = "ml_unit_direct"
                    self.query_logger.info("Успешно использован MLUnit напрямую")
                    return response
        except Exception as e:
            self.query_logger.warning(f"MLUnit недоступен: {e}")
        
        # Уровень 5: Memory Manager для простых ответов
        try:
            if hasattr(self, 'memory_manager') and self.memory_manager:
                # Пытаемся найти похожий запрос в памяти
                memory_response = self.memory_manager.search_similar(query, top_k=1)
                if memory_response and len(memory_response) > 0:
                    similar_item = memory_response[0]
                    if hasattr(similar_item, 'response') or (isinstance(similar_item, dict) and 'response' in similar_item):
                        response_text = similar_item.response if hasattr(similar_item, 'response') else similar_item.get('response', '')
                        if response_text:
                            response = {
                                "response": response_text,
                                "confidence": 0.6,
                                "fallback_level": 5,
                                "source": "memory_manager",
                                "similarity_score": getattr(similar_item, 'similarity', 0.0),
                                "timestamp": time.time()
                            }
                            self.query_logger.info("Успешно использован memory_manager")
                            return response
        except Exception as e:
            self.query_logger.warning(f"Memory manager недоступен: {e}")
        
        # Уровень 6: Базовый ответ на основе ключевых слов
        try:
            response = self._generate_basic_fallback_response(query)
            response["fallback_level"] = 6
            response["source"] = "basic_fallback"
            self.query_logger.warning("Использован базовый fallback-ответ")
            return response
        except Exception as e:
            self.query_logger.error(f"Ошибка в базовом fallback: {e}")
        
        # Финальный fallback если все уровни провалились
        processing_time = time.time() - start_time
        self.query_logger.error(f"Все уровни fallback провалились для запроса: {query[:50]}...")
        return {
            "response": "Извините, система временно недоступна. Пожалуйста, попробуйте переформулировать запрос или обратиться позже.",
            "status": "error",
            "fallback_level": 7,
            "source": "final_fallback",
            "error": "All fallback levels failed",
            "processing_time": processing_time,
            "timestamp": time.time(),
            "metadata": {
                "original_query_length": len(query),
                "system_status": "critical_degradation"
            }
        }
    
    def _generate_basic_fallback_response(self, query: str) -> Dict[str, Any]:
        """Генерирует базовый ответ на основе анализа ключевых слов."""
        query_lower = query.lower()
        
        # Базовые паттерны ответов
        if any(word in query_lower for word in ['привет', 'здравствуй', 'hello', 'hi']):
            response_text = "Здравствуйте! Я система CogniFlex. К сожалению, мои основные компоненты временно недоступны, но я рада вам помочь в рамках своих ограниченных возможностей."
        elif any(word in query_lower for word in ['как дела', 'how are you', 'что нового']):
            response_text = "Спасибо за интерес! Система работает в ограниченном режиме из-за технических трудностей. Я стараюсь помочь в рамках доступных возможностей."
        elif any(word in query_lower for word in ['помощь', 'help', 'помоги']):
            response_text = "Я готова помочь, но мои возможности сейчас ограничены. Попробуйте переформулировать запрос или обратитесь позже, когда система восстановится."
        elif any(word in query_lower for word in ['спасибо', 'thank', 'благодарю']):
            response_text = "Всегда пожалуйста! Рада была помочь, несмотря на временные ограничения системы."
        elif '?' in query or any(word in query_lower for word in ['что', 'где', 'когда', 'почему', 'как']):
            response_text = "Интересный вопрос! К сожалению, из-за временных технических трудностей я не могу дать полный ответ. Попробуйте обратиться позже, когда система восстановится."
        else:
            response_text = "Я получила ваш запрос, но из-за временных ограничений системы не могу обработать его в полной мере. Попробуйте позже или переформулируйте запрос."
        
        return {
            "response": response_text,
            "confidence": 0.2,
            "status": "limited",
            "timestamp": time.time(),
            "metadata": {
                "fallback_type": "keyword_based",
                "query_category": self._categorize_query(query_lower)
            }
        }
    
    def _categorize_query(self, query_lower: str) -> str:
        """Категоризирует запрос по ключевым словам."""
        categories = {
            'greeting': ['привет', 'здравствуй', 'hello', 'hi', 'добрый'],
            'question': ['что', 'где', 'когда', 'почему', 'как', '?'],
            'help': ['помощь', 'help', 'помоги', 'подскажи'],
            'gratitude': ['спасибо', 'thank', 'благодарю', 'благодарю'],
            'farewell': ['пока', 'до свидания', 'goodbye', 'bye'],
            'system': ['система', 'работа', 'статус', 'состояние']
        }
        
        for category, keywords in categories.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        
        return 'general'
    
    def start(self) -> bool:
        """Запускает все компоненты системы."""
        if not self.initialized:
            self.query_logger.error("Невозможно запустить неинициализированное ядро")
            return False
        
        start_time = time.time()
        self.query_logger.info("Запуск ядра CogniFlex...")
        
        try:
            components_started = 0
            components_skipped = 0
            components_failed = 0
            
            for name, component in self.components.items():
                if hasattr(component, 'start'):
                    try:
                        component_start = time.time()
                        if name == 'neuromorphic_simulator' and hasattr(component, 'use_nest') and getattr(component, 'use_nest', False):
                            self.query_logger.info("Пропуск автозапуска нейроморфного симулятора (NEST-режим)")
                            components_skipped += 1
                            continue
                        
                        # Check if component has BaseComponent state management
                        if hasattr(component, 'get_state'):
                            from cogniflex.core.base_component import ComponentState
                            state = component.get_state()
                            if state == ComponentState.RUNNING:
                                self.query_logger.debug(f"Компонент {name} уже запущен")
                                components_started += 1
                                continue
                            elif state == ComponentState.STARTING:
                                self.query_logger.debug(f"Компонент {name} уже запускается")
                                components_started += 1
                                continue
                            elif state not in [ComponentState.READY, ComponentState.UNINITIALIZED]:
                                self.query_logger.warning(f"Компонент {name} не готов к запуску (состояние: {state})")
                                components_failed += 1
                                continue
                        
                        result = component.start()
                        component_time = time.time() - component_start
                        if result is not False:
                            self.query_logger.info(f"Компонент {name} запущен за {component_time:.4f} сек")
                            components_started += 1
                        else:
                            self.query_logger.warning(f"Компонент {name} отклонил запуск")
                            components_failed += 1
                    except Exception as e:
                        self.query_logger.warning(f"Ошибка при запуске компонента {name}: {e}", exc_info=True)
                        self.metrics_manager.record_error(f"component_{name}_start_failed")
                        components_failed += 1
                else:
                    # Components without start() method are considered "started"
                    components_skipped += 1
                    self.query_logger.debug(f"Компонент {name} не имеет метода start()")
            
            total_components = len(self.components)
            self.query_logger.info(f"Итоги запуска: {components_started}/{total_components} запущено, {components_skipped} пропущено, {components_failed} неудачно")
            
            if components_started < total_components * 0.5:
                self.query_logger.warning(f"ВНИМАНИЕ: Запущено только {components_started}/{total_components} компонентов")
                self.metrics_manager.record_warning("insufficient_components_started")
            
            self.running = True
            
            # Проверяем и запускаем GUI если доступен
            self._start_gui_if_available()
            
            try:
                if getattr(self, 'background', None):
                    self.background.start()
                    self.query_logger.info("BackgroundCoordinator запущен")
            except Exception as e:
                self.query_logger.warning(f"Не удалось запустить BackgroundCoordinator: {e}")
            
            total_time = time.time() - start_time
            self.query_logger.info(f"Ядро CogniFlex успешно запущено за {total_time:.4f} сек")
            return True
        except Exception as e:
            self.query_logger.error(f"Ошибка запуска ядра: {e}", exc_info=True)
            self.metrics_manager.record_error("core_start_failed")
            if self.state_manager:
                self.state_manager.set_state(SystemState.ERROR, f"Ошибка запуска: {e}")
            return False
    
    def _start_gui_if_available(self) -> None:
        """Запускает GUI если доступен и система полностью инициализирована."""
        try:
            # Проверяем наличие GUI компонентов
            if hasattr(self, 'components') and 'gui' in self.components:
                gui_component = self.components['gui']
                if hasattr(gui_component, 'start_gui'):
                    self.query_logger.info("Запуск GUI после полной инициализации системы...")
                    gui_component.start_gui()
                else:
                    self.query_logger.debug("GUI компонент найден, но не имеет метода start_gui")
            else:
                self.query_logger.debug("GUI компонент не найден в системе")
        except Exception as e:
            self.query_logger.warning(f"Ошибка при запуске GUI: {e}")
    
    def _check_system_ready_for_training(self) -> bool:
        """Проверяет готовность системы к запуску обучения."""
        try:
            # Система должна быть полностью инициализирована и запущена
            if not (self.initialized and self.running):
                return False
            
            # Ключевые компоненты должны быть здоровы
            required_components = ['model_manager', 'ml_unit', 'memory_manager', 'text_processor']
            for comp_name in required_components:
                if comp_name in self.components:
                    comp = self.components[comp_name]
                    if hasattr(comp, 'health_check'):
                        health = comp.health_check()
                        if isinstance(health, dict):
                            if not health.get('healthy', False):
                                self.query_logger.warning(f"Компонент {comp_name} не здоров: {health}")
                                return False
                        else:
                            if not health:
                                self.query_logger.warning(f"Компонент {comp_name} не здоров")
                                return False
                    else:
                        self.query_logger.debug(f"Компонент {comp_name} не имеет метода health_check")
                else:
                    self.query_logger.warning(f"Обязательный компонент {comp_name} не найден, но продолжаем проверку...")
            
            # Проверяем доступность ресурсов для обучения
            if hasattr(self, 'resource_manager'):
                try:
                    cpu_usage = float(self.resource_manager.get_cpu_usage())
                    mem_usage = float(self.resource_manager.get_memory_usage())
                    
                    # Обучение запускается только если загрузка CPU < 90% и RAM < 95%
                    if cpu_usage > 0.9 or mem_usage > 0.95:
                        self.query_logger.info(f"Система не готова к обучению: CPU={cpu_usage:.1%}, RAM={mem_usage:.1%}")
                        return False
                except Exception:
                    pass
            
            return True
        except Exception as e:
            self.query_logger.error(f"Ошибка проверки готовности к обучению: {e}")
            return False
    
    def stop(self):
        """Останавливает все компоненты системы."""
        with self._shutdown_lock:
            if self._shutting_down:
                self.query_logger.debug("Система уже завершает работу")
                return
            if not self.running:
                self.query_logger.debug("Попытка остановки уже остановленной системы")
                return
            
            self._shutting_down = True
        
        stop_time = time.time()
        self.query_logger.info("Остановка ядра CogniFlex...")
        
        try:
            try:
                if getattr(self, 'background', None):
                    self.background.stop()
                    self.query_logger.info("BackgroundCoordinator остановлен")
            except Exception as e:
                self.query_logger.warning(f"Ошибка остановки BackgroundCoordinator: {e}")
            
            if self.state_manager:
                self.state_manager.set_state(SystemState.SHUTTING_DOWN, "Начало остановки системы")
            
            if self.resource_manager:
                self.resource_manager.stop_monitoring()
            
            for name, component in self.components.items():
                try:
                    if hasattr(component, 'stop'):
                        component.stop()
                        self.query_logger.debug(f"Компонент {name} остановлен")
                except Exception as e:
                    self.query_logger.error(f"Ошибка остановки компонента {name}: {e}")
            
            self.running = False
            
            if self.state_manager:
                self.state_manager.set_state(SystemState.OFFLINE, "Система остановлена")
            
            total_time = time.time() - stop_time
            self.metrics_manager.record_system_shutdown(total_time)
            self.query_logger.info(f"Ядро CogniFlex остановлено за {total_time:.4f} сек")
        except Exception as e:
            self.query_logger.error(f"Ошибка остановки ядра: {e}", exc_info=True)
            self.metrics_manager.record_error("core_stop_failed")
            if self.state_manager:
                self.state_manager.record_error(e, "core_brain")
    
    def start_background_services(self) -> None:
        """Явный запуск автопилота (если требуется вне start())."""
        try:
            if getattr(self, 'background', None):
                self.background.start()
        except Exception as e:
            self.query_logger.warning(f"start_background_services: {e}")
    
    def stop_background_services(self) -> None:
        """Явная остановка автопилота (если требуется вне stop())."""
        try:
            if getattr(self, 'background', None):
                self.background.stop()
        except Exception as e:
            self.query_logger.warning(f"stop_background_services: {e}")
    
    def signal_user_activity(self) -> None:
        """Сигнал активности пользователя для троттлинга фоновых задач."""
        try:
            if getattr(self, 'background', None):
                self.background.signal_user_activity()
        except Exception:
            pass
    
    def reboot(self) -> bool:
        """Перезагружает ядро: безопасно останавливает, заново инициализирует и запускает систему."""
        self._log_throttled(self.query_logger, logging.INFO, "core_reboot_request", "Запрос перезагрузки ядра")
        
        try:
            if getattr(self, 'running', False):
                self.query_logger.info("Остановка системы перед перезагрузкой...")
                try:
                    self.stop()
                except Exception as e:
                    self.query_logger.error(f"Ошибка при остановке перед перезагрузкой: {e}", exc_info=True)
            
            self.initialized = False
            self.query_logger.info("Повторная инициализация ядра...")
            
            if not self.initialize():
                self.query_logger.error("Перезагрузка прервана: ошибка повторной инициализации")
                return False
            
            self.query_logger.info("Запуск ядра после перезагрузки...")
            
            if not self.start():
                self.query_logger.error("Перезагрузка прервана: ошибка запуска после инициализации")
                return False
            
            self._log_throttled(self.query_logger, logging.INFO, "core_reboot_success", "Перезагрузка ядра успешно завершена")
            return True
        except Exception as e:
            self.query_logger.error(f"Ошибка перезагрузки ядра: {e}", exc_info=True)
            return False
    
    def setup_smart_cache_eviction(self):
        """Настраивает умное вытеснение кэша токенов"""
        try:
            if not hasattr(self, 'token_cache') or not self.token_cache:
                self.query_logger.warning("Token кэш недоступен для умного вытеснения")
                return
            
            # Подписываемся на события памяти
            if self.events:
                self.events.subscribe('memory_pressure', self._handle_memory_pressure)
                self.events.subscribe('cache_eviction_needed', self._handle_cache_eviction)
                self.query_logger.info("Подписались на события управления кэшем")
            
            # Настраиваем мониторинг кэша
            self._setup_cache_monitoring()
            self.query_logger.info("Умное вытеснение кэша настроено")
            
        except Exception as e:
            self.query_logger.error(f"Ошибка настройки умного вытеснения: {e}")
    
    def _setup_cache_monitoring(self):
        """Настраивает мониторинг состояния кэша"""
        try:
            if self.background:
                # Создаем простой детектор давления памяти
                class MemoryPressureDetector:
                    def __init__(self, callback):
                        self.callback = callback
                    
                    def probe(self, context):
                        """Пробует состояние памяти и возвращает задачи при необходимости"""
                        try:
                            import psutil
                            memory = psutil.virtual_memory()
                            memory_percent = memory.percent / 100.0
                            
                            # Проверяем VRAM если доступно
                            vram_pressure = 0.0
                            try:
                                import torch
                                if torch.cuda.is_available():
                                    vram_used = torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory
                                    vram_pressure = vram_used
                            except ImportError:
                                pass
                            
                            # Если давление памяти высокое, возвращаем задачу вытеснения
                            if memory_percent > 0.85 or vram_pressure > 0.9:
                                return [{
                                    'type': 'cache_eviction',
                                    'priority': 'high',
                                    'data': {
                                        'source': 'ram' if memory_percent > 0.85 else 'vram',
                                        'memory_percent': memory_percent,
                                        'vram_pressure': vram_pressure
                                    }
                                }]
                            return []
                        except Exception:
                            return []
                
                # Создаем и регистрируем детектор
                memory_detector = MemoryPressureDetector(self._check_memory_pressure)
                self.background.register_detector(memory_detector)
                self.query_logger.info("Детектор давления памяти зарегистрирован")
        except Exception as e:
            self.query_logger.warning(f"Ошибка настройки мониторинга кэша: {e}")
    
    def _check_memory_pressure(self):
        """Проверяет давление памяти и инициирует вытеснение"""
        try:
            if not hasattr(self, 'token_cache') or not self.token_cache:
                return
            
            # Получаем статистику кэша
            cache_stats = self.token_cache.get_cache_stats()
            
            # Проверяем использование RAM
            memory = psutil.virtual_memory()
            memory_percent = memory.percent / 100.0
            
            # Проверяем VRAM если доступно
            vram_pressure = 0.0
            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory
                vram_pressure = vram_used
            
            # Определяем необходимость вытеснения
            needs_eviction = False
            eviction_source = None
            
            if memory_percent > 0.85:  # 85% RAM
                needs_eviction = True
                eviction_source = 'ram'
            elif vram_pressure > 0.9:  # 90% VRAM
                needs_eviction = True
                eviction_source = 'vram'
            
            if needs_eviction:
                self.query_logger.info(f"Обнаружено давление памяти: {eviction_source}={memory_percent:.2f}, VRAM={vram_pressure:.2f}")
                
                # Публикуем событие
                if self.events:
                    self.events.trigger('memory_pressure', {
                        'source': eviction_source,
                        'memory_percent': memory_percent,
                        'vram_pressure': vram_pressure,
                        'cache_stats': cache_stats
                    })
                
                # Выполняем умное вытеснение
                self._perform_smart_eviction(eviction_source, memory_percent, vram_pressure)
        
        except Exception as e:
            self.query_logger.error(f"Ошибка проверки давления памяти: {e}")
    
    def _handle_memory_pressure(self, event_data):
        """Обрабатывает событие давления памяти"""
        try:
            source = event_data.get('source', 'unknown')
            memory_percent = event_data.get('memory_percent', 0.0)
            vram_pressure = event_data.get('vram_pressure', 0.0)
            
            self.query_logger.info(f"Обработка давления памяти из {source}: RAM={memory_percent:.2f}, VRAM={vram_pressure:.2f}")
            
            # Выполняем вытеснение с учетом источника
            self._perform_smart_eviction(source, memory_percent, vram_pressure)
            
        except Exception as e:
            self.query_logger.error(f"Ошибка обработки давления памяти: {e}")
    
    def _handle_cache_eviction(self, event_data):
        """Обрабатывает событие необходимости вытеснения кэша"""
        try:
            eviction_type = event_data.get('type', 'lru')
            target_tokens = event_data.get('target_tokens', 100)
            
            self.query_logger.info(f"Выполнение вытеснения кэша: {eviction_type}, токенов: {target_tokens}")
            
            if eviction_type == 'smart':
                self._perform_smart_eviction('system', 0.9, 0.9)
            else:
                self._perform_basic_eviction(target_tokens)
                
        except Exception as e:
            self.query_logger.error(f"Ошибка обработки вытеснения кэша: {e}")
    
    def _perform_smart_eviction(self, source, memory_percent, vram_pressure):
        """Выполняет умное вытеснение с учетом приоритетов и метаданных"""
        try:
            if not hasattr(self, 'token_cache') or not self.token_cache:
                return
            
            # Определяем стратегию вытеснения
            if source == 'vram':
                # Вытеснение из VRAM в RAM
                self._evict_vram_to_ram()
            elif source == 'ram':
                # Вытеснение из RAM в SSD
                self._evict_ram_to_ssd()
            else:
                # Комплексное вытеснение
                if vram_pressure > 0.8:
                    self._evict_vram_to_ram()
                if memory_percent > 0.8:
                    self._evict_ram_to_ssd()
            
            # Обновляем статистику
            self._update_cache_metrics()
            
        except Exception as e:
            self.query_logger.error(f"Ошибка умного вытеснения: {e}")
    
    def _evict_vram_to_ram(self):
        """Вытесняет токены из VRAM в RAM"""
        try:
            if not hasattr(self.token_cache, 'vram_cache'):
                return
            
            vram_cache = self.token_cache.vram_cache
            ram_cache = self.token_cache.ram_cache
            
            # Получаем список токенов в VRAM с метаданными
            tokens_to_evict = []
            for token_id, token_data in vram_cache.items():
                # Получаем метаданные токена
                metadata = self.token_cache.token_metadata.get(token_id, {})
                last_access = metadata.get('last_access', 0)
                access_count = metadata.get('access_count', 0)
                relevance = metadata.get('relevance_score', 0.5)
                
                tokens_to_evict.append({
                    'id': token_id,
                    'data': token_data,
                    'last_access': last_access,
                    'access_count': access_count,
                    'relevance': relevance
                })
            
            # Сортируем по релевантности и времени доступа
            tokens_to_evict.sort(key=lambda x: (x['relevance'], x['last_access']))
            
            # Вытесняем 25% токенов из VRAM
            evict_count = max(1, len(tokens_to_evict) // 4)
            evicted = 0
            
            for token_info in tokens_to_evict[:evict_count]:
                token_id = token_info['id']
                token_data = token_info['data']
                
                # Перемещаем в RAM
                ram_cache.put(token_id, token_data)
                vram_cache.pop(token_id, None)
                
                # Обновляем метаданные
                self.token_cache.token_metadata[token_id] = {
                    **self.token_cache.token_metadata.get(token_id, {}),
                    'location': 'ram',
                    'evicted_from': 'vram',
                    'eviction_time': time.time()
                }
                
                evicted += 1
            
            self.query_logger.info(f"Вытеснено {evicted} токенов из VRAM в RAM")
            
            # Публикуем событие
            if self.events:
                self.events.trigger('vram_to_ram_eviction', {
                    'evicted_count': evicted,
                    'timestamp': time.time()
                })
                
        except Exception as e:
            self.query_logger.error(f"Ошибка вытеснения VRAM->RAM: {e}")
    
    def _evict_ram_to_ssd(self):
        """Вытесняет токены из RAM в SSD"""
        try:
            if not hasattr(self.token_cache, 'ram_cache') or not hasattr(self.token_cache, 'disk_cache'):
                return
            
            ram_cache = self.token_cache.ram_cache
            disk_cache = self.token_cache.disk_cache
            
            # Получаем список токенов в RAM
            tokens_to_evict = []
            for token_id, token_data in ram_cache.items():
                # Получаем метаданные
                metadata = self.token_cache.token_metadata.get(token_id, {})
                last_access = metadata.get('last_access', 0)
                access_count = metadata.get('access_count', 0)
                relevance = metadata.get('relevance_score', 0.5)
                
                tokens_to_evict.append({
                    'id': token_id,
                    'data': token_data,
                    'last_access': last_access,
                    'access_count': access_count,
                    'relevance': relevance
                })
            
            # Сортируем по релевантности и времени доступа
            tokens_to_evict.sort(key=lambda x: (x['relevance'], x['last_access']))
            
            # Вытесняем 30% токенов из RAM в SSD
            evict_count = max(1, len(tokens_to_evict) * 3 // 10)
            evicted = 0
            
            for token_info in tokens_to_evict[:evict_count]:
                token_id = token_info['id']
                token_data = token_info['data']
                
                # Сохраняем на SSD
                if disk_cache.save_token(token_id, token_data):
                    ram_cache.pop(token_id, None)
                    
                    # Обновляем метаданные
                    self.token_cache.token_metadata[token_id] = {
                        **self.token_cache.token_metadata.get(token_id, {}),
                        'location': 'ssd',
                        'evicted_from': 'ram',
                        'eviction_time': time.time()
                    }
                    
                    evicted += 1
            
            self.query_logger.info(f"Вытеснено {evicted} токенов из RAM в SSD")
            
            # Публикуем событие
            if self.events:
                self.events.trigger('ram_to_ssd_eviction', {
                    'evicted_count': evicted,
                    'timestamp': time.time()
                })
                
        except Exception as e:
            self.query_logger.error(f"Ошибка вытеснения RAM->SSD: {e}")
    
    def _perform_basic_eviction(self, target_tokens):
        """Выполняет базовое вытеснение LRU"""
        try:
            if hasattr(self, 'token_cache') and self.token_cache and hasattr(self.token_cache, '_evict_one_lru'):
                evicted = 0
                for _ in range(min(target_tokens, 100)):  # Ограничиваем 100 токенов за раз
                    self.token_cache._evict_one_lru()
                    evicted += 1
                
                self.query_logger.info(f"Выполнено базовое вытеснение: {evicted} токенов")
                
        except Exception as e:
            self.query_logger.error(f"Ошибка базового вытеснения: {e}")
    
    def _update_cache_metrics(self):
        """Обновляет метрики кэша"""
        try:
            if hasattr(self, 'metrics_manager') and self.metrics_manager:
                if hasattr(self.token_cache, 'get_cache_stats'):
                    stats = self.token_cache.get_cache_stats()
                    
                    # Записываем метрики
                    self.metrics_manager.record_query_metrics(
                        cache_vram_hits=stats.get('vram_hits', 0),
                        cache_ram_hits=stats.get('ram_hits', 0),
                        cache_disk_hits=stats.get('disk_hits', 0),
                        cache_evictions=stats.get('evictions', 0),
                        cache_efficiency=stats.get('cache_efficiency', 0.0)
                    )
                    
        except Exception as e:
            self.query_logger.debug(f"Ошибка обновления метрик кэша: {e}")
    
    def get_cache_health_status(self) -> Dict[str, Any]:
        """Возвращает детальный статус здоровья кэша"""
        try:
            if not hasattr(self, 'token_cache') or not self.token_cache:
                return {'status': 'unavailable', 'message': 'Token кэш недоступен'}
            
            stats = self.token_cache.get_cache_stats()
            memory = psutil.virtual_memory()
            
            # Рассчитываем здоровье кэша
            total_requests = stats.get('total_requests', 0)
            total_hits = stats.get('vram_hits', 0) + stats.get('ram_hits', 0) + stats.get('disk_hits', 0)
            hit_rate = total_hits / max(1, total_requests)
            
            # Определяем статус
            if hit_rate > 0.8:
                status = 'excellent'
            elif hit_rate > 0.6:
                status = 'good'
            elif hit_rate > 0.4:
                status = 'fair'
            else:
                status = 'poor'
            
            return {
                'status': status,
                'hit_rate': hit_rate,
                'memory_usage': memory.percent,
                'cache_stats': stats,
                'recommendations': self._get_cache_recommendations(stats, memory.percent)
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _get_cache_recommendations(self, stats: Dict, memory_percent: float) -> List[str]:
        """Генерирует рекомендации по оптимизации кэша"""
        recommendations = []
        
        try:
            # Проверяем hit rate
            total_requests = stats.get('total_requests', 0)
            total_hits = stats.get('vram_hits', 0) + stats.get('ram_hits', 0) + stats.get('disk_hits', 0)
            hit_rate = total_hits / max(1, total_requests)
            
            if hit_rate < 0.5:
                recommendations.append("Низкий hit rate кэша. Рассмотрите увеличение размера кэша.")
            
            # Проверяем использование памяти
            if memory_percent > 85:
                recommendations.append("Высокое использование памяти. Рекомендуется агрессивное вытеснение.")
            
            # Проверяем баланс кэшей
            vram_hits = stats.get('vram_hits', 0)
            ram_hits = stats.get('ram_hits', 0)
            disk_hits = stats.get('disk_hits', 0)
            
            if vram_hits == 0 and torch.cuda.is_available():
                recommendations.append("VRAM кэш не используется. Проверьте настройки GPU.")
            
            if disk_hits > ram_hits * 2:
                recommendations.append("Частое обращение к SSD. Рассмотрите увеличение RAM кэша.")
            
        except Exception as e:
            recommendations.append(f"Ошибка анализа рекомендаций: {e}")
        
        return recommendations
    
    def _setup_module_recovery_strategies(self):
        """Регистрирует проверки здоровья и стратегии восстановления модулей в DeferredCommandSystem."""
        if not getattr(self, 'deferred_system', None):
            self.query_logger.debug("DeferredCommandSystem недоступна, пропускаем настройку восстановления модулей")
            return
        
        ds = self.deferred_system
        
        def register(module_key: str, component: Any, init_method_name: Optional[str] = None):
            def health_ok() -> bool:
                try:
                    if hasattr(component, 'health_check'):
                        status = component.health_check()
                        if isinstance(status, dict):
                            return bool(status.get('healthy', False))
                        return bool(status)
                    return component is not None
                except Exception as e:
                    self.query_logger.warning(f"Health-check исключение для {module_key}: {e}")
                    return False
            
            def recover() -> bool:
                try:
                    if hasattr(component, 'recover'):
                        ok = component.recover()
                        self.query_logger.info(f"Восстановление {module_key} через метод компонента: {'успех' if ok else 'неудача'}")
                        return bool(ok)
                    
                    initializer = getattr(self, 'component_initializer', None)
                    if initializer and init_method_name and hasattr(initializer, init_method_name):
                        method = getattr(initializer, init_method_name)
                        ok = method(self)
                        self.query_logger.info(f"Восстановление {module_key} через ComponentInitializer.{init_method_name}: {'успех' if ok else 'неудача'}")
                        return bool(ok)
                    
                    return False
                except Exception as e:
                    self.query_logger.error(f"Ошибка восстановления модуля {module_key}: {e}", exc_info=True)
                    return False
            
            try:
                ds.add_module_health_check(module_key, health_ok)
                ds.add_module_recovery_strategy(module_key, recover)
            except Exception as e:
                self.query_logger.error(f"Ошибка регистрации стратегии восстановления для {module_key}: {e}")
        
        for module_key, init_method in [
            ('ml_unit', 'initialize_ml_unit'),
            ('knowledge_graph', 'initialize'),
            ('memory_manager', 'initialize_memory_manager'),
        ]:
            if module_key in self.components and self.components[module_key]:
                register(module_key, self.components[module_key], init_method)
        
        for attr_name, init_method in [
            ('response_generator', 'initialize_response_generator'),
            ('text_processor', 'initialize_text_processor'),
            ('token_processor', 'initialize_token_processor'),
        ]:
            component = getattr(self, attr_name, None)
            if component is not None:
                register(attr_name, component, init_method)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Возвращает системные метрики."""
        self.query_logger.debug("Запрос системных метрик")
        metrics = self.metrics_manager.get_metrics()
        self.query_logger.debug(f"Получены системные метрики: {metrics}")
        return metrics
    
    def emit_metric(self, metric: Dict[str, Any]) -> bool:
        """Проксирует нормализованную метрику в SystemMetricsManager.emit()."""
        try:
            if hasattr(self.metrics_manager, "emit"):
                return bool(self.metrics_manager.emit(metric))
        except Exception:
            pass
        return False
    
    def emit_metrics(self, metrics: List[Dict[str, Any]]) -> int:
        """Проксирует список нормализованных метрик в SystemMetricsManager.emit_many()."""
        try:
            if hasattr(self.metrics_manager, "emit_many"):
                return int(self.metrics_manager.emit_many(metrics))
        except Exception:
            pass
        return 0
    
    def flush_emitted_metrics(self) -> List[Dict[str, Any]]:
        """Возвращает буфер нормализованных метрик из менеджера и очищает его."""
        try:
            if hasattr(self.metrics_manager, "flush"):
                return list(self.metrics_manager.flush())
        except Exception:
            pass
        return []
    
    def get_status(self) -> Dict[str, Any]:
        """Возвращает расширенный статус системы."""
        status = {
            "initialized": self.initialized,
            "running": self.running,
            "components": len(self.components),
            "metrics": self.metrics_manager.get_metrics() if hasattr(self.metrics_manager, 'get_metrics') else {}
        }
        
        if self.state_manager:
            status["system_state"] = self.state_manager.get_system_summary()
            status["health"] = {"status": self.state_manager.get_state().value}
        
        if self.resource_manager:
            status["resources"] = self.resource_manager.get_resource_summary()
        
        if self.config_manager:
            status["config_valid"] = self.config_manager.validate_config()
        
        return status
    
    def debug_message(self, message: str) -> str:
        """
        Debug communication bridge - receive commands and return responses.
        Use this for direct system interaction and debugging.
        
        Commands:
        - "status" - Return system status
        - "health" - Return health check
        - "test" - Test response
        - "memory" - Return memory stats
        - Or any message will be echoed back
        
        Args:
            message: Debug command or message
            
        Returns:
            Debug response string
        """
        msg = message.strip().lower()
        
        if msg == "status":
            return f"Status: fractal_ready={self.fractal_ready}, model={type(self.fractal_model_manager).__name__ if self.fractal_model_manager else None}"
        
        elif msg == "health":
            health = {
                'fractal_ready': self.fractal_ready,
                'model': self.fractal_model_manager.initialized if self.fractal_model_manager else False,
                'learning': hasattr(self, 'self_dialog_learning') and self.self_dialog_learning is not None
            }
            return f"Health: {health}"
        
        elif msg == "test":
            if self.fractal_model_manager:
                return self.fractal_model_manager.generate_response("Привет", max_tokens=30)
            return "Model not available"
        
        elif msg == "memory":
            if hasattr(self, 'memory_manager') and self.memory_manager:
                try:
                    return f"Memory: initialized={getattr(self.memory_manager, 'initialized', False)}"
                except:
                    return "Memory manager error"
            return "Memory manager not available"
        
        else:
            return f"[DEBUG] Received: '{message}'. System ready. Commands: status, health, test, memory"
    
    def _ensure_module_entry(self, name: str) -> Dict[str, Any]:
        """Гарантирует запись о модуле в реестре управления."""
        if name not in self.module_control:
            self.module_control[name] = {
                "enabled": True,
                "status": "unknown",
                "last_error": None,
                "last_change": time.time(),
            }
        return self.module_control[name]
    
    def enable_module(self, name: str) -> bool:
        """Включает модуль логически."""
        entry = self._ensure_module_entry(name)
        entry["enabled"] = True
        entry["last_change"] = time.time()
        self.query_logger.info(f"Модуль '{name}' включен")
        return True
    
    def disable_module(self, name: str, stop_if_running: bool = True) -> bool:
        """Отключает модуль и опционально останавливает его."""
        entry = self._ensure_module_entry(name)
        entry["enabled"] = False
        entry["last_change"] = time.time()
        
        component = self.components.get(name)
        try:
            if stop_if_running and component and hasattr(component, 'stop'):
                component.stop()
                entry["status"] = "stopped"
                self.query_logger.info(f"Модуль '{name}' остановлен при отключении")
            else:
                self.query_logger.info(f"Модуль '{name}' отключен")
            return True
        except Exception as e:
            entry["last_error"] = str(e)
            self.query_logger.error(f"Ошибка при отключении модуля '{name}': {e}", exc_info=True)
            return False
    
    def start_module(self, name: str) -> bool:
        """Запускает модуль, если он включен."""
        entry = self._ensure_module_entry(name)
        if not entry.get("enabled", True):
            self.query_logger.warning(f"Попытка запуска отключенного модуля '{name}'")
            return False
        
        component = self.components.get(name)
        if not component or not hasattr(component, 'start'):
            self.query_logger.warning(f"Компонент '{name}' не найден или не поддерживает start()")
            return False
        
        try:
            component.start()
            entry["status"] = "running"
            entry["last_error"] = None
            entry["last_change"] = time.time()
            self.query_logger.info(f"Модуль '{name}' запущен")
            return True
        except Exception as e:
            entry["last_error"] = str(e)
            entry["status"] = "error"
            if hasattr(self.metrics_manager, 'record_error'):
                self.metrics_manager.record_error(f"module_{name}_start_failed")
            self.query_logger.error(f"Ошибка запуска модуля '{name}': {e}", exc_info=True)
            return False
    
    def stop_module(self, name: str) -> bool:
        """Останавливает модуль."""
        entry = self._ensure_module_entry(name)
        component = self.components.get(name)
        
        if not component or not hasattr(component, 'stop'):
            self.query_logger.warning(f"Компонент '{name}' не найден или не поддерживает stop()")
            return False
        
        try:
            component.stop()
            entry["status"] = "stopped"
            entry["last_error"] = None
            entry["last_change"] = time.time()
            self.query_logger.info(f"Модуль '{name}' остановлен")
            return True
        except Exception as e:
            entry["last_error"] = str(e)
            entry["status"] = "error"
            self.query_logger.error(f"Ошибка остановки модуля '{name}': {e}", exc_info=True)
            return False
    
    def get_module_status(self, name: str) -> Dict[str, Any]:
        """Возвращает агрегированный статус модуля."""
        entry = self._ensure_module_entry(name)
        component = self.components.get(name)
        
        status = {
            "enabled": entry.get("enabled", True),
            "status": entry.get("status", "unknown"),
            "last_error": entry.get("last_error"),
            "last_change": entry.get("last_change"),
            "running": bool(getattr(component, 'running', False)) if component else False,
            "healthy": None,
        }
        
        try:
            if component and hasattr(component, 'health_check'):
                hc = component.health_check()
                status["healthy"] = bool(hc.get('healthy')) if isinstance(hc, dict) else bool(hc)
        except Exception:
            status["healthy"] = False
        
        return status
    
    def list_modules(self) -> Dict[str, Dict[str, Any]]:
        """Сводный список модулей и их статусов."""
        result = {}
        for name in sorted(self.components.keys()):
            result[name] = self.get_module_status(name)
        return result
    
    def is_model_ready(self, model_id: str) -> bool:
        """True, если модель загружена и доступна в ModelManager."""
        try:
            mm = getattr(self, 'model_manager', None)
            if not mm:
                return False
            models = getattr(mm, 'models', {})
            return model_id in models
        except Exception:
            return False
    
    def ensure_model_available(self, model_id: str, wait: bool = False, timeout_s: float = 0.0) -> Dict[str, Any]:
        """Инициирует загрузку модели и опционально ожидает готовности."""
        info = {"requested": model_id, "started": False, "ready": False, "waited": 0.0, "remaining": None}
        
        mm = getattr(self, 'model_manager', None)
        if not mm:
            info["error"] = "ModelManager недоступен"
            return info
        
        try:
            if not self.is_model_ready(model_id):
                started = bool(mm.load_model(model_id)) if hasattr(mm, 'load_model') else False
                info["started"] = started
            else:
                info["started"] = False
            
            if wait:
                t0 = time.time()
                deadline = t0 + max(0.0, float(timeout_s))
                while time.time() < deadline:
                    if self.is_model_ready(model_id):
                        info["ready"] = True
                        break
                    time.sleep(0.2)
                info["waited"] = round(time.time() - t0, 3)
                info["remaining"] = max(0.0, round(deadline - time.time(), 3))
            else:
                info["ready"] = self.is_model_ready(model_id)
            
            return info
        except Exception as e:
            info["error"] = str(e)
            return info
    
    def get_contradiction_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику по противоречиям."""
        self._log_throttled(self.query_logger, logging.INFO, "contradiction_stats_request", "Запрос статистики противоречий")
        
        contradiction_manager = None
        
        for attr_name in ['contradiction_manager', 'contradiction_resolver']:
            if hasattr(self, 'components') and attr_name in self.components:
                contradiction_manager = self.components[attr_name]
                break
            if hasattr(self, attr_name):
                contradiction_manager = getattr(self, attr_name)
                break
        
        if contradiction_manager is None and hasattr(self, 'component_initializer'):
            for attr_name in ['contradiction_manager', 'contradiction_resolver']:
                contradiction_manager = getattr(self.component_initializer, attr_name, None)
                if contradiction_manager:
                    break
        
        if contradiction_manager:
            try:
                stats_start = time.time()
                stats = contradiction_manager.get_contradiction_stats()
                stats_time = time.time() - stats_start
                self._log_throttled(self.query_logger, logging.INFO, "contradiction_stats_received", 
                                   f"Статистика противоречий получена за {stats_time:.4f} сек")
                return stats
            except Exception as e:
                self.query_logger.error(f"Ошибка получения статистики противоречий: {e}", exc_info=True)
                return {"error": str(e), "timestamp": time.time()}
        
        self.query_logger.warning("Менеджер противоречий недоступен")
        return {}
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Возвращает системные метрики."""
        base = self.system_metrics_manager.get_metrics() if self.system_metrics_manager else {}
        
        try:
            neu = self.components.get('neuromorphic_simulator') if hasattr(self, 'components') else None
            if neu is None and hasattr(self, 'neuromorphic_simulator'):
                neu = getattr(self, 'neuromorphic_simulator')
            
            if neu:
                neu_status = {
                    "available": True,
                    "running": bool(getattr(neu, 'running', False)),
                    "use_nest": bool(getattr(neu, 'use_nest', False)),
                }
                
                if hasattr(neu, 'get_system_health'):
                    health = neu.get_system_health()
                    neu_status.update({
                        "health_status": health.get("status"),
                        "health_score": health.get("health_score"),
                        "interaction_strength": (health.get("analysis", {}) or {}).get("interaction_strength"),
                        "total_activities": (health.get("analysis", {}) or {}).get("total_activities"),
                        "timestamp": health.get("timestamp")
                    })
                
                base["neuromorphic"] = neu_status
            else:
                base["neuromorphic"] = {"available": False}
        except Exception as e:
            try:
                base["neuromorphic_error"] = str(e)
            except Exception:
                pass
        
        return base
    
    def get_system_dashboard_data(self) -> Dict[str, Any]:
        """Возвращает данные для системного дашборда."""
        self._log_throttled(self.query_logger, logging.INFO, "dashboard_request", "Запрос данных для системного дашборда")
        
        dashboard_start = time.time()
        
        try:
            data = {
                "timestamp": time.time(),
                "metrics": self.system_metrics_manager.get_metrics() if self.system_metrics_manager else {},
                "health": self.get_system_health(),
                "contradiction_stats": self.get_contradiction_statistics(),
                "learning_opportunities": self._get_learning_opportunities(),
                "system_info": self._get_system_info()
            }
            
            dashboard_time = time.time() - dashboard_start
            self._log_throttled(self.query_logger, logging.INFO, "dashboard_ready", 
                               f"Данные дашборда сформированы за {dashboard_time:.4f} сек")
            self.query_logger.debug(f"Данные дашборда: {data}")
            return data
        except Exception as e:
            dashboard_time = time.time() - dashboard_start
            self.query_logger.error(f"Ошибка формирования данных дашборда за {dashboard_time:.4f} сек: {e}", exc_info=True)
            return {
                "error": str(e),
                "timestamp": time.time(),
                "partial_data": {
                    "metrics": {},
                    "health": self.get_system_health()
                }
            }
    
    def _get_learning_opportunities(self) -> List[Dict[str, Any]]:
        """Возвращает возможности для обучения."""
        try:
            opportunities = []
            
            if 'ml_unit' in self.components and self.components['ml_unit']:
                ml_unit = self.components['ml_unit']
                if hasattr(ml_unit, 'get_learning_opportunities'):
                    opportunities.extend(ml_unit.get_learning_opportunities())
            
            if not opportunities:
                opportunities = [
                    {
                        "type": "pattern_analysis",
                        "description": "Анализ паттернов в запросах пользователей",
                        "priority": "medium",
                        "timestamp": time.time()
                    },
                    {
                        "type": "knowledge_expansion",
                        "description": "Расширение базы знаний",
                        "priority": "low",
                        "timestamp": time.time()
                    }
                ]
            
            return opportunities
        except Exception as e:
            self.query_logger.error(f"Ошибка получения возможностей обучения: {e}", exc_info=True)
            return []
    
    def add_deferred_command(self, command: callable, *args, **kwargs):
        """Добавляет отложенную команду для выполнения после полной инициализации."""
        self.query_logger.info(f"Добавлена отложенная команда: {getattr(command, '__name__', 'lambda')}")
        self.deferred_commands.append((command, args, kwargs))
    
    def get_response_metadata(self, query: str) -> Dict[str, Any]:
        """Возвращает метаданные ответа на запрос."""
        self.query_logger.debug(f"Запрос метаданных для ответа на запрос: '{query[:50]}{'...' if len(query) > 50 else ''}'")
        
        metadata = {
            "timestamp": time.time(),
            "contradictions_detected": False,
            "personalized": False,
            "components_used": [],
            "query_length": len(query)
        }
        
        contradiction_resolver = None
        for attr_name in ['contradiction_resolver', 'contradiction_manager']:
            if hasattr(self, attr_name) and getattr(self, attr_name):
                contradiction_resolver = getattr(self, attr_name)
                break
            if hasattr(self, 'components') and attr_name in self.components:
                contradiction_resolver = self.components[attr_name]
                break
        
        if contradiction_resolver:
            contradictions_start = time.time()
            contradictions = contradiction_resolver.get_active_contradictions()
            contradictions_time = time.time() - contradictions_start
            relevant = [c for c in contradictions if query.lower() in c.get('concept', '').lower()]
            
            if relevant:
                metadata["contradictions_detected"] = True
                metadata["contradictions"] = relevant
                self.query_logger.info(f"Обнаружено {len(relevant)} релевантных противоречий за {contradictions_time:.4f} сек")
            else:
                self.query_logger.debug(f"Противоречия не обнаружены за {contradictions_time:.4f} сек")
        
        if hasattr(self, 'adaptation_manager') and self.adaptation_manager:
            metadata["personalized"] = True
            metadata["components_used"].append("adaptation_manager")
        
        if hasattr(self, 'text_processor') and self.text_processor:
            metadata["components_used"].append("text_processor")
        
        if 'knowledge_graph' in self.components and self.components['knowledge_graph']:
            metadata["components_used"].append("knowledge_graph")
        
        self.query_logger.debug(f"Сформированы метаданные ответа: {metadata}")
        return metadata
    
    def _on_metrics_event(self, data: Dict[str, Any]):
        """Обработчик событий метрик."""
        try:
            if self.metrics_manager and hasattr(self.metrics_manager, 'record_event'):
                self.metrics_manager.record_event(data)
        except Exception as e:
            self.query_logger.debug(f"Ошибка обработки события метрик: {e}")


# Backward compatibility alias
CogniFlexCore = CoreBrain


def setup_logging():
    """Настраивает логгирование для системы."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Основная функция запуска системы."""
    setup_logging()
    logger.info("Логгирование настроено")
    
    core = CoreBrain()
    logger.info("Создан экземпляр CoreBrain")
    
    startup_start = time.time()
    
    if core.initialize():
        logger.info("Ядро успешно инициализировано")
        
        if core.start():
            logger.info("Система CogniFlex успешно запущена")
            startup_time = time.time() - startup_start
            logger.info(f"Полное время запуска системы: {startup_time:.4f} сек")
            
            logger.info("Система работает. Введите 'exit' для остановки или нажмите Ctrl+C")
            
            def _cli_exit_listener():
                """Слушает ввод из терминала и останавливает систему по команде 'exit'."""
                try:
                    while core.running:
                        try:
                            user_input = input()
                        except EOFError:
                            break
                        if isinstance(user_input, str) and user_input.strip().lower() in ("exit", "quit", "q"):
                            logger.info("Получена команда 'exit' из терминала — инициируем остановку...")
                            try:
                                core.stop()
                            except Exception as e:
                                logger.error(f"Ошибка при остановке по команде exit: {e}", exc_info=True)
                            break
                except Exception as e:
                    logger.error(f"Ошибка в потоке чтения команд CLI: {e}", exc_info=True)
            
            _cli_listener_thread = threading.Thread(target=_cli_exit_listener, name="CLIExitListener", daemon=True)
            _cli_listener_thread.start()
            
            try:
                logger.info("Ожидание команд. Введите 'exit' для завершения или Ctrl+C")
                while core.running:
                    time.sleep(1)
                    if time.time() % 30 < 0.1:
                        health = core.get_system_health()
                        logger.info(f"Текущее состояние системы: {health['status']}")
                        if health['status'] != 'healthy':
                            logger.warning("Система работает в деградированном состоянии!")
            except KeyboardInterrupt:
                logger.info("Получен сигнал остановки от пользователя")
            except Exception as e:
                logger.critical(f"Критическая ошибка в основном цикле: {e}", exc_info=True)
            finally:
                logger.info("Начало процесса остановки системы")
                core.stop()
                logger.info("Система остановлена")
        else:
            logger.error("Не удалось запустить ядро после инициализации")
    else:
        logger.critical("Не удалось инициализировать ядро системы")

    # Методы для совместимости с GUI
    def get_resource_snapshot(self) -> Dict[str, Any]:
        """Возвращает снимок использования ресурсов."""
        try:
            if hasattr(self, 'resource_manager') and self.resource_manager:
                return {
                    'cpu_usage': self.resource_manager.get_cpu_usage(),
                    'memory_usage': self.resource_manager.get_memory_usage(),
                    'disk_usage': self.resource_manager.get_disk_usage() if hasattr(self.resource_manager, 'get_disk_usage') else 0,
                    'timestamp': time.time()
                }
        except Exception as e:
            self.query_logger.warning(f"Ошибка получения снимка ресурсов: {e}")
        return {}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Возвращает статистику кэша."""
        try:
            cache_stats = {}
            if hasattr(self, 'hybrid_cache') and self.hybrid_cache:
                cache_stats['hybrid_cache'] = getattr(self.hybrid_cache, 'get_stats', lambda: {})()
            if hasattr(self, 'memory_manager') and self.memory_manager:
                cache_stats['memory_manager'] = getattr(self.memory_manager, 'get_stats', lambda: {})()
            return cache_stats
        except Exception as e:
            self.query_logger.warning(f"Ошибка получения статистики кэша: {e}")
        return {}
    
    def tokenize_query(self, query: str) -> Dict[str, Any]:
        """Токенизирует запрос и возвращает информацию."""
        try:
            if hasattr(self, 'text_processor') and self.text_processor:
                tokens = self.text_processor.tokenize(query)
                return {
                    'tokens': tokens,
                    'token_count': len(tokens) if isinstance(tokens, list) else 0,
                    'processing_time': 0.1
                }
            elif hasattr(self, 'ml_unit') and self.ml_unit:
                # Пробуем токенизировать через MLUnit
                if hasattr(self.ml_unit, 'tokenizer'):
                    tokens = self.ml_unit.tokenizer.encode(query)
                    return {
                        'tokens': tokens,
                        'token_count': len(tokens),
                        'processing_time': 0.1
                    }
        except Exception as e:
            self.query_logger.warning(f"Ошибка токенизации запроса: {e}")
        
        # Базовая токенизация как fallback
        words = query.split()
        return {
            'tokens': words,
            'token_count': len(words),
            'processing_time': 0.01
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Возвращает детальный статус системы."""
        try:
            return {
                'initialized': self.initialized,
                'running': self.running,
                'components_count': len(self.components) if hasattr(self, 'components') else 0,
                'components': list(self.components.keys()) if hasattr(self, 'components') else [],
                'timestamp': time.time()
            }
        except Exception as e:
            self.query_logger.warning(f"Ошибка получения статуса системы: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_components_status(self) -> Dict[str, Dict[str, Any]]:
        """Возвращает статус всех компонентов системы."""
        status = {}
        
        key_components = [
            'fractal_storage', 'self_reasoning_engine', 'reasoning_integration',
            'knowledge_graph', 'memory_manager', 'hybrid_cache',
            'ml_unit', 'model_manager', 'qwen_model_manager',
            'curiosity_engine', 'learning_manager', 'self_dialog_learning',
            'ethics_framework', 'contradiction_manager', 'event_bus'
        ]
        
        for comp_name in key_components:
            component = self.components.get(comp_name) or getattr(self, comp_name, None)
            
            if component is None:
                status[comp_name] = {'status': 'not_initialized', 'healthy': False}
            else:
                try:
                    # Пробуем получить статус через разные атрибуты
                    is_healthy = True
                    detail = 'initialized'
                    
                    if hasattr(component, 'initialized'):
                        is_healthy = getattr(component, 'initialized', False)
                    elif hasattr(component, 'health_check'):
                        health = component.health_check()
                        is_healthy = health.get('healthy', True) if isinstance(health, dict) else bool(health)
                    
                    if hasattr(component, 'get_stats'):
                        try:
                            detail = component.get_stats()
                        except:
                            pass
                    
                    status[comp_name] = {
                        'status': 'ok' if is_healthy else 'error',
                        'healthy': is_healthy,
                        'detail': detail
                    }
                except Exception as e:
                    status[comp_name] = {'status': 'error', 'healthy': False, 'error': str(e)}
        
        return status
    
    def shutdown(self, save_state: bool = True) -> Dict[str, Any]:
        """Graceful shutdown системы с сохранением состояния."""
        if self._shutting_down:
            return {'status': 'already_shutting_down'}
        
        with self._shutdown_lock:
            self._shutting_down = True
            self.running = False
            
            results = {'saved': [], 'errors': []}
            
            # Сохраняем состояние FractalStorage
            if save_state:
                try:
                    fractal_storage = getattr(self, 'fractal_storage', None)
                    if fractal_storage and hasattr(fractal_storage, 'flush'):
                        fractal_storage.flush()
                        results['saved'].append('fractal_storage')
                except Exception as e:
                    results['errors'].append(f'fractal_storage: {e}')
                
                # Сохраняем гибридный кэш
                try:
                    hybrid_cache = getattr(self, 'hybrid_cache', None)
                    if hybrid_cache and hasattr(hybrid_cache, 'save'):
                        hybrid_cache.save()
                        results['saved'].append('hybrid_cache')
                except Exception as e:
                    results['errors'].append(f'hybrid_cache: {e}')
            
            # Останавливаем фоновые потоки
            try:
                if hasattr(self, 'deferred_system') and self.deferred_system:
                    self.deferred_system.shutdown()
            except Exception as e:
                results['errors'].append(f'deferred_system: {e}')
            
            self.query_logger.info(f"Система остановлена. Сохранено: {results['saved']}, Ошибки: {results['errors']}")
            
            return {
                'status': 'shutdown_complete',
                'saved': results['saved'],
                'errors': results['errors']
            }

if __name__ == "__main__":
    main()