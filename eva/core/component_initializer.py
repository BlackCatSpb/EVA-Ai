"""
ComponentInitializer - Единый инициализатор компонентов системы ЕВА
Версия: 2.0.0
Поддерживаемые компоненты: 21
"""
import os
import sys
import logging
import time
import threading
from typing import Dict, Any, List, Set, Optional, Callable, Tuple

logger = logging.getLogger("eva.component_initializer")

def _ensure_eva_path():
    """Ensure ЕВА path is in sys.path using dynamic path detection."""
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
    
    try:
        os.chdir(eva_root)
    except OSError:
        pass
    except Exception:
        pass
    
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
    
    # Полный список из 23 компонента системы
    COMPONENT_LIST = [
        # Системные компоненты (3)
        'event_bus',
        'resource_manager', 
        'config_manager',
        
        # Память и кэширование (2)
        'memory_manager',
        'hybrid_cache',
        
        # Знания и обработка (2)
        'knowledge_graph',
        'text_processor',
        
        # ML компоненты (2)
        'ml_unit',
        'model_manager',
        
        # Основная логика (3)
        'query_processor',
        'response_generator',
        'reasoning_engine',
        
        # Обучение (3)
        'training_orchestrator',
        'learning_manager',
        'learning_scheduler',
        
        # Аналитика (3)
        'system_monitor',
        'metrics_collector',
        'analytics_manager',
        
        # Специализированные (6)
        'contradiction_manager',
        'ethics_framework',
        'qwen_api_enhancer',
        'adaptation_manager',
        'web_search_engine',
        'gui',
        
        # Fractal Reasoning (2) - добавлены позже
        'fractal_storage',
        'self_reasoning_engine',
    ]
    
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
        self.logger = logging.getLogger("eva.component_initializer")
        self.component_configs: Dict[str, Any] = {}
        
        # Определяем зависимости между компонентами
        self._define_dependencies()
        
        # Регистрируем фабрики компонентов
        self._register_component_factories()
        
        self.logger.info("ComponentInitializer инициализирован")
    
    def _validate_dependencies(self, component_name: str) -> Tuple[bool, List[str]]:
        """
        Проверяет и валидирует зависимости компонента перед инициализацией.
        
        Args:
            component_name: Имя компонента для проверки
            
        Returns:
            Tuple[bool, List[str]]: (валидность, список проблем)
        """
        issues = []
        
        if component_name not in self.component_dependencies:
            issues.append(f"Component {component_name} not found in dependencies")
            return False, issues
        
        dependencies = self.component_dependencies[component_name]
        
        for dep in dependencies:
            if dep not in self.component_factories:
                issues.append(f"Dependency {dep} not registered as factory")
                continue
                
            if dep in self.failed_components:
                issues.append(f"Dependency {dep} previously failed to initialize")
                
            if dep not in self.initialized_components and dep not in self.failed_components:
                issues.append(f"Dependency {dep} not yet initialized")
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            self.logger.warning(f"Dependency validation failed for {component_name}: {issues}")
        
        return is_valid, issues
    
    def _define_dependencies(self):
        """Определяет зависимости между компонентами."""
        self.component_dependencies = {
            # Системные компоненты не имеют зависимостей
            'event_bus': [],
            'resource_manager': [],
            'config_manager': [],
            
            # Память не зависит от системных
            'memory_manager': [],
            'hybrid_cache': ['memory_manager'],
            
            # Знания зависят от событий
            'knowledge_graph': ['event_bus'],
            'text_processor': ['hybrid_cache'],
            
            # ML зависит от памяти и знаний
            'ml_unit': ['memory_manager', 'knowledge_graph'],
            'model_manager': ['ml_unit'],
            
            # Логика зависит от обработки и ML
            'query_processor': ['text_processor', 'knowledge_graph'],
            'response_generator': ['query_processor'],  # Убрали model_manager как обязательную зависимость
            'reasoning_engine': ['knowledge_graph'],
            
            # Обучение зависит от ML
            'training_orchestrator': ['ml_unit'],  # Убрали model_manager как обязательную зависимость
            'learning_manager': ['training_orchestrator', 'knowledge_graph'],
            'learning_scheduler': ['learning_manager'],
            
            # Аналитика зависит от мониторинга
            'analytics_manager': ['system_monitor'],
            'system_monitor': ['resource_manager'],
            'metrics_collector': ['system_monitor'],
            
            # Специализированные зависят от знаний
            'contradiction_manager': ['knowledge_graph'],
            'adaptation_manager': ['analytics_manager'],
            'ethics_framework': ['knowledge_graph'],
            'web_search_engine': ['knowledge_graph'],
            'gui': [],
            
            # Fractal Reasoning компоненты (фрактальное хранилище для саморассуждений)
            'fractal_storage': [],
            'self_reasoning_engine': ['fractal_storage', 'knowledge_graph'],
        }
    
    def _register_component_factories(self):
        """Регистрирует фабрики для создания всех 21 компонентов."""
        
        # ===== СИСТЕМНЫЕ КОМПОНЕНТЫ =====
        
        def create_event_bus():
            try:
                from eva.core.event_bus import EventBus
                event_bus = EventBus()
                self.core_brain.event_bus = event_bus
                self.logger.info("[OK] EventBus создан")
                return event_bus
            except Exception as e:
                self.logger.error(f"[FAIL] Ошибка создания event_bus: {e}", exc_info=True)
                self.failed_components.add('event_bus')
                return None
        
        def create_resource_manager():
            try:
                from eva.core.resource_manager import ResourceManager
                config_manager = getattr(self.core_brain, 'config_manager', None)
                resource_manager = ResourceManager(config_manager=config_manager)
                self.core_brain.resource_manager = resource_manager
                self.logger.info("[OK] ResourceManager создан")
                return resource_manager
            except Exception as e:
                self.logger.error(f"[FAIL] Ошибка создания resource_manager: {e}", exc_info=True)
                self.failed_components.add('resource_manager')
                return None
        
        def create_config_manager():
            try:
                from eva.core.config_manager import ConfigManager
                config_manager = ConfigManager()
                self.core_brain.config_manager = config_manager
                self.logger.info("[OK] ConfigManager создан")
                return config_manager
            except Exception as e:
                self.logger.error(f"[FAIL] Ошибка создания config_manager: {e}", exc_info=True)
                self.failed_components.add('config_manager')
                return None
        
        # ===== ПАМЯТЬ И КЭШИРОВАНИЕ =====
        
        def create_memory_manager():
            try:
                from eva.memory.memory_manager import MemoryManager
                cache_dir = os.path.join(
                    getattr(self.core_brain, 'cache_dir', './cache'),
                    'memory'
                )
                os.makedirs(cache_dir, exist_ok=True)
                memory_manager = MemoryManager(
                    brain=self.core_brain,
                    cache_dir=cache_dir
                )
                if hasattr(memory_manager, 'initialize'):
                    init_result = memory_manager.initialize()
                    if init_result is False:
                        self.logger.warning("[WARN] MemoryManager.initialize() вернул False - возможны проблемы")
                self.memory_manager = memory_manager
                self.core_brain.memory_manager = memory_manager
                self.logger.info("[OK] MemoryManager создан")
                return memory_manager
            except Exception as e:
                self.logger.error(f"[FAIL] Ошибка создания memory_manager: {e}", exc_info=True)
                self.failed_components.add('memory_manager')
                return None
        
        def create_hybrid_cache():
            try:
                self.logger.info("Начало создания HybridTokenCache...")
                self.logger.debug(f"core_brain тип: {type(self.core_brain)}")
                self.logger.debug(f"core_brain атрибуты: {dir(self.core_brain)}")
                
                # Проверяем, есть ли уже кэш в brain
                existing = getattr(self.core_brain, 'token_cache', None) or getattr(self.core_brain, 'hybrid_cache', None)
                if existing is not None:
                    self.logger.info(f"Используем существующий HybridTokenCache: {id(existing)}")
                    hybrid_cache = existing
                else:
                    from eva.memory.hybrid_token_cache import get_shared_cache
                    self.logger.info("Импортирован get_shared_cache")
                    hybrid_cache = get_shared_cache(self.core_brain, "default")
                    self.logger.info(f"Создан/получен синглтон HybridTokenCache: {id(hybrid_cache)}")
                
                if hybrid_cache is None:
                    self.logger.error("HybridTokenCache вернул None!")
                    return None
                
                # Регистрируем в components
                if not hasattr(self.core_brain, 'components'):
                    self.core_brain.components = {}
                self.core_brain.components['hybrid_cache'] = hybrid_cache
                self.logger.info("Добавлен в brain.components")
                
                # Обратная совместимость
                self.core_brain.token_cache = hybrid_cache
                self.core_brain.hybrid_cache = hybrid_cache
                self.logger.info("Установлены обратные ссылки")
                
                self.logger.info(f"[OK] HybridTokenCache создан: {id(hybrid_cache)}")
                return hybrid_cache
                
            except Exception as e:
                self.logger.error(f"[FAIL] Ошибка создания hybrid_cache: {e}", exc_info=True)
                self.failed_components.add('hybrid_cache')
                return None
        
        # ===== ЗНАНИЯ И ОБРАБОТКА =====
        
        def create_knowledge_graph():
            try:
                _ensure_eva_path()
                from eva.knowledge.knowledge_graph_integrated import IntegratedKnowledgeGraph
                event_bus = getattr(self.core_brain, 'event_bus', None)
                knowledge_graph = IntegratedKnowledgeGraph(
                    brain=self.core_brain,
                    event_bus=event_bus,
                    name="knowledge_graph"
                )
                if hasattr(knowledge_graph, 'initialize'):
                    knowledge_graph.initialize()
                self.knowledge_graph = knowledge_graph
                self.core_brain.knowledge_graph = knowledge_graph
                self.logger.info("[OK] KnowledgeGraph создан")
                return knowledge_graph
            except Exception as e:
                self.logger.error(f"[FAIL] Ошибка создания knowledge_graph: {e}", exc_info=True)
                self.failed_components.add('knowledge_graph')
                return None
        
        def create_qwen_api_enhancer():
            """Создает QwenAPIEnhancer для обогащения знаний"""
            try:
                import os
                from eva.knowledge.qwen_api_enhancer import QwenAPIEnhancer
                
                api_key = os.environ.get('OPENROUTER_API_KEY', '')
                enhancer = QwenAPIEnhancer(api_key=api_key, enable_fallbacks=True)
                
                self.core_brain.qwen_api_enhancer = enhancer
                self.logger.info(f"[OK] QwenAPIEnhancer создан: {enhancer.get_status()}")
                return enhancer
            except Exception as e:
                self.logger.error(f"[FAIL] Ошибка создания qwen_api_enhancer: {e}", exc_info=True)
                self.failed_components.add('qwen_api_enhancer')
                return None
        
        def create_text_processor():
            try:
                _ensure_eva_path()
                from eva.mlearning.unified_text_processor import UnifiedTextProcessor
                text_processor = UnifiedTextProcessor(brain=self.core_brain)
                hybrid_cache = getattr(self.core_brain, 'hybrid_cache', None)
                if hybrid_cache:
                    text_processor.hybrid_cache = hybrid_cache
                    self.logger.info("   └─ Гибридный кэш подключен")
                text_processor.initialize()
                text_processor._setup_component()
                self.core_brain.text_processor = text_processor
                self.logger.info("[OK] TextProcessor создан")
                return text_processor
            except Exception as e:
                self.logger.error(f"[FAIL] Ошибка создания text_processor: {e}", exc_info=True)
                self.failed_components.add('text_processor')
                return None
        
        # ===== ML КОМПОНЕНТЫ =====
        
        def create_ml_unit():
            try:
                _ensure_eva_path()
                from eva.mlearning.ml_unit import MLUnit
                cache_dir = os.path.join(
                    getattr(self.core_brain, 'cache_dir', './cache'),
                    'ml_unit'
                )
                os.makedirs(cache_dir, exist_ok=True)
                ml_unit = MLUnit(
                    brain=self.core_brain,
                    cache_dir=cache_dir,
                    max_workers=4
                )
                if hasattr(ml_unit, 'initialize'):
                    ml_unit.initialize()
                hybrid_cache = getattr(self.core_brain, 'hybrid_cache', None)
                if hybrid_cache:
                    ml_unit.hybrid_cache = hybrid_cache
                    self.logger.info("   └─ Гибридный кэш подключен")
                self.ml_unit = ml_unit
                self.core_brain.ml_unit = ml_unit
                self.logger.info("[OK] MLUnit создан")
                return ml_unit
            except Exception as e:
                self.logger.error(f"[FAIL] Ошибка создания ml_unit: {e}", exc_info=True)
                self.failed_components.add('ml_unit')
                return None
        
        def create_model_manager():
            try:
                self.logger.info("Начало создания ModelManager...")
                from eva.mlearning.hybrid_model_manager import HybridModelManager
                
                self.logger.info("Создание экземпляра HybridModelManager...")
                model_manager = HybridModelManager(
                    brain=self.core_brain,
                    max_vram_gb=0.5,
                    max_ssd_gb=2.0
                )
                
                self.logger.info("Вызов initialize()...")
                init_result = None
                try:
                    if hasattr(model_manager, 'initialize'):
                        init_result = model_manager.initialize()
                    else:
                        init_result = True
                    
                    if init_result is False:
                        self.logger.error("[FAIL] Не удалось инициализировать HybridModelManager - initialize() вернул False")
                        self.logger.warning("[WARN] Продолжаем без ModelManager (некоторые компоненты будут недоступны)")
                        self.failed_components.add('model_manager')
                        return None
                    
                    self.logger.info("[OK] ModelManager успешно инициализирован")
                    
                except Exception as init_error:
                    self.logger.error(f"[FAIL] Исключение при инициализации ModelManager: {init_error}", exc_info=True)
                    self.logger.warning("[WARN] Продолжаем без ModelManager (некоторые компоненты будут недоступны)")
                    self.failed_components.add('model_manager')
                    return None
                    
                self.model_manager = model_manager
                self.core_brain.model_manager = model_manager
                self.logger.info("[OK] ModelManager создан")
                return model_manager
            except Exception as e:
                self.logger.error(f"[FAIL] Ошибка создания model_manager: {e}", exc_info=True)
                self.failed_components.add('model_manager')
                return None
        
        # ===== ОСНОВНАЯ ЛОГИКА =====
        
        def create_query_processor():
            try:
                from eva.core.query_processor import QueryProcessor
                query_processor = QueryProcessor(brain=self.core_brain)
                if hasattr(query_processor, 'initialize'):
                    init_result = query_processor.initialize()
                    if init_result is False:
                        self.logger.warning("[WARN] QueryProcessor.initialize() вернул False")
                self.core_brain.query_processor = query_processor
                self.logger.info("[OK] QueryProcessor создан")
                return query_processor
            except Exception as e:
                self.logger.error(f"[FAIL] Ошибка создания query_processor: {e}", exc_info=True)
                self.failed_components.add('query_processor')
                return None
        
        def create_response_generator():
            try:
                from eva.core.response_generator import ResponseGenerator
                response_generator = ResponseGenerator(brain=self.core_brain)
                if hasattr(response_generator, 'initialize'):
                    response_generator.initialize()
                self.core_brain.response_generator = response_generator
                self.logger.info("[OK] ResponseGenerator создан")
                return response_generator
            except Exception as e:
                self.logger.error(f"[FAIL] Ошибка создания response_generator: {e}", exc_info=True)
                self.failed_components.add('response_generator')
                return None
        
        def create_reasoning_engine():
            try:
                from eva.core.reasoning_engine import ReasoningEngine
                reasoning_engine = ReasoningEngine(brain=self.core_brain)
                if hasattr(reasoning_engine, 'initialize'):
                    reasoning_engine.initialize()
                self.core_brain.reasoning_engine = reasoning_engine
                self.logger.info("[OK] ReasoningEngine создан")
                return reasoning_engine
            except Exception as e:
                self.logger.error(f"[FAIL] Ошибка создания reasoning_engine: {e}", exc_info=True)
                self.failed_components.add('reasoning_engine')
                return None
        
        # ===== ОБУЧЕНИЕ =====
        
        def create_training_orchestrator():
            try:
                from eva.mlearning.training_orchestrator import TrainingOrchestrator
                training_orchestrator = TrainingOrchestrator(brain=self.core_brain)
                if hasattr(training_orchestrator, 'initialize'):
                    training_orchestrator.initialize()
                self.core_brain.training_orchestrator = training_orchestrator
                self.logger.info("[OK] TrainingOrchestrator создан")
                return training_orchestrator
            except Exception as e:
                self.logger.error(f"[FAIL] Ошибка создания training_orchestrator: {e}", exc_info=True)
                self.failed_components.add('training_orchestrator')
                return None

        def create_learning_manager():
            try:
                from eva.learning.learning_manager import LearningManager
                learning_manager = LearningManager(brain=self.core_brain)
                if hasattr(learning_manager, 'initialize'):
                    learning_manager.initialize()
                self.core_brain.learning_manager = learning_manager
                self.logger.info("[OK] LearningManager создан")
                return learning_manager
            except Exception as e:
                self.logger.error(f"[FAIL] Ошибка создания learning_manager: {e}", exc_info=True)
                self.failed_components.add('learning_manager')
                return None
        
        def create_learning_scheduler():
            try:
                from eva.core.learning_scheduler import LearningScheduler
                attention_system = getattr(self.core_brain, 'attention_system', None)
                if attention_system is None:
                    self.logger.warning("[WARN] attention_system не найден - используется DummyAttentionSystem")
                    class DummyAttentionSystem:
                        def __init__(self):
                            self.pending_opportunities = []
                    attention_system = DummyAttentionSystem()
                learning_scheduler = LearningScheduler(attention_system)
                if hasattr(learning_scheduler, 'initialize'):
                    init_result = learning_scheduler.initialize()
                    if init_result is False:
                        self.logger.warning("[WARN] LearningScheduler.initialize() вернул False")
                self.core_brain.learning_scheduler = learning_scheduler
                self.logger.info("[OK] LearningScheduler создан")
                return learning_scheduler
            except Exception as e:
                self.logger.error(f"[FAIL] Ошибка создания learning_scheduler: {e}", exc_info=True)
                self.failed_components.add('learning_scheduler')
                return None
        
        # ===== АНАЛИТИКА =====
        
        def create_analytics_manager():
            try:
                from eva.analytics.analytics_manager import AnalyticsManager
                analytics_manager = AnalyticsManager(brain=self.core_brain)
                if hasattr(analytics_manager, 'initialize'):
                    analytics_manager.initialize()
                self.core_brain.analytics_manager = analytics_manager
                self.logger.info("[OK] AnalyticsManager создан")
                return analytics_manager
            except Exception as e:
                self.logger.error(f"[FAIL] Ошибка создания analytics_manager: {e}", exc_info=True)
                self.failed_components.add('analytics_manager')
                return None
        
        def create_system_monitor():
            try:
                from eva.monitoring.system_monitor import SystemMonitor
                system_monitor = SystemMonitor()
                if hasattr(system_monitor, 'initialize'):
                    system_monitor.initialize()
                self.core_brain.system_monitor = system_monitor
                self.logger.info("[OK] SystemMonitor создан")
                return system_monitor
            except Exception as e:
                self.logger.error(f"[FAIL] Ошибка создания system_monitor: {e}", exc_info=True)
                self.failed_components.add('system_monitor')
                return None
        
        def create_metrics_collector():
            try:
                try:
                    from eva.core.metrics_collector import MetricsCollector
                except ImportError:
                    from eva.analytics.analytics_manager import AnalyticsManager as MetricsCollector
                metrics_collector = MetricsCollector(brain=self.core_brain)
                if hasattr(metrics_collector, 'initialize'):
                    init_result = metrics_collector.initialize()
                    if init_result is False:
                        self.logger.warning("[WARN] MetricsCollector.initialize() вернул False")
                self.core_brain.metrics_collector = metrics_collector
                self.logger.info("[OK] MetricsCollector создан")
                return metrics_collector
            except Exception as e:
                self.logger.error(f"[FAIL] Ошибка создания metrics_collector: {e}", exc_info=True)
                self.failed_components.add('metrics_collector')
                return None
        
        # ===== СПЕЦИАЛИЗИРОВАННЫЕ =====
        
        def create_contradiction_manager():
            try:
                from eva.contradiction.contradiction_manager import ContradictionManager
                contradiction_manager = ContradictionManager(brain=self.core_brain)
                if hasattr(contradiction_manager, 'initialize'):
                    contradiction_manager.initialize()
                self.core_brain.contradiction_manager = contradiction_manager
                self.logger.info("[OK] ContradictionManager создан")
                return contradiction_manager
            except Exception as e:
                self.logger.error(f"[FAIL] Ошибка создания contradiction_manager: {e}", exc_info=True)
                self.failed_components.add('contradiction_manager')
                return None
        
        def create_adaptation_manager():
            try:
                from eva.adaptation.adaptation_core import AdaptationManager
                adaptation_manager = AdaptationManager(brain=self.core_brain)
                if hasattr(adaptation_manager, 'initialize'):
                    adaptation_manager.initialize()
                self.core_brain.adaptation_manager = adaptation_manager
                self.logger.info("[OK] AdaptationManager создан")
                return adaptation_manager
            except Exception as e:
                self.logger.error(f"[FAIL] Ошибка создания adaptation_manager: {e}", exc_info=True)
                self.failed_components.add('adaptation_manager')
                return None
        
        def create_ethics_framework():
            try:
                from eva.ethics.ethics_framework import EthicsFramework
                ethics_framework = EthicsFramework(brain=self.core_brain)
                self.core_brain.ethics_framework = ethics_framework
                self.logger.info("[OK] EthicsFramework создан")
                return ethics_framework
            except Exception as e:
                self.logger.error(f"[FAIL] Ошибка создания ethics_framework: {e}", exc_info=True)
                self.failed_components.add('ethics_framework')
                return None
        
        def create_gui():
            try:
                from eva.gui.core_gui import ЕВАGUI
                
                # Создаем основной GUI с brain
                gui = ЕВАGUI(brain=self.core_brain)
                self.core_brain.gui = gui
                self.logger.info("[OK] ЕВАGUI создан")
                return gui
            except Exception as e:
                self.logger.error(f"[FAIL] Ошибка создания gui: {e}", exc_info=True)
                self.failed_components.add('gui')
                return None
        
        def create_web_search_engine():
            try:
                from eva.websearch.web_search_engine import WebSearchEngine
                web_search = WebSearchEngine(brain=self.core_brain)
                self.core_brain.web_search_engine = web_search
                self.logger.info("[OK] WebSearchEngine создан")
                return web_search
            except Exception as e:
                self.logger.error(f"[FAIL] Ошибка создания web_search_engine: {e}", exc_info=True)
                self.failed_components.add('web_search_engine')
                return None
        
        # ===== FRACTAL REASONING КОМПОНЕНТЫ =====
        
        def create_fractal_storage():
            """Создает FractalStorage для хранения цепочек рассуждений."""
            try:
                try:
                    from eva.reasoning.fractal_ml import FractalStorage
                except ImportError:
                    self.logger.warning("[WARN] FractalStorage не найден в fractal_ml - пропускаем")
                    self.failed_components.add('fractal_storage')
                    return None
                
                storage_dir = os.path.join(
                    getattr(self.core_brain, 'cache_dir', './cache'),
                    'fractal_reasoning'
                )
                os.makedirs(storage_dir, exist_ok=True)
                
                fractal_storage = FractalStorage(storage_dir=storage_dir)
                
                # Регистрируем в core_brain
                self.core_brain.fractal_storage = fractal_storage
                
                # Регистрируем в components если есть
                if hasattr(self.core_brain, 'components'):
                    self.core_brain.components['fractal_storage'] = fractal_storage
                
                self.logger.info(f"[OK] FractalStorage создан: {storage_dir}")
                return fractal_storage
            except Exception as e:
                self.logger.error(f"[FAIL] Ошибка создания fractal_storage: {e}", exc_info=True)
                self.failed_components.add('fractal_storage')
                return None
        
        def create_self_reasoning_engine():
            """Создает Self-Reasoning Engine для саморассуждений с поддержкой рекурсии."""
            try:
                try:
                    from eva.reasoning import SelfReasoningEngine
                except ImportError:
                    self.logger.warning("[WARN] SelfReasoningEngine не найден в eva.reasoning - пропускаем")
                    self.failed_components.add('self_reasoning_engine')
                    return None
                
                try:
                    from eva.reasoning.integration import ReasoningIntegration
                except ImportError:
                    ReasoningIntegration = None
                    self.logger.debug("ReasoningIntegration не найден - интеграция недоступна")
                
                # Получаем конфигурацию из brain_config
                reasoning_config = self.core_brain.config.get('reasoning', {}) if hasattr(self.core_brain, 'config') else {}
                
                # Получаем fractal_storage
                fractal_storage = getattr(self.core_brain, 'fractal_storage', None)
                
                # Создаем движок рассуждений с рекурсивной поддержкой
                self_reasoning_engine = SelfReasoningEngine(
                    brain=self.core_brain,
                    config={
                        'max_iterations': reasoning_config.get('max_iterations', 5),
                        'confidence_threshold': reasoning_config.get('confidence_threshold', 0.75),
                        'max_recursion_depth': reasoning_config.get('max_recursion_depth', 3)
                    }
                )
                
                # Подключаем fractal_storage если есть
                if fractal_storage:
                    self_reasoning_engine.fractal_storage = fractal_storage
                    self_reasoning_engine._init_retriever()
                
                # Также пробуем интеграцию через ReasoningIntegration
                if ReasoningIntegration:
                    try:
                        reasoning_integration = ReasoningIntegration(self.core_brain)
                        reasoning_integration.reasoning_engine = self_reasoning_engine
                        reasoning_integration.enabled = True
                        self.core_brain.reasoning_integration = reasoning_integration
                        self.logger.info("[OK] ReasoningIntegration также создан")
                    except Exception as e:
                        self.logger.debug(f"ReasoningIntegration не создан: {e}")
                
                # Регистрируем в core_brain
                self.core_brain.self_reasoning_engine = self_reasoning_engine
                
                self.logger.info("[OK] SelfReasoningEngine создан с рекурсивной поддержкой")
                return self_reasoning_engine
            except Exception as e:
                self.logger.error(f"[FAIL] Ошибка создания self_reasoning_engine: {e}", exc_info=True)
                self.failed_components.add('self_reasoning_engine')
                return None
        
        # Регистрируем все фабрики
        self.component_factories = {
            # Системные
            'event_bus': create_event_bus,
            'resource_manager': create_resource_manager,
            'config_manager': create_config_manager,
            # Память
            'memory_manager': create_memory_manager,
            'hybrid_cache': create_hybrid_cache,
            # Знания
            'knowledge_graph': create_knowledge_graph,
            'qwen_api_enhancer': create_qwen_api_enhancer,
            'text_processor': create_text_processor,
            # ML
            'ml_unit': create_ml_unit,
            'model_manager': create_model_manager,
            # Логика
            'query_processor': create_query_processor,
            'response_generator': create_response_generator,
            'reasoning_engine': create_reasoning_engine,
            # Обучение
            'training_orchestrator': create_training_orchestrator,
            'learning_manager': create_learning_manager,
            'learning_scheduler': create_learning_scheduler,
            # Аналитика
            'analytics_manager': create_analytics_manager,
            'system_monitor': create_system_monitor,
            'metrics_collector': create_metrics_collector,
            # Специализированные
            'contradiction_manager': create_contradiction_manager,
            'adaptation_manager': create_adaptation_manager,
            'ethics_framework': create_ethics_framework,
            'web_search_engine': create_web_search_engine,
            'gui': create_gui,
            
            # Fractal Reasoning
            'fractal_storage': create_fractal_storage,
            'self_reasoning_engine': create_self_reasoning_engine,
        }
        
        self.logger.info(f"Зарегистрировано {len(self.component_factories)} фабрик компонентов")
    
    def _check_dependencies(self, component_name: str) -> Tuple[bool, List[str]]:
        """
        Проверяет, инициализированы ли все зависимости компонента.
        
        Args:
            component_name: Имя компонента для проверки
            
        Returns:
            Tuple[bool, List[str]]: (успех, список отсутствующих зависимостей)
        """
        dependencies = self.component_dependencies.get(component_name, [])
        missing = []
        
        for dep in dependencies:
            if dep not in self.initialized_components:
                missing.append(dep)
        
        if missing:
            return False, missing
        return True, []
    
    # Опциональные компоненты - их отсутствие не блокирует запуск системы
    OPTIONAL_COMPONENTS = {
        'qwen_api_enhancer',
        'web_search_engine',
        'gui',
    }
    
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
            
            components_to_init = component_list or self.COMPONENT_LIST
            
            success_count = 0
            failed_count = 0
            skipped_count = 0
            
            # Последовательная инициализация с проверкой зависимостей
            for component_name in components_to_init:
                if component_name not in self.component_factories:
                    self.logger.warning(f"[WARN] Фабрика для {component_name} не найдена - пропущено")
                    skipped_count += 1
                    continue
                
                # Проверяем зависимости
                deps_ok, missing_deps = self._validate_dependencies(component_name)
                if not deps_ok:
                    self.logger.error(f"[FAIL] {component_name}: зависимости не готовы - {missing_deps}")
                    self.failed_components.add(component_name)
                    failed_count += 1
                    continue
                
                try:
                    # Создаем компонент через фабрику
                    factory = self.component_factories[component_name]
                    component = factory()

                    if component is not None:
                        # Set up event subscriptions if the component has the method
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
                            # Components with BaseComponent base class - initialize handles event setup
                            try:
                                if hasattr(component, 'is_initialized'):
                                    if not component.is_initialized:
                                        component.initialize()
                            except Exception:
                                pass
                        
                        if not event_integration_ok:
                            self.logger.warning(f"   [WARN] Неполная интеграция с системой событий для {component_name}")
                        
                        # Publish component initialized event
                        try:
                            event_bus = getattr(self.core_brain, 'event_bus', None)
                            if event_bus is not None:
                                # Import here to avoid circular imports
                                from eva.core.event_bus import EventTypes, Event
                                event = Event(
                                    event_type=EventTypes.COMPONENT_INITIALIZED,
                                    source=component_name,
                                    data={'component_name': component_name}
                                )
                                event_bus.publish_sync(event)
                        except Exception as e:
                            self.logger.debug(f"   └─ Failed to publish COMPONENT_INITIALIZED event for {component_name}: {e}")
                        # Регистрируем успешно инициализированный компонент
                        self.initialized_components.add(component_name)
                        self.component_instances[component_name] = component
                        self.core_brain.components[component_name] = component
                        self.logger.info(f"[OK] {component_name} инициализирован")
                        success_count += 1
                    else:
                        self.logger.error(f"[FAIL] {component_name} не был создан")
                        self.failed_components.add(component_name)
                        failed_count += 1
                        
                except Exception as e:
                    self.logger.error(f"[FAIL] {component_name}: {e}", exc_info=True)
                    self.failed_components.add(component_name)
                    failed_count += 1
            
            # Пост-инициализация связей
            if success_count > 0:
                try:
                    self.post_initialize_connections()
                    self.logger.info("[OK] Пост-инициализация связей выполнена")
                except Exception as e:
                    self.logger.error(f"[FAIL] Ошибка пост-инициализации: {e}")
            
            # Итоговый отчет
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
            
            # Проверяем только обязательные компоненты
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
        try:
            self.logger.info("Установка связей между компонентами...")
            
            # ===== Гибридный кэш =====
            self.logger.info(f"Проверка гибридного кэша в brain.components: {hasattr(self.core_brain, 'components')}")
            if hasattr(self.core_brain, 'components'):
                self.logger.info(f"Ключи в brain.components: {list(self.core_brain.components.keys())}")
            
            hybrid_cache = self.core_brain.components.get('hybrid_cache')
            self.logger.info(f"hybrid_cache из brain.components: {hybrid_cache}")
            
            if hybrid_cache is not None:
                self.core_brain.hybrid_cache = hybrid_cache
                self.logger.info("Установка гибридного кэша в brain.hybrid_cache")
                
                # Подключаем ко всем компонентам, которые могут его использовать
                components_to_connect = [
                    ('memory_manager', 'memory_manager'),
                    ('ml_unit', 'ml_unit'),
                    ('text_processor', 'text_processor'),
                    ('model_manager', 'model_manager'),
                    ('query_processor', 'query_processor'),
                    ('response_generator', 'response_generator')
                ]
                
                for comp_name, comp_key in components_to_connect:
                    # Ищем компонент в component_instances, а не в brain.components
                    component = self.component_instances.get(comp_key)
                    if component is not None:
                        # Принудительно устанавливаем гибридный кэш
                        if hasattr(component, 'hybrid_cache'):
                            old_cache = component.hybrid_cache
                            component.hybrid_cache = hybrid_cache
                            
                            if old_cache is hybrid_cache:
                                self.logger.info(f"   [OK] {comp_name}: уже был подключен")
                            elif old_cache is not None:
                                self.logger.info(f"   [UPD] {comp_name}: заменен старый кэш")
                            else:
                                self.logger.info(f"   └─ hybrid_cache → {comp_name}")
                        else:
                            # Некоторые компоненты могут поддерживать установку кэша через атрибут
                            try:
                                setattr(component, 'hybrid_cache', hybrid_cache)
                                self.logger.info(f"   └─ hybrid_cache → {comp_name} (через setattr)")
                            except Exception as e:
                                self.logger.debug(f"   [WARN] {comp_name}: не удалось установить hybrid_cache: {e}")
                    else:
                        self.logger.debug(f"   [WARN] {comp_name}: компонент не найден")
            
            # ===== ML Unit связи =====
            ml_unit = self.component_instances.get('ml_unit')
            if ml_unit is not None:
                model_manager = self.component_instances.get('model_manager')
                if model_manager is not None and hasattr(ml_unit, 'model_manager'):
                    ml_unit.model_manager = model_manager
                    self.logger.info("   └─ ml_unit → model_manager")
                
                text_processor = self.component_instances.get('text_processor')
                if text_processor is not None and hasattr(ml_unit, 'text_processor'):
                    ml_unit.text_processor = text_processor
                    self.logger.info("   └─ ml_unit → text_processor")
            
            # ===== Response Generator связи =====
            response_generator = self.component_instances.get('response_generator')
            if response_generator is not None:
                model_manager = self.component_instances.get('model_manager')
                if model_manager is not None and hasattr(response_generator, 'model_manager'):
                    response_generator.model_manager = model_manager
                    self.logger.info("   └─ response_generator → model_manager")
                
                reasoning_engine = self.component_instances.get('reasoning_engine')
                if reasoning_engine is not None and hasattr(response_generator, 'reasoning_engine'):
                    response_generator.reasoning_engine = reasoning_engine
                    self.logger.info("   └─ response_generator → reasoning_engine")
            
            self.logger.info("[OK] Все связи установлены")
            
        except Exception as e:
            self.logger.error(f"[FAIL] Ошибка установки связей: {e}", exc_info=True)
            raise
    
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
        """
        Возвращает статус инициализации компонентов.
        
        Returns:
            Dict: Статус инициализации
        """
        total = len(self.component_factories)
        initialized = len(self.initialized_components)
        failed = len(self.failed_components)
        
        return {
            'initialized': list(self.initialized_components),
            'failed': list(self.failed_components),
            'total_factories': total,
            'success_count': initialized,
            'failed_count': failed,
            'success_rate': initialized / max(1, total),
            'components': {
                name: {
                    'initialized': name in self.initialized_components,
                    'instance': self.component_instances.get(name),
                    'dependencies': self.component_dependencies.get(name, [])
                }
                for name in self.component_factories.keys()
            }
        }
    
    def retry_failed_components(self) -> Dict[str, bool]:
        """
        Повторяет инициализацию неудачных компонентов.
        
        Returns:
            Dict: Результаты повторной инициализации
        """
        results = {}
        
        for component_name in list(self.failed_components):
            if component_name not in self.component_factories:
                continue
            
            try:
                self.logger.info(f"Повторная инициализация: {component_name}")
                
                deps_ok, _ = self._check_dependencies(component_name)
                if not deps_ok:
                    self.logger.warning(f"[WARN] Зависимости не готовы для {component_name}")
                    results[component_name] = False
                    continue
                
                factory = self.component_factories[component_name]
                component = factory()
                
                if component:
                    self.initialized_components.add(component_name)
                    self.failed_components.discard(component_name)
                    self.component_instances[component_name] = component
                    results[component_name] = True
                    self.logger.info(f"[OK] {component_name} инициализирован")
                else:
                    results[component_name] = False
                    
            except Exception as e:
                self.logger.error(f"[FAIL] Ошибка повторной инициализации {component_name}: {e}")
                results[component_name] = False
        
        return results
    
    def shutdown_components(self):
        """Корректно завершает работу всех компонентов."""
        if getattr(self, '_shutdown_complete', False):
            self.logger.debug("Компоненты уже завершили работу")
            return
        
        self.logger.info("=" * 60)
        self.logger.info("ЗАВЕРШЕНИЕ РАБОТЫ КОМПОНЕНТОВ")
        self.logger.info("=" * 60)
        
        # Обратный порядок для корректного завершения
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
        """
        Возвращает информацию о здоровье компонента.
        
        Args:
            component_name: Имя компонента
            
        Returns:
            Dict: Информация о здоровье
        """
        component = self.component_instances.get(component_name)
        
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
            'dependencies': self.component_dependencies.get(component_name, [])
        }
        
        # Проверяем зависимости
        for dep in health_info['dependencies']:
            if dep not in self.initialized_components:
                health_info['healthy'] = False
                health_info['missing_dependency'] = dep
                break
        
        # Проверяем состояние компонента
        if hasattr(component, 'is_healthy'):
            health_info['healthy'] = component.is_healthy()
        elif hasattr(component, 'health_check'):
            health_info['health_status'] = component.health_check()
        
        return health_info
    
    def get_all_component_health(self) -> Dict[str, Dict[str, Any]]:
        """
        Возвращает информацию о здоровье всех компонентов.
        
        Returns:
            Dict: Информация о здоровье всех компонентов
        """
        return {
            name: self.get_component_health(name)
            for name in self.component_factories.keys()
        }


# ============================================================================
# Тестирование
# ============================================================================

if __name__ == "__main__":
    """Тестирование ComponentInitializer."""
    
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ ComponentInitializer")
    print("=" * 60)
    
    # Создаем mock CoreBrain для тестирования
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