"""
init_factories.py - Factory functions for creating EVA system components.
Contains all create_* functions for EventBus, ResourceManager, MemoryManager, etc.
"""
import os
import logging

logger = logging.getLogger("eva_ai.component_initializer.factories")


def _ensure_eva_path():
    """Ensure EVA path is in sys.path using dynamic path detection."""
    import sys
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


def create_event_bus(initializer):
    try:
        from eva_ai.core.event_bus import EventBus
        event_bus = EventBus()
        initializer.core_brain.event_bus = event_bus
        initializer.logger.info("[OK] EventBus создан")
        return event_bus
    except Exception as e:
        initializer.logger.error(f"[FAIL] Ошибка создания event_bus: {e}", exc_info=True)
        initializer.failed_components.add('event_bus')
        return None


def create_resource_manager(initializer):
    try:
        from eva_ai.core.resource_manager import ResourceManager
        config_manager = getattr(initializer.core_brain, 'config_manager', None)
        resource_manager = ResourceManager(config_manager=config_manager)
        initializer.core_brain.resource_manager = resource_manager
        initializer.logger.info("[OK] ResourceManager создан")
        return resource_manager
    except Exception as e:
        initializer.logger.error(f"[FAIL] Ошибка создания resource_manager: {e}", exc_info=True)
        initializer.failed_components.add('resource_manager')
        return None


def create_config_manager(initializer):
    try:
        from eva_ai.core.config_manager import ConfigManager
        config_manager = ConfigManager()
        initializer.core_brain.config_manager = config_manager
        initializer.logger.info("[OK] ConfigManager создан")
        return config_manager
    except Exception as e:
        initializer.logger.error(f"[FAIL] Ошибка создания config_manager: {e}", exc_info=True)
        initializer.failed_components.add('config_manager')
        return None


def create_memory_manager(initializer):
    _ensure_eva_path()
    try:
        # Проверяем, не создан ли уже
        existing = getattr(initializer.core_brain, 'memory_manager', None)
        if existing is not None:
            initializer.logger.info(f"Используем существующий MemoryManager: {id(existing)}")
            initializer.memory_manager = existing
            initializer.core_brain.memory_manager = existing
            return existing
        
        # Проверяем, не создан ли уже в component_instances
        if hasattr(initializer, 'component_instances'):
            existing = initializer.component_instances.get('memory_manager')
            if existing is not None:
                initializer.logger.info(f"Используем существующий MemoryManager из component_instances: {id(existing)}")
                initializer.memory_manager = existing
                initializer.core_brain.memory_manager = existing
                return existing
        
        # Simple import approach - используем тот же подход что и в memory_initializer.py
        try:
            from eva_ai.memory.memory_manager import MemoryManager
        except ImportError as e:
            initializer.logger.warning(f"Не удалось импортировать MemoryManager: {e}")
            # Skip this component - memory_manager будет инициализирован в другом месте
            initializer.logger.info("[SKIP] MemoryManager будет инициализирован в memory_initializer")
            return None
        
        initializer.logger.info("MemoryManager импортирован успешно")
        
        cache_dir = os.path.join(
            getattr(initializer.core_brain, 'cache_dir', './cache'),
            'memory'
        )
        os.makedirs(cache_dir, exist_ok=True)
        
        event_bus = getattr(initializer.core_brain, 'event_bus', None)
        deferred_system = getattr(initializer.core_brain, 'deferred_system', None)
        fractal_graph_v2 = getattr(initializer.core_brain, 'fractal_graph_v2', None)
        
        memory_manager = MemoryManager(
            brain=initializer.core_brain,
            cache_dir=cache_dir,
            event_bus=event_bus,
            deferred_system=deferred_system,
            fractal_graph_v2=fractal_graph_v2
        )
        if hasattr(memory_manager, 'initialize'):
            init_result = memory_manager.initialize()
            if init_result is False:
                initializer.logger.warning("[WARN] MemoryManager.initialize() вернул False - возможны проблемы")
        initializer.memory_manager = memory_manager
        initializer.core_brain.memory_manager = memory_manager
        initializer.logger.info("[OK] MemoryManager создан")
        return memory_manager
    except Exception as e:
        initializer.logger.error(f"[FAIL] Ошибка создания memory_manager: {e}", exc_info=True)
        initializer.failed_components.add('memory_manager')
        return None


def create_hybrid_cache(initializer):
    _ensure_eva_path()
    try:
        # Проверяем, не создан ли уже
        existing = getattr(initializer.core_brain, 'token_cache', None) or getattr(initializer.core_brain, 'hybrid_cache', None)
        if existing is not None:
            initializer.logger.info(f"Используем существующий HybridTokenCache: {id(existing)}")
            hybrid_cache = existing
        else:
            initializer.logger.info("Начало создания HybridTokenCache...")
            initializer.logger.debug(f"core_brain тип: {type(initializer.core_brain)}")
            initializer.logger.debug(f"core_brain атрибуты: {dir(initializer.core_brain)}")

            # Try multiple import approaches
            get_shared_cache = None
            try:
                from eva_ai.memory.hybrid_token_cache import get_shared_cache
            except ImportError:
                try:
                    from eva_ai.memory import get_shared_cache
                except ImportError:
                    pass
            
            if get_shared_cache is None:
                raise RuntimeError("Cannot import get_shared_cache")
            
            initializer.logger.info("Импортирован get_shared_cache")
            hybrid_cache = get_shared_cache(initializer.core_brain, "default")
            initializer.logger.info(f"Создан/получен синглтон HybridTokenCache: {id(hybrid_cache)}")

        if hybrid_cache is None:
            initializer.logger.error("HybridTokenCache вернул None!")
            return None

        if not hasattr(initializer.core_brain, 'components'):
            initializer.core_brain.components = {}
        initializer.core_brain.components['hybrid_cache'] = hybrid_cache
        initializer.logger.info("Добавлен в brain.components")

        initializer.core_brain.token_cache = hybrid_cache
        initializer.core_brain.hybrid_cache = hybrid_cache
        initializer.logger.info("Установлены обратные ссылки")

        initializer.logger.info(f"[OK] HybridTokenCache создан: {id(hybrid_cache)}")
        return hybrid_cache

    except Exception as e:
        initializer.logger.error(f"[FAIL] Ошибка создания hybrid_cache: {e}", exc_info=True)
        initializer.failed_components.add('hybrid_cache')
        return None


def create_qwen_api_enhancer(initializer):
    """Создает QwenAPIEnhancer для обогащения знаний"""
    try:
        from eva_ai.knowledge.qwen_api_enhancer import QwenAPIEnhancer

        api_key = os.environ.get('OPENROUTER_API_KEY', '')
        enhancer = QwenAPIEnhancer(api_key=api_key)

        initializer.core_brain.qwen_api_enhancer = enhancer
        initializer.logger.info(f"[OK] QwenAPIEnhancer создан")
        return enhancer
    except Exception as e:
        initializer.logger.error(f"[FAIL] Ошибка создания qwen_api_enhancer: {e}", exc_info=True)
        initializer.failed_components.add('qwen_api_enhancer')
        return None


def create_text_processor(initializer):
    try:
        _ensure_eva_path()
        from eva_ai.mlearning.unified_text_processor import UnifiedTextProcessor
        text_processor = UnifiedTextProcessor(brain=initializer.core_brain)
        hybrid_cache = getattr(initializer.core_brain, 'hybrid_cache', None)
        if hybrid_cache:
            text_processor.hybrid_cache = hybrid_cache
            initializer.logger.info("   +-- Hybrid cache podklyuchen")
        text_processor.initialize()
        text_processor._setup_component()
        initializer.core_brain.text_processor = text_processor
        initializer.logger.info("[OK] TextProcessor sozdan")
        return text_processor
    except Exception as e:
        initializer.logger.error(f"[FAIL] Ошибка создания text_processor: {e}", exc_info=True)
        initializer.failed_components.add('text_processor')
        return None


def create_ml_unit(initializer):
    try:
        _ensure_eva_path()
        from eva_ai.mlearning.ml_unit import MLUnit
        cache_dir = os.path.join(
            getattr(initializer.core_brain, 'cache_dir', './cache'),
            'ml_unit'
        )
        os.makedirs(cache_dir, exist_ok=True)
        ml_unit = MLUnit(
            brain=initializer.core_brain,
            cache_dir=cache_dir,
            max_workers=4
        )
        if hasattr(ml_unit, 'initialize'):
            ml_unit.initialize()
        hybrid_cache = getattr(initializer.core_brain, 'hybrid_cache', None)
        if hybrid_cache:
            ml_unit.hybrid_cache = hybrid_cache
            initializer.logger.info("   └─ Гибридный кэш подключен")
        initializer.ml_unit = ml_unit
        initializer.core_brain.ml_unit = ml_unit
        initializer.logger.info("[OK] MLUnit создан")
        return ml_unit
    except Exception as e:
        initializer.logger.error(f"[FAIL] Ошибка создания ml_unit: {e}", exc_info=True)
        initializer.failed_components.add('ml_unit')
        return None


def create_model_manager(initializer):
    """HybridModelManager removed - using UnifiedGenerator/FractalModelManager only"""
    initializer.logger.info("HybridModelManager disabled - using UnifiedGenerator pipeline")
    initializer.failed_components.add('model_manager')
    return None


def create_query_processor(initializer):
    try:
        from eva_ai.core.query_processor import QueryProcessor
        query_processor = QueryProcessor(brain=initializer.core_brain)
        if hasattr(query_processor, 'initialize'):
            init_result = query_processor.initialize()
            if init_result is False:
                initializer.logger.warning("[WARN] QueryProcessor.initialize() вернул False")
        initializer.core_brain.query_processor = query_processor
        initializer.logger.info("[OK] QueryProcessor создан")
        return query_processor
    except Exception as e:
        initializer.logger.error(f"[FAIL] Ошибка создания query_processor: {e}", exc_info=True)
        initializer.failed_components.add('query_processor')
        return None


def create_response_generator(initializer):
    try:
        from eva_ai.core.response_generator import ResponseGenerator
        response_generator = ResponseGenerator(brain=initializer.core_brain)
        if hasattr(response_generator, 'initialize'):
            response_generator.initialize()
        initializer.core_brain.response_generator = response_generator
        initializer.logger.info("[OK] ResponseGenerator создан")
        return response_generator
    except Exception as e:
        initializer.logger.error(f"[FAIL] Ошибка создания response_generator: {e}", exc_info=True)
        initializer.failed_components.add('response_generator')
        return None


def create_reasoning_engine(initializer):
    """ReasoningEngine removed - using SelfReasoningEngine only"""
    initializer.logger.info("ReasoningEngine disabled - using SelfReasoningEngine")
    initializer.failed_components.add('reasoning_engine')
    return None


def create_analytics_manager(initializer):
    try:
        from eva_ai.analytics.analytics_manager import AnalyticsManager
        analytics_manager = AnalyticsManager(brain=initializer.core_brain)
        if hasattr(analytics_manager, 'initialize'):
            analytics_manager.initialize()
        initializer.core_brain.analytics_manager = analytics_manager
        initializer.logger.info("[OK] AnalyticsManager создан")
        return analytics_manager
    except Exception as e:
        initializer.logger.error(f"[FAIL] Ошибка создания analytics_manager: {e}", exc_info=True)
        initializer.failed_components.add('analytics_manager')
        return None


def create_system_monitor(initializer):
    try:
        from eva_ai.monitoring.system_monitor import SystemMonitor
        system_monitor = SystemMonitor()
        if hasattr(system_monitor, 'initialize'):
            system_monitor.initialize()
        initializer.core_brain.system_monitor = system_monitor
        initializer.logger.info("[OK] SystemMonitor создан")
        return system_monitor
    except Exception as e:
        initializer.logger.error(f"[FAIL] Ошибка создания system_monitor: {e}", exc_info=True)
        initializer.failed_components.add('system_monitor')
        return None


def create_metrics_collector(initializer):
    try:
        try:
            from eva_ai.core.metrics_collector import MetricsCollector
        except ImportError:
            from eva_ai.analytics.analytics_manager import AnalyticsManager as MetricsCollector
        metrics_collector = MetricsCollector(brain=initializer.core_brain)
        if hasattr(metrics_collector, 'initialize'):
            init_result = metrics_collector.initialize()
            if init_result is False:
                initializer.logger.warning("[WARN] MetricsCollector.initialize() вернул False")
        initializer.core_brain.metrics_collector = metrics_collector
        initializer.logger.info("[OK] MetricsCollector создан")
        return metrics_collector
    except Exception as e:
        initializer.logger.error(f"[FAIL] Ошибка создания metrics_collector: {e}", exc_info=True)
        initializer.failed_components.add('metrics_collector')
        return None


def create_contradiction_manager(initializer):
    try:
        from eva_ai.contradiction.contradiction_manager import ContradictionManager
        contradiction_manager = ContradictionManager(brain=initializer.core_brain)
        if hasattr(contradiction_manager, 'initialize'):
            contradiction_manager.initialize()
        initializer.core_brain.contradiction_manager = contradiction_manager
        initializer.logger.info("[OK] ContradictionManager создан")
        return contradiction_manager
    except Exception as e:
        initializer.logger.error(f"[FAIL] Ошибка создания contradiction_manager: {e}", exc_info=True)
        initializer.failed_components.add('contradiction_manager')
        return None


def create_adaptation_manager(initializer):
    try:
        from eva_ai.adaptation.adaptation_core import AdaptationManager
        adaptation_manager = AdaptationManager(brain=initializer.core_brain)
        if hasattr(adaptation_manager, 'initialize'):
            adaptation_manager.initialize()
        initializer.core_brain.adaptation_manager = adaptation_manager
        initializer.logger.info("[OK] AdaptationManager создан")
        return adaptation_manager
    except Exception as e:
        initializer.logger.error(f"[FAIL] Ошибка создания adaptation_manager: {e}", exc_info=True)
        initializer.failed_components.add('adaptation_manager')
        return None


def create_ethics_framework(initializer):
    try:
        from eva_ai.ethics.ethics_framework import EthicsFramework
        event_bus = getattr(initializer.core_brain, 'event_bus', None) or getattr(initializer.core_brain, '_new_event_bus', None)
        ethics_framework = EthicsFramework(brain=initializer.core_brain, event_bus=event_bus)
        initializer.core_brain.ethics_framework = ethics_framework
        initializer.logger.info("[OK] EthicsFramework создан")
        return ethics_framework
    except Exception as e:
        initializer.logger.error(f"[FAIL] Ошибка создания ethics_framework: {e}", exc_info=True)
        initializer.failed_components.add('ethics_framework')
        return None


def create_web_search_engine(initializer):
    try:
        from eva_ai.websearch.web_search_integrated import IntegratedWebSearchEngine
        web_search = IntegratedWebSearchEngine(brain=initializer.core_brain)
        
        # Принудительная инициализация
        if hasattr(web_search, 'initialize') and not web_search.is_initialized:
            try:
                web_search.initialize()
                initializer.logger.info("[OK] IntegratedWebSearchEngine инициализирован")
            except Exception as init_err:
                initializer.logger.warning(f"[WARN] Не удалось инициализировать web_search_engine: {init_err}")
        
        initializer.core_brain.web_search_engine = web_search
        initializer.logger.info("[OK] WebSearchEngine с Tavily создан")
        return web_search
    except Exception as e:
        initializer.logger.error(f"[FAIL] Ошибка создания web_search_engine: {e}", exc_info=True)
        initializer.failed_components.add('web_search_engine')
        return None


def create_fractal_graph_v2(initializer):
    """Получает или создаёт fractal_graph_v2."""
    try:
        existing = getattr(initializer.core_brain, 'fractal_graph_v2', None)
        if existing is not None:
            initializer.logger.info(f"Используем существующий fractal_graph_v2: {id(existing)}")
            return existing
        
        from eva_ai.memory.fractal_graph_v2 import FractalMemoryGraph
        config = initializer.core_brain.config.get('fractal_graph_v2', {}) if hasattr(initializer.core_brain, 'config') else {}
        
        embedding_device = config.get('embedding_device', None)
        if embedding_device is None:
            try:
                import torch
                embedding_device = 'cuda' if torch.cuda.is_available() else 'cpu'
                initializer.logger.info(f"Auto-detected embedding device: {embedding_device}")
            except ImportError:
                embedding_device = 'cpu'
                initializer.logger.info("PyTorch not available, using CPU for embeddings")
        
        event_bus = getattr(initializer.core_brain, 'event_bus', None)
        
        fg = FractalMemoryGraph(
            storage_dir=config.get('storage_dir'),
            embedding_device=embedding_device,
            event_bus=event_bus
        )
        initializer.core_brain.fractal_graph_v2 = fg
        
        if hasattr(initializer.core_brain, 'components'):
            initializer.core_brain.components['fractal_graph_v2'] = fg
        
        initializer.logger.info("[OK] FractalGraphV2 создан")
        
        _load_model_into_graph(fg, initializer)
        
        return fg
    except Exception as e:
        initializer.logger.error(f"[FAIL] Ошибка создания fractal_graph_v2: {e}")
        initializer.failed_components.add('fractal_graph_v2')
        return None


def _load_model_into_graph(fg, initializer):
    """Загрузить данные модели в граф памяти."""
    try:
        model_config = initializer.core_brain.config.get('model', {}) if hasattr(initializer.core_brain, 'config') else {}
        
        model_paths = [
            model_config.get('logic_model_path', ''),
            model_config.get('context_model_path', ''),
            model_config.get('model_a_gguf_path', '')
        ]
        
        # Дедупликация путей
        seen = set()
        unique_paths = []
        for p in model_paths:
            if p and p not in seen:
                seen.add(p)
                unique_paths.append(p)
        
        from eva_ai.memory.fractal_graph_v2.gguf_parser import extract_to_graph, clear_and_reload_model_graph
        
        for model_path in unique_paths:
            if model_path and os.path.exists(model_path):
                initializer.logger.info(f"Загрузка данных модели в граф: {model_path}")
                
                try:
                    result = extract_to_graph(model_path, fg)
                    initializer.logger.info(f"Добавлено узлов модели: {result.get('nodes_added', 0)}")
                except Exception as e:
                    initializer.logger.warning(f"Ошибка загрузки модели {model_path}: {e}")
        
        initializer.logger.info("[OK] Данные модели загружены в граф")
    except Exception as e:
        initializer.logger.warning(f"Не удалось загрузить модель в граф: {e}")


def create_knowledge_components(initializer):
    """
    Создаёт компоненты работы с знаниями на основе FractalGraph v2.
    БЕЗ KG адаптера - напрямую используем FGv2.
    """
    try:
        # Получаем FGv2
        fg = getattr(initializer.core_brain, 'fractal_graph_v2', None)
        if fg is None:
            components = getattr(initializer.core_brain, 'components', {})
            fg = components.get('fractal_graph_v2')
        
        if fg is None:
            initializer.logger.warning("[WARN] FGv2 не найден, компоненты знаний не созданы")
            return None
        
        # Создаём ConceptExtractor для извлечения концептов из запросов
        try:
            from eva_ai.knowledge.concept_extractor import create_concept_extractor
            concept_extractor = create_concept_extractor(
                fractal_graph=fg,
                brain=initializer.core_brain
            )
            initializer.core_brain.concept_extractor = concept_extractor
            initializer.core_brain.components['concept_extractor'] = concept_extractor
            initializer.logger.info("[OK] ConceptExtractor создан")
        except Exception as ce:
            initializer.logger.warning(f"[WARN] ConceptExtractor не создан: {ce}")
        
        # Создаём ContradictionGenerator для генерации противоречий (шаблоны)
        try:
            from eva_ai.contradiction.contradiction_generator import create_contradiction_generator
            contr_generator = create_contradiction_generator(
                brain=initializer.core_brain,
                fractal_graph=fg
            )
            initializer.core_brain.contradiction_generator = contr_generator
            initializer.core_brain.components['contradiction_generator'] = contr_generator
            initializer.logger.info("[OK] ContradictionGenerator создан (шаблоны)")
        except Exception as cge:
            initializer.logger.warning(f"[WARN] ContradictionGenerator не создан: {cge}")
        
        # Создаём ContradictionMiner для обнаружения противоречий в графе (анализ)
        try:
            from eva_ai.contradiction.contradiction_miner import create_contradiction_miner
            
            event_bus = getattr(initializer.core_brain, 'event_bus', None) or getattr(initializer.core_brain, '_new_event_bus', None)
            deferred_system = getattr(initializer.core_brain, 'deferred_system', None)
            
            contradiction_miner = create_contradiction_miner(
                brain=initializer.core_brain,
                event_bus=event_bus,
                deferred_system=deferred_system,
                config={
                    'enabled': True,
                    'dry_run': False,
                    'sim_threshold': 0.75,
                    'contra_threshold': 0.65,
                    'max_candidates_per_cycle': 3
                }
            )
            
            initializer.core_brain.contradiction_miner = contradiction_miner
            initializer.core_brain.components['contradiction_miner'] = contradiction_miner
            
            # Запускаем
            contradiction_miner.start()
            
            initializer.logger.info("[OK] ContradictionMiner создан и запущен (анализ графа)")
        except Exception as cme:
            initializer.logger.warning(f"[WARN] ContradictionMiner не создан: {cme}")
        
         # Создаём ConceptMiner для глубокого анализа кластеров
        try:
            from eva_ai.knowledge.concept_miner import create_concept_miner
            
            # Получаем необходимые компоненты
            event_bus = getattr(initializer.core_brain, 'event_bus', None) or getattr(initializer.core_brain, '_new_event_bus', None)
            deferred_system = getattr(initializer.core_brain, 'deferred_system', None)
            
            concept_miner = create_concept_miner(
                brain=initializer.core_brain,
                event_bus=event_bus,
                deferred_system=deferred_system,
                config={
                    'enabled': True,
                    'dry_run': False,
                    'max_candidates_per_cycle': 3,
                    'enable_web_search_validation': False  # Пока отключим для скорости
                }
            )
            
            initializer.core_brain.concept_miner = concept_miner
            initializer.core_brain.components['concept_miner'] = concept_miner
            
            # Запускаем ConceptMiner
            concept_miner.start()
            
            initializer.logger.info("[OK] ConceptMiner (глубокий анализ) создан и запущен")
        except Exception as cme:
            initializer.logger.warning(f"[WARN] ConceptMiner не создан: {cme}")
         
        # Создаём Wikipedia Knowledge Base для enrichment концептов
        try:
            from eva_ai.knowledge.wikipedia_kb import get_wikipedia_kb
            wikipedia_kb = get_wikipedia_kb()
            initializer.core_brain.wikipedia_kb = wikipedia_kb
            initializer.core_brain.components['wikipedia_kb'] = wikipedia_kb
            initializer.logger.info("[OK] Wikipedia Knowledge Base создан")
        except Exception as wkbe:
            initializer.logger.warning(f"[WARN] Wikipedia Knowledge Base не создан: {wkbe}")

        initializer.logger.info("[OK] KnowledgeGraph адаптер (FGv2) создан")
        return kg_adapter
    except Exception as e:
        initializer.logger.error(f"[FAIL] Ошибка создания KnowledgeGraph адаптера: {e}")
        return None


def create_fractal_storage(initializer):
    """Создает FractalStorage для хранения цепочек рассуждений."""
    try:
        try:
            from eva_ai.reasoning.fractal_ml import FractalStorage
        except ImportError:
            initializer.logger.warning("[WARN] FractalStorage не найден в fractal_ml - пропускаем")
            initializer.failed_components.add('fractal_storage')
            return None

        storage_dir = os.path.join(
            getattr(initializer.core_brain, 'cache_dir', './cache'),
            'fractal_reasoning'
        )
        os.makedirs(storage_dir, exist_ok=True)

        fractal_storage = FractalStorage(storage_dir=storage_dir)

        initializer.core_brain.fractal_storage = fractal_storage

        if hasattr(initializer.core_brain, 'components'):
            initializer.core_brain.components['fractal_storage'] = fractal_storage

        initializer.logger.info(f"[OK] FractalStorage создан: {storage_dir}")
        return fractal_storage
    except Exception as e:
        initializer.logger.error(f"[FAIL] Ошибка создания fractal_storage: {e}", exc_info=True)
        initializer.failed_components.add('fractal_storage')
        return None


def create_self_reasoning_engine(initializer):
    """Создает Self-Reasoning Engine для саморассуждений с поддержкой рекурсии."""
    try:
        try:
            from eva_ai.reasoning import SelfReasoningEngine
        except ImportError:
            initializer.logger.warning("[WARN] SelfReasoningEngine не найден в eva_ai.reasoning - пропускаем")
            initializer.failed_components.add('self_reasoning_engine')
            return None

        try:
            from eva_ai.reasoning.integration import ReasoningIntegration
        except ImportError:
            ReasoningIntegration = None
            initializer.logger.debug("ReasoningIntegration не найден - интеграция недоступна")

        reasoning_config = initializer.core_brain.config.get('reasoning', {}) if hasattr(initializer.core_brain, 'config') else {}

        fractal_storage = getattr(initializer.core_brain, 'fractal_storage', None)

        two_model_pipeline = getattr(initializer.core_brain, 'two_model_pipeline', None)
        
        event_bus = getattr(initializer.core_brain, 'event_bus', None) or getattr(initializer.core_brain, '_new_event_bus', None)

        self_reasoning_engine = SelfReasoningEngine(
            brain=initializer.core_brain,
            two_model_pipeline=two_model_pipeline,
            event_bus=event_bus,
            config={
                'max_iterations': reasoning_config.get('max_iterations', 5),
                'confidence_threshold': reasoning_config.get('confidence_threshold', 0.75),
                'max_recursion_depth': reasoning_config.get('max_recursion_depth', 3)
            }
        )

        if fractal_storage:
            self_reasoning_engine.fractal_storage = fractal_storage
            self_reasoning_engine._init_retriever()

        if ReasoningIntegration:
            try:
                reasoning_integration = ReasoningIntegration(initializer.core_brain)
                reasoning_integration.reasoning_engine = self_reasoning_engine
                reasoning_integration.enabled = True
                initializer.core_brain.reasoning_integration = reasoning_integration
                initializer.logger.info("[OK] ReasoningIntegration также создан")
            except Exception as e:
                initializer.logger.debug(f"ReasoningIntegration не создан: {e}")

        initializer.core_brain.self_reasoning_engine = self_reasoning_engine

        initializer.logger.info("[OK] SelfReasoningEngine создан с рекурсивной поддержкой")
        return self_reasoning_engine
    except Exception as e:
        initializer.logger.error(f"[FAIL] Ошибка создания self_reasoning_engine: {e}", exc_info=True)
        initializer.failed_components.add('self_reasoning_engine')
        return None


def create_enhanced_reasoning_engine(initializer):
    """EnhancedReasoningEngine removed - dead code, never called"""
    initializer.logger.info("EnhancedReasoningEngine disabled - dead code removed")
    initializer.failed_components.add('enhanced_reasoning_engine')
    return None


def create_fcp_pipeline(initializer):
    """Создаёт FCPPipelineV15 - основной FCP пайплайн с GNN инъекцией."""
    initializer.logger.info("[FCP] === create_fcp_pipeline STARTED ===")
    try:
        from eva_ai.core.fcp_pipeline import FCPPipelineV15
        
        config = initializer.core_brain.config.get('fcp_pipeline', {})
        initializer.logger.info(f"[FCP] config: {config}")
        
        if not config.get('enabled', False):
            initializer.logger.info("FCPPipelineV15 disabled in config")
            return None
        
        model_path = config.get('model_path')
        if not model_path:
            model_path = os.path.join(
                getattr(initializer.core_brain, 'cache_dir', './cache'),
                'models',
                'ruadapt_qwen3_4b_openvino'
            )
        
        initializer.logger.info(f"[FCP] model_path: {model_path}")
        initializer.logger.info(f"[FCP] exists: {os.path.exists(model_path)}")
        
        if not os.path.exists(model_path):
            initializer.logger.error(f"[FCP] Model path does not exist!")
            return None
        
        graph_path = config.get('graph_path')
        gnn_ov_path = config.get('gnn_ov_path')
        lora_dir = config.get('lora_dir')
        
        initializer.logger.info(f"[FCP] Creating FCPPipelineV15...")
        pipeline = FCPPipelineV15(
            model_path=model_path,
            graph_path=graph_path,
            gnn_ov_path=gnn_ov_path,
            lora_dir=lora_dir
        )
        
        initializer.logger.info(f"[FCP] FCPPipelineV15 created: {pipeline}")
        initializer.logger.info(f"[FCP] inner pipeline: {getattr(pipeline, 'pipeline', 'NO ATTR')}")
        
        initializer.core_brain.fcp_pipeline = pipeline
        if hasattr(initializer.core_brain, 'components'):
            initializer.core_brain.components['fcp_pipeline'] = pipeline
        
        initializer.logger.info("[FCP] === create_fcp_pipeline SUCCESS ===")
        return pipeline
        
    except Exception as e:
        initializer.logger.error(f"[FCP] EXCEPTION: {e}", exc_info=True)
        initializer.failed_components.add('fcp_pipeline')
        return None


def register_all_factories(initializer):
    """Registers all component factories on the given initializer instance."""
    initializer.component_factories = {
        'event_bus': lambda: create_event_bus(initializer),
        'resource_manager': lambda: create_resource_manager(initializer),
        'config_manager': lambda: create_config_manager(initializer),
        'memory_manager': lambda: create_memory_manager(initializer),
        'hybrid_cache': lambda: create_hybrid_cache(initializer),
        'fractal_graph_v2': lambda: create_fractal_graph_v2(initializer),
        'qwen_api_enhancer': lambda: create_qwen_api_enhancer(initializer),
        'text_processor': lambda: create_text_processor(initializer),
        'ml_unit': lambda: create_ml_unit(initializer),
        'model_manager': lambda: create_model_manager(initializer),
        'query_processor': lambda: create_query_processor(initializer),
        'response_generator': lambda: create_response_generator(initializer),
        'reasoning_engine': lambda: create_reasoning_engine(initializer),
        'analytics_manager': lambda: create_analytics_manager(initializer),
        'system_monitor': lambda: create_system_monitor(initializer),
        'metrics_collector': lambda: create_metrics_collector(initializer),
        'contradiction_manager': lambda: create_contradiction_manager(initializer),
        'adaptation_manager': lambda: create_adaptation_manager(initializer),
        'ethics_framework': lambda: create_ethics_framework(initializer),
        'web_search_engine': lambda: create_web_search_engine(initializer),
        'fractal_storage': lambda: create_fractal_storage(initializer),
        'self_reasoning_engine': lambda: create_self_reasoning_engine(initializer),
        'enhanced_reasoning_engine': lambda: create_enhanced_reasoning_engine(initializer),
        'fcp_pipeline': lambda: create_fcp_pipeline(initializer),
        'wikipedia_kb': lambda: get_wikipedia_kb()
    }
    initializer.logger.info(f"[REGISTER] All factories: {list(initializer.component_factories.keys())}")
    initializer.logger.info(f"Зарегистрировано {len(initializer.component_factories)} фабрик компонентов")
