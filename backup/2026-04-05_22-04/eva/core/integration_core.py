"""
Integration Core - Main ЕВАIntegrator class, initialization, and lifecycle.
"""

import logging
import os
import time
import threading
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from .event_system import EventBus
from .core_brain import CoreBrain
from .fractal_attention_system import FractalAttentionSystem
from .self_dialog_manager import SelfDialogManager
from .contradiction_resolver import ContradictionResolver
from .learning_scheduler import LearningScheduler
from .system_optimizer import SystemOptimizer
from .response_generator import ResponseGenerator
from .reasoning_engine import ReasoningEngine
try:
    from ..generation.generation_coordinator import GenerationCoordinator
except ImportError:
    from ..generation.generation_coordinator import UnifiedGenerationCoordinator as GenerationCoordinator
from ..memory.memory_manager import MemoryManager
from ..knowledge.knowledge_graph import KnowledgeGraph
from .integration_adapters import (
    _handle_query_received,
    _handle_tokenize_request,
    _handle_tokens_ready,
    _handle_hot_window_ready,
    _handle_response_generated,
    _handle_contradiction_detected,
    _handle_learning_opportunity,
    _handle_self_dialog_request,
    _handle_ethical_check_request,
    _tokenize_text,
    _enhance_prompt_with_reasoning,
)
from .integration_events import _setup_event_subscriptions
from .integration_sync import (
    _learning_scheduler_worker,
    _system_optimizer_worker,
    _health_monitor_worker,
)

logger = logging.getLogger("eva.integration")


class ЕВАIntegrator:
    """
    Центральный интегратор системы ЕВА.
    Координирует работу всех компонентов через событийную шину.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, brain=None):
        self.config = config or {}
        self.core_brain = brain
        
        self.event_bus = EventBus(
            timeline_maxlen=self.config.get('timeline_maxlen', 1000),
            enable_timeline=self.config.get('enable_timeline', True)
        )

        self.response_generator = None
        self.generation_coordinator = None
        self.memory_manager = None
        self.knowledge_graph = None
        self.fractal_attention = None

        self.initialized = False
        self.running = False
        self._shutdown_event = threading.Event()
        self._processing_lock = threading.Lock()
        self._active_requests = {}

        self._executor = ThreadPoolExecutor(max_workers=self.config.get('max_workers', 8))

        self.metrics = {
            'total_requests': 0,
            'successful_responses': 0,
            'average_processing_time': 0.0,
            'active_sessions': 0
        }

        logger.info("ЕВАIntegrator готов")

    def initialize(self) -> bool:
        """Инициализация всех компонентов системы."""
        try:
            logger.info("Начало инициализации системы ЕВА")

            if self.core_brain is None:
                logger.info("CoreBrain не передан, создаём новый экземпляр")
                self.core_brain = CoreBrain(self.config)
            else:
                logger.info("Используем переданный CoreBrain")

            if not getattr(self.core_brain, 'initialized', False):
                if not self.core_brain.initialize():
                    logger.error("Не удалось инициализировать CoreBrain")
                    return False
            else:
                logger.info("CoreBrain уже инициализирован")

            self._initialize_components()
            self._setup_event_subscriptions()

            self.fractal_attention = FractalAttentionSystem(self.core_brain)
            self.core_brain.attention_system = self.fractal_attention

            self._start_background_processes()

            self.initialized = True
            logger.info("Система ЕВА готова")

            return True

        except Exception as e:
            logger.error(f"Ошибка инициализации системы: {e}", exc_info=True)
            return False

    def _initialize_components(self):
        """Инициализация основных компонентов."""
        try:
            self.response_generator = ResponseGenerator(
                brain=self.core_brain,
                model_manager=getattr(self.core_brain, 'model_manager', None)
            )

            generation_config = self.core_brain.generation_config or {}
            self.generation_coordinator = GenerationCoordinator(
                brain=self.core_brain,
                model_name=generation_config.get('model_name', 'sberbank-ai/rugpt3large_based_on_gpt2'),
                num_workers=generation_config.get('num_workers', 4),
                cache_dir=generation_config.get('cache_dir', './cache'),
                max_cache_size_gb=generation_config.get('cache_config', {}).get('target_memory_gb', 50)
            )
            logger.info("GenerationCoordinator готов")

            if hasattr(self.core_brain, 'memory_manager'):
                self.memory_manager = self.core_brain.memory_manager
            else:
                from ..memory.memory_manager import MemoryManager
                cache_dir = os.path.join(getattr(self.core_brain, 'cache_dir', './cache'), 'memory')
                self.memory_manager = MemoryManager(cache_dir=cache_dir, brain=self.core_brain)
            logger.info("MemoryManager готов")

            if hasattr(self.core_brain, 'knowledge_graph'):
                self.knowledge_graph = self.core_brain.knowledge_graph
            else:
                from ..knowledge.knowledge_graph import KnowledgeGraph
                self.knowledge_graph = KnowledgeGraph(self.core_brain)
            logger.info("KnowledgeGraph готов")

            reasoning_config = self.config.get('reasoning', {})
            self.reasoning_engine = ReasoningEngine(
                brain=self.core_brain,
                config=reasoning_config
            )
            logger.info("ReasoningEngine готов")

        except Exception as e:
            logger.error(f"Ошибка инициализации компонентов: {e}")

    def _start_background_processes(self):
        """Запуск фоновых процессов."""
        try:
            if hasattr(self.fractal_attention, 'learning_scheduler'):
                learning_thread = threading.Thread(
                    target=self._learning_scheduler_worker,
                    daemon=True,
                    name="LearningScheduler"
                )
                learning_thread.start()
                logger.info("LearningScheduler активен")

            if hasattr(self.fractal_attention, 'system_optimizer'):
                optimizer_thread = threading.Thread(
                    target=self._system_optimizer_worker,
                    daemon=True,
                    name="SystemOptimizer"
                )
                optimizer_thread.start()
                logger.info("SystemOptimizer активен")

            health_thread = threading.Thread(
                target=self._health_monitor_worker,
                daemon=True,
                name="HealthMonitor"
            )
            health_thread.start()
            logger.info("HealthMonitor активен")

        except Exception as e:
            logger.error(f"Ошибка запуска фоновых процессов: {e}")

    def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Обработка пользовательского запроса."""
        if not self.initialized:
            return {
                'status': 'error',
                'error': 'Система не готова',
                'response': 'Извините, система временно недоступна.'
            }

        start_time = time.time()
        request_id = f"req_{int(start_time * 1000)}"

        try:
            request_data = {
                'request_id': request_id,
                'query': query,
                'context': context or {},
                'timestamp': start_time,
                'status': 'processing'
            }

            with self._processing_lock:
                self._active_requests[request_id] = request_data

            self.event_bus.trigger('query_received', request_data)

            timeout = self.config.get('processing_timeout', 30.0)
            end_time = time.time() + timeout

            while time.time() < end_time:
                with self._processing_lock:
                    current_data = self._active_requests.get(request_id, {})

                if current_data.get('status') in ['completed', 'error']:
                    break

                time.sleep(0.1)

            with self._processing_lock:
                final_data = self._active_requests.get(request_id, {})
                if request_id in self._active_requests:
                    del self._active_requests[request_id]

            processing_time = time.time() - start_time
            self._update_metrics(final_data.get('status') == 'completed', processing_time)

            return final_data

        except Exception as e:
            logger.error(f"Ошибка обработки запроса {request_id}: {e}")

            with self._processing_lock:
                if request_id in self._active_requests:
                    del self._active_requests[request_id]

            return {
                'status': 'error',
                'error': str(e),
                'response': 'Извините, произошла ошибка при обработке запроса.',
                'processing_time': time.time() - start_time
            }

    def _update_request_status(self, request_id: str, status: str, **kwargs):
        """Обновление статуса запроса."""
        with self._processing_lock:
            if request_id in self._active_requests:
                self._active_requests[request_id]['status'] = status
                self._active_requests[request_id].update(kwargs)

    def _update_metrics(self, successful: bool, processing_time: float):
        """Обновление метрик производительности."""
        try:
            self.metrics['total_requests'] += 1
            if successful:
                self.metrics['successful_responses'] += 1

            current_avg = self.metrics['average_processing_time']
            total_requests = self.metrics['total_requests']
            if total_requests > 0:
                self.metrics['average_processing_time'] = (current_avg * (total_requests - 1) + processing_time) / total_requests

        except Exception as e:
            logger.error(f"Ошибка обновления метрик: {e}")

    def get_system_health(self) -> Dict[str, Any]:
        """Получение состояния здоровья системы."""
        try:
            health = {
                'status': 'healthy',
                'timestamp': time.time(),
                'components': {},
                'metrics': self.metrics.copy(),
                'active_requests': len(self._active_requests)
            }

            components_status = {
                'core_brain': self.core_brain is not None and self.core_brain.running,
                'response_generator': self.response_generator is not None,
                'generation_coordinator': self.generation_coordinator is not None,
                'memory_manager': self.memory_manager is not None,
                'knowledge_graph': self.knowledge_graph is not None,
                'fractal_attention': self.fractal_attention is not None,
                'event_bus': self.event_bus is not None
            }

            health['components'] = components_status

            critical_components = ['core_brain', 'response_generator', 'generation_coordinator']
            if not all(components_status.get(comp, False) for comp in critical_components):
                health['status'] = 'degraded'
                health['issues'] = [comp for comp in critical_components if not components_status.get(comp, False)]

            return health

        except Exception as e:
            logger.error(f"Ошибка получения здоровья системы: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }

    def get_system_stats(self) -> Dict[str, Any]:
        """Получение статистики системы."""
        try:
            stats = {
                'uptime': time.time() - getattr(self, '_start_time', time.time()),
                'event_bus_stats': self.event_bus.get_event_stats(),
                'performance_stats': self.event_bus.get_performance_stats(),
                'health': self.get_system_health(),
                'metrics': self.metrics.copy(),
                'active_requests': len(self._active_requests)
            }

            return stats

        except Exception as e:
            logger.error(f"Ошибка получения статистики системы: {e}")
            return {'error': str(e)}

    def start_self_dialog(self):
        """Запуск сессии самодиалога."""
        try:
            self.event_bus.trigger('self_dialog_request', {
                'timestamp': time.time(),
                'reason': 'manual_request'
            })
            logger.info("Запрошена сессия самодиалога")

        except Exception as e:
            logger.error(f"Ошибка запуска самодиалога: {e}")

    def optimize_system(self):
        """Принудительная оптимизация системы."""
        try:
            if hasattr(self.fractal_attention, 'system_optimizer'):
                logger.info("Запуск принудительной оптимизации")
                optimizer = self.fractal_attention.system_optimizer
                if hasattr(optimizer, 'run_optimization') and callable(getattr(optimizer, 'run_optimization', None)):
                    optimizer.run_optimization()
                elif hasattr(optimizer, 'optimize') and callable(getattr(optimizer, 'optimize', None)):
                    optimizer.optimize()
                elif hasattr(optimizer, 'start_optimization_monitor'):
                    optimizer.start_optimization_monitor()
                else:
                    logger.warning("SystemOptimizer не имеет метода оптимизации")
                logger.info("Оптимизация выполнена")
            else:
                logger.warning("SystemOptimizer недоступен")

        except Exception as e:
            logger.error(f"Ошибка оптимизации системы: {e}")

    def shutdown(self):
        """Корректное завершение работы системы."""
        try:
            logger.info("Завершение работы ЕВА")

            self._shutdown_event.set()

            if self.generation_coordinator:
                self.generation_coordinator.cleanup()

            if self.core_brain:
                self.core_brain.stop()

            if hasattr(self, '_executor'):
                self._executor.shutdown(wait=True)

            self.running = False
            logger.info("Работа ЕВА завершена")

        except Exception as e:
            logger.error(f"Ошибка при завершении работы: {e}")

    def __del__(self):
        """Деструктор."""
        try:
            if hasattr(self, 'running') and self.running:
                self.shutdown()
        except Exception:
            pass


IntegrationLayer = ЕВАIntegrator
