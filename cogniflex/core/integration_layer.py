"""
Интеграционный слой CogniFlex - центральный контроллер для обработки запросов.

Объединяет все компоненты системы через событийную шину согласно фрактальной архитектуре:
1. GUI → Ядро системы (query_received)
2. Ядро → Координатор токенизации (tokenize_request)
3. Токенизация → Фрактальное хранилище (tokens_ready)
4. Фрактальное хранилище → Генератор ответов (hot_window_ready)
5. Генератор → GUI (response_generated)
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
from ..generation.generation_coordinator import GenerationCoordinator
from ..memory.memory_manager import MemoryManager
from ..knowledge.knowledge_graph import KnowledgeGraph

logger = logging.getLogger("cogniflex.integration")

class CogniFlexIntegrator:
    """
    Центральный интегратор системы CogniFlex.

    Координирует работу всех компонентов через событийную шину,
    обеспечивая последовательную обработку запросов пользователя.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, brain=None):
        """
        Инициализация интегратора.

        Args:
            config: Конфигурация системы
            brain: Экземпляр CoreBrain (опционально)
        """
        self.config = config or {}
        self.core_brain = brain  # Store provided brain instance
        
        self.event_bus = EventBus(
            timeline_maxlen=self.config.get('timeline_maxlen', 1000),
            enable_timeline=self.config.get('enable_timeline', True)
        )

        # Основные компоненты системы
        self.response_generator = None
        self.generation_coordinator = None
        self.memory_manager = None
        self.knowledge_graph = None
        self.fractal_attention = None

        # Состояние системы
        self.initialized = False
        self.running = False
        self._shutdown_event = threading.Event()
        self._processing_lock = threading.Lock()
        self._active_requests = {}  # request_id -> request_data

        # Пул потоков для параллельной обработки
        self._executor = ThreadPoolExecutor(max_workers=self.config.get('max_workers', 8))

        # Метрики производительности
        self.metrics = {
            'total_requests': 0,
            'successful_responses': 0,
            'average_processing_time': 0.0,
            'active_sessions': 0
        }

        logger.info("CogniFlexIntegrator готов")

    def initialize(self) -> bool:
        """
        Инициализация всех компонентов системы.

        Returns:
            bool: True если инициализация успешна
        """
        try:
            logger.info("Начало инициализации системы CogniFlex")

            # 1. Используем переданный CoreBrain или создаём новый если не передан
            if self.core_brain is None:
                logger.info("CoreBrain не передан, создаём новый экземпляр")
                self.core_brain = CoreBrain(self.config)
            else:
                logger.info("Используем переданный CoreBrain")

            # Инициализируем CoreBrain если ещё не инициализирован
            if not getattr(self.core_brain, 'initialized', False):
                if not self.core_brain.initialize():
                    logger.error("Не удалось инициализировать CoreBrain")
                    return False
            else:
                logger.info("CoreBrain уже инициализирован")

            # 2. Инициализация компонентов
            self._initialize_components()

            # 3. Настройка подписок на события
            self._setup_event_subscriptions()

            # 4. Инициализация фрактальной системы внимания
            self.fractal_attention = FractalAttentionSystem(self.core_brain)

            # 5. Запуск фоновых процессов
            self._start_background_processes()

            self.initialized = True
            logger.info("Система CogniFlex готова")

            return True

        except Exception as e:
            logger.error(f"Ошибка инициализации системы: {e}", exc_info=True)
            return False

    def _initialize_components(self):
        """Инициализация основных компонентов."""
        try:
            # Генератор ответов
            self.response_generator = ResponseGenerator(
                brain=self.core_brain,
                model_manager=getattr(self.core_brain, 'model_manager', None)
            )

            # Координатор генерации
            generation_config = self.core_brain.generation_config or {}
            self.generation_coordinator = GenerationCoordinator(
                brain=self.core_brain,
                model_name=generation_config.get('model_name', 'sberbank-ai/rugpt3large_based_on_gpt2'),
                num_workers=generation_config.get('num_workers', 4),
                cache_dir=generation_config.get('cache_dir', './cache'),
                max_cache_size_gb=generation_config.get('cache_config', {}).get('target_memory_gb', 50)
            )
            logger.info("GenerationCoordinator готов")

            # Менеджер памяти
            if hasattr(self.core_brain, 'memory_manager'):
                self.memory_manager = self.core_brain.memory_manager
            else:
                from ..memory.memory_manager import MemoryManager
                cache_dir = os.path.join(getattr(self.core_brain, 'cache_dir', './cache'), 'memory')
                self.memory_manager = MemoryManager(cache_dir=cache_dir, brain=self.core_brain)
            logger.info("MemoryManager готов")

            # Граф знаний
            if hasattr(self.core_brain, 'knowledge_graph'):
                self.knowledge_graph = self.core_brain.knowledge_graph
            else:
                from ..knowledge.knowledge_graph import KnowledgeGraph
                self.knowledge_graph = KnowledgeGraph(self.core_brain)
            logger.info("KnowledgeGraph готов")

            # Движок рассуждений (внутренний диалог)
            reasoning_config = self.config.get('reasoning', {})
            self.reasoning_engine = ReasoningEngine(
                brain=self.core_brain,
                config=reasoning_config
            )
            logger.info("ReasoningEngine готов")

        except Exception as e:
            logger.error(f"Ошибка инициализации компонентов: {e}")

    def _setup_event_subscriptions(self):
        """Настройка подписок на события."""
        try:
            # 1. Подписка на query_received (максимальный приоритет)
            self.event_bus.subscribe(
                'query_received',
                self._handle_query_received,
                priority=10
            )

            # 2. Подписка на tokenize_request
            self.event_bus.subscribe(
                'tokenize_request',
                self._handle_tokenize_request,
                priority=8
            )

            # 3. Подписка на tokens_ready
            self.event_bus.subscribe(
                'tokens_ready',
                self._handle_tokens_ready,
                priority=7
            )

            # 4. Подписка на hot_window_ready
            self.event_bus.subscribe(
                'hot_window_ready',
                self._handle_hot_window_ready,
                priority=6
            )

            # 5. Подписка на response_generated
            self.event_bus.subscribe(
                'response_generated',
                self._handle_response_generated,
                priority=5
            )

            # 6. Подписка на contradiction_detected
            self.event_bus.subscribe(
                'contradiction_detected',
                self._handle_contradiction_detected,
                priority=9
            )

            # 7. Подписка на learning_opportunity
            self.event_bus.subscribe(
                'learning_opportunity',
                self._handle_learning_opportunity,
                priority=4
            )

            # 8. Подписка на self_dialog_request
            self.event_bus.subscribe(
                'self_dialog_request',
                self._handle_self_dialog_request,
                priority=3
            )

            # 9. Подписка на ethical_check_request
            self.event_bus.subscribe(
                'ethical_check_request',
                self._handle_ethical_check_request,
                priority=8
            )

            logger.info("Подписки на события настроены")

        except Exception as e:
            logger.error(f"Ошибка настройки подписок: {e}")

    def _start_background_processes(self):
        """Запуск фоновых процессов."""
        try:
            # 1. Запуск системы самообучения
            if hasattr(self.fractal_attention, 'learning_scheduler'):
                learning_thread = threading.Thread(
                    target=self._learning_scheduler_worker,
                    daemon=True,
                    name="LearningScheduler"
                )
                learning_thread.start()
                logger.info("LearningScheduler активен")

            # 2. Запуск системы самооптимизации
            if hasattr(self.fractal_attention, 'system_optimizer'):
                optimizer_thread = threading.Thread(
                    target=self._system_optimizer_worker,
                    daemon=True,
                    name="SystemOptimizer"
                )
                optimizer_thread.start()
                logger.info("SystemOptimizer активен")

            # 3. Запуск мониторинга здоровья системы
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
        """
        Обработка пользовательского запроса.

        Args:
            query: Текст запроса
            context: Дополнительный контекст

        Returns:
            Dict с результатом обработки
        """
        if not self.initialized:
            return {
                'status': 'error',
                'error': 'Система не готова',
                'response': 'Извините, система временно недоступна.'
            }

        start_time = time.time()
        request_id = f"req_{int(start_time * 1000)}"

        try:
            # Создаем данные запроса
            request_data = {
                'request_id': request_id,
                'query': query,
                'context': context or {},
                'timestamp': start_time,
                'status': 'processing'
            }

            # Регистрируем активный запрос
            with self._processing_lock:
                self._active_requests[request_id] = request_data

            # Публикуем событие query_received
            self.event_bus.trigger('query_received', request_data)

            # Ожидаем завершения обработки (асинхронно через события)
            timeout = self.config.get('processing_timeout', 30.0)
            end_time = time.time() + timeout

            while time.time() < end_time:
                with self._processing_lock:
                    current_data = self._active_requests.get(request_id, {})

                if current_data.get('status') in ['completed', 'error']:
                    break

                time.sleep(0.1)

            # Получаем финальный результат
            with self._processing_lock:
                final_data = self._active_requests.get(request_id, {})
                if request_id in self._active_requests:
                    del self._active_requests[request_id]

            # Обновляем метрики
            processing_time = time.time() - start_time
            self._update_metrics(final_data.get('status') == 'completed', processing_time)

            return final_data

        except Exception as e:
            logger.error(f"Ошибка обработки запроса {request_id}: {e}")

            # Очистка
            with self._processing_lock:
                if request_id in self._active_requests:
                    del self._active_requests[request_id]

            return {
                'status': 'error',
                'error': str(e),
                'response': 'Извините, произошла ошибка при обработке запроса.',
                'processing_time': time.time() - start_time
            }

    def _handle_query_received(self, request_data: Dict[str, Any]):
        """Обработка события query_received."""
        try:
            request_id = request_data['request_id']
            query = request_data['query']

            logger.info(f"Обработка запроса {request_id}: '{query[:50]}...'")

            # 1. Инициализируем фокус внимания
            if self.fractal_attention:
                focus_result = self.fractal_attention._initialize_attention_focus(query)
                request_data['attention_focus'] = focus_result

            # 2. Публикуем запрос на токенизацию
            tokenize_data = {
                'request_id': request_id,
                'query': query,
                'context': request_data.get('context', {}),
                'attention_focus': request_data.get('attention_focus', {})
            }

            self.event_bus.trigger('tokenize_request', tokenize_data)

        except Exception as e:
            logger.error(f"Ошибка обработки query_received: {e}")
            self._update_request_status(request_id, 'error', error=str(e))

    def _handle_tokenize_request(self, tokenize_data: Dict[str, Any]):
        """Обработка события tokenize_request."""
        try:
            request_id = tokenize_data['request_id']
            query = tokenize_data['query']

            logger.info(f"Токенизация запроса {request_id}")

            # Используем координатор генерации для токенизации
            if self.generation_coordinator:
                # Получаем токены
                tokens = self._tokenize_text(query)

                # Публикуем событие tokens_ready
                tokens_data = {
                    'request_id': request_id,
                    'query': query,
                    'tokens': tokens,
                    'context': tokenize_data.get('context', {}),
                    'attention_focus': tokenize_data.get('attention_focus', {})
                }

                self.event_bus.trigger('tokens_ready', tokens_data)

            else:
                logger.error("GenerationCoordinator недоступен")
                self._update_request_status(request_id, 'error', error="Токенизатор недоступен")

        except Exception as e:
            logger.error(f"Ошибка токенизации: {e}")
            self._update_request_status(request_id, 'error', error=str(e))

    def _handle_tokens_ready(self, tokens_data: Dict[str, Any]):
        """Обработка события tokens_ready."""
        try:
            request_id = tokens_data['request_id']
            tokens = tokens_data['tokens']

            logger.info(f"Токены готовы для запроса {request_id}")

            # Формируем горячее окно через фрактальное хранилище
            if self.memory_manager and hasattr(self.memory_manager, 'create_hot_window'):
                hot_window = self.memory_manager.create_hot_window(
                    tokens=tokens,
                    context=tokens_data.get('context', {}),
                    attention_focus=tokens_data.get('attention_focus', {})
                )

                # Публикуем событие hot_window_ready
                window_data = {
                    'request_id': request_id,
                    'query': tokens_data['query'],
                    'tokens': tokens,
                    'hot_window': hot_window,
                    'context': tokens_data.get('context', {}),
                    'attention_focus': tokens_data.get('attention_focus', {})
                }

                self.event_bus.trigger('hot_window_ready', window_data)

            else:
                logger.warning("MemoryManager недоступен, пропускаем формирование горячего окна")
                # Публикуем с пустым окном
                window_data = {
                    'request_id': request_id,
                    'query': tokens_data['query'],
                    'tokens': tokens,
                    'hot_window': {},
                    'context': tokens_data.get('context', {}),
                    'attention_focus': tokens_data.get('attention_focus', {})
                }
                self.event_bus.trigger('hot_window_ready', window_data)

        except Exception as e:
            logger.error(f"Ошибка обработки tokens_ready: {e}")
            self._update_request_status(request_id, 'error', error=str(e))

    def _handle_hot_window_ready(self, window_data: Dict[str, Any]):
        """Обработка события hot_window_ready с внутренним рассуждением."""
        try:
            request_id = window_data['request_id']
            query = window_data['query']
            hot_window = window_data['hot_window']

            logger.info(f"Горячее окно готово для запроса {request_id}")

            # === ВНУТРЕННЕЕ РАССУЖДЕНИЕ ===
            # Запускаем многоуровневый анализ перед генерацией ответа
            reasoning_result = None
            if self.reasoning_engine:
                try:
                    logger.info(f"Запуск рассуждения для запроса {request_id}")
                    reasoning_result = self.reasoning_engine.reason(
                        query=query,
                        context={
                            'hot_window': hot_window,
                            'attention_focus': window_data.get('attention_focus', {}),
                            'request_id': request_id
                        }
                    )
                    logger.info(f"Рассуждение завершено: {reasoning_result.get('reasoning_steps', 0)} шагов, "
                              f"уверенность={reasoning_result.get('confidence', 0):.2f}")
                except Exception as e:
                    logger.error(f"Ошибка рассуждения: {e}")
                    reasoning_result = None

            # Проверяем на противоречия (дополнительно к тому что в reasoning)
            contradictions = []
            if hasattr(self.fractal_attention, 'contradiction_resolver'):
                contradictions = self.fractal_attention.contradiction_resolver.detect_contradictions(
                    query, hot_window
                )

            # Генерируем ответ с учетом результатов рассуждения
            if self.response_generator:
                # Формируем контекст с инсайтами из рассуждения
                generation_context = {
                    'hot_window': hot_window,
                    'contradictions': contradictions,
                    'attention_focus': window_data.get('attention_focus', {}),
                    'reasoning_result': reasoning_result  # Добавляем результаты рассуждения
                }

                # Если есть результаты рассуждения - используем их для улучшения промпта
                if reasoning_result and reasoning_result.get('confidence', 0) > 0.5:
                    enhanced_prompt = self._enhance_prompt_with_reasoning(query, reasoning_result)
                else:
                    enhanced_prompt = query

                response_result = self.response_generator.generate_response(
                    prompt=enhanced_prompt,
                    context=generation_context
                )

                # Проверяем этическую корректность
                if contradictions:
                    self.event_bus.trigger('contradiction_detected', {
                        'request_id': request_id,
                        'contradictions': contradictions
                    })

                # Публикуем событие response_generated
                response_data = {
                    'request_id': request_id,
                    'query': query,
                    'response': response_result,
                    'hot_window': hot_window,
                    'contradictions': contradictions,
                    'reasoning_result': reasoning_result,  # Сохраняем результаты рассуждения
                    'processing_time': time.time() - window_data.get('timestamp', time.time())
                }

                self.event_bus.trigger('response_generated', response_data)

            else:
                logger.error("ResponseGenerator недоступен")
                self._update_request_status(request_id, 'error', error="Генератор ответов недоступен")

        except Exception as e:
            logger.error(f"Ошибка обработки hot_window_ready: {e}")
            self._update_request_status(request_id, 'error', error=str(e))

    def _handle_response_generated(self, response_data: Dict[str, Any]):
        """Обработка события response_generated."""
        try:
            request_id = response_data['request_id']

            logger.info(f"Ответ сгенерирован для запроса {request_id}")

            # Обновляем статус запроса
            with self._processing_lock:
                if request_id in self._active_requests:
                    self._active_requests[request_id].update({
                        'status': 'completed',
                        'response': response_data.get('response', {}),
                        'processing_time': response_data.get('processing_time', 0.0)
                    })

            # Проверяем возможности обучения
            if hasattr(self.fractal_attention, 'learning_scheduler'):
                learning_opportunities = self.fractal_attention.learning_scheduler.identify_learning_opportunities(
                    response_data.get('query', '')
                )

                if learning_opportunities:
                    self.event_bus.trigger('learning_opportunity', {
                        'request_id': request_id,
                        'opportunities': learning_opportunities
                    })

        except Exception as e:
            logger.error(f"Ошибка обработки response_generated: {e}")

    def _handle_contradiction_detected(self, contradiction_data: Dict[str, Any]):
        """Обработка события contradiction_detected."""
        try:
            request_id = contradiction_data['request_id']
            contradictions = contradiction_data['contradictions']

            logger.info(f"Обнаружено {len(contradictions)} противоречий для запроса {request_id}")

            # Разрешаем противоречия
            if hasattr(self.fractal_attention, 'contradiction_resolver'):
                for contradiction in contradictions:
                    resolution = self.fractal_attention.contradiction_resolver.resolve_contradiction(contradiction)
                    logger.info(f"Противоречие разрешено: {resolution}")

        except Exception as e:
            logger.error(f"Ошибка обработки противоречий: {e}")

    def _handle_learning_opportunity(self, learning_data: Dict[str, Any]):
        """Обработка события learning_opportunity."""
        try:
            opportunities = learning_data['opportunities']

            logger.info(f"Обнаружено {len(opportunities)} возможностей обучения")

            # Запускаем сессии обучения
            for opportunity in opportunities:
                if hasattr(self.fractal_attention, 'learning_scheduler'):
                    success = self.fractal_attention.learning_scheduler.schedule_learning_session(opportunity)
                    if success:
                        logger.info(f"Запланирована сессия обучения: {opportunity.get('description', '')}")

        except Exception as e:
            logger.error(f"Ошибка обработки возможностей обучения: {e}")

    def _handle_self_dialog_request(self, dialog_data: Dict[str, Any]):
        """Обработка события self_dialog_request."""
        try:
            logger.info("Запуск сессии самодиалога")

            if hasattr(self.fractal_attention, 'dialog_manager'):
                self.fractal_attention.dialog_manager.start_session()

        except Exception as e:
            logger.error(f"Ошибка запуска самодиалога: {e}")

    def _handle_ethical_check_request(self, ethical_data: Dict[str, Any]):
        """Обработка события ethical_check_request."""
        try:
            request_id = ethical_data.get('request_id', '')
            content = ethical_data.get('content', '')

            logger.info(f"Проверка этической корректности для запроса {request_id}")

            # Здесь должна быть интеграция с этической рамкой
            # Пока заглушка
            ethical_result = {
                'request_id': request_id,
                'score': 0.8,
                'approved': True,
                'recommendations': []
            }

            # Обновляем данные запроса
            self._update_request_status(request_id, 'ethical_check_completed', ethical_result=ethical_result)

        except Exception as e:
            logger.error(f"Ошибка этической проверки: {e}")

    def _tokenize_text(self, text: str) -> List[str]:
        """Токенизация текста."""
        try:
            if self.generation_coordinator and hasattr(self.generation_coordinator, 'tokenizer'):
                tokens = self.generation_coordinator.tokenizer.tokenize(text)
                return [str(token) for token in tokens]
            else:
                # Fallback токенизация
                return text.split()
        except Exception as e:
            logger.error(f"Ошибка токенизации: {e}")
            return text.split()

    def _update_request_status(self, request_id: str, status: str, **kwargs):
        """Обновление статуса запроса."""
        with self._processing_lock:
            if request_id in self._active_requests:
                self._active_requests[request_id]['status'] = status
                self._active_requests[request_id].update(kwargs)

    def _enhance_prompt_with_reasoning(self, query: str, reasoning_result: Dict) -> str:
        """Улучшает промпт на основе результатов рассуждения"""
        try:
            # Получаем инсайты из рассуждения
            insights = reasoning_result.get('reasoning_phases', [])
            confidence = reasoning_result.get('confidence', 0)
            
            # Если уверенность высокая - добавляем контекст
            if confidence > 0.6 and insights:
                # Формируем улучшенный промпт
                enhanced = f"""На основе анализа:
Запрос: {query}

Ключевые аспекты для учета:
- Тип запроса: {reasoning_result.get('reasoning_phases', ['general'])[0] if reasoning_result.get('reasoning_phases') else 'general'}
- Уверенность анализа: {confidence:.0%}
- Найдено инсайтов: {reasoning_result.get('insights_count', 0)}

Сформируйте точный и полезный ответ:"""
                return enhanced
            
            # Если уверенность низкая - возвращаем оригинал
            return query
            
        except Exception as e:
            logger.debug(f"Ошибка улучшения промпта: {e}")
            return query

    def _update_metrics(self, successful: bool, processing_time: float):
        """Обновление метрик производительности."""
        try:
            self.metrics['total_requests'] += 1
            if successful:
                self.metrics['successful_responses'] += 1

            # Обновляем среднее время обработки
            current_avg = self.metrics['average_processing_time']
            total_requests = self.metrics['total_requests']
            self.metrics['average_processing_time'] = (current_avg * (total_requests - 1) + processing_time) / total_requests

        except Exception as e:
            logger.error(f"Ошибка обновления метрик: {e}")

    def _learning_scheduler_worker(self):
        """Фоновый воркер планировщика обучения."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Проверяем возможности обучения каждые 5 минут
                    time.sleep(300)

                    if hasattr(self.fractal_attention, 'learning_scheduler'):
                        high_priority = self.fractal_attention.learning_scheduler.get_high_priority_opportunities()

                        if high_priority:
                            logger.info(f"Найдено {len(high_priority)} высокоприоритетных возможностей обучения")
                            # Можно запустить обучение автоматически

                except Exception as e:
                    logger.error(f"Ошибка в планировщике обучения: {e}")
                    time.sleep(60)

        except Exception as e:
            logger.error(f"Критическая ошибка в learning_scheduler_worker: {e}")

    def _system_optimizer_worker(self):
        """Фоновый воркер оптимизатора системы."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Оптимизируем систему каждые 10 минут
                    time.sleep(600)

                    if hasattr(self.fractal_attention, 'system_optimizer'):
                        logger.info("Запуск оптимизации системы")
                        self.fractal_attention.system_optimizer.optimize_system()

                except Exception as e:
                    logger.error(f"Ошибка в оптимизаторе системы: {e}")
                    time.sleep(120)

        except Exception as e:
            logger.error(f"Критическая ошибка в system_optimizer_worker: {e}")

    def _health_monitor_worker(self):
        """Фоновый воркер мониторинга здоровья системы."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Проверяем здоровье каждую минуту
                    time.sleep(60)

                    health_data = self.get_system_health()
                    logger.debug(f"Здоровье системы: {health_data}")

                    # Если есть проблемы, публикуем событие
                    if health_data.get('status') != 'healthy':
                        self.event_bus.trigger('system_health_check', health_data, priority_override=10)

                except Exception as e:
                    logger.error(f"Ошибка в мониторе здоровья: {e}")
                    time.sleep(30)

        except Exception as e:
            logger.error(f"Критическая ошибка в health_monitor_worker: {e}")

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

            # Проверяем основные компоненты
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

            # Определяем общий статус
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
                self.fractal_attention.system_optimizer.optimize_system()
                logger.info("Оптимизация выполнена")
            else:
                logger.warning("SystemOptimizer недоступен")

        except Exception as e:
            logger.error(f"Ошибка оптимизации системы: {e}")

    def shutdown(self):
        """Корректное завершение работы системы."""
        try:
            logger.info("Завершение работы CogniFlex")

            # Устанавливаем флаг завершения
            self._shutdown_event.set()

            # Останавливаем компоненты
            if self.generation_coordinator:
                self.generation_coordinator.cleanup()

            if self.core_brain:
                self.core_brain.stop()

            # Закрываем пул потоков
            if hasattr(self, '_executor'):
                self._executor.shutdown(wait=True)

            self.running = False
            logger.info("Работа CogniFlex завершена")

        except Exception as e:
            logger.error(f"Ошибка при завершении работы: {e}")

    def __del__(self):
        """Деструктор."""
        try:
            if hasattr(self, 'running') and self.running:
                self.shutdown()
        except Exception:
            pass


# Alias for backward compatibility
IntegrationLayer = CogniFlexIntegrator
