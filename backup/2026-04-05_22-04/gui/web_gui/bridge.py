"""
GUI Bridge - интеграция Web GUI с ЕВА Core
"""
import logging
import threading
from typing import Optional, Callable, Any, Dict

logger = logging.getLogger("eva.webgui.bridge")


class GUIBridge:
    """
    Мост между ЕВА Core и Web GUI.
    Обеспечивает двустороннюю коммуникацию и передачу событий.
    """
    
    def __init__(self, brain=None, integrator=None):
        self.brain = brain
        self.integrator = integrator
        self.web_gui = None
        self._event_subscriptions = {}
        self._reasoning_callbacks = []
        self._training_callbacks = []
        self._lock = threading.Lock()
        
        logger.info("GUIBridge инициализирован")
    
    def set_web_gui(self, web_gui):
        """Установка ссылки на Web GUI"""
        self.web_gui = web_gui
        logger.info("Web GUI подключен к GUIBridge")
        
        self._setup_core_subscriptions()
    
    def _setup_core_subscriptions(self):
        """Настройка подписок на события Core"""
        
        def subscribe_to_brain():
            if not self.brain:
                return
            
            if hasattr(self.brain, 'events') and self.brain.events:
                self.brain.events.subscribe('query_received', self._on_query_received)
                self.brain.events.subscribe('response_generated', self._on_response_generated)
                self.brain.events.subscribe('training_progress', self._on_training_progress)
                self.brain.events.subscribe('reasoning_step', self._on_reasoning_step)
                self.brain.events.subscribe('system_error', self._on_system_error)
                
                logger.info("Подписки на события brain настроены")
            
            if hasattr(self.brain, 'event_bus') and self.brain.event_bus:
                self._subscribe_to_new_event_bus()
            
            # Подписка на deferred_system
            if hasattr(self.brain, 'deferred_system') and self.brain.deferred_system:
                self._setup_deferred_system_access()
            
            if hasattr(self.brain, 'on_model_load'):
                self.brain.on_model_load.append(self._on_model_load)
            
            if hasattr(self.brain, 'on_models_ready'):
                self.brain.on_models_ready.append(self._on_models_ready)
        
        thread = threading.Thread(target=subscribe_to_brain, daemon=True)
        thread.start()
    
    def _subscribe_to_new_event_bus(self):
        """Подписка на новую EventBus для получения событий"""
        try:
            from eva.core.event_bus import EventTypes
            
            event_bus = self.brain.event_bus
            
            # Системные события
            event_bus.subscribe(EventTypes.SYSTEM_ERROR, self._on_new_system_error)
            event_bus.subscribe(EventTypes.SYSTEM_READY, self._on_system_ready)
            event_bus.subscribe(EventTypes.SYSTEM_STOP, self._on_system_stop)
            
            # События компонентов
            event_bus.subscribe(EventTypes.COMPONENT_INITIALIZED, self._on_new_component_initialized)
            event_bus.subscribe(EventTypes.COMPONENT_STARTED, self._on_component_started)
            event_bus.subscribe(EventTypes.COMPONENT_STOPPED, self._on_component_stopped)
            event_bus.subscribe(EventTypes.COMPONENT_ERROR, self._on_component_error)
            
            # События обучения
            event_bus.subscribe(EventTypes.LEARNING_PROGRESS, self._on_new_learning_progress)
            event_bus.subscribe(EventTypes.LEARNING_COMPLETED, self._on_learning_completed)
            event_bus.subscribe(EventTypes.LEARNING_STARTED, self._on_learning_started)
            
            # События пайплайна
            event_bus.subscribe(EventTypes.PIPELINE_COMPLETE, self._on_new_pipeline_complete)
            event_bus.subscribe(EventTypes.PIPELINE_START, self._on_pipeline_start)
            event_bus.subscribe(EventTypes.PIPELINE_MODEL_A_COMPLETE, self._on_pipeline_model_a)
            event_bus.subscribe(EventTypes.PIPELINE_MODEL_B_COMPLETE, self._on_pipeline_model_b)
            
            # Противоречия
            event_bus.subscribe(EventTypes.CONTRADICTION_DETECTED, self._on_new_contradiction_detected)
            event_bus.subscribe(EventTypes.CONTRADICTION_RESOLVED, self._on_contradiction_resolved)
            
            # Память
            event_bus.subscribe(EventTypes.MEMORY_WARNING, self._on_memory_warning)
            event_bus.subscribe(EventTypes.MEMORY_OPTIMIZED, self._on_memory_optimized)
            
            # Знания
            event_bus.subscribe(EventTypes.KNOWLEDGE_UPDATED, self._on_knowledge_updated)
            event_bus.subscribe(EventTypes.KNOWLEDGE_ADDED, self._on_knowledge_added)
            
            logger.info("Подписки на новую EventBus настроены (17 событий)")
        except Exception as e:
            logger.warning("Не удалось подписаться на новую EventBus: {}".format(e))
    
    def _setup_deferred_system_access(self):
        """Настройка доступа к системе отложенных команд"""
        try:
            self.deferred_system = self.brain.deferred_system
            
            logger.info("Доступ к DeferredCommandSystem настроен")
            logger.info("DeferredCommandSystem: {}".format(type(self.deferred_system).__name__))
        except Exception as e:
            logger.warning("Не удалось настроить доступ к DeferredCommandSystem: {}".format(e))
    
    def _on_new_system_error(self, event):
        """Обработка системной ошибки из новой EventBus"""
        data = event.data if hasattr(event, 'data') else {}
        self._on_system_error(data)
    
    def _on_new_component_initialized(self, event):
        """Обработка инициализации компонента из новой EventBus"""
        data = event.data if hasattr(event, 'data') else {}
        logger.debug("Component initialized: {}".format(data))
        if self.web_gui:
            getattr(self.web_gui, "emit_system_notification", lambda x: None)({
                'type': 'success',
                'message': f"Компонент инициализирован: {data.get('component_name', 'unknown')}"
            })
    
    def _on_new_learning_progress(self, event):
        """Обработка прогресса обучения из новой EventBus"""
        data = event.data if hasattr(event, 'data') else {}
        self._on_training_progress(data)
    
    def _on_new_pipeline_complete(self, event):
        """Обработка завершения пайплайна из новой EventBus"""
        data = event.data if hasattr(event, 'data') else {}
        response = data.get('response', '')
        if response and self.web_gui:
            getattr(self.web_gui, "emit_system_notification", lambda x: None)({
                'type': 'info',
                'message': f"Ответ сгенерирован: {len(response)} символов"
            })
    
    def _on_new_contradiction_detected(self, event):
        """Обработка обнаружения противоречия из новой EventBus"""
        data = event.data if hasattr(event, 'data') else {}
        logger.warning("Contradiction detected: {}".format(data))
        if self.web_gui:
            getattr(self.web_gui, "emit_system_notification", lambda x: None)({
                'type': 'warning',
                'message': f"Обнаружено противоречие: {data.get('description', 'unknown')}"
            })
    
    def _on_query_received(self, data: Dict[str, Any]):
        """Обработка полученного запроса"""
        logger.debug("Query received: {}...".format(data.get('query', '')[:50]))
        
        if self.web_gui:
            getattr(self.web_gui, "emit_system_notification", lambda x: None)({
                'type': 'info',
                'message': f"Получен запрос: {data.get('query', '')[:30]}..."
            })
    
    def _on_response_generated(self, data: Dict[str, Any]):
        """Обработка сгенерированного ответа"""
        response = data.get('response', '')
        reasoning = data.get('reasoning', [])
        
        logger.debug("Response generated: {} chars".format(len(response)))
        
        if reasoning and self.web_gui:
            for step in reasoning:
                getattr(self.web_gui, "emit_reasoning_update", lambda x: None)({
                    'step': step.get('step', 0),
                    'title': step.get('title', ''),
                    'detail': step.get('detail', ''),
                    'timestamp': data.get('timestamp')
                })
    
    def _on_training_progress(self, data: Dict[str, Any]):
        """Обработка прогресса обучения"""
        if self.web_gui:
            getattr(self.web_gui, "emit_training_update", lambda x: None)({
                'state': data.get('state', 'training'),
                'progress': data.get('progress', 0),
                'epoch': data.get('epoch', 0),
                'max_epochs': data.get('max_epochs', 0),
                'step': data.get('step', 0),
                'loss': data.get('loss'),
                'log': data.get('log'),
                'log_level': data.get('log_level', 'info')
            })
    
    def _on_reasoning_step(self, data: Dict[str, Any]):
        """Обработка шага рассуждения"""
        if self.web_gui:
            getattr(self.web_gui, "emit_reasoning_update", lambda x: None)({
                'text': data.get('text', ''),
                'step': data.get('step', 0),
                'timestamp': data.get('timestamp')
            })
    
    def _on_system_error(self, data: Dict[str, Any]):
        """Обработка системной ошибки"""
        error_msg = data.get('error', 'Unknown error')
        logger.warning("System error: {}".format(error_msg))
        
        if self.web_gui:
            getattr(self.web_gui, "emit_system_notification", lambda x: None)({
                'type': 'error',
                'message': f"Ошибка системы: {error_msg}"
            })
    
    def _on_model_load(self, data: Dict[str, Any]):
        """Обработка загрузки модели"""
        event = data.get('event', '')
        
        if event == 'model_load_start':
            message = f"Загрузка модели: {data.get('name', 'unknown')}"
        elif event == 'model_load_complete':
            message = f"Модель загружена: {data.get('name', 'unknown')}"
        elif event == 'model_load_error':
            message = f"Ошибка загрузки: {data.get('error', 'unknown')}"
        else:
            return
        
        if self.web_gui:
            getattr(self.web_gui, "emit_system_notification", lambda x: None)({
                'type': 'info' if 'error' not in event else 'error',
                'message': message
            })
    
    def _on_models_ready(self, data=None):
        """Обработка готовности моделей"""
        if self.web_gui:
            getattr(self.web_gui, "emit_system_notification", lambda x: None)({
                'type': 'success',
                'message': 'Все модели готовы к работе'
            })
    
    # ---- Новые обработчики событий EventBus ----
    
    def _on_system_ready(self, event):
        """Обработка готовности системы"""
        data = event.data if hasattr(event, 'data') else {}
        logger.info("System ready received")
        if self.web_gui:
            getattr(self.web_gui, "emit_system_notification", lambda x: None)({
                'type': 'success',
                'message': 'Система готова к работе'
            })
    
    def _on_system_stop(self, event):
        """Обработка остановки системы"""
        data = event.data if hasattr(event, 'data') else {}
        logger.info("System stop received")
        if self.web_gui:
            getattr(self.web_gui, "emit_system_notification", lambda x: None)({
                'type': 'info',
                'message': 'Система останавливается'
            })
    
    def _on_component_started(self, event):
        """Обработка запуска компонента"""
        data = event.data if hasattr(event, 'data') else {}
        if self.web_gui:
            getattr(self.web_gui, "emit_system_notification", lambda x: None)({
                'type': 'info',
                'message': 'Компонент запущен: {}'.format(data.get('component_name', 'unknown'))
            })
    
    def _on_component_stopped(self, event):
        """Обработка остановки компонента"""
        data = event.data if hasattr(event, 'data') else {}
        if self.web_gui:
            getattr(self.web_gui, "emit_system_notification", lambda x: None)({
                'type': 'info',
                'message': 'Компонент остановлен: {}'.format(data.get('component_name', 'unknown'))
            })
    
    def _on_component_error(self, event):
        """Обработка ошибки компонента"""
        data = event.data if hasattr(event, 'data') else {}
        logger.error("Component error: {}".format(data))
        if self.web_gui:
            getattr(self.web_gui, "emit_system_notification", lambda x: None)({
                'type': 'error',
                'message': 'Ошибка компонента: {}'.format(data.get('component', 'unknown'))
            })
    
    def _on_learning_completed(self, event):
        """Обработка завершения обучения"""
        data = event.data if hasattr(event, 'data') else {}
        if self.web_gui:
            getattr(self.web_gui, "emit_system_notification", lambda x: None)({
                'type': 'success',
                'message': 'Обучение завершено: {}'.format(data.get('type', 'unknown'))
            })
    
    def _on_learning_started(self, event):
        """Обработка начала обучения"""
        data = event.data if hasattr(event, 'data') else {}
        if self.web_gui:
            getattr(self.web_gui, "emit_system_notification", lambda x: None)({
                'type': 'info',
                'message': 'Начато обучение: {}'.format(data.get('type', 'unknown'))
            })
    
    def _on_pipeline_start(self, event):
        """Обработка начала пайплайна"""
        data = event.data if hasattr(event, 'data') else {}
        if self.web_gui:
            getattr(self.web_gui, "emit_system_notification", lambda x: None)({
                'type': 'info',
                'message': 'Начат пайплайн: {}'.format(data.get('query_id', 'unknown'))
            })
    
    def _on_pipeline_model_a(self, event):
        """Обработка завершения модели A"""
        data = event.data if hasattr(event, 'data') else {}
        logger.debug("Pipeline model A complete")
    
    def _on_pipeline_model_b(self, event):
        """Обработка завершения модели B"""
        data = event.data if hasattr(event, 'data') else {}
        logger.debug("Pipeline model B complete")
    
    def _on_contradiction_resolved(self, event):
        """Обработка разрешения противоречия"""
        data = event.data if hasattr(event, 'data') else {}
        if self.web_gui:
            getattr(self.web_gui, "emit_system_notification", lambda x: None)({
                'type': 'success',
                'message': 'Противоречие разрешено'
            })
    
    def _on_memory_warning(self, event):
        """Обработка предупреждения о памяти"""
        data = event.data if hasattr(event, 'data') else {}
        if self.web_gui:
            getattr(self.web_gui, "emit_system_notification", lambda x: None)({
                'type': 'warning',
                'message': 'Предупреждение о памяти: {}'.format(data.get('warning', 'unknown'))
            })
    
    def _on_memory_optimized(self, event):
        """Обработка оптимизации памяти"""
        data = event.data if hasattr(event, 'data') else {}
        if self.web_gui:
            getattr(self.web_gui, "emit_system_notification", lambda x: None)({
                'type': 'info',
                'message': 'Память оптимизирована'
            })
    
    def _on_knowledge_updated(self, event):
        """Обработка обновления знаний"""
        data = event.data if hasattr(event, 'data') else {}
        if self.web_gui:
            getattr(self.web_gui, "emit_system_notification", lambda x: None)({
                'type': 'info',
                'message': 'База знаний обновлена'
            })
    
    def _on_knowledge_added(self, event):
        """Обработка добавления знаний"""
        data = event.data if hasattr(event, 'data') else {}
        if self.web_gui:
            getattr(self.web_gui, "emit_system_notification", lambda x: None)({
                'type': 'info',
                'message': 'Добавлены новые знания'
            })
    
    def send_message(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Отправка сообщения через Core"""
        try:
            if self.integrator:
                return self.integrator.process_query(query, context or {})
            elif self.brain and hasattr(self.brain, 'process_query'):
                return self.brain.process_query(query, context or {})
            else:
                return {'status': 'error', 'response': 'Система недоступна'}
        except Exception as e:
            logger.error("Error sending message: {}".format(e))
            return {'status': 'error', 'response': str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Получение статуса системы"""
        status = {
            'status': 'unknown',
            'components': 0,
            'timestamp': None
        }
        
        try:
            if self.integrator and hasattr(self.integrator, 'get_system_health'):
                health = self.integrator.get_system_health()
                status.update(health)
            elif self.brain:
                if hasattr(self.brain, 'running') and self.brain.running:
                    status['status'] = 'active'
                if hasattr(self.brain, 'components'):
                    status['components'] = len(self.brain.components)
        except Exception as e:
            logger.error("Error getting system status: {}".format(e))
        
        return status
    
    def get_memory_graph_data(self) -> Dict[str, Any]:
        """Получение данных графа памяти"""
        graph_data = {
            'nodes': [],
            'edges': [],
            'stats': {}
        }
        
        try:
            if self.brain and hasattr(self.brain, 'memory_manager'):
                mm = self.brain.memory_manager
                
                if hasattr(mm, 'get_graph_data'):
                    graph_data = mm.get_graph_data()
                elif hasattr(mm, 'nodes'):
                    nodes = mm.nodes
                    graph_data['nodes'] = [
                        {
                            'id': n.get('id', i) if isinstance(n, dict) else i,
                            'label': n.get('content', '')[:50] if isinstance(n, dict) else str(n)[:50],
                            'type': n.get('type', 'memory') if isinstance(n, dict) else 'memory'
                        }
                        for i, n in enumerate(nodes[:100])
                    ]
                    graph_data['stats']['total_nodes'] = len(nodes)
                    
                if hasattr(mm, 'edges'):
                    graph_data['edges'] = [
                        {'from': e.get('from', ''), 'to': e.get('to', '')}
                        for e in mm.edges[:200]
                    ]
                    
        except Exception as e:
            logger.error("Error getting memory graph: {}".format(e))
        
        return graph_data
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Получение метрик системы"""
        metrics = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'cache_hit_rate': 0.0,
            'active_queries': 0
        }
        
        try:
            if self.brain:
                if hasattr(self.brain, 'get_resource_snapshot'):
                    snapshot = self.brain.get_resource_snapshot()
                    metrics.update(snapshot)
                if hasattr(self.brain, 'get_cache_stats'):
                    cache = self.brain.get_cache_stats()
                    metrics['cache_hit_rate'] = cache.get('hit_rate', 0.0)
        except Exception as e:
            logger.error("Error getting metrics: {}".format(e))
        
        return metrics
    
    def start_training(self, config: Optional[Dict] = None):
        """Запуск обучения"""
        try:
            if self.integrator and hasattr(self.integrator, 'start_training'):
                self.integrator.start_training(config or {})
            elif self.brain and hasattr(self.brain, 'start_training'):
                self.brain.start_training(config or {})
            else:
                logger.warning("Training not available")
        except Exception as e:
            logger.error("Error starting training: {}".format(e))
    
    def start_self_dialog(self):
        """Запуск самодиалога"""
        try:
            if self.integrator and hasattr(self.integrator, 'start_self_dialog'):
                self.integrator.start_self_dialog()
            elif self.brain and hasattr(self.brain, 'start_self_dialog'):
                self.brain.start_self_dialog()
            else:
                logger.warning("Self dialog not available")
        except Exception as e:
            logger.error("Error starting self dialog: {}".format(e))
    
    def optimize_system(self):
        """Оптимизация системы"""
        try:
            if self.integrator and hasattr(self.integrator, 'optimize_system'):
                self.integrator.optimize_system()
            elif self.brain and hasattr(self.brain, 'optimize_system'):
                self.brain.optimize_system()
            else:
                logger.warning("Optimization not available")
        except Exception as e:
            logger.error("Error optimizing system: {}".format(e))


class NetworkBridge:
    """
    Адаптер для сетевого доступа к ЕВА.
    Позволяет подключаться к удалённому серверу.
    """
    
    def __init__(self, host: str = '127.0.0.1', port: int = 5555, use_ssl: bool = False):
        self.host = host
        self.port = port
        self.use_ssl = use_ssl
        self.connected = False
        self._socket = None
        
        logger.info("NetworkBridge инициализирован: {}:{}".format(host, port))
    
    def connect(self) -> bool:
        """Подключение к серверу"""
        try:
            import socket
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.connect((self.host, self.port))
            self.connected = True
            logger.info("Подключено к {}:{}".format(self.host, self.port))
            return True
        except Exception as e:
            logger.error("Ошибка подключения: {}".format(e))
            self.connected = False
            return False
    
    def disconnect(self):
        """Отключение от сервера"""
        if self._socket:
            self._socket.close()
            self._socket = None
        self.connected = False
        logger.info("Отключено от сервера")
    
    def send_command(self, command: str, data: Optional[Dict] = None) -> Optional[Dict]:
        """Отправка команды на сервер"""
        if not self.connected:
            logger.warning("Не подключено к серверу")
            return None
        
        try:
            import json
            
            message = json.dumps({
                'command': command,
                'data': data or {}
            })
            
            self._socket.sendall(message.encode('utf-8'))
            
            response = self._socket.recv(4096)
            return json.loads(response.decode('utf-8'))
            
        except Exception as e:
            logger.error("Ошибка отправки команды: {}".format(e))
            return None
    
    def start_gui(self, brain=None, integrator=None):
        """Запуск Web GUI с интеграцией"""
        from .server import create_app
        
        bridge = GUIBridge(brain=brain, integrator=integrator)
        
        web_gui = create_app(brain=brain, integrator=integrator)
        
        bridge.set_web_gui(web_gui)
        
        return web_gui


def create_gui_bridge(brain=None, integrator=None) -> GUIBridge:
    """Создание моста GUI"""
    return GUIBridge(brain=brain, integrator=integrator)


def start_web_gui(brain=None, integrator=None, host: str = '127.0.0.1', port: int = 5555):
    """Запуск Web GUI с интеграцией в Core"""
    from .server import create_app
    
    bridge = GUIBridge(brain=brain, integrator=integrator)
    
    web_gui = create_app(brain=brain, integrator=integrator, host=host, port=port)
    
    bridge.set_web_gui(web_gui)
    
    logger.info("Web GUI запущен на http://{}:{}".format(host, port))
    
    return web_gui, bridge
