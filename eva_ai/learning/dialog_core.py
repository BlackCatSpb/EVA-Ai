"""Main class SelfDialogLearning, initialization, lifecycle, and main dialog loop."""
import logging
import time
import threading
import queue
import json
import os
from typing import Dict, List, Any, Optional, Callable

from eva_ai.learning.dialog_types import DialogRole, DialogTurn, LearningType, SelfDialog
from eva_ai.learning.dialog_topics import DialogTopicsMixin
from eva_ai.learning.dialog_generation import DialogGenerationMixin
from eva_ai.learning.dialog_learning import DialogLearningMixin
from eva_ai.learning.dialog_concepts import DialogConceptsMixin
from eva_ai.learning.interest_scorer import InterestScorer

logger = logging.getLogger("eva_ai.self_dialog_learning")


class SelfDialogLearning(DialogTopicsMixin, DialogGenerationMixin, DialogLearningMixin, DialogConceptsMixin):
    """
    Система самообучения через самодиалог.

    Основные функции:
    1. Мониторинг взаимодействий с пользователем
    2. Создание самодиалогов для анализа и обучения
    3. Выявление пробелов в знаниях
    4. Генерация обучающих сценариев
    5. Обновление базы знаний
    6. Автоматическое выполнение возможностей для обучения
    
    Интеграция с DeferredCommandSystem:
    - Использует DeferredCommandSystem для координации с другими модулями
    - Публикует события в EventBus для координации
    """

    def __init__(self, brain=None, config: Optional[Dict[str, Any]] = None):
        self.brain = brain
        self.config = config or {}

        self.enabled = self.config.get("enabled", True)
        self.auto_dialog_interval = self.config.get("auto_dialog_interval", 300)
        self.auto_learning_interval = self.config.get("auto_learning_interval", 60)
        self.max_dialog_turns = self.config.get("max_dialog_turns", 15)
        self.min_quality_threshold = self.config.get("min_quality_threshold", 0.9)
        self.auto_execute_enabled = self.config.get("auto_execute_enabled", True)

        self.min_priority_threshold = self.config.get("min_priority_threshold", 0.1)

        self.dialog_queue = queue.Queue()
        self.active_dialogs: Dict[str, SelfDialog] = {}
        self.dialog_history: List[SelfDialog] = []
        self.max_history_size = self.config.get("max_history_size", 100)

        self.running = False
        self.stop_event = threading.Event()
        self.last_learning_check = 0
        self.last_dialog_check = 0

        self._in_self_dialog = False

        self.learning_callbacks: List[Callable] = []

        self._recently_processed_topics: Dict[str, float] = {}
        self._topic_ttl = 600

        self.stats = {
            "total_dialogs": 0,
            "successful_learning": 0,
            "knowledge_gaps_identified": 0,
            "actions_taken": 0,
            "opportunities_executed": 0,
            "opportunities_found": 0
        }

        self.interest_scorer = InterestScorer()
        
        # DeferredCommandSystem integration
        self.deferred_system = None
        if brain and hasattr(brain, 'deferred_system'):
            self.deferred_system = brain.deferred_system
        
        # EventBus integration
        self._event_bus = None
        if brain:
            self._event_bus = getattr(brain, 'event_bus', None) or getattr(brain, '_new_event_bus', None)
        
        # Инициализация DialogConceptsMixin
        DialogConceptsMixin.__init__(self)
        
        # Подписка на события куратора
        self._setup_curator_events()
        
        # Подписка на события шины
        self._subscription_ids: List[str] = []
        self._setup_event_bus_subscriptions()
        
        logger.info("SelfDialogLearningSystem инициализирована")

    def _setup_curator_events(self):
        """Подписаться на события куратора графа."""
        if not self.brain:
            return
        
        event_bus = getattr(self.brain, 'event_bus', None) or getattr(self.brain, '_new_event_bus', None)
        if not event_bus:
            return
        
        try:
            event_bus.subscribe("curator.knowledge_extracted", self._on_curator_knowledge_extracted, priority=3)
            event_bus.subscribe("curator.graph_optimized", self._on_curator_graph_optimized, priority=3)
            event_bus.subscribe("curator.cleanup_done", self._on_curator_cleanup_done, priority=3)
            logger.debug("Подписка на события куратора установлена")
        except Exception as e:
            logger.debug(f"Ошибка подписки на события куратора: {e}")
    
    def _setup_event_bus_subscriptions(self):
        """Подписка на события EventBus для координации."""
        if not self._event_bus:
            return
        
        events_to_subscribe = [
            ("system.idle", "_on_system_idle"),
            ("system.state_changed", "_on_system_state_changed"),
            ("concept.confirmed", "_on_concept_confirmed"),
            ("contradiction.detected", "_on_contradiction_detected"),
        ]
        
        for event_type, handler_name in events_to_subscribe:
            try:
                if hasattr(self, handler_name):
                    handler = getattr(self, handler_name)
                    if hasattr(self._event_bus, 'subscribe'):
                        sub_id = self._event_bus.subscribe(event_type, handler, priority=5)
                        self._subscription_ids.append(sub_id)
                        logger.debug(f"Подписка на {event_type}: {sub_id}")
            except Exception as e:
                logger.debug(f"Ошибка подписки на {event_type}: {e}")
        
        if self._subscription_ids:
            logger.info(f"SelfDialogLearning подписан на {len(self._subscription_ids)} событий шины")

    def _on_curator_knowledge_extracted(self, data: Dict[str, Any]):
        """Обработка извлечённых знаний от куратора."""
        count = data.get('count', 0)
        if count > 0:
            logger.info(f"Куратор извлёк {count} новых знаний")
            self.stats["knowledge_gaps_identified"] += count

    def _on_curator_graph_optimized(self, data: Dict[str, Any]):
        """Обработка оптимизации графа куратором."""
        groups_created = data.get('groups_created', 0)
        if groups_created > 0:
            logger.debug(f"Куратор создал {groups_created} групп")

    def _on_curator_cleanup_done(self, data: Dict[str, Any]):
        """Обработка чистки графа куратором."""
        orphans = data.get('orphans_removed', 0)
        if orphans > 0:
            logger.debug(f"Куратор удалил {orphans} сиротских узлов")
    
    def _on_system_idle(self, event):
        """При простое системы - можно выполнить фоновые задачи."""
        if self.deferred_system and not self.dialog_queue.empty():
            try:
                from eva_ai.core.deferred_command_system import CommandPriority
                self.deferred_system.add_command(
                    command=self._process_queued_dialogs,
                    priority=CommandPriority.LOW,
                    command_id=f"self_dialog_process_{int(time.time())}"
                )
            except Exception as e:
                logger.debug(f"Не удалось добавить в DeferredCommandSystem: {e}")
    
    def _on_system_state_changed(self, event):
        """При изменении состояния системы."""
        data = event.data if hasattr(event, 'data') else {}
        new_state = data.get('state', '')
        if new_state == 'idle':
            self._on_system_idle(event)
    
    def _on_concept_confirmed(self, event):
        """При подтверждении концепта - добавить в очередь для обсуждения."""
        data = event.data if hasattr(event, 'data') else {}
        concept_name = data.get('concept_name', '')
        if concept_name:
            self.queue_concept_for_dialog(concept_name, priority=data.get('confidence', 0.5))
    
    def _on_contradiction_detected(self, event):
        """При обнаружении противоречия - добавить в очередь для разрешения."""
        data = event.data if hasattr(event, 'data') else {}
        contr_id = data.get('contradiction_id', '')
        concept = data.get('concept', '')
        if contr_id and concept:
            self.queue_contradiction_for_resolution(contr_id, concept, priority=data.get('priority', 0.7))
    
    def _process_queued_dialogs(self):
        """Обработать диалоги из очереди через DeferredCommandSystem."""
        processed = 0
        max_per_cycle = 3
        
        while processed < max_per_cycle:
            try:
                task = self.dialog_queue.get_nowait()
                self._process_task(task)
                processed += 1
            except queue.Empty:
                break

    def start(self):
        """Запускает систему самообучения."""
        if self.running:
            logger.warning("Система самообучения уже запущена")
            return False

        if not self.enabled:
            logger.info("Система самообучения отключена в конфигурации")
            return False

        self.running = True
        self.stop_event.clear()

        self._cleanup_low_priority_opportunities()

        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="SelfDialogLearning"
        )
        self.worker_thread.start()

        logger.info("Система самообучения запущена")
        return True

    def stop(self) -> None:
        """Останавливает систему самообучения."""
        if not self.running:
            return

        self.running = False
        self.stop_event.set()

        # Отписываемся от EventBus
        if self._event_bus and self._subscription_ids:
            for sub_id in self._subscription_ids:
                try:
                    self._event_bus.unsubscribe(sub_id)
                except Exception:
                    pass
            self._subscription_ids.clear()
            logger.debug("SelfDialogLearning отписался от EventBus")

        if hasattr(self, 'worker_thread') and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
            if self.worker_thread.is_alive():
                logger.warning("Worker thread did not stop within timeout, continuing shutdown")

        logger.info("Система самообучения остановлена")

    def _cleanup_low_priority_opportunities(self):
        """Очищает возможности с низким приоритетом при старте."""
        if not self.brain:
            return

        try:
            analyzer_core = None
            if hasattr(self.brain, 'self_analyzer') and self.brain.self_analyzer:
                analyzer_core = getattr(self.brain.self_analyzer, 'analyzer_core', None)
            elif hasattr(self.brain, 'analyzer_core'):
                analyzer_core = self.brain.analyzer_core

            if analyzer_core:
                try:
                    conn = getattr(analyzer_core, 'db_path', None)
                    if conn:
                        import sqlite3
                        with sqlite3.connect(conn) as db_conn:
                            cursor = db_conn.cursor()
                            cursor.execute('''
                                UPDATE learning_opportunities
                                SET executed = 1, execution = ?, last_updated = ?
                                WHERE priority < 0.3
                            ''', (json.dumps({"skipped": True, "reason": "low_priority_startup"}), time.time()))
                            deleted = cursor.rowcount
                            db_conn.commit()
                            if deleted > 0:
                                logger.info(f"Очищено {deleted} возможностей с низким приоритетом")
                except Exception as e:
                    logger.warning(f"Ошибка очистки низкоприоритетных возможностей: {type(e).__name__}: {e}", exc_info=True)
        except Exception as e:
            logger.warning(f"Ошибка при очистке: {type(e).__name__}: {e}", exc_info=True)

    def _worker_loop(self):
        """Основной рабочий цикл системы."""
        logger.info("Рабочий цикл самообучения запущен")
        
        idle_counter = 0  # Счетчик простоя
        last_activity = time.time()

        while not self.stop_event.is_set():
            try:
                task_processed = False
                try:
                    task = self.dialog_queue.get(timeout=1)
                    self._process_task(task)
                    task_processed = True
                    last_activity = time.time()
                    idle_counter = 0
                except queue.Empty:
                    idle_counter += 1

                if self.auto_execute_enabled:
                    self._check_and_execute_learning_opportunities()

                current_time = time.time()
                
                # Генерация диалогов по расписанию
                if current_time - self.last_dialog_check > 300:
                    self._generate_dialog_from_conversations()
                    self.last_dialog_check = current_time

                # Компактификация накопленного контекста в простое (каждые 60 сек без задач)
                if idle_counter > 60 and (current_time - last_activity > 60):
                    self._process_pending_context_compactions()
                    idle_counter = 0  # Сброс после обработки

                time.sleep(1)

            except Exception as e:
                logger.error(f"Ошибка в рабочем цикле самообучения: {e}")
                time.sleep(5)

        logger.info("Рабочий цикл самообучения завершен")
    
    def _process_pending_context_compactions(self):
        """Обрабатывает накопленные контексты для компактификации в простое системы."""
        try:
            if not self.brain:
                return
            
            # Проверяем гибридный кэш на наличие необработанных контекстов
            cache = getattr(self.brain, 'hybrid_cache', None)
            if not cache:
                return
            
            # Используем get_recent вместо search_keys
            raw_context_keys = cache.get_recent("raw_context:", limit=10)
            
            if not raw_context_keys:
                return
            
            logger.info(f"Обработка {len(raw_context_keys)} контекстов в простое")
            
            for key in raw_context_keys:
                try:
                    raw_data = cache.get(key)
                    if not raw_data:
                        continue
                    
                    # Компактифицируем
                    result = self.compact_context(
                        raw_data.get("items", []),
                        raw_data.get("query", ""),
                        target_size=raw_data.get("target_size", 2000),
                        method="semantic_extraction"
                    )
                    
                    # Сохраняем компактный контекст
                    compacted_key = key.replace("raw_context:", "compacted_context:")
                    cache.put(compacted_key, result, ttl=7200)  # 2 часа
                    
                    # Удаляем сырой контекст
                    cache.delete(key)
                    
                    logger.debug(f"Контекст {key} компактифицирован: {result['compression_ratio']:.1%}")
                    
                except Exception as e:
                    logger.warning(f"Ошибка обработки контекста {key}: {e}")
                    continue
            
            if raw_context_keys:
                logger.info(f"Компактификация завершена для {len(raw_context_keys)} контекстов")
                
        except Exception as e:
            logger.debug(f"Ошибка в _process_pending_context_compactions: {e}")

    def _generate_dialog_from_conversations(self) -> None:
        """
        Generates self-dialog from concepts, contradictions, or conversation history.
        
        Priority:
        1. Contradictions for resolution
        2. Concepts from queue
        3. Conversation history (fallback)
        """
        if not self.brain:
            return
        
        current_time = time.time()
        
        # Check interval
        if current_time - self.last_dialog_check < self.auto_dialog_interval:
            return
        
        self.last_dialog_check = current_time
        
        # Try to get topic from concepts/contradictions queue first
        next_topic = self._get_next_dialog_topic()
        
        if next_topic:
            topic_type = next_topic['type']
            title = next_topic['title']
            data = next_topic['data']
            
            logger.info(f"Creating self-dialog from {topic_type}: {title[:50]}...")
            
            # Create dialog with appropriate type
            dialog_id = f"dialog_{int(time.time() * 1000)}"
            dialog = SelfDialog(
                id=dialog_id,
                topic=title,
                turns=[],
                start_time=time.time()
            )
            
            self.active_dialogs[dialog_id] = dialog
            
            try:
                if topic_type == 'concept':
                    self._run_concept_dialog(dialog, data)
                elif topic_type == 'contradiction':
                    self._run_contradiction_dialog(dialog, data)
                
                dialog.end_time = time.time()
                self._finalize_dialog(dialog)
                self.stats["total_dialogs"] += 1
                
            except Exception as e:
                logger.error(f"Error in {topic_type} dialog: {e}")
                dialog.outcome = f"error: {str(e)}"
                dialog.end_time = time.time()
                
            return
        
        # Fallback to conversation history
        super()._generate_dialog_from_conversations()

    def create_dialog(self, topic: str, context: Optional[Dict[str, Any]] = None):
        """
        Создает новый самодиалог.

        Args:
            topic: Тема диалога
            context: Дополнительный контекст
            
        Использует DeferredCommandSystem для координации если доступен.
        """
        task = {
            "type": "create_dialog",
            "topic": topic,
            "context": context
        }
        
        # Пробуем использовать DeferredCommandSystem для приоритизации
        if self.deferred_system:
            try:
                from eva_ai.core.deferred_command_system import CommandPriority
                
                # Определяем приоритет по контексту
                priority = CommandPriority.NORMAL
                if context:
                    if context.get('trigger') == 'interest_scorer':
                        priority = CommandPriority.HIGH
                    elif context.get('feedback', {}).get('rating', 5) < 3:
                        priority = CommandPriority.HIGH
                
                self.deferred_system.add_command(
                    command=self._create_self_dialog,
                    args=(topic, context),
                    kwargs={},
                    priority=priority,
                    command_id=f"self_dialog_{int(time.time() * 1000)}"
                )
                
                # Публикуем событие о создании диалога
                if self._event_bus:
                    self._event_bus.publish("self_dialog.scheduled", {
                        "topic": topic,
                        "priority": priority.name,
                        "source": "user_request"
                    })
                
                return
            except Exception as e:
                logger.debug(f"DeferredCommandSystem unavailable: {e}")
        
        # Fallback на локальную очередь
        self.dialog_queue.put(task)

    def _create_self_dialog(self, topic: str, context: Optional[Dict[str, Any]] = None):
        """Внутренний метод создания самодиалога."""
        dialog_id = f"dialog_{int(time.time() * 1000)}"

        dialog = SelfDialog(
            id=dialog_id,
            topic=topic,
            turns=[],
            start_time=time.time()
        )

        self.active_dialogs[dialog_id] = dialog

        try:
            self._run_dialog(dialog, context)
        except Exception as e:
            logger.error(f"Ошибка выполнения самодиалога {dialog_id}: {e}")
            dialog.outcome = f"error: {str(e)}"

        dialog.end_time = time.time()

        self._finalize_dialog(dialog)

        self.stats["total_dialogs"] += 1

    def _process_task(self, task: Dict[str, Any]):
        """Обрабатывает задачу из очереди."""
        task_type = task.get("type")

        if task_type == "create_dialog":
            self._create_self_dialog(task.get("topic"), task.get("context"))
        elif task_type == "analyze_interaction":
            self._analyze_interaction(task.get("query"), task.get("response"), task.get("feedback"))
        elif task_type == "trigger_learning":
            self._trigger_learning(task.get("gap"))
        elif task_type == "compact_context":
            self._process_context_compaction(task.get("context_id"), task.get("context_data"))

    def _process_context_compaction(self, context_id: str, context_data: Dict[str, Any]):
        """Обрабатывает компактификацию контекста в фоне."""
        try:
            items = context_data.get("items", [])
            query = context_data.get("query", "")
            target_size = context_data.get("target_size", 2000)
            
            result = self.compact_context(
                items, query, target_size, method="semantic_extraction"
            )
            
            # Сохраняем результат для последующего использования
            if self.brain and hasattr(self.brain, 'hybrid_cache'):
                cache_key = f"compacted_context:{context_id}"
                self.brain.hybrid_cache.put(cache_key, result, ttl=3600)  # TTL 1 час
            
            logger.info(f"Контекст {context_id} компактифицирован: "
                       f"{result['compression_ratio']:.1%} сжатия, "
                       f"сохранность {result['semantic_preserved']:.1%}")
            
        except Exception as e:
            logger.error(f"Ошибка компактификации контекста {context_id}: {e}")

    def _finalize_dialog(self, dialog: SelfDialog):
        """Финализирует диалог и выполняет действия с proper cleanup."""
        try:
            if dialog.id in self.active_dialogs:
                del self.active_dialogs[dialog.id]

            dialog.turns.clear()

            self.dialog_history.append(dialog)

            if len(self.dialog_history) > self.max_history_size:
                excess = len(self.dialog_history) - self.max_history_size
                self.dialog_history = self.dialog_history[excess:]

            for gap in dialog.knowledge_gaps:
                try:
                    self._trigger_learning(gap)
                except Exception as gap_error:
                    logger.warning(f"Failed to trigger learning for gap '{gap}': {gap_error}")

            for callback in self.learning_callbacks:
                try:
                    callback(dialog)
                except Exception as callback_error:
                    logger.error(f"Ошибка в callback обучения: {callback_error}")

            logger.info(f"Самодиалог завершен: {dialog.id}, исход: {dialog.outcome}")

        except Exception as e:
            logger.error(f"Error in dialog finalization: {e}", exc_info=True)
        finally:
            dialog.turns = []
            dialog.knowledge_gaps = []

    def analyze_interaction(self, query: str, response: str, feedback: Optional[Dict[str, Any]] = None):
        """
        Анализирует взаимодействие с пользователем.

        Args:
            query: Запрос пользователя
            response: Ответ системы
            feedback: Обратная связь (опционально)
        """
        self.dialog_queue.put({
            "type": "analyze_interaction",
            "query": query,
            "response": response,
            "feedback": feedback
        })

    def _analyze_interaction(self, query: str, response: str, feedback: Optional[Dict[str, Any]] = None):
        """Внутренний метод анализа взаимодействия."""
        is_negative = False

        if feedback:
            if feedback.get("rating", 5) < 3:
                is_negative = True
            if feedback.get("corrected", False):
                is_negative = True

        # Сохраняем опыт в FractalGraphV2
        quality_score = 0.5
        if feedback:
            rating = feedback.get("rating", 5)
            quality_score = rating / 10.0
        
        if self.brain and hasattr(self.brain, 'fractal_graph_v2'):
            try:
                fg = self.brain.fractal_graph_v2
                if hasattr(fg, 'save_experience'):
                    model_used = feedback.get('model_used', 'self_dialog') if feedback else 'self_dialog'
                    fg.save_experience(
                        query=query,
                        response=response,
                        model_used=model_used,
                        quality_score=quality_score
                    )
                    logger.debug(f"Saved experience to FractalGraphV2: {query[:30]}...")
            except Exception as e:
                logger.debug(f"Error saving to FractalGraphV2: {e}")

        if is_negative:
            self.create_dialog(
                topic=f"Улучшение ответа на: {query[:50]}",
                context={
                    "user_query": query,
                    "system_response": response,
                    "feedback": feedback
                }
            )

    # ===== МЕТОДЫ КОМПАКТИФИКАЦИИ КОНТЕКСТА =====
    
    def compact_context(
        self,
        context_items: List[Dict[str, Any]],
        query: str,
        target_size: int = 2000,
        method: str = "semantic_extraction"
    ) -> Dict[str, Any]:
        """
        Компактифицирует контекст для подачи в генерацию.
        
        Args:
            context_items: Список элементов контекста (узлы графа, результаты поиска)
            query: Запрос пользователя для определения релевантности
            target_size: Целевой размер контекста в символах
            method: Метод компактификации ("semantic_extraction" или "hierarchical_summary")
            
        Returns:
            Dict с полями:
                - compacted_context: сжатый контекст (строка)
                - original_size: размер оригинала
                - compacted_size: размер сжатого
                - compression_ratio: коэффициент сжатия
                - semantic_preserved: оценка сохранения смысла (0-1)
                - metadata: доп. информация
        """
        if not context_items:
            return {
                "compacted_context": "",
                "original_size": 0,
                "compacted_size": 0,
                "compression_ratio": 1.0,
                "semantic_preserved": 1.0,
                "metadata": {"method": method, "items_processed": 0}
            }
        
        # Оцениваем релевантность каждого элемента
        scored_items = self._score_context_relevance(context_items, query)
        
        # Сортируем по релевантности
        scored_items.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Компактифицируем согласно методу
        if method == "semantic_extraction":
            compacted = self._compact_by_semantic_extraction(scored_items, target_size)
        elif method == "hierarchical_summary":
            compacted = self._compact_hierarchically(scored_items, target_size)
        else:
            compacted = self._compact_by_truncation(scored_items, target_size)
        
        original_size = sum(len(str(item.get("content", item.get("text", "")))) for item in context_items)
        compacted_size = len(compacted)
        
        # Оцениваем сохранность семантики
        semantic_preserved = self._estimate_semantic_preservation(
            context_items, compacted, query
        )
        
        return {
            "compacted_context": compacted,
            "original_size": original_size,
            "compacted_size": compacted_size,
            "compression_ratio": compacted_size / max(original_size, 1),
            "semantic_preserved": semantic_preserved,
            "metadata": {
                "method": method,
                "items_processed": len(context_items),
                "items_included": len(scored_items[:10])  # Топ-10
            }
        }
    
    def _score_context_relevance(
        self,
        items: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """Оценивает релевантность элементов контекста к запросу."""
        import re
        
        query_words = set(re.findall(r"\w+", query.lower()))
        scored = []
        
        for item in items:
            content = str(item.get("content", item.get("text", item.get("snippet", ""))))
            content_lower = content.lower()
            
            # Простая оценка по ключевым словам
            word_matches = sum(1 for word in query_words if word in content_lower)
            relevance = word_matches / max(len(query_words), 1)
            
            # Бонус за длину (информативность)
            length_score = min(len(content) / 1000, 0.3)  # Макс +0.3
            
            # Бонус за тип контента
            content_type = item.get("type", "unknown")
            type_bonus = {
                "concept": 0.2,
                "fact": 0.15,
                "response": 0.1,
                "query": 0.05
            }.get(content_type, 0)
            
            final_score = relevance + length_score + type_bonus
            
            item["relevance_score"] = min(1.0, final_score)
            scored.append(item)
        
        return scored
    
    def _compact_by_semantic_extraction(
        self,
        scored_items: List[Dict[str, Any]],
        target_size: int
    ) -> str:
        """Компактифицирует путем извлечения ключевых смыслов."""
        parts = []
        current_size = 0
        
        # Берем топ-результаты пока не достигнем целевого размера
        for item in scored_items[:15]:  # Макс 15 элементов
            content = str(item.get("content", item.get("text", item.get("snippet", ""))))
            
            # Если элемент слишком длинный - суммаризируем
            if len(content) > 300:
                content = self._extract_key_sentences(content, max_sentences=3)
            
            if current_size + len(content) > target_size:
                # Берем только часть
                remaining = target_size - current_size
                if remaining > 100:  # Минимум 100 символов
                    parts.append(content[:remaining])
                break
            
            parts.append(content)
            current_size += len(content) + 2  # +2 на разделитель
        
        return " | ".join(parts)
    
    def _compact_hierarchically(
        self,
        scored_items: List[Dict[str, Any]],
        target_size: int
    ) -> str:
        """Создает иерархическое резюме контекста."""
        # Группируем по типам
        by_type = {}
        for item in scored_items:
            item_type = item.get("type", "other")
            if item_type not in by_type:
                by_type[item_type] = []
            by_type[item_type].append(item)
        
        summary_parts = []
        
        # Сначала концепты (самые важные)
        if "concept" in by_type:
            concepts = by_type["concept"][:3]
            concept_texts = [str(c.get("content", ""))[:150] for c in concepts]
            summary_parts.append("Концепты: " + "; ".join(concept_texts))
        
        # Затем факты
        if "fact" in by_type and sum(len(p) for p in summary_parts) < target_size * 0.7:
            facts = by_type["fact"][:3]
            fact_texts = [str(f.get("content", ""))[:120] for f in facts]
            summary_parts.append("Факты: " + "; ".join(fact_texts))
        
        # Остальное
        remaining_size = target_size - sum(len(p) for p in summary_parts)
        if remaining_size > 200:
            other_items = []
            for item_type, items in by_type.items():
                if item_type not in ["concept", "fact"]:
                    other_items.extend(items[:2])
            
            other_texts = [str(i.get("content", ""))[:100] for i in other_items[:3]]
            if other_texts:
                summary_parts.append("Дополнительно: " + "; ".join(other_texts))
        
        return "\n".join(summary_parts)
    
    # ===== DUAL CIRCUIT INTEGRATION - Фаза 3 =====
    
    def trigger_self_dialog(self, reason: str = "manual") -> bool:
        """
        Триггер для запуска самообучения по требованию.
        
        Запускает цикл обработки если есть работа в очереди.
        
        Args:
            reason: Причина триггера ('manual', 'concept', 'contradiction', etc.)
            
        Returns:
            True если запущен
        """
        # Проверяем есть ли работа
        has_work = bool(self._concept_queue) or bool(self._contradiction_topics)
        
        if not has_work:
            logger.info(f"Нет работы в очереди, триггер '{reason}' отклонен")
            return False
        
        logger.info(f"Триггер '{reason}' активирован: {len(self._concept_queue)} концептов, {len(self._contradiction_topics)} противоречий")
        
        # Запускаем обработку
        self._process_with_dual_circuit_batch()
        
        return True
    
    def _process_with_dual_circuit_batch(self) -> None:
        """
        Обрабатывает все концепты и противоречия через generate_dual_circuit.
        
        Работает пока есть работа в очереди.
        После каждого цикла запускает GraphCurator.
        """
        processed_count = 0
        max_iterations = 20  # Защита от бесконечного цикла
        
        while processed_count < max_iterations:
            # Проверяем есть ли работа
            next_topic = self._get_next_dialog_topic()
            
            if not next_topic:
                logger.info(f"DualCircuit batch завершен: {processed_count} итераций")
                break
            
            try:
                topic_type = next_topic['type']
                title = next_topic['title']
                data = next_topic['data']
                
                logger.info(f"Обрабатываем {topic_type}: {title[:50]}...")
                
                # Вызываем generate_dual_circuit
                result = self._run_dual_circuit_for_topic(title, data)
                
                if result:
                    processed_count += 1
                    
                    # Извлекаем концепты из рассуждений и добавляем в очередь
                    if result.get('concepts_extracted'):
                        for concept in result['concepts_extracted'][:5]:
                            self.queue_concept_for_dialog(concept, priority=0.6)
                    
                    logger.info(f"DualCircuit итерация {processed_count}: сохранено {result.get('saved_to_graph', 0)} в граф")
                
            except Exception as e:
                logger.error(f"Ошибка в _process_with_dual_circuit_batch: {e}")
                continue
        
        # Запускаем GraphCurator после завершения всех итераций
        if processed_count > 0:
            self._run_graph_curator_after_cycle()
    
    def _run_dual_circuit_for_topic(self, title: str, data: Dict[str, Any]) -> Optional[Dict]:
        """
        Запускает generate_dual_circuit для конкретной темы.
        
        Args:
            title: Название темы
            data: Данные т��мы
            
        Returns:
            Результат generate_dual_circuit или None
        """
        # Получаем DualGenerator
        dual_gen = self._get_dual_generator()
        
        if not dual_gen:
            logger.warning("DualGenerator недоступен")
            return None
        
        # Формируем запрос
        query = title
        if data.get('name'):
            query = f"Исследуй концепт: {data['name']}"
        elif data.get('concept'):
            query = f"Разреши противоречие в концепте: {data['concept']}"
        
        try:
            # Вызываем dual circuit
            result = dual_gen.generate_dual_circuit(
                query=query,
                save_to_graph=True,
                extract_concepts=True
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка _run_dual_circuit_for_topic: {e}")
            return None
    
    def _get_dual_generator(self):
        """Получить DualGenerator из brain."""
        try:
            if self.brain and hasattr(self.brain, 'two_model_pipeline'):
                pipeline = self.brain.two_model_pipeline
                
                # Пробуем разные атрибуты
                if hasattr(pipeline, 'dual_generator'):
                    return pipeline.dual_generator
                
                # Для HybridPipelineAdapter
                if hasattr(pipeline, 'dual_generator'):
                    return pipeline.dual_generator
                
                # Для DualGenerator напрямую
                if hasattr(pipeline, 'generate_dual_circuit'):
                    return pipeline
                    
        except Exception as e:
            logger.warning(f"Cannot get DualGenerator: {e}")
        
        return None
    
    def _run_graph_curator_after_cycle(self) -> None:
        """
        Запускает GraphCurator после завершения цикла dual circuit.
        
        Точка интеграции с GraphCurator.
        """
        if not self.brain:
            return
        
        try:
            # Пробуем получить graph_curator
            graph_curator = getattr(self.brain, 'graph_curator', None)
            
            if graph_curator and hasattr(graph_curator, 'force_curation'):
                logger.info("Запускаем GraphCurator после цикла dual circuit")
                graph_curator.force_curation()
            else:
                # Пробуем напрямую через fractal_graph_v2
                fgv2 = getattr(self.brain, 'fractal_graph_v2', None)
                if fgv2 and hasattr(fgv2, 'save'):
                    fgv2.save()
                    logger.info("FractalGraphV2 сохранен после цикла")
                    
        except Exception as e:
            logger.warning(f"GraphCurator error: {e}")
        
        return " | ".join(summary_parts)
    
    def _compact_by_truncation(
        self,
        scored_items: List[Dict[str, Any]],
        target_size: int
    ) -> str:
        """Простое усечение до целевого размера."""
        all_content = " | ".join([
            str(item.get("content", item.get("text", "")))[:200]
            for item in scored_items[:10]
        ])
        
        if len(all_content) > target_size:
            return all_content[:target_size] + "..."
        return all_content
    
    def _extract_key_sentences(self, text: str, max_sentences: int = 3) -> str:
        """Извлекает ключевые предложения из текста."""
        import re
        
        # Разбиваем на предложения
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return text[:200]
        
        # Берем первое и последнее (обычно содержат суть) + самые длинные
        key_sentences = [sentences[0]]
        
        if len(sentences) > 1:
            # Добавляем самое длинное из середины
            middle = sentences[1:-1] if len(sentences) > 2 else []
            if middle:
                longest = max(middle, key=len)
                key_sentences.append(longest)
            
            if len(sentences) > 2:
                key_sentences.append(sentences[-1])
        
        return ". ".join(key_sentences[:max_sentences]) + "."
    
    def _estimate_semantic_preservation(
        self,
        original_items: List[Dict[str, Any]],
        compacted: str,
        query: str
    ) -> float:
        """Оценивает, насколько хорошо сохранился смысл при компактификации."""
        import re
        
        # Подсчитываем ключевые слова запроса в сжатом контексте
        query_words = set(re.findall(r"\w+", query.lower()))
        compacted_lower = compacted.lower()
        
        matches = sum(1 for word in query_words if word in compacted_lower)
        coverage = matches / max(len(query_words), 1)
        
        # Оценка по длине (не слишком ли сильно сжали)
        original_text = " ".join([str(i.get("content", "")) for i in original_items])
        length_ratio = len(compacted) / max(len(original_text), 1)
        
        # Если сжали слишком сильно - штраф
        if length_ratio < 0.1:
            coverage *= 0.7
        
        return min(1.0, coverage)
    
    def schedule_context_compaction(self, context_id: str, context_data: Dict[str, Any]):
        """
        Запланирует компактификацию контекста для выполнения в простое.
        Контекст сохраняется в очередь и обрабатывается при отсутствии запросов.
        """
        self.dialog_queue.put({
            "type": "compact_context",
            "context_id": context_id,
            "context_data": context_data,
            "priority": "low"  # Низкий приоритет - выполняется в простое
        })
        logger.debug(f"Запланирована компактификация контекста: {context_id}")

    def check_and_create_dialog(self, query: str, query_embedding: List[float] = None) -> bool:
        """
        Проверить интересность запроса и создать самодиалог если нужно.
        
        Args:
            query: Запрос пользователя
            query_embedding: Эмбеддинг запроса (опционально)
            
        Returns:
            True если создан самодиалог, False иначе
        """
        if self.interest_scorer.is_interesting(query):
            logger.info(f"Запрос признан интересным: {query[:50]}...")
            self.create_dialog(
                topic=f"Обучение на интересный запрос: {query[:50]}",
                context={
                    "user_query": query,
                    "query_embedding": query_embedding,
                    "trigger": "interest_scorer"
                }
            )
            return True
        return False

    def register_learning_callback(self, callback: Callable):
        """Регистрирует callback для завершения обучения."""
        self.learning_callbacks.append(callback)

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику системы."""
        return {
            **self.stats,
            "active_dialogs": len(self.active_dialogs),
            "total_history": len(self.dialog_history),
            "queue_size": self.dialog_queue.qsize()
        }

    def get_recent_learning(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Возвращает недавние результаты обучения."""
        recent = self.dialog_history[-limit:] if self.dialog_history else []
        return [
            {
                "id": d.id,
                "topic": d.topic,
                "outcome": d.outcome,
                "learning_type": d.learning_type.value if d.learning_type else None,
                "gaps": d.knowledge_gaps,
                "actions": d.actions_taken,
                "duration": d.end_time - d.start_time if d.end_time else 0
            }
            for d in recent
        ]

    def get_knowledge_gaps_summary(self) -> List[Dict[str, Any]]:
        """Возвращает сводку пробелов в знаниях."""
        all_gaps: Dict[str, int] = {}

        for dialog in self.dialog_history:
            for gap in dialog.knowledge_gaps:
                all_gaps[gap] = all_gaps.get(gap, 0) + 1

        return [
            {"gap": gap, "count": count}
            for gap, count in sorted(all_gaps.items(), key=lambda x: x[1], reverse=True)
        ]


def create_self_dialog_learning(brain=None, config: Optional[Dict[str, Any]] = None) -> SelfDialogLearning:
    """Factory function to create a SelfDialogLearning instance."""
    return SelfDialogLearning(brain=brain, config=config)
