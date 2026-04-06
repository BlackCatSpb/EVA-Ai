"""Main class SelfDialogLearning, initialization, lifecycle, and main dialog loop."""
import logging
import time
import threading
import queue
import json
import os
from typing import Dict, List, Any, Optional, Callable

from eva.learning.dialog_types import DialogRole, DialogTurn, LearningType, SelfDialog
from eva.learning.dialog_topics import DialogTopicsMixin
from eva.learning.dialog_generation import DialogGenerationMixin
from eva.learning.dialog_learning import DialogLearningMixin

logger = logging.getLogger("eva.self_dialog_learning")


class SelfDialogLearning(DialogTopicsMixin, DialogGenerationMixin, DialogLearningMixin):
    """
    Система самообучения через самодиалог.

    Основные функции:
    1. Мониторинг взаимодействий с пользователем
    2. Создание самодиалогов для анализа и обучения
    3. Выявление пробелов в знаниях
    4. Генерация обучающих сценариев
    5. Обновление базы знаний
    6. Автоматическое выполнение возможностей для обучения
    """

    def __init__(self, brain=None, config: Optional[Dict[str, Any]] = None):
        """
        Инициализирует систему самообучения.

        Args:
            brain: Ссылка на ядро ЕВА
            config: Конфигурация системы
        """
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

        logger.info("SelfDialogLearningSystem инициализирована")

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

        while not self.stop_event.is_set():
            try:
                try:
                    task = self.dialog_queue.get(timeout=1)
                    self._process_task(task)
                except queue.Empty:
                    pass

                if self.auto_execute_enabled:
                    self._check_and_execute_learning_opportunities()

                current_time = time.time()
                if current_time - self.last_dialog_check > 300:
                    self._generate_dialog_from_conversations()

                time.sleep(1)

            except Exception as e:
                logger.error(f"Ошибка в рабочем цикле самообучения: {e}")
                time.sleep(5)

        logger.info("Рабочий цикл самообучения завершен")

    def create_dialog(self, topic: str, context: Optional[Dict[str, Any]] = None):
        """
        Создает новый самодиалог.

        Args:
            topic: Тема диалога
            context: Дополнительный контекст
        """
        self.dialog_queue.put({
            "type": "create_dialog",
            "topic": topic,
            "context": context
        })

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

        if is_negative:
            self.create_dialog(
                topic=f"Улучшение ответа на: {query[:50]}",
                context={
                    "user_query": query,
                    "system_response": response,
                    "feedback": feedback
                }
            )

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
