"""Система самообучения через самодиалог для CogniFlex.

Этот модуль реализует механизм самообучения системы через создание
внутреннего диалога между различными аспектами системы (AI assistant,
critic, learner, etc.) для выявления пробелов в знаниях и их заполнения.
"""
import logging
import time
import threading
import queue
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("cogniflex.self_dialog_learning")

class DialogRole(Enum):
    """Роли участников самодиалога."""
    ASSISTANT = "assistant"
    CRITIC = "critic"
    LEARNER = "learner"
    TEACHER = "teacher"
    OBSERVER = "observer"

class LearningType(Enum):
    """Типы обучения."""
    EXPANSION = "expansion"
    REFINEMENT = "refinement"
    UPDATING = "updating"
    INTEGRATION = "integration"

@dataclass
class DialogTurn:
    """Один ход в самодиалоге."""
    role: DialogRole
    content: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0

@dataclass
class SelfDialog:
    """Полный самодиалог системы с самим собой."""
    id: str
    topic: str
    turns: List[DialogTurn]
    start_time: float
    end_time: Optional[float] = None
    outcome: Optional[str] = None
    learning_type: Optional[LearningType] = None
    knowledge_gaps: List[str] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)

class SelfDialogLearningSystem:
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
            brain: Ссылка на ядро CogniFlex
            config: Конфигурация системы
        """
        self.brain = brain
        self.config = config or {}
        
        self.enabled = self.config.get("enabled", True)
        self.auto_dialog_interval = self.config.get("auto_dialog_interval", 300)
        self.auto_learning_interval = self.config.get("auto_learning_interval", 60)
        self.max_dialog_turns = self.config.get("max_dialog_turns", 10)
        self.min_quality_threshold = self.config.get("min_quality_threshold", 0.6)
        self.auto_execute_enabled = self.config.get("auto_execute_enabled", True)
        
        # Порог приоритета для выполнения обучения (снижен с 0.3 до 0.1)
        self.min_priority_threshold = self.config.get("min_priority_threshold", 0.1)
        
        self.dialog_queue = queue.Queue()
        self.active_dialogs: Dict[str, SelfDialog] = {}
        self.dialog_history: List[SelfDialog] = []
        self.max_history_size = self.config.get("max_history_size", 100)
        
        self.running = False
        self.stop_event = threading.Event()
        self.last_learning_check = 0
        
        self.learning_callbacks: List[Callable] = []
        
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
    
    def stop(self):
        """Останавливает систему самообучения."""
        if not self.running:
            return
        
        self.running = False
        self.stop_event.set()
        
        if hasattr(self, 'worker_thread') and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        
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
                        db_conn = sqlite3.connect(conn)
                        cursor = db_conn.cursor()
                        cursor.execute('''
                            UPDATE learning_opportunities 
                            SET executed = 1, execution = ?, last_updated = ?
                            WHERE priority < 0.3
                        ''', (json.dumps({"skipped": True, "reason": "low_priority_startup"}), time.time()))
                        deleted = cursor.rowcount
                        db_conn.commit()
                        db_conn.close()
                        if deleted > 0:
                            logger.info(f"Очищено {deleted} возможностей с низким приоритетом")
                except Exception as e:
                    logger.debug(f"Ошибка очистки низкоприоритетных возможностей: {e}")
        except Exception as e:
            logger.debug(f"Ошибка при очистке: {e}")
    
    def _worker_loop(self):
        """Основной рабочий цикл системы."""
        logger.info("Рабочий цикл самообучения запущен")
        
        while not self.stop_event.is_set():
            try:
                try:
                    task = self.dialog_queue.get(timeout=1)
                    self._process_task(task)
                except Exception as e:
                    logger.warning(f"Queue timeout in learning cycle: {e}")
                
                if self.auto_execute_enabled:
                    self._check_and_execute_learning_opportunities()
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Ошибка в рабочем цикле самообучения: {e}")
                time.sleep(5)
        
        logger.info("Рабочий цикл самообучения завершен")
    
    def _check_and_execute_learning_opportunities(self):
        """Проверяет и автоматически выполняет возможности для обучения."""
        current_time = time.time()
        if current_time - self.last_learning_check < self.auto_learning_interval:
            return
        
        self.last_learning_check = current_time
        
        try:
            opportunities = self._get_learning_opportunities()
            
            if not opportunities:
                return
            
            self.stats["opportunities_found"] += len(opportunities)
            logger.info(f"Найдено {len(opportunities)} возможностей для обучения")
            
            for opportunity in opportunities:
                self._execute_learning_opportunity(opportunity)
                
        except Exception as e:
            logger.error(f"Ошибка при проверке возможностей для обучения: {e}")
    
    def _get_learning_opportunities(self) -> List[Dict[str, Any]]:
        """Получает невыполненные возможности для обучения."""
        if not self.brain:
            return []
        
        opportunities = []
        min_priority = self.min_priority_threshold  # Используем настраиваемый порог
        
        if hasattr(self.brain, 'self_analyzer') and self.brain.self_analyzer:
            try:
                opportunities = self.brain.self_analyzer.get_learning_opportunities(
                    executed=False,
                    min_priority=min_priority,
                    limit=10
                )
            except Exception as e:
                logger.debug(f"Ошибка получения возможностей от self_analyzer: {e}")
        
        if not opportunities and hasattr(self.brain, 'analyzer_core') and self.brain.analyzer_core:
            try:
                opportunities = self.brain.analyzer_core.get_learning_opportunities(
                    executed=False,
                    min_priority=min_priority,
                    limit=10
                )
            except Exception as e:
                logger.debug(f"Ошибка получения возможностей от analyzer_core: {e}")
        
        if hasattr(self.brain, 'learning_opportunity_manager') and self.brain.learning_opportunity_manager:
            try:
                ml_opportunities = self.brain.learning_opportunity_manager.get_learning_opportunities(
                    status='pending',
                    limit=10
                )
                if ml_opportunities:
                    opportunities.extend(ml_opportunities)
            except Exception as e:
                logger.debug(f"Ошибка получения возможностей от learning_opportunity_manager: {e}")
        
        real_opportunities = [
            op for op in opportunities 
            if isinstance(op, dict) and op.get('id') is not None and not op.get('executed', False)
        ]
        
        return real_opportunities
    
    def _execute_learning_opportunity(self, opportunity: Dict[str, Any]):
        """Выполняет возможность для обучения."""
        try:
            opportunity_id = opportunity.get('id')
            concept = opportunity.get('concept', 'unknown')
            opportunity_type = opportunity.get('opportunity_type', 'expansion')
            priority = opportunity.get('priority', 0.5)
            domain = opportunity.get('domain', 'general')
            
            # Используем настраиваемый порог приоритета
            if priority < self.min_priority_threshold:
                logger.debug(f"Пропуск возможности с низким приоритетом: {concept} ({priority:.2f})")
                self._mark_opportunity_executed(opportunity_id, {"skipped": True, "reason": "low_priority"})
                return
            
            logger.info(f"Выполняется обучение: {concept} (тип: {opportunity_type}, приоритет: {priority:.2f})")
            
            learned_content = self._perform_learning(concept, opportunity_type, domain, opportunity)
            
            self._mark_opportunity_executed(opportunity_id, learned_content)
            
            self.stats["opportunities_executed"] += 1
            self.stats["successful_learning"] += 1
            
            logger.info(f"Обучение завершено: {concept}")
            
        except Exception as e:
            logger.error(f"Ошибка выполнения возможности обучения: {e}")
    
    def _perform_learning(self, concept: str, opportunity_type: str, domain: str, 
                         opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Выполняет процесс обучения для концепта."""
        result = {
            "concept": concept,
            "type": opportunity_type,
            "domain": domain,
            "learned": True,
            "timestamp": time.time()
        }
        
        if opportunity_type == "expansion":
            result["content"] = self._learn_expansion(concept, domain, opportunity)
        elif opportunity_type == "refinement":
            result["content"] = self._learn_refinement(concept, domain, opportunity)
        elif opportunity_type == "updating":
            result["content"] = self._learn_updating(concept, domain, opportunity)
        elif opportunity_type == "integration":
            result["content"] = self._learn_integration(concept, domain, opportunity)
        else:
            result["content"] = self._learn_generic(concept, domain, opportunity)
            result["learned"] = True
        
        self._store_learned_content(result)
        
        return result
    
    def _learn_expansion(self, concept: str, domain: str, opportunity: Dict[str, Any]) -> str:
        """Расширяет знания по концепту."""
        suggested_actions = opportunity.get('suggested_actions', [])
        
        if self.brain and hasattr(self.brain, 'knowledge_graph') and self.brain.knowledge_graph:
            try:
                self.brain.knowledge_graph.add_node(
                    concept,
                    node_type="learned_knowledge",
                    domain=domain,
                    metadata={
                        "learned_at": time.time(),
                        "type": "expansion",
                        "source": "self_dialog_learning"
                    }
                )
                logger.debug(f"Добавлен узел знаний: {concept}")
            except Exception as e:
                logger.debug(f"Ошибка добавления узла: {e}")
        
        learning_text = f"Расширение знаний: {concept}"
        if suggested_actions:
            learning_text += f". Действия: {'; '.join(suggested_actions[:2])}"
        
        return learning_text
    
    def _learn_refinement(self, concept: str, domain: str, opportunity: Dict[str, Any]) -> str:
        """Уточняет существующие знания."""
        evidence = opportunity.get('evidence', [])
        
        refinement_text = f"Уточнение знаний: {concept}"
        if evidence:
            refinement_text += f". Основание: {'; '.join(evidence[:2])}"
        
        if self.brain and hasattr(self.brain, 'knowledge_graph') and self.brain.knowledge_graph:
            try:
                self.brain.knowledge_graph.update_node(
                    concept,
                    metadata={
                        "refined_at": time.time(),
                        "type": "refinement",
                        "source": "self_dialog_learning",
                        "evidence": evidence
                    }
                )
            except Exception as e:
                logger.warning(f"Error updating knowledge graph in _learn_refining: {e}")
        
        return refinement_text
    
    def _learn_updating(self, concept: str, domain: str, opportunity: Dict[str, Any]) -> str:
        """Обновляет устаревшие знания."""
        suggested_actions = opportunity.get('suggested_actions', [])
        
        update_text = f"Обновление знаний: {concept}"
        if suggested_actions:
            update_text += f". Необходимые действия: {'; '.join(suggested_actions[:2])}"
        
        if self.brain:
            if hasattr(self.brain, 'adaptation_manager') and self.brain.adaptation_manager:
                try:
                    self.brain.adaptation_manager.update_adaptation(
                        concept,
                        {"updated": True, "updated_at": time.time()}
                    )
                except Exception as e:
                    logger.warning(f"Error updating adaptation: {e}")
            
            if hasattr(self.brain, 'knowledge_graph') and self.brain.knowledge_graph:
                try:
                    self.brain.knowledge_graph.update_node(
                        concept,
                        metadata={
                            "updated_at": time.time(),
                            "type": "updating",
                            "source": "self_dialog_learning"
                        }
                    )
                except Exception as e:
                    logger.warning(f"Error updating knowledge graph in _learn_updating: {e}")
        
        return update_text
    
    def _learn_integration(self, concept: str, domain: str, opportunity: Dict[str, Any]) -> str:
        """Интегрирует новые знания."""
        evidence = opportunity.get('evidence', [])
        
        integration_text = f"Интеграция знаний: {concept}"
        if evidence:
            integration_text += f". Связи: {'; '.join(evidence[:2])}"
        
        if self.brain and hasattr(self.brain, 'knowledge_graph') and self.brain.knowledge_graph:
            try:
                self.brain.knowledge_graph.add_node(
                    concept,
                    node_type="integrated_knowledge",
                    domain=domain,
                    metadata={
                        "integrated_at": time.time(),
                        "type": "integration",
                        "source": "self_dialog_learning",
                        "evidence": evidence
                    }
                )
            except Exception as e:
                logger.warning(f"Error adding node in _learn_integration: {e}")
        
        return integration_text
    
    def _learn_generic(self, concept: str, domain: str, opportunity: Dict[str, Any]) -> str:
        """Общее обучение."""
        suggested_actions = opportunity.get('suggested_actions', [])
        
        learning_text = f"Изучение концепта: {concept}"
        if suggested_actions:
            learning_text += f". Выполнено: {'; '.join(suggested_actions[:3])}"
        
        return learning_text
    
    def _store_learned_content(self, result: Dict[str, Any]):
        """Сохраняет изученный контент."""
        if not self.brain:
            return
        
        try:
            if hasattr(self.brain, 'memory_manager') and self.brain.memory_manager:
                try:
                    self.brain.memory_manager.store(
                        f"learned_{result['concept']}",
                        {
                            **result,
                            "stored_at": time.time(),
                            "source": "self_dialog_learning"
                        }
                    )
                except Exception as e:
                    logger.warning(f"Error storing learned content: {e}")
                    
        except Exception as e:
            logger.debug(f"Ошибка сохранения изученного контента: {e}")
    
    def _mark_opportunity_executed(self, opportunity_id: str, result: Dict[str, Any]):
        """Отмечает возможность как выполненную."""
        if not self.brain or not opportunity_id:
            return
        
        try:
            if hasattr(self.brain, 'self_analyzer') and self.brain.self_analyzer:
                if hasattr(self.brain.self_analyzer, 'analyzer_core') and self.brain.self_analyzer.analyzer_core:
                    self._mark_in_analyzer_core(opportunity_id, result)
                elif hasattr(self.brain.self_analyzer, 'learning_opportunity_manager'):
                    self._mark_in_learning_manager(opportunity_id, result)
            
            elif hasattr(self.brain, 'analyzer_core') and self.brain.analyzer_core:
                self._mark_in_analyzer_core(opportunity_id, result)
            
            elif hasattr(self.brain, 'learning_opportunity_manager'):
                self._mark_in_learning_manager(opportunity_id, result)
                
        except Exception as e:
            logger.debug(f"Ошибка отметки выполненной возможности: {e}")
    
    def _mark_in_analyzer_core(self, opportunity_id: str, result: Dict[str, Any]):
        """Отмечает возможность как выполненную в analyzer_core."""
        try:
            analyzer_core = None
            if hasattr(self.brain, 'self_analyzer') and self.brain.self_analyzer:
                analyzer_core = getattr(self.brain.self_analyzer, 'analyzer_core', None)
            elif hasattr(self.brain, 'analyzer_core'):
                analyzer_core = self.brain.analyzer_core
            
            if analyzer_core:
                db_path = getattr(analyzer_core, 'db_path', None)
                if db_path:
                    import sqlite3
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute('''
                        UPDATE learning_opportunities 
                        SET executed = 1, execution = ?, last_updated = ?
                        WHERE id = ?
                    ''', (json.dumps(result), time.time(), opportunity_id))
                    conn.commit()
                    conn.close()
        except Exception as e:
            logger.warning(f"Error marking opportunity executed: {e}")
    
    def _mark_in_learning_manager(self, opportunity_id: str, result: Dict[str, Any]):
        """Отмечает возможность как выполненную в learning_opportunity_manager."""
        try:
            if hasattr(self.brain.learning_opportunity_manager, 'execute_learning_opportunity'):
                self.brain.learning_opportunity_manager.execute_learning_opportunity(opportunity_id)
        except Exception as e:
            logger.warning(f"Error in _mark_in_learning_manager: {e}")
    
    def _process_task(self, task: Dict[str, Any]):
        """Обрабатывает задачу из очереди."""
        task_type = task.get("type")
        
        if task_type == "create_dialog":
            self._create_self_dialog(task.get("topic"), task.get("context"))
        elif task_type == "analyze_interaction":
            self._analyze_interaction(task.get("query"), task.get("response"), task.get("feedback"))
        elif task_type == "trigger_learning":
            self._trigger_learning(task.get("gap"))
    
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
    
    def _run_dialog(self, dialog: SelfDialog, context: Optional[Dict[str, Any]] = None):
        """Выполняет самодиалог."""
        assistant_prompt = self._generate_assistant_prompt(dialog.topic, context)
        
        turn_count = 0
        while turn_count < self.max_dialog_turns and len(dialog.turns) < self.max_dialog_turns:
            turn_count += 1
            
            if turn_count == 1:
                role = DialogRole.ASSISTANT
                content = self._simulate_assistant_response(assistant_prompt)
            elif turn_count == 2:
                role = DialogRole.CRITIC
                content = self._simulate_critic_response(dialog.turns, dialog.topic)
            elif turn_count == 3:
                role = DialogRole.LEARNER
                content = self._simulate_learner_response(dialog.turns, dialog.topic)
            elif turn_count == 4:
                role = DialogRole.TEACHER
                content = self._simulate_teacher_response(dialog.turns, dialog.topic)
            else:
                role = DialogRole.OBSERVER
                content = self._simulate_observer_response(dialog.turns, dialog.topic)
            
            turn = DialogTurn(
                role=role,
                content=content,
                timestamp=time.time()
            )
            turn.quality_score = self._assess_turn_quality(turn, dialog)
            
            dialog.turns.append(turn)
            
            if self._should_end_dialog(dialog):
                break
        
        self._analyze_dialog_outcome(dialog)
    
    def _generate_assistant_prompt(self, topic: str, context: Optional[Dict[str, Any]]) -> str:
        """Генерирует промпт для первого хода ассистента."""
        prompt = f"Тема: {topic}\n"
        
        if context:
            if "user_query" in context:
                prompt += f"Вопрос пользователя: {context['user_query']}\n"
            if "system_response" in context:
                prompt += f"Ответ системы: {context['system_response']}\n"
            if "knowledge_gaps" in context:
                prompt += f"Известные пробелы: {', '.join(context['knowledge_gaps'])}\n"
        
        prompt += "\nСистема анализирует эту тему..."
        return prompt
    
    def _simulate_assistant_response(self, prompt: str) -> str:
        """Симулирует ответ ассистента с использованием SelfReasoningEngine."""
        topic_match = prompt.split("Тема:")[1].split("\n")[0].strip() if "Тема:" in prompt else "тему"
        
        # Пробуем использовать SelfReasoningEngine
        if self.brain and hasattr(self.brain, 'self_reasoning_engine') and self.brain.self_reasoning_engine:
            try:
                result = self.brain.self_reasoning_engine.process_query(f"Объясни: {topic_match}")
                if result and result.get('response'):
                    return result['response'][:500]
            except Exception as e:
                logger.debug(f"Не удалось использовать SelfReasoningEngine: {e}")
        
        # Пробуем использовать process_query
        if self.brain and hasattr(self.brain, 'process_query'):
            try:
                response = self.brain.process_query(f"Объясни: {topic_match}")
                if response and isinstance(response, dict):
                    response = response.get('response', response.get('text', ''))
                return response[:500] if response and len(response) > 500 else (response or '')
            except Exception as e:
                logger.debug(f"Не удалось получить ответ от brain: {e}")
        
        # Fallback - простой ответ
        return f"Анализ темы '{topic_match}': система исследует базовые аспекты проблемы."
    
    def _simulate_critic_response(self, turns: List[DialogTurn], topic: str) -> str:
        """Симулирует ответ критика с проверкой противоречий."""
        if not turns:
            return "Нет данных для критики."
        
        last_content = turns[-1].content if turns else ""
        
        # Проверяем противоречия если доступен ContradictionManager
        contradictions_found = []
        if self.brain and hasattr(self.brain, 'contradiction_manager') and self.brain.contradiction_manager:
            try:
                contr_result = self.brain.contradiction_manager.detect_contradictions()
                if contr_result and isinstance(contr_result, dict):
                    contradictions = contr_result.get('contradictions', [])
                    if contradictions:
                        contradictions_found = [str(c)[:50] for c in contradictions[:3]]
            except Exception as e:
                logger.debug(f"Ошибка проверки противоречий: {e}")
        
        gaps = self._identify_knowledge_gaps(last_content, topic)
        
        result_parts = []
        if contradictions_found:
            result_parts.append(f"Противоречия: {', '.join(contradictions_found)}")
        
        gaps_text = "; ".join(gaps) if gaps else "не выявлены"
        result_parts.append(f"Пробелы в знаниях: {gaps_text}")
        
        return "Критический анализ: " + ". ".join(result_parts) + "."
    
    def _simulate_learner_response(self, turns: List[DialogTurn], topic: str) -> str:
        """Симулирует ответ обучающегося."""
        gaps = self._identify_knowledge_gaps("", topic)
        
        if gaps:
            actions = [f"Изучить: {gap}" for gap in gaps[:3]]
            return f"План обучения: {'; '.join(actions)}."
        
        return "Обучение: текущие знания достаточны для данной темы."
    
    def _simulate_teacher_response(self, turns: List[DialogTurn], topic: str) -> str:
        """Симулирует ответ учителя."""
        critic_feedback = ""
        learner_plan = ""
        
        for turn in turns:
            if turn.role == DialogRole.CRITIC:
                critic_feedback = turn.content
            elif turn.role == DialogRole.LEARNER:
                learner_plan = turn.content
        
        gaps = self._identify_knowledge_gaps("", topic)
        
        if gaps:
            recommendations = [f"Рекомендация: углубить знания в области {gap}" for gap in gaps[:2]]
            return f"Наставление: {'; '.join(recommendations)}."
        
        return "Текущий уровень знаний соответствует требованиям."
    
    def _simulate_observer_response(self, turns: List[DialogTurn], topic: str) -> str:
        """Симулирует ответ наблюдателя."""
        avg_quality = sum(t.quality_score for t in turns) / len(turns) if turns else 0
        
        if avg_quality >= self.min_quality_threshold:
            outcome = "положительный"
        elif avg_quality >= 0.4:
            outcome = "нейтральный"
        else:
            outcome = "требует улучшения"
        
        gaps = self._identify_knowledge_gaps("", topic)
        
        return f"Наблюдение: общий исход диалога {outcome}. Требуется работа над {len(gaps)} аспектами."
    
    def _identify_knowledge_gaps(self, content: str, topic: str) -> List[str]:
        """Выявляет пробелы в знаниях."""
        gaps = []
        
        topic_lower = topic.lower()
        
        if "анализ" in topic_lower or "analysis" in topic_lower:
            gaps.append("методы анализа данных")
        if "обучение" in topic_lower or "learning" in topic_lower:
            gaps.append("алгоритмы машинного обучения")
        if "знание" in topic_lower or "knowledge" in topic_lower:
            gaps.append("управление знаниями")
        if any(word in topic_lower for word in ["когнитив", "cognitive", "мышление", "thinking"]):
            gaps.append("когнитивные функции")
        
        if not gaps:
            gaps.append("общие концепции предметной области")
        
        self.stats["knowledge_gaps_identified"] += len(gaps)
        
        return gaps
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Извлекает сущности из текста простыми методами.
        
        Args:
            text: Входной текст для анализа
            
        Returns:
            Словарь с типами сущностей и списком найденных сущностей
        """
        entities = {
            "persons": [],
            "organizations": [],
            "locations": [],
            "concepts": [],
            "technologies": [],
            "numbers": []
        }
        
        if not text:
            return entities
        
        text_lower = text.lower()
        words = text.split()
        
        person_patterns = [
            "человек", "пользователь", "специалист", "эксперт", "ученый",
            "разработчик", "администратор", "аналитик", "инженер"
        ]
        
        org_patterns = [
            "компания", "организация", "команда", "группа", "институт",
            "университет", "лаборатория", "предприятие", "корпорация"
        ]
        
        location_patterns = [
            "страна", "город", "регион", "область", "континент",
            "европа", "азия", "россия", "сша", "германия", "франция"
        ]
        
        concept_patterns = [
            "концепция", "принцип", "теория", "модель", "система",
            "метод", "подход", "стратегия", "алгоритм", "процесс"
        ]
        
        tech_patterns = [
            "python", "java", "c++", "javascript", "tensorflow", "pytorch",
            "神经网络", "машинное обучение", "глубинное обучение", "nlp",
            "transformer", "bert", "gpt", "llm"
        ]
        
        for i, word in enumerate(words):
            word_lower = word.lower().strip(".,!?;:()[]{}")
            
            for pattern in person_patterns:
                if pattern in word_lower:
                    entities["persons"].append(word)
                    break
            
            for pattern in org_patterns:
                if pattern in word_lower:
                    entities["organizations"].append(word)
                    break
            
            for pattern in location_patterns:
                if pattern in word_lower:
                    entities["locations"].append(word)
                    break
            
            for pattern in concept_patterns:
                if pattern in word_lower:
                    entities["concepts"].append(word)
                    break
            
            for pattern in tech_patterns:
                if pattern in word_lower:
                    entities["technologies"].append(word)
                    break
            
            if word_lower.isdigit() and len(word_lower) > 1:
                entities["numbers"].append(word_lower)
        
        for key in entities:
            entities[key] = list(set(entities[key]))[:10]
        
        return entities
    
    def _assess_turn_quality(self, turn: DialogTurn, dialog: SelfDialog) -> float:
        """Оценивает качество хода диалога."""
        score = 0.5
        
        if len(turn.content) > 50:
            score += 0.1
        
        if any(word in turn.content.lower() for word in ["анализ", "проблем", "решени", "вывод"]):
            score += 0.2
        
        if turn.role == DialogRole.CRITIC and "пробел" in turn.content.lower():
            score += 0.1
        
        if turn.role == DialogRole.LEARNER and ("изучить" in turn.content.lower() or "план" in turn.content.lower()):
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _should_end_dialog(self, dialog: SelfDialog) -> bool:
        """Определяет, нужно ли завершить диалог."""
        if len(dialog.turns) >= self.max_dialog_turns:
            return True
        
        if len(dialog.turns) >= 4:
            observer_turns = [t for t in dialog.turns if t.role == DialogRole.OBSERVER]
            if observer_turns and observer_turns[-1].quality_score >= self.min_quality_threshold:
                return True
        
        return False
    
    def _analyze_dialog_outcome(self, dialog: SelfDialog):
        """Анализирует исход диалога."""
        if not dialog.turns:
            dialog.outcome = "no_content"
            dialog.learning_type = None
            return
        
        avg_quality = sum(t.quality_score for t in dialog.turns) / len(dialog.turns)
        
        if avg_quality >= self.min_quality_threshold:
            dialog.outcome = "successful"
            
            if avg_quality >= 0.8:
                dialog.learning_type = LearningType.REFINEMENT
            else:
                dialog.learning_type = LearningType.EXPANSION
        else:
            dialog.outcome = "needs_improvement"
            dialog.learning_type = LearningType.UPDATING
        
        gaps = []
        for turn in dialog.turns:
            if turn.role == DialogRole.CRITIC:
                gap_text = turn.content
                if "пробелы" in gap_text.lower():
                    gap_start = gap_text.lower().find(":") + 1
                    gap_end = gap_text.find(".")
                    if gap_start > 0 and gap_end > gap_start:
                        gap_content = gap_text[gap_start:gap_end].strip()
                        gaps = [g.strip() for g in gap_content.split(";") if g.strip()]
        
        dialog.knowledge_gaps = gaps
        
        actions = []
        for turn in dialog.turns:
            if turn.role == DialogRole.LEARNER:
                if "план" in turn.content.lower():
                    actions.append("Создан план обучения")
                if "изучить" in turn.content.lower():
                    actions.append("Инициировано изучение нового материала")
            elif turn.role == DialogRole.TEACHER:
                if "рекомендац" in turn.content.lower():
                    actions.append("Даны рекомендации по улучшению")
        
        dialog.actions_taken = actions
        
        if dialog.outcome == "successful":
            self.stats["successful_learning"] += 1
        
        self.stats["actions_taken"] += len(actions)
    
    def _finalize_dialog(self, dialog: SelfDialog):
        """Финализирует диалог и выполняет действия."""
        if dialog.id in self.active_dialogs:
            del self.active_dialogs[dialog.id]
        
        self.dialog_history.append(dialog)
        
        if len(self.dialog_history) > self.max_history_size:
            self.dialog_history = self.dialog_history[-self.max_history_size:]
        
        for gap in dialog.knowledge_gaps:
            self._trigger_learning(gap)
        
        for callback in self.learning_callbacks:
            try:
                callback(dialog)
            except Exception as e:
                logger.error(f"Ошибка в callback обучения: {e}")
        
        logger.info(f"Самодиалог завершен: {dialog.id}, исход: {dialog.outcome}")
    
    def _trigger_learning(self, gap: str):
        """Запускает процесс обучения для указанного пробела."""
        if not self.brain:
            return
        
        try:
            if hasattr(self.brain, 'analyzer_core') and self.brain.analyzer_core:
                self.brain.analyzer_core.add_learning_opportunity(
                    concept=gap,
                    opportunity_type="expansion",
                    priority=0.7,
                    domain="self_dialog_learning",
                    evidence=[f"Выявлено в самодиалоге: {gap}"],
                    suggested_actions=[f"Изучить {gap}", "Обновить связанные знания"]
                )
                logger.info(f"Добавлена возможность обучения: {gap}")
        except Exception as e:
            logger.error(f"Ошибка добавления возможности обучения: {e}")
    
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
