"""Learning from dialogs, knowledge extraction, and graph updates for self-dialog learning."""
from __future__ import annotations

import logging
import time
import json
import os
from typing import Dict, List, Any, Optional

from eva.learning.dialog_types import DialogRole, DialogTurn, LearningType, SelfDialog

logger = logging.getLogger("eva.self_dialog_learning")


class DialogLearningMixin:
    """Mixin for learning from dialogs, knowledge extraction, and graph updates."""

    def _get_fractal_graph(self):
        """Получить FractalGraphV2 если доступен."""
        if not self.brain:
            return None
        return getattr(self.brain, 'fractal_graph_v2', None)

    def _add_to_fractal_graph(self, content: str, node_type: str = "learned",
                               level: int = 1, metadata: Dict = None) -> Optional[str]:
        """Добавить знание в FractalGraphV2."""
        fg = self._get_fractal_graph()
        if not fg or not hasattr(fg, 'add_node'):
            return None
        
        try:
            node = fg.add_node(
                content=content[:500],
                node_type=node_type,
                level=level,
                confidence=0.7,
                metadata=metadata or {},
                auto_vectorize=True,
                auto_cluster=True
            )
            logger.debug(f"Added to FractalGraphV2: {content[:50]}...")
            return node.id if node else None
        except Exception as e:
            logger.debug(f"Error adding to FractalGraphV2: {e}")
            return None

    def _check_contradiction_fractal(self, content: str) -> Dict[str, Any]:
        """Проверить противоречие в FractalGraphV2."""
        fg = self._get_fractal_graph()
        if not fg or not hasattr(fg, 'check_contradiction'):
            return {"is_contradiction": False, "reason": "FractalGraphV2 not available"}
        
        try:
            return fg.check_contradiction(content)
        except Exception as e:
            logger.debug(f"Error checking contradiction: {e}")
            return {"is_contradiction": False, "error": str(e)}

    def _check_and_execute_learning_opportunities(self) -> None:
        """Проверяет и автоматически выполняет возможности для обучения."""
        current_time = time.time()
        if current_time - self.last_learning_check < self.auto_learning_interval:
            return

        self.last_learning_check = current_time

        try:
            opportunities = self._get_learning_opportunities()

            if not opportunities:
                self._generate_dialog_from_conversations()
                return

            self.stats["opportunities_found"] += len(opportunities)
            if len(opportunities) > 0:
                logger.debug(f"Найдено {len(opportunities)} возможностей для обучения")

            for opportunity in opportunities:
                self._execute_learning_opportunity(opportunity)

        except Exception as e:
            logger.error(f"Ошибка при проверке возможностей для обучения: {e}")

    def _get_learning_opportunities(self) -> List[Dict[str, Any]]:
        """Получает невыполненные возможности для обучения."""
        if not self.brain:
            return []

        opportunities = []
        min_priority = self.min_priority_threshold

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
                ml_opportunities = self.brain.learning_opportunity_manager.get_learning_opportunities()
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
            if not opportunity_id:
                return
            concept = opportunity.get('concept', 'unknown')
            opportunity_type = opportunity.get('opportunity_type', 'expansion')
            priority = opportunity.get('priority', 0.5)
            domain = opportunity.get('domain', 'general')

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

        # Используем FractalGraphV2 если доступен
        fg = self._get_fractal_graph()
        if fg and hasattr(fg, 'add_node'):
            try:
                # Проверяем на противоречия
                contradiction = self._check_contradiction_fractal(concept)
                if contradiction and contradiction.get('is_contradiction'):
                    logger.info(f"Обнаружено противоречие при расширении: {concept}")
                
                node = fg.add_node(
                    content=concept,
                    node_type="learned_knowledge",
                    level=1,
                    confidence=0.7,
                    metadata={
                        "learned_at": time.time(),
                        "type": "expansion",
                        "source": "self_dialog_learning",
                        "domain": domain
                    },
                    auto_vectorize=True,
                    auto_cluster=True
                )
                logger.debug(f"Добавлен узел в FractalGraphV2: {concept}")
            except Exception as e:
                logger.debug(f"Ошибка добавления в FractalGraphV2: {e}")

        # Также добавляем в классический knowledge_graph для совместимости
        knowledge_graph = getattr(self.brain, 'knowledge_graph', None) if self.brain else None
        if knowledge_graph:
            try:
                add_node = getattr(knowledge_graph, 'add_node', None)
                if add_node:
                    add_node(
                        name=concept,
                        node_type="learned_knowledge",
                        domain=domain,
                        meta={
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

        # FractalGraphV2: используем self_dialogue для проверки и уточнения
        fg = self._get_fractal_graph()
        if fg and hasattr(fg, 'self_dialogue'):
            try:
                result = fg.self_dialogue(f"Уточнение: {concept}")
                logger.debug(f"Self-dialogue for refinement: {result.get('action', 'none')}")
            except Exception as e:
                logger.debug(f"Error in self_dialogue: {e}")

        # Классический knowledge_graph
        knowledge_graph = getattr(self.brain, 'knowledge_graph', None) if self.brain else None
        if knowledge_graph:
            try:
                if hasattr(knowledge_graph, 'update_node') and hasattr(knowledge_graph, 'search_nodes'):
                    nodes = knowledge_graph.search_nodes(concept, limit=1)
                    if nodes and len(nodes) > 0:
                        node = nodes[0]
                        node_id = getattr(node, 'id', None)
                        if node_id:
                            knowledge_graph.update_node(
                                node_id=node_id,
                                new_description=f"Refined at {time.time()}, type: refinement, evidence: {evidence}",
                                source="self_dialog_learning"
                            )
                elif hasattr(knowledge_graph, 'add_node'):
                    knowledge_graph.add_node(
                        name=concept,
                        description=f"Refined at {time.time()}, type: refinement, evidence: {evidence}",
                        source="self_dialog_learning"
                    )
            except Exception as e:
                logger.warning(f"Error updating knowledge graph in _learn_refining: {e}")

        if not self.brain:
            return refinement_text

        memory_manager = getattr(self.brain, 'memory_manager', None)
        if memory_manager and hasattr(memory_manager, 'add_memory'):
            try:
                memory_manager.add_memory(
                    "semantic",
                    {
                        "concept": concept,
                        "type": "refinement",
                        "domain": domain,
                        "evidence": evidence,
                        "refined_at": time.time(),
                        "source": "self_dialog_learning"
                    },
                    {"type": "refinement"},
                    None
                )
            except Exception as e:
                logger.debug(f"Error storing refinement: {e}")

        return refinement_text

    def _learn_updating(self, concept: str, domain: str, opportunity: Dict[str, Any]) -> str:
        """Обновляет устаревшие знания."""
        suggested_actions = opportunity.get('suggested_actions', [])

        update_text = f"Обновление знаний: {concept}"
        if suggested_actions:
            update_text += f". Необходимые действия: {'; '.join(suggested_actions[:2])}"

        if self.brain:
            adaptation_manager = getattr(self.brain, 'adaptation_manager', None)
            if adaptation_manager:
                try:
                    update_adaptation = getattr(adaptation_manager, 'update_adaptation', None)
                    if update_adaptation:
                        update_adaptation(
                            concept,
                            {"updated": True, "updated_at": time.time()}
                        )
                except Exception as e:
                    logger.warning(f"Error updating adaptation: {e}")

            if hasattr(self.brain, 'knowledge_graph') and self.brain.knowledge_graph:
                try:
                    if hasattr(self.brain.knowledge_graph, 'update_node') and hasattr(self.brain.knowledge_graph, 'search_nodes'):
                        nodes = self.brain.knowledge_graph.search_nodes(concept, limit=1)
                        if nodes and len(nodes) > 0:
                            node = nodes[0]
                            node_id = getattr(node, 'id', None)
                            if node_id:
                                self.brain.knowledge_graph.update_node(
                                    node_id=node_id,
                                    new_description=f"Updated at {time.time()}, type: updating",
                                    source="self_dialog_learning"
                                )
                    elif hasattr(self.brain.knowledge_graph, 'add_node'):
                        self.brain.knowledge_graph.add_node(
                            name=concept,
                            description=f"Updated at {time.time()}, type: updating",
                            source="self_dialog_learning"
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
                if hasattr(self.brain.knowledge_graph, 'add_node'):
                    node_id = f"integrated_{hash(concept) % 1000000}"
                    self.brain.knowledge_graph.add_node(
                        name=concept,
                        node_id=node_id,
                        node_type="integrated_knowledge",
                        domain=domain,
                        meta={
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
            memory_manager = getattr(self.brain, 'memory_manager', None)
            if memory_manager:
                try:
                    add_memory = getattr(memory_manager, 'add_memory', None)
                    if add_memory:
                        add_memory(
                            "semantic",
                            {
                                **result,
                                "stored_at": time.time(),
                                "source": "self_dialog_learning"
                            },
                            {"type": "learned_content"},
                            None
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
                if db_path and os.path.exists(db_path):
                    import sqlite3
                    with sqlite3.connect(db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute('''
                            UPDATE learning_opportunities
                            SET executed = 1, execution = ?, last_updated = ?
                            WHERE id = ?
                        ''', (json.dumps(result), time.time(), opportunity_id))
                        conn.commit()
        except Exception as e:
            logger.warning(f"Error marking opportunity executed: {e}")

    def _mark_in_learning_manager(self, opportunity_id: str, result: Dict[str, Any]):
        """Отмечает возможность как выполненную в learning_opportunity_manager."""
        try:
            if getattr(self.brain, 'learning_opportunity_manager', None) and hasattr(self.brain.learning_opportunity_manager, 'execute_learning_opportunity'):
                self.brain.learning_opportunity_manager.execute_learning_opportunity(opportunity_id)
        except Exception as e:
            logger.warning(f"Error in _mark_in_learning_manager: {e}")

    def _trigger_learning(self, gap: str) -> None:
        """Запускает процесс обучения для указанного пробела."""
        if not self.brain:
            return

        try:
            if hasattr(self.brain, 'analyzer_core') and self.brain.analyzer_core:
                if hasattr(self.brain.analyzer_core, 'add_learning_opportunity'):
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
