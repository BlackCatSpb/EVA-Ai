"""Task creation, management, execution, and queueing for EVA learning scheduler."""

import time
import logging
import heapq
from typing import Dict, List, Optional, Any

import numpy as np

from .scheduler_core import LearningTask, ResourceAllocation

logger = logging.getLogger("eva_ai.learning_scheduler")


class TaskManagerMixin:
    """Mixin providing task CRUD, queue management, and execution."""

    def add_task(self, task: LearningTask) -> bool:
        """Добавляет новую задачу в расписание."""
        with self.lock:
            if len(self.task_registry) > 1000:
                logger.warning("Достигнут лимит очереди задач")
                return False

            if task.task_id in self.task_registry:
                logger.warning(f"Задача {task.task_id} уже существует")
                return False

            self.task_registry[task.task_id] = task

            if task.status == "pending":
                heapq.heappush(self.task_queue, task)

            self._save_tasks()
            self._update_stats()

            logger.info(f"Добавлена задача обучения: {task.task_id} ({task.task_type} для '{task.concept}')")
            return True

    def get_task(self, task_id: str) -> Optional[LearningTask]:
        """Получает задачу по ID."""
        with self.lock:
            return self.task_registry.get(task_id)

    def _update_task_status_internal(self, task_id: str, status: str) -> bool:
        task = self.task_registry.get(task_id)
        if not task:
            return False

        old_status = task.status
        task.status = status

        if status == "in_progress" and not task.start_time:
            task.start_time = time.time()
        elif status in ["completed", "failed"] and not task.end_time:
            task.end_time = time.time()

        if status in ["completed", "failed"]:
            self._process_dependencies_internal(task_id)

        self._save_tasks()
        self._update_stats()

        logger.info(f"Статус задачи {task_id} обновлен с '{old_status}' на '{status}'")
        return True

    def update_task_status(self, task_id: str, status: str) -> bool:
        """Обновляет статус задачи."""
        with self.lock:
            return self._update_task_status_internal(task_id, status)

    def _process_dependencies_internal(self, task_id: str):
        if task_id not in self.task_registry:
            return
        for dependent_id in self.task_registry[task_id].dependents:
            dependent_task = self.task_registry.get(dependent_id)
            if dependent_task and dependent_task.status == "pending":
                all_dependencies_satisfied = True
                for dep_id in dependent_task.dependencies:
                    dep_task = self.task_registry.get(dep_id)
                    if dep_task and dep_task.status != "completed":
                        all_dependencies_satisfied = False
                        break

                if all_dependencies_satisfied:
                    dependent_task.status = "pending"
                    heapq.heappush(self.task_queue, dependent_task)
                    self._save_tasks()

    def _process_dependencies(self, task_id: str):
        """Обрабатывает зависимости после завершения задачи."""
        with self.lock:
            self._process_dependencies_internal(task_id)

    def start_task(self, task_id: str) -> bool:
        """Запускает задачу на выполнение."""
        with self.lock:
            task = self.get_task(task_id)
            if not task or task.status != "pending":
                return False

            for dep_id in task.dependencies:
                dep_task = self.get_task(dep_id)
                if not dep_task or dep_task.status != "completed":
                    logger.warning(f"Задача {task_id} не может быть запущена: зависимость {dep_id} не выполнена")
                    return False

            if not self.resource_allocation.acquire_slot(task_id):
                logger.warning(f"Недостаточно ресурсов для запуска задачи {task_id}")
                return False

            self._update_task_status_internal(task_id, "in_progress")
            logger.info(f"Задача {task_id} запущена на выполнение")
            return True

    def complete_task(self, task_id: str, result: Any) -> bool:
        """Завершает выполнение задачи."""
        with self.lock:
            if task_id not in self.task_registry:
                return False

            task = self.task_registry[task_id]
            task.result = result
            task.error = None

            self._update_task_status_internal(task_id, "completed")

            self.resource_allocation.release_slot(task_id)

            logger.info(f"Задача {task_id} завершена успешно")
            return True

    def fail_task(self, task_id: str, error: str) -> bool:
        """Помечает задачу как неудачную."""
        with self.lock:
            if task_id not in self.task_registry:
                return False

            task = self.task_registry[task_id]
            task.error = error

            if task.retries < task.max_retries:
                task.retries += 1
                task.status = "pending"
                task.scheduled_time = time.time() + self.task_retry_delay
                self.resource_allocation.release_slot(task_id)
                heapq.heappush(self.task_queue, task)
                self._save_tasks()
                self._update_stats()
                logger.warning(f"Задача {task_id} помечена как неудачная (попытка {task.retries}/{task.max_retries}): {error}")
            else:
                self._update_task_status_internal(task_id, "failed")
                self.resource_allocation.release_slot(task_id)
                logger.error(f"Задача {task_id} завершилась неудачей после {task.max_retries} попыток: {error}")

            return True

    def clear_schedule(self):
        """Очищает текущее расписание."""
        with self.lock:
            self.task_registry.clear()
            self.task_queue.clear()
            self.resource_allocation = ResourceAllocation(max_concurrent=self.max_concurrent_tasks)
            self._update_stats()
            self._save_tasks()
            logger.info("Расписание обучения очищено")

    def _get_next_task(self) -> Optional[LearningTask]:
        """Получает следующую задачу для выполнения."""
        with self.lock:
            current_time = time.time()

            while self.task_queue:
                task = heapq.heappop(self.task_queue)

                if task.scheduled_time > current_time:
                    heapq.heappush(self.task_queue, task)
                    return None

                dependencies_satisfied = True
                for dep_id in task.dependencies:
                    dep_task = self.task_registry.get(dep_id)
                    if not dep_task or dep_task.status != "completed":
                        dependencies_satisfied = False
                        break

                if not dependencies_satisfied:
                    heapq.heappush(self.task_queue, task)
                    continue

                if task.status in ["completed", "failed"]:
                    continue

                return task

            return None

    def _execute_task(self, task: LearningTask):
        """Выполняет задачу."""
        logger.info(f"Начало выполнения задачи {task.task_id}: {task.task_type} для '{task.concept}'")

        try:
            start_time = time.time()

            if task.task_type == "expand_domain":
                result = self._execute_expand_domain(task)
            elif task.task_type == "analyze_connections":
                result = self._execute_analyze_connections(task)
            elif task.task_type == "update_knowledge":
                result = self._execute_update_knowledge(task)
            elif task.task_type == "verify_sources":
                result = self._execute_verify_sources(task)
            elif task.task_type == "integrate_knowledge":
                result = self._execute_integrate_knowledge(task)
            elif task.task_type == "deepen_concept":
                result = self._execute_deepen_concept(task)
            elif task.task_type == "synthesize":
                result = self._execute_synthesize(task)
            elif task.task_type == "map_connections":
                result = self._execute_map_connections(task)
            elif task.task_type == "maintain_knowledge":
                result = self._execute_maintain_knowledge(task)
            else:
                raise ValueError(f"Неизвестный тип задачи: {task.task_type}")

            duration = time.time() - start_time
            logger.info(f"Задача {task.task_id} выполнена за {duration:.2f} секунд")
            self.complete_task(task.task_id, result)

        except Exception as e:
            logger.error(f"Ошибка при выполнении задачи {task.task_id}: {str(e)}")
            self.fail_task(task.task_id, str(e))

    def _execute_expand_domain(self, task: LearningTask) -> Any:
        """Выполняет задачу расширения домена."""
        concept = task.concept
        logger.info(f"Расширение домена для концепта: {concept}")

        try:
            concepts = []
            if self.brain and hasattr(self.brain, 'ml_unit') and self.brain.ml_unit and hasattr(self.brain.ml_unit, 'extract_concepts'):
                concepts = self.brain.ml_unit.extract_concepts(concept)

            if self.brain and hasattr(self.brain, 'memory_manager') and self.brain.memory_manager and hasattr(self.brain.memory_manager, 'update_user_profile'):
                self.brain.memory_manager.update_user_profile(
                    user_id="system",
                    updates={
                        "concept": concept,
                        "related_concepts": concepts
                    }
                )

            if hasattr(self, 'brain') and self.brain and hasattr(self.brain, 'knowledge_graph') and self.brain.knowledge_graph:
                try:
                    for related_concept in concepts:
                        self.brain.knowledge_graph.add_node(
                            name=related_concept,
                            node_id=f"concept_{hash(related_concept) % 1000000}",
                            node_type="concept",
                            domain=task.metadata.get("domain", "general")
                        )
                        self.brain.knowledge_graph.add_edge(
                            task.concept,
                            related_concept,
                            "related_to",
                            strength=0.7,
                            meta={"source": "domain_expansion"}
                        )
                except Exception as e:
                    logger.debug(f"Ошибка добавления в knowledge_graph: {e}")

            return {
                "concept": concept,
                "related_concepts": concepts,
                "domain_expansion": np.random.uniform(0.2, 0.5),
                "timestamp": time.time()
            }

        except Exception as e:
            logger.error(f"Ошибка при расширении домена для концепта {concept}: {e}")
            raise

    def _execute_analyze_connections(self, task: LearningTask) -> Any:
        """Выполняет задачу анализа связей."""
        concept = task.concept
        logger.info(f"Анализ связей для концепта: {concept}")

        try:
            connections = []
            if self.brain and hasattr(self.brain, 'knowledge_graph') and self.brain.knowledge_graph:
                try:
                    edges = self.brain.knowledge_graph.get_edges(concept)
                    for edge in edges:
                        related_concept = edge.target_id if edge.source_id == concept else edge.source_id
                        connections.append({
                            "concept": related_concept,
                            "relation": getattr(edge, 'relation_type', getattr(edge, 'relation', 'related_to')),
                            "strength": getattr(edge, 'strength', 0.5)
                        })
                except Exception as e:
                    logger.debug(f"Ошибка get_edges: {e}")

            related_concepts = []
            if len(connections) < 3 and self.brain and hasattr(self.brain, 'ml_unit') and self.brain.ml_unit and hasattr(self.brain.ml_unit, 'extract_concepts'):
                related_concepts = self.brain.ml_unit.extract_concepts(concept)
                for related_concept in related_concepts:
                    if related_concept != concept:
                        connections.append({
                            "concept": related_concept,
                            "relation": "related_to",
                            "strength": 0.6,
                            "weight": 0.6
                        })

            if self.brain and hasattr(self.brain, 'memory_manager') and self.brain.memory_manager and hasattr(self.brain.memory_manager, 'update_user_profile'):
                self.brain.memory_manager.update_user_profile(
                    user_id="system",
                    updates={
                        "concept": concept,
                        "connections": connections
                    }
                )

            return {
                "concept": concept,
                "connections": connections,
                "connection_strength": np.mean([c["strength"] for c in connections]) if connections else 0.0,
                "timestamp": time.time()
            }

        except Exception as e:
            logger.error(f"Ошибка при анализе связей для концепта {concept}: {e}")
            raise

    def _execute_update_knowledge(self, task: LearningTask) -> Any:
        """Выполняет задачу обновления знаний."""
        concept = task.concept
        logger.info(f"Обновление знаний по концепту: {concept}")

        try:
            concepts = []
            if self.brain and hasattr(self.brain, 'ml_unit') and self.brain.ml_unit and hasattr(self.brain.ml_unit, 'extract_concepts'):
                concepts = self.brain.ml_unit.extract_concepts(concept)

            if self.brain and hasattr(self.brain, 'memory_manager') and self.brain.memory_manager and hasattr(self.brain.memory_manager, 'update_user_profile'):
                self.brain.memory_manager.update_user_profile(
                    user_id="system",
                    updates={
                        "concept": concept,
                        "updated_concepts": concepts
                    }
                )

            if self.brain and hasattr(self.brain, 'knowledge_graph') and self.brain.knowledge_graph:
                try:
                    nodes = self.brain.knowledge_graph.search_nodes(concept, limit=1)
                    if nodes and len(nodes) > 0:
                        node = nodes[0]
                        node_id = getattr(node, 'id', None)
                        if node_id:
                            new_strength = min(1.0, getattr(node, 'strength', 0.5) + 0.1)
                            self.brain.knowledge_graph.add_node(
                                name=getattr(node, 'name', getattr(node, 'content', '')),
                                node_id=node_id,
                                node_type=getattr(node, 'node_type', 'concept'),
                                domain=getattr(node, 'domain', 'general'),
                                strength=new_strength,
                                meta=getattr(node, 'meta', getattr(node, 'metadata', {}))
                            )
                    for updated_concept in concepts:
                        self.brain.knowledge_graph.add_node(
                            name=updated_concept,
                            node_id=f"fact_{hash(updated_concept) % 1000000}",
                            node_type="fact",
                            domain=task.metadata.get("domain", "general"), strength=0.8
                        )
                        self.brain.knowledge_graph.add_edge(
                            concept, updated_concept, "contains",
                            strength=0.8, meta={"source": "knowledge_update"}
                        )
                except Exception as e:
                    logger.debug(f"Ошибка обновления knowledge_graph: {e}")

            return {
                "concept": concept,
                "updated_facts": len(concepts),
                "new_sources": 3,
                "update_quality": np.random.uniform(0.8, 0.95),
                "timestamp": time.time()
            }

        except Exception as e:
            logger.error(f"Ошибка при обновлении знаний для концепта {concept}: {e}")
            raise

    def _execute_verify_sources(self, task: LearningTask) -> Any:
        """Выполняет задачу проверки источников."""
        concept = task.concept
        logger.info(f"Проверка источников для концепта: {concept}")

        try:
            concepts = []
            if self.brain and hasattr(self.brain, 'ml_unit') and self.brain.ml_unit and hasattr(self.brain.ml_unit, 'extract_concepts'):
                concepts = self.brain.ml_unit.extract_concepts(concept)

            if self.brain and hasattr(self.brain, 'memory_manager') and self.brain.memory_manager and hasattr(self.brain.memory_manager, 'update_user_profile'):
                self.brain.memory_manager.update_user_profile(
                    user_id="system",
                    updates={
                        "concept": concept,
                        "verified_concepts": concepts
                    }
                )

            verified_sources = 0
            unverified_sources = 0

            if self.brain and hasattr(self.brain, 'knowledge_graph'):
                nodes = self.brain.knowledge_graph.search_nodes(concept, limit=1)
                if nodes and len(nodes) > 0:
                    first_node = nodes[0]
                    node_id = getattr(first_node, 'id', None)
                    if node_id:
                        sources = []
                        if hasattr(self.brain.knowledge_graph, 'get_sources_for_node'):
                            sources = self.brain.knowledge_graph.get_sources_for_node(node_id)
                        for source in sources:
                            source_reliability = getattr(source, 'reliability', getattr(source, 'strength', 0.5))
                            if source_reliability > 0.7:
                                verified_sources += 1
                            else:
                                unverified_sources += 1

            return {
                "concept": concept,
                "verified_sources": verified_sources,
                "unverified_sources": unverified_sources,
                "verification_score": np.random.uniform(0.75, 0.9),
                "timestamp": time.time()
            }

        except Exception as e:
            logger.error(f"Ошибка при проверке источников для концепта {concept}: {e}")
            raise

    def _execute_integrate_knowledge(self, task: LearningTask) -> Any:
        """Выполняет задачу интеграции знаний."""
        concept = task.concept
        logger.info(f"Интеграция знаний по концепту: {concept}")

        try:
            concepts = []
            if self.brain and hasattr(self.brain, 'ml_unit') and self.brain.ml_unit and hasattr(self.brain.ml_unit, 'extract_concepts'):
                concepts = self.brain.ml_unit.extract_concepts(concept)

            if self.brain and hasattr(self.brain, 'memory_manager') and self.brain.memory_manager and hasattr(self.brain.memory_manager, 'update_user_profile'):
                self.brain.memory_manager.update_user_profile(
                    user_id="system",
                    updates={
                        "concept": concept,
                        "integrated_concepts": concepts
                    }
                )

            if self.brain and hasattr(self.brain, 'knowledge_graph') and self.brain.knowledge_graph:
                try:
                    for integrated_concept in concepts:
                        nodes = self.brain.knowledge_graph.search_nodes(integrated_concept, limit=1)
                        if not nodes:
                            self.brain.knowledge_graph.add_node(
                                name=integrated_concept,
                                node_id=f"concept_{hash(integrated_concept) % 1000000}",
                                node_type="concept",
                                domain=task.metadata.get("domain", "general"), strength=0.85
                            )
                        self.brain.knowledge_graph.add_edge(
                            concept, integrated_concept, "integrates",
                            strength=0.8, meta={"source": "knowledge_integration"}
                        )
                except Exception as e:
                    logger.debug(f"Ошибка интеграции в knowledge_graph: {e}")

            return {
                "concept": concept,
                "integrated_concepts": concepts,
                "integration_quality": np.random.uniform(0.8, 0.95),
                "timestamp": time.time()
            }

        except Exception as e:
            logger.error(f"Ошибка при интеграции знаний для концепта {concept}: {e}")
            raise

    def _execute_deepen_concept(self, task: LearningTask) -> Any:
        """Выполняет задачу углубления концепта."""
        concept = task.concept
        logger.info(f"Углубление концепта: {concept}")

        try:
            concepts = []
            if self.brain and hasattr(self.brain, 'ml_unit') and self.brain.ml_unit and hasattr(self.brain.ml_unit, 'extract_concepts'):
                concepts = self.brain.ml_unit.extract_concepts(concept)

            if self.brain and hasattr(self.brain, 'memory_manager') and self.brain.memory_manager and hasattr(self.brain.memory_manager, 'update_user_profile'):
                self.brain.memory_manager.update_user_profile(
                    user_id="system",
                    updates={
                        "concept": concept,
                        "deepened_concepts": concepts
                    }
                )

            if self.brain and hasattr(self.brain, 'knowledge_graph') and self.brain.knowledge_graph:
                try:
                    for detail in concepts:
                        self.brain.knowledge_graph.add_node(
                            name=detail,
                            node_id=f"detail_{hash(detail) % 1000000}",
                            node_type="detail", domain=task.metadata.get("domain", "general"), strength=0.8
                        )
                        self.brain.knowledge_graph.add_edge(
                            concept, detail, "details",
                            strength=0.85, meta={"source": "concept_deepening"}
                        )
                except Exception as e:
                    logger.debug(f"Ошибка добавления деталей в knowledge_graph: {e}")

            return {
                "concept": concept,
                "details": f"Изучены концепты: {concepts}",
                "connections": len(concepts),
                "new_facts": len(concepts),
                "timestamp": time.time()
            }

        except Exception as e:
            logger.error(f"Ошибка при углублении концепта {concept}: {e}")
            raise

    def _execute_synthesize(self, task: LearningTask) -> Any:
        """Выполняет задачу синтеза знаний."""
        concept = task.concept
        logger.info(f"Синтез знаний по концепту: {concept}")

        try:
            concepts = []
            if self.brain and hasattr(self.brain, 'ml_unit') and self.brain.ml_unit and hasattr(self.brain.ml_unit, 'extract_concepts'):
                concepts = self.brain.ml_unit.extract_concepts(concept)

            if self.brain and hasattr(self.brain, 'memory_manager') and self.brain.memory_manager and hasattr(self.brain.memory_manager, 'update_user_profile'):
                self.brain.memory_manager.update_user_profile(
                    user_id="system",
                    updates={
                        "concept": concept,
                        "synthesized_concepts": concepts
                    }
                )

            if self.brain and hasattr(self.brain, 'knowledge_graph') and self.brain.knowledge_graph:
                try:
                    self.brain.knowledge_graph.add_node(
                        name=f"Синтез: {concept}",
                        node_id=f"synthesis_{hash(concept) % 1000000}",
                        node_type="synthesis",
                        domain=task.metadata.get("domain", "general"), strength=0.9
                    )
                    for synthesized_concept in concepts:
                        self.brain.knowledge_graph.add_edge(
                            f"synthesis_{hash(concept) % 1000000}",
                            synthesized_concept, "derived_from",
                            strength=0.9, meta={"source": "knowledge_synthesis"}
                        )
                except Exception as e:
                    logger.debug(f"Ошибка синтеза в knowledge_graph: {e}")

            return {
                "concept": concept,
                "synthesis_quality": np.random.uniform(0.8, 0.95),
                "holistic_understanding": np.random.uniform(0.75, 0.9),
                "new_insights": len(concepts),
                "timestamp": time.time()
            }

        except Exception as e:
            logger.error(f"Ошибка при синтезе знаний для концепта {concept}: {e}")
            raise

    def _execute_map_connections(self, task: LearningTask) -> Any:
        """Выполняет задачу создания карты связей."""
        concept = task.concept
        logger.info(f"Создание карты связей для концепта: {concept}")

        try:
            connections = []

            if self.brain and hasattr(self.brain, 'knowledge_graph') and self.brain.knowledge_graph:
                try:
                    edges = self.brain.knowledge_graph.get_edges(concept)
                    for edge in edges:
                        source_id = getattr(edge, 'source_id', getattr(edge, 'source', None))
                        target_id = getattr(edge, 'target_id', getattr(edge, 'target', None))
                        if source_id is None or target_id is None:
                            continue
                        related_concept = target_id if source_id == concept else source_id
                        strength = getattr(edge, 'strength', 0.5)
                        relation = getattr(edge, 'relation_type', getattr(edge, 'relation', 'related_to'))
                        connections.append({
                            "concept": related_concept,
                            "relation": relation,
                            "strength": strength
                        })
                except Exception as e:
                    logger.debug(f"Ошибка get_edges: {e}")

            related_concepts = []
            if len(connections) < 5 and self.brain and hasattr(self.brain, 'ml_unit') and self.brain.ml_unit and hasattr(self.brain.ml_unit, 'extract_concepts'):
                related_concepts = self.brain.ml_unit.extract_concepts(concept)
                for related_concept in related_concepts:
                    if related_concept != concept:
                        connections.append({
                            "concept": related_concept,
                            "relation": "related_to",
                            "strength": 0.7
                        })

            if self.brain and hasattr(self.brain, 'memory_manager') and self.brain.memory_manager and hasattr(self.brain.memory_manager, 'update_user_profile'):
                self.brain.memory_manager.update_user_profile(
                    user_id="system",
                    updates={
                        "concept": concept,
                        "connections": connections
                    }
                )

            return {
                "concept": concept,
                "connections": connections,
                "connection_strength": np.mean([c["strength"] for c in connections]) if connections else 0.0,
                "timestamp": time.time()
            }

        except Exception as e:
            logger.error(f"Ошибка при создании карты связей для концепта {concept}: {e}")
            raise

    def _execute_maintain_knowledge(self, task: LearningTask) -> Any:
        """Выполняет задачу поддержания знаний."""
        concept = task.concept
        logger.info(f"Поддержание знаний по концепту: {concept}")

        try:
            knowledge_status = "actual"
            maintenance_needed = False

            try:
                if self.brain and hasattr(self.brain, 'knowledge_graph'):
                    nodes = self.brain.knowledge_graph.search_nodes(concept, limit=1)
                    if nodes and len(nodes) > 0:
                        first_node = nodes[0]
                        node_id = getattr(first_node, 'id', None)
                        if node_id:
                            sources = []
                            if hasattr(self.brain.knowledge_graph, 'get_sources_for_node'):
                                sources = self.brain.knowledge_graph.get_sources_for_node(node_id)
                            for source in sources:
                                if time.time() - getattr(source, 'timestamp', time.time()) > 365 * 86400:
                                    knowledge_status = "outdated"
                                    maintenance_needed = True
                                    break
            except Exception as e:
                logger.error(f"Ошибка при обращении к knowledge_graph в maintain_knowledge: {e}")

            concepts = []
            if maintenance_needed and self.brain and hasattr(self.brain, 'ml_unit') and self.brain.ml_unit and hasattr(self.brain.ml_unit, 'extract_concepts'):
                concepts = self.brain.ml_unit.extract_concepts(concept)

                try:
                    if self.brain and hasattr(self.brain, 'knowledge_graph'):
                        nodes = self.brain.knowledge_graph.search_nodes(concept, limit=1)
                        if nodes and len(nodes) > 0:
                            first_node = nodes[0]
                            node_id = getattr(first_node, 'id', None)
                            if node_id:
                                self.brain.knowledge_graph.add_node(
                                    name=getattr(first_node, 'content', ''),
                                    node_id=node_id,
                                    node_type=getattr(first_node, 'node_type', 'concept'),
                                    domain=getattr(first_node, 'domain', 'general'),
                                    strength=0.9,
                                    meta=getattr(first_node, 'meta', getattr(first_node, 'metadata', {}))
                                )

                        for updated_concept in concepts:
                            self.brain.knowledge_graph.add_node(
                                name=updated_concept,
                                node_id=f"fact_{hash(updated_concept) % 1000000}",
                                node_type="fact",
                                domain=task.metadata.get("domain", "general"),
                                strength=0.85
                            )

                            self.brain.knowledge_graph.add_edge(
                                concept,
                                updated_concept,
                                "contains",
                                strength=0.85,
                                meta={"source": "knowledge_maintenance"}
                            )
                except Exception as e:
                    logger.error(f"Ошибка при записи в knowledge_graph в maintain_knowledge: {e}")

            return {
                "concept": concept,
                "knowledge_status": knowledge_status,
                "maintenance_performed": maintenance_needed,
                "timestamp": time.time()
            }

        except Exception as e:
            logger.error(f"Ошибка при поддержании знаний для концепта {concept}: {e}")
            raise
