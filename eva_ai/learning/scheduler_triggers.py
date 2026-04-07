"""Trigger handling, event-based scheduling, and conditions for EVA learning scheduler."""

import time
import logging
from typing import Dict, List, Optional, Any

import numpy as np

from .scheduler_core import LearningTask

logger = logging.getLogger("eva_ai.learning_scheduler")


class TriggerMixin:
    """Mixin providing learning plan creation, triggers, and adaptive scheduling."""

    def create_learning_plan(self, concepts: List[str], depth: int = 2) -> List[Dict]:
        """Создает план обучения для указанных концептов."""
        plan = []

        for concept in concepts:
            knowledge_state = self._assess_knowledge_state(concept)
            learning_type = self._determine_learning_type(knowledge_state)
            tasks = self._create_learning_tasks(concept, learning_type, depth)

            plan.append({
                "concept": concept,
                "knowledge_state": knowledge_state,
                "learning_type": learning_type,
                "tasks": tasks
            })

        return plan

    def _assess_knowledge_state(self, concept: str) -> Dict[str, float]:
        """Оценивает состояние знаний по концепту."""
        depth = 0.5
        breadth = 0.5
        recency = 0.5
        coherence = 0.5

        if self.brain and hasattr(self.brain, 'knowledge_graph'):
            try:
                nodes = self.brain.knowledge_graph.search_nodes(concept, limit=1)

                if nodes and len(nodes) > 0:
                    first_node = nodes[0]
                    node_id = getattr(first_node, 'id', None)
                    if node_id is None:
                        node_id = concept

                    edges = self.brain.knowledge_graph.get_edges(node_id)
                    depth = min(0.9, len(edges) * 0.1)

                    domains = set()
                    for edge in edges:
                        source_id = getattr(edge, 'source_id', getattr(edge, 'source', None))
                        target_id = getattr(edge, 'target_id', getattr(edge, 'target', None))
                        if source_id is None or target_id is None:
                            continue
                        related_id = target_id if source_id == node_id else source_id
                        related_node = self.brain.knowledge_graph.get_node(related_id)
                        if related_node:
                            domains.add(getattr(related_node, 'domain', None))
                    breadth = min(0.9, len([d for d in domains if d is not None]) * 0.2)

                    last_updated = getattr(first_node, 'last_updated', time.time())
                    time_diff = time.time() - last_updated
                    recency = max(0.1, 1.0 - min(1.0, time_diff / (365 * 86400)))

                    try:
                        kg_health = self.brain.knowledge_graph.get_graph_health()
                        coherence = kg_health["statistics"]["coherence"]
                    except Exception:
                        coherence = 0.5
            except Exception as e:
                logger.error(f"Ошибка при обращении к knowledge_graph: {e}")

        overall = (depth * 0.3 + breadth * 0.3 + recency * 0.2 + coherence * 0.2)

        return {
            "depth": depth,
            "breadth": breadth,
            "recency": recency,
            "coherence": coherence,
            "overall": overall,
            "assessment_date": time.time()
        }

    def _determine_learning_type(self, knowledge_state: Dict[str, float]) -> str:
        """Определяет тип обучения на основе состояния знаний."""
        DEPTH_THRESHOLD = 0.4
        BREADTH_THRESHOLD = 0.4
        RECENCY_THRESHOLD = 0.5
        COHERENCE_THRESHOLD = 0.5

        if knowledge_state["depth"] < DEPTH_THRESHOLD:
            return "deepen"

        if knowledge_state["breadth"] < BREADTH_THRESHOLD:
            return "expand"

        if knowledge_state["recency"] < RECENCY_THRESHOLD:
            return "update"

        if knowledge_state["coherence"] < COHERENCE_THRESHOLD:
            return "integrate"

        return "maintain"

    def _create_learning_tasks(self, concept: str, learning_type: str, depth: int) -> List[Dict]:
        """Создает задачи обучения для концепта."""
        tasks = []
        base_priority = 0.8

        if learning_type == "deepen":
            tasks.append({
                "task_id": f"deepen_{hash(concept) % 1000000}_{depth}",
                "task_type": "deepen_concept",
                "concept": concept,
                "priority": base_priority,
                "scheduled_time": time.time(),
                "metadata": {
                    "description": f"Углубить знание по концепту '{concept}'",
                    "expected_outcome": "Понимание нюансов и деталей концепта",
                    "resources": ["Дополнительные источники", "Экспертные материалы"]
                },
                "dependencies": []
            })

            if depth > 1 and len(tasks) > 0:
                first_task_id = tasks[0]["task_id"]
                tasks.append({
                    "task_id": f"analyze_{hash(concept) % 1000000}_{depth}",
                    "task_type": "analyze_connections",
                    "concept": concept,
                    "priority": base_priority * 0.9,
                    "scheduled_time": time.time() + 1800,
                    "metadata": {
                        "description": f"Анализ связей концепта '{concept}' с другими концептами",
                        "expected_outcome": "Понимание контекста и связей концепта",
                        "resources": ["Связанные концепты", "Контекстуальный анализ"]
                    },
                    "dependencies": [first_task_id]
                })

                tasks.append({
                    "task_id": f"map_{hash(concept) % 1000000}_{depth}",
                    "task_type": "map_connections",
                    "concept": concept,
                    "priority": base_priority * 0.9,
                    "scheduled_time": time.time() + 2400,
                    "metadata": {
                        "description": f"Создать карту концептов для '{concept}'",
                        "expected_outcome": "Визуальное представление связей концептов",
                        "resources": ["Карта знаний", "Графический инструмент"]
                    },
                    "dependencies": [first_task_id]
                })

        elif learning_type == "expand":
            tasks.append({
                "task_id": f"expand_{hash(concept) % 1000000}_{depth}",
                "task_type": "expand_domain",
                "concept": concept,
                "priority": base_priority,
                "scheduled_time": time.time(),
                "metadata": {
                    "description": f"Расширить знания по домену концепта '{concept}'",
                    "expected_outcome": "Понимание смежных концептов и областей",
                    "resources": ["Смежные домены", "Кросс-доменные источники"]
                },
                "dependencies": []
            })

            if depth > 1 and len(tasks) > 0:
                first_task_id = tasks[0]["task_id"]
                tasks.append({
                    "task_id": f"relate_{hash(concept) % 1000000}_{depth}",
                    "task_type": "analyze_connections",
                    "concept": concept,
                    "priority": base_priority * 0.9,
                    "scheduled_time": time.time() + 1800,
                    "metadata": {
                        "description": f"Анализ связей концепта '{concept}' с другими доменами",
                        "expected_outcome": "Понимание междоменной интеграции",
                        "resources": ["Междоменные связи", "Интеграционные модели"]
                    },
                    "dependencies": [first_task_id]
                })

                tasks.append({
                    "task_id": f"synthesize_{hash(concept) % 1000000}_{depth}",
                    "task_type": "synthesize",
                    "concept": concept,
                    "priority": base_priority * 0.85,
                    "scheduled_time": time.time() + 3600,
                    "metadata": {
                        "description": f"Синтезировать знания по концепту '{concept}'",
                        "expected_outcome": "Целостное понимание концепта",
                        "resources": ["Методы синтеза", "Интеграционные модели"]
                    },
                    "dependencies": [first_task_id]
                })

        elif learning_type == "update":
            tasks.append({
                "task_id": f"update_{hash(concept) % 1000000}_{depth}",
                "task_type": "update_knowledge",
                "concept": concept,
                "priority": base_priority,
                "scheduled_time": time.time(),
                "metadata": {
                    "description": f"Обновить знания по концепту '{concept}'",
                    "expected_outcome": "Актуальная информация по концепту",
                    "resources": ["Новые источники", "Экспертные обновления"]
                },
                "dependencies": []
            })

            if depth > 1 and len(tasks) > 0:
                first_task_id = tasks[0]["task_id"]
                tasks.append({
                    "task_id": f"verify_{hash(concept) % 1000000}_{depth}",
                    "task_type": "verify_sources",
                    "concept": concept,
                    "priority": base_priority * 0.9,
                    "scheduled_time": time.time() + 1800,
                    "metadata": {
                        "description": f"Проверить источники информации по концепту '{concept}'",
                        "expected_outcome": "Надежные и актуальные источники",
                        "resources": ["Верификационные инструменты", "Экспертные оценки"]
                    },
                    "dependencies": [first_task_id]
                })

        elif learning_type == "integrate":
            tasks.append({
                "task_id": f"integrate_{hash(concept) % 1000000}_{depth}",
                "task_type": "integrate_knowledge",
                "concept": concept,
                "priority": base_priority,
                "scheduled_time": time.time(),
                "metadata": {
                    "description": f"Интегрировать знания по концепту '{concept}'",
                    "expected_outcome": "Согласованное и структурированное знание",
                    "resources": ["Интеграционные методы", "Структурные модели"]
                },
                "dependencies": []
            })

            if depth > 1 and len(tasks) > 0:
                first_task_id = tasks[0]["task_id"]
                tasks.append({
                    "task_id": f"coherence_{hash(concept) % 1000000}_{depth}",
                    "task_type": "map_connections",
                    "concept": concept,
                    "priority": base_priority * 0.9,
                    "scheduled_time": time.time() + 1800,
                    "metadata": {
                        "description": f"Проверить согласованность связей концепта '{concept}'",
                        "expected_outcome": "Логически согласованные связи",
                        "resources": ["Методы проверки согласованности", "Логические модели"]
                    },
                    "dependencies": [first_task_id]
                })

        else:
            tasks.append({
                "task_id": f"maintain_{hash(concept) % 1000000}_{depth}",
                "task_type": "maintain_knowledge",
                "concept": concept,
                "priority": base_priority * 0.7,
                "scheduled_time": time.time(),
                "metadata": {
                    "description": f"Поддерживать знания по концепту '{concept}'",
                    "expected_outcome": "Актуальное и надежное знание",
                    "resources": ["Системы мониторинга", "Механизмы обновления"]
                },
                "dependencies": []
            })

        return tasks

    def generate_learning_plan_report(self, concepts: List[str], depth: int = 2) -> str:
        """Возвращает текстовый отчет о плане обучения."""
        plan = self.create_learning_plan(concepts, depth)
        report = "=== ПЛАН ОБУЧЕНИЯ ===\n\n"

        for i, item in enumerate(plan, 1):
            concept = item["concept"]
            state = item["knowledge_state"]
            report += f"{i}. {concept.upper()}\n"
            report += f"   Текущее состояние: {state['overall']:.2f}\n"
            report += f"   Глубина: {state['depth']:.2f}, Ширина: {state['breadth']:.2f}\n"
            report += f"   Актуальность: {state['recency']:.2f}, Связность: {state['coherence']:.2f}\n"
            report += f"   Рекомендуемый тип обучения: {item['learning_type'].upper()}\n"
            report += "   ЗАДАЧИ:\n"

            for j, task in enumerate(item["tasks"], 1):
                report += f"    {j}. {task['metadata']['description']}\n"
                report += f"       Тип: {task['task_type']}, Приоритет: {task['priority']:.2f}\n"
                report += f"       Ожидаемый результат: {task['metadata']['expected_outcome']}\n"
                if task.get("dependencies"):
                    report += f"       Зависимости: {', '.join(task['dependencies'])}\n"

            report += "\n"

        report += "=== КОНЕЦ ПЛАНА ==="
        return report

    def get_concept_domain(self, concept: str) -> str:
        """Определяет домен концепта."""
        concept_lower = concept.lower()

        domain_keywords = {
            "technology": ["искусственный интеллект", "машинное обучение", "нейросеть", "алгоритм", "программирование", "компьютер", "технология", "сеть", "данные", "информация"],
            "science": ["физика", "химия", "биология", "математика", "нейронаука", "наука", "эксперимент", "теория", "формула", "исследование"],
            "philosophy": ["сознание", "этика", "философия", "мышление", "разум", "философский", "мышление", "дух", "сущность", "бытие"],
            "art": ["искусство", "литература", "музыка", "живопись", "театр", "кино", "творчество", "стиль", "художник", "поэзия"],
            "health": ["здоровье", "медицина", "болезнь", "лечение", "анатомия", "физиология", "психология", "терапия", "диагностика", "профилактика"]
        }

        counts = {domain: 0 for domain in domain_keywords}
        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                if keyword in concept_lower:
                    counts[domain] += 1

        dominant_domain = max(counts, key=counts.get)
        if counts[dominant_domain] == 0:
            return "general"

        return dominant_domain

    def create_adaptive_learning_plan(self, user_id: str, context: Optional[Dict] = None) -> List[Dict]:
        """Создает адаптивный план обучения на основе контекста пользователя."""
        context = context or {}

        user_profile = self._get_user_profile(user_id)
        target_concepts = self._determine_target_concepts(user_profile, context)
        learning_plan = self.create_learning_plan(target_concepts)
        adapted_plan = self._adapt_learning_plan(learning_plan, user_profile, context)

        return adapted_plan

    def _get_user_profile(self, user_id: str) -> Dict:
        """Получает профиль пользователя."""
        if self.brain and hasattr(self.brain, 'adaptation_manager') and self.brain.adaptation_manager:
            try:
                profile_obj = self.brain.adaptation_manager.get_user_profile(user_id)
                if profile_obj and hasattr(profile_obj, 'to_dict'):
                    return profile_obj.to_dict()
            except Exception as e:
                logger.debug(f"Ошибка получения профиля пользователя: {e}")
        return {
            "user_id": user_id,
            "preferences": {},
            "interaction_history": [],
            "knowledge_level": "beginner",
            "learning_style": "visual",
            "cultural_profile": {},
            "timestamp": time.time(),
            "last_updated": time.time()
        }

    def _determine_target_concepts(self, user_profile: Dict, context: Dict) -> List[str]:
        """Определяет целевые концепты для обучения."""
        current_knowledge = {
            "technology": 0.3,
            "science": 0.5,
            "philosophy": 0.2,
            "art": 0.1
        }

        preferences = user_profile.get("preferences") if isinstance(user_profile.get("preferences"), dict) else {}
        user_interests = preferences.get("preferred_domains", ["technology"]) if isinstance(preferences, dict) else ["technology"]

        weak_areas = [
            domain for domain, level in current_knowledge.items()
            if level < 0.5 and domain in user_interests
        ]

        if not weak_areas:
            weak_areas = user_interests

        concepts = []
        for domain in weak_areas[:2]:
            concepts.append(f"основы_{domain}")
            concepts.append(f"продвинутые_{domain}")

        return concepts

    def _adapt_learning_plan(self, learning_plan: List[Dict], user_profile: Dict, context: Dict) -> List[Dict]:
        """Адаптирует план обучения под пользователя."""
        adapted_plan = []

        for item in learning_plan:
            knowledge_level = user_profile["knowledge_level"]
            if knowledge_level == "beginner":
                item["tasks"] = [task for task in item["tasks"] if "deepen" not in task["task_type"]]
            elif knowledge_level == "expert":
                if item["learning_type"] == "deepen":
                    item["priority"] *= 1.2

            learning_style = user_profile["learning_style"]
            for task in item["tasks"]:
                if learning_style == "visual" and task["task_type"] == "map_connections":
                    task["priority"] *= 1.3
                elif learning_style == "auditory" and task["task_type"] == "synthesize":
                    task["priority"] *= 1.2

            adapted_plan.append(item)

        return adapted_plan
