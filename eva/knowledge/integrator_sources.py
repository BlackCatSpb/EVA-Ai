"""Source management, fetching data, external APIs for KnowledgeIntegrator."""
import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger("eva.knowledge_integrator")


def _init_source_reliability(self):
    """Инициализирует базовые значения надежности источников."""
    official_sources = [
        "wikipedia.org", "nih.gov", "nasa.gov", "who.int", "un.org",
        "science.gov", "nature.com", "springer.com", "ieee.org"
    ]
    for source in official_sources:
        self.source_reliability[source] = 0.95

    academic_sources = [
        "edu", "ac.uk", "edu.au", "scholar.google", "jstor.org"
    ]
    for source in academic_sources:
        self.source_reliability[source] = 0.85

    news_sources = [
        "bbc.com", "reuters.com", "apnews.com", "nytimes.com", "theguardian.com"
    ]
    for source in news_sources:
        self.source_reliability[source] = 0.75

    social_sources = [
        "blogspot.com", "wordpress.com", "medium.com", "twitter.com", "facebook.com"
    ]
    for source in social_sources:
        self.source_reliability[source] = 0.45


def get_source_reliability(self, source: str) -> float:
    """
    Возвращает надежность источника.

    Args:
        source: Источник информации

    Returns:
        float: Надежность источника (0.0-1.0)
    """
    if not source:
        return 0.5

    if source in self.source_reliability:
        return self.source_reliability[source]

    for domain, reliability in self.source_reliability.items():
        if domain in source:
            return reliability

    tld = source.split('.')[-1]
    if tld in ['gov', 'edu', 'ac.uk']:
        return 0.8
    elif tld in ['org', 'net']:
        return 0.65
    else:
        return 0.5


def update_source_reliability(self, source: str, reliability_change: float):
    """
    Обновляет надежность источника на основе новых данных.

    Args:
        source: Источник информации
        reliability_change: Изменение надежности (-1.0 to 1.0)
    """
    current_reliability = self.get_source_reliability(source)
    new_reliability = current_reliability + reliability_change * 0.1
    new_reliability = max(0.1, min(0.99, new_reliability))
    self.source_reliability[source] = new_reliability

    logger.debug(f"Надежность источника '{source}' обновлена: {current_reliability:.2f} -> {new_reliability:.2f}")


def _load_history_for_dynamic_updates(self):
    """Загружает историю для динамического обновления надежности источников и доменов."""
    try:
        history = []
        kg = getattr(self, "knowledge_graph", None)
        if kg is not None and hasattr(kg, "get_recent_changes"):
            history = kg.get_recent_changes(limit=100) or []

        for event in history:
            if isinstance(event, dict) and event.get("action") in ["node_update", "node_creation"]:
                if event.get("user_id"):
                    self.update_source_reliability(f"user_{event['user_id']}", 0.1)

        logger.debug("История загружена для динамического обновления надежности")

    except Exception as e:
        logger.error(f"Ошибка загрузки истории для динамических обновлений: {e}")


def learn_from_user_feedback(self, feedback: Dict[str, Any], user_id: Optional[str] = None) -> bool:
    """
    Учится на основе пользовательского фидбэка.

    Args:
        feedback: Данные фидбэка
        user_id: ID пользователя, предоставившего фидбэк

    Returns:
        bool: Успешно ли выполнено
    """
    try:
        logger.info("Обучение на основе пользовательского фидбэка...")

        feedback_type = feedback.get("feedback_type")
        if feedback_type == "correction":
            return self._handle_correction(feedback, user_id)

        elif feedback_type == "suggestion":
            return self._handle_suggestion(feedback, user_id)

        elif feedback_type == "rating":
            return self._handle_rating(feedback, user_id)

        elif feedback_type == "contradiction_report":
            return self._handle_contradiction_report(feedback, user_id)

        return False

    except Exception as e:
        logger.error(f"Ошибка обучения на основе фидбэка: {e}")
        return False


def _handle_contradiction_report(self, feedback: Dict[str, Any], user_id: Optional[str] = None) -> bool:
    """
    Обрабатывает сообщение пользователя о противоречии в знаниях.

    Args:
        feedback: Данные фидбэка
        user_id: ID пользователя

    Returns:
        bool: Успешно ли обработано
    """
    try:
        concept = feedback["concept"]
        description = feedback["description"]
        evidence = feedback.get("evidence", [])

        logger.info(f"Обработка сообщения о противоречии для концепта '{concept}': {description}")

        contradiction = {
            "concept": concept,
            "description": description,
            "evidence": evidence,
            "user_reported": True,
            "user_id": user_id,
            "timestamp": time.time()
        }

        resolved = self._resolve_contradiction(contradiction)

        if user_id:
            reliability_change = 0.2 if resolved else -0.1
            self.update_source_reliability(f"user_{user_id}", reliability_change)

        if resolved:
            kg = getattr(self, "knowledge_graph", None)
            if kg is not None and hasattr(kg, "record_history"):
                try:
                    kg.record_history(
                        f"contradiction_{concept}",
                        "contradiction_resolved",
                        None,
                        {"concept": concept, "description": description},
                        user_id
                    )
                except Exception as ex:
                    logger.debug(f"Не удалось записать историю в KnowledgeGraph: {ex}")

        logger.info(f"Сообщение о противоречии для концепта '{concept}' {'разрешено' if resolved else 'не разрешено'}")
        return resolved

    except Exception as e:
        logger.error(f"Ошибка обработки сообщения о противоречии: {e}")
        return False


def _handle_correction(self, feedback: Dict[str, Any], user_id: Optional[str] = None) -> bool:
    """Обрабатывает коррекцию от пользователя."""
    try:
        concept = feedback["concept"]
        correction_type = feedback["correction_type"]
        correction = feedback["correction"]

        nodes = self.knowledge_graph.search_nodes(concept, limit=1)
        if not nodes:
            self.knowledge_graph.add_concept(
                concept,
                correction,
                domain=feedback.get("domain", "general"),
                strength=0.9,
                user_id=user_id
            )
            return True

        node = nodes[0]

        if correction_type == "inaccuracy":
            updated = self.knowledge_graph.update_concept(
                node.id,
                new_description=correction,
                strength=min(1.0, node.strength + 0.1),
                user_id=user_id
            )
            return updated

        elif correction_type == "missing_info":
            if not node.meta or not isinstance(node.meta, dict):
                return False
            new_description = f"{node.meta['description']}\n\nДополнение: {correction}"
            updated = self.knowledge_graph.update_concept(
                node.id,
                new_description=new_description,
                strength=min(1.0, node.strength + 0.05),
                user_id=user_id
            )
            return updated

        return False

    except Exception as e:
        logger.error(f"Ошибка обработки коррекции для концепта '{feedback['concept']}': {e}")
        return False


def _handle_suggestion(self, feedback: Dict[str, Any], user_id: Optional[str] = None) -> bool:
    """Обрабатывает предложение от пользователя."""
    try:
        suggestion_type = feedback["suggestion_type"]

        if suggestion_type == "new_concept":
            self.knowledge_graph.add_concept(
                feedback["concept"],
                feedback["description"],
                domain=feedback.get("domain", "general"),
                strength=0.7,
                user_id=user_id
            )
            return True

        elif suggestion_type == "new_connection":
            source_node = self.knowledge_graph.search_nodes(feedback["source_concept"], limit=1)
            target_node = self.knowledge_graph.search_nodes(feedback["target_concept"], limit=1)

            if source_node and target_node:
                self.knowledge_graph.add_edge(
                    source_node[0].id,
                    target_node[0].id,
                    feedback["relation"],
                    strength=0.6,
                    user_id=user_id
                )
                return True

        return False

    except Exception as e:
        logger.error(f"Ошибка обработки предложения: {e}")
        return False


def _handle_rating(self, feedback: Dict[str, Any], user_id: Optional[str] = None) -> bool:
    """Обрабатывает оценку от пользователя."""
    try:
        concept = feedback["concept"]
        rating = feedback["rating"]

        nodes = self.knowledge_graph.search_nodes(concept, limit=1)
        if not nodes:
            return False

        node = nodes[0]

        strength_change = (rating - 3) * 0.05
        new_strength = max(0.1, min(1.0, node.strength + strength_change))

        if not node.meta or not isinstance(node.meta, dict) or "description" not in node.meta:
            return False
        return self.knowledge_graph.update_concept(
            node.id,
            new_description=node.meta.get("description", ""),
            strength=new_strength,
            user_id=user_id
        )

    except Exception as e:
        logger.error(f"Ошибка обработки оценки для концепта '{feedback['concept']}': {e}")
        return False
