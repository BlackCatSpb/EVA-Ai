"""
SRE Feedback Module — user feedback processing, outcome learning, confidence threshold adaptation, cross-session learning.
"""

import time
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def process_user_feedback(self, query: str, feedback: str, rating: float) -> Dict[str, Any]:
    """
    Обработать фидбек пользователя для самообучения
    """
    try:
        if self.fractal_storage:
            chain = self.fractal_storage.get_reasoning_chain(query)

            for node in chain:
                old_conf = node.context.get("confidence", 0.5) if node.context else 0.5
                adjusted_conf = (old_conf + rating) / 2

                if node.context:
                    node.context["confidence"] = adjusted_conf
                    node.context["last_feedback_rating"] = rating
                    node.context["feedback_time"] = time.time()

            self.fractal_storage._save()

            if rating < 0.3:
                self._trigger_self_learning(query, feedback)

            logger.info(f"Обработан фидбек для '{query[:30]}...': rating={rating:.2f}")

            return {
                "status": "processed",
                "chain_updated": len(chain),
                "triggered_learning": rating < 0.3
            }

        return {"status": "no_storage", "message": "FractalStorage не доступен"}

    except Exception as e:
        logger.error(f"Ошибка обработки фидбека: {e}")
        return {"status": "error", "message": str(e)}


def learn_from_outcome(self, query: str, outcome: Dict[str, Any]) -> Dict[str, Any]:
    """
    Выучить результат для будущих рассуждений
    """
    try:
        success = outcome.get("success", False)
        user_rating = outcome.get("rating", 0.5)
        response_quality = outcome.get("response_quality", "unknown")

        learning_record = {
            "query": query,
            "success": success,
            "rating": user_rating,
            "quality": response_quality,
            "timestamp": time.time()
        }

        if self.fractal_storage:
            self.fractal_storage.add_node(
                content=f"Learning: {query[:50]}",
                node_type="learning_record",
                level=1,
                context=learning_record
            )

        if user_rating < 0.4:
            logger.info(f"Низкий рейтинг для '{query[:30]}...', адаптирую параметры")
            self.confidence_threshold = max(0.5, self.confidence_threshold - 0.05)
        elif user_rating > 0.8:
            self.confidence_threshold = min(0.9, self.confidence_threshold + 0.02)

        return {
            "status": "learned",
            "new_threshold": self.confidence_threshold,
            "query": query[:30]
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


def _update_confidence_threshold(self, rating: float) -> None:
    """Адаптивное обновление порога уверенности на основе рейтинга"""
    if rating < 0.4:
        self.confidence_threshold = max(0.5, self.confidence_threshold - 0.05)
    elif rating > 0.8:
        self.confidence_threshold = min(0.9, self.confidence_threshold + 0.02)


def refine_reasoning_chain(self, node_id: str, correction: str) -> Dict[str, Any]:
    """
    Уточнить цепочку рассуждений после коррекции пользователя
    """
    if not self.fractal_storage:
        return {"status": "error", "message": "Нет хранилища"}

    try:
        node = self.fractal_storage.get_node(node_id)
        if not node:
            return {"status": "error", "message": "Узел не найден"}

        if not node.context:
            node.context = {}

        corrections = node.context.get("corrections", [])
        corrections.append({
            "correction": correction,
            "timestamp": time.time()
        })
        node.context["corrections"] = corrections
        node.context["needs_rethink"] = True

        self.fractal_storage._save()

        logger.info(f"Добавлена коррекция для узла {node_id[:16]}...")

        return {"status": "processed", "node_id": node_id}

    except Exception as e:
        return {"status": "error", "message": str(e)}


def self_correct(self, query: str, user_correction: str) -> Dict[str, Any]:
    """
    Самоисправление на основе коррекции пользователя
    """
    similar = self.retrieve_similar_reasoning(query)

    correction_prompt = f"""Пользователь указал, что предыдущий ответ был неверным.
Коррекция: {user_correction}

Вопрос: {query}

Дай исправленный ответ учитывая коррекцию:"""

    corrected_response = self._generate_with_qwen(correction_prompt)

    if not corrected_response:
        corrected_response = user_correction
        logger.warning("Generation returned empty, using user correction as response")

    if self.fractal_storage:
        self.fractal_storage.add_reasoning_step(
            query=query,
            step_content=f"SELF-CORRECTED: {corrected_response}",
            confidence=0.6,
            iteration=999
        )

    return {
        "corrected_response": corrected_response,
        "similar_used": len(similar),
        "status": "corrected"
    }


def adaptive_recursion_depth(self, query_complexity: float) -> int:
    """
    Адаптивная глубина рекурсии на основе сложности запроса
    """
    base_depth = 1

    if query_complexity > 0.7:
        return min(self.max_recursion_depth, base_depth + 2)
    elif query_complexity > 0.4:
        return min(self.max_recursion_depth, base_depth + 1)

    return base_depth


def cross_session_learning(self, query: str) -> List[Dict]:
    """
    Учиться на прошлых сессиях - находим похожие запросы
    """
    if not self.fractal_retriever:
        return []

    try:
        similar = self.fractal_retriever.retrieve_with_embedding(
            query=query,
            top_k=10
        )

        learned = []
        for item in similar:
            ctx = item.get("context", {})
            rating = ctx.get("last_feedback_rating", 0.5)

            if rating >= 0.6:
                learned.append({
                    "query": item.get("content", "")[:100],
                    "rating": rating,
                    "node_id": item.get("id")
                })

        return learned

    except Exception as e:
        logger.warning(f"Ошибка кросс-сессионного обучения: {e}")
        return []


def _trigger_self_learning(self, query: str, reason: str):
    """
    Триггерировать самообучение при низком качестве
    """
    try:
        curiosity = getattr(self.brain, 'curiosity_engine', None)
        if curiosity and hasattr(curiosity, 'trigger_self_learning'):
            curiosity.trigger_self_learning(query, reason)
            logger.info(f"Триггернуто самообучение для: {query[:30]}...")
        else:
            logger.debug("CuriosityEngine не доступен для самообучения")
    except Exception as e:
        logger.debug(f"Не удалось триггернуть самообучение: {e}")


def get_feedback_stats(self) -> Dict[str, Any]:
    """Получить статистику фидбека"""
    if not self.fractal_storage:
        return {"status": "no_storage"}

    total_feedback = 0
    positive = 0
    negative = 0

    for node in self.fractal_storage.nodes.values():
        if node.context and "last_feedback_rating" in node.context:
            total_feedback += 1
            rating = node.context["last_feedback_rating"]
            if rating >= 0.5:
                positive += 1
            else:
                negative += 1

    return {
        "total_feedback": total_feedback,
        "positive": positive,
        "negative": negative,
        "current_threshold": self.confidence_threshold
    }
