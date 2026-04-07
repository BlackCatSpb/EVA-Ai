"""
Модуль сценариев этических ситуаций для ЕВА - генерация, сопоставление
"""
import os
import logging
import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

logger = logging.getLogger("eva_ai.ethics.situations")


class EthicalAssessment:
    """Класс для представления этической оценки."""
    def __init__(self, principle_name: str, score: float, confidence: float,
                 violation_detected: bool, severity: str):
        self.principle_name = principle_name
        self.score = score
        self.confidence = confidence
        self.violation_detected = violation_detected
        self.severity = severity


class EthicalPrinciple:
    """Класс для представления этического принципа."""
    def __init__(self, name: str, description: str, weight: float,
                 threshold: float, category: str, last_updated: float, active: bool):
        self.name = name
        self.description = description
        self.weight = weight
        self.threshold = threshold
        self.category = category
        self.last_updated = last_updated
        self.active = active


class EthicalDecision:
    def __init__(self, decision, confidence, justification, alternatives, assessment, requires_human_review):
        self.decision = decision
        self.confidence = confidence
        self.justification = justification
        self.alternatives = alternatives
        self.assessment = assessment
        self.requires_human_review = requires_human_review


class SituationsScenariosMixin:
    """Миксин для генерации и сопоставления сценариев."""

    def handle_situation(self, context: Dict[str, Any]) -> EthicalDecision:
        """
        Обрабатывает этическую ситуацию и принимает решение.

        Args:
            context: Контекст ситуации

        Returns:
            EthicalDecision: Принятое решение
        """
        if not context:
            return self._get_default_decision({}, "Пустой контекст")

        try:
            assessments = self.risk_assessor.assess_risk(context)

            requires_human_review = self._requires_human_review(assessments)

            decision, justification, alternatives = self._generate_decision(
                assessments, context, requires_human_review
            )

            ethical_decision = EthicalDecision(
                decision=decision,
                confidence=self._calculate_confidence(assessments),
                justification=justification,
                alternatives=alternatives,
                assessment=assessments,
                requires_human_review=requires_human_review
            )

            self._cache_solution(context, ethical_decision)

            self._update_statistics(ethical_decision)

            logger.info(f"Принято этическое решение: {decision} (требует человеческого вмешательства: {requires_human_review})")
            return ethical_decision

        except Exception as e:
            logger.error(f"Ошибка обработки этической ситуации: {e}")
            return self._get_default_decision(context, str(e))

    def _requires_human_review(self, assessments: List[EthicalAssessment]) -> bool:
        """Определяет, требуется ли человеческое вмешательство."""
        if not assessments:
            return True

        for assessment in assessments:
            if assessment.violation_detected and assessment.severity == "high":
                return True
            if assessment.confidence < 0.4:
                return True

        high_risk_principles = [
            a for a in assessments
            if a.violation_detected and a.severity in ["high", "medium"]
        ]

        if len(high_risk_principles) > 1:
            principle_weights = {
                a.principle_name: a.score * (1 - a.score)
                for a in high_risk_principles
            }

            if len(principle_weights) > 1:
                max_weight = max(principle_weights.values())
                min_weight = min(principle_weights.values())
                if max_weight - min_weight > 0.3:
                    return True

        return False

    def _generate_decision(self, assessments: List[EthicalAssessment],
                          context: Dict[str, Any],
                          requires_human_review: bool) -> Tuple[str, str, List[Dict[str, Any]]]:
        """
        Генерирует решение на основе оценок.

        Returns:
            Tuple[str, str, List[Dict[str, Any]]]: (решение, обоснование, альтернативы)
        """
        if requires_human_review:
            return (
                "defer_to_human",
                "Этическая ситуация требует человеческого вмешательства из-за высокого риска или неопределенности",
                []
            )

        if not assessments:
            return (
                "proceed_with_caution",
                "Недостаточно данных для полной оценки, действие разрешено с осторожностью",
                []
            )

        principle_scores = {
            a.principle_name: a.score * self._get_principle_weight(a.principle_name)
            for a in assessments
        }

        highest_risk_principle = min(principle_scores, key=principle_scores.get)
        risk_score = principle_scores[highest_risk_principle]

        if risk_score < 0.3:
            return (
                "reject_action",
                f"Отклонение действия из-за серьезного нарушения этического принципа {highest_risk_principle}",
                self._generate_alternatives(highest_risk_principle, context)
            )
        elif risk_score < 0.6:
            return (
                "modify_action",
                f"Модификация действия для уменьшения риска по принципу {highest_risk_principle}",
                self._generate_alternatives(highest_risk_principle, context)
            )
        else:
            return (
                "proceed",
                "Действие соответствует этическим стандартам",
                []
            )

    def _get_principle_weight(self, principle_name: str) -> float:
        """Возвращает вес принципа с учетом текущего контекста."""
        try:
            principle = self.principles_manager.get_principle_by_name(principle_name)
            if principle:
                return principle.weight
        except Exception as e:
            logger.warning(f"Ошибка получения веса принципа {principle_name}: {e}")
        return 1.0

    def _generate_alternatives(self, highest_risk_principle: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Генерирует альтернативные решения."""
        alternatives = []

        if highest_risk_principle == "non_harm":
            alternatives.append({
                "action": "provide_warning",
                "description": "Предоставить предупреждение о потенциальном вреде",
                "expected_outcome": "Пользователь будет информирован о рисках"
            })
            alternatives.append({
                "action": "suggest_alternative",
                "description": "Предложить менее рискованный вариант",
                "expected_outcome": "Снижение потенциального вреда"
            })

        elif highest_risk_principle == "autonomy":
            alternatives.append({
                "action": "provide_more_information",
                "description": "Предоставить больше информации для принятия решения",
                "expected_outcome": "Повышение информированности пользователя"
            })
            alternatives.append({
                "action": "offer_choice",
                "description": "Предложить несколько вариантов действий",
                "expected_outcome": "Увеличение автономии пользователя"
            })

        elif highest_risk_principle == "justice":
            alternatives.append({
                "action": "check_for_bias",
                "description": "Проверить ответ на наличие предвзятости",
                "expected_outcome": "Устранение потенциальной дискриминации"
            })
            alternatives.append({
                "action": "provide_context",
                "description": "Добавить контекст для справедливой оценки",
                "expected_outcome": "Более сбалансированный ответ"
            })

        if self.solutions_cache.get("solutions"):
            similar_solutions = [
                s for s in self.solutions_cache["solutions"]
                if highest_risk_principle in s.get("relevant_principles", [])
            ]

            for solution in similar_solutions[:2]:
                alternatives.append({
                    "action": solution["decision"],
                    "description": f"Адаптировано из предыдущего решения: {solution['justification']}",
                    "expected_outcome": "Аналогичная ситуация была успешно разрешена ранее"
                })

        return alternatives

    def _cache_solution(self, context: Dict[str, Any], decision: EthicalDecision):
        """Кэширует решение для будущих ссылок."""
        try:
            cache_key = self._generate_cache_key(context)

            solution_entry = {
                "key": cache_key,
                "timestamp": time.time(),
                "context_summary": self._summarize_context(context),
                "relevant_principles": [a.principle_name for a in decision.assessment],
                "decision": decision.decision,
                "justification": decision.justification,
                "confidence": decision.confidence,
                "requires_human_review": decision.requires_human_review
            }

            self.solutions_cache["solutions"].insert(0, solution_entry)

            if len(self.solutions_cache["solutions"]) > 100:
                self.solutions_cache["solutions"] = self.solutions_cache["solutions"][:100]

            self._save_cache()

        except Exception as e:
            logger.error(f"Ошибка кэширования решения: {e}")

    def _generate_cache_key(self, context: Dict[str, Any]) -> str:
        """Генерирует уникальный ключ для кэша на основе контекста."""
        context_str = ""
        if "query" in context:
            context_str += str(context["query"]) + " "
        if "response" in context:
            context_str += str(context["response"])

        return hashlib.md5(context_str.encode()).hexdigest()

    def _summarize_context(self, context: Dict[str, Any]) -> str:
        """Создает краткое описание контекста."""
        summary = []

        if "query" in context:
            query_str = str(context["query"])
            summary.append(f"Запрос: {query_str[:100]}{'...' if len(query_str) > 100 else ''}")

        if "response" in context:
            response_str = str(context["response"])
            summary.append(f"Ответ: {response_str[:100]}{'...' if len(response_str) > 100 else ''}")

        if "user_profile" in context and isinstance(context["user_profile"], dict):
            if "preferences" in context["user_profile"]:
                prefs = context["user_profile"]["preferences"]
                if isinstance(prefs, list):
                    summary.append(f"Преференции: {', '.join(str(p) for p in prefs[:3])}")

        return " | ".join(summary)

    def _update_statistics(self, decision: EthicalDecision):
        """Обновляет статистику по этическим решениям."""
        pass

    def _get_default_decision(self, context: Dict[str, Any], error: str) -> EthicalDecision:
        """Возвращает решение по умолчанию в случае ошибки."""
        return EthicalDecision(
            decision="error",
            confidence=0.1,
            justification=f"Ошибка обработки этической ситуации: {error}",
            alternatives=[],
            assessment=[],
            requires_human_review=True
        )
