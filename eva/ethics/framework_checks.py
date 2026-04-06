"""
Модуль проверок этической рамки для ЕВА - индивидуальные проверки, оценка, вычисление
"""
import logging
import time
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .framework_principles import EthicalPrinciple
from .framework_violations import EthicalDecision

logger = logging.getLogger("eva.ethics")

@dataclass
class EthicalAssessment:
    """Результат этической оценки."""
    principle_scores: Dict[str, float] = field(default_factory=dict)
    violations: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    # Дополнительные поля для risk_assessment
    principle_name: Optional[str] = None
    score: float = 0.0
    explanation: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    violation_detected: bool = False
    severity: str = "low"

@dataclass
class EthicsAnalysisResult:
    overall_score: float
    violations: List[Dict[str, Any]]
    recommendations: List[str]
    principle_scores: Dict[str, float]


class EthicsChecksMixin:
    """Миксин для выполнения этических проверок и оценок."""

    def analyze_content(self, content: str, context: Optional[Dict[str, Any]] = None) -> EthicsAnalysisResult:
        result = self.analyze_request(content, context=context)
        try:
            principle_scores = result.get('principle_scores', {}) if isinstance(result, dict) else {}
            violations = result.get('violations', []) if isinstance(result, dict) else []
            recommendations = result.get('recommendations', []) if isinstance(result, dict) else []
            max_score = 0.0
            try:
                max_score = float(max(principle_scores.values())) if principle_scores else 0.0
            except Exception:
                max_score = 0.0
            overall_score = max(0.0, min(1.0, 1.0 - max_score))
            return EthicsAnalysisResult(
                overall_score=overall_score,
                violations=violations,
                recommendations=recommendations,
                principle_scores=principle_scores
            )
        except Exception:
            return EthicsAnalysisResult(
                overall_score=1.0,
                violations=[],
                recommendations=[],
                principle_scores={}
            )

    def analyze_response(self, query: str, response: str) -> Dict[str, Any]:
        analysis = self.analyze_content(response, context={"query": query})
        return {
            "overall_score": analysis.overall_score,
            "violations": analysis.violations,
            "recommendations": analysis.recommendations,
            "principle_scores": analysis.principle_scores
        }

    def analyze_request(self, request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Анализирует запрос на соответствие этическим принципам.

        Args:
            request: Текст запроса пользователя
            context: Дополнительный контекст

        Returns:
            Dict: Результат анализа
        """
        start_time = time.time()

        with self.lock:
            self.stats["total_assessments"] += 1
            self.stats["last_assessment"] = time.time()

            assessment = self._assess_request(request, context)

            violations = []
            for principle_name, score in assessment.principle_scores.items():
                principle = self.principles.get(principle_name)
                if principle and score > principle.threshold:
                    violation = EthicalDecision(
                        approved=False,
                        principle=principle_name,
                        severity=score,
                        description=f"Нарушение принципа {principle_name}",
                        context=context or {}
                    )
                    violations.append(violation)
                    self.violations.append(violation)

                    self.stats["violations_detected"] += 1
                    if score > 0.9:
                        self.stats["high_severity_violations"] += 1

            self._save_violations()

            result = {
                "approved": len(violations) == 0,
                "violations": [v.__dict__ for v in violations],
                "principle_scores": assessment.principle_scores,
                "confidence": assessment.confidence,
                "timestamp": time.time(),
                "processing_time": time.time() - start_time
            }

            if not result["approved"]:
                result["response"] = self._generate_rejection_response(violations)

            if violations:
                logger.warning(f"Обнаружено {len(violations)} этических нарушений в запросе")
                for violation in violations:
                    logger.warning(f"Нарушение: {violation.principle} (серьезность: {violation.severity:.2f})")
            else:
                logger.info("Запрос прошел этическую проверку успешно")

            return result

    def _assess_request(self, request: str, context: Optional[Dict[str, Any]]) -> EthicalAssessment:
        """
        Оценивает запрос на соответствие этическим принципам.

        Args:
            request: Текст запроса
            context: Дополнительный контекст

        Returns:
            EthicalAssessment: Результат оценки
        """
        principle_scores = {}
        recommendations = []

        for name, principle in self.principles.items():
            score = self._evaluate_principle(request, context, principle)
            principle_scores[name] = score

            if score > principle.threshold:
                recommendations.append(
                    f"Внимание: возможное нарушение принципа '{name}' (уровень: {score:.2f}). "
                    "Рекомендуется пересмотреть запрос или предоставить дополнительные разъяснения."
                )

        confidence = self._calculate_confidence(principle_scores)

        return EthicalAssessment(
            violations=[],
            recommendations=recommendations,
            principle_scores=principle_scores,
            confidence=confidence
        )

    def _evaluate_principle(self, request: str, context: Optional[Dict[str, Any]],
                           principle: EthicalPrinciple) -> float:
        """
        Оценивает запрос на соответствие конкретному этическому принципу.

        Args:
            request: Текст запроса
            context: Дополнительный контекст
            principle: Этический принцип

        Returns:
            float: Уровень нарушения (0.0-1.0)
        """
        score = 0.0

        if principle.category == "privacy":
            score = self._evaluate_privacy(request, context)
        elif principle.category == "safety":
            score = self._evaluate_safety(request, context)
        elif principle.category == "fairness":
            score = self._evaluate_fairness(request, context)
        elif principle.category == "transparency":
            score = self._evaluate_transparency(request, context)
        elif principle.category == "autonomy":
            score = self._evaluate_autonomy(request, context)
        elif principle.category == "beneficence":
            score = self._evaluate_beneficence(request, context)
        elif principle.category == "accountability":
            score = self._evaluate_accountability(request, context)

        return min(max(score, 0.0), 1.0)

    def _evaluate_privacy(self, request: str, context: Optional[Dict[str, Any]]) -> float:
        """Оценивает запрос на соответствие принципу приватности."""
        score = 0.0

        personal_info_keywords = [
            "имя", "фамилия", "адрес", "телефон", "email", "почта", "номер",
            "паспорт", "документ", "личные данные", "конфиденциальная", "приватная"
        ]

        for keyword in personal_info_keywords:
            if keyword in request.lower():
                score += 0.15

        if re.search(r"\b[А-Я][а-я]+\s+[А-Я][а-я]+\b", request):
            score += 0.1

        if "о ком-то" in request.lower() or "человеке" in request.lower():
            score += 0.1

        if "конфиденциальная" in request.lower() or "секрет" in request.lower():
            score += 0.2

        return min(score, 1.0)

    def _evaluate_safety(self, request: str, context: Optional[Dict[str, Any]]) -> float:
        """Оценивает запрос на соответствие принципу безопасности."""
        score = 0.0

        dangerous_keywords = [
            "убить", "навредить", "повредить", "опасно", "опасность", "вред", "взрыв",
            "оружие", "наркотик", "наркотики", "взлом", "взломать", "взломать", "обмануть"
        ]

        for keyword in dangerous_keywords:
            if keyword in request.lower():
                score += 0.2

        if "призыв" in request.lower() and ("насилие" in request.lower() or "агрессия" in request.lower()):
            score += 0.3

        if "как" in request.lower() and ("сделать взрыв" in request.lower() or "создать оружие" in request.lower()):
            score += 0.4

        return min(score, 1.0)

    def _evaluate_fairness(self, request: str, context: Optional[Dict[str, Any]]) -> float:
        """Оценивает запрос на соответствие принципу справедливости."""
        score = 0.0

        discriminatory_keywords = [
            "расист", "расизм", "расистский", "националист", "национализм",
            "ксенофоб", "ксенофобия", "сексист", "сексизм", "гомофоб", "гомофобия"
        ]

        for keyword in discriminatory_keywords:
            if keyword in request.lower():
                score += 0.2

        if "все [A-ZА-Я][a-zа-я]+" in request.lower() and "плохие" in request.lower():
            score += 0.15

        if "лучшая раса" in request.lower() or "лучшая национальность" in request.lower():
            score += 0.25

        return min(score, 1.0)

    def _evaluate_transparency(self, request: str, context: Optional[Dict[str, Any]]) -> float:
        """Оценивает запрос на соответствие принципу прозрачности."""
        score = 0.0

        if "секрет" in request.lower() and "алгоритм" in request.lower():
            score += 0.2

        if "как обмануть" in request.lower() or "как скрыть" in request.lower():
            score += 0.3

        if "манипулировать" in request.lower() or "ввести в заблуждение" in request.lower():
            score += 0.25

        return min(score, 1.0)

    def _evaluate_autonomy(self, request: str, context: Optional[Dict[str, Any]]) -> float:
        """Оценивает запрос на соответствие принципу автономии."""
        score = 0.0

        if "контролировать" in request.lower() and "мысли" in request.lower():
            score += 0.3

        if "заставить выбрать" in request.lower() or "манипулировать выбором" in request.lower():
            score += 0.25

        if "отменить права" in request.lower() or "ограничить возможности" in request.lower():
            score += 0.2

        return min(score, 1.0)

    def _evaluate_beneficence(self, request: str, context: Optional[Dict[str, Any]]) -> float:
        """Оценивает запрос на соответствие принципу пользы."""
        score = 0.0

        if "вред" in request.lower() and ("пользователю" in request.lower() or "людям" in request.lower()):
            score += 0.4

        if "бесполезная информация" in request.lower() or "пустая трата времени" in request.lower():
            score += 0.15

        if "ложная информация" in request.lower() or "дезинформация" in request.lower():
            score += 0.25

        return min(score, 1.0)

    def _evaluate_accountability(self, request: str, context: Optional[Dict[str, Any]]) -> float:
        """Оценивает запрос на соответствие принципу подотчетности."""
        score = 0.0

        if "анонимно" in request.lower() and ("действие" in request.lower() or "операция" in request.lower()):
            score += 0.25

        if "скрыть ответственность" in request.lower() or "избежать ответственности" in request.lower():
            score += 0.3

        if "уйти от последствий" in request.lower() or "избежать наказания" in request.lower():
            score += 0.2

        return min(score, 1.0)

    def _calculate_confidence(self, principle_scores: Dict[str, float]) -> float:
        """Вычисляет общий уровень доверия к оценке."""
        total_weight = 0
        weighted_sum = 0

        for name, score in principle_scores.items():
            principle = self.principles.get(name)
            weight = principle.weight if principle else 1.0
            weighted_sum += score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.5

    def _generate_rejection_response(self, violations: List["EthicalDecision"]) -> str:
        """
        Генерирует ответ при отклонении запроса по этическим соображениям.

        Args:
            violations: Список обнаруженных нарушений

        Returns:
            str: Ответ для пользователя
        """
        highest_severity = max(v.severity for v in violations)
        primary_violation = next(v for v in violations if v.severity == highest_severity)

        principle_key = primary_violation.principle.lower()

        if principle_key == "privacy":
            return (
                "Извините, но ваш запрос затрагивает вопросы приватности и конфиденциальности. "
                "Я не могу обрабатывать запросы, связанные с личной информацией других людей или "
                "конфиденциальными данными. Пожалуйста, переформулируйте ваш запрос так, чтобы он "
                "не нарушал принципы приватности."
            )
        elif principle_key == "non-maleficence" or principle_key == "non_maleficence":
            return (
                "Извините, но ваш запрос может привести к потенциальному вреду или опасности. "
                "Я не могу участвовать в обсуждении или предоставлении информации, которая может "
                "нанести вред людям или нарушить безопасность. Пожалуйста, переформулируйте ваш запрос."
            )
        elif principle_key == "justice":
            return (
                "Извините, но ваш запрос содержит элементы дискриминации или несправедливости. "
                "Я не могу поддерживать или распространять информацию, которая нарушает принципы "
                "справедливости и равенства. Пожалуйста, переформулируйте ваш запрос более нейтрально."
            )
        else:
            return (
                "Извините, но ваш запрос не соответствует этическим стандартам системы. "
                "Пожалуйста, переформулируйте запрос так, чтобы он соответствовал принципам "
                "уважения, безопасности и пользы для всех пользователей."
            )

    def check_with_context(
        self,
        text: str,
        query: str = "",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Проверяет текст на этические нарушения с учётом контекста

        Args:
            text: Текст для проверки
            query: Оригинальный запрос
            context: Дополнительный контекст

        Returns:
            Dict с результатами проверки и рекомендациями
        """
        logger.info("Проверка этики с контекстом...")

        try:
            if hasattr(self, 'analyze_content'):
                analysis = self.analyze_content(text, context)
                if hasattr(analysis, 'violations'):
                    violations = analysis.violations if isinstance(analysis.violations, list) else []
                else:
                    violations = []
            else:
                violations = []

            critical = [v for v in violations if v.get('severity', 0) > 0.8]
            warnings = [v for v in violations if 0.5 < v.get('severity', 0) <= 0.8]
            minor = [v for v in violations if v.get('severity', 0) <= 0.5]

            overall_score = 1.0 - (len(critical) * 0.3 + len(warnings) * 0.1)
            overall_score = max(0.0, min(1.0, overall_score))

            return {
                'violations': violations,
                'critical_violations': critical,
                'warnings': warnings,
                'minor_violations': minor,
                'overall_score': overall_score,
                'has_violations': len(critical) > 0 or len(warnings) > 0,
                'violation_count': len(violations)
            }

        except Exception as e:
            logger.warning(f"Ошибка при check_with_context: {e}")
            return {
                'violations': [],
                'overall_score': 0.5,
                'has_violations': False,
                'violation_count': 0
            }

    def generate_regeneration_prompt(
        self,
        ethics_result: Dict[str, Any],
        query: str = "",
        response: str = ""
    ) -> str:
        """
        Генерирует промпт для регенерации при этических нарушениях

        Args:
            ethics_result: Результат check_with_context()
            query: Оригинальный запрос
            response: Текущий ответ

        Returns:
            str: Промпт для Qwen
        """
        critical = ethics_result.get('critical_violations', [])
        warnings = ethics_result.get('warnings', [])

        if not critical and not warnings:
            return ""

        parts = []

        if critical:
            parts.append("КРИТИЧЕСКИЕ НАРУШЕНИЯ:")
            for i, v in enumerate(critical[:2], 1):
                principle = v.get('principle', 'unknown')
                desc = v.get('description', v.get('message', ''))
                parts.append(f"{i}. [{principle}] {desc}")

        if warnings:
            parts.append("\nПРЕДУПРЕЖДЕНИЯ:")
            for i, v in enumerate(warnings[:2], 1):
                principle = v.get('principle', 'unknown')
                desc = v.get('description', v.get('message', ''))
                parts.append(f"{i}. [{principle}] {desc}")

        prompt = 'Обнаружены этические нарушения в ответе:\n"'
        prompt += "\n".join(parts)
        prompt += "\n\nПереформулируй ответ, устранив нарушения. Будь этичной."

        return prompt
