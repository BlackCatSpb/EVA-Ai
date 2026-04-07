"""
Модуль нарушений этической рамки для ЕВА - обнаружение, логирование, отчётность
"""
import os
import logging
import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

from .violation_id_manager import (
    generate_violation_id,
    parse_violation_id,
    is_valid_violation_id,
    get_violation_principle,
    get_violation_timestamp
)
from .framework_principles import EthicalPrinciple

logger = logging.getLogger("eva_ai.ethics")

@dataclass
class EthicalDecision:
    """Представляет этическое решение."""
    approved: bool
    principle: str
    severity: float
    description: str
    context: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolution: Optional[str] = None
    resolution_timestamp: Optional[float] = None
    source: str = "system"
    violation_id: str = field(init=False)

    def __post_init__(self):
        """Генерируем уникальный ID при создании объекта."""
        self.violation_id = generate_violation_id(self.principle, self.timestamp)

@dataclass
class EthicalReview:
    """Результат этического обзора контента."""
    content: str
    review_type: str
    reviewer: str
    decision: EthicalDecision
    timestamp: float = field(default_factory=time.time)


class EthicsViolationsMixin:
    """Миксин для управления нарушениями, сохранения и отчётности."""

    def _load_violations_and_stats(self):
        """Загружает нарушения и статистику из хранилища."""
        try:
            cache_dir = self.cache_dir
            os.makedirs(cache_dir, exist_ok=True)

            violations_file = os.path.join(cache_dir, 'violations.json')
            if os.path.exists(violations_file):
                with open(violations_file, 'r', encoding='utf-8') as f:
                    loaded_violations = json.load(f)
                    for vdata in loaded_violations:
                        try:
                            self.violations.append(EthicalDecision(
                                approved=vdata.get("approved", False),
                                principle=vdata.get("principle", "unknown"),
                                severity=vdata.get("severity", 0.0),
                                description=vdata.get("description", ""),
                                context=vdata.get("context", {}),
                                timestamp=vdata.get("timestamp", time.time()),
                                resolved=vdata.get("resolved", False),
                                resolution=vdata.get("resolution"),
                                resolution_timestamp=vdata.get("resolution_timestamp"),
                                source=vdata.get("source", "system")
                            ))
                        except Exception:
                            pass
                    logger.debug(f"Загружено {len(loaded_violations)} нарушений")

            stats_file = os.path.join(cache_dir, 'ethics_stats.json')
            if os.path.exists(stats_file):
                with open(stats_file, 'r', encoding='utf-8') as f:
                    self.stats = json.load(f)
                    logger.debug("Статистика этики загружена")

        except Exception as e:
            logger.error(f"Ошибка загрузки нарушений и статистики: {e}")

    def _save_violations(self):
        """Сохраняет нарушения в файл."""
        try:
            violations_data = []
            for violation in self.violations:
                violations_data.append({
                    "approved": violation.approved,
                    "principle": violation.principle,
                    "severity": violation.severity,
                    "description": violation.description,
                    "context": violation.context,
                    "timestamp": violation.timestamp,
                    "resolved": violation.resolved,
                    "resolution": violation.resolution,
                    "resolution_timestamp": violation.resolution_timestamp,
                    "source": violation.source,
                    "violation_id": violation.violation_id
                })

            with open(self.violations_file, 'w', encoding='utf-8') as f:
                json.dump(violations_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения нарушений: {e}", exc_info=True)

    def _save_principles(self):
        """Сохраняет принципы в файл."""
        try:
            principles_data = {}
            for name, principle in self.principles.items():
                principles_data[name] = {
                    "name": principle.name,
                    "description": principle.description,
                    "weight": principle.weight,
                    "threshold": principle.threshold,
                    "category": principle.category,
                    "priority": principle.priority
                }

            with open(self.principles_file, 'w', encoding='utf-8') as f:
                json.dump(principles_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения принципов: {e}", exc_info=True)

    def _save_stats(self):
        """Сохраняет статистику в файл."""
        try:
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения статистики: {e}", exc_info=True)

    def get_violation_history(self, limit: int = 50,
                              principle: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Возвращает историю нарушений.

        Args:
            limit: Максимальное количество записей
            principle: Фильтр по принципу

        Returns:
            List[Dict]: Список нарушений
        """
        with self.lock:
            filtered = self.violations
            if principle:
                filtered = [v for v in filtered if v.principle == principle]

            filtered = sorted(filtered, key=lambda x: x.timestamp, reverse=True)

            result = [v.__dict__ for v in filtered[:limit]]
            return result

    def resolve_violation(self, violation_id: str, resolution: str,
                         reviewer: str = "system") -> bool:
        """
        Разрешает нарушение.

        Args:
            violation_id: ID нарушения
            resolution: Описание решения
            reviewer: Кто разрешил нарушение

        Returns:
            bool: Успешно ли разрешено
        """
        with self.lock:
            if not is_valid_violation_id(violation_id):
                logger.warning(f"Недействительный ID нарушения: {violation_id}")
                return False

            expected_principle = get_violation_principle(violation_id)
            if expected_principle and expected_principle not in self.principles:
                logger.warning(f"Принцип '{expected_principle}' из ID не найден в системе")
                return False

            for violation in self.violations:
                if violation.violation_id == violation_id:
                    violation.resolved = True
                    violation.resolution = resolution
                    violation.resolution_timestamp = time.time()

                    self.stats["resolved_violations"] += 1
                    self.stats["pending_reviews"] = max(0, self.stats["pending_reviews"] - 1)

                    self._save_violations()
                    self._save_stats()

                    logger.info(f"Нарушение {violation_id} разрешено")
                    return True

            logger.warning(f"Нарушение {violation_id} не найдено для разрешения")
            return False

    def get_active_violations(self) -> List[Dict[str, Any]]:
        """Возвращает список активных (неразрешенных) нарушений."""
        with self.lock:
            active = [v for v in self.violations if not v.resolved]
            active = sorted(active, key=lambda x: x.severity, reverse=True)
            return [v.__dict__ for v in active]

    def get_ethics_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику этической рамки."""
        with self.lock:
            return {
                "total_assessments": self.stats["total_assessments"],
                "violations_detected": self.stats["violations_detected"],
                "high_severity_violations": self.stats["high_severity_violations"],
                "resolved_violations": self.stats["resolved_violations"],
                "pending_reviews": self.stats["pending_reviews"],
                "last_assessment": self.stats["last_assessment"],
                "violation_rate": (self.stats["violations_detected"] / self.stats["total_assessments"]) \
                    if self.stats["total_assessments"] > 0 else 0
            }

    def export_ethics_data(self, file_path: str) -> bool:
        """
        Экспортирует данные этической рамки в файл.

        Args:
            file_path: Путь к файлу для экспорта

        Returns:
            bool: Успешно ли экспортировано
        """
        try:
            export_data = {
                "metadata": {
                    "format_version": "1.0",
                    "exported_at": time.time(),
                    "system": "ЕВА"
                },
                "principles": {
                    name: {
                        "name": principle.name,
                        "description": principle.description,
                        "weight": principle.weight,
                        "threshold": principle.threshold,
                        "category": principle.category,
                        "priority": principle.priority
                    } for name, principle in self.principles.items()
                },
                "violations": [{
                    "approved": v.approved,
                    "principle": v.principle,
                    "severity": v.severity,
                    "description": v.description,
                    "context": v.context,
                    "timestamp": v.timestamp,
                    "resolved": v.resolved,
                    "resolution": v.resolution,
                    "resolution_timestamp": v.resolution_timestamp,
                    "source": v.source,
                    "violation_id": v.violation_id
                } for v in self.violations],
                "statistics": self.stats
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Данные этической рамки экспортированы в {file_path}")
            return True

        except Exception as e:
            logger.error(f"Ошибка экспорта данных этической рамки: {e}", exc_info=True)
            return False

    def import_ethics_data(self, file_path: str) -> bool:
        """
        Импортирует данные этической рамки из файла.

        Args:
            file_path: Путь к файлу для импорта

        Returns:
            bool: Успешно ли импортировано
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)

            self.principles = {}
            for name, data in import_data["principles"].items():
                self.principles[name] = EthicalPrinciple(
                    name=data["name"],
                    description=data["description"],
                    weight=data["weight"],
                    threshold=data["threshold"],
                    category=data["category"],
                    priority=data["priority"]
                )

            self.violations = []
            for data in import_data["violations"]:
                if "violation_id" in data and is_valid_violation_id(data["violation_id"]):
                    self.violations.append(EthicalDecision(
                        approved=data["approved"],
                        principle=data["principle"],
                        severity=data["severity"],
                        description=data["description"],
                        context=data["context"],
                        timestamp=data["timestamp"],
                        resolved=data["resolved"],
                        resolution=data.get("resolution"),
                        resolution_timestamp=data.get("resolution_timestamp"),
                        source=data.get("source", "system")
                    ))
                else:
                    violation = EthicalDecision(
                        approved=data["approved"],
                        principle=data["principle"],
                        severity=data["severity"],
                        description=data["description"],
                        context=data["context"],
                        timestamp=data["timestamp"],
                        resolved=data["resolved"],
                        resolution=data.get("resolution"),
                        resolution_timestamp=data.get("resolution_timestamp"),
                        source=data.get("source", "system")
                    )
                    self.violations.append(violation)

            self.stats = import_data["statistics"]

            self._save_principles()
            self._save_violations()
            self._save_stats()

            logger.info(f"Данные этической рамки импортированы из {file_path}")
            return True

        except Exception as e:
            logger.error(f"Ошибка импорта данных этической рамки: {e}", exc_info=True)
            return False

    def _analyze_ethical_trends(self) -> Dict[str, Any]:
        """
        Анализирует тенденции в этических нарушениях.

        Returns:
            Dict: Результат анализа тенденций
        """
        trends = {
            "by_principle": defaultdict(int),
            "by_severity": {
                "low": 0,
                "medium": 0,
                "high": 0
            },
            "by_category": defaultdict(int),
            "time_trends": []
        }

        recent_violations = self.violations[-100:]

        for violation in recent_violations:
            trends["by_principle"][violation.principle] += 1

            principle = self.principles.get(violation.principle)
            if principle:
                trends["by_category"][principle.category] += 1

            if violation.severity > 0.8:
                trends["by_severity"]["high"] += 1
            elif violation.severity > 0.6:
                trends["by_severity"]["medium"] += 1
            else:
                trends["by_severity"]["low"] += 1

        now = time.time()
        week_ago = now - 7 * 24 * 3600
        daily_counts = [0] * 7

        for violation in self.violations:
            if violation.timestamp >= week_ago:
                day = int((violation.timestamp - week_ago) / (24 * 3600))
                if 0 <= day < 7:
                    daily_counts[day] += 1

        trends["time_trends"] = daily_counts

        return dict(trends)

    def generate_ethics_report(self) -> Dict[str, Any]:
        """
        Генерирует отчет по этической активности.

        Returns:
            Dict: Отчет
        """
        with self.lock:
            trends = self._analyze_ethical_trends()

            main_issues = []
            if trends["by_severity"]["high"] > 0:
                high_principle = max(trends["by_principle"], key=trends["by_principle"].get)
                main_issues.append(f"Высокая частота нарушений принципа '{high_principle}'")

            if trends["by_severity"]["high"] > 5:
                main_issues.append("Критически высокий уровень серьезных нарушений")

            recommendations = []
            if trends["by_severity"]["high"] > 0:
                recommendations.append(
                    "Рассмотрите возможность усиления проверок для принципа с наибольшим "
                    "количеством нарушений"
                )
            if trends["time_trends"][-1] > trends["time_trends"][-2] * 1.5:
                recommendations.append(
                    "Наблюдается рост числа нарушений, рекомендуется провести анализ причин"
                )

            return {
                "timestamp": time.time(),
                "statistics": self.get_ethics_statistics(),
                "trends": trends,
                "main_issues": main_issues,
                "recommendations": recommendations
            }
