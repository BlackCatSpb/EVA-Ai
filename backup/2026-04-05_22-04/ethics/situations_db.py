"""
Модуль базы данных этических ситуаций для ЕВА - загрузка, хранение, управление
"""
import os
import logging
import json
import time
from typing import Dict, List, Optional, Any

logger = logging.getLogger("eva.ethics.situations")

class EthicalIssue:
    def __init__(self, name, description, type, priority, evidence, timestamp=None, resolved=False, resolution=None):
        self.name = name
        self.description = description
        self.type = type
        self.priority = priority
        self.evidence = evidence
        self.timestamp = timestamp or time.time()
        self.resolved = resolved
        self.resolution = resolution


class SituationsDBMixin:
    """Миксин для управления базой данных этических ситуаций."""

    def _load_cache(self, file_path: str) -> Dict[str, Any]:
        """Загружает кэш из файла."""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {"solutions": [], "reviews": []}
        except Exception as e:
            logger.error(f"Ошибка загрузки кэша: {e}")
            return {"solutions": [], "reviews": []}

    def _save_cache(self):
        """Сохраняет кэш в файл."""
        try:
            with open(self.solutions_cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.solutions_cache, f, ensure_ascii=False, indent=2)

            with open(self.review_cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.review_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения кэша: {e}")

    def _load_ethical_issues(self):
        """Загружает этические проблемы из файла."""
        try:
            issues_file = os.path.join(self.cache_dir, "ethical_issues.json")

            if os.path.exists(issues_file):
                with open(issues_file, 'r', encoding='utf-8') as f:
                    issues_data = json.load(f)

                self.ethical_issues = [
                    EthicalIssue(
                        name=issue["name"],
                        description=issue["description"],
                        type=issue["type"],
                        priority=issue["priority"],
                        evidence=issue["evidence"],
                        timestamp=issue.get("timestamp", time.time()),
                        resolved=issue.get("resolved", False),
                        resolution=issue.get("resolution")
                    ) for issue in issues_data
                ]

                logger.debug(f"Загружено {len(self.ethical_issues)} этических проблем")
            else:
                self.ethical_issues = [
                    EthicalIssue(
                        name="нейроэстетика",
                        description="Отсутствие знаний о взаимодействии нейронауки и эстетики",
                        type="incomplete",
                        priority=0.6,
                        evidence=["Запросы о нейроэстетике не могут быть полноценно обработаны"]
                    ),
                    EthicalIssue(
                        name="этика_искусственного_интеллекта",
                        description="Противоречивые подходы к этике ИИ в разных источниках",
                        type="contradictory",
                        priority=0.7,
                        evidence=["Разные источники дают противоречивые рекомендации"]
                    ),
                    EthicalIssue(
                        name="автономия_человека",
                        description="Недостаток знаний о балансе автономии человека и ИИ",
                        type="missing",
                        priority=0.65,
                        evidence=["Частые запросы о контроле над ИИ"]
                    )
                ]

                self._save_ethical_issues()

                logger.info("Созданы базовые этические проблемы")

        except Exception as e:
            logger.error(f"Ошибка загрузки этических проблем: {e}")
            self.ethical_issues = []

    def _save_ethical_issues(self):
        """Сохраняет этические проблемы в файл."""
        try:
            issues_file = os.path.join(self.cache_dir, "ethical_issues.json")
            issues_data = [
                {
                    "name": issue.name,
                    "description": issue.description,
                    "type": issue.type,
                    "priority": issue.priority,
                    "evidence": issue.evidence,
                    "timestamp": issue.timestamp,
                    "resolved": issue.resolved,
                    "resolution": issue.resolution
                } for issue in self.ethical_issues
            ]

            with open(issues_file, 'w', encoding='utf-8') as f:
                json.dump(issues_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения этических проблем: {e}")

    def get_ethical_issues(self, limit: int = 10, min_priority: float = 0.5) -> List[EthicalIssue]:
        """
        Возвращает список этических проблем.

        Args:
            limit: Максимальное количество проблем
            min_priority: Минимальный приоритет

        Returns:
            List[EthicalIssue]: Список этических проблем
        """
        issues = [
            issue for issue in self.ethical_issues
            if not issue.resolved and issue.priority >= min_priority
        ]
        issues.sort(key=lambda x: x.priority, reverse=True)

        return issues[:limit]

    def add_ethical_issue(self, issue: EthicalIssue):
        """
        Добавляет новую этическую проблему.

        Args:
            issue: Этическая проблема
        """
        if not issue:
            return

        self.ethical_issues.append(issue)
        self._save_ethical_issues()
        logger.info(f"Добавлена новая этическая проблема: {issue.name} (приоритет: {issue.priority})")

    def resolve_ethical_issue(self, issue_name: str, resolution: Dict[str, Any]):
        """
        Помечает этическую проблему как решенную.

        Args:
            issue_name: Название проблемы
            resolution: Описание решения
        """
        if not issue_name:
            return

        for issue in self.ethical_issues:
            if issue.name == issue_name and not issue.resolved:
                issue.resolved = True
                issue.resolution = resolution
                self._save_ethical_issues()
                logger.info(f"Этическая проблема решена: {issue_name}")
                return
