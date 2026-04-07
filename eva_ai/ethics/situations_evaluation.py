"""
Модуль оценки этических ситуаций для ЕВА - алгоритмы оценки, скоринг
"""
import os
import logging
import json
import time
from typing import Dict, List, Optional, Any
from io import BytesIO

logger = logging.getLogger("eva_ai.ethics.situations")

try:
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    np = None


class SituationsEvaluationMixin:
    """Миксин для алгоритмов оценки и скоринга этических ситуаций."""

    def _calculate_confidence(self, assessments) -> float:
        """Рассчитывает общую уверенность в решении."""
        if not assessments:
            return 0.5

        total_confidence = sum(a.confidence for a in assessments)
        avg_confidence = total_confidence / len(assessments)

        scores = [a.score for a in assessments]
        if len(scores) > 1 and np:
            variance = np.var(scores)
            confidence = avg_confidence * (1 - min(0.5, variance))
        else:
            confidence = avg_confidence

        return max(0.1, min(1.0, confidence))

    def get_situation_dashboard_data(self) -> Dict[str, Any]:
        """
        Возвращает данные для дашборда этических ситуаций.

        Returns:
            Dict[str, Any]: Данные для дашборда
        """
        try:
            recent_solutions = self.solutions_cache.get("solutions", [])[:10]

            decision_counts = {}
            for solution in recent_solutions:
                decision = solution.get("decision", "unknown")
                decision_counts[decision] = decision_counts.get(decision, 0) + 1

            open_issues = [i for i in self.ethical_issues if not i.resolved]
            high_priority_issues = [i for i in open_issues if i.priority >= 0.7]

            try:
                risk_data = self.risk_assessor.get_risk_dashboard_data()
            except Exception as e:
                logger.warning(f"Ошибка получения данных о рисках: {e}")
                risk_data = {}

            return {
                "total_solutions": len(self.solutions_cache.get("solutions", [])),
                "recent_solutions": recent_solutions,
                "decision_counts": decision_counts,
                "open_issues_count": len(open_issues),
                "high_priority_issues_count": len(high_priority_issues),
                "high_priority_issues": [
                    {
                        "name": i.name,
                        "description": i.description,
                        "priority": i.priority,
                        "type": i.type
                    } for i in high_priority_issues[:5]
                ],
                "risk_data": risk_data,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Ошибка получения данных дашборда: {e}")
            return {"error": str(e), "timestamp": time.time()}

    def generate_situation_visualization(self, view_type: str = "issues") -> str:
        """
        Генерирует визуализацию данных о ситуациях.

        Args:
            view_type: Тип визуализации

        Returns:
            str: Изображение в формате base64
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib недоступен для визуализации")
            return ""

        try:
            dashboard_data = self.get_situation_dashboard_data()

            if "error" in dashboard_data:
                return ""

            fig = Figure(figsize=(10, 6), dpi=100)
            ax = fig.add_subplot(111)

            if view_type == "issues":
                issues = [i["name"] for i in dashboard_data.get("high_priority_issues", [])]
                priorities = [i["priority"] for i in dashboard_data.get("high_priority_issues", [])]

                if issues and priorities:
                    y_pos = range(len(issues))
                    ax.barh(y_pos, priorities, align='center', color='salmon')
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(issues)
                    ax.invert_yaxis()
                    ax.set_xlabel('Приоритет (0-1)')
                    ax.set_title('Высокоприоритетные этические проблемы')
                    ax.set_xlim(0, 1)
                else:
                    ax.text(0.5, 0.5, 'Нет данных для отображения',
                           ha='center', va='center', transform=ax.transAxes)

            elif view_type == "decisions":
                decision_counts = dashboard_data.get("decision_counts", {})
                if decision_counts:
                    decisions = list(decision_counts.keys())
                    counts = list(decision_counts.values())

                    ax.pie(counts, labels=decisions, autopct='%1.1f%%', startangle=90)
                    ax.axis('equal')
                    ax.set_title('Распределение этических решений')
                else:
                    ax.text(0.5, 0.5, 'Нет данных для отображения',
                           ha='center', va='center', transform=ax.transAxes)

            buf = BytesIO()
            fig.tight_layout()
            canvas = FigureCanvasAgg(fig)
            canvas.print_png(buf)

            buf.seek(0)
            import base64
            img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            return f"data:image/png;base64,{img_data}"

        except Exception as e:
            logger.error(f"Ошибка генерации визуализации ситуаций: {e}")
            return ""

    def export_ethics_data(self, file_path: str) -> bool:
        """
        Экспортирует данные этической рамки в файл.

        Args:
            file_path: Путь к файлу для экспорта

        Returns:
            bool: Успешно ли экспортировано
        """
        if not file_path:
            return False

        try:
            export_data = {
                "metadata": {
                    "export_time": time.time(),
                    "format_version": "1.0"
                },
                "solutions": self.solutions_cache.get("solutions", []),
                "reviews": self.review_cache.get("reviews", []),
                "ethical_issues": [
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
                ],
                "system_health": self.get_system_health()
            }

            try:
                principles = self.principles_manager.get_all_principles()
                export_data["principles"] = [
                    {
                        "id": pid,
                        **principle.__dict__
                    } for pid, principle in principles.items()
                ]
            except Exception as e:
                logger.warning(f"Ошибка экспорта принципов: {e}")
                export_data["principles"] = []

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Данные этической рамки экспортированы в {file_path}")
            return True

        except Exception as e:
            logger.error(f"Ошибка экспорта данных этической рамки: {e}")
            return False

    def import_ethics_data(self, file_path: str) -> bool:
        """
        Импортирует данные этической рамки из файла.

        Args:
            file_path: Путь к файлу для импорта

        Returns:
            bool: Успешно ли импортировано
        """
        if not file_path or not os.path.exists(file_path):
            return False

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if "principles" in data:
                try:
                    for principle_data in data["principles"]:
                        principle = EthicalPrinciple(
                            name=principle_data["name"],
                            description=principle_data["description"],
                            weight=principle_data["weight"],
                            threshold=principle_data["threshold"],
                            category=principle_data["category"],
                            last_updated=principle_data["last_updated"],
                            active=principle_data["active"]
                        )
                        self.principles_manager.add_principle(principle)
                except Exception as e:
                    logger.error(f"Ошибка импорта принципов: {e}")

            if "solutions" in data:
                self.solutions_cache["solutions"] = data["solutions"]
                self._save_cache()

            if "reviews" in data:
                self.review_cache["reviews"] = data["reviews"]
                self._save_cache()

            if "ethical_issues" in data:
                from .situations_db import EthicalIssue
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
                    ) for issue in data["ethical_issues"]
                ]
                self._save_ethical_issues()

            logger.info(f"Данные этической рамки импортированы из {file_path}")
            return True

        except Exception as e:
            logger.error(f"Ошибка импорта данных этической рамки: {e}")
            return False

    def get_system_health(self) -> Dict[str, Any]:
        """
        Возвращает отчет о здоровье системы этической рамки.

        Returns:
            Dict: Отчет о здоровье
        """
        try:
            try:
                principles = self.principles_manager.get_all_principles()
                total_principles = len(principles)

                low_compliance_count = 0
                for principle_id, principle in principles.items():
                    try:
                        history = self.principles_manager.get_assessment_history(principle_id, days=7)
                        if history:
                            avg_score = sum(item["score"] for item in history) / len(history)
                            if avg_score < principle.threshold * 0.8:
                                low_compliance_count += 1
                    except Exception as e:
                        logger.warning(f"Ошибка анализа принципа {principle_id}: {e}")
            except Exception as e:
                logger.warning(f"Ошибка получения данных о принципах: {e}")
                total_principles = 0
                low_compliance_count = 0

            health_score = 100.0

            if low_compliance_count > 0:
                health_score -= min(40, low_compliance_count * 15)

            open_issues = len([i for i in self.ethical_issues if not i.resolved])
            if open_issues > 5:
                health_score -= min(30, (open_issues - 2) * 5)

            recommendations = []
            if low_compliance_count > 0:
                recommendations.append(
                    f"Обнаружено {low_compliance_count} принципов с низким уровнем соблюдения. "
                    "Рассмотрите возможность улучшения обработки соответствующих сценариев."
                )

            if open_issues > 0:
                recommendations.append(
                    f"Есть {open_issues} нерешенных этических проблем. "
                    "Рекомендуется сосредоточиться на их решении."
                )

            if not recommendations:
                recommendations.append(
                    "Этическая рамка работает стабильно. Продолжайте мониторинг для "
                    "раннего выявления потенциальных проблем."
                )

            return {
                "health_score": max(0, min(100, health_score)),
                "total_principles": total_principles,
                "low_compliance_count": low_compliance_count,
                "open_issues_count": open_issues,
                "recommendations": recommendations,
                "timestamp": time.time()
            }

        except Exception as e:
            logger.error(f"Ошибка получения отчета о здоровье системы: {e}")
            return {
                "health_score": 0,
                "error": str(e),
                "timestamp": time.time()
            }

    def close(self):
        """Закрывает обработчик этических ситуаций и освобождает ресурсы."""
        logger.info("Закрытие обработчика этических ситуаций...")
        try:
            self._save_cache()
            self._save_ethical_issues()
        except Exception as e:
            logger.error(f"Ошибка при закрытии: {e}")
        logger.info("Обработчик этических ситуаций закрыт")
