"""Monitoring, progress tracking, health checks, and metrics for EVA learning scheduler."""

import time
import json
import logging
from typing import Dict, List, Optional, Any

import numpy as np

from .scheduler_core import LearningTask

logger = logging.getLogger("eva_ai.learning_scheduler")


class MonitorMixin:
    """Mixin providing monitoring, health checks, statistics, and diagnostics."""

    def get_scheduler_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику планировщика."""
        with self.lock:
            return {
                "total_tasks": self.stats["total_tasks"],
                "completed_tasks": self.stats["completed_tasks"],
                "failed_tasks": self.stats["failed_tasks"],
                "pending_tasks": self.stats["pending_tasks"],
                "in_progress_tasks": self.stats["in_progress_tasks"],
                "tasks_per_hour": self._calculate_tasks_per_hour(),
                "avg_completion_time": self._calculate_average_completion_time(),
                "failure_rate": self._calculate_failure_rate(),
                "resource_usage": self.resource_allocation.get_slot_usage(),
                "timestamp": time.time()
            }

    def _calculate_tasks_per_hour(self) -> float:
        """Рассчитывает количество задач в час."""
        with self.lock:
            if self.stats["completed_tasks"] == 0:
                return 0.0

            uptime = time.time() - self.start_time
            if uptime <= 0:
                return 0.0

            return (self.stats["completed_tasks"] / max(1, uptime)) * 3600

    def _calculate_average_completion_time(self) -> float:
        """Рассчитывает среднее время выполнения задачи."""
        with self.lock:
            completed_tasks = [task for task in self.task_registry.values() if task.status == "completed"]
            if not completed_tasks:
                return 0.0

            durations = [task.get_duration() for task in completed_tasks if task.get_duration() is not None]
            if not durations:
                return 0.0

            return np.mean(durations)

    def _calculate_failure_rate(self) -> float:
        """Рассчитывает процент неудачных задач."""
        with self.lock:
            if self.stats["total_tasks"] == 0:
                return 0.0

            return self.stats["failed_tasks"] / self.stats["total_tasks"]

    def get_scheduler_health_report(self) -> Dict[str, Any]:
        """Возвращает отчет о здоровье планировщика."""
        stats = self.get_scheduler_statistics()

        health_score = 100.0

        if stats["pending_tasks"] > 50:
            health_score -= min(30, stats["pending_tasks"] * 0.5)

        if stats["failure_rate"] > 0.2:
            health_score -= min(40, stats["failure_rate"] * 200)
        elif stats["failure_rate"] > 0.1:
            health_score -= min(20, stats["failure_rate"] * 100)

        if stats["avg_completion_time"] > 300:
            health_score -= min(20, (stats["avg_completion_time"] - 300) / 10)

        recommendations = []
        if stats["pending_tasks"] > 50:
            recommendations.append(
                "Очень высокая загрузка планировщика. Рассмотрите возможность "
                "увеличения количества рабочих потоков или оптимизации задач."
            )
        elif stats["pending_tasks"] > 20:
            recommendations.append(
                "Высокая загрузка планировщика. Проверьте приоритеты задач и "
                "рассмотрите возможность увеличения ресурсов."
            )

        if stats["failed_tasks"] > 10:
            recommendations.append(
                "Высокое количество неудачных задач. Проверьте задачи с высоким приоритетом "
                "и увеличьте таймаут для сложных задач."
            )
        elif stats["failed_tasks"] > 5:
            recommendations.append(
                "Умеренное количество неудачных задач. Проверьте задачи, которые "
                "часто завершаются неудачей и оптимизируйте их."
            )

        return {
            "health_score": max(0, min(100, health_score)),
            "statistics": stats,
            "recommendations": recommendations,
            "timestamp": time.time()
        }

    def get_scheduler_diagnostics(self) -> Dict[str, Any]:
        """Возвращает диагностику планировщика."""
        health = self.get_scheduler_health_report()
        stats = self.get_scheduler_statistics()

        active_tasks = []
        with self.lock:
            for task_id in self.resource_allocation.get_active_tasks():
                task = self.task_registry.get(task_id)
                if task:
                    active_tasks.append({
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "concept": task.concept,
                        "priority": task.priority,
                        "duration": task.get_duration()
                    })

        queued_tasks = []
        with self.lock:
            for task in sorted(self.task_queue, key=lambda x: (x.priority, x.scheduled_time)):
                queued_tasks.append({
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "concept": task.concept,
                    "priority": task.priority,
                    "scheduled_time": task.scheduled_time,
                    "is_overdue": task.is_overdue()
                })

        return {
            "health": health,
            "statistics": stats,
            "active_tasks": active_tasks,
            "queued_tasks": queued_tasks,
            "resource_allocation": {
                "max_concurrent": self.resource_allocation.max_concurrent,
                "current_concurrent": self.resource_allocation.current_concurrent,
                "usage": self.resource_allocation.get_slot_usage()
            },
            "timestamp": time.time()
        }

    def export_scheduler_diagnostics(self, file_path: str) -> bool:
        """Экспортирует диагностику планировщика в файл."""
        try:
            diagnostics = {
                "metadata": {
                    "export_time": time.time(),
                    "format_version": "1.0"
                },
                "scheduler_health": self.get_scheduler_health_report(),
                "scheduler_statistics": self.get_scheduler_statistics(),
                "diagnostics": self.get_scheduler_diagnostics()
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(diagnostics, f, ensure_ascii=False, indent=2)

            logger.info(f"Диагностика планировщика экспортирована в {file_path}")
            return True

        except Exception as e:
            logger.error(f"Ошибка экспорта диагностики планировщика: {e}")
            return False

    def get_system_summary(self) -> str:
        """Возвращает краткую сводку о системе."""
        stats = self.get_scheduler_statistics()
        health = self.get_scheduler_health_report()

        summary = (
            f"Планировщик задач обучения\n"
            f"{'=' * 30}\n\n"
            f"Задачи: всего {stats['total_tasks']}, "
            f"выполнено {stats['completed_tasks']}, "
            f"неудачных {stats['failed_tasks']}\n"
            f"Очередь: {stats['pending_tasks']} задач, "
            f"выполняется {stats['in_progress_tasks']}\n"
            f"Ресурсы: загружено {stats['resource_usage']:.0%} "
            f"({self.resource_allocation.current_concurrent}/{self.resource_allocation.max_concurrent})\n\n"
            f"Здоровье системы: {health['health_score']:.1f}/100\n"
        )

        if health["recommendations"]:
            summary += "Рекомендации:\n"
            for i, rec in enumerate(health["recommendations"], 1):
                summary += f"{i}. {rec}\n"

        return summary
