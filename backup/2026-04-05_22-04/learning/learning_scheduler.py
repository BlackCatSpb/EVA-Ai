"""Модуль планировщика задач обучения для ЕВА - управление задачами обучения и их выполнение"""

from .scheduler_core import LearningTask, ResourceAllocation, LearningSchedulerCore
from .scheduler_tasks import TaskManagerMixin
from .scheduler_triggers import TriggerMixin
from .scheduler_monitor import MonitorMixin


class LearningScheduler(LearningSchedulerCore, TaskManagerMixin, TriggerMixin, MonitorMixin):
    """Планировщик задач обучения для ЕВА - управление задачами обучения и их выполнение."""
    pass


def create_scheduler(brain=None, cache_dir=None):
    """Factory function to create a LearningScheduler instance."""
    return LearningScheduler(brain=brain, cache_dir=cache_dir)


def create_learning_task(task_id, task_type, concept, priority, scheduled_time, **kwargs):
    """Factory function to create a LearningTask."""
    return LearningTask(
        task_id=task_id,
        task_type=task_type,
        concept=concept,
        priority=priority,
        scheduled_time=scheduled_time,
        **kwargs
    )
