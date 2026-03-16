"""
LearningOpportunityDetector: выявляет возможности для обучения
"""
from __future__ import annotations
from typing import List, Dict, Any
from .base_detector import BaseDetector


class LearningOpportunityDetector(BaseDetector):
    name = "learning"
    cooldown_sec = 15.0

    def _do_probe(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        brain = context.get('brain')
        if not brain:
            return []

        # Считаем, что обучение имеет смысл при наличии pending/in_progress возможностей
        stats = {}
        try:
            op_list = []
            if hasattr(brain, '_get_learning_opportunities'):
                op_list = brain._get_learning_opportunities() or []
            # оценка: если есть хотя бы одна возможность — возвращаем TrainingJob
            if op_list:
                stats['count'] = len(op_list)
        except Exception:
            pass

        # Проверим, не идёт ли уже обучение
        trainer = getattr(brain, 'memory_graph_trainer', None)
        if trainer is None:
            sa = getattr(brain, 'self_analyzer', None)
            trainer = getattr(sa, 'memory_graph_trainer', None) if sa else None
        if trainer is None:
            return []
        try:
            busy = False
            if hasattr(trainer, 'is_training'):
                busy = bool(trainer.is_training() if callable(trainer.is_training) else trainer.is_training)
            elif hasattr(trainer, 'is_busy'):
                busy = bool(trainer.is_busy() if callable(trainer.is_busy) else trainer.is_busy)
            if busy:
                return []
        except Exception:
            return []

        if not stats.get('count'):
            return []

        # Формируем заявку на TrainingJob
        return [{
            "job_type": "TrainingJob",
            "params": {},
        }]
    cooldown_sec = 15.0

    def _do_probe(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        brain = context.get('brain')
        if not brain:
            return []

        # Считаем, что обучение имеет смысл при наличии pending/in_progress возможностей
        stats = {}
        try:
            op_list = []
            if hasattr(brain, '_get_learning_opportunities'):
                op_list = brain._get_learning_opportunities() or []
            # оценка: если есть хотя бы одна возможность — возвращаем TrainingJob
            if op_list:
                stats['count'] = len(op_list)
        except (AttributeError, TypeError, RuntimeError) as e:
            logger.debug(f"Error getting learning opportunities: {e}")

        # Проверим, не идёт ли уже обучение
        trainer = getattr(brain, 'memory_graph_trainer', None)
        if trainer is None:
            sa = getattr(brain, 'self_analyzer', None)
            trainer = getattr(sa, 'memory_graph_trainer', None) if sa else None
        if trainer is None:
            return []
        try:
            busy = False
            if hasattr(trainer, 'is_training'):
                busy = bool(trainer.is_training() if callable(trainer.is_training) else trainer.is_training)
            elif hasattr(trainer, 'is_busy'):
                busy = bool(trainer.is_busy() if callable(trainer.is_busy) else trainer.is_busy)
            if busy:
                return []
        except (AttributeError, TypeError, RuntimeError) as e:
            logger.debug(f"Error checking trainer status: {e}")
            return []

        if not stats.get('count'):
            return []

        # Формируем заявку на TrainingJob
        return [{
            "job_type": "TrainingJob",
            "params": {},
        }]
