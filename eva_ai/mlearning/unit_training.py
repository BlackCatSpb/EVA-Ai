"""Training orchestration and model updates for MLUnit"""
from __future__ import annotations

import logging

logger = logging.getLogger("eva_ai.ml_unit")


def _init_training_orchestrator(self):
    """Обучение теперь через SelfDialogLearning, TrainingOrchestrator не используется."""
    logger.info("TrainingOrchestrator отключен - обучение через SelfDialogLearning")
    return True
