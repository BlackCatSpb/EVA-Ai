"""Система самообучения через самодиалог для ЕВА.

Этот модуль реализует механизм самообучения системы через создание
внутреннего диалога между различными аспектами системы (AI assistant,
critic, learner, etc.) для выявления пробелов в знаниях и их заполнения.

Модуль рефакторирован: основная логика перенесена в dialog_core.py,
dialog_topics.py, dialog_generation.py, dialog_learning.py.
"""
from eva_ai.learning.dialog_types import DialogRole, DialogTurn, LearningType, SelfDialog
from eva_ai.learning.dialog_core import SelfDialogLearning, create_self_dialog_learning

SelfDialogLearningSystem = SelfDialogLearning

__all__ = [
    "DialogRole",
    "DialogTurn",
    "LearningType",
    "SelfDialog",
    "SelfDialogLearning",
    "SelfDialogLearningSystem",
    "create_self_dialog_learning",
]
