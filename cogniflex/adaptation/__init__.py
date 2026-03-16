"""Пакет адаптации для CogniFlex - управление профилями пользователей и адаптация системы"""

from .adaptation_core import AdaptationManager
from .adaptation_profiles import UserFeedback, UserProfile

__all__ = [
    'AdaptationManager',
    'UserFeedback',
    'UserProfile'
]

# Логирование инициализации пакета
import logging
logger = logging.getLogger("cogniflex.adaptation")
logger.debug("Пакет адаптации инициализирован")
