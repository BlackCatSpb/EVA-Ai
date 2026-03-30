"""
Стратегии разрешения противоречий в системе ЕВА
"""
import logging
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger("eva.contradiction.strategies")

class ContradictionResolutionStrategy(ABC):
    """
    Абстрактная стратегия разрешения противоречий.

    Определяет интерфейс для различных подходов к разрешению
    противоречий в знаниях системы.
    """

    def __init__(self, name: str, description: str = ""):
        """
        Инициализирует стратегию разрешения.

        Args:
            name: Название стратегии
            description: Описание стратегии
        """
        self.name = name
        self.description = description or f"Стратегия {name}"
        self.logger = logging.getLogger(f"eva.contradiction.strategies.{name}")

    @abstractmethod
    def resolve(self, contradiction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Разрешает противоречие.

        Args:
            contradiction_data: Данные о противоречии

        Returns:
            Результат разрешения противоречия
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """Возвращает информацию о стратегии."""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.__class__.__name__
        }


class ConservativeStrategy(ContradictionResolutionStrategy):
    """
    Консервативная стратегия - сохраняет существующее знание,
    игнорируя новое противоречивое знание.
    """

    def __init__(self):
        super().__init__(
            "conservative",
            "Сохраняет существующее знание, игнорируя противоречия"
        )

    def resolve(self, contradiction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Разрешает противоречие консервативно."""
        return {
            "action": "keep_existing",
            "reason": "Консервативная стратегия: сохраняем существующее знание",
            "confidence": 0.8
        }


class MajorityVoteStrategy(ContradictionResolutionStrategy):
    """
    Стратегия большинства - выбирает вариант, поддерживаемый
    большинством источников.
    """

    def __init__(self):
        super().__init__(
            "majority_vote",
            "Выбирает вариант с наибольшим количеством источников"
        )

    def resolve(self, contradiction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Разрешает противоречие по принципу большинства."""
        # Простая реализация - возвращает первый вариант
        return {
            "action": "select_first",
            "reason": "Стратегия большинства: выбираем первый вариант",
            "confidence": 0.6
        }


class ConfidenceBasedStrategy(ContradictionResolutionStrategy):
    """
    Стратегия на основе уверенности - выбирает вариант с
    наибольшей уверенностью.
    """

    def __init__(self):
        super().__init__(
            "confidence_based",
            "Выбирает вариант с наибольшей уверенностью"
        )

    def resolve(self, contradiction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Разрешает противоречие на основе уверенности."""
        return {
            "action": "select_highest_confidence",
            "reason": "Стратегия уверенности: выбираем наиболее уверенный вариант",
            "confidence": 0.7
        }


class LearningBasedStrategy(ContradictionResolutionStrategy):
    """
    Стратегия на основе обучения - использует исторические данные
    для выбора лучшего варианта.
    """

    def __init__(self):
        super().__init__(
            "learning_based",
            "Использует обучение для выбора лучшего варианта"
        )

    def resolve(self, contradiction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Разрешает противоречие с использованием обучения."""
        return {
            "action": "learning_based_selection",
            "reason": "Обучающая стратегия: используем исторические данные",
            "confidence": 0.5
        }


# Фабрика стратегий
class ContradictionStrategyFactory:
    """Фабрика для создания стратегий разрешения противоречий."""

    _strategies = {
        "conservative": ConservativeStrategy,
        "majority_vote": MajorityVoteStrategy,
        "confidence_based": ConfidenceBasedStrategy,
        "learning_based": LearningBasedStrategy
    }

    @classmethod
    def create_strategy(cls, strategy_name: str) -> ContradictionResolutionStrategy:
        """
        Создает стратегию по имени.

        Args:
            strategy_name: Имя стратегии

        Returns:
            Экземпляр стратегии

        Raises:
            ValueError: Если стратегия не найдена
        """
        if strategy_name not in cls._strategies:
            available = list(cls._strategies.keys())
            raise ValueError(f"Неизвестная стратегия '{strategy_name}'. Доступные: {available}")

        return cls._strategies[strategy_name]()

    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Возвращает список доступных стратегий."""
        return list(cls._strategies.keys())
