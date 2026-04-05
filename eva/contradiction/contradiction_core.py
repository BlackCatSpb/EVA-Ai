"""
Модуль управления противоречиями для системы ЕВА.
"""
from eva.contradiction.core_detection import Contradiction, ContradictionCore

__all__ = ['Contradiction', 'ContradictionCore', 'ContradictionDetector', 'OptimizedContradictionDetector']

ContradictionDetector = ContradictionCore
OptimizedContradictionDetector = ContradictionCore
