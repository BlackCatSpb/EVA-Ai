"""
Модуль для обработки естественного языка в CogniFlex.

Содержит классы и утилиты для токенизации, нормализации текста
и других операций обработки естественного языка.
"""

from .text_processor import TextProcessor

# Псевдоним для совместимости
UnifiedTextProcessor = TextProcessor

__all__ = ['TextProcessor', 'UnifiedTextProcessor']
