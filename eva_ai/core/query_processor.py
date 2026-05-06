"""Модуль обработки запросов для ЕВА.

Модуль был разделён на:
- processor_core.py — Main class, initialization, lifecycle
- processor_pipeline.py — Processing pipeline, stages
- processor_handlers.py — Query handlers, response formatting
"""
from .processor_core import QueryProcessor

__all__ = ['QueryProcessor']
